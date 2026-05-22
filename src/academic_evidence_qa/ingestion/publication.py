"""模块名称：索引代预留、核验与原子发布。

用途：通过 PostgreSQL Library 行锁实现同库互斥及发布前冲突检查。
核心设计：模型/网络校验在事务外；事务仅冻结输入或切换活动指针。
工程要点：失败不改活动指针；同库构建通过 Library 行锁和构建指针互斥。
"""

from datetime import datetime, timezone
import json
from uuid import UUID, uuid4

from sqlalchemy import select

from academic_evidence_qa.infrastructure.database.models import (
    IndexChunk,
    IndexGeneration,
    Library,
)
from academic_evidence_qa.ingestion.index_builder import BuiltIndex, content_hash


class PublicationError(Exception):
    """发布条件不满足；异常内容为稳定错误码。"""


def _check_manifest(generation, built):
    """核验全部构建身份、chunk 内容及冻结词法统计，避免同 attempt 内容错配。"""
    expected = {
        "library_id": str(generation.library_id),
        "generation_id": str(generation.generation_id),
        "attempt_id": str(generation.attempt_id),
        "membership_revision": generation.membership_revision,
        "collection_name": generation.collection_name,
        "expected_active_generation_id": str(generation.expected_active_generation_id)
        if generation.expected_active_generation_id
        else None,
    }
    if any(built.manifest.get(key) != value for key, value in expected.items()):
        raise PublicationError("manifest_identity_conflict")
    if (
        content_hash(built.manifest) != built.manifest_sha256
        or content_hash(built.manifest["lexical_statistics"]) != built.manifest["statistics_sha256"]
        or content_hash(json.loads(json.dumps(built.chunks, default=str)))
        != built.manifest["chunks_sha256"]
    ):
        raise PublicationError("manifest_hash_mismatch")
    if built.manifest["chunk_ids"] != [
        str(item["chunk_id"]) for item in built.chunks
    ] or built.manifest["chunk_ids"] != [item["id"] for item in built.points]:
        raise PublicationError("manifest_chunk_identity_mismatch")


async def reserve_generation(database, library_id: UUID) -> IndexGeneration:
    """名称：预留构建。用途：短事务锁定 Library，拒绝同库并发构建。

    Args:
        database: 已组装的 Database 生命周期。
        library_id: 存在的目标库。
    Returns:
        固定修订、attempt 和预期旧指针的新 building generation。
    """
    async with database.transaction() as session:
        library = await session.scalar(
            select(Library).where(Library.library_id == library_id).with_for_update()
        )
        if library is None:
            raise PublicationError("library_missing")
        if library.building_generation_id is not None:
            raise PublicationError("library_build_in_progress")
        identity, attempt = uuid4(), uuid4()
        generation = IndexGeneration(
            generation_id=identity,
            library_id=library_id,
            attempt_id=attempt,
            membership_revision=library.membership_revision,
            expected_active_generation_id=library.active_generation_id,
            collection_name=f"aeqa_{library_id.hex[:8]}_{identity.hex}_{attempt.hex}",
            state="building",
            manifest={},
        )
        session.add(generation)
        await session.flush()
        library.building_generation_id = identity
        return generation


async def save_built_index(database, generation, built: BuiltIndex) -> None:
    """名称：保存冻结清单。用途：核验身份后持久化 chunk，不执行发布。

    Args:
        database, generation: 预留构建的数据库与身份。
        built: 同 attempt 生成的暂存结果。
    Returns:
        None；错误身份不会覆盖原清单。
    """
    _check_manifest(generation, built)
    async with database.transaction() as session:
        library = await session.scalar(
            select(Library).where(Library.library_id == generation.library_id).with_for_update()
        )
        current = await session.get(IndexGeneration, generation.generation_id, with_for_update=True)
        _check_current(library, current, generation)
        if current.manifest or built.manifest["attempt_id"] != str(current.attempt_id):
            raise PublicationError("manifest_identity_conflict")
        if content_hash(built.manifest) != built.manifest_sha256:
            raise PublicationError("manifest_hash_mismatch")
        current.manifest = built.manifest
        current.manifest_sha256 = built.manifest_sha256
        session.add_all(
            IndexChunk(
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in data.items()
                }
            )
            for data in built.chunks
        )


def _check_current(library, current, generation) -> None:
    """核验库锁定关系与当前 attempt；不放宽已发布或失败状态。"""
    if (
        library is None
        or current is None
        or current.library_id != generation.library_id
        or current.attempt_id != generation.attempt_id
        or library.building_generation_id != generation.generation_id
        or current.state != "building"
    ):
        raise PublicationError("build_attempt_mismatch")


async def publish_generation(
    database, generation, built: BuiltIndex, qdrant, verify_sources
) -> dict:
    """名称：校验并发布。用途：完整外部核验后在短事务原子切换指针。

    Args:
        database, generation, built: 同一已冻结构建的身份与内容。
        qdrant: 逐点核验正式存储。
        verify_sources: 再次确认来源可读性和哈希的异步回调。
    Returns:
        真实 Qdrant 完整性与 smoke 证据。
    Raises:
        PublicationError: 修订、attempt、旧指针或数据库清单不一致。
    """
    _check_manifest(generation, built)
    await verify_sources()
    evidence = qdrant.verify(generation.collection_name, built.points)
    async with database.transaction() as session:
        library = await session.scalar(
            select(Library).where(Library.library_id == generation.library_id).with_for_update()
        )
        current = await session.get(IndexGeneration, generation.generation_id, with_for_update=True)
        _check_current(library, current, generation)
        if library.membership_revision != current.membership_revision:
            raise PublicationError("membership_changed")
        if library.active_generation_id != current.expected_active_generation_id:
            raise PublicationError("active_generation_conflict")
        if (
            current.manifest_sha256 != built.manifest_sha256
            or content_hash(current.manifest) != built.manifest_sha256
        ):
            raise PublicationError("manifest_hash_mismatch")
        chunk_ids = set(
            await session.scalars(
                select(IndexChunk.chunk_id).where(IndexChunk.generation_id == current.generation_id)
            )
        )
        if chunk_ids != {UUID(identity) for identity in built.manifest["chunk_ids"]}:
            raise PublicationError("database_chunk_id_set_mismatch")
        rows = (
            await session.scalars(
                select(IndexChunk)
                .where(IndexChunk.generation_id == current.generation_id)
                .order_by(IndexChunk.reading_order)
            )
        ).all()
        actual_chunks = [
            {key: getattr(row, key) for key in data}
            for row, data in zip(rows, built.chunks, strict=True)
        ]
        if (
            content_hash(json.loads(json.dumps(actual_chunks, default=str)))
            != built.manifest["chunks_sha256"]
        ):
            raise PublicationError("database_chunk_content_mismatch")
        if library.active_generation_id is not None:
            previous = await session.get(
                IndexGeneration, library.active_generation_id, with_for_update=True
            )
            if (
                previous is None
                or previous.library_id != library.library_id
                or previous.state != "published"
            ):
                raise PublicationError("previous_generation_invalid")
            previous.state = "retired"
        current.state = "published"
        current.published_at = datetime.now(timezone.utc)
        library.active_generation_id = current.generation_id
        library.building_generation_id = None
    return evidence


async def fail_generation(database, generation, error_code: str) -> None:
    """名称：记录构建失败。用途：仅释放同 attempt 的构建指针，保留旧活动代。

    Args:
        database, generation: 本次已预留的身份。
        error_code: 脱敏稳定码，不保存论文或第三方异常原文。
    Returns:
        None；集合和旧代均保留，不自动恢复崩溃构建。
    """
    async with database.transaction() as session:
        library = await session.scalar(
            select(Library).where(Library.library_id == generation.library_id).with_for_update()
        )
        current = await session.get(IndexGeneration, generation.generation_id, with_for_update=True)
        _check_current(library, current, generation)
        current.state = "failed"
        current.error_code = error_code
        library.building_generation_id = None
