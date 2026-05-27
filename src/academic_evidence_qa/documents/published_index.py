"""模块名称：管理复用的已发布索引核验。

用途：在受理事务之外核验不可变清单、数据库块及 Qdrant 来源与向量结构。
核心设计：不加载模型；从已冻结块重新计算 sparse，dense 仅检查结构与有限值。
工程要点：不删除集合；与原发布逐值 dense 验证的职责分离，不宣称重新推理。
"""

import asyncio
import json
import math

from sqlalchemy import select

from academic_evidence_qa.documents.source_reader import verify_snapshot
from academic_evidence_qa.infrastructure.database.models import (
    Library,
    IndexGeneration,
    IndexChunk,
    DocumentVersion,
    ParseSnapshotRecord,
)
from academic_evidence_qa.infrastructure.qdrant_store import QdrantStore
from academic_evidence_qa.ingestion.index_builder import content_hash
from academic_evidence_qa.retrieval.lexical import freeze_statistics, document_vector


async def verify_active_index(database, store, library_id, *, max_json_bytes, qdrant_url):
    """名称：完整发布复用核验。用途：无 Library 行锁时验证固定活动代。

    Args:
        database, store, library_id: 明确本地库和存储。
        max_json_bytes, qdrant_url: 有界快照读取及 Qdrant 地址。
    Returns:
        成功时返回已核验 generation ID；不完整返回 None，交由正式 Job 重建。
    """
    try:
        async with database.sessions() as session:
            library = await session.get(Library, library_id)
            if not library or not library.active_generation_id:
                return None
            generation = await session.get(IndexGeneration, library.active_generation_id)
            manifest = generation.manifest
            if (
                generation.state != "published"
                or content_hash(manifest) != generation.manifest_sha256
            ):
                return None
            for member in manifest["members"]:
                from uuid import UUID

                version = await session.get(DocumentVersion, UUID(member["document_version_id"]))
                snapshot = await session.get(ParseSnapshotRecord, UUID(member["parse_snapshot_id"]))
                if version.paper_id != UUID(member["paper_id"]):
                    return None
                await verify_snapshot(
                    session, version, snapshot, store, max_json_bytes=max_json_bytes
                )
            rows = (
                await session.scalars(
                    select(IndexChunk)
                    .where(IndexChunk.generation_id == generation.generation_id)
                    .order_by(IndexChunk.reading_order)
                )
            ).all()
            fields = tuple(IndexChunk.__table__.columns.keys())
            chunks = json.loads(
                json.dumps(
                    [{key: getattr(row, key) for key in fields} for row in rows], default=str
                )
            )
            if (
                content_hash(chunks) != manifest["chunks_sha256"]
                or [row["chunk_id"] for row in chunks] != manifest["chunk_ids"]
            ):
                return None
            statistics = freeze_statistics([row["retrieval_text"] for row in chunks])
            if (
                statistics.sha256 != manifest["statistics_sha256"]
                or statistics.to_dict() != manifest["lexical_statistics"]
            ):
                return None

        # 同步网络放在线程中且有客户端超时；不跨网络持有数据库事务。
        def check_points():
            qdrant = QdrantStore(qdrant_url)
            try:
                actual, offset = {}, None
                while True:
                    points, offset = qdrant.client.scroll(
                        generation.collection_name,
                        limit=128,
                        offset=offset,
                        with_vectors=True,
                        with_payload=True,
                    )
                    for point in points:
                        actual[str(point.id)] = point
                    if offset is None:
                        break
                expected = []
                for row in chunks:
                    point = actual.get(row["chunk_id"])
                    if point is None or not isinstance(point.vector, dict):
                        raise ValueError("published_vectors_missing")
                    dense = point.vector.get("dense", [])
                    if (
                        len(dense) != 1024
                        or not all(math.isfinite(value) for value in dense)
                        or abs(sum(value * value for value in dense) - 1) > 1e-3
                    ):
                        raise ValueError("published_dense_invalid")
                    expected.append(
                        {
                            "id": row["chunk_id"],
                            "dense": dense,
                            "sparse": document_vector(row["retrieval_text"], statistics),
                            "payload": {
                                "library_id": str(library_id),
                                "generation_id": str(generation.generation_id),
                                "attempt_id": str(generation.attempt_id),
                                "paper_id": row["paper_id"],
                                "document_version_id": row["document_version_id"],
                                "parse_snapshot_id": row["parse_snapshot_id"],
                                "chunk_id": row["chunk_id"],
                                "block_type": row["block_type"],
                                "source_ranges": row["source_ranges"],
                                "source_locations": row["source_locations"],
                                "statistics_sha256": statistics.sha256,
                            },
                        }
                    )
                qdrant.verify(generation.collection_name, expected)
            finally:
                qdrant.close()

        task = asyncio.create_task(asyncio.to_thread(check_points))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # 同步客户端有超时；取消等待其关闭，避免遗留网络线程持有资源。
            from contextlib import suppress

            with suppress(Exception):
                await task
            raise
        return generation.generation_id
    except Exception:
        return None
