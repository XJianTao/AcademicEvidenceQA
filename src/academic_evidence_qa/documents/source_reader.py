"""模块名称：固定成员与解析来源读取。

用途：按 Library 当前修订选择明确版本及最早有效的同配置快照。
核心设计：数据库身份先冻结，随后在事务外验证原件和标准化产物哈希。
工程要点：缺失、rejected、损坏或跨库来源均使整个构建失败。
"""

from dataclasses import dataclass
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from academic_evidence_qa.contracts.documents import ParsedDocument
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.documents.parser import load_normalized_document, parsed_document_payload
from academic_evidence_qa.infrastructure.database.models import (
    DocumentVersion,
    LibraryPaper,
    ParseSnapshotRecord,
    StoredFile,
    ParseBlock,
    ParseSection,
    IndexGeneration,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore, StoredObject


class SourceError(Exception):
    """固定来源无法核验；仅公开稳定错误码。"""


@dataclass(frozen=True)
class FixedSource:
    """名称：固定来源。用途：把成员选版、快照与完整标准化文档绑定。"""

    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    file_sha256: str
    parser_backend: str
    parser_config_id: str
    adapter_version: str
    document: ParsedDocument


async def read_fixed_sources(
    session: AsyncSession, library_id: UUID, store: LocalFileStore, *, max_json_bytes: int
) -> list[FixedSource]:
    """名称：读取固定来源。用途：完整读取 active 成员，不自动选最新版。

    Args:
        session: 只读会话；不持有 Library 行锁或跨模型运行的事务。
        library_id: 目标库，不复用其他库成员。
        store: 该库对应的可信 FileStore。
        max_json_bytes: 标准化 JSON 大小上限。
    Returns:
        按 paper ID 排序的已核验固定来源。
    Raises:
        SourceError: 任一成员缺少有效快照或原件/产物损坏。
    """
    members = (
        await session.scalars(
            select(LibraryPaper)
            .where(LibraryPaper.library_id == library_id, LibraryPaper.status == "active")
            .order_by(LibraryPaper.paper_id)
        )
    ).all()
    results = []
    for member in members:
        version = await session.get(DocumentVersion, member.selected_document_version_id)
        if version is None or version.paper_id != member.paper_id:
            raise SourceError("selected_version_invalid")
        stored = await session.get(StoredFile, version.file_id)
        if stored is None or stored.file_sha256 != version.file_sha256:
            raise SourceError("source_file_identity_invalid")
        snapshots = (
            await session.scalars(
                select(ParseSnapshotRecord)
                .where(
                    ParseSnapshotRecord.document_version_id == version.document_version_id,
                    ParseSnapshotRecord.parser_backend == member.parser_backend,
                    ParseSnapshotRecord.parser_config_id == member.parser_config_id,
                    ParseSnapshotRecord.quality_state.in_(("usable", "usable_with_warnings")),
                )
                .order_by(ParseSnapshotRecord.created_at, ParseSnapshotRecord.parse_snapshot_id)
            )
        ).all()
        if not snapshots:
            raise SourceError("usable_snapshot_missing")
        try:
            store.verify(
                StoredObject(
                    stored.file_id, stored.file_sha256, stored.object_key, stored.size_bytes
                )
            )
        except Exception as error:
            raise SourceError("source_file_corrupt") from error
        selected = None
        for snapshot in snapshots:
            if snapshot.input_sha256 != version.file_sha256:
                continue
            try:
                document = await verify_snapshot(
                    session, version, snapshot, store, max_json_bytes=max_json_bytes
                )
            except Exception:
                continue
            selected = FixedSource(
                member.paper_id,
                version.document_version_id,
                snapshot.parse_snapshot_id,
                version.file_sha256,
                member.parser_backend,
                member.parser_config_id,
                snapshot.adapter_version,
                document,
            )
            break
        if selected is None:
            raise SourceError("snapshot_integrity_failed")
        results.append(selected)
    return results


async def verify_snapshot(session, version, snapshot, store, *, max_json_bytes, file_executor=None):
    """名称：固定快照完整核验。用途：构建、复用和回源共享同一映射检查。

    Args:
        session, version, snapshot, store: 明确来源及可信存储，调用者已核验库范围。
        max_json_bytes: 标准化 JSON 预算。
    Returns:
        已核验原件、两种产物与数据库逐块/逐章节映射的文档。
    """
    if (
        snapshot.document_version_id != version.document_version_id
        or snapshot.input_sha256 != version.file_sha256
    ):
        raise SourceError("snapshot_identity_invalid")
    stored = await session.get(StoredFile, version.file_id)
    if stored is None or stored.file_sha256 != version.file_sha256:
        raise SourceError("source_file_identity_invalid")

    def load_files():
        store.verify(
            StoredObject(stored.file_id, stored.file_sha256, stored.object_key, stored.size_bytes)
        )
        store.verify_artifact_reference(
            snapshot.raw_artifact_key, expected_sha256=snapshot.raw_artifact_sha256
        )
        store.verify_artifact_reference(
            snapshot.normalized_artifact_key, expected_sha256=snapshot.normalized_artifact_sha256
        )
        return load_normalized_document(
            store.data_root / snapshot.normalized_artifact_key, max_json_bytes=max_json_bytes
        )

    document = (await file_executor.run(load_files))[0] if file_executor else load_files()
    if document.quality_state != snapshot.quality_state or not document.blocks:
        raise SourceError("snapshot_integrity_failed")
    payload = json.loads(json.dumps(parsed_document_payload(document)))
    for model, key, fields in (
        (
            ParseBlock,
            "blocks",
            (
                "block_type",
                "reading_order",
                "text",
                "locations",
                "raw_provenance",
                "quality_flags",
            ),
        ),
        (ParseSection, "sections", ("title", "level", "reading_order", "locations")),
    ):
        rows = (
            await session.scalars(
                select(model)
                .where(model.parse_snapshot_id == snapshot.parse_snapshot_id)
                .order_by(model.reading_order)
            )
        ).all()
        actual = [{field: getattr(row, field) for field in fields} for row in rows]
        expected = sorted(payload[key], key=lambda item: item["reading_order"])
        if actual != expected:
            raise SourceError("snapshot_database_mapping_mismatch")
    if document.quality_state not in ("usable", "usable_with_warnings") or not document.blocks:
        raise SourceError("snapshot_integrity_failed")
    return document


async def read_scoped_source(
    database, store, request, *, outline=False, max_json_bytes, file_executor=None
):
    """名称：有范围的目录/原文读取。用途：固定发布身份或明确未发布预览。

    Args:
        database, store: 正式来源组件。
        request: 明确库、generation、版本、快照及分页预算。
        outline: 仅返回章节目录；默认返回原块内容。
        max_json_bytes: 完整产物核验预算。
    Returns:
        带物理位置、质量标记和续读坐标的白名单结果。
    Raises:
        DocumentError: 跨库/版本/快照为 404；完整性失败为 409。
    """
    async with database.sessions() as session:
        member = await session.scalar(
            select(LibraryPaper).where(
                LibraryPaper.library_id == request.library_id,
                LibraryPaper.paper_id == request.paper_id,
                LibraryPaper.status == "active",
            )
        )
        version = await session.get(DocumentVersion, request.document_version_id)
        snapshot = await session.get(ParseSnapshotRecord, request.parse_snapshot_id)
        if (
            member is None
            or version is None
            or snapshot is None
            or version.paper_id != request.paper_id
            or snapshot.document_version_id != version.document_version_id
        ):
            raise DocumentError("source_not_found", 404)
        if request.preview:
            if (
                request.generation_id is not None
                or member.selected_document_version_id != version.document_version_id
                or (member.parser_backend, member.parser_config_id)
                != (snapshot.parser_backend, snapshot.parser_config_id)
            ):
                raise DocumentError("source_not_found", 404)
        else:
            if request.generation_id is None:
                raise DocumentError("generation_required", 422)
            generation = await session.get(IndexGeneration, request.generation_id)
            identity = (
                str(request.paper_id),
                str(version.document_version_id),
                str(snapshot.parse_snapshot_id),
            )
            if (
                generation is None
                or generation.library_id != request.library_id
                or generation.state not in ("published", "retired")
                or not any(
                    (
                        item.get("paper_id"),
                        item.get("document_version_id"),
                        item.get("parse_snapshot_id"),
                    )
                    == identity
                    for item in generation.manifest.get("members", [])
                )
            ):
                raise DocumentError("source_not_found", 404)
            # 发布清单不可变，但读取仍核验哈希，避免损坏的绑定放行来源。
            from academic_evidence_qa.ingestion.index_builder import content_hash

            if content_hash(generation.manifest) != generation.manifest_sha256:
                raise DocumentError("generation_integrity_failed", 409)
        try:
            await verify_snapshot(
                session,
                version,
                snapshot,
                store,
                max_json_bytes=max_json_bytes,
                file_executor=file_executor,
            )
        except Exception as error:
            raise DocumentError("snapshot_integrity_failed", 409) from error
        sections = list(
            (
                await session.scalars(
                    select(ParseSection)
                    .where(ParseSection.parse_snapshot_id == snapshot.parse_snapshot_id)
                    .order_by(ParseSection.reading_order)
                )
            ).all()
        )
        base = {
            "library_id": str(request.library_id),
            "generation_id": str(request.generation_id) if request.generation_id else None,
            "paper_id": str(request.paper_id),
            "document_version_id": str(version.document_version_id),
            "parse_snapshot_id": str(snapshot.parse_snapshot_id),
            "publication_state": "unpublished" if request.preview else "published",
        }

        async def recheck_member():
            # READ COMMITTED 的新语句读取当前成员状态，防止长核验期间
            # 已软移除的成员仍返回原文；不把文件读取放进成员写锁事务。
            current = (
                await session.execute(
                    select(
                        LibraryPaper.status,
                        LibraryPaper.selected_document_version_id,
                        LibraryPaper.parser_backend,
                        LibraryPaper.parser_config_id,
                    ).where(
                        LibraryPaper.library_id == request.library_id,
                        LibraryPaper.paper_id == request.paper_id,
                    )
                )
            ).first()
            if (
                current is None
                or current.status != "active"
                or (
                    request.preview
                    and (
                        current.selected_document_version_id != version.document_version_id
                        or (current.parser_backend, current.parser_config_id)
                        != (snapshot.parser_backend, snapshot.parser_config_id)
                    )
                )
            ):
                raise DocumentError("source_not_found", 404)

        if outline:
            selected = sections[request.offset : request.offset + request.limit]
            await recheck_member()
            return {
                **base,
                "items": [
                    {
                        "section_id": str(item.section_id),
                        "title": item.title,
                        "level": item.level,
                        "reading_order": item.reading_order,
                        "locations": item.locations,
                    }
                    for item in selected
                ],
                "next_offset": request.offset + len(selected)
                if request.offset + len(selected) < len(sections)
                else None,
            }
        query = (
            select(ParseBlock)
            .where(ParseBlock.parse_snapshot_id == snapshot.parse_snapshot_id)
            .order_by(ParseBlock.reading_order)
        )
        if request.block_id:
            query = query.where(ParseBlock.block_id == request.block_id)
        if request.section_id:
            section = next(
                (item for item in sections if item.section_id == request.section_id), None
            )
            if section is None:
                raise DocumentError("section_not_found", 404)
            # 章节直到下一个同级或更高层标题，包含其下的子章节。
            following = next(
                (
                    item
                    for item in sections
                    if item.reading_order > section.reading_order
                    and (item.level or 1) <= (section.level or 1)
                ),
                None,
            )
            query = query.where(ParseBlock.reading_order >= section.reading_order)
            if following:
                query = query.where(ParseBlock.reading_order < following.reading_order)
        blocks = list((await session.scalars(query)).all())
        if request.block_id and not blocks:
            raise DocumentError("block_not_found", 404)
        budget, results = request.max_characters, []
        next_offset, next_char = None, 0
        if request.char_offset and (
            request.offset >= len(blocks) or request.char_offset >= len(blocks[request.offset].text)
        ):
            raise DocumentError("source_offset_invalid", 422)
        for position in range(request.offset, min(len(blocks), request.offset + request.limit)):
            block = blocks[position]
            start = request.char_offset if position == request.offset else 0
            text = block.text[start : start + budget]
            results.append(
                {
                    "block_id": str(block.block_id),
                    "block_type": block.block_type,
                    "reading_order": block.reading_order,
                    "text": text,
                    "char_start": start,
                    "char_end": start + len(text),
                    "locations": block.locations,
                    "quality_flags": block.quality_flags,
                }
            )
            budget -= len(text)
            if start + len(text) < len(block.text):
                next_offset, next_char = position, start + len(text)
                break
            if position + 1 < len(blocks):
                next_offset, next_char = position + 1, 0
            else:
                next_offset = None
            if budget == 0:
                break
        await recheck_member()
        return {**base, "items": results, "next_offset": next_offset, "next_char_offset": next_char}


async def load_evidence_pack(database, library_id, retrieval_run_id):
    """名称：加载固定证据封装。用途：核对 registry 与封装哈希，不依赖 Qdrant。

    Args:
        database, library_id, retrieval_run_id: 显式可信上下文，跨库返回404。
    Returns:
        通过持久映射一致性核验的 EvidencePack；尚未替代来源可读性检查。
    """
    from academic_evidence_qa.contracts.rag import EvidencePack
    from academic_evidence_qa.infrastructure.database.models import RetrievalRun, EvidenceRecord
    from academic_evidence_qa.ingestion.index_builder import content_hash

    async with database.sessions() as session:
        run = await session.get(RetrievalRun, retrieval_run_id)
        if run is None or run.library_id != library_id:
            raise DocumentError("evidence_not_found", 404)
        if content_hash(run.pack) != run.pack_sha256:
            raise DocumentError("evidence_integrity_failed", 409)
        pack = EvidencePack.model_validate(run.pack)
        if (
            pack.scope.library_id != library_id
            or pack.retrieval_run_id != retrieval_run_id
            or pack.scope.generation_id != run.generation_id
        ):
            raise DocumentError("evidence_integrity_failed", 409)
        rows = (
            await session.scalars(
                select(EvidenceRecord).where(EvidenceRecord.retrieval_run_id == retrieval_run_id)
            )
        ).all()
        if {row.evidence_id for row in rows} != {e.evidence_id for e in pack.evidence}:
            raise DocumentError("evidence_integrity_failed", 409)
        expected = {e.evidence_id: e.model_dump(mode="json") for e in pack.evidence}
        if any(
            content_hash(row.payload) != row.payload_sha256
            or row.payload != expected[row.evidence_id]
            for row in rows
        ):
            raise DocumentError("evidence_integrity_failed", 409)
    return pack


async def verify_evidence_sources(
    database, store, pack, *, max_json_bytes, file_executor, cache=None
):
    """名称：复用固定证据来源核验。用途：按快照一次装载，逐区间核验来源及位置。

    Args:
        pack: 已核验 registry 的封装。
        cache: 仅当前请求的已核验 ParsedDocument，不跨请求缓存来源。
        max_json_bytes, file_executor: 现有有界读取约束。
    Returns:
        None；坏源、范围外三元组及错误原文均硬失败。
    """
    cache = {} if cache is None else cache
    identities = {
        (m["paper_id"], m["document_version_id"], m["parse_snapshot_id"])
        for m in pack.scope.members
    }
    async with database.sessions() as session:
        for evidence in pack.evidence:
            if (
                str(evidence.paper_id),
                str(evidence.document_version_id),
                str(evidence.parse_snapshot_id),
            ) not in identities:
                raise DocumentError("evidence_out_of_scope", 409)
            version = await session.get(DocumentVersion, evidence.document_version_id)
            snapshot = await session.get(ParseSnapshotRecord, evidence.parse_snapshot_id)
            if version is None or snapshot is None or version.paper_id != evidence.paper_id:
                raise DocumentError("evidence_source_invalid", 409)
            if evidence.parse_snapshot_id not in cache:
                try:
                    cache[evidence.parse_snapshot_id] = await verify_snapshot(
                        session,
                        version,
                        snapshot,
                        store,
                        max_json_bytes=max_json_bytes,
                        file_executor=file_executor,
                    )
                except Exception as error:
                    raise DocumentError("snapshot_integrity_failed", 409) from error
            for part in evidence.ranges:
                block = await session.get(ParseBlock, part.block_id)
                if (
                    block is None
                    or block.parse_snapshot_id != evidence.parse_snapshot_id
                    or block.reading_order != part.block_reading_order
                    or not 0 <= part.start < part.end <= len(block.text)
                    or part.text != block.text[part.start : part.end]
                    or part.block_type != block.block_type
                    or part.locations != block.locations
                    or part.quality_flags != block.quality_flags
                ):
                    raise DocumentError("evidence_range_invalid", 409)


async def read_registered_evidence(database, store, request, *, max_json_bytes, file_executor):
    """名称：有范围证据分页。用途：新取证检查完整 active scope，历史答案显式豁免移除。

    Args:
        request: library/run/evidence 及可选已保存 answer_id，均需同一绑定。
        max_json_bytes, file_executor: 现有有界来源 I/O。
    Returns:
        固定证据区间的分页片段及续读偏移；历史读取不得用于扩展新 scope。
    """
    from academic_evidence_qa.infrastructure.database.models import AnswerArtifact

    pack = await load_evidence_pack(database, request.library_id, request.retrieval_run_id)
    historical = request.answer_id is not None
    if historical:
        async with database.sessions() as session:
            answer = await session.get(AnswerArtifact, request.answer_id)
            if (
                answer is None
                or answer.library_id != request.library_id
                or answer.retrieval_run_id != pack.retrieval_run_id
                or answer.state != "completed"
            ):
                raise DocumentError("historical_answer_not_found", 404)

    async def check_members():
        async with database.sessions() as session:
            active = set(
                (
                    await session.scalars(
                        select(LibraryPaper.paper_id).where(
                            LibraryPaper.library_id == request.library_id,
                            LibraryPaper.status == "active",
                        )
                    )
                ).all()
            )
        if not {UUID(m["paper_id"]) for m in pack.scope.members}.issubset(active):
            raise DocumentError("scope_membership_changed", 409)

    if not historical:
        await check_members()
    evidence = next((e for e in pack.evidence if e.evidence_id == request.evidence_id), None)
    if evidence is None:
        raise DocumentError("evidence_not_found", 404)
    await verify_evidence_sources(
        database, store, pack, max_json_bytes=max_json_bytes, file_executor=file_executor
    )
    if request.offset >= len(evidence.ranges):
        raise DocumentError("source_offset_invalid", 422)
    part = evidence.ranges[request.offset]
    if request.char_offset >= len(part.text):
        raise DocumentError("source_offset_invalid", 422)
    text = part.text[request.char_offset : request.char_offset + request.max_characters]
    end = request.char_offset + len(text)
    next_offset = request.offset if end < len(part.text) else request.offset + 1
    next_char = end if end < len(part.text) else 0
    if not historical:
        await check_members()
    return {
        "evidence_id": evidence.evidence_id,
        "retrieval_run_id": pack.retrieval_run_id,
        "library_id": request.library_id,
        "generation_id": pack.scope.generation_id,
        "paper_id": evidence.paper_id,
        "document_version_id": evidence.document_version_id,
        "parse_snapshot_id": evidence.parse_snapshot_id,
        "historical": historical,
        "items": [
            {
                **part.model_dump(mode="json"),
                "text": text,
                "start": part.start + request.char_offset,
                "end": part.start + end,
            }
        ],
        "quality_flags": evidence.quality_flags,
        "next_offset": next_offset if next_offset < len(evidence.ranges) else None,
        "next_char_offset": next_char,
    }
