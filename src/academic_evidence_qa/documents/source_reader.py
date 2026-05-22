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
from academic_evidence_qa.documents.parser import load_normalized_document, parsed_document_payload
from academic_evidence_qa.infrastructure.database.models import (
    DocumentVersion,
    LibraryPaper,
    ParseSnapshotRecord,
    StoredFile,
    ParseBlock,
    ParseSection,
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
                store.verify_artifact_reference(
                    snapshot.raw_artifact_key, expected_sha256=snapshot.raw_artifact_sha256
                )
                store.verify_artifact_reference(
                    snapshot.normalized_artifact_key,
                    expected_sha256=snapshot.normalized_artifact_sha256,
                )
                document = load_normalized_document(
                    store.data_root / snapshot.normalized_artifact_key,
                    max_json_bytes=max_json_bytes,
                )
                if document.quality_state != snapshot.quality_state or not document.blocks:
                    continue
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
