"""模块名称：可信 Agent 来源登记与历史读取。

用途：通过独立交付身份保存多次取证和直接读取来源，不借用一次问答 answer_id。
核心设计：RAG 重建来源并核验引用，宿主再把通过检查的 final 写入 append-only 会话。
工程要点：登记不等于交付；历史接口只供可信宿主代理，不借历史授权进行新取证。
"""

import json
from uuid import UUID

from sqlalchemy import func, select

from academic_evidence_qa.answering.citations import normalize_citation_aliases
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.documents.source_reader import (
    load_evidence_pack,
    verify_evidence_sources,
    verify_snapshot,
)
from academic_evidence_qa.infrastructure.database.models import (
    AgentDelivery,
    DocumentVersion,
    IndexGeneration,
    Library,
    LibraryPaper,
    ParseBlock,
    ParseSection,
    ParseSnapshotRecord,
)
from academic_evidence_qa.ingestion.index_builder import content_hash

DELIVERY_MAX_BYTES = 256 * 1024


def _delivery_payload(row):
    """验证持久白名单及摘要；损坏记录不能作为历史授权。"""
    payload = row.payload
    if (
        not isinstance(payload, dict)
        or content_hash(payload) != row.payload_sha256
        or payload.get("contract") != "m3_agent_delivery_v1"
        or payload.get("delivery_id") != str(row.delivery_id)
        or payload.get("library_id") != str(row.library_id)
        or payload.get("generation_id") != str(row.generation_id)
        or payload.get("registration_status") != "registered"
        or payload.get("citation_check_status") != "passed"
    ):
        raise DocumentError("delivery_integrity_failed", 409)
    return {**payload, "payload_sha256": row.payload_sha256}


async def load_agent_delivery(database, library_id, delivery_id):
    """名称：读取登记身份。用途：恢复同一交付，不把 registered 提升为 final。

    Args:
        database, library_id, delivery_id: 可信宿主固定的上下文。
    Returns:
        经摘要核验的登记结果；最终交付状态仍由宿主会话记录确定。
    """
    async with database.sessions() as session:
        row = await session.get(AgentDelivery, delivery_id)
        if row is None or row.library_id != library_id:
            raise DocumentError("delivery_not_found", 404)
        return _delivery_payload(row)


async def _fixed_identity(session, library_id, generation_id, paper_id, version_id, snapshot_id):
    """核验发布清单、库和三元组；registry 哈希不能替代发布来源授权。"""
    generation = await session.get(IndexGeneration, generation_id)
    version = await session.get(DocumentVersion, version_id)
    snapshot = await session.get(ParseSnapshotRecord, snapshot_id)
    if (
        generation is None
        or generation.library_id != library_id
        or generation.state not in ("published", "retired")
        or version is None
        or version.paper_id != paper_id
        or snapshot is None
        or snapshot.document_version_id != version_id
        or snapshot.input_sha256 != version.file_sha256
    ):
        raise DocumentError("source_not_found", 404)
    manifest = generation.manifest
    if (
        not isinstance(manifest, dict)
        or content_hash(manifest) != generation.manifest_sha256
        or manifest.get("library_id") != str(library_id)
        or manifest.get("generation_id") != str(generation_id)
        or manifest.get("collection_name") != generation.collection_name
        or manifest.get("attempt_id") != str(generation.attempt_id)
        or manifest.get("membership_revision") != generation.membership_revision
        or not isinstance(manifest.get("members"), list)
    ):
        raise DocumentError("generation_integrity_failed", 409)
    matches = [
        item
        for item in manifest["members"]
        if isinstance(item, dict) and item.get("paper_id") == str(paper_id)
    ]
    if (
        len(matches) != 1
        or matches[0].get("document_version_id") != str(version_id)
        or matches[0].get("parse_snapshot_id") != str(snapshot_id)
        or matches[0].get("file_sha256") != version.file_sha256
    ):
        raise DocumentError("source_not_found", 404)
    return version, snapshot


async def _active_members(session, library_id, paper_ids):
    """在耗时来源检查前后核验完整取证范围，避免以旧 registry 扩展当前权限。"""
    active = set(
        (
            await session.scalars(
                select(LibraryPaper.paper_id).where(
                    LibraryPaper.library_id == library_id,
                    LibraryPaper.paper_id.in_(paper_ids),
                    LibraryPaper.status == "active",
                )
            )
        ).all()
    )
    if not set(paper_ids).issubset(active):
        raise DocumentError("scope_membership_changed", 409)


class AgentDeliveryService:
    """名称：Agent 来源服务。用途：在既有来源组件上核验、幂等登记和回读。

    Args:
        database, store: 现有 PostgreSQL 与 FileStore；不依赖生成模型或索引服务。
        max_json_bytes, file_executor: 沿用解析来源的文件和字节边界。
    Returns:
        只保存真实来源的独立登记，不执行生成或宿主 final 写入。
    """

    def __init__(self, database, store, *, max_json_bytes, file_executor=None):
        self.database, self.store = database, store
        self.max_json_bytes, self.file_executor = max_json_bytes, file_executor

    async def _sources(self, library_id, generation_id, sources, *, historical=False):
        """重建白名单正文，不采信宿主输入的原文、位置或 quality flags。"""
        packs, snapshots, canonical, paper_ids = {}, {}, [], set()
        for source in sources:
            if source.kind == "evidence":
                run_id = source.retrieval_run_id
                if run_id not in packs:
                    try:
                        pack = await load_evidence_pack(self.database, library_id, run_id)
                    except DocumentError:
                        raise
                    except Exception as error:
                        raise DocumentError("evidence_integrity_failed", 409) from error
                    if pack.scope.generation_id != generation_id:
                        raise DocumentError("evidence_out_of_scope", 409)
                    async with self.database.sessions() as session:
                        try:
                            for member in pack.scope.members:
                                identity = tuple(
                                    UUID(member[key])
                                    for key in (
                                        "paper_id",
                                        "document_version_id",
                                        "parse_snapshot_id",
                                    )
                                )
                                await _fixed_identity(session, library_id, generation_id, *identity)
                                paper_ids.add(identity[0])
                        except (KeyError, TypeError, ValueError) as error:
                            raise DocumentError("evidence_integrity_failed", 409) from error
                        if not historical:
                            await _active_members(session, library_id, paper_ids)
                    await verify_evidence_sources(
                        self.database,
                        self.store,
                        pack,
                        max_json_bytes=self.max_json_bytes,
                        file_executor=self.file_executor,
                        cache=snapshots,
                    )
                    packs[run_id] = pack
                pack = packs[run_id]
                evidence = next(
                    (item for item in pack.evidence if item.evidence_id == source.evidence_id), None
                )
                if evidence is None:
                    raise DocumentError("evidence_not_found", 404)
                async with self.database.sessions() as session:
                    await _fixed_identity(
                        session,
                        library_id,
                        generation_id,
                        evidence.paper_id,
                        evidence.document_version_id,
                        evidence.parse_snapshot_id,
                    )
                item = {
                    "alias": source.alias,
                    "kind": "evidence",
                    "source_id": str(evidence.evidence_id),
                    "retrieval_run_id": str(run_id),
                    "evidence_id": str(evidence.evidence_id),
                    "upstream_alias": evidence.alias,
                    "paper_id": str(evidence.paper_id),
                    "document_version_id": str(evidence.document_version_id),
                    "parse_snapshot_id": str(evidence.parse_snapshot_id),
                    "text": "\n\n".join(part.text for part in evidence.ranges),
                    "ranges": [part.model_dump(mode="json") for part in evidence.ranges],
                    "quality_flags": evidence.quality_flags,
                }
                paper_ids.add(evidence.paper_id)
            else:
                paper_ids.add(source.paper_id)
                async with self.database.sessions() as session:
                    version, snapshot = await _fixed_identity(
                        session,
                        library_id,
                        generation_id,
                        source.paper_id,
                        source.document_version_id,
                        source.parse_snapshot_id,
                    )
                    if not historical:
                        await _active_members(session, library_id, paper_ids)
                    if source.parse_snapshot_id not in snapshots:
                        try:
                            snapshots[source.parse_snapshot_id] = await verify_snapshot(
                                session,
                                version,
                                snapshot,
                                self.store,
                                max_json_bytes=self.max_json_bytes,
                                file_executor=self.file_executor,
                            )
                        except Exception as error:
                            raise DocumentError("snapshot_integrity_failed", 409) from error
                    item = await self._direct_source(session, source)
            canonical.append(item)
        if not historical:
            async with self.database.sessions() as session:
                await _active_members(session, library_id, paper_ids)
        return canonical, paper_ids

    async def _direct_source(self, session, source):
        """把固定原块区间或目录标题转换为真实来源，正文不由客户端提交。"""
        item = {
            "alias": source.alias,
            "kind": source.kind,
            "paper_id": str(source.paper_id),
            "document_version_id": str(source.document_version_id),
            "parse_snapshot_id": str(source.parse_snapshot_id),
        }
        if source.kind == "block":
            block = await session.get(ParseBlock, source.block_id)
            if (
                block is None
                or block.parse_snapshot_id != source.parse_snapshot_id
                or not 0 <= source.char_start < source.char_end <= len(block.text)
            ):
                raise DocumentError("source_range_invalid", 409)
            text = block.text[source.char_start : source.char_end]
            return {
                **item,
                "source_id": str(block.block_id),
                "block_id": str(block.block_id),
                "char_start": source.char_start,
                "char_end": source.char_end,
                "text": text,
                "ranges": [
                    {
                        "block_id": str(block.block_id),
                        "block_reading_order": block.reading_order,
                        "start": source.char_start,
                        "end": source.char_end,
                        "text": text,
                        "block_type": block.block_type,
                        "locations": block.locations,
                        "quality_flags": block.quality_flags,
                    }
                ],
                "quality_flags": block.quality_flags,
            }
        section = await session.get(ParseSection, source.section_id)
        if (
            section is None
            or section.parse_snapshot_id != source.parse_snapshot_id
            or not section.title
        ):
            raise DocumentError("section_not_found", 404)
        return {
            **item,
            "source_id": str(section.section_id),
            "section_id": str(section.section_id),
            "title": section.title,
            "text": section.title,
            "reading_order": section.reading_order,
            "locations": section.locations,
            "ranges": [],
            "quality_flags": [],
        }

    async def register(self, request):
        """名称：登记论文事实回答。用途：先重放，再核验来源并在短事务内冻结结果。

        Args:
            request: 已严格校验的宿主输入；delivery_id 兼作稳定幂等身份。
        Returns:
            registered 响应；引用失败不持久正文，也不触发生成式修复。
        Raises:
            DocumentError: 错绑、坏源、缺失引用、冲突或来源预算超限。
        """
        raw = request.model_dump(mode="json")
        raw["sources"].sort(key=lambda item: int(item["alias"][1:]))
        digest = content_hash(raw)
        async with self.database.sessions() as session:
            existing = await session.get(AgentDelivery, request.delivery_id)
            if existing is not None:
                if existing.request_sha256 != digest:
                    raise DocumentError("delivery_idempotency_conflict", 409)
                return _delivery_payload(existing)
        ordered = sorted(request.sources, key=lambda item: int(item.alias[1:]))
        sources, paper_ids = await self._sources(request.library_id, request.generation_id, ordered)
        text, aliases, errors = normalize_citation_aliases(
            request.text, {source["alias"] for source in sources}
        )
        if errors:
            raise DocumentError("delivery_citation_invalid", 422)
        payload = {
            "contract": "m3_agent_delivery_v1",
            "delivery_id": str(request.delivery_id),
            "library_id": str(request.library_id),
            "generation_id": str(request.generation_id),
            "text": text,
            "citations": [
                {key: item[key] for key in ("alias", "kind", "source_id")}
                for alias in aliases
                for item in sources
                if item["alias"] == alias
            ],
            "sources": sources,
            "citation_check_status": "passed",
            "support_check_status": "not_run",
            "registration_status": "registered",
        }
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > DELIVERY_MAX_BYTES:
            raise DocumentError("delivery_sources_too_large", 413)
        payload_sha256 = content_hash(payload)
        async with self.database.transaction() as session:
            # 无行可锁时也按稳定 delivery 串行；幂等重放先于当前成员检查。
            lock_key = int(request.delivery_id.hex[:15], 16)
            await session.execute(select(func.pg_advisory_xact_lock(lock_key)))
            existing = await session.get(AgentDelivery, request.delivery_id)
            if existing is not None:
                if existing.request_sha256 != digest:
                    raise DocumentError("delivery_idempotency_conflict", 409)
                return _delivery_payload(existing)
            if await session.get(Library, request.library_id, with_for_update=True) is None:
                raise DocumentError("library_not_found", 404)
            # 与成员写操作共用 Library 行锁，耗时文件核验不进入该锁事务。
            await _active_members(session, request.library_id, paper_ids)
            session.add(
                AgentDelivery(
                    delivery_id=request.delivery_id,
                    library_id=request.library_id,
                    generation_id=request.generation_id,
                    request_sha256=digest,
                    payload=payload,
                    payload_sha256=payload_sha256,
                )
            )
        return {**payload, "payload_sha256": payload_sha256}

    async def verified_source(self, library_id, delivery_id, alias):
        """名称：核验交付历史来源。用途：仅实际被正文引用的别名具有历史读取能力。"""
        from academic_evidence_qa.api.schemas import AgentDeliveryRequest

        payload = await load_agent_delivery(self.database, library_id, delivery_id)
        if alias not in {item["alias"] for item in payload["citations"]}:
            raise DocumentError("delivery_source_not_found", 404)
        original = next((item for item in payload["sources"] if item["alias"] == alias), None)
        if original is None:
            raise DocumentError("delivery_integrity_failed", 409)
        keys = {
            "evidence": ("retrieval_run_id", "evidence_id"),
            "block": (
                "paper_id",
                "document_version_id",
                "parse_snapshot_id",
                "block_id",
                "char_start",
                "char_end",
            ),
            "section": ("paper_id", "document_version_id", "parse_snapshot_id", "section_id"),
        }
        try:
            request = AgentDeliveryRequest.model_validate(
                {
                    "delivery_id": delivery_id,
                    "library_id": library_id,
                    "generation_id": payload["generation_id"],
                    "text": payload["text"],
                    "sources": [
                        {
                            "alias": alias,
                            "kind": original["kind"],
                            **{key: original[key] for key in keys[original["kind"]]},
                        }
                    ],
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DocumentError("delivery_integrity_failed", 409) from error
        sources, _ = await self._sources(
            library_id, request.generation_id, request.sources, historical=True
        )
        if sources[0] != original:
            raise DocumentError("delivery_integrity_failed", 409)
        return payload, original

    async def read_source(
        self, library_id, delivery_id, alias, *, offset=0, char_offset=0, max_characters=12000
    ):
        """名称：有界历史回读。用途：只读取登记区间或目录标题，不扩大为整块或整章。"""
        payload, source = await self.verified_source(library_id, delivery_id, alias)
        ranges = source["ranges"]
        if source["kind"] == "section":
            ranges = [
                {
                    "section_id": source["section_id"],
                    "reading_order": source["reading_order"],
                    "text": source["text"],
                    "locations": source["locations"],
                    "start": 0,
                    "end": len(source["text"]),
                }
            ]
        if offset >= len(ranges) or char_offset >= len(ranges[offset]["text"]):
            raise DocumentError("source_offset_invalid", 422)
        part = ranges[offset]
        text = part["text"][char_offset : char_offset + max_characters]
        end = char_offset + len(text)
        next_offset = offset if end < len(part["text"]) else offset + 1
        return {
            "contract": "m3_agent_source_v1",
            "delivery_id": str(delivery_id),
            "library_id": str(library_id),
            "generation_id": payload["generation_id"],
            **{
                key: source[key]
                for key in (
                    "alias",
                    "kind",
                    "source_id",
                    "paper_id",
                    "document_version_id",
                    "parse_snapshot_id",
                )
            },
            **{key: source[key] for key in ("retrieval_run_id", "evidence_id") if key in source},
            "historical": True,
            "items": [
                {
                    **part,
                    "text": text,
                    "start": part["start"] + char_offset,
                    "end": part["start"] + end,
                }
            ],
            "quality_flags": source["quality_flags"],
            "next_offset": next_offset if next_offset < len(ranges) else None,
            "next_char_offset": end if end < len(part["text"]) else 0,
        }
