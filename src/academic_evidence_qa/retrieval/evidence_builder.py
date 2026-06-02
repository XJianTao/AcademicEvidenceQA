"""模块名称：原文证据组装与封装预算。

用途：复用候选检索，按固定块区间扩展、重叠去重并保存证据 registry。
核心设计：只扩展同块上下文及解析器明确的相关块；不猜公式/图表的语义关系。
工程要点：同请求复用已核验文档；完整 JSON 计量，超限舍弃完整证据而非截断 JSON。
"""

from uuid import UUID
from sqlalchemy import select

from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.contracts.rag import Evidence, EvidenceRange, EvidencePack
from academic_evidence_qa.infrastructure.database.models import (
    ParseBlock,
    RetrievalRun,
    EvidenceRecord,
)
from academic_evidence_qa.infrastructure.generation_client import estimated_tokens, serialize
from academic_evidence_qa.ingestion.index_builder import content_hash


def measure_pack(pack):
    """计量包含标签、别名、位置、分页和状态的完整封装，迭代消除计数字段自引用。"""
    for _ in range(5):
        payload = pack.model_dump(mode="json")
        size, tokens = len(serialize(payload)), estimated_tokens(payload)
        if (size, tokens) == (pack.serialized_bytes, pack.token_count):
            break
        pack.serialized_bytes, pack.token_count = size, tokens


def select_evidence(result, blocks_by_snapshot, *, token_budget, max_bytes):
    """名称：选择原文证据。用途：按候选次序扩展240字符，合并重叠原块区间。

    Args:
        result: 已核验固定来源的候选响应，windows 仅用于重排而非原文定位。
        blocks_by_snapshot: 来自同次快照核验后的数据库块映射。
        token_budget, max_bytes: 完整封装限制，未知 tokenizer 保守估算。
    Returns:
        原文 EvidencePack；超长表保留原块分页读取能力，不伪造精确行框。
    """
    pack = EvidencePack(
        scope=result.scope,
        evidence=[],
        status="ready",
        degraded_reasons=list(result.statuses),
        timings=result.timings.copy(),
    )
    intervals = {}
    priority = []
    candidate_flags = {}
    for candidate in result.candidates:
        blocks = blocks_by_snapshot[candidate.parse_snapshot_id]
        for source in candidate.source_ranges:
            order = source["block_reading_order"]
            block = blocks.get(order)
            if block is None:
                raise DocumentError("candidate_source_mismatch", 409)
            start, end = source["start"], source["end"]
            if not (
                type(start) is int and type(end) is int and 0 <= start <= end <= len(block.text)
            ):
                raise DocumentError("candidate_range_invalid", 409)
            orders = [order]
            related = block.raw_provenance.get("related_block_reading_orders", [])
            if isinstance(related, list):
                orders += [o for o in related if type(o) is int and o in blocks][:4]
            for selected_order in dict.fromkeys(orders):
                selected = blocks[selected_order]
                left, right = (
                    (max(0, start - 240), min(len(selected.text), end + 240))
                    if selected_order == order
                    else (0, len(selected.text))
                )
                key = (
                    candidate.paper_id,
                    candidate.document_version_id,
                    candidate.parse_snapshot_id,
                    selected_order,
                )
                if key not in intervals:
                    intervals[key] = []
                    priority.append(key)
                intervals[key].append((left, right))
                candidate_flags.setdefault(key, set()).update(candidate.quality_flags)
    for key in priority:
        paper, version, snapshot, order = key
        block = blocks_by_snapshot[snapshot][order]
        merged = []
        for left, right in sorted(intervals[key]):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        for left, right in merged:
            if left == right:
                continue
            evidence = Evidence(
                alias=f"E{len(pack.evidence) + 1}",
                paper_id=paper,
                document_version_id=version,
                parse_snapshot_id=snapshot,
                quality_flags=sorted(candidate_flags[key] | set(block.quality_flags)),
                ranges=[
                    EvidenceRange(
                        block_id=block.block_id,
                        block_reading_order=order,
                        start=left,
                        end=right,
                        text=block.text[left:right],
                        block_type=block.block_type,
                        locations=block.locations,
                        quality_flags=block.quality_flags,
                    )
                ],
            )
            pack.evidence.append(evidence)
            measure_pack(pack)
            if pack.token_count > token_budget or pack.serialized_bytes > max_bytes:
                pack.evidence.pop()
                pack.truncated = True
                pack.continuation = "read_original_block_with_source_pagination"
    pack.uncovered_paper_ids = [
        UUID(m["paper_id"])
        for m in pack.scope.members
        if UUID(m["paper_id"]) not in {e.paper_id for e in pack.evidence}
    ]
    pack.status = (
        "ready"
        if pack.evidence
        else "budget_exhausted"
        if result.candidates
        else "insufficient_evidence"
    )
    measure_pack(pack)
    # 最终状态/覆盖字段也占预算；按整个证据重新选择，绝不切割封装。
    while pack.evidence and (pack.token_count > token_budget or pack.serialized_bytes > max_bytes):
        pack.evidence.pop()
        pack.truncated = True
        pack.status = "ready" if pack.evidence else "budget_exhausted"
        pack.uncovered_paper_ids = [
            UUID(m["paper_id"])
            for m in pack.scope.members
            if UUID(m["paper_id"]) not in {e.paper_id for e in pack.evidence}
        ]
        measure_pack(pack)
    return pack


class EvidenceBuilder:
    """名称：取证服务。用途：检索/来源复用、最终成员复核和原子保存映射。

    Args:
        retrieval: 既有唯一检索管线，包含数据库/有界 I/O 与模型 owner。
        max_bytes: 证据封装硬字节上限。
    Returns:
        retrieve 返回已持久化 EvidencePack，不调用生成式模型。
    """

    def __init__(self, retrieval, *, max_bytes=48000):
        self.retrieval, self.max_bytes = retrieval, max_bytes

    async def retrieve(self, request, *, source_cache=None):
        cache = {} if source_cache is None else source_cache
        result = await self.retrieval.retrieve(request, source_cache=cache)
        if "recall_unavailable" in result.statuses:
            raise DocumentError("recall_unavailable", 503)
        blocks = {}
        async with self.retrieval.database.sessions() as session:
            for snapshot_id in cache:
                rows = (
                    await session.scalars(
                        select(ParseBlock).where(ParseBlock.parse_snapshot_id == snapshot_id)
                    )
                ).all()
                blocks[snapshot_id] = {row.reading_order: row for row in rows}
        pack = select_evidence(
            result, blocks, token_budget=request.evidence_token_budget, max_bytes=self.max_bytes
        )
        await self.retrieval._verify_active_members(pack.scope)
        await self.save(pack, request)
        await self.retrieval._verify_active_members(pack.scope)
        return pack

    async def save(self, pack, request):
        """同事务保存完整封装及 registry，任何失败不得留下部分证据映射。"""
        payload = pack.model_dump(mode="json")
        async with self.retrieval.database.transaction() as session:
            session.add(
                RetrievalRun(
                    retrieval_run_id=pack.retrieval_run_id,
                    library_id=pack.scope.library_id,
                    generation_id=pack.scope.generation_id,
                    request=request.model_dump(mode="json"),
                    pack=payload,
                    pack_sha256=content_hash(payload),
                )
            )
            await session.flush()
            for evidence in pack.evidence:
                payload = evidence.model_dump(mode="json")
                session.add(
                    EvidenceRecord(
                        evidence_id=evidence.evidence_id,
                        retrieval_run_id=pack.retrieval_run_id,
                        payload=payload,
                        payload_sha256=content_hash(payload),
                    )
                )
