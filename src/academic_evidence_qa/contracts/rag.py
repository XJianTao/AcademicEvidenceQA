"""模块名称：固定检索、证据与一次问答契约。

用途：声明固定范围、候选来源、持久证据及答案校验状态。
核心设计：一次请求绑定不可变代及成员修订，分数与失败分别表达。
工程要点：严格预算；来源校验失败不得作为模型降级返回。
"""

from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class RetrievalRequest(BaseModel):
    """名称：候选请求。用途：限定库、论文及有界召回，不静默裁剪问题。"""

    model_config = ConfigDict(extra="forbid")
    library_id: UUID
    generation_id: UUID | None = None
    paper_ids: list[UUID] = Field(default_factory=list, max_length=20)
    query: str = Field(min_length=1, max_length=4000)
    mode: Literal["dense", "bm25", "rrf", "hybrid_rerank"] = "hybrid_rerank"
    recall_limit: int = Field(default=40, ge=1, le=40)
    limit: int = Field(default=10, ge=1, le=60)


class FixedScope(BaseModel):
    """名称：固定范围。用途：保留本次实际 collection、修订与来源成员。"""

    library_id: UUID
    generation_id: UUID
    membership_revision: int
    collection_name: str
    members: list[dict[str, str]]


class RerankWindow(BaseModel):
    """名称：重排窗口。用途：记录父检索文本中的字符半开区间与实际 pair 长度。"""

    char_start: int
    char_end: int
    pair_tokens: int
    score: float | None = None


class Candidate(BaseModel):
    """名称：来源候选。用途：贯穿固定身份、原块映射及各路排序证据。

    Args:
        source_ranges, source_locations: 已核验父 chunk 的完整来源映射。
        windows: retrieval_text 中的区间；通过父 source_ranges 回源，不冒充原块偏移。
    Returns:
        M2a-1 候选；分数不代表正确概率。
    """

    chunk_id: UUID
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    generation_id: UUID
    retrieval_text: str
    source_ranges: list[dict[str, Any]]
    source_locations: list[dict[str, Any]]
    quality_flags: list[str]
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    windows: list[RerankWindow] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """名称：候选响应。用途：区分零命中、单路故障与不完整重排。"""

    contract: Literal["m2a1_candidates"] = "m2a1_candidates"
    scope: FixedScope
    candidates: list[Candidate]
    statuses: list[str]
    configuration: dict[str, Any]
    timings: dict[str, float]


class EvidenceRequest(RetrievalRequest):
    """名称：正式取证请求。用途：在候选范围上声明封装预算，不依赖生成模型。"""

    evidence_token_budget: int = Field(default=6000, ge=1, le=16000)


class AnswerRequest(EvidenceRequest):
    """名称：有界问答请求。用途：固定范围，最多两次补检索及一次同证据引用修复。"""

    max_output_tokens: int = Field(default=1200, ge=1, le=4096)
    max_supplemental_retrievals: int = Field(default=2, ge=0, le=2)
    allow_citation_repair: bool = True


class EvidenceRange(BaseModel):
    """名称：原块区间。用途：保存数据库块身份和原文半开区间，非检索窗口偏移。"""

    block_id: UUID
    block_reading_order: int
    start: int
    end: int
    text: str
    block_type: str
    locations: list[dict[str, Any]]
    quality_flags: list[str]


class Evidence(BaseModel):
    """名称：持久证据。用途：跨重启回读固定来源，质量标记不得提升。"""

    evidence_id: UUID = Field(default_factory=uuid4)
    alias: str
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    ranges: list[EvidenceRange]
    quality_flags: list[str]


class EvidencePack(BaseModel):
    """名称：证据封装。用途：报告预算、固定范围与未覆盖情况，不声称全文覆盖。"""

    retrieval_run_id: UUID = Field(default_factory=uuid4)
    scope: FixedScope
    evidence: list[Evidence]
    status: Literal["ready", "insufficient_evidence", "budget_exhausted"]
    budget_mode: Literal["estimated"] = "estimated"
    token_count: int = 0
    serialized_bytes: int = 0
    truncated: bool = False
    continuation: str | None = None
    uncovered_paper_ids: list[UUID] = Field(default_factory=list)
    degraded_reasons: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    supplemental_retrievals: list[dict[str, Any]] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)


class EvidenceReadRequest(BaseModel):
    """名称：证据回读。用途：绑定库、取证记录及可选历史答案，ID 不作为权限。"""

    model_config = ConfigDict(extra="forbid")
    library_id: UUID
    retrieval_run_id: UUID
    evidence_id: UUID
    answer_id: UUID | None = None
    offset: int = Field(default=0, ge=0, le=100000)
    char_offset: int = Field(default=0, ge=0, le=10000000)
    max_characters: int = Field(default=12000, ge=1, le=32000)


class Citation(BaseModel):
    """名称：答案引用。用途：固定本次别名与已持久化证据的映射。"""

    alias: str
    evidence_id: UUID


class Answer(BaseModel):
    """名称：答案交付。用途：完整性与语义支持分开，无效草稿不作为 final 交付。"""

    answer_id: UUID
    retrieval_run_id: UUID
    status: Literal["answered", "insufficient_evidence", "budget_exhausted", "citation_failed"]
    text: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    citation_check_status: Literal["passed", "failed", "not_run"] = "not_run"
    support_check_status: Literal["not_run"] = "not_run"
    diagnostics: list[str] = Field(default_factory=list)
    repair_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    budget_mode: Literal["estimated"] = "estimated"
    stream_status: Literal["not_enabled"] = "not_enabled"
