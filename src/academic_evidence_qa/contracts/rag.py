"""模块名称：M2a-1 候选检索契约。

用途：声明固定范围、候选来源与排序状态，不提前定义证据或答案。
核心设计：一次请求绑定不可变代及成员修订，分数与失败分别表达。
工程要点：严格预算；来源校验失败不得作为模型降级返回。
"""

from typing import Any, Literal
from uuid import UUID

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
