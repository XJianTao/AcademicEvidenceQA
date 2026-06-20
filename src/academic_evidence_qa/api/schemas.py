"""模块名称：M1 管理 HTTP 契约。

用途：校验明确身份、修订与来源预算；与 ORM 分离。
核心设计：拒绝额外字段，不允许本地路径或隐式远端授权。
工程要点：分页与文本预算在请求边界固定，UUID 不能绕过库归属。
"""

from uuid import UUID
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictRequest(BaseModel):
    """名称：请求基类。用途：拒绝未约定字段，避免静默忽略配置。"""

    model_config = ConfigDict(extra="forbid")


class LibraryRequest(StrictRequest):
    """建库无需附加元数据。"""


class UploadRequest(StrictRequest):
    """上传元字段；文件哈希由接收层计算，外发必须显式选择。"""

    library_id: UUID
    expected_membership_revision: int = Field(ge=0)
    paper_id: UUID | None = None
    parser_backend: Literal["mineru_remote", "docling"]
    allow_remote_processing: bool


class MemberRequest(StrictRequest):
    """成员选版，不创建未授权远端任务。"""

    selected_document_version_id: UUID
    expected_membership_revision: int = Field(ge=0)
    parser_backend: str = Field(min_length=1, max_length=128)
    parser_config_id: str = Field(min_length=1, max_length=128)


class RevisionRequest(StrictRequest):
    """常规重建或移除所固定的库修订。"""

    expected_membership_revision: int = Field(ge=0)


class SourceRequest(StrictRequest):
    """固定来源：已发布 generation 或显式管理预览。"""

    library_id: UUID
    generation_id: UUID | None = None
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    preview: bool = False
    section_id: UUID | None = None
    block_id: UUID | None = None
    offset: int = Field(default=0, ge=0, le=100000)
    char_offset: int = Field(default=0, ge=0, le=10000000)
    limit: int = Field(default=20, ge=1, le=100)
    max_characters: int = Field(default=12000, ge=1, le=32000)


class PaperFindRequest(StrictRequest):
    """名称：固定发布元数据查询。用途：仅在宿主允许池内筛选和稳定续页。"""

    library_id: UUID
    generation_id: UUID
    paper_ids: list[UUID] = Field(min_length=1, max_length=20)
    query: str = Field(default="", max_length=200, strict=True)
    limit: int = Field(default=10, ge=1, le=20, strict=True)
    cursor: str | None = Field(default=None, max_length=2048, strict=True)

    @field_validator("paper_ids")
    @classmethod
    def reject_duplicate_papers(cls, value):
        """允许池必须显式且无重复，不能把无效输入静默规范成成功。"""
        if len(value) != len(set(value)):
            raise ValueError("duplicate_paper_ids")
        return value


class LibraryDiscoverRequest(StrictRequest):
    """名称：显式全库发现。用途：在固定发布代内搜索所有仍活动的论文元数据。"""

    generation_id: UUID
    query: str = Field(default="", max_length=200, strict=True)
    limit: int = Field(default=10, ge=1, le=20, strict=True)
    cursor: str | None = Field(default=None, max_length=2048, strict=True)


class AgentEvidenceSource(StrictRequest):
    """登记真实 retrieval evidence，不接收客户端提供的正文或版本覆盖。"""

    kind: Literal["evidence"]
    alias: str = Field(pattern=r"^E[1-9][0-9]*$", max_length=16, strict=True)
    retrieval_run_id: UUID
    evidence_id: UUID


class AgentBlockSource(StrictRequest):
    """登记实际读取的原块半开区间，不伪造 retrieval evidence。"""

    kind: Literal["block"]
    alias: str = Field(pattern=r"^E[1-9][0-9]*$", max_length=16, strict=True)
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    block_id: UUID
    char_start: int = Field(ge=0, le=10000000, strict=True)
    char_end: int = Field(gt=0, le=10000000, strict=True)


class AgentSectionSource(StrictRequest):
    """只登记目录标题及其位置；章节正文须逐块登记。"""

    kind: Literal["section"]
    alias: str = Field(pattern=r"^E[1-9][0-9]*$", max_length=16, strict=True)
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    section_id: UUID


AgentSource = Annotated[
    AgentEvidenceSource | AgentBlockSource | AgentSectionSource, Field(discriminator="kind")
]


class AgentDeliveryRequest(StrictRequest):
    """名称：可信宿主交付登记。用途：固定正文、全部来源和独立 delivery 身份。"""

    delivery_id: UUID
    library_id: UUID
    generation_id: UUID
    text: str = Field(min_length=1, max_length=32000, strict=True)
    sources: list[AgentSource] = Field(min_length=1, max_length=64)

    @field_validator("sources")
    @classmethod
    def reject_duplicate_aliases(cls, value):
        """别名必须在整个交付内唯一，多次取证不得覆盖先前映射。"""
        if len({item.alias for item in value}) != len(value):
            raise ValueError("duplicate_citation_alias")
        return value


class JobResponse(BaseModel):
    """持久任务响应；解析 succeeded 不表示库 generation 发布。"""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    job_type: Literal["parse", "index"]
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    stage: str
    attempt_id: UUID | None
    error_code: str | None
    cancel_requested: bool
    retry_of_job_id: UUID | None
    retry_job_ids: list[UUID]
    remote_stage: Literal["submit", "upload", "poll", "download"] | None = None
    failure_reason: str | None = Field(
        default=None,
        pattern=r"^(http_[1-5][0-9]{2}|proxy_connect_failed|connect_timeout|read_timeout|tls_failed|connection_failed|invalid_response|remote_rejection|transport_failure)$",
    )
    library_id: UUID | None
    document_version_id: UUID | None = None
    parse_snapshot_id: UUID | None = None
    index_job_ids: list[UUID] = Field(default_factory=list)
    generation_id: UUID | None = None
    membership_revision: int | None = None


class StrictResponse(BaseModel):
    """名称：响应基类。用途：拒绝服务意外泄漏字段，不序列化 ORM。"""

    model_config = ConfigDict(extra="forbid")


class LibraryResponse(StrictResponse):
    """建库的冻结身份与初始修订。"""

    library_id: UUID
    membership_revision: int


class UploadResponse(LibraryResponse):
    """受理、原件登记或完整复用，分别表达解析和发布身份。"""

    paper_id: UUID
    document_version_id: UUID
    file_sha256: str
    status: Literal["accepted", "registered", "reused"]
    job_id: UUID | None
    index_job_id: UUID | None
    parse_job_id: UUID | None = None
    parse_snapshot_id: UUID | None = None
    generation_id: UUID | None = None


class MemberResponse(LibraryResponse):
    """成员意图响应，无变化时不产生新的构建意图。"""

    paper_id: UUID
    selected_document_version_id: UUID
    status: Literal["active", "removed"]
    changed: bool
    index_job_id: UUID | None


class JobMutationResponse(StrictResponse):
    """重建、取消与重试响应；终态继续通过 GET job 查询。"""

    job_id: UUID
    job_type: Literal["parse", "index"]
    library_id: UUID | None = None
    membership_revision: int | None = None
    cancel_requested: bool | None = None
    retry_of_job_id: UUID | None = None
    attempt_id: UUID | None = None
    index_job_id: UUID | None = None


class PaperResponse(StrictResponse):
    """论文白名单元数据；不返回本地文件位置。"""

    paper_id: UUID
    title: str | None
    document_version_ids: list[UUID]
    status: Literal["active", "removed"] | None = None
    selected_document_version_id: UUID | None = None
    published_source: dict | None = None
    pending: bool | None = None
    parser_backend: str | None = None
    parser_config_id: str | None = None


class PaperListResponse(StrictResponse):
    """有界论文分页及待发布/活动身份。"""

    items: list[PaperResponse]
    next_offset: int | None
    library_id: UUID | None
    membership_revision: int | None
    active_generation_id: UUID | None


class PaperFindItemResponse(StrictResponse):
    """名称：发布论文消歧项。用途：用固定来源身份补足不可靠展示元数据。"""

    paper_id: UUID
    title: str | None
    filename: str | None
    filename_aliases: list[str] = Field(default_factory=list)
    document_version_id: UUID
    parse_snapshot_id: UUID
    publication_status: Literal["published"]
    metadata_warnings: list[str]
    disambiguation: dict


class PaperFindResponse(StrictResponse):
    """固定范围内查询结果；无匹配是成功空页，不代替范围错误。"""

    contract: Literal["m3_paper_find_v1"]
    library_id: UUID
    generation_id: UUID
    items: list[PaperFindItemResponse]
    next_cursor: str | None
    has_more: bool


class LibraryDiscoverResponse(PaperFindResponse):
    """显式库级发现响应，不能被当作当前取证允许池。"""

    contract: Literal["m3_library_discovery_v1"]


class AgentDeliveryResponse(StrictResponse):
    """经来源核验的登记结果；最终交付以宿主 append-only final 为准。"""

    contract: Literal["m3_agent_delivery_v1"]
    delivery_id: UUID
    library_id: UUID
    generation_id: UUID
    text: str
    citations: list[dict]
    sources: list[dict]
    citation_check_status: Literal["passed"]
    support_check_status: Literal["not_run"]
    registration_status: Literal["registered"]
    payload_sha256: str


class SourceItemResponse(StrictResponse):
    """目录项或原块白名单；位置保留解析契约的 provenance。"""

    reading_order: int
    locations: list[dict]
    section_id: UUID | None = None
    title: str | None = None
    level: int | None = None
    block_id: UUID | None = None
    block_type: str | None = None
    text: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    quality_flags: list[str] | None = None


class SourceResponse(StrictResponse):
    """固定库/版本/快照的分页来源，不是 M2 Evidence registry。"""

    library_id: UUID
    generation_id: UUID | None
    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    publication_state: Literal["published", "unpublished"]
    items: list[SourceItemResponse]
    next_offset: int | None
    next_char_offset: int | None = None


class EncodingRequest(BaseModel):
    """内部编码只接受小批次；总字节预算由路由在 JSON 解码前核验。"""

    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=4)
    configuration_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
