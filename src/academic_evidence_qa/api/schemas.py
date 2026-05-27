"""模块名称：M1 管理 HTTP 契约。

用途：校验明确身份、修订与来源预算；与 ORM 分离。
核心设计：拒绝额外字段，不允许本地路径或隐式远端授权。
工程要点：分页与文本预算在请求边界固定，UUID 不能绕过库归属。
"""

from uuid import UUID
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


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
