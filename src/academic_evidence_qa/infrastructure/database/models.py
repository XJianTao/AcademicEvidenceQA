"""模块名称：文档、解析与索引数据实体

持久化文件、论文、不可变版本、库成员与最小待构建意图。

核心设计：
  - 全局哈希唯一；复合外键保证文件哈希和选版归属；每库独立修订。

工程要点：
  - 仅映射当前已实现的文档、解析和索引任务；
  - 版本不可变由迁移内数据库触发器保护。
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """本阶段实体的声明基类，仅用于类型化映射与迁移比较。"""


class ManagementRequest(Base):
    """名称：管理幂等记录。用途：请求与结果同事务冻结，重放先于修订核验。"""

    __tablename__ = "management_requests"
    idempotency_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_sha256: Mapped[str] = mapped_column(String(64))
    response_status: Mapped[int]
    response_body: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Library(Base):
    """论文库

    隔离每库成员修订和未来活动指针。

    Args:
        library_id: 默认生成 UUID；membership_revision 从零开始。
        active_generation_id: 可空的未来发布指针，当前不赋予索引语义。
    Returns:
        由事务服务锁定和更新的库实体，不作为外部响应。
    """

    __tablename__ = "libraries"
    __table_args__ = (CheckConstraint("membership_revision >= 0", name="ck_library_revision"),)
    library_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    membership_revision: Mapped[int] = mapped_column(BigInteger, server_default="0")
    # M1-C 才建立 generation 外键；本阶段不会修改或解释该指针。
    active_generation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            "index_generations.generation_id", name="fk_library_active_generation", use_alter=True
        )
    )
    building_generation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey(
            "index_generations.generation_id", name="fk_library_building_generation", use_alter=True
        )
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Paper(Base):
    """论文身份

    区分论文与正文文件，不自动合并元数据。

    Args:
        paper_id: 默认生成 UUID。
        title: 可空的展示元数据，不设置唯一约束。
    Returns:
        可被多个库复用、包含多个不可变版本的论文实体。
    """

    __tablename__ = "papers"
    paper_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class StoredFile(Base):
    """原件登记

    记录服务生成的对象身份与内容校验信息。

    Args:
        file_id, object_key: FileStore 生成的身份和相对键。
        file_sha256, size_bytes: 实际流式计算的哈希和字节数。
    Returns:
        全局唯一的文件记录；数据库触发器阻止覆盖/删除。
    """

    __tablename__ = "stored_files"
    __table_args__ = (
        UniqueConstraint("file_sha256", name="uq_file_sha256"),
        UniqueConstraint("object_key", name="uq_file_object_key"),
        UniqueConstraint("file_id", "file_sha256", name="uq_file_identity_hash"),
        CheckConstraint("size_bytes >= 0", name="ck_file_size"),
        CheckConstraint("file_sha256 ~ '^[0-9a-f]{64}$'", name="ck_file_sha256"),
    )
    file_id: Mapped[UUID] = mapped_column(primary_key=True)
    file_sha256: Mapped[str] = mapped_column(String(64))
    object_key: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentVersion(Base):
    """正文版本

    绑定固定论文身份与固定原件。

    Args:
        document_version_id: 默认生成 UUID。
        paper_id: 所属论文；file_id/file_sha256 共同引用原件。
    Returns:
        同哈希全局唯一且不可改写/删除的版本，选版由每库成员另行保存。
    """

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("file_sha256", name="uq_version_sha256"),
        UniqueConstraint("paper_id", "document_version_id", name="uq_version_paper_identity"),
        ForeignKeyConstraint(
            ["file_id", "file_sha256"],
            ["stored_files.file_id", "stored_files.file_sha256"],
            name="fk_version_file_hash",
        ),
    )
    document_version_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.paper_id"))
    file_id: Mapped[UUID]
    file_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentMetadata(Base):
    """名称：版本展示元数据。用途：独立登记原文件名，不改写不可变来源。

    Args:
        document_version_id: 已登记的固定版本身份，也是唯一主键。
        original_filenames: 按确认顺序保存首名和别名；历史未知可为空。
    Returns:
        只供展示和发现的名称集合；不参与原件路径、解析配置或索引身份。
    """

    __tablename__ = "document_metadata"
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id"), primary_key=True
    )
    original_filenames: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


class LibraryPaper(Base):
    """论文库成员

    表达每库独立待构建选版和软移除状态。

    Args:
        library_id, paper_id: 复合主键，同时保证唯一成员关系。
        selected_document_version_id: 必须属于同一 paper 的版本。
        parser_backend, parser_config_id, status: 明确配置与 active/removed 状态。
    Returns:
        下次构建意图，不表示 generation 已发布；软移除保留原件和版本。
    """

    __tablename__ = "library_papers"
    __table_args__ = (
        # 复合主键已经实现 UNIQUE(library_id, paper_id)，避免 PostgreSQL 折叠冗余约束。
        ForeignKeyConstraint(
            ["paper_id", "selected_document_version_id"],
            ["document_versions.paper_id", "document_versions.document_version_id"],
            name="fk_member_selected_paper_version",
        ),
        CheckConstraint("status IN ('active', 'removed')", name="ck_member_status"),
        CheckConstraint("length(trim(parser_backend)) > 0", name="ck_member_parser"),
        CheckConstraint("length(trim(parser_config_id)) > 0", name="ck_member_config"),
    )
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.paper_id"), primary_key=True)
    selected_document_version_id: Mapped[UUID]
    parser_backend: Mapped[str] = mapped_column(String(128))
    parser_config_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BuildIntent(Base):
    """名称：持久化索引任务。用途：保留成员变更意图并执行租约构建。

    保证成员变化不会仅保留在内存。

    Args:
        library_id, membership_revision: 实际成员修改后的库与修订，联合唯一。
        build_intent_id: 默认生成 UUID；created_at 由数据库记录。
    Returns:
        与成员/修订同事务提交的 Job，C2 原地扩展，不另建队列。
    """

    __tablename__ = "build_intents"
    __table_args__ = (
        UniqueConstraint(
            "library_id",
            "membership_revision",
            "request_sequence",
            name="uq_intent_library_revision",
        ),
        CheckConstraint("membership_revision >= 0", name="ck_intent_revision"),
        CheckConstraint(
            "state IN ('queued','running','succeeded','failed','cancelled')",
            name="ck_index_job_state",
        ),
        CheckConstraint("lease_version >= 0", name="ck_index_job_lease_version"),
    )
    build_intent_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    state: Mapped[str] = mapped_column(String(16), server_default="queued")
    stage: Mapped[str] = mapped_column(String(32), server_default="queued")
    cancel_requested: Mapped[bool] = mapped_column(server_default="false")
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_version: Mapped[int] = mapped_column(BigInteger, server_default="0")
    attempt_id: Mapped[UUID | None]
    generation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("index_generations.generation_id", name="fk_job_generation")
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"))
    membership_revision: Mapped[int] = mapped_column(BigInteger)
    request_sequence: Mapped[int] = mapped_column(BigInteger, server_default="0")
    retry_of_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("build_intents.build_intent_id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ParseTask(Base):
    """最小解析任务

    保存固定 attempt、远端标识与失败诊断；C3 增加数据库时钟租约及安全队列领取。

    Args:
        document_version_id: 固定输入版本，任务不会随成员选版变化。
        parser_backend, parser_config_id: 不含密钥的稳定配置身份。
        allow_remote_processing: 每次导入明确记录的外发授权。
    Returns:
        可在同一明确 job_id 下继续远端查询的持久化检查点。
    """

    __tablename__ = "parse_tasks"
    index_job_id: Mapped[UUID | None] = mapped_column(ForeignKey("build_intents.build_intent_id"))
    __table_args__ = (
        CheckConstraint(
            "state IN ('queued', 'executing', 'failed', 'cancelled', 'completed')",
            name="ck_parse_task_state",
        ),
        CheckConstraint("length(trim(parser_backend)) > 0", name="ck_parse_task_backend"),
        CheckConstraint("length(trim(parser_config_id)) > 0", name="ck_parse_task_config"),
    )
    parse_task_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id")
    )
    parser_backend: Mapped[str] = mapped_column(String(128))
    parser_config_id: Mapped[str] = mapped_column(String(128))
    allow_remote_processing: Mapped[bool]
    state: Mapped[str] = mapped_column(String(16), server_default="queued")
    stage: Mapped[str] = mapped_column(String(64), server_default="accepted")
    attempt_id: Mapped[UUID] = mapped_column(default=uuid4)
    executor_token: Mapped[UUID | None]
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    library_id: Mapped[UUID | None] = mapped_column(ForeignKey("libraries.library_id"))
    retry_of_job_id: Mapped[UUID | None] = mapped_column(ForeignKey("parse_tasks.parse_task_id"))
    parse_snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("parse_snapshots.parse_snapshot_id")
    )
    cancel_requested: Mapped[bool] = mapped_column(server_default="false")
    remote_task_id: Mapped[str | None] = mapped_column(String(512))
    error_code: Mapped[str | None] = mapped_column(String(128))
    checkpoint: Mapped[dict] = mapped_column(JSON, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ParseSnapshotRecord(Base):
    """不可变解析快照元数据

    文件产物已先由 FileStore 原子保存，本表仅在短事务内提交可回收的引用。
    """

    __tablename__ = "parse_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "parser_backend",
            "parser_config_id",
            "adapter_version",
            "normalized_artifact_sha256",
            name="uq_snapshot_content_identity",
        ),
        CheckConstraint(
            "quality_state IN ('usable', 'usable_with_warnings', 'rejected')",
            name="ck_snapshot_quality",
        ),
        CheckConstraint("input_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_input_sha256"),
        CheckConstraint("raw_artifact_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_raw_sha256"),
        CheckConstraint(
            "normalized_artifact_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_normalized_sha256"
        ),
    )
    parse_snapshot_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id")
    )
    input_sha256: Mapped[str] = mapped_column(String(64))
    parser_backend: Mapped[str] = mapped_column(String(128))
    parser_config_id: Mapped[str] = mapped_column(String(128))
    adapter_version: Mapped[str] = mapped_column(String(128))
    remote_version: Mapped[str] = mapped_column(String(256), server_default="unknown")
    actual_resource_config: Mapped[dict] = mapped_column(JSON)
    quality_state: Mapped[str] = mapped_column(String(32))
    quality_metrics: Mapped[dict] = mapped_column(JSON)
    raw_artifact_key: Mapped[str] = mapped_column(String(256))
    raw_artifact_sha256: Mapped[str] = mapped_column(String(64))
    normalized_artifact_key: Mapped[str] = mapped_column(String(256))
    normalized_artifact_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ParseSection(Base):
    """不可变快照章节；阅读顺序在同一快照内唯一。"""

    __tablename__ = "parse_sections"
    __table_args__ = (
        UniqueConstraint("parse_snapshot_id", "reading_order", name="uq_section_reading_order"),
        CheckConstraint("reading_order >= 0", name="ck_section_reading_order"),
    )
    section_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    parse_snapshot_id: Mapped[UUID] = mapped_column(ForeignKey("parse_snapshots.parse_snapshot_id"))
    title: Mapped[str | None] = mapped_column(String(4096))
    level: Mapped[int | None]
    reading_order: Mapped[int]
    locations: Mapped[list] = mapped_column(JSON)


class ParseBlock(Base):
    """不可变快照块；正文、位置和质量标记与所属快照共同冻结。"""

    __tablename__ = "parse_blocks"
    __table_args__ = (
        UniqueConstraint("parse_snapshot_id", "reading_order", name="uq_block_reading_order"),
        CheckConstraint("reading_order >= 0", name="ck_block_reading_order"),
        CheckConstraint("length(trim(block_type)) > 0", name="ck_block_type"),
    )
    block_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    parse_snapshot_id: Mapped[UUID] = mapped_column(ForeignKey("parse_snapshots.parse_snapshot_id"))
    block_type: Mapped[str] = mapped_column(String(128))
    reading_order: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    locations: Mapped[list] = mapped_column(JSON)
    raw_provenance: Mapped[dict] = mapped_column(JSON)
    quality_flags: Mapped[list] = mapped_column(JSON)


class IndexGeneration(Base):
    """名称：索引代。用途：冻结一次完整构建的输入、统计与 collection 绑定。

    Args:
        generation_id, attempt_id: 两种独立且不可复用的构建身份。
        manifest: 固定成员、chunk、模型与词法统计的完整清单。
    Returns:
        仅在 building 阶段可发布或失败的数据库记录。
    """

    __tablename__ = "index_generations"
    __table_args__ = (
        UniqueConstraint("collection_name", name="uq_generation_collection"),
        UniqueConstraint("attempt_id", name="uq_generation_attempt"),
        CheckConstraint(
            "state IN ('building', 'published', 'retired', 'failed')", name="ck_generation_state"
        ),
        CheckConstraint("membership_revision >= 0", name="ck_generation_revision"),
    )
    generation_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"))
    attempt_id: Mapped[UUID]
    membership_revision: Mapped[int] = mapped_column(BigInteger)
    expected_active_generation_id: Mapped[UUID | None]
    collection_name: Mapped[str] = mapped_column(String(160))
    state: Mapped[str] = mapped_column(String(16), server_default="building")
    manifest: Mapped[dict] = mapped_column(JSON, server_default="{}")
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IndexChunk(Base):
    """名称：冻结索引块。用途：保存可回源的原块区间与派生检索文本。"""

    __tablename__ = "index_chunks"
    __table_args__ = (
        UniqueConstraint("generation_id", "reading_order", name="uq_index_chunk_order"),
        CheckConstraint("embedding_tokens > 0 AND embedding_tokens <= 600", name="ck_chunk_tokens"),
        CheckConstraint("reranker_tokens > 0", name="ck_chunk_reranker_tokens"),
    )
    chunk_id: Mapped[UUID] = mapped_column(primary_key=True)
    generation_id: Mapped[UUID] = mapped_column(ForeignKey("index_generations.generation_id"))
    parse_snapshot_id: Mapped[UUID] = mapped_column(ForeignKey("parse_snapshots.parse_snapshot_id"))
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id")
    )
    paper_id: Mapped[UUID] = mapped_column(ForeignKey("papers.paper_id"))
    reading_order: Mapped[int]
    block_type: Mapped[str] = mapped_column(String(128))
    retrieval_text: Mapped[str] = mapped_column(Text)
    source_ranges: Mapped[list] = mapped_column(JSON)
    source_locations: Mapped[list] = mapped_column(JSON)
    quality_flags: Mapped[list] = mapped_column(JSON)
    embedding_tokens: Mapped[int]
    reranker_tokens: Mapped[int]


class RetrievalRun(Base):
    """名称：固定取证记录。用途：持久 scope/封装及请求，回读不依赖旧向量索引。"""

    __tablename__ = "retrieval_runs"
    retrieval_run_id: Mapped[UUID] = mapped_column(primary_key=True)
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"))
    generation_id: Mapped[UUID] = mapped_column(ForeignKey("index_generations.generation_id"))
    request: Mapped[dict] = mapped_column(JSON)
    pack: Mapped[dict] = mapped_column(JSON)
    pack_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvidenceRecord(Base):
    """名称：证据注册记录。用途：将全局证据 ID 绑定唯一取证记录及不可变原文区间。"""

    __tablename__ = "evidence_registry"
    evidence_id: Mapped[UUID] = mapped_column(primary_key=True)
    retrieval_run_id: Mapped[UUID] = mapped_column(ForeignKey("retrieval_runs.retrieval_run_id"))
    payload: Mapped[dict] = mapped_column(JSON)
    payload_sha256: Mapped[str] = mapped_column(String(64))


class AnswerArtifact(Base):
    """名称：答案与幂等屏障。用途：外部调用前先占 key，未知结果禁止重复付费生成。"""

    __tablename__ = "answer_artifacts"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_answer_idempotency"),
        CheckConstraint(
            "state IN ('pending', 'completed', 'failed', 'unknown')", name="ck_answer_state"
        ),
    )
    answer_id: Mapped[UUID] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_sha256: Mapped[str] = mapped_column(String(64))
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"))
    retrieval_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("retrieval_runs.retrieval_run_id")
    )
    state: Mapped[str] = mapped_column(String(16))
    configuration: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict | None] = mapped_column(JSON)
    draft: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
