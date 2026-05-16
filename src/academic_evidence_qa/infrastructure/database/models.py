"""模块：M1-A 数据实体

持久化文件、论文、不可变版本、库成员与最小待构建意图。

核心设计：
  - 全局哈希唯一；复合外键保证文件哈希和选版归属；每库独立修订。

工程要点：
  - 不预建解析、索引或 Worker 实体；
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
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """本阶段实体的声明基类，仅用于类型化映射与迁移比较。"""


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
    active_generation_id: Mapped[UUID | None]
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
    """待构建意图

    保证成员变化不会仅保留在内存。

    Args:
        library_id, membership_revision: 实际成员修改后的库与修订，联合唯一。
        build_intent_id: 默认生成 UUID；created_at 由数据库记录。
    Returns:
        与成员/修订同事务提交的记录，M1-C 才接入领取/合并/执行。
    """

    __tablename__ = "build_intents"
    __table_args__ = (
        UniqueConstraint("library_id", "membership_revision", name="uq_intent_library_revision"),
        CheckConstraint("membership_revision > 0", name="ck_intent_revision"),
    )
    build_intent_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    library_id: Mapped[UUID] = mapped_column(ForeignKey("libraries.library_id"))
    membership_revision: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
