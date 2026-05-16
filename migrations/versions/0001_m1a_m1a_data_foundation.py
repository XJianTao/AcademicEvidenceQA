"""模块名称：m1a_data_foundation。

版本化数据库结构变更。

核心设计：
  - 明确升级与降级，不依赖运行时 ORM 自动建表。

工程要点：
  - 降级可能删除本迁移的数据，仅由管理者显式执行。
"""

from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa


revision: str = "0001_m1a"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级数据底座。

    创建六表及来源不可变保护。

    Args:
        无；连接与事务由 Alembic 管理，目标库由调用方显式选择。
    Returns:
        None；迁移版本由 Alembic 记录，已到 head 时不重复执行本函数。
    """
    # 实际建表迁移独立于 ORM；约束命名用于真实 PostgreSQL 一致性核对。
    op.create_table(
        "libraries",
        sa.Column("library_id", sa.Uuid(), nullable=False),
        sa.Column("membership_revision", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("active_generation_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("membership_revision >= 0", name="ck_library_revision"),
        sa.PrimaryKeyConstraint("library_id"),
    )
    op.create_table(
        "papers",
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("paper_id"),
    )
    op.create_table(
        "stored_files",
        sa.Column("file_id", sa.Uuid(), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column("object_key", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("file_sha256 ~ '^[0-9a-f]{64}$'", name="ck_file_sha256"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_file_size"),
        sa.PrimaryKeyConstraint("file_id"),
        sa.UniqueConstraint("file_id", "file_sha256", name="uq_file_identity_hash"),
        sa.UniqueConstraint("file_sha256", name="uq_file_sha256"),
        sa.UniqueConstraint("object_key", name="uq_file_object_key"),
    )
    op.create_table(
        "build_intents",
        sa.Column("build_intent_id", sa.Uuid(), nullable=False),
        sa.Column("library_id", sa.Uuid(), nullable=False),
        sa.Column("membership_revision", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("membership_revision > 0", name="ck_intent_revision"),
        sa.ForeignKeyConstraint(
            ["library_id"],
            ["libraries.library_id"],
        ),
        sa.PrimaryKeyConstraint("build_intent_id"),
        sa.UniqueConstraint("library_id", "membership_revision", name="uq_intent_library_revision"),
    )
    op.create_table(
        "document_versions",
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("file_id", sa.Uuid(), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["file_id", "file_sha256"],
            ["stored_files.file_id", "stored_files.file_sha256"],
            name="fk_version_file_hash",
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.paper_id"],
        ),
        sa.PrimaryKeyConstraint("document_version_id"),
        sa.UniqueConstraint("file_sha256", name="uq_version_sha256"),
        sa.UniqueConstraint("paper_id", "document_version_id", name="uq_version_paper_identity"),
    )
    op.create_table(
        "library_papers",
        sa.Column("library_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("selected_document_version_id", sa.Uuid(), nullable=False),
        sa.Column("parser_backend", sa.String(length=128), nullable=False),
        sa.Column("parser_config_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("status IN ('active', 'removed')", name="ck_member_status"),
        sa.CheckConstraint("length(trim(parser_backend)) > 0", name="ck_member_parser"),
        sa.CheckConstraint("length(trim(parser_config_id)) > 0", name="ck_member_config"),
        sa.ForeignKeyConstraint(
            ["library_id"],
            ["libraries.library_id"],
        ),
        sa.ForeignKeyConstraint(
            ["paper_id", "selected_document_version_id"],
            ["document_versions.paper_id", "document_versions.document_version_id"],
            name="fk_member_selected_paper_version",
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.paper_id"],
        ),
        sa.PrimaryKeyConstraint("library_id", "paper_id"),
    )
    # 不可变身份由数据库保护，绕过 service 的直接 SQL 也不能改写/删除来源。
    op.execute("""
        CREATE FUNCTION reject_immutable_source_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'immutable_source' USING ERRCODE = '23514';
        END;
        $$
    """)
    for table_name in ("document_versions", "stored_files"):
        op.execute(f"""
            CREATE TRIGGER reject_source_change BEFORE UPDATE OR DELETE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION reject_immutable_source_change()
        """)


def downgrade() -> None:
    """降级数据底座

    显式移除本版本表和触发器函数。

    Args:
        无；仅由管理员显式调用 Alembic 降级，本轮验收不执行。
    Returns:
        None；这是删除本版本数据的管理操作，不能当作测试清理入口。
    """
    # 管理员显式降级才撤销来源保护；本轮验证不执行降级、不删除数据。
    op.drop_table("library_papers")
    op.drop_table("document_versions")
    op.drop_table("build_intents")
    op.drop_table("stored_files")
    op.drop_table("papers")
    op.drop_table("libraries")
    op.execute("DROP FUNCTION reject_immutable_source_change()")
