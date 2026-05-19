"""模块名称：m1b_parse_snapshots。

受限解析任务与不可变解析快照数据库结构。

核心设计：任务可更新检查点，成功快照、章节和块由数据库触发器禁止改写或删除。

工程要点：迁移不创建 Worker、索引 generation 或业务 HTTP API。
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_m1b"
down_revision: str | None = "0001_m1a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建解析任务、快照、章节与块。

    Args:
        无；由 Alembic 在操作者显式选定的数据库中执行。
    Returns:
        None；快照不可变规则在此迁移内落到 PostgreSQL。
    """
    op.create_table(
        "parse_tasks",
        sa.Column("parse_task_id", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("parser_backend", sa.String(length=128), nullable=False),
        sa.Column("parser_config_id", sa.String(length=128), nullable=False),
        sa.Column("allow_remote_processing", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("stage", sa.String(length=64), server_default="accepted", nullable=False),
        sa.Column("attempt_id", sa.Uuid(), nullable=False),
        sa.Column("executor_token", sa.Uuid(), nullable=True),
        sa.Column(
            "cancel_requested", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("remote_task_id", sa.String(length=512), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("checkpoint", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False),
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
        sa.CheckConstraint(
            "state IN ('queued', 'executing', 'failed', 'cancelled', 'completed')",
            name="ck_parse_task_state",
        ),
        sa.CheckConstraint("length(trim(parser_backend)) > 0", name="ck_parse_task_backend"),
        sa.CheckConstraint("length(trim(parser_config_id)) > 0", name="ck_parse_task_config"),
        sa.ForeignKeyConstraint(["document_version_id"], ["document_versions.document_version_id"]),
        sa.PrimaryKeyConstraint("parse_task_id"),
    )
    op.create_table(
        "parse_snapshots",
        sa.Column("parse_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("parser_backend", sa.String(length=128), nullable=False),
        sa.Column("parser_config_id", sa.String(length=128), nullable=False),
        sa.Column("adapter_version", sa.String(length=128), nullable=False),
        sa.Column(
            "remote_version", sa.String(length=256), server_default="unknown", nullable=False
        ),
        sa.Column("actual_resource_config", sa.JSON(), nullable=False),
        sa.Column("quality_state", sa.String(length=32), nullable=False),
        sa.Column("quality_metrics", sa.JSON(), nullable=False),
        sa.Column("raw_artifact_key", sa.String(length=256), nullable=False),
        sa.Column("raw_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("normalized_artifact_key", sa.String(length=256), nullable=False),
        sa.Column("normalized_artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "quality_state IN ('usable', 'usable_with_warnings', 'rejected')",
            name="ck_snapshot_quality",
        ),
        sa.CheckConstraint("input_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_input_sha256"),
        sa.CheckConstraint("raw_artifact_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_raw_sha256"),
        sa.CheckConstraint(
            "normalized_artifact_sha256 ~ '^[0-9a-f]{64}$'", name="ck_snapshot_normalized_sha256"
        ),
        sa.ForeignKeyConstraint(["document_version_id"], ["document_versions.document_version_id"]),
        sa.PrimaryKeyConstraint("parse_snapshot_id"),
        sa.UniqueConstraint(
            "document_version_id",
            "parser_backend",
            "parser_config_id",
            "adapter_version",
            "normalized_artifact_sha256",
            name="uq_snapshot_content_identity",
        ),
    )
    op.create_table(
        "parse_sections",
        sa.Column("section_id", sa.Uuid(), nullable=False),
        sa.Column("parse_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=4096), nullable=True),
        sa.Column("level", sa.Integer(), nullable=True),
        sa.Column("reading_order", sa.Integer(), nullable=False),
        sa.Column("locations", sa.JSON(), nullable=False),
        sa.CheckConstraint("reading_order >= 0", name="ck_section_reading_order"),
        sa.ForeignKeyConstraint(["parse_snapshot_id"], ["parse_snapshots.parse_snapshot_id"]),
        sa.PrimaryKeyConstraint("section_id"),
        sa.UniqueConstraint("parse_snapshot_id", "reading_order", name="uq_section_reading_order"),
    )
    op.create_table(
        "parse_blocks",
        sa.Column("block_id", sa.Uuid(), nullable=False),
        sa.Column("parse_snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("block_type", sa.String(length=128), nullable=False),
        sa.Column("reading_order", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("locations", sa.JSON(), nullable=False),
        sa.Column("raw_provenance", sa.JSON(), nullable=False),
        sa.Column("quality_flags", sa.JSON(), nullable=False),
        sa.CheckConstraint("reading_order >= 0", name="ck_block_reading_order"),
        sa.CheckConstraint("length(trim(block_type)) > 0", name="ck_block_type"),
        sa.ForeignKeyConstraint(["parse_snapshot_id"], ["parse_snapshots.parse_snapshot_id"]),
        sa.PrimaryKeyConstraint("block_id"),
        sa.UniqueConstraint("parse_snapshot_id", "reading_order", name="uq_block_reading_order"),
    )
    op.execute("""
        CREATE FUNCTION reject_immutable_parse_snapshot_change() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'immutable_parse_snapshot' USING ERRCODE = '23514';
        END;
        $$
    """)
    for table_name in ("parse_snapshots", "parse_sections", "parse_blocks"):
        op.execute(f"""
            CREATE TRIGGER reject_parse_snapshot_change BEFORE UPDATE OR DELETE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION reject_immutable_parse_snapshot_change()
        """)


def downgrade() -> None:
    """移除本迁移结构，仅供管理员显式数据删除操作使用。"""
    for table_name in ("parse_snapshots", "parse_sections", "parse_blocks"):
        op.execute(f"DROP TRIGGER IF EXISTS reject_parse_snapshot_change ON {table_name}")
    op.execute("DROP FUNCTION IF EXISTS reject_immutable_parse_snapshot_change()")
    op.drop_table("parse_blocks")
    op.drop_table("parse_sections")
    op.drop_table("parse_snapshots")
    op.drop_table("parse_tasks")
