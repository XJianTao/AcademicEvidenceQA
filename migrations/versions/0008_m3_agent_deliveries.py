"""模块名称：M3 Agent 独立来源登记迁移。

用途：增量保存可信宿主的固定交付来源，不改写原件或一次问答历史。
核心设计：只新增独立表，兼容全部历史版本、快照和费用记录。
工程要点：登记不是 final；回退会丢历史授权，禁止自动删除。
"""

from alembic import op
import sqlalchemy as sa

revision = "0008_m3_deliveries"
down_revision = "0007_m3_names"
branch_labels = None
depends_on = None


def upgrade():
    """名称：增量来源登记。用途：增加独立 delivery 表，不回填或改写旧业务数据。"""
    op.create_table(
        "agent_deliveries",
        sa.Column("delivery_id", sa.Uuid(), primary_key=True),
        sa.Column("library_id", sa.Uuid(), sa.ForeignKey("libraries.library_id"), nullable=False),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("index_generations.generation_id"),
            nullable=False,
        ),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade():
    """禁止静默删除来源历史，保留策略须人工评审。"""
    raise RuntimeError("m3_deliveries_downgrade_requires_manual_review")
