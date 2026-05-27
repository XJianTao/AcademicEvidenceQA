"""模块名称：M1-C3 管理契约迁移。

用途：持久幂等、解析租约/关联及同修订显式重建。
核心设计：默认 sequence=0 保留每修订的原始成员意图；显式请求分配新序号。
工程要点：只增量迁移，不修改历史任务、诊断或产物。
"""

from alembic import op
import sqlalchemy as sa

revision = "0005_m1c3"
down_revision = "0004_m1c2"
branch_labels = None
depends_on = None


def upgrade():
    """名称：增量管理迁移。用途：保留原任务并允许独立新请求。"""
    op.drop_constraint("uq_intent_library_revision", "build_intents", type_="unique")
    op.drop_constraint("ck_intent_revision", "build_intents", type_="check")
    op.add_column(
        "build_intents",
        sa.Column("request_sequence", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "build_intents",
        sa.Column("retry_of_job_id", sa.Uuid(), sa.ForeignKey("build_intents.build_intent_id")),
    )
    op.create_unique_constraint(
        "uq_intent_library_revision",
        "build_intents",
        ["library_id", "membership_revision", "request_sequence"],
    )
    op.create_check_constraint("ck_intent_revision", "build_intents", "membership_revision >= 0")
    for column in (
        sa.Column("index_job_id", sa.Uuid(), sa.ForeignKey("build_intents.build_intent_id")),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("library_id", sa.Uuid(), sa.ForeignKey("libraries.library_id")),
        sa.Column("retry_of_job_id", sa.Uuid(), sa.ForeignKey("parse_tasks.parse_task_id")),
        sa.Column(
            "parse_snapshot_id", sa.Uuid(), sa.ForeignKey("parse_snapshots.parse_snapshot_id")
        ),
    ):
        op.add_column("parse_tasks", column)
    op.create_table(
        "management_requests",
        sa.Column("idempotency_key", sa.String(128), primary_key=True),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=False),
        sa.Column("response_body", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade():
    """名称：拒绝无损回退。用途：多个同修订请求不能折叠为历史唯一键。"""
    raise RuntimeError("m1c3_downgrade_requires_manual_review")
