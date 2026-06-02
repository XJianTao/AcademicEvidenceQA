"""模块名称：M2a-2 持久证据与答案迁移。

用途：增量建立取证记录、证据 registry 和生成幂等屏障。
核心设计：原来源不复制到新业务版本，不修改历史解析或索引记录。
工程要点：外键保留原来源；回退涉及已保存答案，必须人工评审。
"""

from alembic import op
import sqlalchemy as sa

revision = "0006_m2a2"
down_revision = "0005_m1c3"
branch_labels = None
depends_on = None


def upgrade():
    """名称：证据增量迁移。用途：只建立本阶段使用的三张表。"""
    op.create_table(
        "retrieval_runs",
        sa.Column("retrieval_run_id", sa.Uuid(), primary_key=True),
        sa.Column("library_id", sa.Uuid(), sa.ForeignKey("libraries.library_id"), nullable=False),
        sa.Column(
            "generation_id",
            sa.Uuid(),
            sa.ForeignKey("index_generations.generation_id"),
            nullable=False,
        ),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("pack", sa.JSON(), nullable=False),
        sa.Column("pack_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "evidence_registry",
        sa.Column("evidence_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "retrieval_run_id",
            sa.Uuid(),
            sa.ForeignKey("retrieval_runs.retrieval_run_id"),
            nullable=False,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_sha256", sa.String(64), nullable=False),
    )
    op.create_table(
        "answer_artifacts",
        sa.Column("answer_id", sa.Uuid(), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_sha256", sa.String(64), nullable=False),
        sa.Column("library_id", sa.Uuid(), sa.ForeignKey("libraries.library_id"), nullable=False),
        sa.Column("retrieval_run_id", sa.Uuid(), sa.ForeignKey("retrieval_runs.retrieval_run_id")),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("response", sa.JSON()),
        sa.Column("draft", sa.Text()),
        sa.Column("error_code", sa.String(128)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_answer_idempotency"),
        sa.CheckConstraint(
            "state IN ('pending', 'completed', 'failed', 'unknown')", name="ck_answer_state"
        ),
    )


def downgrade():
    """名称：禁止自动丢弃。用途：保存的答案和证据需人工决定保留策略。"""
    raise RuntimeError("m2a2_downgrade_requires_manual_review")
