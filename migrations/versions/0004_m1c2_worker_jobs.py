"""模块名称：M1-C2 持久化索引任务迁移。

用途：原地扩展 BuildIntent，保留已有意图、修订与唯一性。
核心设计：状态与阶段分离；租约版本单调递增，attempt 每次领取更新。
工程要点：不改历史迁移，不删除历史 generation 或 collection。
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_m1c2"
down_revision = "0003_m1c1"
branch_labels = None
depends_on = None


def upgrade():
    """名称：扩展意图。用途：已有记录默认 queued，可执行且不丢失。"""
    for column in (
        sa.Column("state", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("lease_owner", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("lease_version", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("attempt_id", sa.Uuid()),
        sa.Column("generation_id", sa.Uuid()),
        sa.Column("error_code", sa.String(128)),
    ):
        op.add_column("build_intents", column)
    op.create_foreign_key(
        "fk_job_generation",
        "build_intents",
        "index_generations",
        ["generation_id"],
        ["generation_id"],
    )
    op.create_check_constraint(
        "ck_index_job_state",
        "build_intents",
        "state IN ('queued','running','succeeded','failed','cancelled')",
    )
    op.create_check_constraint("ck_index_job_lease_version", "build_intents", "lease_version >= 0")


def downgrade():
    """名称：显式回退。用途：仅供管理员使用，不由 Worker 自动调用。"""
    op.drop_constraint("fk_job_generation", "build_intents", type_="foreignkey")
    op.drop_constraint("ck_index_job_state", "build_intents", type_="check")
    op.drop_constraint("ck_index_job_lease_version", "build_intents", type_="check")
    for name in (
        "error_code",
        "generation_id",
        "attempt_id",
        "lease_version",
        "lease_expires_at",
        "lease_owner",
        "cancel_requested",
        "stage",
        "state",
    ):
        op.drop_column("build_intents", name)
