"""模块名称：M3 文件名展示元数据迁移。

用途：为固定文档版本增量登记可核验的原文件名。
核心设计：独立表兼容全部历史记录，不修改原件、版本或解析快照。
工程要点：历史缺名继续为未知；回退会丢失已确认名称，需人工评审。
"""

from alembic import op
import sqlalchemy as sa

revision = "0007_m3_names"
down_revision = "0006_m2a2"
branch_labels = None
depends_on = None


def upgrade():
    """名称：文件名增量迁移。用途：新增可空名称集合，不回填猜测值。"""
    op.create_table(
        "document_metadata",
        sa.Column(
            "document_version_id",
            sa.Uuid(),
            sa.ForeignKey("document_versions.document_version_id"),
            primary_key=True,
        ),
        sa.Column("original_filenames", sa.JSON(), nullable=True),
    )


def downgrade():
    """名称：禁止自动丢弃。用途：已确认的名称需人工决定保留策略。"""
    raise RuntimeError("m3_names_downgrade_requires_manual_review")
