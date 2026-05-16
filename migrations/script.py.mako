"""模块：${message}

版本化数据库结构变更。

核心设计：
  - 明确升级与降级，不依赖运行时 ORM 自动建表。

工程要点：
  - 降级可能删除本迁移的数据，仅由管理者显式执行。
"""

from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    """应用本版本迁移；无返回值。"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """显式撤销本版本结构；不自动执行。"""
    ${downgrades if downgrades else "pass"}
