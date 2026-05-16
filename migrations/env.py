"""模块：同步迁移入口

独立于异步服务执行 PostgreSQL 迁移和 ORM 差异检查。

核心设计：
  - 连接从环境读取，测试可注入专用数据库连接。


工程要点：
  - 不记录 URL、不自动创建业务数据库，应用连接与迁移连接分开。
"""

from alembic import context
from sqlalchemy import create_engine, pool

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.database.models import Base


def run_with_connection(connection) -> None:
    """执行同步迁移

    在专用连接中运行版本化结构变更。

    Args:
        connection: 管理者选定的同步连接，测试注入隔离验收库连接。
    Returns:
        None；不自行创建数据库，不把 ORM 自动建表作为迁移。
    """
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(
        url=load_settings().database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    supplied_connection = context.config.attributes.get("connection")
    if supplied_connection is not None:
        run_with_connection(supplied_connection)
    else:
        engine = create_engine(
            load_settings().database_url, poolclass=pool.NullPool, hide_parameters=True
        )
        with engine.connect() as connection:
            run_with_connection(connection)
        engine.dispose()
