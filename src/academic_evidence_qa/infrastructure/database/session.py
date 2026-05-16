"""模块：数据库会话生命周期

构造异步 engine 和短事务上下文；Alembic 单独使用同步 engine。

核心设计：
  - 提交/回滚由调用方的 transaction 管理，service 不隐式提交。

工程要点：
  - 异常必须离开事务上下文；
  - 禁止捕获领域错误后提交部分登记。
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def create_database_event_loop() -> asyncio.AbstractEventLoop:
    """创建数据库兼容事件循环

    显式满足 Windows psycopg 限制。

    Args:
        无；用于 asyncio.run(..., loop_factory=create_database_event_loop)。
    Returns:
        Windows 使用 SelectorEventLoop；其他平台使用默认新循环。
    工程要点：不修改全局 policy；未来 API 入口也需显式配置兼容循环。
    """
    loop = asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


class Database:
    """数据库生命周期

    集中持有连接池和会话工厂。

    Args:
        database_url: 不记录到日志的连接配置。
    Returns:
        可显式 dispose 的数据库实例；无业务实体初始化副作用。
    """

    def __init__(self, database_url: URL):
        self.engine = create_async_engine(
            database_url, pool_pre_ping=True, echo=False, hide_parameters=True
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """短事务

        成功提交，任何异常回滚并关闭会话。

        Args:
            无；业务在 yield 的会话中执行，不跨事务复用锁定对象。
        Returns:
            异步上下文内的 AsyncSession。
        """
        async with self.sessions.begin() as session:
            yield session

    async def close(self) -> None:
        """关闭数据库

        在进程或测试生命周期结束时释放连接池。

        Args:
            无；调用方先完成使用该实例的事务，并等待本方法结束。
        Returns:
            None；不会删除业务数据或关闭其他实例的连接池。
        """
        await self.engine.dispose()
