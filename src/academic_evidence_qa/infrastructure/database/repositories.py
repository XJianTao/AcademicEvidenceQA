"""模块：文档数据查询与锁定

提供实际需要的库锁、哈希串行锁、版本和成员查询。

核心设计：
  - 库行锁防止陈旧修订覆盖；
  - 事务级哈希锁串行化跨库相同内容登记。

工程要点：
  - 统一先锁 Library 再锁哈希；
  - populate_existing 防止会话缓存沿用旧修订。
"""

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from academic_evidence_qa.infrastructure.database.models import (
    DocumentVersion,
    Library,
    LibraryPaper,
)


async def lock_library(session: AsyncSession, library_id: UUID) -> Library | None:
    """锁定库

    读取锁取得后的最新修订。

    Args:
        session: 调用方持有的显式事务。
        library_id: 本次唯一目标库；多个库操作须另行规定统一锁顺序。
    Returns:
        锁定的 Library，或不存在时 None；锁保留到提交/回滚。
    """
    return await session.scalar(
        select(Library)
        .where(Library.library_id == library_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


async def lock_file_hash(session: AsyncSession, file_sha256: str) -> None:
    """锁定内容登记

    新哈希没有行可锁时使用事务 advisory lock。

    Args:
        session: 显式事务，同库操作先持有 Library 锁。
        file_sha256: 已计算的 64 位十六进制 SHA-256。
    Returns:
        None；使用前 64 bit 有符号锁键，碰撞仅导致额外串行，不改变哈希校验。
    """
    lock_key = int.from_bytes(bytes.fromhex(file_sha256[:16]), "big", signed=True)
    await session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})


async def find_version_by_hash(
    session: AsyncSession,
    file_sha256: str,
) -> DocumentVersion | None:
    """按内容查版本

    查找全局唯一的已登记版本。

    Args:
        session: 已取得该哈希事务锁的会话，避免首次登记竞争。
        file_sha256: 完整 SHA-256，而不是 advisory lock 的截短键。
    Returns:
        已有不可变版本；内容尚未登记时返回 None。
    """
    return await session.scalar(
        select(DocumentVersion).where(DocumentVersion.file_sha256 == file_sha256)
    )


async def find_member(
    session: AsyncSession,
    library_id: UUID,
    paper_id: UUID,
) -> LibraryPaper | None:
    """查询库成员

    读取每库独立选版与状态。

    Args:
        session: 已持有目标库行锁的显式事务，避免读写竞态。
        library_id: 目标库身份。
        paper_id: 论文身份，与 library_id 共同定位唯一成员。
    Returns:
        刷新会话缓存后的成员，包括 removed；不存在时返回 None。
    """
    return await session.scalar(
        select(LibraryPaper)
        .where(
            LibraryPaper.library_id == library_id,
            LibraryPaper.paper_id == paper_id,
        )
        .execution_options(populate_existing=True)
    )
