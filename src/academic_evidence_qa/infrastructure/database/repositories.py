"""模块名称：文档查询与索引任务短事务

提供实际需要的库锁、哈希串行锁、版本和成员查询。

核心设计：
  - 库行锁防止陈旧修订覆盖；
  - 事务级哈希锁串行化跨库相同内容登记。

工程要点：
  - 索引统一 Library→Job→generation；文档统一 Library→哈希；
  - populate_existing 防止会话缓存沿用旧修订。
"""

from datetime import timedelta
import math
from uuid import UUID, uuid4

from sqlalchemy import select, text, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from academic_evidence_qa.infrastructure.database.models import (
    DocumentVersion,
    Library,
    LibraryPaper,
    ParseSnapshotRecord,
    ParseTask,
    BuildIntent,
    IndexGeneration,
)

from academic_evidence_qa.contracts.jobs import IndexJobLease, JobLeaseError


async def claim_index_job(database, owner: str, lease_seconds: float, *, library_id=None):
    """名称：领取索引任务。用途：短事务跳过锁定库，防止同库并行。

    Args:
        database, owner, lease_seconds: 数据库、非空执行器身份与正租期。
        library_id: 可选限定库，验证入口不会消费其他库。
    Returns:
        新租约或 None；过期接管保留旧失败代并生成新 attempt。
    """
    if (
        not owner.strip()
        or len(owner) > 128
        or not math.isfinite(lease_seconds)
        or lease_seconds <= 0
    ):
        raise ValueError("invalid_job_lease_config")
    async with database.transaction() as session:
        eligible = (
            select(BuildIntent.build_intent_id)
            .where(
                BuildIntent.library_id == Library.library_id,
                or_(
                    BuildIntent.state == "queued",
                    (BuildIntent.state == "running")
                    & (BuildIntent.lease_expires_at <= func.clock_timestamp()),
                ),
            )
            .exists()
        )
        query = (
            select(Library)
            .where(
                eligible,
                ~select(BuildIntent.build_intent_id)
                .where(
                    BuildIntent.library_id == Library.library_id,
                    BuildIntent.state == "running",
                    BuildIntent.lease_expires_at > func.clock_timestamp(),
                )
                .exists(),
            )
            .order_by(Library.library_id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        # 显式 C1 构建不是可接管租约，跳过其库而非让它饿死其他库。
        query = query.where(
            or_(
                Library.building_generation_id.is_(None),
                select(BuildIntent.build_intent_id)
                .where(
                    BuildIntent.library_id == Library.library_id,
                    BuildIntent.generation_id == Library.building_generation_id,
                    BuildIntent.state == "running",
                    BuildIntent.lease_expires_at <= func.clock_timestamp(),
                )
                .exists(),
            )
        )
        if library_id is not None:
            query = query.where(Library.library_id == library_id)
        # 只在短事务处理候选；被有效租约占用的库不阻塞其他 Library。
        for library in (await session.scalars(query)).all():
            now = await session.scalar(select(func.clock_timestamp()))
            jobs = (
                await session.scalars(
                    select(BuildIntent)
                    .where(BuildIntent.library_id == library.library_id)
                    .order_by(BuildIntent.membership_revision)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            if any(job.state == "running" and job.lease_expires_at > now for job in jobs):
                continue
            for job in jobs:
                if job.state not in ("queued", "running"):
                    continue
                # C1 的显式构建不是可接管 Job，不能擅自释放其指针。
                if (
                    library.building_generation_id is not None
                    and library.building_generation_id != job.generation_id
                ):
                    continue
                if job.generation_id is not None:
                    generation = await session.get(
                        IndexGeneration, job.generation_id, with_for_update=True
                    )
                    if generation.state == "building":
                        generation.state, generation.error_code = "failed", "lease_expired"
                    if library.building_generation_id == job.generation_id:
                        library.building_generation_id = None
                if job.cancel_requested or job.membership_revision != library.membership_revision:
                    job.state = "cancelled" if job.cancel_requested else "failed"
                    job.error_code = (
                        "job_cancelled" if job.cancel_requested else "membership_superseded"
                    )
                    job.stage = "finished"
                    continue
                job.state, job.stage = "running", "claimed"
                job.lease_owner = owner
                job.lease_version += 1
                job.lease_expires_at = now + timedelta(seconds=lease_seconds)
                job.attempt_id, job.generation_id, job.error_code = uuid4(), None, None
                return IndexJobLease(
                    job.build_intent_id,
                    library.library_id,
                    job.membership_revision,
                    owner,
                    job.lease_version,
                    job.attempt_id,
                )
    return None


async def check_index_lease(session, lease: IndexJobLease, *, allow_cancel=False):
    """名称：事务内 fencing 核验。用途：调用者先锁 Library，再锁 Job。

    Args:
        session, lease: 显式短事务与不可变领取身份。
        allow_cancel: 仅失败收尾允许读取已持久化取消。
    Returns:
        有效 running Job；旧 owner/version/attempt 或过期均拒绝。
    """
    job = await session.get(BuildIntent, lease.job_id, with_for_update=True, populate_existing=True)
    now = await session.scalar(select(func.clock_timestamp()))
    if (
        job is None
        or job.library_id != lease.library_id
        or job.membership_revision != lease.membership_revision
        or job.state != "running"
        or job.lease_owner != lease.owner
        or job.lease_version != lease.version
        or job.attempt_id != lease.attempt_id
        or job.lease_expires_at is None
        or job.lease_expires_at <= now
    ):
        raise JobLeaseError("job_lease_lost")
    if job.cancel_requested and not allow_cancel:
        raise JobLeaseError("job_cancelled")
    return job


async def heartbeat_index_job(database, lease, lease_seconds, *, stage=None):
    """名称：续约与阶段检查点。用途：短事务拒绝旧执行器及取消。

    Args:
        database, lease, lease_seconds: 当前领取与正租期。
        stage: 可选稳定阶段名，不保存输入或第三方原文。
    Returns:
        None；数据库失败不可视作成功续约。
    """
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("invalid_job_lease_config")
    async with database.transaction() as session:
        await lock_library(session, lease.library_id)
        job = await check_index_lease(session, lease)
        now = await session.scalar(select(func.clock_timestamp()))
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        if stage is not None:
            job.stage = stage


async def request_index_cancel(database, job_id):
    """名称：持久化取消。用途：按 Library→Job 与发布互斥。

    Args:
        database, job_id: 正式数据库与目标任务。
    Returns:
        cancel_requested/cancelled/already_completed；不回滚已发布指针。
    """
    async with database.transaction() as session:
        library_id = await session.scalar(
            select(BuildIntent.library_id).where(BuildIntent.build_intent_id == job_id)
        )
        if library_id is None:
            raise JobLeaseError("job_missing")
        await lock_library(session, library_id)
        job = await session.get(BuildIntent, job_id, with_for_update=True)
        if job.state in ("succeeded", "failed", "cancelled"):
            return "already_completed"
        job.cancel_requested = True
        if job.state == "queued":
            job.state, job.stage, job.error_code = "cancelled", "finished", "job_cancelled"
            return "cancelled"
        return "cancel_requested"


async def finish_index_failure(database, lease, error_code):
    """名称：失败或取消收尾。用途：只改仍有效的本租约，不释放新 attempt。

    Args:
        database, lease, error_code: 当前执行与脱敏稳定码。
    Returns:
        终态；过期任务保留 running 供接管，禁止旧执行器补写。
    """
    async with database.transaction() as session:
        library = await lock_library(session, lease.library_id)
        job = await check_index_lease(session, lease, allow_cancel=True)
        if job.generation_id:
            generation = await session.get(IndexGeneration, job.generation_id, with_for_update=True)
            if generation.state == "building":
                generation.state = "failed"
                generation.error_code = "job_cancelled" if job.cancel_requested else error_code
            if library.building_generation_id == job.generation_id:
                library.building_generation_id = None
        job.state = "cancelled" if job.cancel_requested else "failed"
        job.stage = "finished"
        job.error_code = "job_cancelled" if job.cancel_requested else error_code
        return job.state


async def read_index_job_result(database, lease):
    """名称：回读已提交结果。用途：响应丢失后不误报失败、不重复发布。

    Args:
        database, lease: 出现异常的执行上下文；要求完整身份匹配。
    Returns:
        已原子成功的摘要或 None；数据库不可用时不伪造成功。
    """
    async with database.sessions() as session:
        job = await session.get(BuildIntent, lease.job_id)
        if (
            job is None
            or job.state != "succeeded"
            or job.library_id != lease.library_id
            or job.attempt_id != lease.attempt_id
            or job.lease_version != lease.version
            or job.lease_owner != lease.owner
        ):
            return None
        generation = await session.get(IndexGeneration, job.generation_id)
        if (
            generation is None
            or generation.attempt_id != lease.attempt_id
            or generation.state not in ("published", "retired")
        ):
            raise JobLeaseError("job_result_inconsistent")
        return {
            "status": "passed",
            "recovered_committed_result": True,
            "library_id": str(job.library_id),
            "generation_id": str(generation.generation_id),
            "attempt_id": str(job.attempt_id),
            "collection_name": generation.collection_name,
            "membership_revision": job.membership_revision,
            "manifest_sha256": generation.manifest_sha256,
            "paper_count": len(generation.manifest["members"]),
            "chunk_count": len(generation.manifest["chunk_ids"]),
        }


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


async def lock_parse_task(session: AsyncSession, parse_task_id: UUID) -> ParseTask | None:
    """锁定一项最小解析任务，供显式执行、取消和快照提交互斥使用。"""
    return await session.scalar(
        select(ParseTask)
        .where(ParseTask.parse_task_id == parse_task_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


async def find_complete_snapshot(
    session: AsyncSession,
    *,
    document_version_id: UUID,
    parser_backend: str,
    parser_config_id: str,
    adapter_version: str,
    normalized_artifact_sha256: str,
) -> ParseSnapshotRecord | None:
    """按完整内容身份查询现存快照；调用方仍须验证原始/标准化产物哈希。"""
    return await session.scalar(
        select(ParseSnapshotRecord).where(
            ParseSnapshotRecord.document_version_id == document_version_id,
            ParseSnapshotRecord.parser_backend == parser_backend,
            ParseSnapshotRecord.parser_config_id == parser_config_id,
            ParseSnapshotRecord.adapter_version == adapter_version,
            ParseSnapshotRecord.normalized_artifact_sha256 == normalized_artifact_sha256,
        )
    )
