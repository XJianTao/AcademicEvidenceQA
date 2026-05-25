"""模块名称：索引任务执行器。

用途：领取持久化 Job、维持心跳，复用 C1 唯一构建流程。
核心设计：数据库租约是权威；同步模型与网络由 pipeline 在线程执行。
工程要点：心跳失败立即禁止后续阶段；不强杀推理线程，等待有界阶段结束。
"""

import asyncio
import math
from contextlib import suppress

from academic_evidence_qa.contracts.jobs import JobLeaseError, index_error_code as stable_error_code
from academic_evidence_qa.infrastructure.database.repositories import (
    claim_index_job,
    heartbeat_index_job,
    finish_index_failure,
    read_index_job_result,
)
from academic_evidence_qa.ingestion.pipeline import rebuild_library


class IndexJobRunner:
    """名称：单 Worker Runner。用途：顺序执行，避免同进程模型 owner 并发。

    Args:
        database, store, qdrant: 正式存储组件。
        owner, lease_seconds, heartbeat_seconds: 本执行器身份与正心跳间隔。
        pipeline_options: C1 模型路径、读取预算及可选测试模型。
    Returns:
        可重复调用 run_once 的生命周期；不创建 HTTP 服务。
    """

    def __init__(
        self,
        database,
        store,
        qdrant,
        *,
        owner,
        lease_seconds=60,
        heartbeat_seconds=10,
        **pipeline_options,
    ):
        if (
            not owner.strip()
            or not math.isfinite(lease_seconds)
            or not math.isfinite(heartbeat_seconds)
            or not 0 < heartbeat_seconds < lease_seconds / 2
        ):
            raise ValueError("invalid_job_lease_config")
        self.database, self.store, self.qdrant = database, store, qdrant
        self.owner, self.lease_seconds, self.heartbeat_seconds = (
            owner,
            lease_seconds,
            heartbeat_seconds,
        )
        self.pipeline_options = pipeline_options
        self._running = False

    async def run_once(self, *, library_id=None):
        """名称：执行一个任务。用途：无任务返回 None，失败返回脱敏结果。

        Args:
            library_id: 可选目标库；正常 Worker 消费全部可领取意图。
        Returns:
            含 Job/lease/attempt 的结果；进程取消不记录假失败，过期后接管。
        """
        if self._running:
            raise JobLeaseError("worker_already_running")
        self._running = True
        try:
            lease = await claim_index_job(
                self.database, self.owner, self.lease_seconds, library_id=library_id
            )
            if lease is None:
                return None
            return await self.execute(lease)
        finally:
            self._running = False

    async def execute(self, lease):
        """名称：执行已领取租约。用途：接管验证与正常领取共用同一流程。

        Args:
            lease: 本 owner 的当前领取身份，实际有效性仍由数据库核验。
        Returns:
            已发布或明确失败的摘要，不复用过期 collection。
        """
        failure = None

        async def heartbeat():
            nonlocal failure
            while True:
                await asyncio.sleep(self.heartbeat_seconds)
                try:
                    await heartbeat_index_job(self.database, lease, self.lease_seconds)
                except Exception as error:
                    failure = error
                    return

        async def stage_check(stage):
            if failure is not None:
                raise failure
            await heartbeat_index_job(self.database, lease, self.lease_seconds, stage=stage)

        def execution_check():
            # 线程只读停止原因；数据库继续是最终写入和发布权威。
            if failure is not None:
                raise failure

        def on_interrupt():
            nonlocal failure
            failure = JobLeaseError("worker_interrupted")
            pulse.cancel()

        pulse = asyncio.create_task(heartbeat())
        identity = {
            "job_id": str(lease.job_id),
            "lease_owner": lease.owner,
            "lease_version": lease.version,
            "attempt_id": str(lease.attempt_id),
        }
        try:
            result = await rebuild_library(
                self.database,
                lease.library_id,
                self.store,
                self.qdrant,
                lease=lease,
                stage_check=stage_check,
                execution_check=execution_check,
                on_interrupt=on_interrupt,
                **self.pipeline_options,
            )
            return {**result, **identity}
        except asyncio.CancelledError:
            # 模拟进程退出只停止心跳，不伪造终态；下一领取生成新 attempt。
            raise
        except Exception as error:
            code = stable_error_code(error)
            # 发布提交可能成功而本地响应丢失；数据库回读优先于失败收尾。
            with suppress(Exception):
                committed = await read_index_job_result(self.database, lease)
                if committed:
                    return {**committed, **identity}
            try:
                state = await finish_index_failure(self.database, lease, code)
            except Exception:
                state = "lease_lost"
            return {**identity, "status": state, "error_code": code}
        finally:
            pulse.cancel()
            with suppress(asyncio.CancelledError):
                await pulse
