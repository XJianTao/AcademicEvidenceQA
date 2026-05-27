"""模块名称：正式解析编排与 Worker 执行。

用途：M1-B CLI 与 C3 Worker 共用唯一远端到快照链路。
核心设计：网络/受限子进程位于事务外，检查点与快照在短事务内核验身份。
工程要点：不打印第三方响应或签名 URL，取消传到同一受限进程树。
"""

import asyncio
import math
import threading
from contextlib import asynccontextmanager, suppress
from time import monotonic
from datetime import datetime, timedelta
from uuid import uuid4
from sqlalchemy import select, func, or_

from academic_evidence_qa.bootstrap import mineru_config, parser_identity
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.contracts.jobs import index_error_code
from academic_evidence_qa.documents.service import DocumentService
from academic_evidence_qa.infrastructure.database.models import (
    ParseTask,
    DocumentVersion,
    StoredFile,
    ParseSnapshotRecord,
)
from academic_evidence_qa.documents.source_reader import verify_snapshot
from academic_evidence_qa.infrastructure.local_file_store import StoredObject
from academic_evidence_qa.infrastructure.mineru_client import MineruClient

from academic_evidence_qa.documents.parser import (
    ADAPTER_VERSION,
    ParserError,
    normalize_mineru_result,
)
from academic_evidence_qa.infrastructure.process_guard import WindowsProcessGuard
from academic_evidence_qa.infrastructure.result_archive import extract_result_archive


class CancellableParseGuard(WindowsProcessGuard):
    """名称：可取消资源防护。用途：正式 Windows 防护与执行器停止事件绑定。"""

    def __init__(self, cancellation):
        super().__init__()
        self.cancellation = cancellation

    def run(self, request, **kwargs):
        return super().run(request, cancellation=self.cancellation, **kwargs)


async def run_local_stage(cancellation, function, *args, **kwargs):
    """名称：受限本地阶段。用途：取消时等待线程回收进程树，避免后台遗留。

    Args:
        cancellation: 传给 Windows 资源防护的线程事件。
        function, args, kwargs: 有界同步操作及参数。
    Returns:
        操作结果；取消先回收，再向调用方传播。
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancellation.set()
        with suppress(Exception):
            await task
        raise


@asynccontextmanager
async def record_stage(summary, stage):
    """记录脱敏耗时；失败保留阶段，不保存输入/第三方消息。"""
    started = monotonic()
    try:
        yield
    finally:
        summary.setdefault("stages", {})[stage] = {"seconds": round(monotonic() - started, 3)}


async def execute_parse_flow(
    *,
    database,
    service,
    store,
    stored,
    settings,
    client,
    attempt,
    job_id,
    executor_token,
    attempt_dir,
    checkpoint,
    local_cancel,
    remote_cancel,
    summary,
    config_identity,
    context,
    record_stage=record_stage,
    guard_factory=CancellableParseGuard,
    extract=extract_result_archive,
    normalize=normalize_mineru_result,
    completed=None,
):
    """名称：唯一解析编排。用途：从可信原件到原子快照，CLI 与 Worker 共用。

    Args:
        database/service/store: 正式事务与存储组件。
        attempt/job_id/executor_token: 权威身份，checkpoint 验证租约与预算。
        context: 当前阶段；其余参数为预算、取消事件及可注入边界。
        completed: 恢复时已知远端完成结果，禁止重新提交/上传。
    Returns:
        快照身份、标准化文档与两份已核验不可变产物。
    """
    source = store.data_root / stored.object_key
    if completed is None:
        context["stage"] = "submit"
        async with record_stage(summary, context["stage"]):
            await checkpoint()
            submission = await client.submit(
                source,
                attempt=attempt,
                checkpoint_remote_task_id=checkpoint,
                cancellation=remote_cancel,
            )
        for stage in ("upload", "poll", "download"):
            context["stage"] = stage
            async with record_stage(summary, context["stage"]):
                await checkpoint()
                if stage == "upload":
                    await client.upload(
                        source, submission, attempt=attempt, cancellation=remote_cancel
                    )
                elif stage == "poll":
                    completed = await client.wait_for_completion(
                        submission.remote_task_id, attempt=attempt, cancellation=remote_cancel
                    )
                    summary.update(
                        poll_count=completed.poll_count, poll_retries=completed.retry_count
                    )
                else:
                    await client.download(
                        completed,
                        attempt_dir / "result.zip",
                        attempt=attempt,
                        cancellation=remote_cancel,
                    )
    else:
        context["stage"] = "download"
        await checkpoint()
        await client.download(
            completed, attempt_dir / "result.zip", attempt=attempt, cancellation=remote_cancel
        )
    guard = guard_factory(local_cancel)
    context["stage"] = "extract"
    async with record_stage(summary, context["stage"]):
        await checkpoint()
        extracted = await run_local_stage(
            local_cancel,
            extract,
            attempt_dir / "result.zip",
            attempt_dir / "extracted",
            max_files=settings.result_max_files,
            max_extracted_bytes=settings.result_max_extracted_bytes,
            max_json_bytes=settings.result_max_json_bytes,
            timeout_seconds=settings.result_extract_timeout_seconds,
            memory_mib=settings.result_extract_memory_mib,
            guard=guard,
        )
        summary["archive"] = vars(extracted)
    content_lists = list((attempt_dir / "extracted").rglob("*_content_list.json"))
    if len(content_lists) != 1:
        raise ParserError("mineru_content_list_ambiguous_or_missing")
    context["stage"] = "normalize"
    async with record_stage(summary, context["stage"]):
        await checkpoint()
        normalized_path = attempt_dir / "normalized.json"
        document = await run_local_stage(
            local_cancel,
            normalize,
            content_lists[0],
            normalized_path,
            max_json_bytes=settings.result_max_json_bytes,
            timeout_seconds=settings.normalization_timeout_seconds,
            memory_mib=settings.normalization_memory_mib,
            guard=guard,
        )
        summary.update(
            quality_state=document.quality_state,
            quality_metrics=document.quality_metrics,
            warning_count=document.quality_metrics.get("warning_count", 0),
        )
        if document.quality_state == "rejected" or not document.blocks:
            raise ParserError("parser_quality_rejected")
    context["stage"] = "artifacts"
    async with record_stage(summary, context["stage"]):
        await checkpoint()
        with (attempt_dir / "result.zip").open("rb") as stream:
            raw = await run_local_stage(
                local_cancel,
                store.write_artifact,
                "mineru_raw",
                stream,
                max_bytes=settings.result_max_download_bytes,
            )
        with normalized_path.open("rb") as stream:
            normalized = await run_local_stage(
                local_cancel,
                store.write_artifact,
                "normalized",
                stream,
                max_bytes=settings.result_max_json_bytes,
            )
        summary.update(raw_artifact=vars(raw), normalized_artifact=vars(normalized))
    context["stage"] = "snapshot_commit"
    async with record_stage(summary, context["stage"]):
        await checkpoint()
        async with database.transaction() as session:
            snapshot_id = await service.save_parse_snapshot(
                session,
                parse_task_id=job_id,
                executor_token=executor_token,
                document=document,
                adapter_version=ADAPTER_VERSION,
                remote_version="unknown",
                actual_resource_config=config_identity,
                raw_artifact=raw,
                normalized_artifact=normalized,
            )
        summary["parse_snapshot_id"] = str(snapshot_id)
    return snapshot_id, document, raw, normalized


class ParseJobRunner:
    """名称：持久解析执行器。用途：领取、心跳和安全恢复，共用正式解析编排。

    Args:
        database, store, settings: 本进程组件，不在请求内执行。
        lease_seconds, heartbeat_seconds: 数据库时钟租期；token 阻止旧执行器写入。
        client_factory: 可控测试边界，生产使用正式 MinerU 客户端。
    Returns:
        顺序执行的 Runner；不对未知 submit/upload 进行重传。
    """

    def __init__(
        self,
        database,
        store,
        settings,
        *,
        lease_seconds=60,
        heartbeat_seconds=10,
        client_factory=MineruClient,
    ):
        if (
            not all(math.isfinite(value) for value in (lease_seconds, heartbeat_seconds))
            or not 0 < heartbeat_seconds < lease_seconds / 2
        ):
            raise ValueError("invalid_parse_lease_config")
        self.database, self.store, self.settings = database, store, settings
        self.service = DocumentService(store)
        self.lease_seconds, self.heartbeat_seconds = lease_seconds, heartbeat_seconds
        self.client_factory = client_factory
        self._run_lock = asyncio.Lock()

    async def run_once(self, *, library_id=None):
        """串行化同一 Runner 的并发调用；不同进程仍由数据库租约互斥。"""
        async with self._run_lock:
            return await self._execute_once(library_id=library_id)

    async def _execute_once(self, *, library_id=None):
        """名称：领取并执行。用途：短事务 SKIP LOCKED，按版本串行外发。

        Args:
            library_id: 可选消费范围；不改变任务固定输入。
        Returns:
            脱敏结果或 None；过期接管保持原 attempt 与原期限。
        """
        token = uuid4()
        async with self.database.transaction() as session:
            query = (
                select(ParseTask)
                .where(
                    or_(
                        ParseTask.state == "queued",
                        (ParseTask.state == "executing")
                        & (ParseTask.lease_expires_at <= func.clock_timestamp()),
                    )
                )
                .order_by(ParseTask.created_at)
                .with_for_update(skip_locked=True)
            )
            if library_id is not None:
                query = query.where(ParseTask.library_id == library_id)
            task = None
            for candidate in (await session.scalars(query)).all():
                # advisory 锁与活跃租约检查防止两个库同一版本同时上传。
                lock_key = int(candidate.document_version_id.hex[:15], 16)
                locked = await session.scalar(select(func.pg_try_advisory_xact_lock(lock_key)))
                if not locked:
                    continue
                other = await session.scalar(
                    select(ParseTask.parse_task_id)
                    .where(
                        ParseTask.document_version_id == candidate.document_version_id,
                        ParseTask.parser_backend == candidate.parser_backend,
                        ParseTask.parser_config_id == candidate.parser_config_id,
                        ParseTask.parse_task_id != candidate.parse_task_id,
                        ParseTask.state == "executing",
                        ParseTask.lease_expires_at > func.clock_timestamp(),
                    )
                    .limit(1)
                )
                if other:
                    continue
                task = candidate
                break
            if task is None:
                return None
            recovering = task.state == "executing"
            now = await session.scalar(select(func.clock_timestamp()))
            remaining_started = monotonic()
            task.executor_token = token
            task.lease_expires_at = now + timedelta(seconds=self.lease_seconds)
            task.state = "executing"
            await session.flush()
            job_id, attempt_id = task.parse_task_id, task.attempt_id
            checkpoint_data, old_stage, remote_id = (
                dict(task.checkpoint),
                task.stage,
                task.remote_task_id,
            )
            version = await session.get(DocumentVersion, task.document_version_id)
            file = await session.get(StoredFile, version.file_id)
            stored = StoredObject(file.file_id, file.file_sha256, file.object_key, file.size_bytes)
            backend, config_id, allowed = (
                task.parser_backend,
                task.parser_config_id,
                task.allow_remote_processing,
            )
        remote_cancel, local_cancel = asyncio.Event(), threading.Event()
        failure = None
        client = None
        attempt = None
        context = {"stage": old_stage}
        summary = {"job_id": str(job_id), "attempt_id": str(attempt_id), "job_type": "parse"}

        async def pulse_once():
            async with self.database.transaction() as session:
                current = await self.service._locked_executing_task(session, job_id, token)
                if current.cancel_requested:
                    raise DocumentError("parse_task_cancelled", 409)
                clock = await session.scalar(select(func.clock_timestamp()))
                current.lease_expires_at = clock + timedelta(seconds=self.lease_seconds)

        async def heartbeat():
            nonlocal failure
            while True:
                await asyncio.sleep(self.heartbeat_seconds)
                try:
                    await pulse_once()
                except Exception as error:
                    failure = error
                    remote_cancel.set()
                    local_cancel.set()
                    return

        async def checkpoint(remote_task_id=None):
            # submit 返回后必须先冻结远端身份，即使此时收到取消；
            # 过期/错误执行器仍由短事务 fencing 拒绝，不能覆盖检查点。
            preserve_remote_identity = remote_task_id is not None
            if failure is not None and not (
                preserve_remote_identity and str(failure) == "parse_task_cancelled"
            ):
                raise failure
            async with self.database.transaction() as session:
                result = await self.service.checkpoint_parse_task(
                    session,
                    parse_task_id=job_id,
                    executor_token=token,
                    stage=context["stage"],
                    remote_task_id=remote_task_id,
                    checkpoint=attempt.checkpoint(),
                )
            # 先提交已知远端身份，再传播取消；在事务内抛出会回滚检查点。
            if result.cancel_requested:
                remote_cancel.set()
                local_cancel.set()
                raise DocumentError("parse_task_cancelled", 409)

        pulse = asyncio.create_task(heartbeat())
        try:
            await pulse_once()
            # 同版本串行领取后再次核验缓存；文件/JSON 核验不持有任务行锁。
            # 领取事务已经提交，活跃租约阻止另一进程同时外发同一输入。
            if not recovering:
                async with self.database.sessions() as session:
                    snapshots = (
                        await session.scalars(
                            select(ParseSnapshotRecord)
                            .where(
                                ParseSnapshotRecord.document_version_id
                                == version.document_version_id,
                                ParseSnapshotRecord.parser_backend == backend,
                                ParseSnapshotRecord.parser_config_id == config_id,
                                ParseSnapshotRecord.quality_state.in_(
                                    ("usable", "usable_with_warnings")
                                ),
                            )
                            .order_by(ParseSnapshotRecord.created_at)
                        )
                    ).all()
                    cached_id = None
                    for snapshot in snapshots:
                        try:
                            await verify_snapshot(
                                session,
                                version,
                                snapshot,
                                self.store,
                                max_json_bytes=self.settings.result_max_json_bytes,
                            )
                        except Exception:
                            continue
                        cached_id = snapshot.parse_snapshot_id
                        break
                if cached_id is not None:
                    async with self.database.transaction() as session:
                        current = await self.service._locked_executing_task(session, job_id, token)
                        if current.cancel_requested:
                            raise DocumentError("parse_task_cancelled", 409)
                        current.state, current.stage = "completed", "finished"
                        current.parse_snapshot_id = cached_id
                    return {
                        **summary,
                        "status": "passed",
                        "cache_hit": True,
                        "parse_snapshot_id": str(cached_id),
                    }
            identity, expected_config = parser_identity(self.settings)
            if backend != "mineru_remote":
                raise DocumentError("docling_unavailable", 503)
            if not allowed:
                raise DocumentError("remote_processing_not_allowed", 422)
            if config_id != expected_config:
                raise DocumentError("parser_config_unavailable", 503)
            await asyncio.to_thread(self.store.verify, stored)
            client = self.client_factory(mineru_config(self.settings, recovering=recovering))
            completed = None
            if recovering:
                if not remote_id or old_stage in ("accepted", "starting", "submit"):
                    raise DocumentError("submission_unknown", 409)
                if old_stage == "upload":
                    raise DocumentError("upload_state_unknown", 409)
                if checkpoint_data.get("attempt_id") != str(attempt_id):
                    raise DocumentError("parse_task_attempt_conflict", 409)
                if (
                    old_stage
                    not in (
                        "poll",
                        "download",
                        "extract",
                        "normalize",
                        "artifacts",
                        "snapshot_commit",
                    )
                    or checkpoint_data.get("remote_task_id") != remote_id
                ):
                    raise DocumentError("attempt_checkpoint_invalid", 409)
                try:
                    expires = datetime.fromisoformat(checkpoint_data["expires_at"])
                    if expires.tzinfo is None:
                        raise ValueError
                except (KeyError, TypeError, ValueError) as error:
                    raise DocumentError("attempt_checkpoint_invalid", 409) from error
                remaining = (expires - now).total_seconds() - (monotonic() - remaining_started)
                attempt = client.resume_known_poll(
                    attempt_id=str(attempt_id),
                    expires_at=expires,
                    remote_task_id=remote_id,
                    remaining_seconds=remaining,
                )
                context["stage"] = "poll"
                await checkpoint()
                completed = await client.wait_for_completion(
                    remote_id, attempt=attempt, cancellation=remote_cancel
                )
            else:
                attempt = client.begin_attempt(attempt_id=str(attempt_id))
            # 同 attempt 每次执行使用 token 子目录，避免恢复覆盖旧诊断或结果。
            directory = self.store.data_root / "parse_attempts" / attempt_id.hex / token.hex
            directory.mkdir(parents=True, exist_ok=False)
            await execute_parse_flow(
                database=self.database,
                service=self.service,
                store=self.store,
                stored=stored,
                settings=self.settings,
                client=client,
                attempt=attempt,
                job_id=job_id,
                executor_token=token,
                attempt_dir=directory,
                checkpoint=checkpoint,
                local_cancel=local_cancel,
                remote_cancel=remote_cancel,
                summary=summary,
                config_identity=identity,
                context=context,
                completed=completed,
            )
            summary["status"] = "passed"
            return summary
        except asyncio.CancelledError:
            remote_cancel.set()
            local_cancel.set()
            raise
        except Exception as error:
            code = index_error_code(error)
            if code == "index_job_failed":
                code = "parse_job_failed"
            cancelled = code in ("parse_task_cancelled", "cancelled")
            if cancelled and attempt is not None and attempt.remote_task_id:
                code = "remote_cancellation_unknown"
            with suppress(Exception):
                async with self.database.transaction() as session:
                    await self.service.fail_parse_task(
                        session, parse_task_id=job_id, executor_token=token, error_code=code
                    )
            return {**summary, "status": "cancelled" if cancelled else "failed", "error_code": code}
        finally:
            pulse.cancel()
            with suppress(asyncio.CancelledError):
                await pulse
            if client is not None:
                await client.aclose()
