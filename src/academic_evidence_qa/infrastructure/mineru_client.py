"""模块：MinerU 官方在线协议客户端。

用途概要：以可注入传输层执行单 PDF 的提交、预签名上传、轮询和 ZIP 下载，供 M1-B
显式解析任务使用。

核心设计：总期限覆盖全部远端步骤；查询、上传和下载仅重试可安全重放的瞬时失败；
创建 batch 的响应丢失一律标为 ``submission_unknown``，绝不盲目重提。

工程要点：token、Authorization、预签名 URL 与远端错误正文都不进入异常消息或检查点。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from academic_evidence_qa.infrastructure.mineru_http_transport import (
    HttpxMineruTransport,
    MineruTransport,
    MineruTransportError,
    retryable_http_status,
)


_TRANSPORT_ERROR_CODES = {
    "remote_response_too_large",
    "remote_response_invalid",
    "remote_timeout",
    "remote_transport_failure",
    "remote_upload_timeout",
    "remote_upload_transport_failure",
    "remote_upload_http_error",
    "remote_poll_http_error",
    "remote_download_http_error",
    "remote_download_too_large",
    "remote_download_empty",
    "remote_download_destination_exists",
    "remote_download_publish_failure",
    "remote_download_timeout",
    "remote_download_transport_failure",
}


class MineruClientError(RuntimeError):
    """在线协议的稳定失败，不保存远端返回正文或敏感地址。"""

    def __init__(self, error_code: str, *, attempt: MineruAttempt | None = None):
        self.error_code = error_code
        self.attempt_id = attempt.attempt_id if attempt else None
        self.remote_task_id = attempt.remote_task_id if attempt else None
        self.last_stage = attempt.last_stage if attempt else None
        super().__init__(error_code)


@dataclass(frozen=True)
class MineruClientConfig:
    """名称：MinerU 远端限制。用途概要：固定每个显式任务的超时和重试预算。"""

    base_url: str
    api_token: str = field(repr=False)
    model_version: str = "vlm"
    connect_timeout_seconds: float = 10
    read_timeout_seconds: float = 60
    total_timeout_seconds: float = 1800
    max_retries: int = 2
    poll_interval_seconds: float = 2
    max_download_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.model_version != "vlm":
            raise ValueError("M1-B 仅允许 MinerU vlm")
        if not self.api_token:
            raise ValueError("MINERU_API_TOKEN 不能为空")
        if self.max_retries not in range(3):
            raise ValueError("MinerU 可重试失败次数只能为 0 至 2")
        if (
            min(
                self.connect_timeout_seconds,
                self.read_timeout_seconds,
                self.total_timeout_seconds,
                self.poll_interval_seconds,
            )
            <= 0
            or self.max_download_bytes <= 0
        ):
            raise ValueError("MinerU 超时、轮询间隔和下载上限必须为正数")


@dataclass
class MineruAttempt:
    """名称：在线 attempt 执行上下文。用途概要：为所有远端阶段固定同一总预算。

    Args:
        attempt_id: 与持久化解析任务关联的显式执行标识。
        expires_at: 仅供持久化诊断的 UTC 到期时间；不用于放宽本进程预算。
        monotonic_deadline: 当前进程创建一次的单调时钟期限。

    Returns:
        保存远端身份、最后阶段与提交不确定性；不保存 URL 或凭据的上下文。

    工程要点：由 ``begin_attempt`` 创建并仅在原客户端使用；跨进程时无法证明
    monotonic 连续性，因此本阶段显式拒绝恢复，不通过 UTC 重建完整预算。
    """

    attempt_id: str
    expires_at: datetime
    monotonic_deadline: float
    _owner: object = field(repr=False)
    remote_task_id: str | None = None
    last_stage: str = "created"
    submission_state: str = "not_started"

    def checkpoint(self) -> dict[str, str | None]:
        """输出可持久化的安全上下文；不序列化进程内 owner 或签名地址。"""
        return {
            "attempt_id": self.attempt_id,
            "expires_at": self.expires_at.isoformat(),
            "remote_task_id": self.remote_task_id,
            "last_stage": self.last_stage,
            "submission_state": self.submission_state,
        }


@dataclass(frozen=True)
class MineruSubmission:
    """名称：提交结果。用途概要：只保留可持久化的远端 batch 标识。"""

    remote_task_id: str
    upload_url: str = field(repr=False)


@dataclass(frozen=True)
class MineruCompletedResult:
    """名称：完成结果。用途概要：返回仅用于当前下载步骤的受校验 ZIP 地址和统计。"""

    archive_url: str = field(repr=False)
    poll_count: int
    retry_count: int


@dataclass(frozen=True)
class _RetryResult:
    """名称：可重试调用结果。用途概要：分开保存结果与本次调用实际重试次数。"""

    value: Any
    retry_count: int


class MineruClient:
    """名称：MinerU 任务客户端。用途概要：执行单个已授权解析任务的远端协议。"""

    def __init__(
        self,
        config: MineruClientConfig,
        *,
        transport: MineruTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.config = config
        self.api_base = _api_base(config.base_url)
        self._transport = transport or HttpxMineruTransport()
        self._monotonic = monotonic
        self._sleep = sleep
        self._utc_now = utc_now
        self._closed = False
        self._attempt_owner = object()
        self._started_attempt_ids: set[str] = set()

    def begin_attempt(self, *, attempt_id: str) -> MineruAttempt:
        """名称：开始在线 attempt。用途概要：在提交之前固定唯一整体期限。

        Args:
            attempt_id: 由持久化 ParseTask 提供的执行标识，不能包含敏感数据。
        Returns:
            必须传给所有远端阶段的同一个执行上下文。
        """
        self._require_open()
        if not _valid_identifier(attempt_id):
            raise MineruClientError("attempt_id_invalid")
        if attempt_id in self._started_attempt_ids:
            raise MineruClientError("attempt_already_started")
        self._started_attempt_ids.add(attempt_id)
        return MineruAttempt(
            attempt_id=attempt_id,
            expires_at=self._utc_now() + timedelta(seconds=self.config.total_timeout_seconds),
            monotonic_deadline=self._monotonic() + self.config.total_timeout_seconds,
            _owner=self._attempt_owner,
        )

    def restore_attempt(
        self,
        *,
        attempt_id: str,
        expires_at: datetime,
        remote_task_id: str | None,
        last_stage: str,
    ) -> MineruAttempt:
        """名称：拒绝不安全恢复。用途概要：阻止用持久化身份重获完整预算。

        Args:
            attempt_id: 持久化 attempt 标识。
            expires_at: 持久化 UTC 期限；不能证明单调时钟或系统时钟未改变。
            remote_task_id: 已知远端 batch 标识，保留供人工诊断。
            last_stage: 原 attempt 的最后阶段，仅允许协议枚举值。
        Returns:
            不返回上下文；当前 M1-B 不安全重建跨进程 monotonic deadline。
        Raises:
            MineruClientError: ``attempt_resume_unsupported``，不发送任何请求。
        """
        if not _valid_identifier(attempt_id) or (
            remote_task_id is not None and not _valid_identifier(remote_task_id)
        ):
            raise MineruClientError("attempt_checkpoint_invalid")
        if expires_at.tzinfo is None or last_stage not in {
            "created",
            "submit",
            "upload",
            "poll",
            "download",
        }:
            raise MineruClientError("attempt_checkpoint_invalid")
        rejected = MineruAttempt(
            attempt_id,
            expires_at,
            float("-inf"),
            self._attempt_owner,
            remote_task_id=remote_task_id,
            last_stage=last_stage,
        )
        raise MineruClientError("attempt_resume_unsupported", attempt=rejected)

    async def submit(
        self,
        source: Path,
        *,
        attempt: MineruAttempt,
        checkpoint_remote_task_id: Callable[[str], Awaitable[None]],
        cancellation: asyncio.Event | None = None,
    ) -> MineruSubmission:
        """名称：提交单个 PDF。用途概要：创建远端 batch 并在上传前持久化身份。

        提交端点创建远端副作用。网络或响应不确定时不能安全重放，故统一返回
        ``submission_unknown``，让操作者按明确 job_id 人工处置。

        Args:
            source: 经过有界接收与预检的可信原件路径。
            attempt: 在本客户端创建一次且贯穿四个远端阶段的上下文。
            checkpoint_remote_task_id: 上传前必须完成的远端身份持久化回调。
            cancellation: 本地取消信号；不能保证远端已接收文件被撤回。
        Returns:
            远端身份与仅限当前 attempt 使用的上传地址，不应直接写入日志。
        """
        self._enter_stage(attempt, "submit", cancellation)
        if not source.is_file():
            raise MineruClientError("source_pdf_missing", attempt=attempt)
        if attempt.submission_state != "not_started":
            code = (
                "submission_unknown"
                if attempt.submission_state == "unknown"
                else "submission_already_started"
            )
            raise MineruClientError(code, attempt=attempt)
        # POST 发出后即存在不可重放的副作用，任何不确定响应均阻止再次提交。
        attempt.submission_state = "unknown"
        try:
            status, payload = await self._await_cancellable(
                lambda: self._transport.request_json(
                    "POST",
                    self.api_base + "/file-urls/batch",
                    headers={"Authorization": f"Bearer {self.config.api_token}"},
                    json_body={
                        # FileStore 原件使用无扩展名身份；远端使用中性 PDF 名，避免
                        # 扩展名校验失败，也不把操作者原始论文文件名交给服务商。
                        "files": [{"name": "document.pdf"}],
                        "model_version": self.config.model_version,
                    },
                    timeout=self._timeout(attempt),
                ),
                cancellation,
                attempt=attempt,
            )
        except MineruTransportError:
            raise MineruClientError("submission_unknown", attempt=attempt) from None
        if status != 200:
            raise MineruClientError("submission_unknown", attempt=attempt)
        try:
            batch_id, upload_url = _submission_fields(payload)
        except MineruClientError:
            raise MineruClientError("submission_unknown", attempt=attempt) from None
        attempt.remote_task_id = batch_id
        attempt.submission_state = "known"
        await self._await_cancellable(
            lambda: checkpoint_remote_task_id(batch_id), cancellation, attempt=attempt
        )
        return MineruSubmission(batch_id, upload_url)

    async def upload(
        self,
        source: Path,
        submission: MineruSubmission,
        *,
        attempt: MineruAttempt,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        """名称：上传原件。用途概要：流式 PUT，同一内容仅有限次安全重放。

        Args:
            source: 已预检的可信原件。
            submission: 当前 attempt 的提交结果；其中地址禁止持久化。
            attempt: 与提交相同的唯一预算上下文。
            cancellation: 本地取消信号。
        Returns:
            None；超时、取消或上传失败返回稳定领域错误。
        """
        self._enter_stage(attempt, "upload", cancellation)
        self._require_remote_task(attempt, submission.remote_task_id)
        outcome = await self._with_retries(
            lambda: self._upload_once(source, submission.upload_url, attempt),
            attempt=attempt,
            cancellation=cancellation,
        )
        assert outcome.value is None

    async def wait_for_completion(
        self,
        remote_task_id: str,
        *,
        attempt: MineruAttempt,
        cancellation: asyncio.Event | None = None,
    ) -> MineruCompletedResult:
        """名称：查询解析结果。用途概要：轮询已知 batch，继承原 attempt 总期限。

        Args:
            remote_task_id: 与 attempt 绑定且已持久化的 batch 标识。
            attempt: 原客户端仍然持有的上下文；恢复调用不能创建新期限。
            cancellation: 本地取消信号。
        Returns:
            成功结果地址及本次查询/重试次数；不包含远端响应正文。
        """
        self._enter_stage(attempt, "poll", cancellation)
        self._require_remote_task(attempt, remote_task_id)
        polls = retries = 0
        while True:
            outcome = await self._with_retries(
                lambda: self._poll_once(remote_task_id, attempt),
                attempt=attempt,
                cancellation=cancellation,
            )
            status, payload = outcome.value
            polls += 1
            retries += outcome.retry_count
            assert status == 200
            try:
                state, archive_url = _result_fields(payload)
            except MineruClientError as error:
                raise MineruClientError(error.error_code, attempt=attempt) from None
            if state == "done":
                if archive_url is None:
                    raise MineruClientError("remote_result_url_missing", attempt=attempt)
                return MineruCompletedResult(archive_url, polls, retries)
            if state == "failed":
                raise MineruClientError("remote_parse_failed", attempt=attempt)
            # 官方 batch 契约包含上传已完成但尚未入解析队列的 waiting-file；
            # 它仍消耗本 attempt 的原总预算，不能作为失败或重提交理由。
            if state not in {"waiting-file", "pending", "running", "converting"}:
                raise MineruClientError("remote_state_unknown", attempt=attempt)
            await self._sleep_with_deadline(attempt, cancellation)

    async def download(
        self,
        result: MineruCompletedResult,
        destination: Path,
        *,
        attempt: MineruAttempt,
        cancellation: asyncio.Event | None = None,
    ) -> None:
        """名称：下载解析 ZIP。用途概要：在原总期限内流式下载并限制实际字节数。

        Args:
            result: 查询阶段返回的临时签名地址，不能写入日志。
            destination: 当前 attempt 的专用结果路径。
            attempt: 提交和查询共用的上下文。
            cancellation: 本地取消信号。
        Returns:
            None；请求不转发 Authorization，失败不伪装成功。
        """
        self._enter_stage(attempt, "download", cancellation)
        await self._with_retries(
            lambda: self._transport.download_file(
                result.archive_url,
                destination,
                max_bytes=self.config.max_download_bytes,
                timeout=self._timeout(attempt),
            ),
            attempt=attempt,
            cancellation=cancellation,
        )

    async def aclose(self) -> None:
        """关闭传输；取消路径可调用，随后实例拒绝继续发送请求。"""
        if not self._closed:
            self._closed = True
            await self._transport.aclose()

    async def _with_retries(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        attempt: MineruAttempt,
        cancellation: asyncio.Event | None,
    ) -> _RetryResult:
        attempts = 0
        while True:
            try:
                return _RetryResult(
                    await self._await_cancellable(operation, cancellation, attempt=attempt),
                    attempts,
                )
            except MineruTransportError as error:
                if not error.retryable or attempts >= self.config.max_retries:
                    # 只接受本模块定义的稳定码，注入/远端异常正文不能成为日志摘要。
                    code = (
                        error.error_code
                        if error.error_code in _TRANSPORT_ERROR_CODES
                        else "remote_transport_failure"
                    )
                    raise MineruClientError(code, attempt=attempt) from None
                attempts += 1
                await self._sleep_with_deadline(attempt, cancellation)

    async def _upload_once(self, source: Path, upload_url: str, attempt: MineruAttempt) -> None:
        """执行一次预签名上传，并将可重试 HTTP 状态转为统一传输错误。"""
        status = await self._transport.upload_file(
            upload_url, source, timeout=self._timeout(attempt)
        )
        if status in {200, 201, 204}:
            return
        raise MineruTransportError(
            "remote_upload_http_error", retryable=retryable_http_status(status)
        )

    async def _poll_once(self, remote_task_id: str, attempt: MineruAttempt) -> tuple[int, Any]:
        """执行一次状态查询，并将服务端瞬时状态交给公共重试边界。"""
        status, payload = await self._transport.request_json(
            "GET",
            self.api_base + f"/extract-results/batch/{remote_task_id}",
            headers={"Authorization": f"Bearer {self.config.api_token}"},
            json_body=None,
            timeout=self._timeout(attempt),
        )
        if status != 200:
            raise MineruTransportError(
                "remote_poll_http_error", retryable=retryable_http_status(status)
            )
        return status, payload

    async def _await_cancellable(
        self,
        operation: Callable[[], Awaitable[Any]],
        cancellation: asyncio.Event | None,
        *,
        attempt: MineruAttempt,
    ) -> Any:
        self._require_open()
        if cancellation is not None and cancellation.is_set():
            await self.aclose()
            raise MineruClientError("cancelled", attempt=attempt)
        remaining = self._remaining(attempt)
        task = asyncio.create_task(operation())
        cancelled = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        watched = {task, cancelled} if cancelled is not None else {task}
        try:
            done, _ = await asyncio.wait(
                watched, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if cancelled is not None and cancelled in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self.aclose()
                raise MineruClientError("cancelled", attempt=attempt)
            if task not in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self.aclose()
                raise MineruClientError("remote_total_timeout", attempt=attempt)
            if self._monotonic() >= attempt.monotonic_deadline:
                # 已完成任务也须回收异常，避免未读取的异常被 asyncio 写入日志。
                await asyncio.gather(task, return_exceptions=True)
                raise MineruClientError("remote_total_timeout", attempt=attempt) from None
            return await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self.aclose()
            raise
        finally:
            # 外部取消与期限耗尽都必须回收 watcher，避免遗留本地请求或取消协程。
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if cancelled is not None:
                cancelled.cancel()
                await asyncio.gather(cancelled, return_exceptions=True)

    async def _sleep_with_deadline(
        self, attempt: MineruAttempt, cancellation: asyncio.Event | None
    ) -> None:
        remaining = self._remaining(attempt)
        if self.config.poll_interval_seconds >= remaining:
            raise MineruClientError("remote_total_timeout", attempt=attempt)
        await self._await_cancellable(
            lambda: self._sleep(self.config.poll_interval_seconds),
            cancellation,
            attempt=attempt,
        )

    def _remaining(self, attempt: MineruAttempt) -> float:
        remaining = attempt.monotonic_deadline - self._monotonic()
        if remaining <= 0:
            raise MineruClientError("remote_total_timeout", attempt=attempt)
        return remaining

    def _timeout(self, attempt: MineruAttempt) -> httpx.Timeout:
        remaining = self._remaining(attempt)
        return httpx.Timeout(
            connect=min(self.config.connect_timeout_seconds, remaining),
            read=min(self.config.read_timeout_seconds, remaining),
            write=min(self.config.read_timeout_seconds, remaining),
            pool=min(self.config.connect_timeout_seconds, remaining),
        )

    def _enter_stage(
        self, attempt: MineruAttempt, stage: str, cancellation: asyncio.Event | None
    ) -> None:
        if attempt._owner is not self._attempt_owner:
            raise MineruClientError("attempt_context_invalid", attempt=attempt)
        attempt.last_stage = stage
        self._require_open()
        if cancellation is not None and cancellation.is_set():
            raise MineruClientError("cancelled", attempt=attempt)
        self._remaining(attempt)

    def _require_remote_task(self, attempt: MineruAttempt, remote_task_id: str) -> None:
        if not _valid_identifier(remote_task_id):
            raise MineruClientError("remote_task_id_invalid", attempt=attempt)
        if attempt.remote_task_id != remote_task_id:
            raise MineruClientError("attempt_remote_task_mismatch", attempt=attempt)

    def _require_open(self) -> None:
        if self._closed:
            raise MineruClientError("mineru_client_closed")


def _api_base(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MINERU_BASE_URL 必须是无凭据、无查询的 HTTPS 地址")
    stripped = base_url.rstrip("/")
    return stripped if stripped.endswith("/api/v4") else stripped + "/api/v4"


def _safe_signed_url(value: Any) -> str:
    if not isinstance(value, str):
        raise MineruClientError("remote_url_invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # urlsplit 不会主动校验端口；读取 port 才能拒绝非数字/越界端口。
        # 畸形 IPv6/NFKC hostname 的 ValueError 也可能带出签名地址内容。
        _ = parsed.port
    except ValueError:
        raise MineruClientError("remote_url_invalid") from None
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise MineruClientError("remote_url_invalid")
    return value


def _valid_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value))


def _submission_fields(payload: Any) -> tuple[str, str]:
    data = payload.get("data") if isinstance(payload, dict) and payload.get("code") == 0 else None
    if not isinstance(data, dict) or not _valid_identifier(data.get("batch_id")):
        raise MineruClientError("remote_submission_invalid")
    urls = data.get("file_urls")
    if not isinstance(urls, list) or len(urls) != 1:
        raise MineruClientError("remote_submission_invalid")
    return data["batch_id"], _safe_signed_url(urls[0])


def _result_fields(payload: Any) -> tuple[str, str | None]:
    data = payload.get("data") if isinstance(payload, dict) and payload.get("code") == 0 else None
    values = data.get("extract_result") if isinstance(data, dict) else None
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise MineruClientError("remote_result_invalid")
    state = values[0].get("state")
    if not isinstance(state, str):
        raise MineruClientError("remote_state_unknown")
    url = values[0].get("full_zip_url")
    return state, _safe_signed_url(url) if url is not None else None
