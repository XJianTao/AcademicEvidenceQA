"""模块：MinerU HTTP 传输。

用途概要：执行在线解析所需的 JSON 请求、预签名上传和有界 ZIP 下载。

核心设计：非成功 HTTP 响应先按状态处理，成功 JSON 和下载内容分别限制字节数；
下载通过同目录临时文件与独占硬链接发布，已有目标不会被覆盖。

工程要点：请求上下文中的 HTTPX 日志不包含 token 或签名 URL；传输异常仅向
协议层提供稳定错误码和可重试标记，不输出远端响应正文。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from uuid import uuid4
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Protocol

import httpx


_PRIVATE_MINERU_REQUEST: ContextVar[bool] = ContextVar("private_mineru_request", default=False)


class _PrivateRequestLogFilter(logging.Filter):
    """仅抑制本请求上下文中的 HTTP 库记录，避免 INFO URL 和 DEBUG headers 泄漏。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _PRIVATE_MINERU_REQUEST.get()


def _install_private_request_log_filter() -> None:
    """为 HTTPX/HTTPCore 的固定 logger 加上下文过滤，不改变全局日志级别。"""
    names = {
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.connection_pool",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    }
    names.update(name for name in logging.Logger.manager.loggerDict if name.startswith("httpcore."))
    for name in names:
        logger = logging.getLogger(name)
        if not any(isinstance(item, _PrivateRequestLogFilter) for item in logger.filters):
            logger.addFilter(_PrivateRequestLogFilter())


class MineruTransportError(RuntimeError):
    """名称：传输层失败。用途概要：标记稳定错误码及请求是否可安全重放。"""

    def __init__(self, error_code: str, *, retryable: bool):
        self.error_code = error_code
        self.retryable = retryable
        super().__init__(error_code)


class MineruTransport(Protocol):
    """名称：注入式传输边界。用途概要：使协议故障可在本地确定性测试。"""

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        json_body: dict[str, Any] | None,
        timeout: httpx.Timeout,
    ) -> tuple[int, Any]: ...

    async def upload_file(self, url: str, source: Path, *, timeout: httpx.Timeout) -> int: ...

    async def download_file(
        self, url: str, destination: Path, *, max_bytes: int, timeout: httpx.Timeout
    ) -> None: ...

    async def aclose(self) -> None: ...


class HttpxMineruTransport:
    """名称：HTTPX 传输。用途概要：以有限 JSON 和流式文件 I/O 执行 HTTPS 请求。"""

    _MAX_API_JSON_BYTES = 1024 * 1024

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=True)
        _install_private_request_log_filter()

    @asynccontextmanager
    async def _stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[httpx.Response]:
        """把依赖库日志限定为当前协程的私有请求；响应结束后恢复其他请求日志。"""
        token = _PRIVATE_MINERU_REQUEST.set(True)
        try:
            async with self._client.stream(*args, **kwargs) as response:
                yield response
        finally:
            _PRIVATE_MINERU_REQUEST.reset(token)

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None,
        json_body: dict[str, Any] | None,
        timeout: httpx.Timeout,
    ) -> tuple[int, Any]:
        """返回非 200 状态供协议层分类；只解析成功响应的有界 JSON。"""
        try:
            async with self._stream(
                method, url, headers=headers, json=json_body, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    return response.status_code, None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self._MAX_API_JSON_BYTES:
                        raise MineruTransportError("remote_response_too_large", retryable=False)
                try:
                    return response.status_code, json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise MineruTransportError(
                        "remote_response_invalid", retryable=False
                    ) from error
        except MineruTransportError:
            raise
        except httpx.TimeoutException as error:
            raise MineruTransportError("remote_timeout", retryable=True) from error
        except httpx.HTTPError as error:
            raise MineruTransportError("remote_transport_failure", retryable=True) from error

    async def upload_file(self, url: str, source: Path, *, timeout: httpx.Timeout) -> int:
        """流式 PUT 后仅检查状态；不将无用的上传响应正文读入内存。"""
        try:
            async with self._stream(
                "PUT", url, content=_file_chunks(source), timeout=timeout
            ) as response:
                return response.status_code
        except httpx.TimeoutException as error:
            raise MineruTransportError("remote_upload_timeout", retryable=True) from error
        except httpx.HTTPError as error:
            raise MineruTransportError("remote_upload_transport_failure", retryable=True) from error

    async def download_file(
        self, url: str, destination: Path, *, max_bytes: int, timeout: httpx.Timeout
    ) -> None:
        """下载到同目录暂存文件，再原子地建立不可覆盖的目标硬链接。"""
        temporary = destination.with_name(f"{destination.name}.{uuid4().hex}.part")
        try:
            try:
                if destination.exists():
                    raise FileExistsError(destination)
                async with self._stream("GET", url, timeout=timeout) as response:
                    if response.status_code != 200:
                        raise MineruTransportError(
                            "remote_download_http_error",
                            retryable=retryable_http_status(response.status_code),
                        )
                    size_bytes = 0
                    with temporary.open("xb") as output:
                        async for chunk in response.aiter_bytes():
                            size_bytes += len(chunk)
                            if size_bytes > max_bytes:
                                raise MineruTransportError(
                                    "remote_download_too_large", retryable=False
                                )
                            output.write(chunk)
                        output.flush()
                    if size_bytes == 0:
                        raise MineruTransportError("remote_download_empty", retryable=False)
                # replace() 会覆盖旧产物；同目录硬链接的独占创建避免并发检查竞争。
                os.link(temporary, destination)
            except FileExistsError as error:
                raise MineruTransportError(
                    "remote_download_destination_exists", retryable=False
                ) from error
            except OSError as error:
                raise MineruTransportError(
                    "remote_download_publish_failure", retryable=False
                ) from error
        except MineruTransportError:
            raise
        except httpx.TimeoutException as error:
            raise MineruTransportError("remote_download_timeout", retryable=True) from error
        except httpx.HTTPError as error:
            raise MineruTransportError(
                "remote_download_transport_failure", retryable=True
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    async def aclose(self) -> None:
        await self._client.aclose()


async def _file_chunks(path: Path, chunk_bytes: int = 1024 * 1024) -> AsyncIterator[bytes]:
    """以固定块异步读取受理原件，避免上传时把 PDF 整体载入内存。"""
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, chunk_bytes):
            yield chunk


def retryable_http_status(status_code: int) -> bool:
    """只对限流和服务端瞬时错误重试；调用方仍须限制次数和总期限。"""
    return status_code == 429 or 500 <= status_code <= 599
