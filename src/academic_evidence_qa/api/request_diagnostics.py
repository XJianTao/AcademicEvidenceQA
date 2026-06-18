"""模块名称：有界 HTTP 请求关联诊断。

用途：以受控请求标识关联 API 完成日志与脱敏字段验证错误。
核心设计：纯 ASGI 透传流，路由只记录模板；字段名和错误类型来自可信 schema。
工程要点：不读取正文、不记录值或异常消息；独立 CLI handler 不改变第三方日志。
"""

import json
import logging
import re
import time
from typing import cast, get_args
from uuid import uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorType
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from academic_evidence_qa.api import schemas
from academic_evidence_qa.contracts import rag


LOGGER_NAME = "academic_evidence_qa.api.requests"
_logger = logging.getLogger(LOGGER_NAME)
_REQUEST_ID = re.compile(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE"})
_ERROR_TYPES = frozenset(get_args(ErrorType))
_LOCATION_NAMES = frozenset(
    {"body", "path", "query", "header", "cookie", "idempotency-key"}
    | {
        name
        for module in (schemas, rag)
        for model in vars(module).values()
        if isinstance(model, type) and issubclass(model, BaseModel)
        for name in cast(type[BaseModel], model).model_fields
    }
)
_MAX_ERRORS = 16
_MAX_LOCATION_DEPTH = 8


def configure_request_logging() -> None:
    """名称：CLI 关联日志配置。用途：仅为本模块添加一次 INFO 输出。

    Args:
        无；嵌入应用可自行配置同名 logger，不调用此函数。
    Returns:
        None；重复调用不会重复添加 handler，不修改 root 或第三方日志。
    """
    handler_name = "academic_request_diagnostics"
    if not any(handler.name == handler_name for handler in _logger.handlers):
        handler = logging.StreamHandler()
        handler.name = handler_name
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def request_id(scope: Scope) -> str:
    """读取中间件已绑定的受控 UUID，不从异常或业务正文取值。"""
    return scope["state"]["academic_request_id"]


def _route_template(scope: Scope) -> str:
    # 路由对象来自应用注册表，原始 path/query 和动态路径值均不进入日志。
    return getattr(scope.get("route"), "path", "<unmatched>")


def _write_event(event: str, scope: Scope, **fields: object) -> None:
    _logger.info(
        json.dumps(
            {
                "event": event,
                "request_id": request_id(scope),
                "method": scope["method"] if scope["method"] in _METHODS else "OTHER",
                "route": _route_template(scope),
                **fields,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )


def log_request_validation(
    request: Request,
    error: RequestValidationError | ValidationError,
    *,
    body: bool = False,
) -> None:
    """名称：字段验证诊断。用途：保留位置/类型，排除消息、值、ctx 和正文。

    Args:
        request: 带受控关联标识的当前请求。
        error: 框架验证错误或被 request_invalid 包装的 Pydantic 验证错误。
        body: 手动 JSON/schema 验证缺少 body 前缀时显式补齐。
    Returns:
        None；最多记录 16 项、每项 8 层位置，未知字段和类型以固定占位符表示。
    """
    errors = (
        error.errors(include_url=False, include_context=False, include_input=False)
        if isinstance(error, ValidationError)
        else error.errors()
    )
    summaries = []
    for item in errors[:_MAX_ERRORS]:
        location = item.get("loc", ())
        if not isinstance(location, (tuple, list)):
            location = ()
        safe_location = ["body"] if body else []
        for component in location[: _MAX_LOCATION_DEPTH - len(safe_location)]:
            safe_location.append(
                component
                if isinstance(component, str) and component in _LOCATION_NAMES
                else "[index]"
                if type(component) is int
                else "<field>"
            )
        error_type = item.get("type")
        summaries.append(
            {
                "location": safe_location,
                "type": error_type
                if isinstance(error_type, str) and error_type in _ERROR_TYPES
                else "validation_error",
            }
        )
    _write_event(
        "request_validation_failed",
        request.scope,
        validation_errors=summaries,
        errors_truncated=len(errors) > _MAX_ERRORS,
    )


class RequestDiagnosticsMiddleware:
    """名称：HTTP 关联中间件。用途：不缓冲正文地关联并发和流式请求。

    Args:
        app: 下游 ASGI 应用；非 HTTP scope 原样透传。
    Returns:
        中间件实例；有效 UUID 沿用，缺失、重复或非法首部生成新 UUID。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = [value for name, value in scope["headers"] if name.lower() == b"x-request-id"]
        identity = (
            supplied[0].decode("ascii").lower()
            if len(supplied) == 1 and _REQUEST_ID.fullmatch(supplied[0])
            else str(uuid4())
        )
        scope.setdefault("state", {})["academic_request_id"] = identity
        started = time.monotonic()
        status_code, complete = 503, False

        async def send_response(message: Message) -> None:
            nonlocal status_code, complete
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                message = {**message, "headers": [*headers, (b"x-request-id", identity.encode())]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        try:
            await self.app(scope, receive, send_response)
        finally:
            _write_event(
                "request_complete",
                scope,
                status_code=status_code,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                response_complete=complete,
            )
