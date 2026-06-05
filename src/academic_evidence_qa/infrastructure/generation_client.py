"""模块名称：单例异步生成客户端。

用途：封装 AsyncOpenAI 文本生成、有限期限、usage 与稳定错误。
核心设计：SDK 自动重试关闭，仅明确未受理的 429 可有限重试；未知结果不重发。
工程要点：不记录凭据、问题、回答或供应商异常；取消由 async HTTP 栈回收等待连接。
"""

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any, cast

from openai import (
    AsyncOpenAI,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
)
from openai.types.chat import ChatCompletionMessageParam


def serialize(value: Any) -> bytes:
    """中文及封装均计入保守字节估算；不使用 embedding tokenizer。"""
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def estimated_tokens(value: Any) -> int:
    """未知 tokenizer 时每个序列化字节暂按一个 token，加固定协议余量。"""
    return len(serialize(value)) + 256


@dataclass(frozen=True)
class GenerationResult:
    """名称：生成结果。用途：将正文、计费事实和实际请求次数分离。"""

    text: str
    usage: dict[str, Any]
    seconds: float


class GenerationError(Exception):
    """名称：脱敏生成失败。用途：只公开固定类别及保守占用，不携带远端正文。"""

    def __init__(
        self, code: str, reserved_tokens: int, attempts: int, *, diagnostics=None, usage=None
    ):
        self.code, self.reserved_tokens, self.attempts = code, reserved_tokens, attempts
        self.diagnostics, self.usage = diagnostics or {}, usage or {}
        super().__init__(code)


def response_metadata(response, *, attempts, per_attempt):
    """名称：响应结构摘要。用途：在正文校验前保留白名单诊断及可得 usage。

    Args:
        response: SDK 响应；不得序列化完整对象、正文、拒绝文本或私有推理。
        attempts, per_attempt: 实际次数及先前未知尝试的保守预留。
    Returns:
        脱敏结构摘要及归一化 usage；未知枚举只记 other，拒绝字符串透传。
    """
    # 兼容服务可能返回缺失结构；结构错误也不能丢掉已经返回的 usage。
    choices = getattr(response, "choices", None)
    choices = choices if isinstance(choices, list) else []
    message = getattr(choices[0], "message", None) if len(choices) == 1 else None
    content = getattr(message, "content", None)
    finish = getattr(choices[0], "finish_reason", None) if len(choices) == 1 else None
    diagnostics = {
        "choice_count": len(choices),
        "finish_reason": finish
        if finish in ("stop", "length", "tool_calls", "content_filter", "function_call")
        else "other",
        "content_type": type(content).__name__
        if type(content) in (str, list, dict, int, bool, float, type(None))
        else "other",
        "content_empty": content is None or (isinstance(content, str) and not content.strip()),
        "has_refusal": bool(getattr(message, "refusal", None)),
        "has_tool_calls": bool(getattr(message, "tool_calls", None)),
    }
    raw = getattr(response, "usage", None)

    def token(field):
        value = getattr(raw, field, None)
        return value if type(value) is int and value >= 0 else None

    actual = token("total_tokens")
    details = getattr(raw, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    usage = {
        "status": "reported" if actual is not None and actual > 0 else "missing",
        "input_tokens": token("prompt_tokens"),
        "output_tokens": token("completion_tokens"),
        "reasoning_tokens": reasoning if type(reasoning) is int and reasoning >= 0 else None,
        "total_tokens": actual,
        "attempts": attempts,
        "accounted_tokens": (actual or per_attempt) + (attempts - 1) * per_attempt,
    }
    return diagnostics, usage


class GenerationClient:
    """名称：生成边界。用途：生命周期共享 SDK，不与业务重试叠加。

    Args:
        sdk: bootstrap 构建、max_retries=0 的 AsyncOpenAI。
        model: 用户配置的准确模型名。
        timeout_seconds, max_retries: 总期限与仅 429 的有限重试。
        context_tokens, max_bytes: 完整请求含输出预留的保守限制。
    Returns:
        generate 返回归一化结果；close 关闭共享连接池。
    """

    def __init__(
        self,
        sdk: AsyncOpenAI,
        *,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        context_tokens: int,
        max_bytes: int,
    ):
        self.sdk, self.model = sdk, model
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.context_tokens, self.max_bytes = context_tokens, max_bytes

    def request_body(
        self, messages: list[ChatCompletionMessageParam], max_output_tokens: int
    ) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_output_tokens,
            "stream": False,
        }

    def fits(self, messages: list[ChatCompletionMessageParam], max_output_tokens: int) -> bool:
        body = self.request_body(messages, max_output_tokens)
        return (
            len(serialize(body)) <= self.max_bytes
            and estimated_tokens(body) + max_output_tokens <= self.context_tokens
        )

    async def generate(
        self, messages: list[ChatCompletionMessageParam], max_output_tokens: int
    ) -> GenerationResult:
        """名称：一次业务生成。用途：复核实际参数预算，有限 429 重试，不重试未知结果。

        Args:
            messages: 最终完整 system/user 结构，不再追加隐藏上下文。
            max_output_tokens: 实际传入 SDK 的输出硬限制。
        Returns:
            正文及 usage；usage 缺失时保守占用所有尝试的输入/最大输出。
        Raises:
            GenerationError: 稳定错误；CancelledError 原样传播，不承诺远端停止计费。
        """
        if not self.fits(messages, max_output_tokens):
            raise GenerationError("generation_budget_exhausted", 0, 0)
        per_attempt = (
            estimated_tokens(self.request_body(messages, max_output_tokens)) + max_output_tokens
        )
        started, attempts = time.monotonic(), 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                for retry in range(self.max_retries + 1):
                    attempts += 1
                    try:
                        response = await self.sdk.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            # 真实 qwen endpoint 的 max_tokens 未限制 reasoning tokens；
                            # max_completion_tokens 才覆盖完整计费输出，不做隐式参数降级。
                            max_completion_tokens=max_output_tokens,
                            stream=False,
                            timeout=self.timeout_seconds,
                        )
                    except APIStatusError as error:
                        if error.status_code == 429 and retry < self.max_retries:
                            await asyncio.sleep(min(2**retry, 2))
                            continue
                        raise GenerationError(
                            "generation_http_error", attempts * per_attempt, attempts
                        ) from None
                    diagnostics, usage = response_metadata(
                        response, attempts=attempts, per_attempt=per_attempt
                    )
                    choices = getattr(response, "choices", None)
                    message = (
                        getattr(choices[0], "message", None)
                        if isinstance(choices, list) and len(choices) == 1
                        else None
                    )
                    content = getattr(message, "content", None)
                    rejected = []
                    if diagnostics["choice_count"] != 1:
                        rejected.append("choice_count")
                    if diagnostics["finish_reason"] != "stop":
                        rejected.append("finish_reason")
                    if diagnostics["has_tool_calls"]:
                        rejected.append("tool_calls")
                    if diagnostics["has_refusal"]:
                        rejected.append("refusal")
                    if not isinstance(content, str):
                        rejected.append("content_type")
                    elif not content.strip():
                        rejected.append("content_empty")
                    elif len(content.encode()) > self.max_bytes:
                        rejected.append("content_bytes_exceeded")
                    if (
                        usage["output_tokens"] is not None
                        and usage["output_tokens"] > max_output_tokens
                    ):
                        rejected.append("output_tokens_exceeded")
                    if rejected:
                        diagnostics["rejection_categories"] = rejected
                        raise GenerationError(
                            "generation_output_limit_unsupported"
                            if "output_tokens_exceeded" in rejected
                            else "generation_protocol_error",
                            usage["accounted_tokens"],
                            attempts,
                            diagnostics=diagnostics,
                            usage=usage,
                        )
                    return GenerationResult(
                        cast(str, content),  # 上方拒绝类别已保证纯非空文本；仅向类型检查器收窄。
                        usage,
                        time.monotonic() - started,
                    )
        except (TimeoutError, APITimeoutError):
            raise GenerationError("generation_timeout", attempts * per_attempt, attempts) from None
        except APIConnectionError:
            raise GenerationError(
                "generation_connection_error", attempts * per_attempt, attempts
            ) from None
        except (APIResponseValidationError, ValueError, AttributeError, TypeError):
            raise GenerationError(
                "generation_protocol_error",
                attempts * per_attempt,
                attempts,
                diagnostics={"rejection_categories": ["malformed_response"]},
            ) from None
        raise GenerationError("generation_protocol_error", attempts * per_attempt, attempts)

    async def close(self) -> None:
        await self.sdk.close()
