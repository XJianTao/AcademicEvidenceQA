"""模块名称：单例异步生成客户端。

用途：封装 AsyncOpenAI 文本生成、有限期限、usage 与稳定错误。
核心设计：SDK 自动重试关闭，仅明确未受理的 429 可有限重试；未知结果不重发。
工程要点：不记录凭据、问题、回答或供应商异常；取消由 async HTTP 栈回收等待连接。
"""

import asyncio
from dataclasses import dataclass
import json
import time
from typing import Any

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

    def __init__(self, code: str, reserved_tokens: int, attempts: int):
        self.code, self.reserved_tokens, self.attempts = code, reserved_tokens, attempts
        super().__init__(code)


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
                    if (
                        len(response.choices) != 1
                        or response.choices[0].finish_reason != "stop"
                        or response.choices[0].message.tool_calls
                        or response.choices[0].message.refusal
                        or not isinstance(response.choices[0].message.content, str)
                        or not response.choices[0].message.content.strip()
                        or len(response.choices[0].message.content.encode()) > self.max_bytes
                    ):
                        raise GenerationError(
                            "generation_protocol_error", attempts * per_attempt, attempts
                        )
                    usage = response.usage
                    if usage and usage.completion_tokens > max_output_tokens:
                        raise GenerationError(
                            "generation_output_limit_unsupported",
                            max(attempts * per_attempt, usage.total_tokens),
                            attempts,
                        )
                    actual = usage.total_tokens if usage and usage.total_tokens > 0 else None
                    return GenerationResult(
                        response.choices[0].message.content,
                        {
                            "status": "reported" if actual else "missing",
                            "input_tokens": usage.prompt_tokens if usage else None,
                            "output_tokens": usage.completion_tokens if usage else None,
                            "reasoning_tokens": usage.completion_tokens_details.reasoning_tokens
                            if usage and usage.completion_tokens_details
                            else None,
                            "total_tokens": actual,
                            "attempts": attempts,
                            "accounted_tokens": (actual or per_attempt)
                            + (attempts - 1) * per_attempt,
                        },
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
                "generation_protocol_error", attempts * per_attempt, attempts
            ) from None
        raise GenerationError("generation_protocol_error", attempts * per_attempt, attempts)

    async def close(self) -> None:
        await self.sdk.close()
