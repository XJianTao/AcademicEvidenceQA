"""MinerU M0 作业等待策略。

模块名称：MinerU 作业状态等待器
用途概要：为离线验证提供可控的状态轮询、超时、取消和有限重试语义。
核心设计：只依赖注入的状态读取、时钟和休眠函数，不发起网络请求，也不承担持久化任务。
工程要点：未知状态按有限重试处理；终态失败、取消和超时保留明确错误类别。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MineruJobError(RuntimeError):
    """MinerU 验证作业未能得到成功终态。"""


class MineruJobCancelled(MineruJobError):
    """调用方请求取消等待。"""


class MineruJobTimedOut(MineruJobError):
    """等待超过有限时限。"""


def wait_for_mineru_result(
    fetch_status: Callable[[], dict[str, Any]],
    *,
    timeout_seconds: float,
    max_attempts: int,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    cancelled: Callable[[], bool] = lambda: False,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """等待 fake 或真实适配器返回成功状态。

    Args:
        fetch_status: 读取一次远端状态的函数；不得在此函数中隐藏无限重试。
        timeout_seconds: 总等待时间上限。
        max_attempts: 最多读取状态的次数，必须为正数。
        sleep: 可注入休眠函数，测试中可记录而不实际等待。
        monotonic: 可注入单调时钟，避免使用墙上时钟计算超时。
        cancelled: 返回 True 时立即停止等待。
        poll_interval_seconds: 两次状态读取之间的休眠秒数。

    Returns:
        成功终态对应的状态字典。

    Raises:
        MineruJobCancelled: 调用方请求取消。
        MineruJobTimedOut: 超过时间或尝试次数上限。
        MineruJobError: 服务返回失败终态或参数非法。
    """
    if timeout_seconds <= 0 or max_attempts <= 0 or poll_interval_seconds < 0:
        raise ValueError(
            "timeout_seconds、max_attempts 必须为正数，poll_interval_seconds 不得为负数"
        )

    started = monotonic()
    for attempt in range(max_attempts):
        if cancelled():
            raise MineruJobCancelled("MinerU wait cancelled")
        if monotonic() - started >= timeout_seconds:
            raise MineruJobTimedOut("MinerU wait timed out")

        status = fetch_status()
        state = status.get("state")
        if state == "done":
            return status
        if state == "failed":
            raise MineruJobError("MinerU returned failed state")
        if attempt + 1 >= max_attempts:
            break
        sleep(poll_interval_seconds)

    raise MineruJobTimedOut("MinerU retry limit reached before done state")
