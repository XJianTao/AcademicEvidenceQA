"""模块名称：专用有界同步执行器。

用途：模型与来源 I/O 各自使用单线程队列，不阻塞 API 事件循环。
核心设计：取消立即移除尚未开始的任务；运行中线程直到真正结束才让下一任务执行。
工程要点：等待与总期限分离；关闭排队请求并等待在途同步调用，不假称强杀线程。
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable


class RuntimeErrorCode(Exception):
    """仅携带可公开的稳定错误码。"""


@dataclass
class _Work:
    function: Callable[[], Any]
    future: asyncio.Future
    enqueued: float
    wait_seconds: float


class BoundedExecutor:
    """名称：有限队列执行器。用途：串行同步工作，真实完成后释放资源。

    Args:
        capacity: 等待任务上限，不含已开始的一项。
    Returns:
        在同一事件循环内使用并显式关闭的执行器。
    """

    def __init__(self, capacity: int = 4):
        if capacity < 1:
            raise ValueError("executor_capacity_invalid")
        self.capacity = capacity
        self.queue: deque[_Work] = deque()
        self.signal = asyncio.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aeqa-bounded")
        self.closed = False
        self.active = False
        self.worker: asyncio.Task | None = None

    async def run(self, function, *, wait_seconds=30.0, total_seconds=120.0):
        """名称：提交工作。用途：返回结果与排队/执行耗时，期限耗尽不释放在途线程。

        Args:
            function: 自身具有输入/网络/文件预算的同步操作。
            wait_seconds, total_seconds: 正的排队及请求总预算。
        Returns:
            (结果, 耗时字典)。
        Raises:
            RuntimeErrorCode: 队满、关闭、等待或总期限耗尽。
        """
        if self.closed:
            raise RuntimeErrorCode("executor_closed")
        if wait_seconds <= 0 or total_seconds <= 0:
            raise ValueError("executor_timeout_invalid")
        if len(self.queue) >= self.capacity:
            raise RuntimeErrorCode("executor_busy")
        loop = asyncio.get_running_loop()
        work = _Work(function, loop.create_future(), time.monotonic(), wait_seconds)
        self.queue.append(work)
        if self.worker is None:
            self.worker = asyncio.create_task(self._consume())
        self.signal.set()

        def expire_wait():
            if work in self.queue and not work.future.done():
                self.queue.remove(work)
                work.future.set_exception(RuntimeErrorCode("executor_queue_timeout"))

        timer = loop.call_later(wait_seconds, expire_wait)
        try:
            return await asyncio.wait_for(asyncio.shield(work.future), total_seconds)
        except TimeoutError as error:
            work.future.cancel()
            raise RuntimeErrorCode("executor_total_timeout") from error
        except asyncio.CancelledError:
            work.future.cancel()
            raise
        finally:
            timer.cancel()
            if work.future.cancelled() and work in self.queue:
                self.queue.remove(work)

    async def _consume(self):
        while not self.closed or self.queue:
            if not self.queue:
                self.signal.clear()
                await self.signal.wait()
                continue
            work = self.queue.popleft()
            if work.future.done():
                continue
            queued = time.monotonic() - work.enqueued
            if queued >= work.wait_seconds:
                work.future.set_exception(RuntimeErrorCode("executor_queue_timeout"))
                continue
            self.active = True
            started = time.monotonic()
            try:
                result = await asyncio.get_running_loop().run_in_executor(self.pool, work.function)
                if not work.future.done():
                    work.future.set_result(
                        (
                            result,
                            {
                                "queue_seconds": queued,
                                "inference_seconds": time.monotonic() - started,
                            },
                        )
                    )
            except Exception as error:
                if not work.future.done():
                    work.future.set_exception(error)
            finally:
                self.active = False

    async def close(self):
        """拒绝新任务、终结排队请求；真正完成在途工作后关闭专用线程。"""
        self.closed = True
        for work in self.queue:
            if not work.future.done():
                work.future.set_exception(RuntimeErrorCode("executor_closed"))
        self.queue.clear()
        self.signal.set()
        if self.worker:
            await self.worker
        self.pool.shutdown(wait=True)
