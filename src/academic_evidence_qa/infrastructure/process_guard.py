"""模块：Windows 受限子进程防护。

用途概要：以 suspended -> Job Object -> resume 的顺序运行不可信 PDF、归档和标准化子进程。

核心设计：Job Object 设置进程内存限额和 KILL_ON_JOB_CLOSE，使子孙进程随父任务一并终止。

工程要点：防护无法建立时 fail closed；本模块不充当队列或通用任务框架。
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ResourceGuardError(RuntimeError):
    """资源防护错误，error_code 可映射为稳定的领域失败。"""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True)
class GuardedProcessResult:
    """受限进程的实际结果与回收时间。"""

    exit_code: int
    duration_seconds: float
    terminated: bool


@dataclass(frozen=True)
class GuardedProcessRequest:
    """受限运行请求。

    Args:
        command: 固定由服务拼出的可执行命令，不能来自 HTTP 输入。
        working_directory: 服务控制的任务目录。
        timeout_seconds: 总墙钟时限。
        memory_mib: Job Object 的每进程提交内存上限。
    Returns:
        可传给 WindowsProcessGuard.run 的不可变请求。
    """

    command: Sequence[str]
    working_directory: Path
    timeout_seconds: float
    memory_mib: int


if sys.platform == "win32":
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INFINITE = 0xFFFFFFFF

    class _StartupInfo(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(_StartupInfo),
        ctypes.POINTER(_ProcessInformation),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


class WindowsProcessGuard:
    """名称：Windows Job Object 防护。用途概要：运行一项受限子进程。

    Args:
        poll_interval_seconds: 检查 timeout/cancel 的最大等待间隔。
    Returns:
        没有常驻资源的防护实例。
    """

    def __init__(self, poll_interval_seconds: float = 0.05):
        self.poll_interval_seconds = poll_interval_seconds

    def run(
        self,
        request: GuardedProcessRequest,
        *,
        cancellation: threading.Event | None = None,
    ) -> GuardedProcessResult:
        """受限执行固定命令。

        Args:
            request: 服务生成的命令及资源限制。
            cancellation: 可选本地取消标志；触发时终止整个 Job 并等待回收。
        Returns:
            退出码、总耗时和是否由 guard 终止。
        Raises:
            ResourceGuardError: 防护不能建立、超时、取消或子进程无法正常结束。
        """
        if sys.platform != "win32":
            raise ResourceGuardError("resource_guard_unavailable")
        if not request.command or request.timeout_seconds <= 0 or request.memory_mib <= 0:
            raise ResourceGuardError("invalid_resource_limit")
        request.working_directory.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ResourceGuardError("resource_guard_unavailable")
        process_info = _ProcessInformation()
        process_assigned_to_job = False
        try:
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            limits.ProcessMemoryLimit = request.memory_mib * 1024 * 1024
            if not kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ResourceGuardError("resource_guard_unavailable")
            startup = _StartupInfo()
            startup.cb = ctypes.sizeof(startup)
            command_line = ctypes.create_unicode_buffer(
                subprocess.list2cmdline(list(request.command))
            )
            created = kernel32.CreateProcessW(
                None,
                command_line,
                None,
                None,
                False,
                CREATE_SUSPENDED | CREATE_NO_WINDOW,
                None,
                str(request.working_directory),
                ctypes.byref(startup),
                ctypes.byref(process_info),
            )
            if not created:
                raise ResourceGuardError("resource_guard_unavailable")
            if not kernel32.AssignProcessToJobObject(job, process_info.hProcess):
                raise ResourceGuardError("resource_guard_unavailable")
            process_assigned_to_job = True
            if kernel32.ResumeThread(process_info.hThread) == 0xFFFFFFFF:
                raise ResourceGuardError("resource_guard_unavailable")
            terminated = False
            while True:
                remaining = request.timeout_seconds - (time.monotonic() - start)
                if cancellation is not None and cancellation.is_set():
                    terminated = True
                    self._terminate_job(job)
                    self._wait_reaped(process_info.hProcess)
                    raise ResourceGuardError("cancelled")
                if remaining <= 0:
                    terminated = True
                    self._terminate_job(job)
                    self._wait_reaped(process_info.hProcess)
                    raise ResourceGuardError("process_timeout")
                wait_ms = max(1, int(min(remaining, self.poll_interval_seconds) * 1000))
                status = kernel32.WaitForSingleObject(process_info.hProcess, wait_ms)
                if status == WAIT_OBJECT_0:
                    code = wintypes.DWORD()
                    if not kernel32.GetExitCodeProcess(process_info.hProcess, ctypes.byref(code)):
                        raise ResourceGuardError("resource_guard_unavailable")
                    return GuardedProcessResult(code.value, time.monotonic() - start, terminated)
                if status != WAIT_TIMEOUT:
                    raise ResourceGuardError("resource_guard_unavailable")
        finally:
            # Job 归属尚未建立时，KILL_ON_JOB_CLOSE 无法覆盖这个挂起进程。
            if process_info.hProcess and not process_assigned_to_job:
                kernel32.TerminateProcess(process_info.hProcess, 1)
                kernel32.WaitForSingleObject(process_info.hProcess, 5000)
            if process_info.hThread:
                kernel32.CloseHandle(process_info.hThread)
            if process_info.hProcess:
                kernel32.CloseHandle(process_info.hProcess)
            # KILL_ON_JOB_CLOSE is a final guard for a child that escaped normal result handling.
            kernel32.CloseHandle(job)

    @staticmethod
    def _terminate_job(job: int) -> None:
        if not kernel32.TerminateJobObject(job, 1):
            raise ResourceGuardError("process_cleanup_failed")

    @staticmethod
    def _wait_reaped(process: int) -> None:
        if kernel32.WaitForSingleObject(process, 5000) != WAIT_OBJECT_0:
            raise ResourceGuardError("process_cleanup_failed")
