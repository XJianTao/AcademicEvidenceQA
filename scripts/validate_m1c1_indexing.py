"""模块名称：M1-C1 显式重建验证 CLI。

用途：组装正式 pipeline 并以独立数据库连接回读发布结果。
核心设计：只操作操作者指定的已有 Library 与存储，不上传 PDF。
工程要点：输出脱敏摘要；不创建 Worker，不清空库或删除 collection。
"""

import argparse
import asyncio
import json
import ctypes
import sys
import re
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.database.models import IndexGeneration, Library
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore
from academic_evidence_qa.infrastructure.model_runtime import (
    DEFAULT_BGE_PATH,
    DEFAULT_RERANKER_PATH,
)
from academic_evidence_qa.infrastructure.qdrant_store import QdrantStore
from academic_evidence_qa.ingestion.pipeline import rebuild_library
from academic_evidence_qa.ingestion.index_builder import content_hash


def process_peak_memory_mib() -> float | None:
    """读取 Windows 进程生命周期峰值工作集；不可测时明确返回 None，不估算。"""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return round(counters.PeakWorkingSetSize / 1024**2, 2)


async def run(args) -> dict:
    """名称：运行验证。用途：CLI 参数组装与独立连接回读，不复制业务逻辑。"""
    settings = load_settings()
    url = (
        settings.database_url.set(database=args.database_name)
        if args.database_name
        else settings.database_url
    )
    database = Database(url)
    qdrant = QdrantStore(args.qdrant_url)
    try:
        result = await rebuild_library(
            database,
            args.library_id,
            LocalFileStore(args.data_root),
            qdrant,
            model_path=args.model_path,
            reranker_path=args.reranker_path,
            max_json_bytes=settings.result_max_json_bytes,
            batch_size=args.batch_size,
        )
        # 新生命周期保证回读不是写入会话缓存。
        independent = Database(url)
        try:
            async with independent.sessions() as session:
                library = await session.get(Library, args.library_id)
                generation = await session.scalar(
                    select(IndexGeneration).where(
                        IndexGeneration.generation_id == UUID(result["generation_id"])
                    )
                )
                if (
                    library is None
                    or generation is None
                    or library.active_generation_id != generation.generation_id
                    or generation.state != "published"
                    or content_hash(generation.manifest) != result["manifest_sha256"]
                ):
                    raise ValueError("independent_publication_read_failed")
                result["independent_read"] = "passed"
                result["process_peak_working_set_mib"] = process_peak_memory_mib()
        finally:
            await independent.close()
        return result
    finally:
        qdrant.close()
        await database.close()


def main() -> int:
    """名称：验证入口。用途：只输出计数、身份和稳定错误码。"""
    parser = argparse.ArgumentParser(description="M1-C1：重建已有 Library，不执行远端解析")
    parser.add_argument("--library-id", type=UUID, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--database-name")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:56333")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BGE_PATH)
    parser.add_argument("--reranker-path", type=Path, default=DEFAULT_RERANKER_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须为正数")
    try:
        result = asyncio.run(run(args), loop_factory=create_database_event_loop)
        exit_code = 0
    except Exception as error:
        code = (
            str(error)
            if type(error).__module__.startswith("academic_evidence_qa")
            or isinstance(error, ValueError)
            else "validation_failed"
        )
        # 即使 ValueError 来自第三方，也仅接受稳定码，不能回显 URL/凭据/论文正文。
        code = code if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) else "validation_failed"
        result = {"status": "failed", "error_code": code}
        exit_code = 1
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
