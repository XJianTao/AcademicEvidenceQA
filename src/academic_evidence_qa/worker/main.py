"""模块名称：本地解析与索引 Worker 入口。

用途：组装已有存储与单 owner 模型，周期领取索引意图。
核心设计：一次只执行一个 Job，postgres 是唯一权威。
工程要点：不启动 API；显式 --with-parsing 启用获准外发，退出保留租约供安全接管。
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore
from academic_evidence_qa.infrastructure.qdrant_store import QdrantStore
from academic_evidence_qa.infrastructure.model_runtime import (
    DEFAULT_BGE_PATH,
    DEFAULT_RERANKER_PATH,
)
from academic_evidence_qa.ingestion.job_runner import IndexJobRunner
from academic_evidence_qa.ingestion.parse_runner import ParseJobRunner


def argument_parser():
    """名称：参数定义。用途：Worker 与验证入口共用，不包含凭据。

    Args:
        无。
    Returns:
        尚未解析的 ArgumentParser，由调用入口添加验证参数。
    """
    parser = argparse.ArgumentParser(description="M1-C2 本地索引 Worker")
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--library-id", type=UUID)
    parser.add_argument("--owner", default=f"worker-{uuid4().hex}")
    parser.add_argument("--lease-seconds", type=float, default=60)
    parser.add_argument("--heartbeat-seconds", type=float, default=10)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:56333")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BGE_PATH)
    parser.add_argument("--reranker-path", type=Path, default=DEFAULT_RERANKER_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--with-parsing",
        action="store_true",
        help="M1-C3 正式导入调度；CLI 索引验证默认保持 C2 语义",
    )
    return parser


async def run(args, *, once=False):
    """名称：运行生命周期。用途：参数组装，不复制索引业务。

    Args:
        args: 已解析的路径、租约与存储参数。
        once: 验证入口只执行一次。
    Returns:
        单次结果或持续轮询；finally 关闭本进程连接。
    """
    if args.poll_seconds <= 0 or args.batch_size < 1:
        raise ValueError("invalid_worker_config")
    settings = load_settings()
    database = Database(settings.database_url.set(database=args.database_name))
    qdrant = QdrantStore(args.qdrant_url)
    try:
        store = LocalFileStore(args.data_root)
        parse_runner = ParseJobRunner(
            database,
            store,
            settings,
            lease_seconds=args.lease_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        runner = IndexJobRunner(
            database,
            store,
            qdrant,
            owner=args.owner,
            lease_seconds=args.lease_seconds,
            heartbeat_seconds=args.heartbeat_seconds,
            model_path=args.model_path,
            reranker_path=args.reranker_path,
            batch_size=args.batch_size,
            max_json_bytes=settings.result_max_json_bytes,
            require_parse_ready=args.with_parsing,
        )
        while True:
            if args.with_parsing:
                parsed = await parse_runner.run_once(library_id=args.library_id)
                if parsed is not None:
                    print(json.dumps(parsed, ensure_ascii=False))
            result = await runner.run_once(library_id=args.library_id)
            if once or args.once:
                return result
            if result is not None:
                print(json.dumps(result, ensure_ascii=False))
            await asyncio.sleep(args.poll_seconds)
    finally:
        qdrant.close()
        await database.close()


def main():
    """名称：Worker 入口。用途：启动原生轮询，异常不打印敏感配置。

    Args:
        无；从命令行读取参数。
    Returns:
        成功为 0，失败为 1，键盘中断为 130；不清理历史数据。
    """
    args = argument_parser().parse_args()
    try:
        result = asyncio.run(run(args), loop_factory=create_database_event_loop)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result is None or result.get("status") == "passed" else 1
    except KeyboardInterrupt:
        return 130
    except Exception:
        print('{"status":"failed","error_code":"worker_failed"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
