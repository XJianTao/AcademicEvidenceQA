"""模块名称：M1-C2 Worker 验证 CLI。

用途：复用正式 Worker，一次领取后独立连接回读任务与活动指针。
核心设计：只组装参数和摘要，不包含构建、接管或发布逻辑。
工程要点：不创建库、不上传论文、不删除 collection。
"""

import asyncio
import json
from pathlib import Path
from uuid import UUID

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.database.models import (
    BuildIntent,
    Library,
    IndexGeneration,
)
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.worker.main import argument_parser, run
from academic_evidence_qa.ingestion.job_runner import stable_error_code


async def validate(args):
    """名称：独立核验。用途：新连接核对 Job、attempt、generation 与 active。

    Args:
        args: Worker 参数及专用验证数据库名称。
    Returns:
        正式执行结果附独立回读状态；无任务明确返回 not_started。
    Raises:
        ValueError: 任一数据库身份与执行摘要不匹配，禁止标记通过。
    """
    result = await run(args, once=True)
    if result is None:
        return {"status": "not_started", "reason": "no_claimable_job"}
    if result["status"] != "passed":
        return result
    database = Database(load_settings().database_url.set(database=args.database_name))
    try:
        async with database.sessions() as session:
            job = await session.get(BuildIntent, UUID(result["job_id"]))
            if job is None:
                raise ValueError("independent_worker_read_failed")
            library = await session.get(Library, job.library_id)
            generation = await session.get(IndexGeneration, job.generation_id)
            if (
                library is None
                or generation is None
                or job.state != "succeeded"
                or str(job.library_id) != result["library_id"]
                or str(job.generation_id) != result["generation_id"]
                or generation.library_id != job.library_id
                or job.membership_revision != generation.membership_revision
                or job.membership_revision != result["membership_revision"]
                or job.attempt_id != generation.attempt_id
                or str(job.attempt_id) != result["attempt_id"]
                or job.lease_version != result["lease_version"]
                or job.lease_owner != result["lease_owner"]
                or generation.collection_name != result["collection_name"]
                or generation.manifest_sha256 != result["manifest_sha256"]
                or library.active_generation_id != generation.generation_id
                or generation.state != "published"
            ):
                raise ValueError("independent_worker_read_failed")
            result["independent_read"] = "passed"
        return result
    finally:
        await database.close()


def main():
    """名称：验证入口。用途：保存脱敏自验结果，异常不打印原文。

    Args:
        无；从命令行读取参数。
    Returns:
        通过为 0，失败或无可领取任务为 1；写入指定摘要文件。
    """
    parser = argument_parser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(validate(args), loop_factory=create_database_event_loop)
    except Exception as error:
        result = {"status": "failed", "error_code": stable_error_code(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
