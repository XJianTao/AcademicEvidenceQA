"""模块名称：M2a-1 模型共享真实验证。

用途：在专用数据库复用完整既有快照，验证正式 Worker 编码、争用与失败。
核心设计：先核验旧来源，再通过 DocumentService 登记；复用正式 Worker 和独立回读。
工程要点：不上传、不加载权重、不删除数据库或集合；只保存身份与计时。
"""

import argparse
import asyncio
from pathlib import Path
import json
import time
from uuid import UUID, uuid4

from alembic import command
from alembic.config import Config
import httpx
from sqlalchemy import create_engine, select

from prepare_m1a_database import prepare_database
from validate_m1c2_worker import validate as validate_worker
from academic_evidence_qa.api.schemas import RevisionRequest
from academic_evidence_qa.config import load_settings
from academic_evidence_qa.contracts.documents import MemberCommand, ParserSelection
from academic_evidence_qa.contracts.jobs import ParseTaskRequest
from academic_evidence_qa.documents.management import ManagementService
from academic_evidence_qa.documents.service import DocumentService
from academic_evidence_qa.documents.source_reader import verify_snapshot
from academic_evidence_qa.infrastructure.database.models import (
    BuildIntent,
    DocumentVersion,
    Library,
    ParseSnapshotRecord,
    StoredFile,
)
from academic_evidence_qa.infrastructure.database.repositories import request_index_cancel
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore
from academic_evidence_qa.worker.main import argument_parser, run as run_worker


async def copy_verified_source(args, database, store):
    """名称：复制核验来源。用途：保留旧快照完整正文与结构，不重新解析。

    Args:
        args: 明确旧数据库、快照与存储根目录。
        database, store: 新专用库及新存储。
    Returns:
        新库身份、旧/新快照身份及原件哈希。
    """
    original = Database(load_settings().database_url.set(database=args.source_database))
    original_store = LocalFileStore(args.source_root)
    try:
        async with original.sessions() as session:
            snapshot = await session.get(ParseSnapshotRecord, args.snapshot_id)
            if snapshot is None:
                raise ValueError("source_snapshot_missing")
            version = await session.get(DocumentVersion, snapshot.document_version_id)
            document = await verify_snapshot(
                session,
                version,
                snapshot,
                original_store,
                max_json_bytes=load_settings().result_max_json_bytes,
            )
            file = await session.get(StoredFile, version.file_id)
        service = DocumentService(store)
        # 文件复制在短身份事务前完成；只复制已核验的既有产物。
        with (original_store.data_root / file.object_key).open("rb") as stream:
            stored = store.write(stream)
        artifacts = []
        for kind, key in (
            ("raw", snapshot.raw_artifact_key),
            ("normalized", snapshot.normalized_artifact_key),
        ):
            with (original_store.data_root / key).open("rb") as stream:
                artifacts.append(store.write_artifact(kind, stream, max_bytes=536870912))
        async with database.transaction() as session:
            library_id = await service.create_library(session)
            registered = await service.register_verified_document(session, stored)
            task = await service.create_parse_task(
                session,
                ParseTaskRequest(
                    registered.document_version_id,
                    snapshot.parser_backend,
                    snapshot.parser_config_id,
                    False,
                ),
            )
        token = uuid4()
        async with database.transaction() as session:
            await service.begin_parse_task(
                session, task.parse_task_id, token, attempt_id=task.attempt_id
            )
            copied_snapshot = await service.save_parse_snapshot(
                session,
                parse_task_id=task.parse_task_id,
                executor_token=token,
                document=document,
                adapter_version=snapshot.adapter_version,
                remote_version=snapshot.remote_version,
                actual_resource_config=snapshot.actual_resource_config,
                raw_artifact=artifacts[0],
                normalized_artifact=artifacts[1],
            )
        async with database.transaction() as session:
            await service.set_member(
                session,
                MemberCommand(library_id, registered.paper_id, 0),
                registered.document_version_id,
                ParserSelection(snapshot.parser_backend, snapshot.parser_config_id),
            )
        return {
            "library_id": str(library_id),
            "source_snapshot_id": str(args.snapshot_id),
            "copied_snapshot_id": str(copied_snapshot),
            "file_sha256": version.file_sha256,
        }
    finally:
        await original.close()


async def wait_build(database, library_id, pending):
    """在有限期限内等正式任务进入 build；不伪造或修改阶段。"""
    async with asyncio.timeout(120):
        while not pending.done():
            async with database.sessions() as session:
                job = await session.scalar(
                    select(BuildIntent).where(
                        BuildIntent.library_id == library_id, BuildIntent.state == "running"
                    )
                )
                if job and job.stage == "build":
                    return job.build_intent_id
            await asyncio.sleep(0.1)
    raise ValueError("worker_build_not_observed")


async def validate(args, result):
    """名称：真实共享验证。用途：复用正式入口与短事务观察，不复制构建算法。

    Args:
        args: 专用新库、既有来源与正在运行的 API。
        result: 增量保存证据，失败仍保留已完成阶段。
    Returns:
        None；所有关键断言完成才标记 passed。
    """
    database = Database(load_settings().database_url.set(database=args.database_name))
    store = LocalFileStore(args.data_root)
    try:
        result["source"] = await copy_verified_source(args, database, store)
        library_id = UUID(result["source"]["library_id"])
        worker = argument_parser().parse_args(
            [
                "--database-name",
                args.database_name,
                "--data-root",
                str(args.data_root),
                "--library-id",
                str(library_id),
                "--model-api-url",
                args.base_url,
                "--heartbeat-seconds",
                "1",
                "--lease-seconds",
                "60",
            ]
        )
        started = time.monotonic()
        pending = asyncio.create_task(validate_worker(worker))
        await wait_build(database, library_id, pending)
        async with httpx.AsyncClient(
            base_url=args.base_url, trust_env=False, timeout=380
        ) as client:

            async def query():
                response = await client.post(
                    "/v1/retrieval/candidates",
                    json={
                        "library_id": args.query_library,
                        "query": "graph sentiment classification",
                        "mode": "dense",
                        "limit": 3,
                    },
                )
                response.raise_for_status()
                body = response.json()
                if body["statuses"] or not body["candidates"]:
                    raise ValueError("contention_query_failed")
                return {
                    "scope": body["scope"],
                    "timings": body["timings"],
                    "chunk_ids": [c["chunk_id"] for c in body["candidates"]],
                }

            result["contention_queries"] = await asyncio.gather(query(), query(), query())
            result["worker"] = await pending
            if result["worker"].get("independent_read") != "passed":
                raise ValueError("shared_worker_failed")
            result["contention_seconds"] = time.monotonic() - started
            management = ManagementService(database, store, load_settings())
            revision = RevisionRequest(expected_membership_revision=1)
            _, cancel_job = await management.reindex(uuid4().hex, library_id, revision)
            pending = asyncio.create_task(run_worker(worker, once=True))
            await wait_build(database, library_id, pending)
            started = time.monotonic()
            await request_index_cancel(database, UUID(str(cancel_job["job_id"])))
            result["cancel"] = await asyncio.wait_for(pending, 150)
            result["cancel_seconds"] = time.monotonic() - started
            if result["cancel"]["status"] != "cancelled":
                raise ValueError("worker_cancel_failed")
            # 同步推理不可强杀；用之后成功的内部真实编码确认队列已回收。
            metadata = (await client.get("/internal/models")).json()
            response = await client.post(
                "/internal/encode",
                json={
                    "configuration_sha256": metadata["identity"]["configuration_sha256"],
                    "texts": ["graph sentiment recovery"],
                },
            )
            response.raise_for_status()
            encoded = response.json()
            result["post_cancel_encode"] = {
                "status": "passed",
                "timings": encoded["timings"],
                "dimension": len(encoded["vectors"][0]),
            }
        _, unavailable_job = await management.reindex(uuid4().hex, library_id, revision)
        worker.model_api_url = "http://127.0.0.1:58019"
        started = time.monotonic()
        result["unavailable"] = await asyncio.wait_for(run_worker(worker, once=True), 150)
        result["unavailable_seconds"] = time.monotonic() - started
        if result["unavailable"].get("error_code") != "embedding_api_unavailable":
            raise ValueError("unavailable_error_mismatch")
        async with database.sessions() as session:
            library = await session.get(Library, library_id)
            jobs = [
                await session.get(BuildIntent, UUID(str(item["job_id"])))
                for item in (cancel_job, unavailable_job)
            ]
            if (
                str(library.active_generation_id) != result["worker"]["generation_id"]
                or library.building_generation_id is not None
                or [job.state for job in jobs] != ["cancelled", "failed"]
            ):
                raise ValueError("failure_publication_changed")
            result["independent_failure_read"] = "passed"
        result["status"] = "passed"
    finally:
        await database.close()


def main():
    """创建保留型专用库、迁移后执行；错误只记录类型，不输出连接或全文。"""
    parser = argparse.ArgumentParser(description="M2a-1 真实 Worker 共享验收")
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-database", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", type=UUID, required=True)
    parser.add_argument("--query-library", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "failed", "database_name": args.database_name}
    try:
        prepare_database(args.database_name)
        engine = create_engine(load_settings().database_url.set(database=args.database_name))
        try:
            config = Config("alembic.ini")
            with engine.connect() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
        finally:
            engine.dispose()
        asyncio.run(validate(args, result), loop_factory=create_database_event_loop)
    except Exception as error:
        result["error_type"] = type(error).__name__
        # 稳定项目错误码可供恢复；不回显远端消息或数据库错误。
        if isinstance(error, ValueError):
            result["error_code"] = str(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "error_type", "error_code") if k in result}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
