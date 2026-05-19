"""模块名称：M1-B MinerU 全链路验证入口。

用途概要：操作者显式授权一篇 PDF，经正式有界上传、受限解析和短事务保存快照。
核心设计：复用 DocumentService/ParseTask，不启动 Worker；一个 attempt 共享远端总期限。
工程要点：仅操作专用验证库和 data/ignored；输出稳定错误码、哈希和计数，不输出正文或凭据。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from psycopg.rows import dict_row
from sqlalchemy import create_engine

from academic_evidence_qa.config import Settings, load_settings
from academic_evidence_qa.contracts.documents import DocumentError, ParserSelection
from academic_evidence_qa.contracts.jobs import ParseTaskRequest
from academic_evidence_qa.documents.parser import (
    ADAPTER_VERSION,
    ParserError,
    normalize_mineru_result,
)
from academic_evidence_qa.documents.service import DocumentService
from academic_evidence_qa.documents.upload import preflight_and_store_pdf, receive_pdf_stream
from academic_evidence_qa.infrastructure.database.models import ParseTask
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore
from academic_evidence_qa.infrastructure.mineru_client import (
    MineruClient,
    MineruClientConfig,
    MineruClientError,
)
from academic_evidence_qa.infrastructure.process_guard import (
    ResourceGuardError,
    WindowsProcessGuard,
)
from academic_evidence_qa.infrastructure.result_archive import ArchiveError, extract_result_archive

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "data" / "ignored" / "m1b-validation"


def build_argument_parser() -> argparse.ArgumentParser:
    """名称：构造验证 CLI。用途概要：将本地路径限制在人工验证入口。

    Args:
        无；只构造参数，不读取配置或访问网络。
    Returns:
        明确说明外发和脱敏约束的 argparse 解析器。
    """
    parser = argparse.ArgumentParser(
        description="M1-B 验证：将明确获准的 PDF 上传到 MinerU；不会输出凭据、签名 URL 或正文。"
    )
    parser.add_argument("pdf", type=Path, help="操作者指定的本地 PDF；不是未来 HTTP API 路径契约")
    parser.add_argument("--database-name", required=True, help="仅允许 aeqa_m1a_test_ 前缀专用库")
    parser.add_argument(
        "--allow-remote-processing", action="store_true", help="确认此具体 PDF 可外发到 MinerU"
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """在配置读取、创建库和上传之前拒绝缺授权或非验证库。"""
    if not args.allow_remote_processing:
        raise DocumentError("remote_processing_not_authorized", 422)
    if not re.fullmatch(r"aeqa_m1a_test_[a-z0-9_]{1,49}", args.database_name):
        raise DocumentError("validation_database_required", 422)
    if not args.pdf.is_file():
        raise DocumentError("source_pdf_missing", 422)


def prepare_validation_database(settings: Settings, database_name: str):
    """名称：准备隔离验证库。用途概要：复用现有库创建器并仅迁移指定验证库。

    Args:
        settings: 私有配置；不得打印连接字符串。
        database_name: 已校验的专用前缀库名，不清空已有库。
    Returns:
        指向已迁移专用库的 SQLAlchemy URL。
    """
    spec = importlib.util.spec_from_file_location(
        "m1b_database_preparation", ROOT / "scripts" / "prepare_m1a_database.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with redirect_stdout(io.StringIO()):
        module.prepare_database(database_name)
    url = settings.database_url.set(database=database_name)
    engine = create_engine(url, hide_parameters=True)
    try:
        config = Config(str(ROOT / "alembic.ini"))
        with engine.connect() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    finally:
        engine.dispose()
    return url


@asynccontextmanager
async def record_stage(summary: dict, name: str):
    """记录阶段耗时和结果；异常继续向外传播，不写入异常原文。"""
    started = time.monotonic()
    entry = summary["stages"][name] = {"status": "executing"}
    try:
        yield
        entry["status"] = "passed"
    except BaseException:
        entry["status"] = "failed"
        raise
    finally:
        entry["seconds"] = round(time.monotonic() - started, 3)


async def file_chunks(path: Path):
    """按固定 1 MiB 内容块读取，交给正式接收层累计限额和哈希。"""
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
            yield chunk


class CancellableValidationGuard(WindowsProcessGuard):
    """名称：验证进程防护。用途概要：把同一取消事件传给正式 Windows Job Object。

    Args:
        cancellation: 由数据库取消观察器或本地取消设置的线程事件。
    Returns:
        仅补充取消传播的防护；不改变内存/期限/进程树回收行为。
    """

    def __init__(self, cancellation: threading.Event):
        super().__init__()
        self.cancellation = cancellation

    def run(self, request, **kwargs):
        return super().run(request, cancellation=self.cancellation, **kwargs)


async def observe_cancellation(database, job_id, remote_cancel, local_cancel):
    """只观察本任务取消位；不是任务领取、租约或调度器，数据库失联时停止本地工作。"""
    try:
        while True:
            async with database.transaction() as session:
                task = await session.get(ParseTask, job_id)
                if task is None or task.cancel_requested:
                    remote_cancel.set()
                    local_cancel.set()
                    return
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        raise
    except Exception:
        remote_cancel.set()
        local_cancel.set()


def verify_snapshot(url, store, snapshot_id: UUID, document, stored, raw, normalized) -> dict:
    """名称：独立连接核验快照。用途概要：确认数据库、原件和两份不可变产物一致。

    Args:
        url, store: 同一专用库与 FileStore，连接独立于写入会话。
        snapshot_id: 已提交快照身份。
        document, stored, raw, normalized: 正式路径产生的结构和文件描述。
    Returns:
        哈希一致性、章节/块数量及不含正文的少量位置；不返回连接凭据。
    Raises:
        DocumentError: 任何数据库身份、数量或文件哈希不一致。
    """
    store.verify(stored)
    store.verify_artifact(raw)
    store.verify_artifact(normalized)
    with psycopg.connect(
        url.set(drivername="postgresql").render_as_string(hide_password=False),
        row_factory=dict_row,
    ) as connection:
        snapshot = connection.execute(
            "SELECT * FROM parse_snapshots WHERE parse_snapshot_id = %s", (snapshot_id,)
        ).fetchone()
        blocks = connection.execute(
            "SELECT block_id, block_type, reading_order, text, locations FROM parse_blocks "
            "WHERE parse_snapshot_id = %s ORDER BY reading_order",
            (snapshot_id,),
        ).fetchall()
        sections = connection.execute(
            "SELECT count(*) AS n FROM parse_sections WHERE parse_snapshot_id = %s", (snapshot_id,)
        ).fetchone()["n"]
    if (
        snapshot is None
        or snapshot["input_sha256"] != stored.file_sha256
        or snapshot["raw_artifact_sha256"] != raw.sha256
        or snapshot["normalized_artifact_sha256"] != normalized.sha256
        or snapshot["raw_artifact_key"] != raw.object_key
        or snapshot["normalized_artifact_key"] != normalized.object_key
        or sections != len(document.sections)
        or len(blocks) != len(document.blocks)
        or any(
            row["text"] != block.text for row, block in zip(blocks, document.blocks, strict=True)
        )
    ):
        raise DocumentError("snapshot_verification_failed", 409)
    return {"verified": True, "section_count": sections, "block_count": len(blocks)}


async def run_validation(args, summary: dict) -> int:
    """名称：执行一个授权验证。用途概要：贯通正式 M1-B 模块并保留可诊断失败。

    Args:
        args: 已校验 CLI 参数；授权不由全局环境默认值代替。
        summary: 仅允许安全字段的输出；不保存 HTTP 响应或论文正文。
    Returns:
        0 成功；2 输入/预检；3 远端；4 本地/数据库；5 取消。
    工程要点：网络和受限解析在事务外；崩溃后不自动重提交或重建期限。
    """
    database = client = watcher = attempt = job_id = attempt_id = attempt_dir = None
    local_cancel, remote_cancel = threading.Event(), asyncio.Event()
    executor_token = uuid4()
    stage = "configuration"
    try:
        settings = load_settings()
        if not all(
            os.getenv(key)
            for key in ("MINERU_API_TOKEN", "MINERU_BASE_URL", "MINERU_MODEL_VERSION")
        ):
            raise DocumentError("mineru_configuration_missing", 422)
        config = MineruClientConfig(
            base_url=os.environ["MINERU_BASE_URL"],
            api_token=os.environ["MINERU_API_TOKEN"],
            model_version=os.environ["MINERU_MODEL_VERSION"],
            connect_timeout_seconds=settings.remote_connect_timeout_seconds,
            read_timeout_seconds=settings.remote_read_timeout_seconds,
            total_timeout_seconds=settings.mineru_job_timeout_seconds,
            max_retries=settings.mineru_max_retries,
            max_download_bytes=settings.result_max_download_bytes,
        )
        store = LocalFileStore(VALIDATION_ROOT / args.database_name / "store")
        service = DocumentService(store)
        stage = "bounded_receive"
        async with record_stage(summary, stage):
            received = await receive_pdf_stream(
                file_chunks(args.pdf),
                file_store=store,
                max_upload_bytes=settings.pdf_max_upload_bytes,
            )
            summary.update(sha256=received.sha256, byte_count=received.size_bytes)
        stage = "preflight"
        async with record_stage(summary, stage):
            stored, preflight = await preflight_and_store_pdf(
                received,
                file_store=store,
                max_pages=min(settings.pdf_max_pages, 200),
                timeout_seconds=settings.pdf_preflight_timeout_seconds,
                memory_mib=settings.pdf_preflight_memory_mib,
            )
            summary["page_count"] = preflight.page_count
        stage = "database_prepare"
        async with record_stage(summary, stage):
            url = await asyncio.to_thread(prepare_validation_database, settings, args.database_name)
            database = Database(url)
        resource_config = {
            key: value for key, value in vars(settings).items() if isinstance(value, int)
        }
        config_identity = {
            "backend": "mineru_remote",
            "model": "vlm",
            "endpoint_sha256": hashlib.sha256(config.base_url.encode()).hexdigest(),
            "resources": resource_config,
            "adapter_version": ADAPTER_VERSION,
        }
        parser_config_id = hashlib.sha256(
            json.dumps(config_identity, sort_keys=True).encode()
        ).hexdigest()
        stage = "register"
        async with record_stage(summary, stage):
            async with database.transaction() as session:
                library_id = await service.create_library(session)
                registered, _ = await service.register_verified_in_library(
                    session,
                    stored,
                    library_id=library_id,
                    expected_membership_revision=0,
                    parser=ParserSelection("mineru_remote", parser_config_id),
                )
                task = await service.create_parse_task(
                    session,
                    ParseTaskRequest(
                        registered.document_version_id, "mineru_remote", parser_config_id, True
                    ),
                )
                job_id = task.parse_task_id
                attempt_id = task.attempt_id
                started = await service.begin_parse_task(
                    session, job_id, executor_token, attempt_id=attempt_id
                )
                if started.attempt_id != attempt_id:
                    raise DocumentError("parse_task_attempt_conflict", 409)
            summary["attempt_id"] = str(attempt_id)
            attempt_dir = VALIDATION_ROOT / args.database_name / "attempts" / attempt_id.hex
            attempt_dir.mkdir(parents=True, exist_ok=False)
            summary.update(
                library_id=str(library_id),
                file_id=str(stored.file_id),
                document_version_id=str(registered.document_version_id),
                job_id=str(job_id),
            )
        client = MineruClient(config)
        attempt = client.begin_attempt(attempt_id=str(attempt_id))

        async def checkpoint(remote_task_id=None):
            async with database.transaction() as session:
                state = await service.checkpoint_parse_task(
                    session,
                    parse_task_id=job_id,
                    executor_token=executor_token,
                    stage=stage,
                    remote_task_id=remote_task_id,
                    checkpoint=attempt.checkpoint(),
                )
            if state.state == "cancelled":
                remote_cancel.set()
                local_cancel.set()
                raise MineruClientError("parse_task_cancelled", attempt=attempt)

        watcher = asyncio.create_task(
            observe_cancellation(database, job_id, remote_cancel, local_cancel)
        )
        source = store.data_root / stored.object_key
        stage = "submit"
        async with record_stage(summary, stage):
            await checkpoint()
            submission = await client.submit(
                source,
                attempt=attempt,
                checkpoint_remote_task_id=checkpoint,
                cancellation=remote_cancel,
            )
        for stage in ("upload", "poll", "download"):
            async with record_stage(summary, stage):
                await checkpoint()
                if stage == "upload":
                    await client.upload(
                        source, submission, attempt=attempt, cancellation=remote_cancel
                    )
                elif stage == "poll":
                    completed = await client.wait_for_completion(
                        submission.remote_task_id, attempt=attempt, cancellation=remote_cancel
                    )
                    summary.update(
                        poll_count=completed.poll_count, poll_retries=completed.retry_count
                    )
                else:
                    await client.download(
                        completed,
                        attempt_dir / "result.zip",
                        attempt=attempt,
                        cancellation=remote_cancel,
                    )
        guard = CancellableValidationGuard(local_cancel)
        stage = "extract"
        async with record_stage(summary, stage):
            await checkpoint()
            extracted = await asyncio.to_thread(
                extract_result_archive,
                attempt_dir / "result.zip",
                attempt_dir / "extracted",
                max_files=settings.result_max_files,
                max_extracted_bytes=settings.result_max_extracted_bytes,
                max_json_bytes=settings.result_max_json_bytes,
                timeout_seconds=settings.result_extract_timeout_seconds,
                memory_mib=settings.result_extract_memory_mib,
                guard=guard,
            )
            summary["archive"] = vars(extracted)
        content_lists = list((attempt_dir / "extracted").rglob("*_content_list.json"))
        if len(content_lists) != 1:
            raise ParserError("mineru_content_list_ambiguous_or_missing")
        stage = "normalize"
        async with record_stage(summary, stage):
            await checkpoint()
            normalized_path = attempt_dir / "normalized.json"
            document = await asyncio.to_thread(
                normalize_mineru_result,
                content_lists[0],
                normalized_path,
                max_json_bytes=settings.result_max_json_bytes,
                timeout_seconds=settings.normalization_timeout_seconds,
                memory_mib=settings.normalization_memory_mib,
                guard=guard,
            )
            summary.update(
                quality_state=document.quality_state,
                quality_metrics=document.quality_metrics,
                warning_count=document.quality_metrics.get("warning_count", 0),
            )
            if document.quality_state == "rejected" or not document.blocks:
                raise ParserError("parser_quality_rejected")
        stage = "artifacts"
        async with record_stage(summary, stage):
            await checkpoint()
            with (attempt_dir / "result.zip").open("rb") as stream:
                raw = await asyncio.to_thread(
                    store.write_artifact,
                    "mineru_raw",
                    stream,
                    max_bytes=settings.result_max_download_bytes,
                )
            with normalized_path.open("rb") as stream:
                normalized = await asyncio.to_thread(
                    store.write_artifact,
                    "normalized",
                    stream,
                    max_bytes=settings.result_max_json_bytes,
                )
            summary.update(raw_artifact=vars(raw), normalized_artifact=vars(normalized))
        stage = "snapshot_commit"
        async with record_stage(summary, stage):
            await checkpoint()
            async with database.transaction() as session:
                snapshot_id = await service.save_parse_snapshot(
                    session,
                    parse_task_id=job_id,
                    executor_token=executor_token,
                    document=document,
                    adapter_version=ADAPTER_VERSION,
                    remote_version="unknown",
                    actual_resource_config=config_identity,
                    raw_artifact=raw,
                    normalized_artifact=normalized,
                )
            summary["parse_snapshot_id"] = str(snapshot_id)
        stage = "independent_verify"
        async with record_stage(summary, stage):
            summary["database_verification"] = await asyncio.to_thread(
                verify_snapshot, url, store, snapshot_id, document, stored, raw, normalized
            )
        summary["status"], summary["error_code"] = "passed", None
        return 0
    except BaseException as error:
        local_cancel.set()
        remote_cancel.set()
        known = isinstance(
            error, (DocumentError, ParserError, ArchiveError, MineruClientError, ResourceGuardError)
        )
        cancelled = isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) or (
            known
            and error.error_code
            in {"cancelled", "parse_task_cancelled", "process_cancelled", "job_cancelled"}
        )
        code = (
            "parse_task_cancelled"
            if cancelled
            else (error.error_code if known else "validation_failed")
        )
        summary.update(
            status="cancelled" if cancelled else "failed", error_code=code, failed_stage=stage
        )
        if database is not None and job_id is not None:
            try:
                async with database.transaction() as session:
                    task = await session.get(ParseTask, job_id)
                    if cancelled:
                        await service.request_parse_cancellation(session, job_id)
                    if task.state == "executing" and attempt is not None:
                        state = await service.checkpoint_parse_task(
                            session,
                            parse_task_id=job_id,
                            executor_token=executor_token,
                            stage=stage,
                            remote_task_id=attempt.remote_task_id,
                            checkpoint=attempt.checkpoint(),
                        )
                        if state.state != "cancelled":
                            await service.fail_parse_task(
                                session,
                                parse_task_id=job_id,
                                executor_token=executor_token,
                                error_code=code,
                            )
                    elif task.state == "executing":
                        await service.fail_parse_task(
                            session,
                            parse_task_id=job_id,
                            executor_token=executor_token,
                            error_code=code,
                        )
            except Exception:
                summary["failure_checkpoint"] = "not_saved"
        if cancelled:
            return 5
        if isinstance(error, MineruClientError):
            return 3
        # 同一个领域异常类型也可能表示数据库身份或快照一致性失败；
        # 按实际失败阶段分类，避免把已上传后的本地错误误报为用户输入错误。
        return 2 if stage in {"configuration", "bounded_receive", "preflight"} else 4
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if client is not None:
            await client.aclose()
        if database is not None:
            await database.close()
        if attempt is not None:
            summary["attempt_expires_at"] = attempt.expires_at.isoformat()
            summary["last_remote_stage"] = attempt.last_stage
            if attempt.remote_task_id:
                summary["remote_task_id_sha256"] = hashlib.sha256(
                    attempt.remote_task_id.encode()
                ).hexdigest()
        if attempt_dir is not None:
            (attempt_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
            )


def main(argv=None) -> int:
    """名称：CLI 主入口。用途概要：输出脱敏摘要和稳定退出码，不泄漏异常 traceback。

    Args:
        argv: 测试可注入参数，默认读取当前进程命令行。
    Returns:
        0 成功；2 输入/配置/预检；3 远端；4 本地/数据库；5 取消。
    """
    args = build_argument_parser().parse_args(argv)
    summary = {
        "input_file": args.pdf.name,
        "backend": "mineru_remote",
        "model_version": "vlm",
        "stages": {},
        "status": "not_started",
        "error_code": None,
    }
    try:
        validate_arguments(args)
        exit_code = asyncio.run(
            run_validation(args, summary), loop_factory=create_database_event_loop
        )
    except DocumentError as error:
        summary.update(status="failed", error_code=error.error_code)
        exit_code = 2
    except (KeyboardInterrupt, asyncio.CancelledError):
        summary.update(status="cancelled", error_code="parse_task_cancelled")
        exit_code = 5
    except Exception:
        summary.update(status="failed", error_code="validation_failed")
        exit_code = 4
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
