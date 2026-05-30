"""模块名称：本地管理 API 生命周期。

用途：FastAPI 工厂和原生 loopback 服务入口。
核心设计：单进程 API 持有模型 owner，解析与构建仍在独立持久 Worker。
工程要点：统一脱敏错误；loopback 内部推理有界，测试可显式关闭模型。
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from academic_evidence_qa.api.routes import router
from academic_evidence_qa.api.routes import internal_router, retrieval_router
from academic_evidence_qa.bootstrap import initialize_retrieval, close_retrieval
from academic_evidence_qa.infrastructure.inference_executor import RuntimeErrorCode
from academic_evidence_qa.config import load_settings
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.documents.parser import ParserError
from academic_evidence_qa.infrastructure.process_guard import ResourceGuardError
from academic_evidence_qa.documents.management import ManagementService
from academic_evidence_qa.infrastructure.database.session import (
    Database,
    create_database_event_loop,
)
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore


def create_app(settings=None, *, database=None, store=None, with_models=None):
    """名称：API 工厂。用途：注入配置与组件，不连接远端或加载模型。

    Args:
        settings: 显式配置；缺省读取本地 .env。
        database, store: 可注入正式组件；注入数据库由调用方关闭。
    Returns:
        带统一错误契约的 FastAPI 应用。
    """
    settings = settings or load_settings()
    owned = database is None
    with_models = owned if with_models is None else with_models
    database = database or Database(settings.database_url)
    store = store or LocalFileStore(settings.data_root)

    @asynccontextmanager
    async def lifespan(app):
        try:
            if with_models:
                await initialize_retrieval(app)
            yield
        finally:
            if with_models and getattr(app.state, "runtime", None) is not None:
                await close_retrieval(app)
            if owned:
                await database.close()

    app = FastAPI(title="AcademicEvidenceQA 本地管理", lifespan=lifespan)
    app.state.settings, app.state.database, app.state.store = settings, database, store
    app.state.management = ManagementService(database, store, settings)
    app.state.retrieval = None

    @app.exception_handler(RuntimeErrorCode)
    async def runtime_error(request, error):
        return JSONResponse(
            {"error_code": str(error)}, status_code=429 if str(error) == "executor_busy" else 503
        )

    @app.exception_handler(DocumentError)
    async def document_error(request, error):
        return JSONResponse({"error_code": error.error_code}, status_code=error.status_code)

    @app.exception_handler(ParserError)
    async def parser_error(request, error):
        # 只公开固定本地解析错误，不回显不可信 PDF/第三方内容。
        unavailable = str(error) in ("resource_guard_unavailable", "invalid_resource_limit")
        return JSONResponse(
            {
                "error_code": "resource_guard_unavailable"
                if unavailable
                else "pdf_preflight_rejected"
            },
            status_code=503 if unavailable else 422,
        )

    @app.exception_handler(ResourceGuardError)
    async def resource_error(request, error):
        return JSONResponse({"error_code": "resource_guard_unavailable"}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"error_code": "request_invalid"}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse({"error_code": "service_unavailable"}, status_code=503)

    app.include_router(router)
    app.include_router(internal_router)
    app.include_router(retrieval_router)
    return app


def main():
    """名称：原生 API 入口。用途：本地单用户 loopback，Windows 使用兼容事件循环。

    Args:
        无；读取 CLI 数据库/存储/端口参数。
    Returns:
        运行到用户停止，不执行迁移或清理历史资源。
    """
    parser = argparse.ArgumentParser(description="M1-C3 本地单用户管理 API")
    parser.add_argument("--database-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = load_settings()
    settings = replace(
        settings,
        database_url=settings.database_url.set(database=args.database_name),
        data_root=args.data_root,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=args.port,
            log_level="critical",
            access_log=False,
        )
    )
    asyncio.run(server.serve(), loop_factory=create_database_event_loop)


if __name__ == "__main__":
    main()
