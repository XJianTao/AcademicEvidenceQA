"""模块名称：API/Worker 轻量依赖组装。

用途：集中固定解析配置身份与资源生命周期。
核心设计：构造数据库和 FileStore 不加载模型、不调用远端。
工程要点：凭据只在 Worker 创建客户端时读取，配置哈希不含 token。
"""

import hashlib
import json
import os

from academic_evidence_qa.documents.parser import ADAPTER_VERSION
from academic_evidence_qa.infrastructure.mineru_client import MineruClientConfig


async def initialize_retrieval(app):
    """名称：组装模型与检索。用途：API 生命周期内唯一 owner，冷启动不阻塞事件循环。

    Args:
        app: 已组装数据库、存储及配置的应用。
    Returns:
        None；资源写入 state，由对应关闭函数释放。
    """
    from academic_evidence_qa.infrastructure.inference_executor import BoundedExecutor
    from academic_evidence_qa.infrastructure.model_runtime import (
        ModelRuntime,
        DEFAULT_BGE_PATH,
        DEFAULT_RERANKER_PATH,
    )
    from academic_evidence_qa.infrastructure.qdrant_store import QdrantStore
    from academic_evidence_qa.retrieval.pipeline import RetrievalPipeline
    from pathlib import Path
    import time

    state = app.state
    state.inference = BoundedExecutor(state.settings.inference_queue_capacity)
    state.source_io = BoundedExecutor(4)
    started = time.monotonic()
    try:
        state.runtime, _ = await state.inference.run(
            lambda: ModelRuntime(
                Path(os.getenv("BGE_M3_MODEL_PATH", str(DEFAULT_BGE_PATH))),
                Path(os.getenv("RERANKER_MODEL_PATH", str(DEFAULT_RERANKER_PATH))),
                with_reranker=True,
            ),
            total_seconds=300,
        )
        state.model_load_seconds = time.monotonic() - started
        state.qdrant = QdrantStore(state.settings.qdrant_url)
        state.retrieval = RetrievalPipeline(
            state.database,
            state.store,
            state.qdrant,
            state.runtime,
            state.inference,
            state.source_io,
            max_json_bytes=state.settings.result_max_json_bytes,
        )
    except BaseException:
        await state.inference.close()
        await state.source_io.close()
        raise


async def close_retrieval(app):
    """先等所有同步推理与来源读结束，再关闭连接及释放本生命周期权重。"""
    await app.state.inference.close()
    await app.state.source_io.close()
    app.state.qdrant.close()
    app.state.retrieval = None
    app.state.runtime = None


def parser_identity(settings):
    """名称：解析配置身份。用途：API、Worker 和缓存使用相同不可变配置。

    Args:
        settings: 项目实际资源限制。
    Returns:
        不含凭据的配置内容及 SHA-256 标识。
    """
    identity = {
        "backend": "mineru_remote",
        "model": "vlm",
        "endpoint_sha256": hashlib.sha256(
            os.getenv("MINERU_BASE_URL", "https://mineru.net").encode()
        ).hexdigest(),
        "resources": {
            key: value for key, value in vars(settings).items() if isinstance(value, int)
        },
        "adapter_version": ADAPTER_VERSION,
    }
    return identity, hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def mineru_config(settings, *, recovering=False):
    """组装 Worker 客户端；恢复只读请求不重置原失败重试预算。"""
    return MineruClientConfig(
        base_url=os.getenv("MINERU_BASE_URL", "https://mineru.net"),
        api_token=os.getenv("MINERU_API_TOKEN", ""),
        connect_timeout_seconds=settings.remote_connect_timeout_seconds,
        read_timeout_seconds=settings.remote_read_timeout_seconds,
        total_timeout_seconds=settings.mineru_job_timeout_seconds,
        max_retries=0 if recovering else settings.mineru_max_retries,
        max_download_bytes=settings.result_max_download_bytes,
    )
