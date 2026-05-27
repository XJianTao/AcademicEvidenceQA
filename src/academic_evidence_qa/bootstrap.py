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
