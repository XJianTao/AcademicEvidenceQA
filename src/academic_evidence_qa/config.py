"""模块：数据底座配置

读取数据库连接和 FileStore 根目录，不加载模型或远端密钥。

核心设计：
  - 调用方显式构造配置；
  - 环境读取仅在 load_settings 中发生。

工程要点：
  - 不输出连接字符串；
  - DATABASE_URL 必须为 PostgreSQL psycopg 驱动。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url


@dataclass(frozen=True, repr=False)
class Settings:
    """运行配置

    保存连接与存储路径并避免 repr 暴露口令。

    Args:
        database_url: psycopg PostgreSQL URL。
        data_root: 服务自有本地数据根目录。
    Returns:
        供迁移和数据服务组装使用的配置。
    """

    database_url: URL
    data_root: Path
    pdf_max_upload_bytes: int = 100 * 1024 * 1024
    pdf_max_pages: int = 300
    pdf_preflight_timeout_seconds: int = 30
    pdf_preflight_memory_mib: int = 512
    local_parse_timeout_seconds: int = 900
    local_parse_memory_mib: int = 8192
    mineru_job_timeout_seconds: int = 1800
    mineru_max_retries: int = 2
    remote_connect_timeout_seconds: int = 10
    remote_read_timeout_seconds: int = 60
    result_max_download_bytes: int = 512 * 1024 * 1024
    result_max_extracted_bytes: int = 2 * 1024 * 1024 * 1024
    result_max_files: int = 10_000
    result_max_json_bytes: int = 128 * 1024 * 1024
    result_extract_timeout_seconds: int = 120
    result_extract_memory_mib: int = 1024
    normalization_timeout_seconds: int = 120
    normalization_memory_mib: int = 2048
    local_parser_concurrency: int = 1
    parser_queue_capacity: int = 2
    model_api_url: str = "http://127.0.0.1:8000"
    qdrant_url: str = "http://127.0.0.1:56333"
    inference_queue_capacity: int = 4


def load_settings() -> Settings:
    """读取本地配置

    优先 DATABASE_URL，否则使用 Compose 同名变量。

    Args:
        无；从当前进程和项目私有 .env 读取，现有环境优先。
    Returns:
        经驱动校验的配置；不连接数据库、不读取模型。
    Raises:
        ValueError: 数据库驱动不属于本项目支持的 PostgreSQL psycopg。
    """
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    url = (
        make_url(database_url)
        if database_url
        else URL.create(
            "postgresql+psycopg",
            username=os.getenv("POSTGRES_USER", "academic"),
            password=os.getenv("POSTGRES_PASSWORD", "local_only_change_me"),
            host="127.0.0.1",
            port=int(os.getenv("POSTGRES_PORT", "55432")),
            database=os.getenv("POSTGRES_DB", "academic_evidence"),
        )
    )
    if url.drivername != "postgresql+psycopg":
        raise ValueError("DATABASE_URL 必须使用 postgresql+psycopg")

    def positive_int(name: str, default: int) -> int:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError(f"{name} 必须为正整数")
        return value

    return Settings(
        url,
        Path(os.getenv("DATA_ROOT", "data")),
        pdf_max_upload_bytes=positive_int("PDF_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
        pdf_max_pages=positive_int("PDF_MAX_PAGES", 300),
        pdf_preflight_timeout_seconds=positive_int("PDF_PREFLIGHT_TIMEOUT_SECONDS", 30),
        pdf_preflight_memory_mib=positive_int("PDF_PREFLIGHT_MEMORY_MIB", 512),
        local_parse_timeout_seconds=positive_int("LOCAL_PARSE_TIMEOUT_SECONDS", 900),
        local_parse_memory_mib=positive_int("LOCAL_PARSE_MEMORY_MIB", 8192),
        mineru_job_timeout_seconds=positive_int("MINERU_JOB_TIMEOUT_SECONDS", 1800),
        mineru_max_retries=positive_int("MINERU_MAX_RETRIES", 2),
        remote_connect_timeout_seconds=positive_int("REMOTE_CONNECT_TIMEOUT_SECONDS", 10),
        remote_read_timeout_seconds=positive_int("REMOTE_READ_TIMEOUT_SECONDS", 60),
        result_max_download_bytes=positive_int("RESULT_MAX_DOWNLOAD_BYTES", 512 * 1024 * 1024),
        result_max_extracted_bytes=positive_int(
            "RESULT_MAX_EXTRACTED_BYTES", 2 * 1024 * 1024 * 1024
        ),
        result_max_files=positive_int("RESULT_MAX_FILES", 10_000),
        result_max_json_bytes=positive_int("RESULT_MAX_JSON_BYTES", 128 * 1024 * 1024),
        result_extract_timeout_seconds=positive_int("RESULT_EXTRACT_TIMEOUT_SECONDS", 120),
        result_extract_memory_mib=positive_int("RESULT_EXTRACT_MEMORY_MIB", 1024),
        normalization_timeout_seconds=positive_int("NORMALIZATION_TIMEOUT_SECONDS", 120),
        normalization_memory_mib=positive_int("NORMALIZATION_MEMORY_MIB", 2048),
        local_parser_concurrency=positive_int("LOCAL_PARSER_CONCURRENCY", 1),
        parser_queue_capacity=positive_int("PARSER_QUEUE_CAPACITY", 2),
        model_api_url=os.getenv("MODEL_API_URL", "http://127.0.0.1:8000"),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:56333"),
        inference_queue_capacity=positive_int("INFERENCE_QUEUE_CAPACITY", 4),
    )
