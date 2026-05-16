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
    return Settings(url, Path(os.getenv("DATA_ROOT", "data")))
