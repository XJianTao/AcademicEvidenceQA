"""模块：M1-A 专用验证库准备

在既有 PostgreSQL 创建隔离验证库，保留所有用户库和 volume。

核心设计：
  - 只接受 aeqa_m1a_test_ 前缀；
  - 库已存在就复用，不删除或清空。

工程要点：
  - 管理员连接来自私有环境；不输出连接字符串或口令。
"""

import argparse
import re

import psycopg
from psycopg import sql

from academic_evidence_qa.config import load_settings


def prepare_database(database_name: str) -> None:
    """准备验证库

    创建不存在的专用库，拒绝非验证库名。

    Args:
        database_name: 小写英文/数字/下划线，aeqa_m1a_test_ 前缀，最多 63 字符。
    Returns:
        None；输出非敏感库名和 created/existing，不删除任何表。
    """
    if not re.fullmatch(r"aeqa_m1a_test_[a-z0-9_]{1,49}", database_name):
        raise ValueError("仅允许 aeqa_m1a_test_ 前缀的专用库")
    url = load_settings().database_url
    with psycopg.connect(
        url.set(drivername="postgresql").render_as_string(hide_password=False), autocommit=True
    ) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
        ).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
        print(f"{database_name}: {'existing' if exists else 'created'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建 M1-A 专用验证库，不删除用户数据")
    parser.add_argument("database_name")
    prepare_database(parser.parse_args().database_name)
