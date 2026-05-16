"""模块：M1-A 重启持久化实测。

在容器重启前保存实际业务表摘要，重启后独立进程读取并严格核对。

核心设计：
  - 按主键固定排序；全部六张业务表及每个原件的实际哈希均纳入验证。

工程要点：
  - 只接受专用验证库，不删除数据；
  - 报告不包含连接口令或论文正文。
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from sqlalchemy import create_engine, select, text

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.database.models import Base, StoredFile
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore, StoredObject

ROOT = Path(__file__).resolve().parents[1]


def verify_persistence(database_name: str, capture: bool) -> None:
    """核对持久化

    核对真实数据库内容和落盘原件。

    Args:
        database_name: 受限前缀的专用验收库。
        capture: True 首次保存摘要；False 与重启前摘要严格比较。
    Returns:
        None；通过时输出表计数、摘要和原件数，任何缺失或不一致抛异常。
    """
    if not re.fullmatch(r"aeqa_m1a_test_[a-z0-9_]{1,49}", database_name):
        raise ValueError("仅允许专用验证库")
    data_root = ROOT / "data/m1a_validation" / database_name
    baseline = data_root / "persistence_baseline.json"
    store = LocalFileStore(data_root)
    engine = create_engine(
        load_settings().database_url.set(database=database_name), hide_parameters=True
    )
    tables = {}
    file_count = 0
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            version = connection.execute(text("SHOW server_version")).scalar_one()
            assert (
                connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
                == "0001_m1a"
            )
            for table in Base.metadata.sorted_tables:
                rows = connection.execute(select(table).order_by(*table.primary_key.columns)).all()
                encoded = json.dumps(
                    [list(row) for row in rows],
                    default=str,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                tables[table.name] = {
                    "count": len(rows),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            for row in connection.execute(select(StoredFile.__table__)).mappings():
                store.verify(
                    StoredObject(
                        row["file_id"], row["file_sha256"], row["object_key"], row["size_bytes"]
                    )
                )
                file_count += 1
        assert all(value["count"] > 0 for value in tables.values()), "不接受空库持久化验收"
        summary = {
            "database": database_name,
            "server_version": version,
            "tables": tables,
            "verified_originals": file_count,
        }
        if capture:
            if baseline.exists():
                raise FileExistsError("已有摘要必须保留，不能覆盖重启前证据")
            baseline.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            expected = json.loads(baseline.read_text(encoding="utf-8"))
            assert summary == expected, "重启前后数据摘要/计数不一致"
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("CAPTURED" if capture else "PERSISTENCE PASSED")
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1-A 实际表内容与原件持久化验证")
    parser.add_argument("database_name")
    parser.add_argument("--capture", action="store_true")
    arguments = parser.parse_args()
    verify_persistence(arguments.database_name, arguments.capture)
