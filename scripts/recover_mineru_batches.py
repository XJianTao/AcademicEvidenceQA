"""恢复既有 MinerU batch 并仅输出不含论文正文的验证汇总。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from academic_evidence_qa.mineru_adapter import normalize_mineru_block


def main() -> None:
    benchmark_dir = Path(r"E:\Project Files\VscodeProjects\rag_test")
    sys.path.insert(0, str(benchmark_dir))
    from mineru_remote_benchmark import RemoteClient, extract_zip

    run_dir = Path(os.environ["MINERU_RUN_DIR"])
    dotenv_path = benchmark_dir / ".env"
    load_dotenv(dotenv_path, override=False)
    token, base_url = os.getenv("MINERU_API_TOKEN"), os.getenv("MINERU_BASE_URL")
    if not token or not base_url:
        raise RuntimeError("MinerU 本地凭据或服务地址缺失")
    client = RemoteClient(base_url, token, request_timeout=60)
    try:
        for result_path in sorted(run_dir.rglob("result.json")):
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if not record.get("batch_id"):
                continue
            started = time.monotonic()
            deadline = started + 1800
            while True:
                item = client.result(record["batch_id"], deadline)
                state = item.get("state")
                if state == "done":
                    break
                if state == "failed":
                    raise RuntimeError("MinerU batch failed; server detail omitted")
                time.sleep(min(3, max(0, deadline - time.monotonic())))
            with tempfile.TemporaryDirectory(prefix="aeqa-mineru-") as scratch:
                scratch_path = Path(scratch)
                archive = scratch_path / "result.zip"
                client.download_zip(item["full_zip_url"], archive, deadline)
                archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
                extract_zip(archive, scratch_path / "extracted")
                json_files = list((scratch_path / "extracted").rglob("*.json"))
                content_files = [path for path in json_files if "content_list" in path.name]
                if not content_files:
                    raise RuntimeError("MinerU ZIP did not contain a content-list JSON")
                blocks = []
                for path in content_files:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        blocks.extend(block for block in payload if isinstance(block, dict))
                normalized = [normalize_mineru_block(block) for block in blocks]
                flags = Counter(flag for block in normalized for flag in block["quality_flags"])
                pages = [block["physical_page_start"] for block in normalized if block["physical_page_start"]]
                print(json.dumps({
                    "source_sha256": record["source_sha256"], "page_count": record["page_count"],
                    "remote_state": state, "polls_to_completion": record.get("poll_count", 0) + 1,
                    "initial_attempt_seconds": record.get("elapsed_seconds"),
                    "recovery_wait_seconds": round(time.monotonic() - started, 3),
                    "block_count": len(normalized), "page_min": min(pages) if pages else None,
                    "page_max": max(pages) if pages else None, "quality_flags": dict(flags),
                    "normalized_hash": hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    "result_zip_sha256": archive_hash,
                }, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
