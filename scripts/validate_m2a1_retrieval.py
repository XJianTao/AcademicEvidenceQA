"""模块名称：M2a-1 正式 HTTP 检索验证。

用途：比较逐论文与全库四模式，回读父 chunk 原块，不复制检索算法。
核心设计：只读已发布语料，保存身份、分数、区间、哈希与分阶段计时。
工程要点：不上传 PDF、不持久化论文全文；失败保留已完成记录。
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx


def check_original(client, library_id, scope, candidate):
    """名称：父来源抽查。用途：通过正式原文分页核验原块/行组，输出哈希。

    Args:
        client, library_id, scope, candidate: 正式接口及固定来源身份。
    Returns:
        原块哈希与位置；派生表格区间不冒充原块字符偏移。
    """
    identity = {k: candidate[k] for k in ("paper_id", "document_version_id", "parse_snapshot_id")}
    checked = []
    for source_range in candidate["source_ranges"]:
        order = source_range["block_reading_order"]
        offset, block = 0, None
        # offset 是分页索引，不能把 reading_order 无条件当成 offset。
        while block is None:
            response = client.post(
                "/v1/evidence/read",
                json={
                    **identity,
                    "library_id": library_id,
                    "generation_id": scope["generation_id"],
                    "offset": offset,
                    "limit": 100,
                    "max_characters": 32000,
                },
            )
            response.raise_for_status()
            source = response.json()
            block = next((b for b in source["items"] if b["reading_order"] == order), None)
            if block is None:
                next_offset = source["next_offset"]
                if (
                    next_offset is None
                    or next_offset + bool(source.get("next_char_offset")) <= offset
                ):
                    raise ValueError("original_block_mapping_failed")
                offset = next_offset + bool(source.get("next_char_offset"))
        text_parts = [block["text"]]
        char_end = block["char_end"]
        # 只通过 block_id 续读同一原块，不获取整篇全文。
        while (
            source["next_offset"] is not None
            and source.get("next_char_offset", 0) > 0
            and source["next_char_offset"] == char_end
            and source["items"][-1]["block_id"] == block["block_id"]
        ):
            response = client.post(
                "/v1/evidence/read",
                json={
                    **identity,
                    "library_id": library_id,
                    "generation_id": scope["generation_id"],
                    "block_id": block["block_id"],
                    "char_offset": char_end,
                    "limit": 1,
                    "max_characters": 32000,
                },
            )
            response.raise_for_status()
            source = response.json()
            continuation = source["items"][0]
            if (
                continuation["block_id"] != block["block_id"]
                or continuation["reading_order"] != order
                or continuation["char_start"] != char_end
                or continuation["char_end"] <= char_end
            ):
                raise ValueError("original_block_mapping_failed")
            text_parts.append(continuation["text"])
            char_end = continuation["char_end"]
        text = "".join(text_parts)
        if not source_range.get("derived") and not 0 <= source_range["start"] <= source_range[
            "end"
        ] <= len(text):
            raise ValueError("original_range_failed")
        checked.append(
            {
                "reading_order": order,
                "range": source_range,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "locations": block["locations"],
                "characters": len(text),
            }
        )
    return checked


def validate(base_url, library_id, cases, records, *, timeout=380):
    """名称：真实联调。用途：正式 API 四模式、明确问题预期与固定身份回源。

    Args:
        base_url, library_id: 已启动 owner 与已发布库。
        cases: 小型人工可复核问题，不是最终冻结评测集。
        records: 已完成记录，异常时仍供调用方保存。
        timeout: 有限单请求预算。
    Returns:
        不含全文的可复核结果，不外推完整质量准确率。
    """
    with httpx.Client(base_url=base_url, trust_env=False, timeout=timeout) as client:
        metadata = client.get("/internal/models")
        metadata.raise_for_status()
        papers = client.get(f"/v1/libraries/{library_id}/papers")
        papers.raise_for_status()
        paper_ids = {item["paper_id"] for item in papers.json()["items"]}
        fixed_generation = papers.json()["active_generation_id"]
        for case in cases:
            paper_id = case.get("paper_id")
            if paper_id and paper_id not in paper_ids:
                raise ValueError("validation_paper_missing")
            for mode in ("dense", "bm25", "rrf", "hybrid_rerank"):
                started = time.monotonic()
                response = client.post(
                    "/v1/retrieval/candidates",
                    json={
                        "library_id": library_id,
                        "generation_id": fixed_generation,
                        "paper_ids": [paper_id] if paper_id else [],
                        "query": case["query"],
                        "mode": mode,
                        "limit": 10,
                    },
                )
                response.raise_for_status()
                result = response.json()
                if (
                    any(s != "rerank_partial" for s in result["statuses"])
                    or not result["candidates"]
                ):
                    raise ValueError("real_retrieval_incomplete")
                candidates = result["candidates"]
                if paper_id and any(c["paper_id"] != paper_id for c in candidates):
                    raise ValueError("paper_filter_failed")
                checked = check_original(client, library_id, result["scope"], candidates[0])
                safe_candidates = []
                for candidate in candidates:
                    text = candidate.pop("retrieval_text")
                    candidate["retrieval_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    candidate["expected_terms_present"] = [
                        term for term in case["terms"] if term.lower() in text.lower()
                    ]
                    safe_candidates.append(candidate)
                records.append(
                    {
                        **case,
                        "mode": mode,
                        "scope": result["scope"],
                        "statuses": result["statuses"],
                        "candidates": safe_candidates,
                        "checked_original_blocks": checked,
                        "timings": result["timings"],
                        "http_seconds": time.monotonic() - started,
                    }
                )
                print(
                    json.dumps({"case": case["name"], "mode": mode, "status": "passed"}), flush=True
                )
        response = client.post(
            "/v1/retrieval/candidates",
            json={
                "library_id": library_id,
                "query": "zzzxxyyqqnonexistent",
                "mode": "bm25",
            },
        )
        response.raise_for_status()
        if response.json()["statuses"] != ["no_hits"] or response.json()["candidates"]:
            raise ValueError("oov_status_failed")
    return {
        "status": "passed",
        "models": metadata.json(),
        "query_count": len(records),
        "records": records,
        "oov": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description="M2a-1 正式服务只读验证")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    try:
        result = validate(
            args.base_url,
            args.library_id,
            json.loads(args.cases.read_text(encoding="utf-8")),
            records,
        )
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("status", "query_count", "error_type") if key in result}
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
