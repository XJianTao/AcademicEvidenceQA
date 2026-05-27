"""模块名称：M1-C3 真实 HTTP 验证入口。

用途：调用独立 API，记录持久任务、发布身份和分页回源的脱敏证据。
核心设计：只编排 HTTP，不创建业务记录、不调用解析器或索引构建函数。
工程要点：输入须事先获准外发；输出只保存身份/哈希/计数，不记录论文全文。
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

import httpx


def run(args):
    """名称：真实管理闭环。用途：提交或接续同一验收，不重复上传。

    Args:
        args: loopback API、样本、保留型摘要和有界等待参数。
    Returns:
        脱敏摘要；任务失败或超时保留原响应身份供诊断。
    """
    output = args.output

    def save(summary):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with httpx.Client(base_url=args.base_url, timeout=90, trust_env=False) as client:
        if args.resume:
            summary = json.loads(output.read_text(encoding="utf-8"))
        else:
            library = client.post(
                "/v1/libraries", json={}, headers={"Idempotency-Key": uuid4().hex}
            )
            library.raise_for_status()
            library_id = library.json()["library_id"]
            summary = {
                "status": "running",
                "base_url": args.base_url,
                "library_id": library_id,
                "uploads": [],
                "source_checks": [],
            }
            save(summary)
            revision = 0
            for path in args.pdf:
                key = uuid4().hex
                with path.open("rb") as stream:
                    result = client.post(
                        "/v1/documents",
                        headers={"Idempotency-Key": key},
                        data={
                            "library_id": library_id,
                            "expected_membership_revision": str(revision),
                            "parser_backend": "mineru_remote",
                            "allow_remote_processing": "true",
                        },
                        files={"file": (path.name, stream, "application/pdf")},
                    )
                result.raise_for_status()
                body = result.json()
                summary["uploads"].append(
                    {
                        "sample": path.name,
                        "idempotency_key": key,
                        "http_status": result.status_code,
                        **body,
                    }
                )
                revision = body["membership_revision"]
                save(summary)
            if args.submit_only:
                return summary
        started = time.monotonic()
        latest_id = summary.get("final_index_job_id", summary["uploads"][-1]["index_job_id"])
        while time.monotonic() - started < args.timeout:
            jobs = []
            for upload in summary["uploads"]:
                for job_id in (upload.get("parse_job_id"), upload["index_job_id"]):
                    if job_id:
                        response = client.get(f"/v1/jobs/{job_id}")
                        response.raise_for_status()
                        jobs.append(response.json())
            summary["jobs"] = jobs
            if all(job["job_id"] != latest_id for job in jobs):
                response = client.get(f"/v1/jobs/{latest_id}")
                response.raise_for_status()
                jobs.append(response.json())
            save(summary)
            final = next(job for job in jobs if job["job_id"] == latest_id)
            if final["state"] == "succeeded":
                break
            if final["state"] in ("failed", "cancelled"):
                summary["status"] = "failed"
                save(summary)
                return summary
            time.sleep(args.poll_seconds)
        else:
            summary["status"] = "waiting_timeout"
            save(summary)
            return summary
        listing = client.get(f"/v1/libraries/{summary['library_id']}/papers").json()
        summary["listing"] = listing
        summary["source_checks"] = []
        for paper in listing["items"]:
            fixed = {
                "library_id": summary["library_id"],
                "generation_id": listing["active_generation_id"],
                "paper_id": paper["paper_id"],
                "document_version_id": paper["published_source"]["document_version_id"],
                "parse_snapshot_id": paper["published_source"]["parse_snapshot_id"],
            }
            outline = client.get(
                f"/v1/papers/{paper['paper_id']}/outline", params={**fixed, "limit": 2}
            )
            outline.raise_for_status()
            source = client.post(
                "/v1/evidence/read", json={**fixed, "limit": 2, "max_characters": 200}
            )
            source.raise_for_status()
            payload = source.json()
            assert payload["items"] and sum(len(item["text"]) for item in payload["items"]) <= 200
            summary["source_checks"].append(
                {
                    **fixed,
                    "outline_count": len(outline.json()["items"]),
                    "next_outline_offset": outline.json()["next_offset"],
                    "blocks": [
                        {key: value for key, value in item.items() if key != "text"}
                        | {"text_sha256": hashlib.sha256(item["text"].encode()).hexdigest()}
                        for item in payload["items"]
                    ],
                    "next_offset": payload["next_offset"],
                    "next_char_offset": payload["next_char_offset"],
                }
            )
        summary["status"] = "passed"
        summary["wait_seconds"] = round(time.monotonic() - started, 3)
        save(summary)
        return summary


def main():
    """名称：验证命令。用途：显式提交或只读接续，输出简短状态。"""
    parser = argparse.ArgumentParser(description="M1-C3 HTTP 验收，PDF 须预先授权外发")
    parser.add_argument("--base-url", default="http://127.0.0.1:58000")
    parser.add_argument("--pdf", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=2100)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if urlsplit(args.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("验证入口仅接受 loopback 地址")
    if not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.poll_seconds)):
        parser.error("等待与轮询间隔须为有限正数")
    if not args.resume and not args.pdf:
        parser.error("提交须提供 --pdf；接续使用 --resume")
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "library_id": result["library_id"],
                "sample_count": len(result["uploads"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] in ("passed", "running") else 1


if __name__ == "__main__":
    raise SystemExit(main())
