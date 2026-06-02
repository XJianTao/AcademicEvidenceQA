"""模块名称：M2a-2 正式问答验证。

用途：通过正式 HTTP 保存小样本问答/引用回读，并记录真实 endpoint 超时/取消。
核心设计：每次请求前持久预留所有可能尝试的预算；未知结果不释放额度。
工程要点：不打印问题、回答、凭据或供应商异常，结果仅写入忽略目录。
"""

import argparse
import asyncio
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx
from openai import AsyncOpenAI

from academic_evidence_qa.config import load_settings
from academic_evidence_qa.infrastructure.generation_client import GenerationClient, GenerationError


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    # 同一目录的后续小样本共享额度；未知预留在进程中断后仍保留。
    ledger = path.parent / "token_budget.json"
    ledger_tmp = ledger.with_suffix(".tmp")
    ledger_tmp.write_text(
        json.dumps(
            {
                "accounted_tokens": result["accounted_tokens"],
                "remaining_tokens": result["remaining_tokens"],
                "latest_output": path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ledger_tmp.replace(ledger)


def reserve(result, amount, path):
    """预留先落盘；脚本中止或 usage 缺失均不按零消耗处理。"""
    if result["accounted_tokens"] + amount > result["token_limit"]:
        raise ValueError("validation_budget_exhausted")
    result["accounted_tokens"] += amount
    result["remaining_tokens"] = result["token_limit"] - result["accounted_tokens"]
    write_result(path, result)


async def endpoint_cleanup(result, path, settings):
    """名称：真实连接清理检查。用途：只发极小文本，不重新上传或解析论文。

    Args:
        result, path: 同一累计额度台账。
        settings: 私有配置，仅传入 SDK，不写入输出。
    Returns:
        None；记录本地超时/取消与共享池关闭事实，不声称远端停止计费。
    """
    for action in ("timeout", "cancel"):
        reserve(result, (settings.generation_context_tokens + 16) * 2, path)
        sdk = AsyncOpenAI(
            base_url=settings.generation_base_url,
            api_key=settings.generation_api_key,
            max_retries=0,
        )
        client = GenerationClient(
            sdk,
            model=settings.generation_model,
            timeout_seconds=0.1 if action == "timeout" else 10,
            max_retries=0,
            context_tokens=settings.generation_context_tokens,
            max_bytes=settings.generation_request_max_bytes,
        )
        started = time.monotonic()
        outcome = "unexpected_completion"
        try:
            task = asyncio.create_task(
                client.generate([{"role": "user", "content": "Reply OK."}], 16)
            )
            if action == "cancel":
                await asyncio.sleep(0.05)
                task.cancel()
            await task
        except asyncio.CancelledError:
            outcome = "cancelled"
        except GenerationError as error:
            outcome = error.code
        finally:
            await client.close()
        result["cleanup"].append(
            {
                "action": action,
                "outcome": outcome,
                "sdk_closed": sdk.is_closed(),
                "seconds": time.monotonic() - started,
                "remote_billing": "unknown_reserved",
            }
        )
        write_result(path, result)


def validate(args):
    """名称：正式链路验证。用途：一次 answer 对应一次检索，引用通过正式接口回读。

    Args:
        args: URL、库、问题文件、忽略目录输出。
    Returns:
        退出码；保留失败与历史记录，禁止自动覆盖既有台账或重新计费。
    """
    if "ignored" not in args.output.parts or args.output.exists():
        raise ValueError("new_ignored_output_required")
    settings = load_settings()
    result = {
        "token_limit": 1_000_000,
        "accounted_tokens": 0,
        "remaining_tokens": 1_000_000,
        "records": [],
        "cleanup": [],
        "status": "running",
    }
    ledger = args.output.parent / "token_budget.json"
    if ledger.exists():
        previous = json.loads(ledger.read_text(encoding="utf-8"))
        result["accounted_tokens"] = previous["accounted_tokens"]
        result["remaining_tokens"] = 1_000_000 - result["accounted_tokens"]
    write_result(args.output, result)
    try:
        with httpx.Client(base_url=args.base_url, trust_env=False, timeout=490) as client:
            metadata = client.get("/internal/models")
            metadata.raise_for_status()
            result["models"] = metadata.json()
            for case in json.loads(args.cases.read_text(encoding="utf-8")):
                maximum = case.get("max_output_tokens", 700)
                reservation = (settings.generation_context_tokens + maximum) * (
                    settings.generation_max_retries + 1
                )
                key = "m2a2-" + uuid4().hex
                record = {**case, "idempotency_key": key, "reservation": reservation}
                result["records"].append(record)
                reserve(result, reservation, args.output)
                started = time.monotonic()
                response = client.post(
                    "/v1/answer",
                    headers={"Idempotency-Key": key},
                    json={
                        "library_id": args.library_id,
                        "paper_ids": case.get("paper_ids", []),
                        "query": case["question"],
                        "mode": case.get("mode", "rrf"),
                        "limit": 6,
                        "evidence_token_budget": case.get("evidence_token_budget", 16000),
                        "max_output_tokens": maximum,
                    },
                )
                record.update(
                    http_status=response.status_code,
                    answer=response.json(),
                    http_seconds=time.monotonic() - started,
                )
                answer = record["answer"]
                if response.status_code == 200:
                    usage = answer.get("usage", {})
                    charged = usage.get("accounted_tokens")
                    # 成功但未调用模型的不足/预算状态为零；其余未知保留预留。
                    if not usage and answer["status"] in (
                        "insufficient_evidence",
                        "budget_exhausted",
                    ):
                        charged = 0
                    if charged is not None:
                        result["accounted_tokens"] += charged - reservation
                        result["remaining_tokens"] = 1_000_000 - result["accounted_tokens"]
                    record["citation_reads"] = []
                    for citation in answer.get("citations", []):
                        read = client.post(
                            "/v1/evidence/read",
                            json={
                                "library_id": args.library_id,
                                "retrieval_run_id": answer["retrieval_run_id"],
                                "evidence_id": citation["evidence_id"],
                                "max_characters": 16000,
                            },
                        )
                        record["citation_reads"].append(
                            {"http_status": read.status_code, "body": read.json()}
                        )
                write_result(args.output, result)
                print(
                    json.dumps(
                        {
                            "case": case["name"],
                            "http_status": response.status_code,
                            "status": answer.get("status", answer.get("error_code")),
                        }
                    ),
                    flush=True,
                )
            if args.cleanup:
                asyncio.run(endpoint_cleanup(result, args.output, settings))
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__)
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "accounted_tokens": result["accounted_tokens"],
                "remaining_tokens": result["remaining_tokens"],
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description="M2a-2 有额度上限的正式问答验证")
    parser.add_argument("--base-url", default="http://127.0.0.1:58010")
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cleanup", action="store_true")
    return validate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
