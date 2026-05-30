"""模块名称：Worker 内部编码客户端。

用途：生产 Worker 仅加载 tokenizer，以有限 HTTP 批次复用 API 模型 owner。
核心设计：先核验身份，再逐响应核验同一配置、维度、范数与条数；失败无本地回退。
工程要点：仅 loopback，有限重试与总期限，第三方消息不进入持久化日志。
"""

import asyncio
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from transformers import AutoTokenizer


class EmbeddingClient:
    """名称：内部批量编码。用途：符合既有构建 runtime 的小接口，不加载权重。

    Args:
        model_path, reranker_path: 仅本地 tokenizer 路径。
        base_url: loopback API；不可用时有限失败。
        expected_identity: 可选冻结配置哈希，不接受静默切换。
    Returns:
        供同步构建线程使用的 client，调用者必须 close。
    """

    def __init__(
        self,
        model_path: Path,
        reranker_path: Path,
        *,
        batch_size: int = 4,
        base_url: str = "http://127.0.0.1:8000",
        expected_identity: str | None = None,
        timeout_seconds: float = 120,
        max_retries: int = 1,
    ):
        if urlsplit(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("embedding_endpoint_not_loopback")
        if not 1 <= batch_size <= 4 or timeout_seconds <= 0 or not 0 <= max_retries <= 2:
            raise ValueError("embedding_client_config_invalid")
        self.model_path = model_path.resolve()
        self.batch_size = batch_size
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.base_url = base_url
        self.closed = False
        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                str(reranker_path), local_files_only=True
            )
            metadata = self._request("GET", "/internal/models")
            self.identity: dict[str, Any] = metadata["identity"]
            self.device, self.dtype = metadata["device"], metadata["dtype"]
            unsigned = {k: v for k, v in self.identity.items() if k != "configuration_sha256"}
            digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
            if (
                self.identity.get("pooling") != "cls"
                or self.identity.get("dimension") != 1024
                or self.identity.get("normalized") is not True
                or self.identity.get("configuration_sha256") != digest
            ):
                raise ValueError("embedding_identity_mismatch")
            if expected_identity and self.identity["configuration_sha256"] != expected_identity:
                raise ValueError("embedding_identity_mismatch")
            # 本地 tokenizer 的具体哈希由 API 身份逐文件确认，禁止构建长度语义漂移。
            for prefix, root in (("embedding", model_path), ("reranker_tokenizer", reranker_path)):
                for name in ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model"):
                    path = root / name
                    if path.is_file():
                        with path.open("rb") as stream:
                            digest = hashlib.file_digest(stream, "sha256").hexdigest()
                        if self.identity["files"].get(f"{prefix}/{name}") != digest:
                            raise ValueError("embedding_identity_mismatch")
        except Exception:
            self.close()
            raise

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.closed:
            raise ValueError("embedding_client_closed")
        # 构建线程使用同步接口；内部事件循环严格限制整轮 HTTP、响应读取和重试。
        # 每轮关闭连接，避免跨 asyncio.run 重用绑定旧事件循环的连接池。
        return asyncio.run(self._request_async(method, path, body))

    async def _request_async(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with httpx.AsyncClient(
                    base_url=self.base_url, trust_env=False, timeout=self.timeout_seconds
                ) as client:
                    for attempt in range(self.max_retries + 1):
                        try:
                            async with client.stream(method, path, json=body) as response:
                                if response.status_code == 200:
                                    data = bytearray()
                                    async for part in response.aiter_bytes():
                                        if len(data) + len(part) > 256 * 1024:
                                            raise ValueError("embedding_response_too_large")
                                        data.extend(part)
                                    try:
                                        result = json.loads(data)
                                    except (ValueError, UnicodeError) as error:
                                        raise ValueError("embedding_response_invalid") from error
                                    if not isinstance(result, dict):
                                        raise ValueError("embedding_response_invalid")
                                    return result
                                if response.status_code not in {429, 503}:
                                    raise ValueError("embedding_api_rejected")
                        except (httpx.TimeoutException, httpx.TransportError):
                            pass
                        if attempt == self.max_retries:
                            raise ValueError("embedding_api_unavailable")
                        await asyncio.sleep(0.2)
        except TimeoutError as error:
            raise ValueError("embedding_api_timeout") from error
        raise ValueError("embedding_api_unavailable")

    def embedding_length(self, text: str) -> int:
        return len(self.embedding_tokenizer(text, truncation=False)["input_ids"])

    def reranker_length(self, text: str) -> int:
        return len(self.reranker_tokenizer(text, truncation=False)["input_ids"])

    def encode(self, texts: list[str]) -> list[list[float]]:
        """名称：编码一个 Worker 批次。用途：核验返回身份及完整有限向量。

        Args:
            texts: 最多 batch_size 条、各不超过 600 tokens。
        Returns:
            与输入一一对应的归一化 1024 维向量。
        """
        if not 1 <= len(texts) <= self.batch_size or any(
            self.embedding_length(t) > 600 for t in texts
        ):
            raise ValueError("embedding_batch_invalid")
        result = self._request(
            "POST",
            "/internal/encode",
            {"texts": texts, "configuration_sha256": self.identity["configuration_sha256"]},
        )
        if result.get("identity") != self.identity:
            raise ValueError("embedding_identity_mismatch")
        vectors = result.get("vectors", [])
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(texts)
            or any(
                not isinstance(v, list)
                or len(v) != 1024
                or not all(type(x) in (int, float) and math.isfinite(x) for x in v)
                or abs(sum(x * x for x in v) - 1) > 0.02
                for v in vectors
            )
        ):
            raise ValueError("embedding_vector_invalid")
        return vectors

    def close(self) -> None:
        """关闭 HTTP 连接，不拥有也不释放 API 权重。"""
        self.closed = True
