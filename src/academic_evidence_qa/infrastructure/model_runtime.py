"""模块名称：本地 BGE-M3 编码运行时。

用途：进程内单实例加载本地模型，按有界 batch 生成 1024 维归一化 dense。
核心设计：只加载本机已存在的权重；串行调用，不建立内部服务。
工程要点：实际 tokenizer 长度先于推理检查，不允许静默截断或非有限向量。
"""

from pathlib import Path
import math
import hashlib
import json
import threading

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

DEFAULT_BGE_PATH = Path(r"D:\ai_models\modelscope_cache\models\BAAI\bge-m3")
DEFAULT_RERANKER_PATH = Path(r"D:\ai_models\modelscope_cache\models\rerank\BAAI\bge-reranker-large")


class ModelRuntime:
    """名称：模型运行时。用途：持有单一 BGE-M3 owner 与两个 tokenizer。

    Args:
        model_path, reranker_path: 已安装的本地目录，不发生模型下载。
        batch_size: 每次 encode 的最大文本条数。
    Returns:
        可在一个显式 CLI 构建期间串行使用的运行时。
    """

    _owner: "ModelRuntime | None" = None
    _owner_lock = threading.Lock()

    def __init__(self, model_path: Path, reranker_path: Path, *, batch_size: int = 4):
        if batch_size < 1 or not model_path.is_dir() or not reranker_path.is_dir():
            raise ValueError("local_model_unavailable")
        self.model_path = model_path.resolve()
        self.reranker_path = reranker_path.resolve()
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
            model_kwargs={"dtype": self.dtype},
        )
        self.embedding_tokenizer = self.model.tokenizer
        pooling = self.model[1]
        if not getattr(pooling, "pooling_mode_cls_token", False) or any(
            getattr(pooling, name, False)
            for name in (
                "pooling_mode_mean_tokens",
                "pooling_mode_max_tokens",
                "pooling_mode_mean_sqrt_len_tokens",
                "pooling_mode_weightedmean_tokens",
                "pooling_mode_lasttoken",
            )
        ):
            raise ValueError("embedding_pooling_configuration_invalid")
        if self.model.max_seq_length < 600:
            raise ValueError("embedding_model_length_insufficient")
        # 固定配置和权重身份；分块哈希不将大权重读入额外内存。
        self.identity = {"pooling": "cls", "dimension": 1024, "normalized": True, "files": {}}
        for root, label in (
            (self.model_path, "embedding"),
            (self.reranker_path, "reranker_tokenizer"),
        ):
            names = (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "sentencepiece.bpe.model",
                "modules.json",
                "1_Pooling/config.json",
            )
            paths = [root / name for name in names if (root / name).is_file()]
            if label == "embedding":
                paths += sorted(root.glob("*.safetensors")) or sorted(
                    root.glob("pytorch_model*.bin")
                )
            for path in paths:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for data in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(data)
                self.identity["files"][f"{label}/{path.relative_to(root).as_posix()}"] = (
                    digest.hexdigest()
                )
        self.identity["configuration_sha256"] = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            str(self.reranker_path), local_files_only=True
        )

    @classmethod
    def owner(cls, model_path: Path, reranker_path: Path, *, batch_size: int = 4) -> "ModelRuntime":
        """名称：获取进程 owner。用途：禁止同进程重复加载大模型。"""
        with cls._owner_lock:
            if cls._owner is None:
                cls._owner = cls(model_path, reranker_path, batch_size=batch_size)
            elif (
                cls._owner.model_path != model_path.resolve()
                or cls._owner.reranker_path != reranker_path.resolve()
                or cls._owner.batch_size != batch_size
            ):
                raise ValueError("model_owner_configuration_conflict")
            return cls._owner

    def embedding_length(self, text: str) -> int:
        """名称：embedding 长度。用途：含特殊 token 的真实编码长度。"""
        return len(
            self.embedding_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        )

    def reranker_length(self, text: str) -> int:
        """名称：reranker 长度。用途：记录独立 tokenizer 的含特殊 token 长度。"""
        return len(
            self.reranker_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """名称：编码 dense。用途：逐批生成有限、归一化的 CLS 向量。

        Args:
            texts: 已完成硬限制检查的 retrieval_text。
        Returns:
            与输入等长的 1024 维 float 向量。
        Raises:
            ValueError: token 超限、维度/范数错误或非有限数值。
        """
        if any(self.embedding_length(text) > 600 for text in texts):
            raise ValueError("embedding_token_limit_exceeded")
        output = []
        for start in range(0, len(texts), self.batch_size):
            vectors = self.model.encode(
                texts[start : start + self.batch_size],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for vector in vectors:
                values = [float(value) for value in vector]
                if len(values) != 1024 or not all(math.isfinite(value) for value in values):
                    raise ValueError("embedding_vector_invalid")
                if not 0.99 <= math.sqrt(sum(value * value for value in values)) <= 1.01:
                    raise ValueError("embedding_vector_not_normalized")
                output.append(values)
        return output
