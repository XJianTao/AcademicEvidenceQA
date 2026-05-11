"""本地模型真实推理与资源记录入口。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


def main() -> None:
    dense_path = Path(os.environ["BGE_M3_MODEL_PATH"])
    reranker_path = Path(os.environ["RERANKER_MODEL_PATH"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    report: dict[str, object] = {
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda, "device": device, "dtype": str(dtype),
        "batch_size": 1, "input_characters": 90,
    }
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embedder = SentenceTransformer(str(dense_path), device=device, local_files_only=True, model_kwargs={"torch_dtype": dtype})
    report["embedding_cold_load_seconds"] = time.perf_counter() - started
    text = "We propose a graph model for collective sentiment analysis with improved F1."
    embedding_started = time.perf_counter()
    vector = embedder.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
    report["embedding_first_seconds"] = time.perf_counter() - embedding_started
    times = []
    for _ in range(3):
        started = time.perf_counter()
        embedder.encode([text], batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
        times.append(time.perf_counter() - started)
    report["embedding_hot_seconds"] = times
    report["embedding_dimension"] = int(vector.shape[1])
    report["embedding_finite"] = bool(torch.isfinite(torch.from_numpy(vector)).all())
    del embedder
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reranker = CrossEncoder(str(reranker_path), device=device, max_length=512, local_files_only=True, automodel_args={"torch_dtype": dtype})
    report["reranker_cold_load_seconds"] = time.perf_counter() - started
    pairs = [["What improves sentiment F1?", "The proposed model improves F1 on sentiment classification."],
             ["What improves sentiment F1?", "The paper studies an unrelated database index."]]
    token_count = len(reranker.tokenizer(pairs[0][0], pairs[0][1], truncation=True, max_length=512)["input_ids"])
    started = time.perf_counter()
    scores = reranker.predict(pairs, batch_size=1, convert_to_numpy=True)
    report["reranker_pair_seconds"] = time.perf_counter() - started
    report["reranker_pair_tokens"] = token_count
    report["reranker_scores"] = [float(score) for score in scores]
    report["relevant_ranked_higher"] = bool(scores[0] > scores[1])
    if device == "cuda":
        report["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        report["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
    try:
        import psutil
        report["process_peak_rss_mib"] = round(psutil.Process().memory_info().peak_wset / 1024**2, 1)
    except (ImportError, AttributeError):
        pass
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
