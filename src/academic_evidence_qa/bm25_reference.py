"""小语料 BM25 参考算分。

模块名称：BM25 手工参考实现
用途概要：为 Qdrant sparse 向量核对提供可复算的 BM25/IDF 基准。
核心设计：每个文档仅做一次 IDF 加权，稀疏点积使用相同 term 权重。
工程要点：实现限于固定小夹具，不作为生产分词或索引管线。
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """按连续拉丁/数字词和单个汉字生成稳定的小样本 token。"""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def bm25_vectors(
    documents: list[str], vocabulary: list[str] | None = None
) -> list[dict[int, float]]:
    """计算以整数词项 ID 为键的 BM25 稀疏权重。"""
    tokenized = [tokenize(document) for document in documents]
    terms = vocabulary or sorted({term for tokens in tokenized for term in tokens})
    ids = {term: index for index, term in enumerate(terms)}
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    result = []
    for terms in tokenized:
        frequencies = Counter(terms)
        weights = {}
        for term, frequency in frequencies.items():
            if term not in ids:
                continue
            idf = math.log(
                1
                + (len(tokenized) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_norm = (
                frequency
                * 2.2
                / (frequency + 1.2 * (0.25 + 0.75 * len(terms) / max(average_length, 1)))
            )
            weights[ids[term]] = idf * length_norm
        result.append(weights)
    return result
