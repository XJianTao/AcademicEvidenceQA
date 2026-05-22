"""模块名称：generation 级 BM25 词法编码。

用途：冻结中英文词表与文档频率，产生 Qdrant sparse dot product 向量。
核心设计：文档侧只有 TF 饱和项，查询去重后仅施加一次正值 IDF。
工程要点：词表按字典序固定；空语料和 OOV 均返回空向量，不启用 Qdrant IDF。
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re

import jieba

K1 = 1.2
B = 0.75
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:[-_][a-zA-Z0-9]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """名称：中英术语分词。用途：保留英文连字符、数字和中文词项。

    Args:
        text: 检索表示或查询文本。
    Returns:
        小写且非空的词项序列；顺序保留用于词频统计。
    """
    tokens = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        term = match.group()
        tokens.extend(word for word in jieba.cut(term, HMM=False) if word.strip()) if re.fullmatch(
            r"[\u3400-\u9fff]+", term
        ) else tokens.append(term)
    return tokens


@dataclass(frozen=True)
class LexicalStatistics:
    """名称：冻结词法统计。用途：在构建与查询间保持词表和 IDF 一致。

    Args:
        vocabulary: 按词项排序的稳定词表。
        document_frequency: 与词表同序的文档频率。
        document_count, average_length: 同一 generation 的 chunk 统计。
    Returns:
        可序列化且可校验 SHA-256 的统计对象。
    """

    vocabulary: tuple[str, ...]
    document_frequency: tuple[int, ...]
    document_count: int
    average_length: float

    def to_dict(self) -> dict:
        return {
            "vocabulary": list(self.vocabulary),
            "document_frequency": list(self.document_frequency),
            "document_count": self.document_count,
            "average_length": self.average_length,
            "k1": K1,
            "b": B,
            "tokenizer": "jieba-0.42.1-hmm-off-v1",
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def freeze_statistics(texts: list[str]) -> LexicalStatistics:
    """名称：冻结语料统计。用途：以所有可检索 chunk 一次计算 N、df 与 avgdl。

    Args:
        texts: 当前 generation 的 retrieval_text 列表。
    Returns:
        空库也有确定性空统计；非空库无词项时显式失败。
    """
    documents = [tokenize(text) for text in texts]
    frequencies = Counter(term for document in documents for term in set(document))
    vocabulary = tuple(sorted(frequencies))
    if documents and not vocabulary:
        raise ValueError("lexical_corpus_empty")
    return LexicalStatistics(
        vocabulary,
        tuple(frequencies[term] for term in vocabulary),
        len(documents),
        sum(map(len, documents)) / len(documents) if documents else 0.0,
    )


def document_vector(text: str, statistics: LexicalStatistics) -> dict[int, float]:
    """名称：文档侧编码。用途：仅编码 TF 饱和与长度归一化。

    Args:
        text: 已冻结的 retrieval_text。
        statistics: 该 generation 的统计。
    Returns:
        稳定词项 ID 到有限正权重的映射。
    """
    terms = tokenize(text)
    if not terms or statistics.average_length <= 0:
        return {}
    counts = Counter(terms)
    index = {term: number for number, term in enumerate(statistics.vocabulary)}
    norm = K1 * (1 - B + B * len(terms) / statistics.average_length)
    return {
        index[term]: count * (K1 + 1) / (count + norm)
        for term, count in counts.items()
        if term in index
    }


def query_vector(query: str, statistics: LexicalStatistics) -> dict[int, float]:
    """名称：查询侧编码。用途：唯一查询词项仅加一次平滑正 IDF。

    Args:
        query: 用户查询，不在本模块执行范围过滤。
        statistics: 已发布 generation 的冻结统计。
    Returns:
        OOV 或空查询返回空向量。
    """
    index = {term: number for number, term in enumerate(statistics.vocabulary)}
    return {
        index[term]: math.log1p(
            (statistics.document_count - statistics.document_frequency[index[term]] + 0.5)
            / (statistics.document_frequency[index[term]] + 0.5)
        )
        for term in set(tokenize(query))
        if term in index
    }


def reference_score(query: str, text: str, statistics: LexicalStatistics) -> float:
    """名称：参考算分。用途：核对 Qdrant sparse dot product 排序与数值。"""
    document = document_vector(text, statistics)
    return sum(
        value * document.get(term_id, 0.0)
        for term_id, value in query_vector(query, statistics).items()
    )
