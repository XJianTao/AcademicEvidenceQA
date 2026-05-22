"""模块名称：独立 generation 的 Qdrant 存储。

用途：创建暂存 collection，批量写入同 point 的 dense/BM25 和来源 payload。
核心设计：每次 attempt 独占 collection；完整滚动读取逐点校验，禁止 alias 发布。
工程要点：不自动删除旧 collection；Qdrant sparse 不设置 IDF modifier。
"""

import math

from qdrant_client import QdrantClient, models


class IndexIntegrityError(Exception):
    """索引校验失败，不得继续发布。"""


class QdrantStore:
    """名称：Qdrant 存储。用途：封装本轮构建、核验和 smoke 查询。

    Args:
        url: 本地 Qdrant HTTP 地址。
    Returns:
        有界超时的同步客户端；调用方负责 close。
    """

    def __init__(self, url: str):
        self.client = QdrantClient(url=url, timeout=30)

    def create(self, collection: str) -> None:
        """名称：创建暂存集合。用途：拒绝已有集合，避免覆盖任何历史证据。"""
        if self.client.collection_exists(collection):
            raise IndexIntegrityError("staging_collection_exists")
        self.client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams()},
        )

    def write(self, collection: str, points: list[dict], *, batch_size: int = 32) -> None:
        """名称：写入 points。用途：分批等待确认，每点同时保存两路向量与身份。

        Args:
            collection: attempt 专属暂存集合。
            points: 已编码且包含 ID、dense、sparse 和 payload 的完整列表。
            batch_size: 有界写入批量。
        Returns:
            None；不表示已经发布或验收通过。
        """
        if batch_size < 1:
            raise ValueError("invalid_qdrant_batch_size")
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=point["id"],
                        vector={
                            "dense": point["dense"],
                            "bm25": models.SparseVector(
                                indices=sorted(point["sparse"]),
                                values=[point["sparse"][key] for key in sorted(point["sparse"])],
                            ),
                        },
                        payload=point["payload"],
                    )
                    for point in points[start : start + batch_size]
                ],
                wait=True,
            )

    def verify(self, collection: str, expected: list[dict]) -> dict:
        """名称：逐点完整性检查。用途：核验精确集合、向量和完整来源 payload。

        Args:
            collection: 尚未发布的集合。
            expected: 本次构建的固定 points，不从远端数量推导期望。
        Returns:
            含点数及真实 filter/dense/sparse smoke 的脱敏证据。
        Raises:
            IndexIntegrityError: 任意缺点、错 ID、向量或来源不一致。
        """
        actual = {}
        config = self.client.get_collection(collection).config.params
        if (
            config.vectors["dense"].size != 1024
            or config.vectors["dense"].distance != models.Distance.COSINE
            or config.sparse_vectors["bm25"].modifier is not None
        ):
            raise IndexIntegrityError("collection_configuration_mismatch")
        offset = None
        while True:
            rows, offset = self.client.scroll(
                collection, limit=128, offset=offset, with_payload=True, with_vectors=True
            )
            for row in rows:
                identity = str(row.id)
                if identity in actual:
                    raise IndexIntegrityError("duplicate_point_id")
                actual[identity] = row
            if offset is None:
                break
        expected_ids = {point["id"] for point in expected}
        if set(actual) != expected_ids or len(expected_ids) != len(expected):
            raise IndexIntegrityError("point_id_set_mismatch")
        for point in expected:
            row = actual[point["id"]]
            vectors = row.vector
            if not isinstance(vectors, dict) or set(vectors) != {"dense", "bm25"}:
                raise IndexIntegrityError("point_vectors_missing")
            dense = vectors["dense"]
            if len(dense) != 1024 or not all(
                math.isfinite(value) for value in dense + point["dense"]
            ):
                raise IndexIntegrityError("dense_vector_invalid")
            # COSINE 可能由服务再次归一化；浮点容差只作用于数值，不放宽身份检查。
            if any(
                abs(left - right) > 1e-5 for left, right in zip(dense, point["dense"], strict=True)
            ):
                raise IndexIntegrityError("dense_vector_mismatch")
            sparse = vectors["bm25"]
            values = dict(zip(sparse.indices, sparse.values, strict=True))
            if set(values) != set(point["sparse"]) or any(
                not math.isfinite(value) or abs(value - point["sparse"][key]) > 1e-5
                for key, value in values.items()
            ):
                raise IndexIntegrityError("sparse_vector_mismatch")
            if row.payload != point["payload"]:
                raise IndexIntegrityError("point_source_mismatch")
        smoke = {"dense": "empty", "filter": "empty", "sparse": "empty"}
        paper_filters = []
        # 多成员库逐篇验证；全部过滤仍读取本代预先编码的同一全库统计。
        for paper_id in sorted({point["payload"]["paper_id"] for point in expected}):
            point = next(item for item in expected if item["payload"]["paper_id"] == paper_id)
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))
                ]
            )
            rows, offset = self.client.scroll(
                collection,
                scroll_filter=query_filter,
                limit=max(1, len(expected)),
                with_payload=True,
                with_vectors=False,
            )
            wanted = {item["id"] for item in expected if item["payload"]["paper_id"] == paper_id}
            if offset is not None or {str(row.id) for row in rows} != wanted:
                raise IndexIntegrityError("paper_filter_id_set_mismatch")
            hits = self.client.query_points(
                collection, using="dense", query=point["dense"], query_filter=query_filter, limit=3
            ).points
            if not hits or any(
                hit.payload["paper_id"] != paper_id
                or hit.payload["statistics_sha256"] != point["payload"]["statistics_sha256"]
                for hit in hits
            ):
                raise IndexIntegrityError("paper_filtered_query_failed")
            paper_filters.append(
                {
                    "paper_id": paper_id,
                    "point_count": len(wanted),
                    "statistics_sha256": point["payload"]["statistics_sha256"],
                    "status": "passed",
                }
            )
        if expected:
            point = expected[0]
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id", match=models.MatchValue(value=point["payload"]["paper_id"])
                    )
                ]
            )
            hits = self.client.query_points(
                collection, using="dense", query=point["dense"], query_filter=query_filter, limit=3
            ).points
            if not hits or any(
                hit.payload["paper_id"] != point["payload"]["paper_id"] for hit in hits
            ):
                raise IndexIntegrityError("filtered_query_failed")
            smoke.update(dense="passed", filter="passed")
            if point["sparse"]:
                hits = self.client.query_points(
                    collection,
                    using="bm25",
                    query=models.SparseVector(
                        indices=sorted(point["sparse"]),
                        values=[point["sparse"][key] for key in sorted(point["sparse"])],
                    ),
                    limit=3,
                ).points
                if not hits:
                    raise IndexIntegrityError("sparse_query_failed")
                smoke["sparse"] = "passed"
        return {
            "point_count": len(actual),
            "exact_id_set": "passed",
            "smoke": smoke,
            "paper_filters": paper_filters,
        }

    def close(self) -> None:
        """关闭本实例客户端，不删除任何 collection。"""
        self.client.close()

    def verify_bm25(self, collection, points, texts, statistics, query: str) -> dict:
        """名称：BM25 数值复核。用途：实际 query 与独立公式逐点对照，不重算过滤统计。

        Args:
            collection, points, texts: 同一构建的集合、身份及检索文本。
            statistics: 冻结 generation 统计。
            query: 固定检查词项，重复词项只计一次。
        Returns:
            查询词项数、命中数和最大绝对误差；OOV 不发查询。
        """
        from collections import Counter
        from academic_evidence_qa.retrieval.lexical import query_vector, tokenize, K1, B

        vector = query_vector(query, statistics)
        if not vector:
            return {"status": "empty", "hit_count": 0}
        scores = {}
        frequencies = dict(zip(statistics.vocabulary, statistics.document_frequency, strict=True))
        for point, text in zip(points, texts, strict=True):
            tokens = tokenize(text)
            counts = Counter(tokens)
            score = 0.0
            for term in set(tokenize(query)):
                if term in frequencies and counts[term]:
                    idf = math.log(
                        1
                        + (statistics.document_count - frequencies[term] + 0.5)
                        / (frequencies[term] + 0.5)
                    )
                    tf = counts[term]
                    score += (
                        idf
                        * tf
                        * (K1 + 1)
                        / (tf + K1 * (1 - B + B * len(tokens) / statistics.average_length))
                    )
            if score > 0:
                scores[point["id"]] = score
        hits = self.client.query_points(
            collection,
            using="bm25",
            query=models.SparseVector(
                indices=sorted(vector), values=[vector[key] for key in sorted(vector)]
            ),
            limit=max(1, len(points)),
        ).points
        if {str(hit.id) for hit in hits} != set(scores):
            raise IndexIntegrityError("bm25_reference_id_mismatch")
        errors = [abs(hit.score - scores[str(hit.id)]) for hit in hits]
        if any(error > 1e-4 for error in errors) or any(
            hits[index].score < hits[index + 1].score for index in range(len(hits) - 1)
        ):
            raise IndexIntegrityError("bm25_reference_score_mismatch")
        return {
            "status": "passed",
            "hit_count": len(hits),
            "max_absolute_error": max(errors, default=0.0),
            "query_term_count": len(vector),
        }
