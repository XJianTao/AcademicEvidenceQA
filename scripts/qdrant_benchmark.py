"""Qdrant dense/sparse/metadata 真实服务验证入口。"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from academic_evidence_qa.bm25_reference import bm25_vectors, tokenize


def main() -> None:
    client = QdrantClient(url="http://127.0.0.1:56333", timeout=10)
    collection = "m0_fixture"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
    )
    documents = [
        ("graph sentiment analysis", [1.0, 0.0, 0.0], "core"),
        ("sentiment classification", [0.0, 1.0, 0.0], "core"),
        ("graph neural network", [0.9, 0.1, 0.0], "challenge"),
    ]
    vocabulary = sorted({term for text, _, _ in documents for term in tokenize(text)})
    vectors = bm25_vectors([text for text, _, _ in documents], vocabulary)
    client.upsert(collection, points=[
        models.PointStruct(
            id=index + 1,
            vector={
                "dense": dense,
                "bm25": models.SparseVector(
                    indices=list(vectors[index]), values=list(vectors[index].values())
                ),
            },
            payload={"text": text, "kind": kind},
        )
        for index, (text, dense, kind) in enumerate(documents)
    ], wait=True)
    query_sparse = bm25_vectors(["graph"], vocabulary)[0]
    sparse_query = models.SparseVector(indices=list(query_sparse), values=list(query_sparse.values()))
    dense = client.query_points(collection, using="dense", query=[1.0, 0.0, 0.0], limit=3).points
    sparse = client.query_points(collection, using="bm25", query=sparse_query, limit=3).points
    filtered = client.query_points(
        collection, using="dense", query=[1.0, 0.0, 0.0], limit=3,
        query_filter=models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="core"))]),
    ).points
    print({"dense_ids": [point.id for point in dense], "sparse_ids": [point.id for point in sparse],
           "filtered_ids": [point.id for point in filtered], "sparse_scores": [point.score for point in sparse]})
    client.close()


if __name__ == "__main__":
    main()
