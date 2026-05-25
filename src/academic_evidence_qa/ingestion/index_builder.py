"""模块名称：冻结 manifest 与索引构建。

用途：将固定来源切分、编码并写入 attempt 专属 Qdrant 暂存 collection。
核心设计：manifest 包含精确 chunk 集合、词法统计与所有来源身份。
工程要点：不持有数据库事务，不调用重排或生成模型；空库不加载模型。
"""

from dataclasses import asdict, dataclass
import hashlib
import json

from academic_evidence_qa.documents.chunker import CHUNKER_VERSION, build_chunks, chunk_id
from academic_evidence_qa.documents.source_reader import FixedSource
from academic_evidence_qa.retrieval.lexical import document_vector, freeze_statistics


def content_hash(value: dict) -> str:
    """名称：冻结内容哈希。用途：统一 JSON 序列化口径。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BuiltIndex:
    """名称：暂存索引结果。用途：绑定冻结清单、point 和 chunk 源映射。"""

    manifest: dict
    manifest_sha256: str
    points: list[dict]
    chunks: list[dict]


def build_index(
    generation, sources: list[FixedSource], runtime, qdrant, *, execution_check=None
) -> BuiltIndex:
    """名称：构建暂存索引。用途：编码固定输入，不执行发布。

    Args:
        generation: 已在短事务中预留的 generation 身份。
        sources: 已核验且完整的成员来源。
        runtime: 有界单 owner 模型；空库可为 None。
        qdrant: 不删除历史集合的存储适配器。
        execution_check: C2 线程安全停止检查，编码批次及外部写入边界执行。
    Returns:
        可持久化且可独立核验的完整构建结果。
    """
    drafts = (
        build_chunks(sources, runtime.embedding_length, runtime.reranker_length) if sources else []
    )
    statistics = freeze_statistics([draft.retrieval_text for draft in drafts])
    dense = []
    if execution_check:
        batch_size = getattr(runtime, "batch_size", 4)
        for offset in range(0, len(drafts), batch_size):
            execution_check()
            dense.extend(
                runtime.encode(
                    [draft.retrieval_text for draft in drafts[offset : offset + batch_size]]
                )
            )
    elif drafts:
        dense = runtime.encode([draft.retrieval_text for draft in drafts])
    if len(dense) != len(drafts):
        raise ValueError("embedding_count_mismatch")
    chunks, points = [], []
    for order, (draft, vector) in enumerate(zip(drafts, dense, strict=True)):
        identity = chunk_id(generation.generation_id, draft, order)
        data = asdict(draft)
        data.update(chunk_id=identity, generation_id=generation.generation_id, reading_order=order)
        chunks.append(data)
        points.append(
            {
                "id": str(identity),
                "dense": vector,
                "sparse": document_vector(draft.retrieval_text, statistics),
                "payload": {
                    "library_id": str(generation.library_id),
                    "generation_id": str(generation.generation_id),
                    "attempt_id": str(generation.attempt_id),
                    "paper_id": str(draft.paper_id),
                    "document_version_id": str(draft.document_version_id),
                    "parse_snapshot_id": str(draft.parse_snapshot_id),
                    "chunk_id": str(identity),
                    "block_type": draft.block_type,
                    "source_ranges": list(draft.source_ranges),
                    "source_locations": list(draft.source_locations),
                    "statistics_sha256": statistics.sha256,
                },
            }
        )
    manifest = {
        "library_id": str(generation.library_id),
        "generation_id": str(generation.generation_id),
        "attempt_id": str(generation.attempt_id),
        "membership_revision": generation.membership_revision,
        "expected_active_generation_id": str(generation.expected_active_generation_id)
        if generation.expected_active_generation_id
        else None,
        "collection_name": generation.collection_name,
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": "local-bge-m3-cls-normalized-1024",
        "model_path": str(runtime.model_path) if runtime else None,
        "model_identity": getattr(runtime, "identity", {}) if runtime else {},
        "members": [
            {
                "paper_id": str(source.paper_id),
                "document_version_id": str(source.document_version_id),
                "parse_snapshot_id": str(source.parse_snapshot_id),
                "file_sha256": source.file_sha256,
                "parser_backend": source.parser_backend,
                "parser_config_id": source.parser_config_id,
                "adapter_version": source.adapter_version,
            }
            for source in sources
        ],
        "chunk_ids": [point["id"] for point in points],
        "chunks_sha256": content_hash(json.loads(json.dumps(chunks, default=str))),
        "lexical_statistics": statistics.to_dict(),
        "statistics_sha256": statistics.sha256,
    }
    if execution_check:
        execution_check()
    qdrant.create(generation.collection_name)
    if execution_check:
        execution_check()
    qdrant.write(generation.collection_name, points)
    return BuiltIndex(manifest, content_hash(manifest), points, chunks)
