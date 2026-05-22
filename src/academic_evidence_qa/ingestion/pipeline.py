"""模块名称：单库显式完整重建 pipeline。

用途：串行编排固定来源、结构切分、编码、暂存、校验和发布。
核心设计：复用正式服务；每次 attempt 独立集合，失败仅记录本代并保留旧指针。
工程要点：不领取任务、不执行远端解析；串行入口复用正式构建与发布服务。
"""

from pathlib import Path
import time
import re
from uuid import UUID

from academic_evidence_qa.documents.source_reader import read_fixed_sources
from academic_evidence_qa.infrastructure.model_runtime import ModelRuntime
from academic_evidence_qa.ingestion.index_builder import build_index
from academic_evidence_qa.retrieval.lexical import LexicalStatistics
from academic_evidence_qa.ingestion.publication import (
    fail_generation,
    publish_generation,
    reserve_generation,
    save_built_index,
)


async def rebuild_library(
    database,
    library_id: UUID,
    store,
    qdrant,
    *,
    model_path: Path,
    reranker_path: Path,
    max_json_bytes: int,
    batch_size: int = 4,
    runtime_factory=ModelRuntime.owner,
) -> dict:
    """名称：完整重建 Library。用途：提供 C1 CLI 与后续 C2 可复用服务。

    Args:
        database, library_id, store, qdrant: 固定目标库的正式存储组件。
        model_path, reranker_path: 现有本地模型与 tokenizer 路径。
        max_json_bytes, batch_size: 显式解析读取和模型批量上限。
        runtime_factory: 测试可注入 fake，真实运行默认单模型 owner。
    Returns:
        仅含身份、计数、阶段耗时和校验证据的摘要。
    Raises:
        Exception: 任一必需阶段失败；旧活动代保持不变。
    """
    started = time.monotonic()

    generation = await reserve_generation(database, library_id)
    try:
        async with database.sessions() as session:
            sources = await read_fixed_sources(
                session, library_id, store, max_json_bytes=max_json_bytes
            )
        read_seconds = time.monotonic() - started
        load_started = time.monotonic()
        runtime = (
            runtime_factory(model_path, reranker_path, batch_size=batch_size) if sources else None
        )
        load_seconds = time.monotonic() - load_started
        build_started = time.monotonic()
        built = build_index(generation, sources, runtime, qdrant)
        build_seconds = time.monotonic() - build_started
        await save_built_index(database, generation, built)
        lexical = built.manifest["lexical_statistics"]
        statistics = LexicalStatistics(
            tuple(lexical["vocabulary"]),
            tuple(lexical["document_frequency"]),
            lexical["document_count"],
            lexical["average_length"],
        )
        bm25_evidence = qdrant.verify_bm25(
            generation.collection_name,
            built.points,
            [chunk["retrieval_text"] for chunk in built.chunks],
            statistics,
            "graph sentiment graph",
        )

        async def verify_sources():
            async with database.sessions() as session:
                verified = await read_fixed_sources(
                    session, library_id, store, max_json_bytes=max_json_bytes
                )
            if [
                (item.paper_id, item.document_version_id, item.parse_snapshot_id)
                for item in verified
            ] != [
                (item.paper_id, item.document_version_id, item.parse_snapshot_id)
                for item in sources
            ]:
                raise ValueError("fixed_source_changed")

        runtime_summary = {
            "device": runtime.device if runtime else "not_loaded",
            "dtype": str(runtime.dtype) if runtime else None,
            "model_identity": runtime.identity if runtime else {},
        }
        evidence = await publish_generation(database, generation, built, qdrant, verify_sources)
        return {
            "status": "passed",
            "library_id": str(library_id),
            "generation_id": str(generation.generation_id),
            "attempt_id": str(generation.attempt_id),
            "collection_name": generation.collection_name,
            "membership_revision": generation.membership_revision,
            "manifest_sha256": built.manifest_sha256,
            "paper_count": len(sources),
            "chunk_count": len(built.chunks),
            **runtime_summary,
            "batch_size": batch_size,
            "source_seconds": round(read_seconds, 3),
            "model_load_seconds": round(load_seconds, 3),
            "build_seconds": round(build_seconds, 3),
            "total_seconds": round(time.monotonic() - started, 3),
            "validation": evidence,
            "bm25_reference": bm25_evidence,
        }
    except Exception as error:
        candidate = str(error)
        code = (
            candidate if re.fullmatch(r"[a-z][a-z0-9_]{0,127}", candidate) else type(error).__name__
        )
        try:
            await fail_generation(database, generation, code[:128])
        except Exception:
            # 数据库不可用时不掩盖原故障；本轮不提供自动恢复，构建指针保留供诊断。
            error.add_note("generation_failure_record_unavailable")
        raise
