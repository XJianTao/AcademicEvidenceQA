"""模块名称：固定范围混合候选检索。

用途：固定发布代后复用冻结 BM25、dense、RRF 与真实重排，并核验候选来源。
核心设计：不追随中途发布指针；同请求按快照核验一次，无跨请求来源缓存。
工程要点：所有 payload 与数据库完整映射比较，来源错误永远硬失败。
"""

import asyncio
import json
import time
from uuid import UUID

from sqlalchemy import select
from qdrant_client import models

from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.contracts.rag import Candidate, FixedScope, RetrievalResult
from academic_evidence_qa.documents.source_reader import verify_snapshot
from academic_evidence_qa.infrastructure.database.models import (
    Library,
    LibraryPaper,
    IndexGeneration,
    IndexChunk,
    DocumentVersion,
    ParseSnapshotRecord,
)
from academic_evidence_qa.ingestion.index_builder import content_hash
from academic_evidence_qa.retrieval.lexical import LexicalStatistics, query_vector
from academic_evidence_qa.retrieval.reranking import rerank
from academic_evidence_qa.retrieval.budget_diagnostics import EvidenceBudgetDiagnostics


def reciprocal_rank_fusion(rankings, *, constant=60):
    """名称：RRF。用途：每路去重后按名次融合，稳定 ID 解决同分。

    Args:
        rankings: 有序 chunk ID 列表的列表。
        constant: 正的名次平滑常量。
    Returns:
        (ID, 融合分数) 的稳定降序列表。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, identity in enumerate(dict.fromkeys(ranking), 1):
            scores[identity] = scores.get(identity, 0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def scope_filter(scope, recall_paper_ids=None):
    """两路共享成员三元组；内部定向子集不改变对外完整固定 scope。"""
    targets = None if recall_paper_ids is None else set(recall_paper_ids)
    members = [m for m in scope.members if targets is None or UUID(m["paper_id"]) in targets]
    if targets is not None and (not targets or len(members) != len(targets)):
        raise DocumentError("supplement_target_out_of_scope", 409)
    return models.Filter(
        must=[
            models.FieldCondition(
                key="library_id", match=models.MatchValue(value=str(scope.library_id))
            ),
            models.FieldCondition(
                key="generation_id", match=models.MatchValue(value=str(scope.generation_id))
            ),
            models.Filter(
                should=[
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key=key, match=models.MatchValue(value=member[key])
                            )
                            for key in ("paper_id", "document_version_id", "parse_snapshot_id")
                        ]
                    )
                    for member in members
                ]
            ),
        ]
    )


class RetrievalPipeline:
    """名称：候选检索服务。用途：组装唯一正式检索流程供 API 和验证 CLI 调用。

    Args:
        database, store, qdrant: 正式存储，不修改已发布数据。
        runtime, inference, io: API owner 及独立有界执行器。
        max_json_bytes: 单快照解析预算。
    Returns:
        通过 retrieve 返回固定 scope 的候选。
    """

    def __init__(self, database, store, qdrant, runtime, inference, io, *, max_json_bytes):
        self.database, self.store, self.qdrant = database, store, qdrant
        self.runtime, self.inference, self.io = runtime, inference, io
        self.max_json_bytes = max_json_bytes

    async def _verify_active_members(self, scope):
        """名称：复核固定成员范围。用途：拒绝已提交软移除，不追随发布切换。

        Args:
            scope: 本次请求已固定的论文范围和 generation。
        Returns:
            None；范围内论文均仍为 active 时继续。
        Raises:
            DocumentError: 范围内任一论文已移除，返回 scope_membership_changed。
        工程要点：独立短读会话，无行锁，不跨模型推理持有事务。
        """
        async with self.database.sessions() as session:
            current = set(
                (
                    await session.scalars(
                        select(LibraryPaper.paper_id).where(
                            LibraryPaper.library_id == scope.library_id,
                            LibraryPaper.status == "active",
                        )
                    )
                ).all()
            )
        if not {UUID(member["paper_id"]) for member in scope.members}.issubset(current):
            raise DocumentError("scope_membership_changed", 409)

    async def retrieve(
        self,
        request,
        *,
        source_cache=None,
        recall_paper_ids=None,
        diagnostics: EvidenceBudgetDiagnostics | None = None,
    ):
        """名称：固定范围检索。用途：有限期限内检索，可内部限定召回子集。

        Args:
            request: 原完整范围和固定代；定向补检索不得缩小此范围。
            source_cache: 请求内已核验快照。
            recall_paper_ids: 仅内部使用的非空论文子集，两路召回及来源拒绝均限制于此。
            diagnostics: 可选离线数值计量；不改变检索或响应契约。
        Returns:
            完整 scope 下的候选；全部成员仍参与返回前 active 复核。
        """
        try:
            async with asyncio.timeout(360):
                return await self._retrieve(
                    request,
                    source_cache=source_cache,
                    recall_paper_ids=recall_paper_ids,
                    diagnostics=diagnostics,
                )
        except TimeoutError as error:
            raise DocumentError("retrieval_timeout", 503) from error

    async def _retrieve(
        self, request, *, source_cache=None, recall_paper_ids=None, diagnostics=None
    ):
        started = time.monotonic()
        statuses, timings = [], {}
        async with self.database.sessions() as session:
            library = await session.get(Library, request.library_id)
            if library is None or not library.active_generation_id:
                raise DocumentError("published_generation_missing", 409)
            generation = await session.get(
                IndexGeneration, request.generation_id or library.active_generation_id
            )
            if (
                generation is None
                or generation.library_id != request.library_id
                or generation.state not in ("published", "retired")
            ):
                raise DocumentError("generation_out_of_scope", 404)
            manifest = generation.manifest
            if (
                content_hash(manifest) != generation.manifest_sha256
                or manifest.get("collection_name") != generation.collection_name
                or manifest.get("library_id") != str(generation.library_id)
                or manifest.get("generation_id") != str(generation.generation_id)
                or manifest.get("attempt_id") != str(generation.attempt_id)
                or manifest.get("membership_revision") != generation.membership_revision
            ):
                raise DocumentError("generation_integrity_failed", 409)
            if manifest["model_identity"] != self.runtime.identity:
                raise DocumentError("embedding_identity_mismatch", 409)
            active = set(
                (
                    await session.scalars(
                        select(LibraryPaper.paper_id).where(
                            LibraryPaper.library_id == request.library_id,
                            LibraryPaper.status == "active",
                        )
                    )
                ).all()
            )
            members = [m for m in manifest["members"] if UUID(m["paper_id"]) in active]
            wanted = set(request.paper_ids)
            if wanted and not wanted.issubset({UUID(m["paper_id"]) for m in members}):
                raise DocumentError("paper_out_of_scope", 404)
            members = [m for m in members if not wanted or UUID(m["paper_id"]) in wanted]
            scope = FixedScope(
                library_id=request.library_id,
                generation_id=generation.generation_id,
                membership_revision=generation.membership_revision,
                collection_name=generation.collection_name,
                members=members,
            )
            rows = list(
                (
                    await session.scalars(
                        select(IndexChunk)
                        .where(IndexChunk.generation_id == generation.generation_id)
                        .order_by(IndexChunk.reading_order)
                    )
                ).all()
            )
            chunks = json.loads(
                json.dumps(
                    [
                        {key: getattr(row, key) for key in IndexChunk.__table__.columns.keys()}
                        for row in rows
                    ],
                    default=str,
                )
            )
            if (
                content_hash(chunks) != manifest["chunks_sha256"]
                or [c["chunk_id"] for c in chunks] != manifest["chunk_ids"]
            ):
                raise DocumentError("generation_integrity_failed", 409)
            lexical = manifest["lexical_statistics"]
            statistics = LexicalStatistics(
                tuple(lexical["vocabulary"]),
                tuple(lexical["document_frequency"]),
                lexical["document_count"],
                lexical["average_length"],
            )
            if statistics.sha256 != manifest["statistics_sha256"]:
                raise DocumentError("statistics_integrity_failed", 409)
            fixed_attempt = str(generation.attempt_id)
        timings["scope_seconds"] = time.monotonic() - started
        if not members:
            return RetrievalResult(
                scope=scope,
                candidates=[],
                statuses=["no_hits"],
                configuration={"mode": request.mode},
                timings=timings,
            )
        # Query 硬限制属于请求错误，不作为单路模型故障掩盖。
        lengths, _ = await self.inference.run(
            lambda: (
                self.runtime.embedding_length(request.query),
                len(
                    self.runtime.reranker_tokenizer(
                        request.query, add_special_tokens=False, truncation=False
                    )["input_ids"]
                ),
            )
        )
        if lengths[0] > 600 or (request.mode == "hybrid_rerank" and lengths[1] > 96):
            raise DocumentError("query_token_limit_exceeded", 422)
        vector = query_vector(request.query, statistics)
        query_filter = scope_filter(scope, recall_paper_ids)

        async def recall(route):
            try:
                if route == "dense":
                    encoded, timing = await self.inference.run(
                        lambda: self.runtime.encode([request.query])
                    )
                    timings.update({f"embedding_{k}": v for k, v in timing.items()})
                    query = encoded[0]
                else:
                    if not vector:
                        return []
                    query = models.SparseVector(
                        indices=sorted(vector), values=[vector[k] for k in sorted(vector)]
                    )
                result, timing = await self.io.run(
                    lambda: self.qdrant.client.query_points(
                        scope.collection_name,
                        using=route,
                        query=query,
                        query_filter=query_filter,
                        limit=request.recall_limit,
                        with_payload=True,
                    ).points
                )
                timings[f"{route}_seconds"] = sum(timing.values())
                return result
            except asyncio.CancelledError:
                raise
            except Exception:
                statuses.append(f"{route}_failed")
                return []

        dense, sparse = await asyncio.gather(
            recall("dense") if request.mode != "bm25" else asyncio.sleep(0, result=[]),
            recall("bm25") if request.mode != "dense" else asyncio.sleep(0, result=[]),
        )
        by_id = {c["chunk_id"]: c for c in chunks}
        identities = {
            (m["paper_id"], m["document_version_id"], m["parse_snapshot_id"])
            for m in members
            if recall_paper_ids is None or UUID(m["paper_id"]) in recall_paper_ids
        }
        for hit in dense + sparse:
            chunk = by_id.get(str(hit.id))
            if (
                chunk is None
                or (chunk["paper_id"], chunk["document_version_id"], chunk["parse_snapshot_id"])
                not in identities
            ):
                raise DocumentError("candidate_out_of_scope", 409)
            expected = {
                "library_id": str(scope.library_id),
                "generation_id": str(scope.generation_id),
                "attempt_id": fixed_attempt,
                "statistics_sha256": statistics.sha256,
                **{
                    k: chunk[k]
                    for k in (
                        "chunk_id",
                        "paper_id",
                        "document_version_id",
                        "parse_snapshot_id",
                        "block_type",
                        "source_ranges",
                        "source_locations",
                    )
                },
            }
            if hit.payload != expected:
                raise DocumentError("candidate_source_mismatch", 409)
        rankings = [[str(h.id) for h in route] for route in (dense, sparse)]
        fused = reciprocal_rank_fusion(rankings)[:60]
        if diagnostics is not None:
            diagnostics.record(
                "recall_counts",
                dense_count=len(dense),
                bm25_count=len(sparse),
                fused_count=len(fused),
            )
        scores_dense = {str(h.id): h.score for h in dense}
        scores_sparse = {str(h.id): h.score for h in sparse}
        ordered = (
            fused
            if request.mode not in ("dense", "bm25")
            else [(str(h.id), None) for h in (dense if request.mode == "dense" else sparse)]
        )
        candidates = [
            Candidate(
                **{k: by_id[identity][k] for k in Candidate.model_fields if k in by_id[identity]},
                dense_score=scores_dense.get(identity),
                bm25_score=scores_sparse.get(identity),
                rrf_score=score,
            )
            for identity, score in ordered
        ]
        if diagnostics is not None:
            diagnostics.record(
                "candidate_pool",
                [c.model_dump(mode="json") for c in candidates],
                item_count=len(candidates),
            )
        source_started = time.monotonic()
        async with self.database.sessions() as session:
            for snapshot_id in sorted({c.parse_snapshot_id for c in candidates}, key=str):
                candidate = next(c for c in candidates if c.parse_snapshot_id == snapshot_id)
                version = await session.get(DocumentVersion, candidate.document_version_id)
                snapshot = await session.get(ParseSnapshotRecord, snapshot_id)
                if version is None or snapshot is None or version.paper_id != candidate.paper_id:
                    raise DocumentError("candidate_source_mismatch", 409)
                try:
                    document = await verify_snapshot(
                        session,
                        version,
                        snapshot,
                        self.store,
                        max_json_bytes=self.max_json_bytes,
                        file_executor=self.io,
                    )
                    if source_cache is not None:
                        source_cache[snapshot_id] = document
                except Exception as error:
                    raise DocumentError("snapshot_integrity_failed", 409) from error
        await self._verify_active_members(scope)
        timings["source_seconds"] = time.monotonic() - source_started
        if request.mode == "hybrid_rerank" and candidates:
            try:
                candidates, rerank_status, timing = await rerank(
                    request.query, candidates, self.runtime, self.inference
                )
                statuses.extend(rerank_status)
                timings.update({f"rerank_{k}": v for k, v in timing.items()})
            except asyncio.CancelledError:
                raise
            except Exception:
                statuses.append("rerank_failed")
                for candidate in candidates:
                    candidate.windows, candidate.rerank_score = [], None
                    candidate.statuses = [s for s in candidate.statuses if s != "rerank_partial"]
        if not candidates:
            statuses.append("recall_unavailable" if statuses else "no_hits")
        # 重排可能耗时较长：成功和降级均须最终复核，且位于降级捕获边界之外。
        # 仅检查固定范围的 active 状态，不因正常发布或选版改变而混用新 generation。
        await self._verify_active_members(scope)
        timings["total_seconds"] = time.monotonic() - started
        if diagnostics is not None:
            diagnostics.record(
                "candidate_reranked",
                [c.model_dump(mode="json") for c in candidates],
                item_count=len(candidates),
            )
        return RetrievalResult(
            scope=scope,
            candidates=candidates[: request.limit],
            statuses=statuses,
            configuration={
                "mode": request.mode,
                "rrf_constant": 60,
                "recall_limit": request.recall_limit,
                "max_pair_tokens": 512,
                "max_parent_windows": 3,
                "max_pairs": 120,
                "model_identity": self.runtime.identity,
                "reranker_identity": self.runtime.reranker_identity,
            },
            timings=timings,
        )
