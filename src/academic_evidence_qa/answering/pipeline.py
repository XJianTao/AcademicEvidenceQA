"""模块名称：一次检索问答与持久交付。

用途：组装证据、一次生成、来源/引用校验及幂等保存，不引入任务平台。
核心设计：先短事务占 key，再在事务外检索/生成；未知结果保留屏障，不自动再次计费。
工程要点：完整实际请求预算复核；交付前检查完整 active scope，发布换代不影响固定版本。
"""

import asyncio
import hashlib
from pathlib import Path
import time
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from academic_evidence_qa.answering.citations import validate_citations
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.contracts.rag import Answer, EvidenceRequest
from academic_evidence_qa.documents.source_reader import load_evidence_pack, verify_evidence_sources
from academic_evidence_qa.infrastructure.database.models import AnswerArtifact
from academic_evidence_qa.infrastructure.generation_client import GenerationError, estimated_tokens
from academic_evidence_qa.ingestion.index_builder import content_hash
from academic_evidence_qa.retrieval.evidence_builder import measure_pack


class AnswerPipeline:
    """名称：固定问答服务。用途：共享取证与单例 SDK，非流式已校验交付。

    Args:
        evidence_builder: 唯一正式取证服务。
        generation: bootstrap 持有的客户端，可缺失但问答必须明确503。
        prompt_path: 版本化提示词，身份包含内容哈希。
    Returns:
        answer 返回保存的 Answer；未知、失败或处理中 key 不重新调用模型。
    """

    def __init__(self, evidence_builder, generation, *, prompt_path=None):
        self.builder, self.generation = evidence_builder, generation
        self.retrieval = evidence_builder.retrieval
        self.database = self.retrieval.database
        path = prompt_path or Path(__file__).resolve().parents[3] / "prompts/rag_answer_v1.md"
        self.prompt = path.read_text(encoding="utf-8")
        self.configuration = {
            "prompt_version": "rag_answer_v1",
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "model": generation.model if generation else None,
            "endpoint_sha256": hashlib.sha256(str(generation.sdk.base_url).encode()).hexdigest()
            if generation and hasattr(generation, "sdk")
            else None,
            "budget_mode": "estimated",
            "stream": False,
            "output_parameter": "max_completion_tokens",
            "max_retries": generation.max_retries if generation else 0,
            "context_tokens": generation.context_tokens if generation else None,
            "max_bytes": generation.max_bytes if generation else None,
        }

    def messages(self, query, pack):
        """发送完整来源标签/别名/封装，不在计量后追加隐藏上下文。"""
        import json

        return [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": query, "evidence_pack": pack.model_dump(mode="json")},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    async def _claim(self, request, key):
        if not key or len(key) > 128 or not key.strip() or key != key.strip():
            raise DocumentError("idempotency_key_required", 422)
        payload = request.model_dump(mode="json")
        payload["paper_ids"] = sorted(set(payload["paper_ids"]))
        digest, answer_id = content_hash(payload), uuid4()
        async with self.database.transaction() as session:
            claimed = await session.scalar(
                insert(AnswerArtifact)
                .values(
                    answer_id=answer_id,
                    idempotency_key=key,
                    request_sha256=digest,
                    library_id=request.library_id,
                    state="pending",
                    configuration=self.configuration,
                )
                .on_conflict_do_nothing(index_elements=[AnswerArtifact.idempotency_key])
                .returning(AnswerArtifact.answer_id)
            )
            if claimed:
                return answer_id, None
            existing = await session.scalar(
                select(AnswerArtifact).where(AnswerArtifact.idempotency_key == key)
            )
            if existing.request_sha256 != digest:
                raise DocumentError("idempotency_conflict", 409)
            if existing.state == "completed":
                return existing.answer_id, Answer.model_validate(existing.response)
            raise DocumentError(
                "answer_result_unknown"
                if existing.state == "unknown"
                else "answer_request_in_progress"
                if existing.state == "pending"
                else existing.error_code or "answer_request_failed",
                409,
            )

    async def _save(
        self,
        answer_id,
        *,
        result=None,
        draft=None,
        error_code=None,
        state="completed",
        retrieval_run_id=None,
        usage=None,
    ):
        async with self.database.transaction() as session:
            row = await session.get(AnswerArtifact, answer_id, with_for_update=True)
            row.state, row.draft, row.error_code = state, draft, error_code
            if retrieval_run_id:
                row.retrieval_run_id = retrieval_run_id
            row.response = result.model_dump(mode="json") if result else {"usage": usage or {}}

    async def answer(self, request, key):
        """名称：一次检索生成。用途：先证据预算再调用，取消/未知不自动重发。

        Args:
            request: 问题、固定范围和预算。
            key: 客户端稳定 Idempotency-Key，相同请求重放不重复计费。
        Returns:
            已持久 Answer；校验失败仅状态与诊断，原草稿留在持久记录。
        Raises:
            DocumentError: 服务不可用、坏源或生成期间范围移除，不能包装无答案。
        """
        answer_id, replay = await self._claim(request, key)
        if replay:
            pack = await load_evidence_pack(
                self.database, request.library_id, replay.retrieval_run_id
            )
            await self.retrieval._verify_active_members(pack.scope)
            await verify_evidence_sources(
                self.database,
                self.retrieval.store,
                pack,
                max_json_bytes=self.retrieval.max_json_bytes,
                file_executor=self.retrieval.io,
            )
            await self.retrieval._verify_active_members(pack.scope)
            return replay
        started, pack, draft, called = time.monotonic(), None, None, False
        accounted_usage = {}
        cache = {}
        try:
            if self.generation is None:
                raise DocumentError("generation_unconfigured", 503)
            # 整个同步请求有界，取证自有360秒期限；不跨任何外部调用持有锁。
            async with asyncio.timeout(360 + self.generation.timeout_seconds):
                pack = await self.builder.retrieve(
                    EvidenceRequest.model_validate(
                        request.model_dump(exclude={"max_output_tokens"})
                    ),
                    source_cache=cache,
                )
                result = Answer(
                    answer_id=answer_id,
                    retrieval_run_id=pack.retrieval_run_id,
                    status="insufficient_evidence",
                    timings=pack.timings.copy(),
                )
                if pack.status == "ready":
                    while pack.evidence and not self.generation.fits(
                        self.messages(request.query, pack), request.max_output_tokens
                    ):
                        pack.evidence.pop()
                        pack.truncated = True
                        pack.uncovered_paper_ids = [
                            UUID(m["paper_id"])
                            for m in pack.scope.members
                            if UUID(m["paper_id"]) not in {e.paper_id for e in pack.evidence}
                        ]
                        pack.status = "ready" if pack.evidence else "budget_exhausted"
                        measure_pack(pack)
                    # 选择改变必须保留原取证记录，另建最终实际封装及 registry。
                    stored = await load_evidence_pack(
                        self.database, request.library_id, pack.retrieval_run_id
                    )
                    if pack != stored:
                        pack.retrieval_run_id = uuid4()
                        # registry 的主键全局唯一；新封装不能复用另一 run 的证据身份。
                        for evidence in pack.evidence:
                            evidence.evidence_id = uuid4()
                        measure_pack(pack)
                        await self.builder.save(pack, request)
                    result.retrieval_run_id = pack.retrieval_run_id
                    if pack.evidence and self.generation.fits(
                        self.messages(request.query, pack), request.max_output_tokens
                    ):
                        await self.retrieval._verify_active_members(pack.scope)
                        # 先关联取证记录，进程意外退出仍可诊断 pending 屏障，不自动恢复付费请求。
                        await self._save(
                            answer_id, state="pending", retrieval_run_id=pack.retrieval_run_id
                        )
                        called = True
                        accounted_usage = {
                            "status": "unknown",
                            "accounted_tokens": (
                                estimated_tokens(
                                    self.generation.request_body(
                                        self.messages(request.query, pack),
                                        request.max_output_tokens,
                                    )
                                )
                                + request.max_output_tokens
                            )
                            * (self.generation.max_retries + 1),
                        }
                        generated = await self.generation.generate(
                            self.messages(request.query, pack), request.max_output_tokens
                        )
                        draft = generated.text
                        accounted_usage = generated.usage
                        result.usage, result.timings["generation_seconds"] = (
                            generated.usage,
                            generated.seconds,
                        )
                        if draft.strip() == "INSUFFICIENT_EVIDENCE":
                            result.status = "insufficient_evidence"
                        else:
                            await verify_evidence_sources(
                                self.database,
                                self.retrieval.store,
                                pack,
                                max_json_bytes=self.retrieval.max_json_bytes,
                                file_executor=self.retrieval.io,
                                cache=cache,
                            )
                            result.citations, result.diagnostics = validate_citations(draft, pack)
                            result.citation_check_status = (
                                "failed" if result.diagnostics else "passed"
                            )
                            result.status = "citation_failed" if result.diagnostics else "answered"
                            result.text = None if result.diagnostics else draft
                    else:
                        result.status = "budget_exhausted"
                elif pack.status == "budget_exhausted":
                    result.status = "budget_exhausted"
                await self.retrieval._verify_active_members(pack.scope)
                result.timings["answer_total_seconds"] = time.monotonic() - started
                await self._save(
                    answer_id, result=result, draft=draft, retrieval_run_id=pack.retrieval_run_id
                )
                await self.retrieval._verify_active_members(pack.scope)
                return result
        except BaseException as error:
            code = (
                error.code
                if isinstance(error, GenerationError)
                else error.error_code
                if isinstance(error, DocumentError)
                else "answer_cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "answer_timeout"
                if isinstance(error, TimeoutError)
                else "answer_failed"
            )
            usage = (
                {
                    "status": "unknown",
                    "accounted_tokens": error.reserved_tokens,
                    "attempts": error.attempts,
                }
                if isinstance(error, GenerationError)
                else accounted_usage
            )
            await asyncio.shield(
                self._save(
                    answer_id,
                    state="unknown" if called else "failed",
                    draft=draft,
                    error_code=code,
                    retrieval_run_id=pack.retrieval_run_id if pack else None,
                    usage=usage,
                )
            )
            if isinstance(error, GenerationError):
                raise DocumentError(code, 504 if code == "generation_timeout" else 503) from None
            if isinstance(error, TimeoutError):
                raise DocumentError(code, 504) from None
            raise
