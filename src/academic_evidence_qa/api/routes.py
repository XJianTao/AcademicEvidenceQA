"""模块名称：管理与固定来源路由。

用途：将严格 HTTP 契约转换到正式服务，不承载业务或后台任务。
核心设计：流式接收/受限预检在事务外，受理由持久服务原子完成。
工程要点：不提供路径输入，模型由生命周期持有；不打印第三方消息。
"""

import asyncio
import re
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Request, Header, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from academic_evidence_qa.api.multipart import receive_management_upload
from academic_evidence_qa.api.schemas import (
    LibraryRequest,
    UploadRequest,
    MemberRequest,
    RevisionRequest,
    SourceRequest,
    JobResponse,
    LibraryResponse,
    UploadResponse,
    MemberResponse,
    JobMutationResponse,
    PaperListResponse,
    SourceResponse,
)
from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.documents.source_reader import read_scoped_source
from academic_evidence_qa.documents.upload import preflight_and_store_pdf, _remove_upload_directory

router = APIRouter(prefix="/v1")
internal_router = APIRouter(prefix="/internal")
retrieval_router = APIRouter(prefix="/v1")


async def bounded_json(request, limit):
    """逐段统计实际字节，在 JSON/Pydantic 解码前拒绝过大的请求。"""
    import json

    data = bytearray()
    async for part in request.stream():
        if len(data) + len(part) > limit:
            raise DocumentError("request_body_too_large", 413)
        data.extend(part)
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("object_required")
        return value
    except (ValueError, UnicodeError) as error:
        raise DocumentError("request_invalid", 422) from error


def require_runtime(request, *, internal=False):
    if internal and (request.client is None or request.client.host not in {"127.0.0.1", "::1"}):
        raise DocumentError("internal_endpoint_forbidden", 403)
    if request.app.state.retrieval is None:
        raise DocumentError("model_owner_unavailable", 503)
    return request.app.state


@internal_router.get("/models")
async def model_metadata(request: Request):
    import os

    state = require_runtime(request, internal=True)
    return {
        "identity": state.runtime.identity,
        "reranker_identity": state.runtime.reranker_identity,
        "device": state.runtime.device,
        "dtype": str(state.runtime.dtype),
        "model_load_seconds": state.model_load_seconds,
        "process_id": os.getpid(),
        "inference_active": state.inference.active,
        "inference_queued": len(state.inference.queue),
    }


@internal_router.post("/encode")
async def encode_batch(request: Request):
    """名称：受限内部编码。用途：Worker 复用 owner；身份不匹配不进入推理。

    Args:
        request: loopback 小 JSON，最多四条且每条不超过 600 tokens。
    Returns:
        向量、原配置身份与排队/推理耗时，不建立公共推理服务。
    """
    from academic_evidence_qa.api.schemas import EncodingRequest

    state = require_runtime(request, internal=True)
    try:
        body = EncodingRequest.model_validate(await bounded_json(request, 64 * 1024))
    except ValidationError as error:
        raise DocumentError("request_invalid", 422) from error
    if body.configuration_sha256 != state.runtime.identity["configuration_sha256"]:
        raise DocumentError("embedding_identity_mismatch", 409)
    if any(not text or len(text) > 16000 for text in body.texts):
        raise DocumentError("embedding_text_invalid", 422)

    def encode():
        if any(state.runtime.embedding_length(t) > 600 for t in body.texts):
            raise DocumentError("embedding_token_limit_exceeded", 422)
        return state.runtime.encode(body.texts)

    vectors, timing = await state.inference.run(encode)
    return {"identity": state.runtime.identity, "vectors": vectors, "timings": timing}


@retrieval_router.post("/retrieval/candidates")
async def retrieve_candidates(request: Request):
    """临时且具名的 Candidate 契约；最终 /retrieve EvidencePack 属于 M2a-2。"""
    from academic_evidence_qa.contracts.rag import RetrievalRequest

    state = require_runtime(request)
    try:
        body = RetrievalRequest.model_validate(await bounded_json(request, 24 * 1024))
    except ValidationError as error:
        raise DocumentError("request_invalid", 422) from error
    return await state.retrieval.retrieve(body)


def response(result, schema):
    """返回服务白名单响应及原始幂等状态码。"""
    status, body = result
    return JSONResponse(
        schema.model_validate(body).model_dump(mode="json", exclude_unset=True), status_code=status
    )


@router.post("/libraries", response_model=LibraryResponse, status_code=201)
async def create_library(
    body: LibraryRequest, request: Request, idempotency_key: str | None = Header(default=None)
):
    return response(
        await request.app.state.management.create_library(idempotency_key), LibraryResponse
    )


@router.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(request: Request, idempotency_key: str | None = Header(default=None)):
    """名称：流式上传受理。用途：可信登记后立即返回，解析由独立 Worker 完成。

    Args:
        request: 原始 multipart 流；不使用框架 spool。
        idempotency_key: 持久幂等键。
    Returns:
        202 受理或 200 复用；拒绝时不创建任务。
    """
    resources = request.app.state
    if not idempotency_key or not re.fullmatch(r"[\x21-\x7e]{1,128}", idempotency_key):
        raise DocumentError("idempotency_key_required", 422)
    received, fields = await receive_management_upload(
        request, resources.store, resources.settings.pdf_max_upload_bytes
    )
    try:
        try:
            body = UploadRequest.model_validate(fields)
        except ValidationError as error:
            raise DocumentError("request_invalid", 422) from error
        if body.parser_backend == "docling":
            raise DocumentError("docling_unavailable", 503)
        if not body.allow_remote_processing:
            raise DocumentError("remote_processing_not_allowed", 422)
        stored, _ = await preflight_and_store_pdf(
            received,
            file_store=resources.store,
            max_pages=resources.settings.pdf_max_pages,
            timeout_seconds=resources.settings.pdf_preflight_timeout_seconds,
            memory_mib=resources.settings.pdf_preflight_memory_mib,
        )
        return response(
            await resources.management.accept_upload(idempotency_key, body, stored), UploadResponse
        )
    finally:
        await asyncio.to_thread(_remove_upload_directory, Path(received.staging_path).parent)


@router.get(
    "/libraries/{library_id}/papers",
    response_model=PaperListResponse,
    response_model_exclude_unset=True,
)
async def library_papers(
    library_id: UUID,
    request: Request,
    offset: int = Query(0, ge=0, le=100000),
    limit: int = Query(20, ge=1, le=100),
):
    return await request.app.state.management.list_papers(
        library_id=library_id, offset=offset, limit=limit
    )


@router.put("/libraries/{library_id}/papers/{paper_id}", response_model=MemberResponse)
async def set_member(
    library_id: UUID,
    paper_id: UUID,
    body: MemberRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None),
):
    return response(
        await request.app.state.management.member_change(
            idempotency_key, library_id, paper_id, body
        ),
        MemberResponse,
    )


@router.delete("/libraries/{library_id}/papers/{paper_id}", response_model=MemberResponse)
async def remove_member(
    library_id: UUID,
    paper_id: UUID,
    body: RevisionRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None),
):
    return response(
        await request.app.state.management.member_change(
            idempotency_key, library_id, paper_id, body, remove=True
        ),
        MemberResponse,
    )


@router.post("/libraries/{library_id}/reindex", response_model=JobMutationResponse, status_code=202)
async def reindex(
    library_id: UUID,
    body: RevisionRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None),
):
    return response(
        await request.app.state.management.reindex(idempotency_key, library_id, body),
        JobMutationResponse,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def read_job(job_id: UUID, request: Request):
    return await request.app.state.management.read_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobMutationResponse)
async def cancel_job(
    job_id: UUID, request: Request, idempotency_key: str | None = Header(default=None)
):
    return response(
        await request.app.state.management.mutate_job(idempotency_key, job_id), JobMutationResponse
    )


@router.post("/jobs/{job_id}/retry", response_model=JobMutationResponse, status_code=202)
async def retry_job(
    job_id: UUID, request: Request, idempotency_key: str | None = Header(default=None)
):
    return response(
        await request.app.state.management.mutate_job(idempotency_key, job_id, retry=True),
        JobMutationResponse,
    )


@router.get("/papers", response_model=PaperListResponse, response_model_exclude_unset=True)
async def papers(
    request: Request, offset: int = Query(0, ge=0, le=100000), limit: int = Query(20, ge=1, le=100)
):
    return await request.app.state.management.list_papers(offset=offset, limit=limit)


@router.get(
    "/papers/{paper_id}/outline", response_model=SourceResponse, response_model_exclude_unset=True
)
async def outline(
    paper_id: UUID,
    request: Request,
    library_id: UUID,
    document_version_id: UUID,
    parse_snapshot_id: UUID,
    generation_id: UUID | None = None,
    preview: bool = False,
    offset: int = Query(0, ge=0, le=100000),
    limit: int = Query(20, ge=1, le=100),
):
    body = SourceRequest(
        library_id=library_id,
        paper_id=paper_id,
        document_version_id=document_version_id,
        parse_snapshot_id=parse_snapshot_id,
        generation_id=generation_id,
        preview=preview,
        offset=offset,
        limit=limit,
    )
    state = request.app.state
    return await read_scoped_source(
        state.database,
        state.store,
        body,
        outline=True,
        max_json_bytes=state.settings.result_max_json_bytes,
        file_executor=getattr(state, "source_io", None),
    )


@router.post("/evidence/read")
async def read_source(request: Request):
    from academic_evidence_qa.contracts.rag import EvidenceReadRequest
    from academic_evidence_qa.documents.source_reader import read_registered_evidence

    state = request.app.state
    raw = await bounded_json(request, 16 * 1024)
    try:
        body = (EvidenceReadRequest if "evidence_id" in raw else SourceRequest).model_validate(raw)
    except ValidationError as error:
        raise DocumentError("request_invalid", 422) from error
    if isinstance(body, EvidenceReadRequest):
        return await read_registered_evidence(
            state.database,
            state.store,
            body,
            max_json_bytes=state.settings.result_max_json_bytes,
            file_executor=getattr(state, "source_io", None),
        )
    return await read_scoped_source(
        state.database,
        state.store,
        body,
        max_json_bytes=state.settings.result_max_json_bytes,
        file_executor=getattr(state, "source_io", None),
    )


@retrieval_router.post("/retrieve")
async def retrieve_evidence(request: Request):
    from academic_evidence_qa.contracts.rag import EvidenceRequest

    state = require_runtime(request)
    try:
        body = EvidenceRequest.model_validate(await bounded_json(request, 24 * 1024))
    except ValidationError as error:
        raise DocumentError("request_invalid", 422) from error
    return await state.evidence.retrieve(body)


@retrieval_router.post("/answer")
async def answer_question(request: Request, idempotency_key: str | None = Header(default=None)):
    from academic_evidence_qa.contracts.rag import AnswerRequest

    state = require_runtime(request)
    try:
        body = AnswerRequest.model_validate(await bounded_json(request, 24 * 1024))
    except ValidationError as error:
        raise DocumentError("request_invalid", 422) from error
    return await state.answering.answer(body, idempotency_key)


@retrieval_router.get("/answers/{answer_id}")
async def saved_answer(answer_id: UUID, request: Request, library_id: UUID):
    # 历史回读无需模型 owner 或生成配置。
    from academic_evidence_qa.infrastructure.database.models import AnswerArtifact

    async with request.app.state.database.sessions() as session:
        row = await session.get(AnswerArtifact, answer_id)
        if row is None or row.library_id != library_id:
            raise DocumentError("answer_not_found", 404)
        if row.state != "completed":
            return {"answer_id": answer_id, "state": row.state, "error_code": row.error_code}
        return {**row.response, "historical": True}
