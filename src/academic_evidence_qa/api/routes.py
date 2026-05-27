"""模块名称：管理与固定来源路由。

用途：将严格 HTTP 契约转换到正式服务，不承载业务或后台任务。
核心设计：流式接收/受限预检在事务外，受理由持久服务原子完成。
工程要点：不提供路径输入、不加载模型、不打印第三方消息。
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
    )


@router.post("/evidence/read", response_model=SourceResponse, response_model_exclude_unset=True)
async def read_source(body: SourceRequest, request: Request):
    state = request.app.state
    return await read_scoped_source(
        state.database, state.store, body, max_json_bytes=state.settings.result_max_json_bytes
    )
