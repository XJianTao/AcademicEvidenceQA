"""模块：有界 PDF 接收与可信登记。

用途概要：从内容字节流写入服务生成的 staging 目录、逐块限额/哈希，并在受限预检后交给短事务登记。

核心设计：上传流的大小限制早于任何完整文件解析；外部调用者不能提供磁盘路径或对象键。

工程要点：断连和拒绝清理未受理暂存，成功后原件可因数据库回滚而保留为可验证残留。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import uuid4

from academic_evidence_qa.contracts.documents import DocumentError, PdfPreflightResult, ReceivedPdf
from academic_evidence_qa.documents.parser import preflight_pdf
from academic_evidence_qa.infrastructure.local_file_store import LocalFileStore, StoredObject


async def receive_pdf_stream(
    chunks: AsyncIterable[bytes],
    *,
    file_store: LocalFileStore,
    max_upload_bytes: int,
    max_chunk_bytes: int = 1024 * 1024,
) -> ReceivedPdf:
    """有界写入单个 PDF 内容流。

    Args:
        chunks: 测试或未来 HTTP 适配层提供的未经信任字节流。
        file_store: 服务自有 staging 根目录。
        max_upload_bytes, max_chunk_bytes: 实际写入前强制执行的累计和单块上限。
    Returns:
        含 SHA-256 与实际字节数的服务暂存对象；尚不能登记为版本。
    Raises:
        DocumentError: 超限为 413，类型/流错误为 422；任何失败都会删除本次暂存目录。
    """
    if max_upload_bytes <= 0 or max_chunk_bytes <= 0:
        raise ValueError("上传限额必须为正数")
    upload_id = uuid4()
    directory = file_store.staging / upload_id.hex
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "upload.pdf"
    digest = hashlib.sha256()
    size_bytes = 0
    first_bytes = bytearray()
    try:
        with path.open("xb") as output:
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise DocumentError("invalid_upload_stream", 422)
                if len(chunk) > max_chunk_bytes:
                    raise DocumentError("upload_chunk_too_large", 413)
                size_bytes += len(chunk)
                if size_bytes > max_upload_bytes:
                    raise DocumentError("upload_size_exceeded", 413)
                if len(first_bytes) < 8:
                    first_bytes.extend(chunk[: 8 - len(first_bytes)])
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if not first_bytes.startswith(b"%PDF-"):
            raise DocumentError("pdf_type_invalid", 422)
        return ReceivedPdf(upload_id, str(path), size_bytes, digest.hexdigest())
    except BaseException:
        await asyncio.to_thread(_remove_upload_directory, directory)
        raise


async def receive_single_pdf_multipart(
    body_chunks: AsyncIterable[bytes],
    *,
    content_type: str,
    file_store: LocalFileStore,
    max_upload_bytes: int,
    max_chunk_bytes: int = 1024 * 1024,
    max_multipart_overhead_bytes: int = 64 * 1024,
) -> ReceivedPdf:
    """增量接收一个 multipart PDF。

    该适配器仅供 M1-B 验证入口和未来 HTTP 边界复用。它在解析边界时仅保留
    必要尾部，文件内容随即交给 ``receive_pdf_stream`` 计数并落盘，因此不会先
    由框架无上限地 spooling。``Content-Length`` 不参与限额判断，缺失或伪造时
    仍以实际读取字节为准。

    Args:
        body_chunks: HTTP body 的原始分段字节流。
        content_type: 含 boundary 参数的 multipart/form-data Content-Type。
        file_store: 服务自有暂存根目录。
        max_upload_bytes: PDF 内容的实际写入上限。
        max_chunk_bytes: 单个写入块上限。
        max_multipart_overhead_bytes: 首部和边界允许的总开销上限。
    Returns:
        与纯内容流路径完全相同的受限暂存结果。
    Raises:
        DocumentError: multipart 结构、字段数或流内容不符合单文件契约。
    """
    boundary = _multipart_boundary(content_type)
    return await receive_pdf_stream(
        _iter_multipart_pdf_content(
            body_chunks,
            boundary=boundary,
            max_overhead_bytes=max_multipart_overhead_bytes,
        ),
        file_store=file_store,
        max_upload_bytes=max_upload_bytes,
        max_chunk_bytes=max_chunk_bytes,
    )


async def preflight_and_store_pdf(
    received: ReceivedPdf,
    *,
    file_store: LocalFileStore,
    max_pages: int,
    timeout_seconds: int,
    memory_mib: int,
) -> tuple[StoredObject, PdfPreflightResult]:
    """执行受限预检并原子保存原件。

    Args:
        received: 只能来自 receive_pdf_stream 的服务暂存结果。
        file_store: 同一服务的可信存储。
        max_pages, timeout_seconds, memory_mib: 严格预检资源限制。
    Returns:
        已保存原件与预检页数；调用者随后才能开启短数据库事务。
    """
    path = Path(received.staging_path)
    expected_staging_root = file_store.staging.resolve()
    try:
        path.resolve().relative_to(expected_staging_root)
    except (OSError, ValueError) as error:
        raise DocumentError("untrusted_staging_path", 422) from error
    if path.name != "upload.pdf" or not path.is_file():
        raise DocumentError("untrusted_staging_path", 422)
    try:
        preflight = await asyncio.to_thread(
            preflight_pdf,
            path,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            memory_mib=memory_mib,
        )
        stored = await asyncio.to_thread(
            file_store.write_verified_staged,
            path,
            expected_sha256=received.sha256,
            expected_size_bytes=received.size_bytes,
        )
        return stored, preflight
    finally:
        # 成功后只删除本次 staging；FileStore 已以硬链接发布原件，数据库失败残留不删除。
        await asyncio.to_thread(_remove_upload_directory, path.parent)


def _remove_upload_directory(directory: Path) -> None:
    if not directory.is_dir() or directory.parent.name != "staging":
        return
    for child in directory.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
    directory.rmdir()


def _multipart_boundary(content_type: str) -> bytes:
    """读取受限 multipart boundary，拒绝缺失或不安全的参数。"""
    media_type, *parameters = [part.strip() for part in content_type.split(";")]
    if media_type.lower() != "multipart/form-data":
        raise DocumentError("multipart_content_type_invalid", 422)
    for parameter in parameters:
        name, separator, value = parameter.partition("=")
        if name.lower() == "boundary" and separator:
            boundary = value.strip().strip('"').encode("ascii", "strict")
            if 1 <= len(boundary) <= 70 and all(32 <= byte < 127 for byte in boundary):
                return boundary
    raise DocumentError("multipart_boundary_invalid", 422)


async def _iter_multipart_pdf_content(
    body_chunks: AsyncIterable[bytes], *, boundary: bytes, max_overhead_bytes: int
) -> AsyncIterable[bytes]:
    """从单个 multipart 文件分段产出内容，不接受普通字段或多个文件。"""
    if max_overhead_bytes <= 0:
        raise ValueError("multipart 开销限额必须为正数")
    opening = b"--" + boundary + b"\r\n"
    marker = b"\r\n--" + boundary
    buffer = bytearray()
    overhead_bytes = 0
    opening_consumed = False
    headers_read = False
    completed = False
    async for incoming in body_chunks:
        if not isinstance(incoming, bytes):
            raise DocumentError("invalid_upload_stream", 422)
        buffer.extend(incoming)
        if not opening_consumed:
            if len(buffer) > max_overhead_bytes:
                raise DocumentError("multipart_overhead_exceeded", 413)
            if len(buffer) < len(opening):
                continue
            if not buffer.startswith(opening):
                raise DocumentError("multipart_structure_invalid", 422)
            del buffer[: len(opening)]
            opening_consumed = True
            overhead_bytes += len(opening)
        if not headers_read:
            if len(buffer) + overhead_bytes > max_overhead_bytes:
                raise DocumentError("multipart_overhead_exceeded", 413)
            separator = buffer.find(b"\r\n\r\n")
            if separator < 0:
                continue
            header_block = bytes(buffer[:separator])
            overhead_bytes += separator + 4
            if overhead_bytes > max_overhead_bytes or not _is_pdf_part_header(header_block):
                raise DocumentError("multipart_part_invalid", 422)
            del buffer[: separator + 4]
            headers_read = True
        if headers_read and not completed:
            boundary_index = buffer.find(marker)
            if boundary_index >= 0:
                suffix_start = boundary_index + len(marker)
                if len(buffer) < suffix_start + 2:
                    continue
                if boundary_index:
                    yield bytes(buffer[:boundary_index])
                suffix = bytes(buffer[suffix_start : suffix_start + 2])
                if suffix != b"--":
                    raise DocumentError("multipart_multiple_parts", 422)
                trailer = bytes(buffer[suffix_start + 2 :])
                if trailer not in (b"", b"\r\n"):
                    raise DocumentError("multipart_structure_invalid", 422)
                completed = True
                buffer.clear()
            else:
                keep = len(marker) + 2
                if len(buffer) > keep:
                    yield bytes(buffer[:-keep])
                    del buffer[:-keep]
    if not completed:
        raise DocumentError("multipart_incomplete", 422)


def _is_pdf_part_header(header_block: bytes) -> bool:
    """确认唯一文件字段声明为 PDF；MIME 仅是初筛，深度校验仍由 pypdf 完成。"""
    try:
        headers = header_block.decode("ascii").lower()
    except UnicodeDecodeError:
        return False
    return (
        "content-disposition: form-data;" in headers
        and "filename=" in headers
        and (
            "content-type: application/pdf" in headers
            or "content-type: application/octet-stream" in headers
        )
    )
