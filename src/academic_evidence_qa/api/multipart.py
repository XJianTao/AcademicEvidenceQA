"""模块名称：有界 multipart 接收。

用途：在 ASGI 原始流上解析既定文件与小字段，不调用框架 UploadFile/spool。
核心设计：每次最多喂入 64 KiB，累计真实字节和每字段/首部预算；PDF 复用正式接收器。
工程要点：断连/结构失败由接收器回收本次 staging；不相信 Content-Length。
"""

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from python_multipart.exceptions import MultipartParseError

from academic_evidence_qa.contracts.documents import DocumentError
from academic_evidence_qa.documents.upload import receive_pdf_stream


async def receive_management_upload(request, store, max_upload_bytes):
    """名称：接收管理上传。用途：有界解析文件和契约字段。

    Args:
        request: 原始 ASGI Request；绝不调用 form()。
        store, max_upload_bytes: 正式 FileStore 和实际 PDF 限额。
    Returns:
        ReceivedPdf 与纯字符串字段；尚未预检或登记。
    Raises:
        DocumentError: 大小、首部、重复字段或 multipart 结构非法。
    """
    media, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary", b"")
    if (
        media != b"multipart/form-data"
        or not 1 <= len(boundary) <= 70
        or not all(32 <= value < 127 for value in boundary)
    ):
        raise DocumentError("multipart_content_type_invalid", 422)
    fields, seen, pending = {}, set(), []
    current = {
        "name": None,
        "headers": {},
        "header_name": bytearray(),
        "header_value": bytearray(),
        "value": bytearray(),
        "size": 0,
    }
    ended, total = False, 0
    allowed = {
        "file",
        "library_id",
        "expected_membership_revision",
        "paper_id",
        "parser_backend",
        "allow_remote_processing",
    }

    def part_begin():
        current.update(
            name=None,
            headers={},
            header_name=bytearray(),
            header_value=bytearray(),
            value=bytearray(),
            size=0,
        )

    def header_data(which, data, start, end):
        current["size"] += end - start
        if current["size"] > 8192:
            raise DocumentError("multipart_overhead_exceeded", 413)
        current[which].extend(data[start:end])

    def header_end():
        name = bytes(current["header_name"]).lower()
        if name in current["headers"]:
            raise DocumentError("multipart_header_duplicate", 422)
        current["headers"][name] = bytes(current["header_value"])
        current["header_name"].clear()
        current["header_value"].clear()

    def headers_finished():
        disposition, params = parse_options_header(
            current["headers"].get(b"content-disposition", b"")
        )
        if disposition != b"form-data":
            raise DocumentError("multipart_part_invalid", 422)
        try:
            name = params.get(b"name", b"").decode("ascii")
        except UnicodeError as error:
            raise DocumentError("multipart_part_invalid", 422) from error
        if name not in allowed or name in seen or (name == "file") != (b"filename" in params):
            raise DocumentError("multipart_part_invalid", 422)
        if name == "file" and current["headers"].get(b"content-type") not in (
            b"application/pdf",
            b"application/octet-stream",
        ):
            raise DocumentError("pdf_type_invalid", 422)
        seen.add(name)
        current["name"] = name

    def part_data(data, start, end):
        if current["name"] == "file":
            pending.append(bytes(data[start:end]))
        else:
            current["value"].extend(data[start:end])
            if len(current["value"]) > 4096:
                raise DocumentError("multipart_field_exceeded", 413)

    def part_end():
        if current["name"] != "file":
            try:
                fields[current["name"]] = current["value"].decode("utf-8")
            except UnicodeError as error:
                raise DocumentError("multipart_field_invalid", 422) from error

    def finish():
        nonlocal ended
        ended = True

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": part_begin,
            "on_header_field": lambda data, start, end: header_data(
                "header_name", data, start, end
            ),
            "on_header_value": lambda data, start, end: header_data(
                "header_value", data, start, end
            ),
            "on_header_end": header_end,
            "on_headers_finished": headers_finished,
            "on_part_data": part_data,
            "on_part_end": part_end,
            "on_end": finish,
        },
    )

    async def file_chunks():
        nonlocal total
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_upload_bytes + 65536:
                raise DocumentError("upload_size_exceeded", 413)
            for offset in range(0, len(chunk), 65536):
                try:
                    parser.write(chunk[offset : offset + 65536])
                except MultipartParseError as error:
                    raise DocumentError("multipart_structure_invalid", 422) from error
                while pending:
                    yield pending.pop(0)
        parser.finalize()
        if not ended or "file" not in seen:
            raise DocumentError("multipart_incomplete", 422)

    received = await receive_pdf_stream(
        file_chunks(), file_store=store, max_upload_bytes=max_upload_bytes
    )
    return received, fields
