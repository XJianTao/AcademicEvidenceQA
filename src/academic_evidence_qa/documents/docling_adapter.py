"""模块：Docling 离线解析适配器。

用途概要：在受限子进程中按需加载 Docling，并将其 Markdown 输出转换为项目统一的 ParsedDocument。
核心设计：父进程不导入 Docling；依赖或模型不可用时返回稳定的 ``docling_unavailable``，绝不回退为无限制执行。
工程要点：本模块不联网、不读取远端凭据；真实解析必须由显式调用方提供已授权的本地 PDF。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from academic_evidence_qa.contracts.documents import ParsedDocument
from academic_evidence_qa.documents.parser import ParserError, normalize_mineru_document
from academic_evidence_qa.infrastructure.process_guard import (
    GuardedProcessRequest,
    ResourceGuardError,
    WindowsProcessGuard,
)


class DoclingAdapterError(ParserError):
    """Docling 离线解析的稳定失败码，不暴露第三方异常原文。"""


def parse_docling_pdf(
    pdf_path: Path,
    *,
    timeout_seconds: int,
    memory_mib: int,
    guard: WindowsProcessGuard | None = None,
) -> ParsedDocument:
    """在受限子进程中执行一次 Docling 解析。

    Args:
        pdf_path: 服务控制的本地 PDF 路径；不接受 HTTP 请求直接传入的路径。
        timeout_seconds, memory_mib: 子进程总时限与内存上限。
        guard: 可注入的真实 Job Object 防护，测试或宿主不可用时必须 fail closed。
    Returns:
        使用统一 ParsedDocument 契约的离线结果。
    Raises:
        DoclingAdapterError: 依赖、模型、输入或资源防护不可用。
    """
    if not pdf_path.is_file():
        raise DoclingAdapterError("docling_input_missing")
    result_path = pdf_path.parent / "docling-result.json"
    request = GuardedProcessRequest(
        command=[
            sys.executable,
            "-m",
            "academic_evidence_qa.documents.docling_adapter",
            str(pdf_path),
            str(result_path),
        ],
        working_directory=pdf_path.parent,
        timeout_seconds=timeout_seconds,
        memory_mib=memory_mib,
    )
    try:
        completed = (guard or WindowsProcessGuard()).run(request)
    except ResourceGuardError as error:
        raise DoclingAdapterError(error.error_code) from error
    if (
        completed.exit_code != 0
        or not result_path.is_file()
        or result_path.stat().st_size > 128 * 1024 * 1024
    ):
        raise DoclingAdapterError("docling_failed")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DoclingAdapterError("docling_failed") from error
    error_code = payload.get("error_code")
    if error_code:
        raise DoclingAdapterError(str(error_code))
    content = payload.get("content_list")
    if not isinstance(content, list):
        raise DoclingAdapterError("docling_invalid_result")
    return normalize_mineru_document(content)


def _docling_child(pdf_path: Path, result_path: Path) -> int:
    """子进程入口：延迟导入第三方并只写筛选后的标准结果。"""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        result_path.write_text(json.dumps({"error_code": "docling_unavailable"}), encoding="utf-8")
        return 0
    try:
        conversion = DocumentConverter().convert(str(pdf_path))
        markdown = conversion.document.export_to_markdown()
        payload: dict[str, Any] = {
            "content_list": [{"type": "text", "text": markdown, "page_idx": None}],
        }
    except Exception:
        payload = {"error_code": "docling_failed"}
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(2)
    raise SystemExit(_docling_child(Path(sys.argv[1]), Path(sys.argv[2])))
