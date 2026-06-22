"""模块名称：受限 PDF 预检与解析结果标准化。

用途概要：在受限子进程中执行 PDF 深度预检，并将 MinerU/Docling 输出转换为统一 ParsedDocument。

核心设计：重型 parser 和 pypdf 均只在子进程入口延迟导入；标准化不会猜测缺失元数据或来源位置。

工程要点：标准化输入应来自受限解包后的文件，原始 provenance 与质量风险必须保留。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from academic_evidence_qa.contracts.documents import (
    ParsedBlock,
    ParsedDocument,
    ParsedSection,
    PdfPreflightResult,
    SourceLocation,
)
from academic_evidence_qa.infrastructure.process_guard import (
    GuardedProcessRequest,
    ResourceGuardError,
    WindowsProcessGuard,
)
from academic_evidence_qa.mineru_adapter import normalize_mineru_block

# v4 接入原文标题入库；配置哈希包含此版本，重新导入时不复用尚未提取标题的旧快照。
ADAPTER_VERSION = "m1b_normalizer_v4"


def extract_paper_title(document: ParsedDocument) -> str | None:
    """从全文唯一且位于首页的非空一级章节提取题目。

    Args:
        document: 已归一化的解析结果；不读取文件名或推断缺失页码。
    Returns:
        合并空白后的原文题目；候选不唯一、位置不确定或超过元数据容量时为空。
    """
    candidates = [
        section for section in document.sections
        if section.level == 1 and section.title and section.title.strip()
    ]
    if len(candidates) != 1:
        return None
    section = candidates[0]
    if not section.locations or any(location.page_index != 0 for location in section.locations):
        return None
    title = " ".join(section.title.split())
    return title if len(title) <= 1024 else None


class ParserError(RuntimeError):
    """解析/预检稳定失败，避免暴露 pypdf 或远端细节。"""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(error_code)


def preflight_pdf(
    pdf_path: Path,
    *,
    max_pages: int,
    timeout_seconds: int,
    memory_mib: int,
    guard: WindowsProcessGuard | None = None,
) -> PdfPreflightResult:
    """严格预检 PDF。

    Args:
        pdf_path: 上传模块生成、仅位于 staging 的 PDF 文件。
        max_pages: 可接受的最大页数。
        timeout_seconds, memory_mib: 子进程总时限和进程提交内存上限。
        guard: 可注入的 Windows Job Object 防护，默认创建真实防护。
    Returns:
        页数和加密状态；成功结果始终为未加密。
    Raises:
        ParserError: 非法、加密、损坏、超页或资源防护失败。
    """
    result_path = pdf_path.parent / "preflight-result.json"
    request = GuardedProcessRequest(
        command=[
            sys.executable,
            "-m",
            "academic_evidence_qa.documents.parser",
            "--preflight",
            str(pdf_path),
            str(result_path),
            str(max_pages),
        ],
        working_directory=pdf_path.parent,
        timeout_seconds=timeout_seconds,
        memory_mib=memory_mib,
    )
    try:
        completed = (guard or WindowsProcessGuard()).run(request)
    except ResourceGuardError as error:
        if error.error_code == "process_timeout":
            raise ParserError("pdf_preflight_timeout") from error
        raise ParserError(error.error_code) from error
    if completed.exit_code != 0 or not result_path.is_file():
        raise ParserError("pdf_preflight_failed")
    # 此文件由受限子进程写入，且 schema 极小；大小仍先限制，再在父进程读取结果。
    if result_path.stat().st_size > 4096:
        raise ParserError("pdf_preflight_failed")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ParserError("pdf_preflight_failed") from error
    error_code = payload.get("error_code")
    if error_code:
        raise ParserError(str(error_code))
    page_count = payload.get("page_count")
    if not isinstance(page_count, int) or page_count < 1 or page_count > max_pages:
        raise ParserError("pdf_page_limit_exceeded")
    if payload.get("is_encrypted") is not False:
        raise ParserError("pdf_encrypted")
    return PdfPreflightResult(page_count=page_count, is_encrypted=False)


def normalize_mineru_document(content_list: list[dict[str, Any]]) -> ParsedDocument:
    """转换 MinerU content-list。

    Args:
        content_list: 受限 JSON 进程已经检查大小和格式后的列表。
    Returns:
        保留表格/算法原始字段、位置与质量标记的统一解析文档。
    Raises:
        ParserError: 输入结构不是预期的 JSON 对象列表。
    """
    if not isinstance(content_list, list):
        raise ParserError("invalid_parser_result")
    blocks: list[ParsedBlock] = []
    sections: list[ParsedSection] = []
    metrics: dict[str, int] = {
        "block_count": 0,
        "nonempty_text_block_count": 0,
        "nul_character_block_count": 0,
        "warning_count": 0,
        "unlocated_count": 0,
    }
    for reading_order, raw in enumerate(content_list):
        if not isinstance(raw, dict):
            raise ParserError("invalid_parser_result")
        converted = normalize_mineru_block(_mineru_adapter_input(raw))
        flags = tuple(str(flag) for flag in converted["quality_flags"])
        # PostgreSQL Text 和 JSONB 均拒绝 U+0000。检查完整原块的键和值，
        # 因而也覆盖章节标题、位置和未用于检索的嵌套 provenance。
        # 不清洗原文；标准化产物仍保留诊断事实，由质量门拒绝提交快照。
        if _contains_nul_character(raw):
            flags += ("nul_character_not_persistable",)
            metrics["nul_character_block_count"] += 1
        location = _location_from_mineru(raw, converted["position_reliability"])
        block_type = str(raw.get("type") or "unknown")
        blocks.append(
            ParsedBlock(
                block_type=block_type,
                reading_order=reading_order,
                text=converted["retrieval_text"],
                locations=(location,),
                # 字段映射只服务于文本提取；来源始终保存远端实际原块，不伪造 original。
                raw_provenance=dict(raw),
                quality_flags=flags,
            )
        )
        metrics["block_count"] += 1
        if converted["retrieval_text"].strip():
            metrics["nonempty_text_block_count"] += 1
        metrics["warning_count"] += len(flags)
        if location.page_index is None:
            metrics["unlocated_count"] += 1
        level = raw.get("text_level")
        explicit_heading_level = level if type(level) is int and level > 0 else None
        if block_type in {"title", "h1", "h2", "h3"} or explicit_heading_level is not None:
            sections.append(
                ParsedSection(
                    title=converted["retrieval_text"] or None,
                    level=explicit_heading_level or _section_level(block_type),
                    reading_order=reading_order,
                    locations=(location,),
                )
            )
    # 远端可能返回非空结构列表却没有任何可读内容；保留块和来源供诊断，
    # 不能只因块数量大于零就把全空正文判为可用。
    if metrics["nonempty_text_block_count"] == 0 or metrics["nul_character_block_count"] > 0:
        return ParsedDocument(tuple(sections), tuple(blocks), "rejected", metrics)
    quality = "usable" if metrics["warning_count"] == 0 else "usable_with_warnings"
    return ParsedDocument(tuple(sections), tuple(blocks), quality, metrics)


def _contains_nul_character(value: Any) -> bool:
    """检查 JSON 字符串、对象键和值；保留命中的原值供诊断。"""
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, dict):
        return any(
            _contains_nul_character(key) or _contains_nul_character(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_nul_character(item) for item in value)
    return False


def normalize_mineru_result(
    input_path: Path,
    output_path: Path,
    *,
    max_json_bytes: int,
    timeout_seconds: int,
    memory_mib: int,
    guard: WindowsProcessGuard | None = None,
) -> ParsedDocument:
    """名称：受限 MinerU 标准化。用途概要：隔离不可信 JSON 的读取与转换。

    Args:
        input_path: 已由受限解包器写入的 MinerU content-list JSON；仅内部路径。
        output_path: 当前 attempt 中尚不存在的标准化产物路径，不覆盖已有证据。
        max_json_bytes: 输入与标准化输出各自的最大字节数，在反序列化前检查。
        timeout_seconds, memory_mib: Windows Job Object 总墙钟期限与内存上限。
        guard: 可注入受控防护；缺少真实防护时不得回退到宿主直接标准化。
    Returns:
        与 output_path 标准化 JSON 绑定的 ParsedDocument。
    Raises:
        ParserError: 超限、非法 JSON、子进程失败或资源防护不可用；失败清理本次输出。
    """
    if max_json_bytes <= 0:
        raise ParserError("invalid_resource_limit")
    if output_path.exists() or input_path.resolve() == output_path.resolve():
        raise ParserError("normalize_output_exists")
    try:
        if not input_path.is_file():
            raise ParserError("normalize_input_missing")
        if input_path.stat().st_size > max_json_bytes:
            raise ParserError("parser_json_limit_exceeded")
    except OSError:
        raise ParserError("normalize_input_missing") from None
    request = GuardedProcessRequest(
        command=[
            sys.executable,
            "-m",
            "academic_evidence_qa.documents.parser",
            "--normalize-mineru",
            str(input_path.resolve()),
            str(output_path.resolve()),
            str(max_json_bytes),
        ],
        working_directory=output_path.parent,
        timeout_seconds=timeout_seconds,
        memory_mib=memory_mib,
    )
    try:
        completed = (guard or WindowsProcessGuard()).run(request)
        if completed.exit_code != 0:
            raise ParserError("normalize_failed")
        return load_normalized_document(output_path, max_json_bytes=max_json_bytes)
    except ResourceGuardError as error:
        output_path.unlink(missing_ok=True)
        code = "normalize_timeout" if error.error_code == "process_timeout" else error.error_code
        raise ParserError(code) from None
    except ParserError:
        output_path.unlink(missing_ok=True)
        raise


def load_normalized_document(path: Path, *, max_json_bytes: int) -> ParsedDocument:
    """名称：恢复受控标准化文档。用途概要：仅加载本项目子进程生成的固定 schema。

    Args:
        path: 可信 attempt 目录内的标准化 JSON，不可用于加载未经标准化的远端输出。
        max_json_bytes: 读取和反序列化前执行的输出大小限制。
    Returns:
        校验基本字段类型并恢复 tuple 位置结构后的 ParsedDocument。
    Raises:
        ParserError: 缺失、超限、非法 schema 或子进程记录的稳定失败码。
    """
    try:
        if max_json_bytes <= 0 or path.stat().st_size > max_json_bytes:
            raise ParserError("parser_json_limit_exceeded")
        with path.open("rb") as stream:
            content = stream.read(max_json_bytes + 1)
        if len(content) > max_json_bytes:
            raise ParserError("parser_json_limit_exceeded")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("error_code") in {
            "parser_json_limit_exceeded",
            "invalid_parser_result",
            "normalize_failed",
        }:
            raise ParserError(payload["error_code"])
        sections = payload["sections"]
        blocks = payload["blocks"]
        quality = payload["quality_state"]
        metrics = payload["quality_metrics"]
        metadata = payload.get("metadata", {})
        if (
            not isinstance(sections, list)
            or not isinstance(blocks, list)
            or quality not in {"usable", "usable_with_warnings", "rejected"}
            or not isinstance(metrics, dict)
            or not all(type(value) is int and value >= 0 for value in metrics.values())
            or not isinstance(metadata, dict)
        ):
            raise ValueError
        return ParsedDocument(
            sections=tuple(_section_from_payload(section) for section in sections),
            blocks=tuple(_block_from_payload(block) for block in blocks),
            quality_state=quality,
            quality_metrics=metrics,
            metadata=metadata,
        )
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        # 不把远端文本、JSON 解码片段或第三方错误原文放入异常摘要。
        raise ParserError("invalid_parser_result") from None


def normalize_pdf_bbox(
    bbox: tuple[float, float, float, float],
    *,
    crop_box: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float, float, float]:
    """将 PDF 原始 bbox 转为旋转 CropBox 的顶左 0–1 坐标。

    Args:
        bbox: PDF 用户空间中的左下、右上坐标。
        crop_box: 当前页 CropBox，坐标保持 PDF 用户空间。
        rotation: 显示旋转角度，仅接受 0/90/180/270。
    Returns:
        截断至 0–1 的左、上、右、下规范化矩形。
    """
    left, bottom, right, top = bbox
    crop_left, crop_bottom, crop_right, crop_top = crop_box
    width, height = crop_right - crop_left, crop_top - crop_bottom
    if width <= 0 or height <= 0 or rotation % 90:
        raise ParserError("invalid_pdf_coordinates")
    points = [(left, bottom), (left, top), (right, bottom), (right, top)]

    def map_point(x: float, y: float) -> tuple[float, float]:
        horizontal = (x - crop_left) / width
        vertical = (y - crop_bottom) / height
        match rotation % 360:
            case 0:
                return horizontal, 1 - vertical
            case 90:
                return vertical, horizontal
            case 180:
                return 1 - horizontal, vertical
            case 270:
                return 1 - vertical, 1 - horizontal
            case _:
                raise ParserError("invalid_pdf_coordinates")

    mapped = [map_point(x, y) for x, y in points]
    xs, ys = zip(*mapped, strict=True)
    values = (min(xs), min(ys), max(xs), max(ys))
    if not all(math.isfinite(value) for value in values):
        raise ParserError("invalid_pdf_coordinates")
    return tuple(max(0.0, min(1.0, value)) for value in values)  # type: ignore[return-value]


def parsed_document_payload(document: ParsedDocument) -> dict[str, Any]:
    """生成稳定的标准化 JSON 载荷，供受限输出和哈希存储使用。"""
    return asdict(document)


def _mineru_adapter_input(raw: dict[str, Any]) -> dict[str, Any]:
    """兼容官方顶层字段和 M0 original 字段，仅映射明确出现的结构与 caption。"""
    adapted = dict(raw)
    original = dict(raw["original"]) if isinstance(raw.get("original"), dict) else {}
    for key in ("table_body", "code_body"):
        if key not in original and key in raw:
            original[key] = raw[key]
    adapted["original"] = original
    captions = [
        raw[key]
        for key in ("caption", "table_caption", "code_caption", "image_caption")
        if key in raw
    ]
    adapted["caption"] = captions
    return adapted


def _locations_from_payload(payload: Any) -> tuple[SourceLocation, ...]:
    """恢复由本项目生成的位置 schema，绝不将原始 bbox 猜成规范化坐标。"""
    if not isinstance(payload, list) or not payload:
        raise ValueError
    locations = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError
        page, bbox = item["page_index"], item["bbox"]
        reliability = item["reliability"]
        provenance = item.get("raw_provenance", {})
        if (
            (page is not None and (type(page) is not int or page < 0))
            or reliability not in {"located", "page_only", "unlocated", "range_incomplete"}
            or not isinstance(provenance, dict)
        ):
            raise ValueError
        if bbox is not None:
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(
                    type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1
                    for value in bbox
                )
                or bbox[0] > bbox[2]
                or bbox[1] > bbox[3]
            ):
                raise ValueError
            bbox = tuple(bbox)
        locations.append(SourceLocation(page, bbox, reliability, provenance))
    return tuple(locations)


def _block_from_payload(payload: Any) -> ParsedBlock:
    """恢复一个标准化块，拒绝将类型错误或不完整产物当成可用快照。"""
    if not isinstance(payload, dict):
        raise ValueError
    kind, order, text = payload["block_type"], payload["reading_order"], payload["text"]
    provenance, flags = payload.get("raw_provenance", {}), payload.get("quality_flags", [])
    if (
        not isinstance(kind, str)
        or type(order) is not int
        or order < 0
        or not isinstance(text, str)
        or not isinstance(provenance, dict)
        or not isinstance(flags, list)
        or not all(isinstance(flag, str) for flag in flags)
    ):
        raise ValueError
    return ParsedBlock(
        kind, order, text, _locations_from_payload(payload["locations"]), provenance, tuple(flags)
    )


def _section_from_payload(payload: Any) -> ParsedSection:
    """恢复显式章节层级；不存在的标题与层级保持 None。"""
    if not isinstance(payload, dict):
        raise ValueError
    title, level, order = payload["title"], payload["level"], payload["reading_order"]
    if (
        (title is not None and not isinstance(title, str))
        or (level is not None and (type(level) is not int or level < 1))
        or type(order) is not int
        or order < 0
    ):
        raise ValueError
    return ParsedSection(title, level, order, _locations_from_payload(payload["locations"]))


def _location_from_mineru(raw: dict[str, Any], reliability: str) -> SourceLocation:
    page_index = raw.get("page_idx")
    page = page_index if isinstance(page_index, int) and page_index >= 0 else None
    bbox = raw.get("bbox")
    normalized_bbox = None
    if (
        isinstance(bbox, list | tuple)
        and len(bbox) == 4
        and all(isinstance(value, int | float) for value in bbox)
    ):
        # MinerU 坐标系在远端结果中并不统一；原值保留但不虚构 PDF 规范化坐标。
        reliability = "page_only" if reliability == "located" else reliability
    if page is None:
        reliability = "unlocated"
    elif reliability == "block_start_only":
        reliability = "range_incomplete"
    elif reliability not in {"located", "page_only", "range_incomplete"}:
        reliability = "page_only"
    return SourceLocation(page, normalized_bbox, reliability, {"raw_bbox": bbox} if bbox else {})


def _section_level(block_type: str) -> int | None:
    return int(block_type[1:]) if len(block_type) == 2 and block_type.startswith("h") else None


def _preflight_child(pdf_path: Path, result_path: Path, max_pages: int) -> int:
    """仅由 `--preflight` 子进程调用；pypdf 延迟导入避免宿主深度解析。"""
    payload: dict[str, Any]
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path), strict=True)
        if reader.is_encrypted:
            payload = {"error_code": "pdf_encrypted"}
        else:
            pages = len(reader.pages)
            payload = (
                {"error_code": "pdf_page_limit_exceeded"}
                if pages < 1 or pages > max_pages
                else {"page_count": pages, "is_encrypted": False}
            )
    except Exception:  # pypdf 的具体异常不应成为外部协议或日志内容。
        payload = {"error_code": "pdf_invalid"}
    result_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return 0


def _normalize_mineru_child(input_path: Path, output_path: Path, max_json_bytes: int) -> int:
    """受限子进程独占入口：先限制原始字节，再解析 JSON、转换和限制输出。"""
    try:
        with input_path.open("rb") as stream:
            content = stream.read(max_json_bytes + 1)
        if len(content) > max_json_bytes:
            raise ParserError("parser_json_limit_exceeded")
        document = normalize_mineru_document(json.loads(content))
        encoded = json.dumps(
            parsed_document_payload(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > max_json_bytes:
            raise ParserError("parser_json_limit_exceeded")
    except ParserError as error:
        encoded = json.dumps({"error_code": error.error_code}).encode("utf-8")
    except Exception:
        encoded = b'{"error_code":"invalid_parser_result"}'
    # exclusive create 防止覆盖历史 evidence；错误输出由宿主识别后删除。
    try:
        with output_path.open("xb") as stream:
            stream.write(encoded)
    except OSError:
        return 1
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", nargs=3, metavar=("PDF", "RESULT", "MAX_PAGES"))
    parser.add_argument("--normalize-mineru", nargs=3, metavar=("INPUT", "OUTPUT", "MAX_BYTES"))
    args = parser.parse_args()
    if args.preflight is not None:
        pdf, result, max_pages = args.preflight
        return _preflight_child(Path(pdf), Path(result), int(max_pages))
    if args.normalize_mineru is not None:
        source, output, max_bytes = args.normalize_mineru
        return _normalize_mineru_child(Path(source), Path(output), int(max_bytes))
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
