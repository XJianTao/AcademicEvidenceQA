"""MinerU 在线结果适配。

模块名称：MinerU 结构块适配器
用途概要：将 MinerU content-list 结构转换成可检索文本和来源元数据。
核心设计：保留原始字段、页码、坐标和可用的引用编号；质量风险只标记不掩盖。
工程要点：不依赖远端服务；空表格/算法仍保留 caption 和 provenance。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_mineru_block(block: dict[str, Any]) -> dict[str, Any]:
    """标准化一个 MinerU content-list 块。

    Args:
        block: MinerU 返回的原始结构块。

    Returns:
        包含检索文本、完整可得 provenance 与质量标记的字典。
    """
    original = block.get("original") if isinstance(block.get("original"), dict) else {}
    parts = [
        _text(block.get("text")),
        _text(original.get("table_body")),
        _text(original.get("code_body")),
        _text(block.get("caption")),
    ]
    retrieval_text = "\n".join(dict.fromkeys(part for part in parts if part))
    page_idx = block.get("page_idx")
    provenance = {
        key: block[key]
        for key in ("type", "page_idx", "bbox", "index", "id", "content")
        if key in block
    }
    if "original" in block:
        provenance["original"] = original
    quality: list[str] = []
    kind = str(block.get("type", "")).lower()
    if "table" in kind and not _text(original.get("table_body")):
        quality.append("empty_table_body")
    if any(token in kind for token in ("code", "algorithm")) and not _text(
        original.get("code_body")
    ):
        quality.append("empty_code_body")
    if "\ufffd" in retrieval_text:
        quality.append("replacement_character")
    if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in retrieval_text):
        quality.append("control_character")
    if re.search(r"\[(?:\s*\d{3,}|\s*\d+\s*[-,]\s*\d+\s*[-,]\s*\d+\s*)\]", retrieval_text):
        quality.append("suspicious_citation_number")
    if not isinstance(page_idx, int) or page_idx < 0:
        quality.append("missing_physical_page")
    return {
        "retrieval_text": retrieval_text,
        "provenance": provenance,
        "physical_page_start": page_idx + 1
        if isinstance(page_idx, int) and page_idx >= 0
        else None,
        "physical_page_end": page_idx + 1 if isinstance(page_idx, int) and page_idx >= 0 else None,
        "position_reliability": "block_start_only" if isinstance(page_idx, int) else "unknown",
        "quality_flags": quality,
    }


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(item for value_item in value if (item := _text(value_item)))
    if isinstance(value, dict):
        return "\n".join(item for value_item in value.values() if (item := _text(value_item)))
    return ""
