"""模块名称：解析块的结构切分。

用途：按章节与块类型构造有界检索文本，并保存原块可确认区间。
核心设计：相邻正文在同一章节内合并，长块先按句子拆分再按 token 预算拆分。
工程要点：标题及特殊 token 计入 600 硬上限；派生表头与说明不冒充原文。
"""

from dataclasses import dataclass, replace
from collections.abc import Callable
from html.parser import HTMLParser
import re
from uuid import UUID, uuid5

from academic_evidence_qa.documents.source_reader import FixedSource

CHUNK_NAMESPACE = UUID("bc72ed51-e676-4ce3-b933-195c1454aa38")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s+|(?<=[。！？])")
HARD_LIMIT = 600
TARGET_LIMIT = 450
CHUNKER_VERSION = "structure-v2"


class _TableRows(HTMLParser):
    """读取已标准化块中的原 HTML；只提取行/单元文本，不推断缺失单元格。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.header_rows: list[bool] = []
        self.header_explicit: list[bool] = []
        self.section: str | None = None
        self.cell_kinds: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"thead", "tbody", "tfoot"}:
            self.section = tag
        elif tag == "tr":
            self.row = []
            self.cell_kinds = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
            self.cell_kinds.append((tag, (dict(attrs).get("scope") or "").lower()))

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append(" ".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            # 明确的行标题优先，不能因数据行含 th 将数值一起重复为列头。
            row_scope = any(scope in {"row", "rowgroup"} for _, scope in self.cell_kinds)
            col_scope = any(scope in {"col", "colgroup"} for _, scope in self.cell_kinds)
            explicit = self.section == "thead" or col_scope
            inferred = (
                self.section is None
                and bool(self.cell_kinds)
                and all(tag == "th" for tag, _ in self.cell_kinds)
            )
            self.header_rows.append(not row_scope and (explicit or inferred))
            self.header_explicit.append(explicit and not row_scope)
            self.row = None
        elif tag in {"thead", "tbody", "tfoot"}:
            self.section = None


def _table_parts(block, prefix: str, count_tokens) -> list[tuple[str, dict]] | None:
    """重复连续列头、caption 与脚注；行范围为原 HTML 零基半开行号，不推测合并单元格。"""
    raw = block.raw_provenance
    original = raw.get("original", {})
    html = raw.get("table_body") or (
        original.get("table_body") if isinstance(original, dict) else None
    )
    if not isinstance(html, str) or "<tr" not in html.lower():
        return None
    table = _TableRows()
    table.feed(html)
    if not table.rows:
        return None
    notes = []
    for key in ("table_caption", "caption", "table_footnote"):
        value = raw.get(key)
        if isinstance(value, str):
            notes.append(value)
        elif isinstance(value, list):
            notes.extend(item for item in value if isinstance(item, str))
    header_count = 0
    while header_count < len(table.rows) and table.header_rows[header_count]:
        header_count += 1
    inferred_header = any(not value for value in table.header_explicit[:header_count])
    # 无结构标记的 td 首行仅作保守显示上下文，并显式标记；明确 tbody 不猜列头。
    if header_count == 0 and not re.search(r"<(?:thead|tbody|tfoot)\b|\bscope\s*=", html, re.I):
        # 混合 th/td 可能是未声明 scope 的行标题，保留为数据而非重复列头。
        first_row = re.search(r"<tr\b[^>]*>(.*?)</tr\s*>", html, re.I | re.S)
        if first_row and not re.search(r"<th\b", first_row[1], re.I):
            header_count = 1
        inferred_header = True
    if header_count == len(table.rows):
        # 全表无可确认数据区时不产生空区间，降级保留每行作为数据。
        header_count = 0
        inferred_header = True
    header = "\n".join(" | ".join(row) for row in table.rows[:header_count])
    context = prefix + "\n".join(notes + [header]) + "\n"
    if count_tokens(context) >= HARD_LIMIT:
        raise ValueError("table_context_exceeds_token_limit")
    groups = []
    text, row_start = context, header_count
    for row_index, row in enumerate(table.rows[header_count:], header_count):
        line = " | ".join(row) + "\n"
        if count_tokens(context + line) > HARD_LIMIT:
            raise ValueError("table_row_exceeds_token_limit")
        if count_tokens(text + line) > TARGET_LIMIT and text != context:
            groups.append(
                (
                    text,
                    {
                        "block_reading_order": block.reading_order,
                        "start": 0,
                        "end": len(block.text),
                        "table_row_start": row_start,
                        "table_row_end": row_index,
                        "derived": True,
                        "table_header_inferred": inferred_header,
                    },
                )
            )
            text, row_start = context, row_index
        text += line
    groups.append(
        (
            text,
            {
                "block_reading_order": block.reading_order,
                "start": 0,
                "end": len(block.text),
                "table_row_start": row_start,
                "table_row_end": len(table.rows),
                "derived": True,
                "table_header_inferred": inferred_header,
            },
        )
    )
    return groups


@dataclass(frozen=True)
class ChunkDraft:
    """名称：切分草稿。用途：冻结派生文本与原文映射，等待索引代身份。"""

    paper_id: UUID
    document_version_id: UUID
    parse_snapshot_id: UUID
    block_type: str
    retrieval_text: str
    source_ranges: tuple[dict, ...]
    source_locations: tuple[dict, ...]
    quality_flags: tuple[str, ...]
    embedding_tokens: int
    reranker_tokens: int


def _location_dict(location) -> dict:
    return {
        "page_index": location.page_index,
        "bbox": list(location.bbox) if location.bbox else None,
        "reliability": location.reliability,
        "raw_provenance": location.raw_provenance,
    }


def _segments(text: str, budget: int, count_tokens: Callable[[str], int]) -> list[tuple[int, int]]:
    """按句末边界切分；单句超限时二分字符长度且绝不静默截断。"""
    boundaries = [0]
    for match in SENTENCE_BOUNDARY.finditer(text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    output = []
    start = 0
    for end in boundaries[1:]:
        if end <= start:
            continue
        if count_tokens(text[start:end]) <= budget:
            continue
        previous = max((edge for edge in boundaries if start < edge < end), default=start)
        if previous > start:
            output.append((start, previous))
            start = previous
        while count_tokens(text[start:end]) > budget:
            low, high = start + 1, end
            while low < high:
                middle = (low + high + 1) // 2
                if count_tokens(text[start:middle]) <= budget:
                    low = middle
                else:
                    high = middle - 1
            if low <= start:
                raise ValueError("single_character_exceeds_token_limit")
            output.append((start, low))
            start = low
    if start < len(text):
        output.append((start, len(text)))
    return output


def build_chunks(
    sources: list[FixedSource],
    embedding_tokens: Callable[[str], int],
    reranker_tokens: Callable[[str], int],
) -> list[ChunkDraft]:
    """名称：构造结构 chunk。用途：在实际 tokenizer 下执行有界切分。

    Args:
        sources: 已核验且固定的成员来源。
        embedding_tokens, reranker_tokens: 两种真实 tokenizer 的含特殊 token 长度。
    Returns:
        全部非空来源的 chunk；每个区间都指回固定原块。
    Raises:
        ValueError: 标题或单字符使限制无法满足。
    """
    chunks = []
    chapter_keys = {}
    for source in sources:
        headings = sorted(source.document.sections, key=lambda item: item.reading_order)
        heading_index = 0
        heading_path: list[str] = []
        for block in sorted(source.document.blocks, key=lambda item: item.reading_order):
            while (
                heading_index < len(headings)
                and headings[heading_index].reading_order <= block.reading_order
            ):
                heading = headings[heading_index]
                if heading.title:
                    level = heading.level or 1
                    heading_path = heading_path[: max(0, level - 1)] + [heading.title]
                heading_index += 1
            body = block.text
            if not body.strip():
                continue
            heading_text = " > ".join(heading_path)
            chapter_keys[(source.parse_snapshot_id, block.reading_order)] = heading_text
            prefix = f"{heading_text}\n" if heading_text and heading_text != body else ""
            if embedding_tokens(prefix) >= HARD_LIMIT:
                raise ValueError("heading_exceeds_token_limit")
            table_parts = (
                _table_parts(block, prefix, embedding_tokens)
                if block.block_type == "table"
                else None
            )
            if table_parts is not None:
                for text, source_range in table_parts:
                    flags = set(block.quality_flags) | {"derived_table_row_group"}
                    if source_range["table_header_inferred"]:
                        flags.add("table_header_inferred")
                    if any(
                        location.reliability == "range_incomplete" for location in block.locations
                    ):
                        flags.add("range_incomplete")
                    # 原适配器可能仅检查通用 text；派生 HTML 文本也必须保留字符风险。
                    if "\ufffd" in text:
                        flags.add("replacement_character")
                    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
                        flags.add("control_character")
                    chunks.append(
                        ChunkDraft(
                            source.paper_id,
                            source.document_version_id,
                            source.parse_snapshot_id,
                            "table",
                            text,
                            (source_range,),
                            tuple(_location_dict(item) for item in block.locations),
                            tuple(sorted(flags)),
                            embedding_tokens(text),
                            reranker_tokens(text),
                        )
                    )
                continue
            # 解析器原始 HTML 和图/公式字段仍保留在源块 provenance；检索文本只含派生文本。
            allowed = min(TARGET_LIMIT, HARD_LIMIT - embedding_tokens(prefix))
            for start, end in _segments(body, allowed, embedding_tokens):
                part = body[start:end]
                if not part:
                    continue
                text = prefix + part
                count = embedding_tokens(text)
                if count > HARD_LIMIT:
                    raise ValueError("chunk_exceeds_token_limit")
                flags = set(block.quality_flags)
                if any(location.reliability == "range_incomplete" for location in block.locations):
                    flags.add("range_incomplete")
                if block.block_type in {"image", "figure"}:
                    flags.add("caption_only_visual_source")
                ranges = ({"block_reading_order": block.reading_order, "start": start, "end": end},)
                chunks.append(
                    ChunkDraft(
                        source.paper_id,
                        source.document_version_id,
                        source.parse_snapshot_id,
                        "references"
                        if heading_path
                        and heading_path[-1].strip().lower()
                        in {"references", "bibliography", "参考文献"}
                        else block.block_type,
                        text,
                        ranges,
                        tuple(_location_dict(item) for item in block.locations),
                        tuple(sorted(flags)),
                        count,
                        reranker_tokens(text),
                    )
                )
    # 同一章节的相邻普通正文合并；特殊结构保持独立，避免跨表/公式猜造关系。
    documents = {
        source.parse_snapshot_id: {block.reading_order: block for block in source.document.blocks}
        for source in sources
    }
    associated = []
    for draft in chunks:
        if draft.block_type not in {"equation", "code", "algorithm", "image", "figure"}:
            associated.append(draft)
            continue
        blocks = documents[draft.parse_snapshot_id]
        block = blocks[draft.source_ranges[0]["block_reading_order"]]
        # 仅接受解析 provenance 中明确的阅读顺序关系，不靠相邻位置猜测定义归属。
        orders = block.raw_provenance.get("related_block_reading_orders", [])
        confirmed = []
        if isinstance(orders, list):
            confirmed = [
                blocks[order]
                for order in orders
                if type(order) is int
                and order in blocks
                and blocks[order].block_type in {"text", "paragraph"}
            ]
        flags = set(draft.quality_flags)
        text, ranges, locations = draft.retrieval_text, draft.source_ranges, draft.source_locations
        if not confirmed:
            flags.add("association_unconfirmed")
        for linked in confirmed:
            candidate = text + "\n" + linked.text
            if embedding_tokens(candidate) > HARD_LIMIT:
                flags.add("association_context_exceeds_budget")
                continue
            text = candidate
            ranges += (
                {
                    "block_reading_order": linked.reading_order,
                    "start": 0,
                    "end": len(linked.text),
                    "association": "explicit_parser_relation",
                },
            )
            locations += tuple(_location_dict(item) for item in linked.locations)
            flags.update(linked.quality_flags)
        associated.append(
            replace(
                draft,
                retrieval_text=text,
                source_ranges=ranges,
                source_locations=locations,
                quality_flags=tuple(sorted(flags)),
                embedding_tokens=embedding_tokens(text),
                reranker_tokens=reranker_tokens(text),
            )
        )
    chunks = associated
    merged = []
    for draft in chunks:
        if merged:
            previous = merged[-1]
            plain = {"text", "paragraph"}
            same_source = previous.parse_snapshot_id == draft.parse_snapshot_id
            adjacent = (
                previous.source_ranges[-1]["block_reading_order"] + 1
                == draft.source_ranges[0]["block_reading_order"]
            )
            previous_prefix = chapter_keys[
                (previous.parse_snapshot_id, previous.source_ranges[-1]["block_reading_order"])
            ]
            current_prefix = chapter_keys[
                (draft.parse_snapshot_id, draft.source_ranges[0]["block_reading_order"])
            ]
            body = draft.retrieval_text
            if current_prefix and body.startswith(current_prefix + "\n"):
                body = body[len(current_prefix) + 1 :]
            text = previous.retrieval_text + "\n" + body
            count = embedding_tokens(text)
            if (
                same_source
                and adjacent
                and previous.block_type in plain
                and draft.block_type in plain
                and previous_prefix == current_prefix
                and count <= TARGET_LIMIT
            ):
                merged[-1] = ChunkDraft(
                    draft.paper_id,
                    draft.document_version_id,
                    draft.parse_snapshot_id,
                    draft.block_type,
                    text,
                    previous.source_ranges + draft.source_ranges,
                    previous.source_locations + draft.source_locations,
                    tuple(sorted(set(previous.quality_flags + draft.quality_flags))),
                    count,
                    reranker_tokens(text),
                )
                continue
        merged.append(draft)
    return merged


def chunk_id(generation_id: UUID, draft: ChunkDraft, order: int) -> UUID:
    """名称：生成 chunk 身份。用途：将同一来源在不同 generation 中明确区分。"""
    return uuid5(
        CHUNK_NAMESPACE, f"{generation_id}:{draft.parse_snapshot_id}:{order}:{draft.source_ranges}"
    )
