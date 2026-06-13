"""模块名称：答案引用完整性。

用途：只解析当前 registry 别名，拒绝伪造/缺失引用，提供可复用纯函数。
核心设计：按非空段落要求引用；语义支持不由 ID 或格式有效性推断。
工程要点：只规范合法引用的空白和组合形式；错误引用原样保留，不发起生成式修复。
"""

import re
from academic_evidence_qa.contracts.rag import Citation, EvidencePack


def normalize_citation_aliases(
    text: str, valid_aliases: set[str]
) -> tuple[str, list[str], list[str]]:
    """名称：检查引用别名。用途：共享格式规则，不虚构 EvidencePack 或证据身份。

    Args:
        text: 模型完整输出，论文内容仍是不可信数据。
        valid_aliases: 当前交付内经过来源核验的别名集合。
    Returns:
        规范文本、实际引用的合法别名和错误类别；非空错误时不得交付正文。
    限制：段落引用是格式规则，不能识别所有无依据事实；support_check_status 必须 not_run。
    """
    aliases = []
    errors = []

    def replace(match):
        value = match.group(0)
        # 不猜测畸形编号，也不部分修正混有未知编号的组合。
        if not re.fullmatch(r"\[\s*E[1-9][0-9]*(?:\s*,\s*E[1-9][0-9]*)*\s*\]", value):
            errors.append("malformed_citation")
            if re.search(r"E[^\]\s,]+", value):
                errors.append("unknown_citation_alias")
            return value
        group = re.findall(r"E[1-9][0-9]*", value)
        aliases.extend(group)
        if any(alias not in valid_aliases for alias in group):
            errors.append("unknown_citation_alias")
            return value
        return "".join(f"[{alias}]" for alias in group)

    # 同一解析器同时负责提取和段落校验；未闭合引用也必须被拒绝。
    normalized = re.sub(r"\[\s*E[^\[\]\n]*(?:\]|(?=\n|\[|$))", replace, text)
    for paragraph in re.split(r"\n\s*\n", normalized.strip()):
        if paragraph.strip() and not re.search(r"\[E[1-9][0-9]*\]", paragraph):
            errors.append("citation_missing")
            break
    if not text.strip():
        errors.append("answer_empty")
    citations = [alias for alias in dict.fromkeys(aliases) if alias in valid_aliases]
    return normalized, citations, list(dict.fromkeys(errors))


def normalize_citations(text: str, pack: EvidencePack) -> tuple[str, list[Citation], list[str]]:
    """名称：兼容 EvidencePack 引用检查。用途：在共享别名规则后绑定真实 evidence ID。

    Args:
        text, pack: 原草稿和当前固定、可回源的 registry。
    Returns:
        规范正文、原有 Citation 契约和错误类别；不推断事实支持。
    """
    registry = {item.alias: item.evidence_id for item in pack.evidence}
    normalized, aliases, errors = normalize_citation_aliases(text, set(registry))
    return normalized, [Citation(alias=a, evidence_id=registry[a]) for a in aliases], errors


def validate_citations(text: str, pack: EvidencePack) -> tuple[list[Citation], list[str]]:
    """兼容只读校验入口；交付方使用 normalize_citations 获取一致的规范文本。"""
    _, citations, errors = normalize_citations(text, pack)
    return citations, errors
