"""模块名称：答案引用完整性。

用途：只解析当前 registry 别名，拒绝伪造/缺失引用，提供可复用纯函数。
核心设计：按非空段落要求引用；语义支持不由 ID 或格式有效性推断。
工程要点：不删除错误引用，不修改草稿，不发起生成式修复。
"""

import re
from academic_evidence_qa.contracts.rag import Citation, EvidencePack


def validate_citations(text: str, pack: EvidencePack) -> tuple[list[Citation], list[str]]:
    """名称：检查本次别名。用途：拒绝未知和缺失引用，保留原草稿诊断。

    Args:
        text: 模型完整输出，论文内容仍是不可信数据。
        pack: 当前固定、可回源 registry；来源可读性由调用方复核。
    Returns:
        有效 Citation 列表和错误类别；非空错误时不得交付已校验正文。
    限制：段落引用是格式规则，不能识别所有无依据事实；support_check_status 必须 not_run。
    """
    registry = {e.alias: e.evidence_id for e in pack.evidence}
    aliases = re.findall(r"\[(E[^\]\s]*)\]", text)
    errors = []
    if any(alias not in registry for alias in aliases):
        errors.append("unknown_citation_alias")
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        if paragraph.strip() and not re.search(r"\[E[1-9][0-9]*\]", paragraph):
            errors.append("citation_missing")
            break
    if not text.strip():
        errors.append("answer_empty")
    citations = [
        Citation(alias=a, evidence_id=registry[a]) for a in dict.fromkeys(aliases) if a in registry
    ]
    return citations, errors
