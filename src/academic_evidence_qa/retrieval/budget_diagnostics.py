"""模块名称：离线证据预算计量。

用途：只收集候选到封装各阶段的数量及 token/字节元数据。
核心设计：独立于 EvidencePack 的可选内存收集器，不改变被预算 JSON 或选择策略。
工程要点：不保存原文、查询、来源标识或任意字符串；未启用时不增加序列化开销。
"""

from dataclasses import dataclass, field
import json
from typing import Any

from academic_evidence_qa.infrastructure.generation_client import estimated_tokens, serialize

STAGES = frozenset(
    {
        "recall_counts",
        "candidate_pool",
        "candidate_reranked",
        "candidate_limit",
        "candidate",
        "source_ranges",
        "expanded_ranges",
        "merged_ranges",
        "empty_pack",
        "evidence",
        "tentative_pack",
        "fallback_pack",
        "final_fields",
        "final_trim",
        "final_pack",
    }
)
METRICS = frozenset(
    {
        "item_count",
        "ordinal",
        "range_count",
        "related_range_count",
        "text_characters",
        "text_utf8_bytes",
        "dense_count",
        "bm25_count",
        "fused_count",
        "token_budget",
        "max_bytes",
        "over_token_budget",
        "over_byte_budget",
        "accepted",
        "removed_count",
    }
)


@dataclass
class EvidenceBudgetDiagnostics:
    """名称：预算阶段收集器。用途：离线调用者显式启用的纯元数据诊断。

    Args:
        events: 数值及固定阶段名；不保留传入 payload 的引用。
    Returns:
        record 追加白名单计量；调用者负责保存，不自动写日志、数据库或 API 响应。
    """

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, payload: Any = None, **metrics: int | bool) -> None:
        """名称：计量阶段。用途：复用现有预算函数并对照未转义 UTF-8 JSON 大小。

        Args:
            stage: 固定白名单阶段名；拒绝外部文本作为标签。
            payload: 临时计量对象；正文只用于计算长度，不存储。
            metrics: 白名单非负整数或布尔值。
        Returns:
            None；estimated_tokens 仍是 ASCII JSON 字节数加 256，并非 tokenizer 实测。
        Raises:
            ValueError: 标签或计量字段不在白名单，或数值非法。
        """
        if stage not in STAGES or any(
            key not in METRICS or type(value) not in (int, bool) or value < 0
            for key, value in metrics.items()
        ):
            raise ValueError("invalid_budget_diagnostic_metadata")
        event: dict[str, Any] = {"stage": stage, **metrics}
        if payload is not None:
            event.update(
                canonical_ascii_bytes=len(serialize(payload)),
                utf8_json_bytes=len(
                    json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                    ).encode("utf-8")
                ),
                estimated_tokens=estimated_tokens(payload),
            )
        self.events.append(event)
