"""模块名称：真实 pair 预算与父 chunk 重排。

用途：按句界分窗，精确记录检索文本字符区间，最高窗口分数聚合父候选。
核心设计：tokenizer 实际 pair 长度是硬约束，超长句再按字符边界二分。
工程要点：不截断 query；每父最多三窗、总共 120 对，未覆盖显式标记。
"""

import re

from academic_evidence_qa.contracts.rag import Candidate, RerankWindow


def make_windows(query, passage, tokenizer, *, max_windows=3):
    """名称：有界重排窗口。用途：保留半开字符区间，不以单文本长度代替 pair。

    Args:
        query, passage: 原始问题与父 retrieval_text，均不静默删尾。
        tokenizer: 本地真实重排 tokenizer。
        max_windows: 父窗口上限。
    Returns:
        (窗口列表, 是否覆盖不完整)。
    """
    if len(tokenizer(query, add_special_tokens=False, truncation=False)["input_ids"]) > 96:
        raise ValueError("reranker_query_limit_exceeded")
    boundaries = [match.end() for match in re.finditer(r"[.!?。！？](?:\s|$)|\n", passage)]
    windows, start = [], 0
    while start < len(passage) and len(windows) < max_windows:
        low, high = start + 1, len(passage)
        end = start
        # 二分只用来寻找可容纳的字符长度；最终 pair 再逐值核验。
        while low <= high:
            mid = (low + high) // 2
            length = len(tokenizer(query, passage[start:mid], truncation=False)["input_ids"])
            if length <= 512:
                end, low = mid, mid + 1
            else:
                high = mid - 1
        if end == start:
            raise ValueError("reranker_pair_limit_exceeded")
        if end < len(passage):
            sentence_end = max((b for b in boundaries if start < b <= end), default=end)
            end = sentence_end
        length = len(tokenizer(query, passage[start:end], truncation=False)["input_ids"])
        if length > 512:
            raise ValueError("reranker_pair_limit_exceeded")
        windows.append(RerankWindow(char_start=start, char_end=end, pair_tokens=length))
        start = end
    return windows, start < len(passage)


async def rerank(query: str, candidates: list[Candidate], runtime, executor):
    """名称：父候选重排。用途：有界共享推理并保留窗口、聚合与未覆盖标记。

    Args:
        query, candidates: 固定范围且已核验来源的候选。
        runtime, executor: API owner 和专用同步队列。
    Returns:
        排序后的候选、状态及排队/推理耗时。
    """

    def prepare():
        pairs, mappings, prepared = [], [], []
        for candidate in candidates:
            windows, partial = make_windows(
                query, candidate.retrieval_text, runtime.reranker_tokenizer
            )
            selected = windows[: max(0, 120 - len(pairs))]
            prepared.append((candidate, selected, partial or len(selected) != len(windows)))
            for window in selected:
                pairs.append((query, candidate.retrieval_text[window.char_start : window.char_end]))
                mappings.append((candidate, window))
        return pairs, mappings, prepared

    (pairs, mappings, prepared), timing = await executor.run(prepare)
    if not pairs:
        return candidates, [], timing
    scores, scoring_time = await executor.run(lambda: runtime.score_pairs(pairs), total_seconds=300)
    # 同步任务真正成功后才修改响应对象；超时/取消后的后台线程不能晚到改写降级结果。
    for candidate, windows, partial in prepared:
        candidate.windows = windows
        if partial:
            candidate.statuses.append("rerank_partial")
    for (candidate, window), score in zip(mappings, scores, strict=True):
        window.score = score
        candidate.rerank_score = max(
            candidate.rerank_score if candidate.rerank_score is not None else float("-inf"), score
        )
    candidates.sort(
        key=lambda c: (
            c.rerank_score is not None,
            c.rerank_score if c.rerank_score is not None else float("-inf"),
            c.rrf_score or 0,
        ),
        reverse=True,
    )
    return (
        candidates,
        ["rerank_partial"] if any("rerank_partial" in c.statuses for c in candidates) else [],
        {key: timing[key] + scoring_time[key] for key in timing},
    )
