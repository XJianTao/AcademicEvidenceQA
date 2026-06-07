import { afterEach, describe, expect, it, vi } from 'vitest';
import { getAnswer, type Answer, type Evidence, type EvidenceReadResult, type Job } from './api';
import { citationEvidence, evidencePhysicalPages, physicalPageNumber, startJobPolling, statusLabel, verifyEvidenceSelection } from './readerState';

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

const evidence: Evidence = {
  evidence_id: 'evidence-1', alias: 'E1', paper_id: 'paper-1', document_version_id: 'version-1', parse_snapshot_id: 'snapshot-1', quality_flags: [],
  ranges: [{ block_id: 'block-1', block_reading_order: 0, start: 0, end: 4, text: 'text', block_type: 'text', locations: [{ page_index: 4 }], quality_flags: [] }],
};
const answer: Answer = {
  answer_id: 'answer-1', retrieval_run_id: 'run-1', status: 'answered', citations: [{ alias: 'E1', evidence_id: evidence.evidence_id }], diagnostics: [],
  scope: { library_id: 'library-1', generation_id: 'generation-old' }, evidence: [evidence],
};
const readResult: EvidenceReadResult = {
  evidence_id: evidence.evidence_id, retrieval_run_id: answer.retrieval_run_id, library_id: 'library-1', generation_id: 'generation-old',
  paper_id: evidence.paper_id, document_version_id: evidence.document_version_id, parse_snapshot_id: evidence.parse_snapshot_id, items: evidence.ranges,
};
const running: Job = { job_id: 'job-1', job_type: 'parse', state: 'running', stage: 'polling' };

describe('阅读状态与来源身份', () => {
  it.each([
    ['answered', '已回答'], ['insufficient_evidence', '证据不足'], ['budget_exhausted', '预算耗尽'], ['citation_failed', '引用校验失败'],
    ['queued', '排队中'], ['running', '处理中'], ['failed', '失败'], ['cancelled', '已取消'],
  ])('保留服务端状态 %s', (status, label) => { expect(statusLabel(status)).toBe(label); });

  it('只把零起始物理 page_index 转为阅读器页码，不用印刷 page 猜测', () => {
    expect(physicalPageNumber(evidence.ranges[0])).toBe(5);
    expect(physicalPageNumber({ ...evidence.ranges[0], locations: [{ page: 88 }] })).toBeNull();
  });

  it('汇总全部区间全部位置的物理页，并按 evidence_id 解析 citation', () => {
    const acrossPages = { ...evidence, ranges: [...evidence.ranges, { ...evidence.ranges[0], locations: [{ page_index: 7 }, { page_index: 4 }, { page_index: 6 }, { page: 88 }] }] };
    expect(evidencePhysicalPages(acrossPages)).toEqual([5, 7, 8]);
    expect(citationEvidence(answer, 'evidence-1')).toBe(evidence);
    expect(citationEvidence(answer, 'E1')).toBeUndefined();
    expect(citationEvidence(answer, 'unrelated')).toBeUndefined();
  });

  it('点击引用等待来源核验成功，再按原答案 generation 打开物理页', async () => {
    let finish!: (result: EvidenceReadResult) => void;
    const reader = vi.fn(() => new Promise<EvidenceReadResult>((resolve) => { finish = resolve; }));
    const openPdf = vi.fn();
    const selection = verifyEvidenceSelection(answer, 'library-1', evidence, reader).then(openPdf);
    expect(reader).toHaveBeenCalledWith({ libraryId: 'library-1', retrievalRunId: 'run-1', evidenceId: 'evidence-1', answerId: 'answer-1' });
    expect(openPdf).not.toHaveBeenCalled();
    finish(readResult);
    await selection;
    expect(openPdf).toHaveBeenCalledWith(expect.objectContaining({ page: 5, pdfUrl: expect.stringContaining('generation_id=generation-old') }));
    const pdfUrl = new URL(openPdf.mock.calls[0][0].pdfUrl, 'http://localhost');
    expect(pdfUrl.searchParams.get('answer_id')).toBe('answer-1');
    expect(pdfUrl.searchParams.get('retrieval_run_id')).toBe('run-1');
  });

  it('历史 PDF 完整来源只取当前答案固定 registry，忽略同 ID 的外部替换身份', async () => {
    const selection = await verifyEvidenceSelection(answer, 'library-1', {
      ...evidence, paper_id: 'other-paper', document_version_id: 'version-new', parse_snapshot_id: 'snapshot-new',
    }, vi.fn().mockResolvedValue(readResult));
    expect(selection.evidence).toBe(evidence);
    const params = new URL(selection.pdfUrl, 'http://localhost').searchParams;
    expect(params.get('paper_id')).toBe('paper-1');
    expect(params.get('document_version_id')).toBe('version-1');
    expect(params.get('parse_snapshot_id')).toBe('snapshot-1');
  });

  it('来源读取失败或身份错配不会打开 PDF', async () => {
    const openPdf = vi.fn();
    await expect(verifyEvidenceSelection(answer, 'library-1', evidence, vi.fn().mockRejectedValue(new Error('source_not_found'))).then(openPdf)).rejects.toThrow('source_not_found');
    await expect(verifyEvidenceSelection(answer, 'library-1', evidence, vi.fn().mockResolvedValue({ ...readResult, generation_id: 'generation-new' })).then(openPdf)).rejects.toThrow('evidence_source_identity_mismatch');
    expect(openPdf).not.toHaveBeenCalled();
  });

  it('拒绝不属于当前答案 registry 的引用', async () => {
    const reader = vi.fn();
    await expect(verifyEvidenceSelection(answer, 'library-1', { ...evidence, evidence_id: 'unrelated' }, reader)).rejects.toThrow('evidence_not_bound_to_answer');
    expect(reader).not.toHaveBeenCalled();
  });

  it('拒绝把历史答案绑定到当前输入的其他库', async () => {
    const reader = vi.fn();
    await expect(verifyEvidenceSelection(answer, 'library-other', evidence, reader)).rejects.toThrow('answer_library_mismatch');
    expect(reader).not.toHaveBeenCalled();
  });

  it('历史答案只读请求保留库身份，unknown 状态不重发生成', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response(JSON.stringify(answer), { status: 200 }));
    expect(await getAnswer('library/1', 'answer/1')).toMatchObject({ answer_id: 'answer-1' });
    expect(fetchMock.mock.calls[0][0]).toBe('/v1/answers/answer%2F1?library_id=library%2F1');
    expect(fetchMock.mock.calls[0][1]?.method ?? 'GET').toBe('GET');
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ answer_id: 'a', state: 'unknown', error_code: 'generation_timeout' }), { status: 200 }));
    await expect(getAnswer('library-1', 'a')).rejects.toMatchObject({ code: 'generation_timeout' });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe('有界任务轮询', () => {
  it('更新 running 状态不会立即重入，并退避至终态后停止', async () => {
    vi.useFakeTimers();
    const fetchJob = vi.fn().mockResolvedValueOnce(running).mockResolvedValueOnce({ ...running, state: 'succeeded' });
    const onJobs = vi.fn();
    const cleanup = startJobPolling(['job-1'], fetchJob, onJobs, vi.fn(), { initialDelayMs: 100, maxDelayMs: 400 });
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(99);
    expect(fetchJob).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchJob).toHaveBeenCalledTimes(2);
    expect(onJobs).toHaveBeenLastCalledWith([expect.objectContaining({ state: 'succeeded' })]);
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetchJob).toHaveBeenCalledTimes(2);
    cleanup();
  });

  it('连续网络失败退避并在上限停止，错误可见', async () => {
    vi.useFakeTimers();
    const fetchJob = vi.fn().mockRejectedValue(new Error('network unavailable'));
    const onError = vi.fn();
    startJobPolling(['job-1'], fetchJob, vi.fn(), onError, { initialDelayMs: 100, maxDelayMs: 400, maxAttempts: 3 });
    await vi.advanceTimersByTimeAsync(100);
    expect(fetchJob).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(199);
    expect(fetchJob).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchJob).toHaveBeenCalledTimes(3);
    expect(onError).toHaveBeenLastCalledWith(expect.objectContaining({ message: '任务轮询已达上限，请手动刷新任务状态。' }));
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetchJob).toHaveBeenCalledTimes(3);
  });

  it('组件卸载清除定时器，并忽略在途请求返回', async () => {
    vi.useFakeTimers();
    let finish!: (job: Job) => void;
    const fetchJob = vi.fn(() => new Promise<Job>((resolve) => { finish = resolve; }));
    const onJobs = vi.fn();
    const cleanup = startJobPolling(['job-1'], fetchJob, onJobs, vi.fn());
    cleanup();
    finish(running);
    await vi.advanceTimersByTimeAsync(10000);
    expect(onJobs).not.toHaveBeenCalled();
    expect(fetchJob).toHaveBeenCalledTimes(1);
  });
});
