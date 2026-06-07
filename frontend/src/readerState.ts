import {
  answerEvidence,
  pdfFileUrl,
  readEvidence,
  shouldPollJob,
  type Answer,
  type Evidence,
  type EvidenceRange,
  type EvidenceReadResult,
  type Job,
} from './api';

export function statusLabel(state: string): string {
  const labels: Record<string, string> = {
    queued: '排队中', running: '处理中', succeeded: '已完成', failed: '失败', cancelled: '已取消',
    answered: '已回答', insufficient_evidence: '证据不足', budget_exhausted: '预算耗尽', citation_failed: '引用校验失败',
  };
  return labels[state] ?? state;
}

export function physicalPageNumber(range: EvidenceRange | undefined): number | null {
  const index = range?.locations[0]?.page_index;
  return typeof index === 'number' && Number.isInteger(index) && index >= 0 ? index + 1 : null;
}

export function evidencePhysicalPages(evidence: Evidence): number[] {
  return [...new Set(evidence.ranges.flatMap((range) => range.locations.flatMap((location) => {
    const index = location.page_index;
    return typeof index === 'number' && Number.isInteger(index) && index >= 0 ? [index + 1] : [];
  })))].sort((left, right) => left - right);
}

export function citationEvidence(answer: Answer, evidenceId: string): Evidence | undefined {
  return answerEvidence(answer).find((item) => item.evidence_id === evidenceId);
}

export interface VerifiedEvidenceSelection {
  evidence: Evidence;
  readResult: EvidenceReadResult;
  pdfUrl: string;
  page: number;
}

export async function verifyEvidenceSelection(
  answer: Answer,
  libraryId: string,
  evidence: Evidence,
  reader: typeof readEvidence = readEvidence,
): Promise<VerifiedEvidenceSelection> {
  // 只接受当前答案 registry 内的证据，回读成功后才交付 PDF 身份。
  const boundEvidence = answerEvidence(answer).find((item) => item.evidence_id === evidence.evidence_id);
  if (!boundEvidence) {
    throw new Error('evidence_not_bound_to_answer');
  }
  if (answer.scope?.library_id && answer.scope.library_id !== libraryId) {
    throw new Error('answer_library_mismatch');
  }
  const generationId = answer.scope?.generation_id;
  if (!generationId) throw new Error('answer_generation_missing');
  const readResult = await reader({
    libraryId,
    retrievalRunId: answer.retrieval_run_id,
    evidenceId: evidence.evidence_id,
    answerId: answer.answer_id,
  });
  const expected = {
    library_id: libraryId,
    generation_id: generationId,
    retrieval_run_id: answer.retrieval_run_id,
    evidence_id: evidence.evidence_id,
    paper_id: boundEvidence.paper_id,
    document_version_id: boundEvidence.document_version_id,
    parse_snapshot_id: boundEvidence.parse_snapshot_id,
  };
  if (Object.entries(expected).some(([key, value]) => readResult[key] !== value)) {
    throw new Error('evidence_source_identity_mismatch');
  }
  const page = physicalPageNumber((readResult.items ?? readResult.ranges)?.[0]);
  if (page === null) throw new Error('evidence_physical_page_missing');
  return {
    evidence: boundEvidence,
    readResult,
    pdfUrl: pdfFileUrl({
      libraryId,
      generationId,
      paperId: boundEvidence.paper_id,
      documentVersionId: boundEvidence.document_version_id,
      parseSnapshotId: boundEvidence.parse_snapshot_id,
      answerId: answer.answer_id,
      retrievalRunId: answer.retrieval_run_id,
    }),
    page,
  };
}

interface JobPollingOptions {
  initialDelayMs?: number;
  maxDelayMs?: number;
  maxAttempts?: number;
}

export function startJobPolling(
  jobIds: string[],
  fetchJob: (jobId: string) => Promise<Job>,
  onJobs: (jobs: Job[]) => void,
  onError: (error: unknown) => void,
  options: JobPollingOptions = {},
): () => void {
  const initialDelay = options.initialDelayMs ?? 3500;
  const maxDelay = options.maxDelayMs ?? 30000;
  const maxAttempts = options.maxAttempts ?? 60;
  const pending = new Set(jobIds);
  let stopped = false;
  let attempts = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const poll = async () => {
    attempts += 1;
    const results = await Promise.allSettled([...pending].map(fetchJob));
    if (stopped) return;
    const updates: Job[] = [];
    for (const result of results) {
      if (result.status === 'fulfilled') {
        updates.push(result.value);
        if (!shouldPollJob(result.value.state)) pending.delete(result.value.job_id);
      } else {
        onError(result.reason);
      }
    }
    if (updates.length) onJobs(updates);
    if (!pending.size) return;
    if (attempts >= maxAttempts) {
      onError(new Error('任务轮询已达上限，请手动刷新任务状态。'));
      return;
    }
    timer = setTimeout(() => void poll(), Math.min(initialDelay * 2 ** (attempts - 1), maxDelay));
  };
  if (pending.size) void poll();
  return () => {
    stopped = true;
    if (timer !== undefined) clearTimeout(timer);
  };
}
