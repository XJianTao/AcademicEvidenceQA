export type JobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type AnswerStatus = 'answered' | 'insufficient_evidence' | 'budget_exhausted' | 'citation_failed';

export interface Library {
  library_id: string;
  membership_revision: number;
}

export interface PublishedSource {
  paper_id?: string;
  generation_id?: string;
  document_version_id?: string;
  parse_snapshot_id?: string;
  generation?: string;
  document_version?: string;
  parse_snapshot?: string;
}

export interface Paper {
  paper_id: string;
  title: string | null;
  document_version_ids: string[];
  selected_document_version_id?: string | null;
  published_source?: PublishedSource | null;
  pending?: boolean | null;
  status?: 'active' | 'removed' | null;
}

export interface PaperList {
  items: Paper[];
  next_offset?: number | null;
  library_id?: string | null;
  membership_revision?: number | null;
  active_generation_id?: string | null;
}

export interface Job {
  job_id: string;
  job_type: 'parse' | 'index';
  state: JobState;
  stage: string;
  attempt_id?: string | null;
  error_code?: string | null;
  cancel_requested?: boolean;
  retry_of_job_id?: string | null;
  retry_job_ids?: string[];
  library_id?: string | null;
  document_version_id?: string | null;
  parse_snapshot_id?: string | null;
  index_job_ids?: string[];
  generation_id?: string | null;
  membership_revision?: number | null;
}

export interface UploadResult extends Library {
  paper_id: string;
  document_version_id: string;
  file_sha256: string;
  status: 'accepted' | 'registered' | 'reused';
  job_id?: string | null;
  index_job_id?: string | null;
  parse_job_id?: string | null;
  parse_snapshot_id?: string | null;
  generation_id?: string | null;
}

export interface EvidenceRange {
  block_id: string;
  block_reading_order: number;
  start: number;
  end: number;
  text: string;
  block_type: string;
  locations: Array<{ page_index?: number; page?: number; [key: string]: unknown }>;
  quality_flags: string[];
}

export interface Evidence {
  evidence_id: string;
  alias: string;
  paper_id: string;
  document_version_id: string;
  parse_snapshot_id: string;
  ranges: EvidenceRange[];
  quality_flags: string[];
}

export interface Citation {
  alias: string;
  evidence_id: string;
}

export interface Answer {
  answer_id: string;
  retrieval_run_id: string;
  status: AnswerStatus;
  text?: string | null;
  citations: Citation[];
  citation_check_status?: string;
  support_check_status?: string;
  diagnostics: string[];
  repair_diagnostics?: Array<Record<string, unknown>>;
  usage?: Record<string, unknown>;
  timings?: Record<string, number>;
  evidence?: Evidence[];
  evidence_pack?: { evidence?: Evidence[]; status?: string };
  scope?: { library_id?: string; generation_id?: string; members?: Array<Record<string, string>> };
}

export function answerEvidence(answer: Answer): Evidence[] {
  return answer.evidence ?? answer.evidence_pack?.evidence ?? [];
}

export function shouldPollJob(state: JobState): boolean {
  return state === 'queued' || state === 'running';
}

export interface EvidenceReadResult {
  evidence_id?: string;
  alias?: string;
  items?: EvidenceRange[];
  ranges?: EvidenceRange[];
  next_offset?: number | null;
  next_char_offset?: number | null;
  [key: string]: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined ?? '').replace(/\/$/, '');

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  const raw = await response.text();
  let payload: unknown = null;
  try {
    payload = raw ? JSON.parse(raw) : null;
  } catch {
    payload = raw;
  }
  if (!response.ok) {
    const body = payload as { error_code?: string; detail?: string } | null;
    const code = body?.error_code ?? `http_${response.status}`;
    throw new ApiError(body?.detail ?? code, response.status, code);
  }
  return payload as T;
}

function idempotencyKey(prefix: string): string {
  return `${prefix}-${crypto.randomUUID()}`;
}

export function createLibrary(): Promise<Library> {
  return request<Library>('/v1/libraries', {
    method: 'POST',
    headers: { 'Idempotency-Key': idempotencyKey('library') },
    body: JSON.stringify({}),
  });
}

export function listPapers(libraryId: string): Promise<PaperList> {
  return request<PaperList>(`/v1/libraries/${encodeURIComponent(libraryId)}/papers?offset=0&limit=100`);
}

export function uploadDocument(
  file: File,
  fields: { libraryId: string; membershipRevision: number; parserBackend: 'mineru_remote' | 'docling'; allowRemoteProcessing: boolean },
): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file, file.name);
  form.append('library_id', fields.libraryId);
  form.append('expected_membership_revision', String(fields.membershipRevision));
  form.append('parser_backend', fields.parserBackend);
  form.append('allow_remote_processing', String(fields.allowRemoteProcessing));
  return request<UploadResult>('/v1/documents', {
    method: 'POST',
    headers: { 'Idempotency-Key': idempotencyKey('upload') },
    body: form,
  });
}

export function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/v1/jobs/${encodeURIComponent(jobId)}`);
}

export async function getAnswer(libraryId: string, answerId: string): Promise<Answer> {
  const payload = await request<Answer | { state: string; error_code?: string }>(
    `/v1/answers/${encodeURIComponent(answerId)}?${new URLSearchParams({ library_id: libraryId })}`,
  );
  if ('state' in payload) {
    throw new ApiError('历史答案尚未完成，保留原状态且不会自动重新生成。', 409, payload.error_code ?? `answer_${payload.state}`);
  }
  return payload;
}

export function cancelJob(jobId: string): Promise<unknown> {
  return request(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    headers: { 'Idempotency-Key': idempotencyKey('cancel') },
  });
}

export function retryJob(jobId: string): Promise<Job> {
  return request<Job>(`/v1/jobs/${encodeURIComponent(jobId)}/retry`, {
    method: 'POST',
    headers: { 'Idempotency-Key': idempotencyKey('retry') },
  });
}

export interface AnswerQuestionInput {
  libraryId: string;
  generationId?: string;
  paperIds: string[];
  query: string;
  evidenceTokenBudget: number;
  maxOutputTokens: number;
}

export function answerQuestion(input: AnswerQuestionInput, requestKey: string): Promise<Answer> {
  return request<Answer>('/v1/answer', {
    method: 'POST',
    headers: { 'Idempotency-Key': requestKey },
    body: JSON.stringify({
      library_id: input.libraryId,
      generation_id: input.generationId || undefined,
      paper_ids: input.paperIds,
      query: input.query,
      mode: 'hybrid_rerank',
      recall_limit: 40,
      limit: 10,
      evidence_token_budget: input.evidenceTokenBudget,
      max_output_tokens: input.maxOutputTokens,
      max_supplemental_retrievals: 2,
      allow_citation_repair: true,
    }),
  });
}

export function readEvidence(input: {
  libraryId: string;
  retrievalRunId: string;
  evidenceId: string;
  answerId?: string;
}): Promise<EvidenceReadResult> {
  return request<EvidenceReadResult>('/v1/evidence/read', {
    method: 'POST',
    body: JSON.stringify({
      library_id: input.libraryId,
      retrieval_run_id: input.retrievalRunId,
      evidence_id: input.evidenceId,
      answer_id: input.answerId,
      max_characters: 32000,
    }),
  });
}

export function pdfFileUrl(input: {
  libraryId: string;
  generationId?: string;
  paperId: string;
  documentVersionId: string;
  parseSnapshotId: string;
  answerId?: string;
  retrievalRunId?: string;
}): string {
  if (Boolean(input.answerId) !== Boolean(input.retrievalRunId)) throw new Error('historical_context_invalid');
  const query = new URLSearchParams({
    library_id: input.libraryId,
    paper_id: input.paperId,
    document_version_id: input.documentVersionId,
    parse_snapshot_id: input.parseSnapshotId,
  });
  if (input.generationId) query.set('generation_id', input.generationId);
  if (input.answerId && input.retrievalRunId) {
    query.set('answer_id', input.answerId);
    query.set('retrieval_run_id', input.retrievalRunId);
  }
  return `${API_BASE}/v1/documents/${encodeURIComponent(input.documentVersionId)}/file?${query}`;
}

export function displayError(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}（HTTP ${error.status}）`;
  if (error instanceof Error) return error.message;
  return '网络请求失败';
}
