import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  answerEvidence,
  answerQuestion,
  createLibrary,
  displayError,
  pdfFileUrl,
  shouldPollJob,
  type Answer,
} from './index';

afterEach(() => vi.restoreAllMocks());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('AcademicEvidenceQA API client', () => {
  it('adds an idempotency key to library creation and sends the expected endpoint', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ library_id: 'lib-1', membership_revision: 0 }));
    const result = await createLibrary();
    expect(result.library_id).toBe('lib-1');
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/v1/libraries');
    expect(init?.method).toBe('POST');
    expect(new Headers(init?.headers).get('Idempotency-Key')).toMatch(/^library-/);
  });

  it('keeps fixed scope fields in the answer request and exposes server status', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({
      answer_id: 'a-1', retrieval_run_id: 'r-1', status: 'insufficient_evidence', citations: [], diagnostics: ['missing paper'],
    }));
    const result = await answerQuestion({ libraryId: 'lib-1', generationId: 'gen-1', paperIds: ['paper-1'], query: 'What changed?', evidenceTokenBudget: 6000, maxOutputTokens: 1200 }, 'answer-fixed-key');
    expect(result.status).toBe('insufficient_evidence');
    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(body).toMatchObject({ library_id: 'lib-1', generation_id: 'gen-1', paper_ids: ['paper-1'], mode: 'hybrid_rerank', recall_limit: 40, limit: 10 });
    expect(new Headers(init?.headers).get('Idempotency-Key')).toBe('answer-fixed-key');
  });

  it('uses only answer-bound evidence and retains terminal polling semantics', () => {
    const answer: Answer = { answer_id: 'a', retrieval_run_id: 'r', status: 'answered', citations: [], diagnostics: [], evidence_pack: { evidence: [{ evidence_id: 'e', alias: 'E1', paper_id: 'p', document_version_id: 'v', parse_snapshot_id: 's', ranges: [], quality_flags: [] }] } };
    expect(answerEvidence(answer).map((item) => item.evidence_id)).toEqual(['e']);
    expect(shouldPollJob('queued')).toBe(true);
    expect(shouldPollJob('running')).toBe(true);
    expect(shouldPollJob('succeeded')).toBe(false);
    expect(shouldPollJob('failed')).toBe(false);
    expect(shouldPollJob('cancelled')).toBe(false);
  });

  it('builds a fixed-identity PDF URL with encoded scope parameters', () => {
    const url = pdfFileUrl({ libraryId: 'lib/1', generationId: 'gen-1', paperId: 'paper-1', documentVersionId: 'version-1', parseSnapshotId: 'snapshot-1' });
    expect(url).toContain('/v1/documents/version-1/file?');
    expect(url).toContain('library_id=lib%2F1');
    expect(url).toContain('generation_id=gen-1');
    expect(url).toContain('parse_snapshot_id=snapshot-1');
  });

  it('历史 PDF 携带成对的原答案与原 run，不允许只传其中一个', () => {
    const input = { libraryId: 'lib-1', generationId: 'gen-old', paperId: 'paper-1', documentVersionId: 'version-old', parseSnapshotId: 'snapshot-old' };
    const query = new URL(pdfFileUrl({ ...input, answerId: 'answer/old', retrievalRunId: 'run/old' }), 'http://localhost').searchParams;
    expect(query.get('answer_id')).toBe('answer/old');
    expect(query.get('retrieval_run_id')).toBe('run/old');
    expect(() => pdfFileUrl({ ...input, answerId: 'answer-old' })).toThrow('historical_context_invalid');
    expect(() => pdfFileUrl({ ...input, retrievalRunId: 'run-old' })).toThrow('historical_context_invalid');
    const normalQuery = new URL(pdfFileUrl(input), 'http://localhost').searchParams;
    expect(normalQuery.has('answer_id')).toBe(false);
    expect(normalQuery.has('retrieval_run_id')).toBe(false);
  });

  it('maps structured API errors without exposing raw response text', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ error_code: 'scope_membership_changed' }, 409));
    await expect(createLibrary()).rejects.toMatchObject({ code: 'scope_membership_changed', status: 409 });
    expect(displayError(new Error('network unavailable'))).toBe('network unavailable');
  });

  it('does not retry a lost answer response and preserves the explicit recovery body and key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new TypeError('response lost'))
      .mockResolvedValueOnce(jsonResponse({ answer_id: 'a-1', retrieval_run_id: 'r-1', status: 'answered', citations: [], diagnostics: [] }));
    const input = { libraryId: 'lib-1', generationId: 'gen-1', paperIds: ['paper-1'], query: 'What changed?', evidenceTokenBudget: 6000, maxOutputTokens: 1200 };
    await expect(answerQuestion(input, 'answer-original')).rejects.toThrow('response lost');
    expect(fetchMock).toHaveBeenCalledOnce();
    await answerQuestion(input, 'answer-original');
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][1]?.body).toBe(fetchMock.mock.calls[1][1]?.body);
    for (const [, init] of fetchMock.mock.calls) expect(new Headers(init?.headers).get('Idempotency-Key')).toBe('answer-original');
  });
});
