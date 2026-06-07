import { describe, expect, it, vi } from 'vitest';
import type { Answer, AnswerQuestionInput } from './api';
import {
  LibraryRequestGuard,
  normalizeAnswerInput,
  prepareAnswerRequest,
  readAnswerRequests,
  sendSavedAnswerRequest,
  updateAnswerRequest,
  verifyAnswerScope,
} from './requestState';

function memoryStorage() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    values,
  };
}

const input: AnswerQuestionInput = {
  libraryId: 'lib-a', generationId: 'gen-a', paperIds: ['paper-1', 'paper-2'], query: 'What changed?',
  evidenceTokenBudget: 6000, maxOutputTokens: 1200,
};
const answer: Answer = {
  answer_id: 'answer-result', retrieval_run_id: 'run-1', status: 'answered', citations: [], diagnostics: [],
  scope: { library_id: 'lib-a', generation_id: 'gen-a' },
};

describe('library request isolation', () => {
  it('rejects an A list response after selecting B', () => {
    const guard = new LibraryRequestGuard('lib-a');
    const first = guard.begin('papers');
    guard.switchTo('lib-b');
    const second = guard.begin('papers');
    expect(guard.isLatest(first)).toBe(false);
    expect(guard.isLatest(second)).toBe(true);
    expect(second.libraryId).toBe('lib-b');
  });

  it('accepts only the newest list or answer request within one library', () => {
    const guard = new LibraryRequestGuard('lib-a');
    const first = guard.begin('papers');
    const second = guard.begin('papers');
    const firstAnswer = guard.begin('answer');
    const historyRead = guard.begin('answer');
    expect(guard.isLatest(first)).toBe(false);
    expect(guard.isLatest(second)).toBe(true);
    expect(guard.isLatest(firstAnswer)).toBe(false);
    expect(guard.isLatest(historyRead)).toBe(true);
  });

  it('does not revive old A callbacks after A to B to A', () => {
    const guard = new LibraryRequestGuard('lib-a');
    const oldA = guard.capture();
    guard.switchTo('lib-b');
    guard.switchTo('lib-a');
    expect(guard.isCurrent(oldA)).toBe(false);
    expect(guard.isCurrent(guard.capture())).toBe(true);
  });

  it.each(['upload', 'manual-job', 'job-action:cancel', 'job-action:retry', 'answer'])('invalidates %s success, error and cleanup tokens on library change', (channel) => {
    const guard = new LibraryRequestGuard('lib-a');
    const request = guard.begin(channel);
    guard.switchTo('lib-b');
    expect(guard.isCurrent(request)).toBe(false);
    expect(guard.isLatest(request)).toBe(false);
    const next = guard.begin(channel);
    expect(guard.isLatest(next)).toBe(true);
  });

  it('invalidates in-flight results on unmount without changing library identity', () => {
    const guard = new LibraryRequestGuard('lib-a');
    const old = guard.capture();
    guard.invalidate();
    expect(guard.isCurrent(old)).toBe(false);
    expect(guard.capture().libraryId).toBe('lib-a');
  });
});

describe('persistent answer identity and recovery', () => {
  it.each([1, 4096])('accepts the backend output boundary %i', (maxOutputTokens) => {
    expect(normalizeAnswerInput({ ...input, maxOutputTokens }).maxOutputTokens).toBe(maxOutputTokens);
  });

  it.each([0, -1, 1.5, NaN, Infinity, 4097])('rejects invalid output %s before persistence or sending', (maxOutputTokens) => {
    const storage = memoryStorage();
    const key = vi.fn();
    expect(() => prepareAnswerRequest(storage, { ...input, maxOutputTokens }, key)).toThrow();
    expect(key).not.toHaveBeenCalled();
    expect(storage.values.size).toBe(0);
  });

  it('enforces the existing backend evidence budget range', () => {
    expect(normalizeAnswerInput({ ...input, evidenceTokenBudget: 16000 }).evidenceTokenBudget).toBe(16000);
    expect(() => normalizeAnswerInput({ ...input, evidenceTokenBudget: 16001 })).toThrow('16000');
  });

  it('keeps unknown 1200 input intact when explicitly preparing 4096, and restores the original key', async () => {
    const storage = memoryStorage();
    prepareAnswerRequest(storage, input, () => 'answer-original');
    updateAnswerRequest(storage, 'answer-original', 'unknown');
    const changed = prepareAnswerRequest(storage, { ...input, maxOutputTokens: 4096 }, () => 'answer-new');
    expect(changed.isNew).toBe(true);
    const reloaded = { getItem: storage.getItem, setItem: storage.setItem };
    expect(readAnswerRequests(reloaded).map((record) => [record.key, record.state, record.input.maxOutputTokens])).toEqual([
      ['answer-original', 'unknown', 1200], ['answer-new', 'prepared', 4096],
    ]);
    const sender = vi.fn().mockResolvedValue(answer);
    await sendSavedAnswerRequest(reloaded, 'answer-original', sender);
    expect(sender).toHaveBeenCalledExactlyOnceWith(input, 'answer-original');
    expect(readAnswerRequests(reloaded)[1]).toEqual(changed.record);
  });

  it('normalizes the fixed scope and reuses the original key for equivalent input', () => {
    const storage = memoryStorage();
    const createKey = vi.fn(() => 'answer-original');
    const first = prepareAnswerRequest(storage, input, createKey);
    const second = prepareAnswerRequest(storage, {
      ...input, libraryId: ' lib-a ', generationId: ' gen-a ', paperIds: [' paper-2 ', 'paper-1', 'paper-1'], query: '  What changed?  ',
    }, createKey);
    expect(first.isNew).toBe(true);
    expect(second.isNew).toBe(false);
    expect(second.record).toEqual(first.record);
    expect(createKey).toHaveBeenCalledOnce();
  });

  it('retains the same input and key after a lost response and storage reload', async () => {
    const storage = memoryStorage();
    const original = prepareAnswerRequest(storage, input, () => 'answer-original').record;
    const sender = vi.fn().mockRejectedValueOnce(new TypeError('response lost')).mockResolvedValueOnce(answer);
    await expect(sendSavedAnswerRequest(storage, original.key, sender)).rejects.toThrow('response lost');
    expect(sender).toHaveBeenCalledTimes(1);
    expect(readAnswerRequests(storage)[0].state).toBe('unknown');
    const restored = prepareAnswerRequest(storage, { ...input, paperIds: [...input.paperIds].reverse() }, () => 'answer-unwanted');
    expect(restored.isNew).toBe(false);
    expect(restored.record.key).toBe(original.key);
    await expect(sendSavedAnswerRequest(storage, restored.record.key, sender)).resolves.toEqual(answer);
    expect(sender.mock.calls).toEqual([[input, original.key], [input, original.key]]);
    expect(readAnswerRequests(storage)[0]).toMatchObject({ state: 'completed', answerId: answer.answer_id });
  });

  it('saves before sending and refuses to send if persistence fails', async () => {
    const storage = memoryStorage();
    const record = prepareAnswerRequest(storage, input, () => 'answer-original').record;
    const brokenStorage = { getItem: storage.getItem, setItem: () => { throw new Error('disk full'); } };
    const sender = vi.fn();
    expect(() => prepareAnswerRequest(brokenStorage, { ...input, query: 'Another question?' })).toThrow('disk full');
    await expect(sendSavedAnswerRequest(brokenStorage, record.key, sender)).rejects.toThrow('disk full');
    expect(sender).not.toHaveBeenCalled();
    expect(readAnswerRequests(storage)[0].state).toBe('prepared');
  });

  it('refuses to create a replacement key when recovery data is corrupt', () => {
    const storage = memoryStorage();
    prepareAnswerRequest(storage, input, () => 'answer-original');
    const storageKey = [...storage.values.keys()][0];
    storage.setItem(storageKey, '{broken json');
    const createKey = vi.fn(() => 'answer-unwanted');
    expect(() => prepareAnswerRequest(storage, input, createKey)).toThrow('恢复记录无效');
    expect(createKey).not.toHaveBeenCalled();
  });

  it('rejects altered saved input even when the old fingerprint is retained', () => {
    const storage = memoryStorage();
    const record = prepareAnswerRequest(storage, input, () => 'answer-original').record;
    storage.setItem([...storage.values.keys()][0], JSON.stringify([{ ...record, input: { ...record.input, query: 'tampered' } }]));
    expect(() => readAnswerRequests(storage)).toThrow('恢复记录无效');
  });

  it.each([
    { libraryId: 'lib-b' }, { generationId: 'gen-b' }, { query: 'A new question?' },
    { paperIds: ['paper-1'] }, { evidenceTokenBudget: 7000 }, { maxOutputTokens: 1400 },
  ])('creates a distinct identity when fixed input changes: %j', (change) => {
    const storage = memoryStorage();
    prepareAnswerRequest(storage, input, () => 'answer-original');
    const changed = prepareAnswerRequest(storage, { ...input, ...change }, () => 'answer-new');
    expect(changed.isNew).toBe(true);
    expect(changed.record.key).toBe('answer-new');
  });

  it('keeps unknown identity even if a foreign answer response is returned', async () => {
    const storage = memoryStorage();
    const record = prepareAnswerRequest(storage, input, () => 'answer-original').record;
    await expect(sendSavedAnswerRequest(storage, record.key, vi.fn().mockResolvedValue({ ...answer, scope: { library_id: 'lib-b', generation_id: 'gen-b' } }))).rejects.toThrow('answer_scope_mismatch');
    expect(readAnswerRequests(storage)[0]).toMatchObject({ key: 'answer-original', state: 'unknown' });
  });

  it('requires a fixed generation and finite positive budgets', () => {
    expect(() => normalizeAnswerInput({ ...input, generationId: '' })).toThrow('必须固定');
    expect(() => normalizeAnswerInput({ ...input, evidenceTokenBudget: NaN })).toThrow('正整数');
    expect(() => normalizeAnswerInput({ ...input, maxOutputTokens: 0 })).toThrow('正整数');
    expect(() => verifyAnswerScope(answer, 'lib-a', 'gen-b')).toThrow('answer_scope_mismatch');
  });

  it('does not send an unsaved request key', async () => {
    const sender = vi.fn();
    await expect(sendSavedAnswerRequest(memoryStorage(), 'answer-unsaved', sender)).rejects.toThrow('answer_request_not_saved');
    expect(sender).not.toHaveBeenCalled();
  });

  it('persists submitted state before the transport begins', async () => {
    const storage = memoryStorage();
    const record = prepareAnswerRequest(storage, input, () => 'answer-original').record;
    await sendSavedAnswerRequest(storage, record.key, vi.fn(async () => {
      expect(readAnswerRequests(storage)[0].state).toBe('submitted');
      return answer;
    }));
    updateAnswerRequest(storage, record.key, 'unknown');
    expect(readAnswerRequests(storage)[0].key).toBe(record.key);
  });
});
