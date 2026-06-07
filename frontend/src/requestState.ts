import { answerQuestion, type Answer, type AnswerQuestionInput } from './api';

export interface LibraryRequestToken {
  libraryId: string;
  epoch: number;
}

export interface LatestRequestToken extends LibraryRequestToken {
  channel: string;
  sequence: number;
}

export class LibraryRequestGuard {
  private epoch = 0;
  private sequences = new Map<string, number>();

  constructor(private libraryId: string) {}

  capture(): LibraryRequestToken { return { libraryId: this.libraryId, epoch: this.epoch }; }

  switchTo(libraryId: string): LibraryRequestToken {
    if (libraryId !== this.libraryId) {
      this.libraryId = libraryId;
      this.invalidate();
    }
    return this.capture();
  }

  invalidate(): void { this.epoch += 1; this.sequences.clear(); }

  begin(channel: string): LatestRequestToken {
    const sequence = (this.sequences.get(channel) ?? 0) + 1;
    this.sequences.set(channel, sequence);
    return { ...this.capture(), channel, sequence };
  }

  isCurrent(token: LibraryRequestToken): boolean {
    return token.libraryId === this.libraryId && token.epoch === this.epoch;
  }

  isLatest(token: LatestRequestToken): boolean {
    return this.isCurrent(token) && this.sequences.get(token.channel) === token.sequence;
  }
}

export interface AnswerRequestRecord {
  key: string;
  fingerprint: string;
  input: AnswerQuestionInput;
  state: 'prepared' | 'submitted' | 'unknown' | 'completed';
  answerId?: string;
}

type RequestStorage = Pick<Storage, 'getItem' | 'setItem'>;
const answerRequestStorageKey = 'academic-evidence-qa.answer-requests.v1';

export function normalizeAnswerInput(input: AnswerQuestionInput): AnswerQuestionInput {
  const normalized = {
    libraryId: input.libraryId.trim(),
    generationId: input.generationId?.trim(),
    paperIds: [...new Set(input.paperIds.map((id) => id.trim()).filter(Boolean))].sort(),
    query: input.query.trim(),
    evidenceTokenBudget: input.evidenceTokenBudget,
    maxOutputTokens: input.maxOutputTokens,
  };
  if (!normalized.libraryId || !normalized.generationId || !normalized.paperIds.length || !normalized.query) {
    throw new Error('问答必须固定 library、generation、论文范围和问题。');
  }
  if (![normalized.evidenceTokenBudget, normalized.maxOutputTokens].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error('问答预算必须为正整数。');
  }
  if (normalized.evidenceTokenBudget > 16000 || normalized.maxOutputTokens > 4096) {
    throw new Error('证据预算不得超过 16000，输出 token 上限不得超过 4096。');
  }
  return normalized;
}

export function readAnswerRequests(storage: RequestStorage): AnswerRequestRecord[] {
  const raw = storage.getItem(answerRequestStorageKey);
  if (!raw) return [];
  try {
    const records = JSON.parse(raw) as AnswerRequestRecord[];
    if (!Array.isArray(records) || records.some((record) => (
      typeof record.key !== 'string' || !record.key.startsWith('answer-') || record.key.length > 128
      || !['prepared', 'submitted', 'unknown', 'completed'].includes(record.state)
      || (record.answerId !== undefined && (typeof record.answerId !== 'string' || !record.answerId))
      || (record.state === 'completed' && !record.answerId)
      || record.fingerprint !== JSON.stringify(record.input)
      || record.fingerprint !== JSON.stringify(normalizeAnswerInput(record.input))
    )) || new Set(records.map((record) => record.key)).size !== records.length
      || new Set(records.map((record) => record.fingerprint)).size !== records.length) throw new Error('invalid_record');
    return records;
  } catch {
    // 损坏的恢复记录不能被静默丢弃，否则未知请求会获得新的付费身份。
    throw new Error('本地问答恢复记录无效；已停止新请求，请保留记录并人工核对。');
  }
}

export function prepareAnswerRequest(
  storage: RequestStorage,
  input: AnswerQuestionInput,
  createKey: () => string = () => `answer-${crypto.randomUUID()}`,
): { record: AnswerRequestRecord; isNew: boolean } {
  const normalized = normalizeAnswerInput(input);
  const fingerprint = JSON.stringify(normalized);
  const records = readAnswerRequests(storage);
  const existing = records.find((record) => record.fingerprint === fingerprint);
  if (existing) return { record: existing, isNew: false };
  const record: AnswerRequestRecord = { key: createKey(), fingerprint, input: normalized, state: 'prepared' };
  if (!record.key.startsWith('answer-') || record.key.length > 128 || records.some((item) => item.key === record.key)) {
    throw new Error('answer_request_key_invalid');
  }
  // 必须先持久化原请求和 key，写入失败时调用方不得发送生成请求。
  storage.setItem(answerRequestStorageKey, JSON.stringify([...records, record]));
  return { record, isNew: true };
}

export function updateAnswerRequest(
  storage: RequestStorage,
  key: string,
  state: AnswerRequestRecord['state'],
  answerId?: string,
): AnswerRequestRecord {
  const records = readAnswerRequests(storage);
  const record = records.find((item) => item.key === key);
  if (!record) throw new Error('answer_request_not_saved');
  record.state = state;
  if (answerId) record.answerId = answerId;
  storage.setItem(answerRequestStorageKey, JSON.stringify(records));
  return record;
}

export function verifyAnswerScope(answer: Answer, libraryId: string, generationId?: string): void {
  if (answer.scope?.library_id !== libraryId || !answer.scope.generation_id
    || (generationId !== undefined && answer.scope.generation_id !== generationId)) {
    throw new Error('answer_scope_mismatch');
  }
}

export async function sendSavedAnswerRequest(
  storage: RequestStorage,
  key: string,
  sender: typeof answerQuestion = answerQuestion,
): Promise<Answer> {
  // 发送前再次读取已保存输入，不使用页面上已被编辑的范围或问题。
  const record = updateAnswerRequest(storage, key, 'submitted');
  try {
    const answer = await sender(record.input, record.key);
    verifyAnswerScope(answer, record.input.libraryId, record.input.generationId);
    updateAnswerRequest(storage, key, 'completed', answer.answer_id);
    return answer;
  } catch (error) {
    try {
      updateAnswerRequest(storage, key, 'unknown');
    } catch {
      // submitted 记录与原 key 仍在；不得因完成状态写入失败创建第二个身份。
      throw new Error('请求结果未能保存；请保留本地记录并用原幂等键恢复。');
    }
    throw error;
  }
}
