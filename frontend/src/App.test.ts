// @vitest-environment jsdom
// 挂载真实 App；仅替换 HTTP 和 PDF.js，不访问模型、上传或外部服务。
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import * as pdfjsLib from 'pdfjs-dist';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Answer, Evidence, PaperList } from './api';
import App from './App';
import { prepareAnswerRequest, readAnswerRequests, updateAnswerRequest } from './requestState';

vi.mock('pdfjs-dist', () => ({
  GlobalWorkerOptions: {},
  getDocument: vi.fn(() => ({ promise: new Promise(() => {}), destroy: vi.fn().mockResolvedValue(undefined) })),
}));

let root: Root | undefined;
let container: HTMLDivElement;
const published: PaperList = {
  library_id: 'library-a', membership_revision: 2, active_generation_id: 'generation-old', next_offset: null,
  items: [{
    paper_id: 'paper-old', title: '已发布论文', status: 'active', pending: true,
    document_version_ids: ['version-old', 'version-new'], selected_document_version_id: 'version-new',
    published_source: { paper_id: 'paper-old', document_version_id: 'version-old', parse_snapshot_id: 'snapshot-old' },
  }],
};

function json(body: unknown): Response { return new Response(JSON.stringify(body), { status: 200 }); }

function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll('button')].find((item) => item.textContent === text);
  if (!found) throw new Error(`找不到按钮：${text}`);
  return found;
}

async function click(element: HTMLElement): Promise<void> { await act(async () => element.click()); }

async function input(id: string, value: string): Promise<void> {
  const element = container.querySelector<HTMLInputElement | HTMLTextAreaElement>(`#${id}`);
  if (!element) throw new Error(`找不到输入：${id}`);
  const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, 'value')?.set?.call(element, value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function mount(): Promise<void> {
  root = createRoot(container);
  await act(async () => root?.render(createElement(App)));
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  localStorage.clear();
  localStorage.setItem('academic-evidence-qa.library-id', 'library-a');
  container = document.createElement('div');
  document.body.append(container);
});

afterEach(async () => {
  if (root) await act(async () => root?.unmount());
  root = undefined;
  document.body.innerHTML = '';
  localStorage.clear();
  vi.restoreAllMocks();
  vi.mocked(pdfjsLib.getDocument).mockClear();
});

describe('旧发布版本的 App 可用性', () => {
  it('默认 4096，可编辑并持久保存实际提交的输出参数', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).includes('/papers?')) return json(published);
      if (url === '/v1/answer') return json({ answer_id: 'answer-new', retrieval_run_id: 'run-new', status: 'insufficient_evidence', citations: [], diagnostics: [], scope: { library_id: 'library-a', generation_id: 'generation-old' }, evidence: [] });
      throw new Error(`未预期请求：${String(url)}`);
    });
    await mount();
    await click(container.querySelector<HTMLInputElement>('.paper-item input')!);
    await click(button('开始问答'));
    const output = container.querySelector<HTMLInputElement>('#max-output-tokens')!;
    expect([output.value, output.min, output.max, output.step, output.required]).toEqual(['4096', '1', '4096', '1', true]);
    await input('question', '说明方法');
    await input('max-output-tokens', '3072');
    await click(button('提交问题'));
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/v1/answer');
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1]?.body)).max_output_tokens).toBe(3072);
    expect(readAnswerRequests(localStorage)[0]).toMatchObject({ state: 'completed', input: { maxOutputTokens: 3072 } });
  });

  it('页面编辑新参数后显式恢复 unknown，仍发送原输入和原 key', async () => {
    const originalInput = { libraryId: 'library-a', generationId: 'generation-old', paperIds: ['paper-old'], query: '原问题', evidenceTokenBudget: 6000, maxOutputTokens: 1200 };
    prepareAnswerRequest(localStorage, originalInput, () => 'answer-original');
    updateAnswerRequest(localStorage, 'answer-original', 'unknown');
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).includes('/papers?')) return json(published);
      throw new Error('response lost');
    });
    await mount();
    await click(button('问答与证据'));
    await input('question', '改后的新问题');
    await input('max-output-tokens', '4096');
    const select = container.querySelector<HTMLSelectElement>('#saved-request')!;
    await act(async () => { select.value = 'answer-original'; select.dispatchEvent(new Event('change', { bubbles: true })); });
    await click(button('恢复原请求（同幂等键）'));
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/v1/answer');
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1]?.body))).toMatchObject({ query: '原问题', max_output_tokens: 1200, generation_id: 'generation-old', paper_ids: ['paper-old'] });
    expect(new Headers(calls[0][1]?.headers).get('Idempotency-Key')).toBe('answer-original');
    expect(readAnswerRequests(localStorage)).toHaveLength(1);
    expect(readAnswerRequests(localStorage)[0]).toMatchObject({ key: 'answer-original', state: 'unknown', input: originalInput });
  });

  it('pending 修订保留可选旧发布来源，明确区分候选选版', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(published));
    await mount();
    expect(container.querySelectorAll('.paper-item')).toHaveLength(1);
    expect(container.querySelector('.paper-item')?.textContent).toContain('发布版本 version-old');
    expect(container.querySelector('.paper-item')?.textContent).toContain('候选选版 version-new（待发布）');
    expect(container.textContent).toContain('已有发布版本继续可问答');
    await click(container.querySelector<HTMLInputElement>('.paper-item input')!);
    expect(button('开始问答').disabled).toBe(false);
    await click(button('开始问答'));
    const generation = container.querySelector<HTMLInputElement>('#generation-id')!;
    expect(generation.value).toBe('generation-old');
    expect(generation.readOnly).toBe(true);
  });

  it.each(['failed', 'cancelled'] as const)('索引 %s 后，旧发布仍可被选择并按原 generation 提问', async (state) => {
    let jobReads = 0;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).includes('/papers?')) return json(published);
      if (url === '/v1/jobs/job-index') {
        jobReads += 1;
        return json({ job_id: 'job-index', library_id: 'library-a', job_type: 'index', state: jobReads === 1 ? 'running' : state, stage: 'finished', error_code: state === 'failed' ? 'index_job_failed' : null });
      }
      if (url === '/v1/answer') return json({ answer_id: 'answer-new', retrieval_run_id: 'run-new', status: 'insufficient_evidence', citations: [], diagnostics: [], scope: { library_id: 'library-a', generation_id: 'generation-old' }, evidence: [] });
      throw new Error(`未预期请求：${String(url)}`);
    });
    await mount();
    await input('saved-job-id', 'job-index');
    await click(button('载入任务状态'));
    expect(jobReads).toBe(2);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/papers?'))).toHaveLength(2);
    expect(container.querySelector('.job-row')?.textContent).toContain(state === 'failed' ? '失败' : '已取消');
    expect(container.querySelectorAll('.paper-item')).toHaveLength(1);
    await click(container.querySelector<HTMLInputElement>('.paper-item input')!);
    await click(button('开始问答'));
    await input('question', '说明旧发布论文的方法');
    await click(button('提交问题'));
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/v1/answer');
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1]?.body))).toMatchObject({
      generation_id: 'generation-old', paper_ids: ['paper-old'], query: '说明旧发布论文的方法',
    });
    expect(container.querySelector('.answer-card')?.textContent).toContain('证据不足');
  });

  it.each([
    ['没有发布来源', { ...published, items: [{ ...published.items[0], published_source: null }] }],
    ['发布快照缺失', { ...published, items: [{ ...published.items[0], published_source: { document_version_id: 'version-old' } }] }],
    ['成员已移除', { ...published, items: [{ ...published.items[0], status: 'removed' }] }],
    ['没有活动代', { ...published, active_generation_id: null }],
  ])('%s 时不可选择或提交新问答', async (_, payload) => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(payload));
    await mount();
    expect(container.querySelectorAll('.paper-item')).toHaveLength(0);
    expect(button('开始问答').disabled).toBe(true);
    await click(button('问答与证据'));
    await input('question', '不应发送的问题');
    expect(button('提交问题').disabled).toBe(true);
    expect(fetchMock.mock.calls.some(([url]) => url === '/v1/answer')).toBe(false);
  });

  it('刷新发现已移除成员后清除旧勾选，不能继续使用失效范围', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(published));
    await mount();
    await click(container.querySelector<HTMLInputElement>('.paper-item input')!);
    expect(button('开始问答').disabled).toBe(false);
    fetchMock.mockResolvedValue(json({ ...published, items: [{ ...published.items[0], status: 'removed' }] }));
    await click(button('载入'));
    expect(container.querySelectorAll('.paper-item')).toHaveLength(0);
    expect(button('开始问答').disabled).toBe(true);
  });
});

describe('历史答案的 App 引用回源', () => {
  it('原答案回读成功后，PDF.js 使用原 pack 来源和成对历史上下文', async () => {
    const evidence: Evidence = {
      evidence_id: 'evidence-old', alias: 'E1', paper_id: 'paper-removed', document_version_id: 'version-history', parse_snapshot_id: 'snapshot-history', quality_flags: [],
      ranges: [{ block_id: 'block-old', block_reading_order: 0, start: 0, end: 4, text: 'text', block_type: 'text', locations: [{ page_index: 2 }], quality_flags: [] }],
    };
    const answer: Answer = {
      answer_id: 'answer-history', retrieval_run_id: 'run-history', status: 'answered', text: '历史正文', citations: [{ alias: 'E1', evidence_id: 'evidence-old' }], diagnostics: [],
      scope: { library_id: 'library-a', generation_id: 'generation-history' }, evidence_pack: { evidence: [evidence] },
    };
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      if (String(url).includes('/papers?')) return json(published);
      if (String(url).startsWith('/v1/answers/answer-history?')) return json(answer);
      if (url === '/v1/evidence/read') return json({
        library_id: 'library-a', generation_id: 'generation-history', retrieval_run_id: 'run-history', evidence_id: evidence.evidence_id,
        paper_id: evidence.paper_id, document_version_id: evidence.document_version_id, parse_snapshot_id: evidence.parse_snapshot_id, items: evidence.ranges,
      });
      throw new Error(`未预期请求：${String(url)}`);
    });
    await mount();
    await input('saved-answer-id', 'answer-history');
    await click(button('载入历史答案'));
    expect(vi.mocked(pdfjsLib.getDocument)).not.toHaveBeenCalled();
    await click(button('E1 · 回读引用'));
    const readCall = fetchMock.mock.calls.find(([url]) => url === '/v1/evidence/read');
    expect(JSON.parse(String(readCall?.[1]?.body))).toMatchObject({ answer_id: 'answer-history', retrieval_run_id: 'run-history', evidence_id: 'evidence-old' });
    expect(vi.mocked(pdfjsLib.getDocument)).toHaveBeenCalledOnce();
    const pdfInput = vi.mocked(pdfjsLib.getDocument).mock.calls[0][0] as { url: string };
    const params = new URL(pdfInput.url, 'http://localhost').searchParams;
    expect(Object.fromEntries(params)).toEqual({
      library_id: 'library-a', generation_id: 'generation-history', paper_id: 'paper-removed',
      document_version_id: 'version-history', parse_snapshot_id: 'snapshot-history', answer_id: 'answer-history', retrieval_run_id: 'run-history',
    });
    expect(container.querySelector<HTMLInputElement>('.page-control input')?.value).toBe('3');
  });
});
