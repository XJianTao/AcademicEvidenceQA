import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import {
  answerEvidence,
  cancelJob,
  createLibrary,
  displayError,
  getJob,
  getAnswer,
  listPapers,
  type Answer,
  type Evidence,
  type EvidenceReadResult,
  type Job,
  type PaperList,
  retryJob,
  uploadDocument,
  shouldPollJob,
} from './api';
import { citationEvidence, evidencePhysicalPages, physicalPageNumber, startJobPolling, statusLabel, verifyEvidenceSelection } from './readerState';
import { LibraryRequestGuard, prepareAnswerRequest, readAnswerRequests, sendSavedAnswerRequest, verifyAnswerScope, type AnswerRequestRecord, type LibraryRequestToken } from './requestState';

type View = 'library' | 'qa';
const storedLibraryKey = 'academic-evidence-qa.library-id';

function readAnswerRecovery() {
  try { return { records: readAnswerRequests(localStorage), error: '' }; }
  catch (error) { return { records: [] as AnswerRequestRecord[], error: displayError(error) }; }
}

function App() {
  const [view, setView] = useState<View>('library');
  const [libraryId, setLibraryId] = useState(() => localStorage.getItem(storedLibraryKey) ?? '');
  const [libraryInput, setLibraryInput] = useState(libraryId);
  const [papers, setPapers] = useState<PaperList | null>(null);
  const [loadingPapers, setLoadingPapers] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [jobError, setJobError] = useState('');
  const [jobIdInput, setJobIdInput] = useState('');
  const trackedJobIds = useRef(new Set<string>());
  const [uploadMessage, setUploadMessage] = useState('');
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [maxOutputTokens, setMaxOutputTokens] = useState('4096');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [answerLibraryId, setAnswerLibraryId] = useState('');
  const [answerIdInput, setAnswerIdInput] = useState('');
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerError, setAnswerError] = useState('');
  const [answerRecovery, setAnswerRecovery] = useState(readAnswerRecovery);
  const [selectedRequestKey, setSelectedRequestKey] = useState('');
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [readResult, setReadResult] = useState<EvidenceReadResult | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState('');
  const [pdfPage, setPdfPage] = useState(1);
  const [activePdf, setActivePdf] = useState('');
  const evidenceRequestRef = useRef(0);
  const requestGuardRef = useRef(new LibraryRequestGuard(libraryId));
  const initialLibraryIdRef = useRef(libraryId);
  const refreshAnswerRecovery = useCallback(() => setAnswerRecovery(readAnswerRecovery()), []);

  const clearAnswer = useCallback(() => {
    evidenceRequestRef.current += 1;
    setAnswer(null); setAnswerLibraryId(''); setSelectedEvidence(null); setReadResult(null);
    setActivePdf(''); setEvidenceError(''); setEvidenceLoading(false); setAnswerError('');
  }, []);

  const trackJobs = useCallback((updates: Job[], token: LibraryRequestToken) => {
    if (!requestGuardRef.current.isCurrent(token)) return false;
    if (updates.some((job) => job.library_id && job.library_id !== token.libraryId)) {
      setJobError('job_library_mismatch');
      return false;
    }
    updates.forEach((job) => trackedJobIds.current.add(job.job_id));
    setJobs((current) => Object.fromEntries([...Object.entries(current), ...updates.map((job) => [job.job_id, job])]));
    return true;
  }, []);

  const loadPapers = useCallback(async (id: string, origin?: LibraryRequestToken) => {
    const targetLibraryId = id.trim();
    const guard = requestGuardRef.current;
    if (!targetLibraryId || (origin && !guard.isCurrent(origin))) return;
    if (!origin) guard.begin('library-selection');
    if (guard.capture().libraryId !== targetLibraryId) {
      // 切库时立即失效全部旧请求；A→B→A 也不能复活第一次 A 的回包。
      guard.switchTo(targetLibraryId);
      clearAnswer(); setAnswerLoading(false); setSelectedPaperIds([]); setJobs({});
      setPapers(null); setJobError(''); setUploadMessage('');
      setSelectedRequestKey(''); trackedJobIds.current.clear();
    }
    const token = guard.begin('papers');
    setLibraryId(targetLibraryId);
    setLibraryInput(targetLibraryId);
    refreshAnswerRecovery();
    setLoadingPapers(true);
    setLibraryError('');
    try {
      localStorage.setItem(storedLibraryKey, targetLibraryId);
      const result = await listPapers(targetLibraryId);
      if (!guard.isLatest(token)) return;
      if (result.library_id && result.library_id !== targetLibraryId) throw new Error('paper_library_mismatch');
      setPapers(result);
    } catch (error) {
      if (!guard.isLatest(token)) return;
      setPapers(null);
      setLibraryError(displayError(error));
    } finally {
      if (guard.isLatest(token)) setLoadingPapers(false);
    }
  }, [clearAnswer, refreshAnswerRecovery]);

  useEffect(() => {
    if (initialLibraryIdRef.current) void loadPapers(initialLibraryIdRef.current);
    return () => { requestGuardRef.current.invalidate(); evidenceRequestRef.current += 1; };
  }, [loadPapers]);

  const activeJobIds = Object.keys(jobs).filter((id) => shouldPollJob(jobs[id].state)).sort().join(',');
  useEffect(() => {
    if (!activeJobIds) return undefined;
    const token = requestGuardRef.current.capture();
    return startJobPolling(activeJobIds.split(','), getJob, (updates) => {
      if (!trackJobs(updates, token)) return;
      if (updates.some((job) => ['succeeded', 'failed', 'cancelled'].includes(job.state))) {
        void loadPapers(token.libraryId, token);
      }
    }, (error) => { if (requestGuardRef.current.isCurrent(token)) setJobError(displayError(error)); });
  }, [activeJobIds, libraryId, loadPapers, trackJobs]);

  const successorJobIds = [...new Set(Object.values(jobs).flatMap((job) => job.index_job_ids ?? []))].filter((id) => !trackedJobIds.current.has(id)).sort().join(',');
  useEffect(() => {
    if (!successorJobIds) return undefined;
    let stopped = false;
    const token = requestGuardRef.current.capture();
    const nextIds = successorJobIds.split(',');
    void Promise.all(nextIds.map(getJob)).then((nextJobs) => {
      if (!stopped) trackJobs(nextJobs, token);
    }).catch((error) => { if (!stopped && requestGuardRef.current.isCurrent(token)) setJobError(displayError(error)); });
    return () => { stopped = true; };
  }, [successorJobIds, libraryId, trackJobs]);

  const publishedPapers = useMemo(
    // pending 属于待发布的整库修订，不能使仍有效的旧发布来源下线。
    () => papers?.active_generation_id ? papers.items.filter((paper) => (
      paper.status === 'active' && paper.published_source?.document_version_id
      && paper.published_source.parse_snapshot_id
      && (!paper.published_source.paper_id || paper.published_source.paper_id === paper.paper_id)
      && (!paper.published_source.generation_id || paper.published_source.generation_id === papers.active_generation_id)
    )) : [],
    [papers],
  );
  const generationId = papers?.active_generation_id ?? '';
  useEffect(() => {
    // 成员移除或来源失效后，旧勾选不能继续进入新的问答请求。
    setSelectedPaperIds((current) => {
      const available = current.filter((id) => publishedPapers.some((paper) => paper.paper_id === id));
      return available.length === current.length ? current : available;
    });
  }, [publishedPapers]);
  const evidenceItems = useMemo(() => (answer ? answerEvidence(answer) : []), [answer]);
  const savedRequests = answerRecovery.records.filter((record) => record.input.libraryId === libraryId);
  const selectedRequest = savedRequests.find((record) => record.key === selectedRequestKey);
  const renderedLibraryToken = requestGuardRef.current.capture();

  const handleCreateLibrary = async () => {
    const guard = requestGuardRef.current;
    const token = guard.begin('library-selection');
    setLibraryError('');
    try {
      const created = await createLibrary();
      if (!guard.isLatest(token)) return;
      await loadPapers(created.library_id);
    } catch (error) { if (guard.isLatest(token)) setLibraryError(displayError(error)); }
  };

  const handleUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const uploadForm = event.currentTarget;
    const form = new FormData(uploadForm);
    const file = form.get('file');
    if (!(file instanceof File) || !file.size) { setUploadMessage('请选择 PDF 文件'); return; }
    if (!libraryId || papers?.membership_revision == null) { setUploadMessage('请先载入 library'); return; }
    const guard = requestGuardRef.current;
    const token = guard.begin('upload');
    setUploadMessage('正在受理上传…');
    try {
      const result = await uploadDocument(file, {
        libraryId: token.libraryId,
        membershipRevision: papers.membership_revision,
        parserBackend: form.get('parser_backend') === 'docling' ? 'docling' : 'mineru_remote',
        allowRemoteProcessing: form.get('allow_remote_processing') === 'on',
      });
      if (!guard.isLatest(token)) return;
      if (result.library_id !== token.libraryId) throw new Error('upload_library_mismatch');
      const jobIds = [result.job_id, result.parse_job_id, result.index_job_id].filter((id): id is string => Boolean(id));
      if (jobIds.length) {
        const fetched = await Promise.all(jobIds.map(getJob));
        if (!guard.isLatest(token) || !trackJobs(fetched, token)) return;
      }
      setUploadMessage(result.status === 'reused' ? '检测到相同文件，已复用既有版本。' : '上传已受理，任务状态会自动更新。');
      await loadPapers(token.libraryId, token);
      if (guard.isLatest(token)) uploadForm.reset();
    } catch (error) { if (guard.isLatest(token)) setUploadMessage(displayError(error)); }
  };

  const handleLoadJob = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!libraryId || !jobIdInput.trim()) return;
    setJobError('');
    const guard = requestGuardRef.current;
    const token = guard.begin('manual-job');
    try {
      const job = await getJob(jobIdInput.trim());
      if (guard.isLatest(token)) trackJobs([job], token);
    } catch (error) { if (guard.isLatest(token)) setJobError(displayError(error)); }
  };

  const handleJobAction = async (job: Job, action: 'cancel' | 'retry', origin: LibraryRequestToken) => {
    const guard = requestGuardRef.current;
    if (!guard.isCurrent(origin) || (job.library_id && job.library_id !== origin.libraryId)) return;
    const token = guard.begin(`job-action:${job.job_id}`);
    try {
      if (action === 'cancel') {
        await cancelJob(job.job_id);
        if (!guard.isLatest(token)) return;
        trackJobs([{ ...job, cancel_requested: true }], token);
        const updated = await getJob(job.job_id);
        if (guard.isLatest(token)) trackJobs([updated], token);
      } else {
        const retry = await retryJob(job.job_id);
        if (!guard.isLatest(token)) return;
        const updated = await getJob(retry.job_id);
        if (guard.isLatest(token)) trackJobs([updated], token);
      }
    } catch (error) { if (guard.isLatest(token)) throw error; }
  };

  const sendAnswerRecord = async (record: AnswerRequestRecord) => {
    const guard = requestGuardRef.current;
    if (record.input.libraryId !== guard.capture().libraryId) return;
    const token = guard.begin('answer');
    clearAnswer(); setAnswerLoading(true); setSelectedRequestKey(record.key);
    try {
      let result: Answer;
      if (record.state === 'completed' && record.answerId) {
        result = await getAnswer(record.input.libraryId, record.answerId);
      } else {
        const pending = sendSavedAnswerRequest(localStorage, record.key);
        refreshAnswerRecovery();
        result = await pending;
      }
      if (!guard.isLatest(token)) return;
      verifyAnswerScope(result, token.libraryId, record.input.generationId);
      setAnswer(result); setAnswerLibraryId(token.libraryId); setView('qa');
    } catch (error) {
      if (guard.isLatest(token)) setAnswerError(`${displayError(error)}。请保留原请求并显式恢复，不会自动换键重试。`);
    } finally {
      if (guard.isLatest(token)) { setAnswerLoading(false); refreshAnswerRecovery(); }
    }
  };

  const handleAnswer = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!libraryId || !generationId.trim() || !selectedPaperIds.length || !query.trim()) return;
    try {
      const prepared = prepareAnswerRequest(localStorage, { libraryId, generationId, paperIds: selectedPaperIds, query, evidenceTokenBudget: 6000, maxOutputTokens: Number(maxOutputTokens) });
      setSelectedRequestKey(prepared.record.key); refreshAnswerRecovery();
      if (!prepared.isNew) {
        setAnswerError('此固定输入已有保存的请求。请使用下方“恢复原请求”读取结果或以原幂等键恢复。');
        return;
      }
      await sendAnswerRecord(prepared.record);
    } catch (error) { setAnswerError(displayError(error)); }
  };

  const handleLoadAnswer = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!libraryId || !answerIdInput.trim()) return;
    const guard = requestGuardRef.current;
    const token = guard.begin('answer');
    clearAnswer(); setAnswerLoading(true);
    try {
      const result = await getAnswer(token.libraryId, answerIdInput.trim());
      if (!guard.isLatest(token)) return;
      verifyAnswerScope(result, token.libraryId);
      setAnswer(result); setAnswerLibraryId(token.libraryId); setView('qa');
    } catch (error) { if (guard.isLatest(token)) setAnswerError(displayError(error)); }
    finally { if (guard.isLatest(token)) setAnswerLoading(false); }
  };

  const handleEvidence = async (evidence: Evidence) => {
    if (!answer || !answerLibraryId) return;
    const token = requestGuardRef.current.capture();
    const requestId = ++evidenceRequestRef.current;
    setSelectedEvidence(null); setReadResult(null); setActivePdf(''); setEvidenceError(''); setEvidenceLoading(true);
    try {
      const selection = await verifyEvidenceSelection(answer, answerLibraryId, evidence);
      if (requestId !== evidenceRequestRef.current || !requestGuardRef.current.isCurrent(token)) return;
      setSelectedEvidence(selection.evidence); setReadResult(selection.readResult);
      setPdfPage(selection.page); setActivePdf(selection.pdfUrl);
    } catch (error) { if (requestId === evidenceRequestRef.current && requestGuardRef.current.isCurrent(token)) setEvidenceError(displayError(error)); }
    finally { if (requestId === evidenceRequestRef.current && requestGuardRef.current.isCurrent(token)) setEvidenceLoading(false); }
  };

  const togglePaper = (paperId: string) => setSelectedPaperIds((current) => current.includes(paperId) ? current.filter((id) => id !== paperId) : [...current, paperId]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div><p className="eyebrow">ACADEMIC EVIDENCE QA</p><h1>固定来源阅读台</h1></div>
        <nav className="tabs" aria-label="主导航">
          <button className={view === 'library' ? 'tab active' : 'tab'} onClick={() => setView('library')}>论文库</button>
          <button className={view === 'qa' ? 'tab active' : 'tab'} onClick={() => setView('qa')}>问答与证据</button>
        </nav>
      </header>

      <main className="workspace">
        <section className="library-bar panel">
          <div className="library-input"><label htmlFor="library-id">Library ID</label><input id="library-id" value={libraryInput} onChange={(event) => setLibraryInput(event.target.value)} placeholder="输入已存在的 library UUID" /><button onClick={() => void loadPapers(libraryInput)} disabled={!libraryInput.trim() || (loadingPapers && libraryInput.trim() === libraryId)}>{loadingPapers ? '载入中…' : '载入'}</button><button className="secondary" onClick={() => void handleCreateLibrary()}>新建 library</button></div>
          {libraryError && <p className="error" role="alert">{libraryError}</p>}
          {papers && <p className="muted">当前修订 {papers.membership_revision ?? '—'} · active generation {papers.active_generation_id ?? '未发布'}</p>}
          <form className="library-input" onSubmit={(event) => void handleLoadAnswer(event)}><label htmlFor="saved-answer-id">历史 Answer ID（只读，不重新生成）</label><input id="saved-answer-id" value={answerIdInput} onChange={(event) => setAnswerIdInput(event.target.value)} placeholder="输入本库已保存的 answer UUID" /><button disabled={!libraryId || !answerIdInput.trim() || answerLoading}>载入历史答案</button></form>
          <form className="library-input" onSubmit={(event) => void handleLoadJob(event)}><label htmlFor="saved-job-id">Job ID（只读载入状态）</label><input id="saved-job-id" value={jobIdInput} onChange={(event) => setJobIdInput(event.target.value)} placeholder="输入本库 parse/index job UUID" /><button disabled={!libraryId || !jobIdInput.trim()}>载入任务状态</button></form>
          {jobError && <p className="error" role="alert">任务状态读取：{jobError}</p>}
          {answerError && view !== 'qa' && <p className="error" role="alert">{answerError}</p>}
          {answerRecovery.error && <p className="error" role="alert">{answerRecovery.error}</p>}
        </section>

        <div className="content-grid">
          <aside className="sidebar">
            <section className="panel"><div className="section-heading"><h2>论文范围</h2><span>{publishedPapers.length} 已发布</span></div>
              {!papers && <p className="empty">载入 library 后选择论文。</p>}
              {papers && !papers.items.length && <p className="empty">此 library 还没有论文。</p>}
              {papers?.items.some((paper) => paper.pending) && <p className="muted">有修订待发布；已有发布版本继续可问答，构建失败或取消不会替换它。</p>}
              {papers && papers.items.length > 0 && !publishedPapers.length && <p className="empty">尚无可问答的已发布论文，请等待索引发布。</p>}
              <div className="paper-list">{publishedPapers.map((paper) => <label className="paper-item" key={paper.paper_id}><input type="checkbox" checked={selectedPaperIds.includes(paper.paper_id)} onChange={() => togglePaper(paper.paper_id)} /><span><strong>{paper.title || '未命名论文'}</strong><small>{paper.paper_id}<br />发布版本 {paper.published_source?.document_version_id}<br />候选选版 {paper.selected_document_version_id ?? '—'}{paper.selected_document_version_id !== paper.published_source?.document_version_id ? '（待发布）' : ''}</small></span></label>)}</div>
            </section>
            <section className="panel upload-panel"><div className="section-heading"><h2>导入 PDF</h2><span>异步任务</span></div><form onSubmit={(event) => void handleUpload(event)}><input name="file" type="file" accept="application/pdf,.pdf" /><label htmlFor="parser-backend">解析方式</label><select id="parser-backend" name="parser_backend" defaultValue="mineru_remote"><option value="mineru_remote">MinerU 远程主解析器</option><option value="docling">Docling 离线降级（需服务可用）</option></select><label className="check"><input name="allow_remote_processing" type="checkbox" />允许上传至远程解析服务</label><button type="submit" disabled={!libraryId || !papers}>提交上传</button></form>{uploadMessage && <p className="muted">{uploadMessage}</p>}</section>
            {Object.values(jobs).length > 0 && <section className="panel"><div className="section-heading"><h2>任务</h2><span>{Object.values(jobs).length}</span></div><div className="job-list">{Object.values(jobs).map((job) => <JobRow key={`${libraryId}:${renderedLibraryToken.epoch}:${job.job_id}`} job={job} onCancel={() => handleJobAction(job, 'cancel', renderedLibraryToken)} onRetry={() => handleJobAction(job, 'retry', renderedLibraryToken)} />)}</div></section>}
          </aside>

          {view === 'library' ? <section className="main-panel panel intro-panel"><p className="eyebrow">READING FLOW</p><h2>从固定版本开始阅读</h2><p>选择已发布论文后进入问答。答案、证据和物理页码都绑定同一 library、generation、版本与解析快照。</p><button onClick={() => setView('qa')} disabled={!selectedPaperIds.length}>开始问答</button></section> : <section className="main-panel qa-layout">
            <section className="panel qa-panel">
              <div className="section-heading"><div><p className="eyebrow">FIXED ANSWER</p><h2>问一个可回源的问题</h2></div><span>{selectedPaperIds.length} 篇范围</span></div>
              <label htmlFor="generation-id">Generation ID（固定已发布版本）</label><input id="generation-id" value={generationId} readOnly placeholder="尚未发布 generation" />
              <form onSubmit={(event) => void handleAnswer(event)}><label htmlFor="question">问题</label><textarea id="question" rows={4} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：两篇论文对该方法的结论有什么共同点？" /><label htmlFor="max-output-tokens">输出 token 上限（含模型推理）</label><input id="max-output-tokens" type="number" min={1} max={4096} step={1} required value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(event.target.value)} /><p className="muted">范围 1–4096，仍受服务端总预算检查。修改后点击提交创建新请求；恢复原请求始终使用其已保存参数和幂等键。</p><button type="submit" disabled={answerLoading || Boolean(answerRecovery.error) || !generationId.trim() || !selectedPaperIds.length || !query.trim()}>{answerLoading ? '检索与回答中…' : '提交问题'}</button></form>
              {answerError && <p className="error" role="alert">{answerError}</p>}
              {savedRequests.length > 0 && <section className="answer-recovery"><h3>已保存的问答请求</h3><label htmlFor="saved-request">选择原请求</label><select id="saved-request" value={selectedRequestKey} onChange={(event) => setSelectedRequestKey(event.target.value)}><option value="">请选择</option>{savedRequests.map((record) => <option key={record.key} value={record.key}>{record.state} · {record.input.query.slice(0, 60)}</option>)}</select>{selectedRequest && <><p className="muted">{selectedRequest.key}<br />固定 generation {selectedRequest.input.generationId} · {selectedRequest.input.paperIds.length} 篇<br />证据预算 {selectedRequest.input.evidenceTokenBudget} · 输出上限 {selectedRequest.input.maxOutputTokens}</p><p>{selectedRequest.input.query}</p><button className="secondary" disabled={answerLoading || Boolean(answerRecovery.error)} onClick={() => void sendAnswerRecord(selectedRequest)}>恢复原请求（同幂等键）</button></>}<p className="muted">恢复使用已保存的范围、问题与预算；已完成请求只读载入，结果未知时以原幂等键核对。</p></section>}
              {answer && <AnswerPanel answer={answer} onSelect={(evidence) => void handleEvidence(evidence)} />}
            </section><EvidencePanel answer={answer} items={evidenceItems} selected={selectedEvidence} readResult={readResult} loading={evidenceLoading} error={evidenceError} onSelect={(evidence) => void handleEvidence(evidence)} onPageSelect={setPdfPage} /><PdfPanel url={activePdf} page={pdfPage} onPageChange={setPdfPage} />
          </section>}
        </div>
      </main>
    </div>
  );
}

function JobRow({ job, onCancel, onRetry }: { job: Job; onCancel: () => Promise<void>; onRetry: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState('');
  const mountedRef = useRef(false);
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; }; }, []);
  const action = async (callback: () => Promise<void>) => { setBusy(true); setActionError(''); try { await callback(); } catch (error) { if (mountedRef.current) setActionError(displayError(error)); } finally { if (mountedRef.current) setBusy(false); } };
  return <div className="job-row"><div><strong>{job.job_type === 'parse' ? '解析' : '索引'} · {job.cancel_requested && shouldPollJob(job.state) ? '取消中' : statusLabel(job.state)}</strong><small>{job.stage || '—'} {job.error_code ? `· ${job.error_code}` : ''}</small>{job.state === 'succeeded' && <small>{job.job_type === 'parse' ? '解析快照已保存，generation 发布仍须索引完成。' : 'generation 已发布。'}</small>}{actionError && <p className="error" role="alert">{actionError}</p>}</div><div>{shouldPollJob(job.state) && <button className="link-button" disabled={busy || job.cancel_requested} onClick={() => void action(onCancel)}>取消</button>}{['failed', 'cancelled'].includes(job.state) && <button className="link-button" disabled={busy} onClick={() => void action(onRetry)}>重试</button>}</div></div>;
}

function AnswerPanel({ answer, onSelect }: { answer: Answer; onSelect: (evidence: Evidence) => void }) {
  return <article className={`answer-card status-${answer.status}`}><div className="section-heading"><h3>{statusLabel(answer.status)}</h3><code>{answer.answer_id}</code></div><p className="muted">固定 generation {answer.scope?.generation_id ?? '缺失'} · retrieval run {answer.retrieval_run_id}</p>{answer.text ? <p className="answer-text">{answer.text}</p> : <p className="empty">服务没有交付答案正文，请查看诊断和证据状态。</p>}<div aria-label="本答案引用">{answer.citations.map((citation) => { const evidence = citationEvidence(answer, citation.evidence_id); return <button className="secondary" key={`${citation.alias}-${citation.evidence_id}`} disabled={!evidence} onClick={() => { if (evidence) onSelect(evidence); }}>{citation.alias}{!evidence ? '（来源缺失）' : ' · 回读引用'}</button>; })}</div><p className="status-line">引用校验：{answer.citation_check_status ?? '未运行'} · 语义支持：{answer.support_check_status ?? '未运行'}</p>{answer.diagnostics.length > 0 && <ul className="diagnostics">{answer.diagnostics.map((diagnostic) => <li key={diagnostic}>{diagnostic}</li>)}</ul>}</article>;
}

function EvidencePanel({ answer, items, selected, readResult, loading, error, onSelect, onPageSelect }: { answer: Answer | null; items: Evidence[]; selected: Evidence | null; readResult: EvidenceReadResult | null; loading: boolean; error: string; onSelect: (evidence: Evidence) => void; onPageSelect: (page: number) => void }) {
  return <section className="panel evidence-panel"><div className="section-heading"><div><p className="eyebrow">SOURCE REGISTRY</p><h2>Evidence / Citation</h2></div><span>{items.length} 条证据</span></div>{!answer && <p className="empty">提交问题后，这里只显示该答案绑定的 EvidencePack。</p>}{answer && !items.length && <p className="empty">当前答案没有可回读证据。</p>}<div className="evidence-list">{items.map((evidence) => <button className={selected?.evidence_id === evidence.evidence_id ? 'evidence-card selected' : 'evidence-card'} key={evidence.evidence_id} onClick={() => onSelect(evidence)}><div><strong>{evidence.alias}</strong><span>{evidence.ranges.length} 个原文区间</span></div><small>paper {evidence.paper_id} · 物理页 {evidencePhysicalPages(evidence).join(', ') || '—'}</small><small>version {evidence.document_version_id}<br />snapshot {evidence.parse_snapshot_id}</small><em>{evidence.quality_flags.join(', ') || 'quality flags: none'}</em></button>)}</div>{loading && <p className="muted">正在核验固定来源…</p>}{error && <p className="error" role="alert">{error}</p>}{readResult && selected && <div className="read-result"><div className="section-heading"><h3>来源回读</h3><span>{(readResult.items ?? readResult.ranges ?? []).length} 区间</span></div><div aria-label="证据物理页跳转">{evidencePhysicalPages(selected).map((page) => <button className="secondary" key={page} onClick={() => onPageSelect(page)}>物理页 {page}</button>)}</div>{(readResult.items ?? readResult.ranges ?? []).map((range) => <blockquote key={`${range.block_id}-${range.start}`}>{range.text}<small>物理页 {physicalPageFromRange(range)}</small></blockquote>)}</div>}</section>;
}

function physicalPageFromRange(range: Evidence['ranges'][number]): string {
  return String(physicalPageNumber(range) ?? '—');
}

function PdfPanel({ url, page, onPageChange }: { url: string; page: number; onPageChange: (page: number) => void }) {
  const [pdfError, setPdfError] = useState('');
  const [pdfLoading, setPdfLoading] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => { pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl; }, []);
  useEffect(() => {
    if (!url || !canvasRef.current) return undefined;
    let cancelled = false;
    const documentTask = pdfjsLib.getDocument({ url, rangeChunkSize: 64 * 1024 });
    let renderTask: ReturnType<pdfjsLib.PDFPageProxy['render']> | undefined;
    const renderPage = async () => {
      setPdfLoading(true); setPdfError('');
      try {
        const document = await documentTask.promise;
        if (cancelled) { await documentTask.destroy(); return; }
        if (page > document.numPages) onPageChange(document.numPages);
        const pdfPage = await document.getPage(Math.min(page, document.numPages));
        const viewport = pdfPage.getViewport({ scale: 1.35 });
        const canvas = canvasRef.current;
        if (!canvas || cancelled) { await document.cleanup(); await documentTask.destroy(); return; }
        const context = canvas.getContext('2d');
        if (!context) throw new Error('canvas_context_unavailable');
        canvas.width = viewport.width; canvas.height = viewport.height;
        renderTask = pdfPage.render({ canvas, canvasContext: context, viewport });
        await renderTask.promise;
        await document.cleanup(); await documentTask.destroy();
      } catch (error) {
        if (!cancelled) setPdfError(error instanceof Error ? error.message : 'PDF 文件读取失败，请核对固定版本和快照身份。');
      } finally { if (!cancelled) setPdfLoading(false); }
    };
    void renderPage();
    return () => {
      cancelled = true;
      renderTask?.cancel();
      void documentTask.destroy().catch(() => undefined);
    };
  }, [url, page, onPageChange]);
  return <section className="panel pdf-panel"><div className="section-heading"><div><p className="eyebrow">FIXED PDF</p><h2>物理页阅读</h2></div><label className="page-control">页码 <input type="number" min={1} value={page} onChange={(event) => onPageChange(Math.max(1, Number(event.target.value) || 1))} /></label></div>{url ? <><div className="pdf-toolbar"><span>{pdfLoading ? '正在按 Range 读取页面…' : `第 ${page} 页`}</span><span>PDF.js · HTTP Range · 来源页码来自 Evidence locations。</span></div><div className="pdf-canvas-wrap"><canvas ref={canvasRef} aria-label={`固定来源 PDF 第 ${page} 页`} /></div>{pdfError && <p className="error" role="alert">{pdfError}</p>}</> : <p className="empty">点击 Evidence 后打开固定 PDF。来源缺少 document_version 或 parse_snapshot 时不会猜测文件。</p>}</section>;
}

export default App;
