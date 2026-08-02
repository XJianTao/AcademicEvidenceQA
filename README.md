# AcademicEvidenceQA

面向科研论文的证据检索服务，为学术问答 Agent 提供 PDF 导入、结构化解析、混合检索、证据回读与来源追溯能力。

**AcademicEvidenceQA 与 academic_qa_harness 共同组成一个证据驱动的学术问答项目。** `academic_qa_harness` 是项目主入口，负责 Agent 调度、对话、工具调用和用户界面；本仓库作为外置 RAG 服务，负责论文数据与证据。两者通过 HTTP 协作，可分别运行和维护。

## 服务能力

- **论文导入与解析**：接收 PDF，进行大小、页数与解析资源预检，调用 MinerU 官方在线 API，保存解析产物并归一化为可索引内容。
- **论文标题提取**：从结构化解析内容中识别唯一且位于第一页的非空一级标题候选，保留原文大小写和符号，不将上传文件名当作论文标题。
- **混合检索**：结合 BGE-M3 向量召回与 BM25 词法召回，通过 RRF 融合候选，再由 BGE reranker 重排。
- **证据组织与回读**：按范围与预算构建 EvidencePack，支持读取证据正文及来源位置，供 Agent 按需补充上下文。
- **可追溯来源**：证据绑定论文版本、解析快照及索引发布代，支持 PDF 原文回读，减少论文更新后引用指向发生漂移的问题。
- **异步任务与恢复**：解析和索引交由独立 Worker 执行，提供任务查询、取消、有限重试、租约与心跳机制。
- **可选独立问答**：配置 OpenAI-compatible 生成服务后可使用 `/v1/answer`；Agent 集成主要使用检索与回读能力，生成模型由应用层单独配置。

## 架构与职责

```text
academic_qa_harness（Agent 与主界面）
                  │ HTTP
                  ▼
AcademicEvidenceQA API ── 本地 Embedding / Reranker
          │                         ▲
          │                    内部推理接口
          ▼                         │
PostgreSQL ◄─────────────── 独立 Worker ── MinerU 在线 API
          │                         │
          └── 本地文件存储           └── Qdrant 向量索引
```

PostgreSQL 保存论文、版本、解析快照、任务和发布状态；Qdrant 保存向量索引；本地文件存储保存 PDF、解析产物及相关文件。索引构建完成后发布新的 generation，检索依据已发布版本执行。

API 进程集中加载 embedding 与 reranker，Worker 通过本机内部接口复用模型，避免重复加载权重。Docker Compose 仅管理 PostgreSQL 和 Qdrant，API 与 Worker 在本机 Python 环境中运行。

## 本地启动

以下命令在仓库根目录使用 PowerShell 执行。当前 PDF 资源隔离实现依赖 Windows Job Object，建议在 Windows 上运行完整导入链路；其他系统不应直接视为已支持。

准备条件：

- Python 3.12，以及可用的 Docker Compose。
- 本地完整的 `BAAI/bge-m3` 与 `BAAI/bge-reranker-large` 模型目录。
- MinerU 在线 API token，以及允许访问 API、上传地址和结果下载 CDN 的网络。

### 1. 安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

已有 `.env` 时保留并编辑现有文件。项目提供 `uv.lock`，使用 uv 的开发者也可通过锁文件安装依赖。模型权重需另行准备，不包含在仓库中。

### 2. 配置本地环境

在 `.env` 中调整下列配置；模型路径须替换为实际目录：

```dotenv
POSTGRES_USER=academic
POSTGRES_PASSWORD=your_local_database_password
POSTGRES_DB=academic_evidence
POSTGRES_PORT=55432
DATA_ROOT=data
QDRANT_URL=http://127.0.0.1:56333
MODEL_API_URL=http://127.0.0.1:8000
BGE_M3_MODEL_PATH=E:/ai_models/bge-m3
RERANKER_MODEL_PATH=E:/ai_models/bge-reranker-large
MINERU_API_TOKEN=your_mineru_token
ACADEMIC_HOST_TOKEN=your_shared_host_token
```

`ACADEMIC_HOST_TOKEN` 用于可信 Agent 宿主的交付登记与历史来源访问，应与 `academic_qa_harness` 的宿主配置一致。真实凭据放在私有 `.env` 或环境变量中。

服务支持可选的 `DATABASE_URL`；设置时必须使用 `postgresql+psycopg` 驱动，它优先于分项 PostgreSQL 配置。API 与 Worker 的数据库名、数据目录必须一致；若调整 API 端口，也需同步修改 `MODEL_API_URL`。

### 3. 启动数据库并迁移

```powershell
docker compose up -d
docker compose ps
python -m alembic upgrade head
```

等待 PostgreSQL 就绪后执行迁移。Compose 在首次初始化时创建 `POSTGRES_DB` 指定的数据库；Alembic 创建表结构，不创建数据库。已有 Docker 数据卷不会因修改 `.env` 自动重建用户或数据库。

### 4. 启动 API

```powershell
python -m academic_evidence_qa.api.app --database-name academic_evidence --data-root data --port 8000
```

首次启动需要加载本地模型。就绪后可打开 `http://127.0.0.1:8000/docs` 查看 OpenAPI 文档；`GET /internal/models` 可用于查看模型服务状态。

### 5. 启动 Worker

另开终端，进入同一仓库并激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m academic_evidence_qa.worker.main --database-name academic_evidence --data-root data --with-parsing --model-api-url http://127.0.0.1:8000
```

API 应先于 Worker 启动。`--with-parsing` 启用导入任务的解析调度；上传请求还必须明确允许远程处理。使用 MinerU 会将 PDF 发送到外部服务，请仅提交有权外发的论文。

## 接入学术 Agent

在 `academic_qa_harness` 中将 RAG 服务地址配置为 `http://127.0.0.1:8000`，并配置对应的宿主 token。论文上传、范围选择、对话及引用浏览建议通过该项目的主界面操作。

主要 HTTP 接口如下，具体字段以源码中的请求模型及运行服务的 OpenAPI 文档为准：

| 接口 | 用途 |
|---|---|
| `POST /v1/libraries` | 创建论文库 |
| `POST /v1/documents` | 上传 PDF 并受理异步任务，要求 `Idempotency-Key` |
| `GET /v1/jobs/{job_id}` | 查询解析或索引任务 |
| `POST /v1/jobs/{job_id}/cancel`、`/retry` | 取消或重试任务 |
| `GET /v1/papers`、`POST /v1/papers/find` | 查询论文目录与发现论文 |
| `POST /v1/retrieve` | 检索并构建证据包 |
| `POST /v1/evidence/read` | 回读证据内容 |
| `GET /v1/documents/{document_version_id}/file` | 读取固定论文版本的 PDF |
| `POST /v1/agent-deliveries` | 登记 Agent 交付及来源身份 |
| `POST /v1/answer` | 可选的独立证据问答 |

一次典型使用流程是：创建论文库 → 上传论文 → Worker 解析并构建索引 → 发布索引版本 → 在 Agent 中选择论文范围 → 检索、回读证据并生成回答。上传受理成功不代表论文已经可检索，需等待对应任务及索引发布完成。

使用独立 `/v1/answer` 时，另行配置 `GENERATION_BASE_URL`、`GENERATION_MODEL` 与 `GENERATION_API_KEY`。这组配置不与 Harness 的模型配置自动共享。

## 数据与运行边界

- 默认数据目录为 `data/`，可由启动参数指定。PDF 与解析产物可在本地存储中找到；数据库记录保存对应文件键和版本身份。
- PostgreSQL 与 Qdrant 数据默认保存在 Compose 的命名卷中。迁移或备份应同时考虑数据库、向量存储和本地文件目录。
- 论文从库中移除采用成员状态管理，不等同于物理删除所有历史版本和文件。
- 当前面向本机单用户使用，API 默认监听 `127.0.0.1`；宿主 token 不构成覆盖全部接口的多用户认证体系。
- 当前管理上传接口使用 MinerU 在线解析，尚不支持通过该接口使用 Docling 离线导入。
- 引用身份与来源可追溯不等于已验证回答语义正确；论文结论仍应结合原文核对。

若任务返回 `remote_download_transport_failure`，应检查结果 CDN 的网络可达性。在线解析完成与结果 ZIP 下载是不同阶段，API 可访问并不保证结果下载可用。处理失败任务时保留 `job_id` 与错误码，并通过任务接口查看状态。

## 目录说明

```text
src/academic_evidence_qa/  API、解析、索引、检索、证据与存储实现
migrations/               PostgreSQL 数据库迁移
prompts/                  独立问答与引用修订的运行时提示词
scripts/                  辅助验证、诊断及恢复脚本
frontend/                 独立论文阅读界面代码
.env.example              本地配置模板
docker-compose.yml        PostgreSQL 与 Qdrant 配置
```

完整产品的主要交互入口位于 `academic_qa_harness`。本仓库的 `frontend/` 是独立阅读界面，不是运行 RAG API 与 Worker 的必要条件。发布内容不包含模型权重、论文数据、运行产物及内部实施记录。
