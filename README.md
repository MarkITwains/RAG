<h1 align="center">PCB-RAG</h1>

<p align="center"><b>把 PCB 设计规范、工艺资料和工程文档变成能提问、带出处的知识库</b></p>

<p align="center">
  Web 问答 · HTTP API · Dify 外部知识库 · Python / Agent 元件
</p>

<p align="center">
  <a href="https://github.com/MarkITwains/RAG/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/MarkITwains/RAG/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="LlamaIndex" src="https://img.shields.io/badge/Built%20with-LlamaIndex-6E56CF">
  <img alt="Milvus" src="https://img.shields.io/badge/Vector%20Store-Milvus-00A1EA">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">博客文章：<a href="https://blog.eecs.top/index.php/archives/3/">https://blog.eecs.top/index.php/archives/3/</a></p>

---

把你的 PCB 文档放进来，入库一次，就可以：

- 在网页里直接提问。答案带 `[n]` 角标，点一下就能看到原文。
- 把它配成 Dify 的外部知识库，接进工作流或对话应用。
- 通过 HTTP API 或 Python 接口，给自己的系统或 Agent 当检索工具。

想了解检索原理、架构、已知问题和修复记录，请看 [原理与设计](docs/ARCHITECTURE.md)。

## 目录

- [适用场景](#适用场景)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
  - [Web 界面](#web-界面)
  - [HTTP API](#http-api)
  - [接入 Dify](#接入-dify)
  - [在 Python / Agent 里调用](#在-python--agent-里调用)
- [日常操作](#日常操作)
- [按需开启的功能](#按需开启的功能)
- [选择模型后端](#选择模型后端)
- [上线前检查](#上线前检查)
- [配置参考](#配置参考)
- [评测](#评测)
- [常见问题](#常见问题)
- [更多文档](#更多文档)
- [公开仓库说明](#公开仓库说明)
- [License](#license)

## 适用场景

| 场景 | 怎么用 | 入口 |
| --- | --- | --- |
| 工程师查规范、查工艺参数 | 在网页里提问，对照引用原文 | `http://localhost:8000/` |
| 在 Dify 里做 PCB 助手 | 配成外部知识库 | `POST /retrieval` |
| 在业务系统里嵌入问答 | 调用问答接口，支持流式输出 | `POST /api/ask`、`POST /api/ask/stream` |
| 自研 Agent 需要查资料 | 直接 import，或用 function calling | `pcb_rag.engine.KnowledgeBase` |
| 多个团队共用一个库 | 打开租户 / 角色权限 | `ACL_ENABLED=1` |

## 快速开始

### 前置要求

- Python 3.10+
- Docker 与 Docker Compose（用于启动 Milvus 向量库）
- 模型服务（二选一）：
  - API：任意 OpenAI 兼容服务的 `BASE_URL` 与 `API_KEY`（推荐，不需要显卡）
  - 本地：安装并启动 [Ollama](https://ollama.com/)

> Windows 用户请在 WSL 里运行。从 PowerShell 启动、停止服务的方法见 [WSL 运行说明](docs/WSL_API.md)。

### 1. 安装

```bash
bash scripts/setup.sh
```

这个脚本会创建 `.venv`、安装依赖、从 `.env.example` 生成 `.env`、建好数据目录、启动 Milvus，最后做一次环境自检。

WSL / Linux 上的 Python 3.12 可以复用已验证的依赖版本：

```bash
python -m pip install -c requirements/constraints-wsl-py312.txt -e ".[runtime,ollama,dev]"
```

要用本地 torch 精排时，再装 `pip install -e ".[local_models]"`。

### 2. 填写 `.env`

最少要填模型接口和服务口令。下面是全部走 API 的例子（以硅基流动为例）：

```bash
AI_API_ONLY=1
LLM_BACKEND=api
EMBED_BACKEND=api
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=Qwen/Qwen3-8B
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024

RERANK_BACKEND=api
RERANK_API_URL=https://api.jina.ai/v1/rerank
RERANK_API_KEY=xxxx
RERANK_API_MODEL=jina-reranker-v2-base-multilingual

# 服务口令：不设置或用占位值 change-me 时，服务拒绝启动
DIFY_API_TOKEN=换成一串随机字符
```

想用本地模型或混搭，见 [选择模型后端](#选择模型后端)。

填好后自检一次，它会列出缺的配置：

```bash
python scripts/check_env.py
```

### 3. 放入文档

把你有权使用的文档放进 `data/clear_docs/`，子目录也会读取。支持 PDF、Word（`.docx` / `.doc`）和 `.txt` / `.md` 等文本文件。

> 入库时默认会修复 `.txt` 文件的编码，并**直接改写原文件**。不想改原文件，请设 `AUTO_FIX_ENCODING=0`。

公开仓库不包含任何原始语料。

### 4. 入库

```bash
bash scripts/run_ingest.sh      # 等价：python -m pcb_rag.ingest
```

入库会先准备新版本，校验通过后才发布。中途失败时，旧版本照常可用。只想预览要处理哪些文件，先设 `INGEST_PLAN_ONLY=1` 再运行。

### 5. 开始提问

命令行问答：

```bash
bash scripts/run_query.sh       # 等价：python -m pcb_rag.query
```

或者启动服务，用浏览器打开 `http://localhost:8000/`：

```bash
bash scripts/serve_api.sh       # 默认端口 8000，可用 API_PORT 修改
```

可以试试这些问题：

```text
4 层 PCB 的阻抗控制需要关注哪些参数？
Altium Designer 中如何处理高速差分线等长？
```

## 使用方式

下面四种方式共用同一套检索和同一份索引，可以同时使用。

### Web 界面

服务启动后访问 `http://localhost:8000/`。页面是单个静态文件（`src/pcb_rag/webui/index.html`），不需要构建，离线部署也能用。

| 能力 | 说明 |
| --- | --- |
| 流式问答 | 检索完先列出证据，答案逐段出现 |
| 证据溯源 | 列出原文片段、章节、得分；答案里的 `[n]` 可点击定位 |
| 检索参数 | 召回条数、相似度阈值可以在页面上直接调 |
| 多轮对话 | 自动带上最近 6 轮问答，后续问题可以说"它"、"上面那个" |
| 状态指示 | 顶栏显示索引、词法索引、精排是否就绪 |

第一次提问如果提示 401，点右上角「设置」，填入 `.env` 里的 `DIFY_API_TOKEN`。口令只存在你的浏览器里。

管理页在 `/manage`。可以查看已发布和待发布的版本、每个文档的处理阶段和失败原因，也可以重试失败的入库。开了权限控制时，管理操作需要管理员身份。

### HTTP API

所有业务接口都要带 `Authorization: Bearer <DIFY_API_TOKEN>`。完整的交互式文档在 `http://localhost:8000/docs`。

**问答：**

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Authorization: Bearer $DIFY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "4 层 PCB 的阻抗控制需要关注哪些参数？", "top_k": 5}'
```

返回里最常用的字段：

| 字段 | 内容 |
| --- | --- |
| `answer` | 答案文本，带 `[n]` 引用角标 |
| `records` | 引用到的原文片段，含来源文件、章节、页码 |
| `citations` | 答案里每个 `[n]` 对应的证据 |
| `degraded` | 哪一路没跑起来（为空表示全链路正常） |
| `revision` | 本次使用的知识库版本 |

**多轮对话：** 有两种做法。

- 自己保存历史：调 `/api/ask`，在 `chat_history` 里传 `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`。
- 让服务端保存：调 `/api/chat`。第一次不传 `session_id`，之后用返回的 `session_id` 继续问。

**全部接口：**

| 接口 | 用途 |
| --- | --- |
| `POST /api/ask` | 检索 + 生成答案，可传对话历史 |
| `POST /api/ask/stream` | 流式问答（SSE）：先推证据，再逐段推答案 |
| `POST /api/ask/plain` | 只返回纯文本答案，适合 Dify 工作流 |
| `POST /api/chat` | 服务端记住会话的问答，返回 `session_id` |
| `POST /api/chat/plain` | 同上，只返回纯文本 |
| `POST /retrieval` | Dify 外部知识库接口，只检索不生成 |
| `POST /api/session/create`、`GET /api/sessions` | 创建、列出会话 |
| `GET /api/session/{id}`、`DELETE /api/session/{id}` | 查看、删除会话 |
| `GET /api/admin/status`、`GET /api/admin/documents` | 查看发布版本和文档处理状态 |
| `POST /api/admin/retry` | 重试未完成的入库 |
| `GET /api/graph/neighbors` | 查看某个实体的图谱关系（需开启 GraphRAG） |
| `GET /health` | 健康检查，给负载均衡和探针用，不需要口令 |
| `GET /health/detail` | 完整配置摘要，需要口令 |
| `GET /metrics` | 请求计数和延迟分位（p50 / p95 / p99） |

### 接入 Dify

先启动服务（`bash scripts/serve_api.sh`），再在 Dify 的「外部知识库」里填：

| 配置项 | 值 |
| --- | --- |
| URL | `http://<你的服务器地址>:8000/retrieval` |
| API Key | `Bearer <你的 DIFY_API_TOKEN>` |

想让 Dify 直接拿到生成好的答案，可以在工作流里用 HTTP 节点调 `/api/ask/plain`。详细步骤见 [Dify 接入指南](docs/DIFY_INTEGRATION_GUIDE.md)。

### 在 Python / Agent 里调用

不经过 HTTP，直接在代码里用：

```python
from pcb_rag.engine import KnowledgeBase

kb = KnowledgeBase()
kb.ensure_ready()                                  # 首次加载需要几十秒

hit = kb.retrieve("4 层板阻抗控制关注什么", top_k=8)
for ev in hit.evidence:
    print(ev.source_name, ev.section, ev.score)    # 逐条证据，不是拼好的字符串

if hit.degraded:                                   # 有一路没跑起来时要处理
    print("降级：", hit.degraded)

ans = kb.answer("那 6 层板呢？", history=[
    {"role": "user", "content": "4 层板阻抗控制关注什么"},
    {"role": "assistant", "content": "..."},
])
print(ans.answer)
```

| 要做的事 | 接口 |
| --- | --- |
| 只检索证据 | `retrieve()` / `aretrieve()` |
| 检索并生成答案 | `answer()` / `aanswer(history=[...])` |
| 查实体的图谱邻域 | `graph_neighbors()` |
| 检查各组件状态 | `health()` |
| 给 LLM 做 function calling | `tool_specs()` / `call_tool()` / `acall_tool()` |

元件不保存会话，历史由调用方传入。可运行的示例：

```bash
python examples/agent_toolkit_demo.py --mode direct --query "4 层板阻抗控制关注什么"
python examples/agent_toolkit_demo.py --mode tools    # 打印 function calling 定义
python examples/agent_toolkit_demo.py --mode react    # 文本协议的 agent 循环
```

更多说明见 [examples/README.md](examples/README.md)。

## 日常操作

### 新增或更新文档

1. 把新文件放进 `data/clear_docs/`，或直接替换旧文件。
2. 重新运行 `bash scripts/run_ingest.sh`。只有内容变了的文件会重新处理。
3. 校验通过后新版本自动发布。正在运行的服务会切到新版本，不用重启。

前提是服务和入库使用同一个 `KB_STATE_PATH`。

### 查看入库状态、重试失败

打开 `/manage` 页面，或调 `GET /api/admin/status` 和 `GET /api/admin/documents`。失败的入库可以在页面上点重试，或调 `POST /api/admin/retry`。

### 从旧集合迁移

旧集合要先迁移，才能用新版的父块、来源信息和图索引。迁移默认只预览，不写数据库：

```bash
python -m pcb_rag.migrate --target-collection pcb_kb_v2             # 预览
python -m pcb_rag.migrate --target-collection pcb_kb_v2 --execute   # 建新集合并输出验证报告
```

确认报告没问题后，加 `--execute --publish` 切换到新集合。之后把 `.env` 里的 `COLLECTION` 改成新集合名。旧集合和快照会保留，可以随时切回。

需要核对旧评测真值时，加 `--mapping-from 旧词法缓存.jsonl --truth-ids 旧chunk编号数组.json`。完整操作顺序见 [修复与验证报告](docs/LIGHTRAG_FIXES_WSL_20261023.md)。

### 全量重建

用新集合名走一遍上面的迁移流程。`INGEST_OVERWRITE=1` 对非空集合会直接报错，不能用来清空重建。

### 补建图谱

已有新版快照时，下面这条命令会补建图谱并直接发布。源文件和切块配置没变时，会复用已有的切块和向量：

```bash
python -m pcb_rag.migrate --stage graph --execute
```

## 按需开启的功能

都在 `.env` 里改，改完重启服务。

| 功能 | 什么时候开 | 开关 | 默认 |
| --- | --- | --- | --- |
| 精排 | 想让最相关的排在前面 | `RERANK_BACKEND=api`（或本地，见下文） | 开（API） |
| HyDE 查询增强 | 问题很短、说得不完整 | `HYDE_ENABLED=1` | 开 |
| 复合问题拆分 | 一句话问了好几件事 | `QUERY_DECOMPOSE_ENABLED=1` | 开 |
| Step-back 背景补充 | 问题太具体，需要上位知识 | `QUERY_STEP_BACK_ENABLED=1` | 开 |
| 引用标注 | 答案要标出处 | `CITATION_ENABLED=1` | 开 |
| 语义缓存 | 重复问题多，想省钱 | `SEMANTIC_CACHE_ENABLED=1` | 开 |
| GraphRAG 图检索 | 问题涉及多个概念之间的关系 | `GRAPH_RAG_ENABLED=1`，先[补建图谱](#补建图谱) | 关 |
| 多轮迭代检索 | 一次查不全，要自己判断再查 | `SELF_RAG_ENABLED=1`，请求里再传 `use_self_rag: true` | 关 |
| 上下文压缩 | 召回内容太长，想省 token | `CONTEXT_COMPRESSION_ENABLED=1` | 关 |
| 会话持久化 | 重启后要保留对话 | `SESSION_STORE=sqlite` | `memory` |
| 多租户权限 | 多个团队共用一个库 | `ACL_ENABLED=1` | 关 |
| PDF 图片描述 | 文档里有大量图（需要视觉模型） | `MULTIMODAL_ENABLED=1` | 关 |
| 各路保底条数 | 想保证某一路至少进几条 | `FUSION_QUOTAS=vector=3,bm25=2,graph=1` | 空（均分） |

## 选择模型后端

LLM、Embedding、Rerank 三个可以分别选，自由混搭。

**全本地（Ollama）**

```bash
AI_API_ONLY=0
LLM_BACKEND=local
EMBED_BACKEND=local
OLLAMA_LLM_MODEL=qwen3.5:35b-a3b-q4_K_M
OLLAMA_EMBED_MODEL=qwen3-embedding:8b-q8_0
```

**全 API（OpenAI 兼容）**：见 [快速开始](#2-填写-env) 的例子。评测用的打分模型可以单独配：

```bash
JUDGE_BACKEND=api
JUDGE_MODEL=glm-5-3-260814
JUDGE_BASE_URL=     # 留空时复用 LLM_BASE_URL
JUDGE_API_KEY=      # 留空时复用 LLM_API_KEY
```

**精排**

```bash
# API 精排（推荐，不占本地显存）
RERANK_BACKEND=api
RERANK_API_URL=https://api.jina.ai/v1/rerank
RERANK_API_KEY=xxxx
RERANK_API_MODEL=jina-reranker-v2-base-multilingual

# 本地精排：hf / qwen3reranker / sbert；不需要精排时设 none
RERANK_BACKEND=qwen3reranker
```

> 混搭示例：LLM 走 API、Embedding 走本地，只需分别设置 `LLM_BACKEND=api` 与 `EMBED_BACKEND=local`，并设 `AI_API_ONLY=0`。

## 上线前检查

| 项目 | 要求 |
| --- | --- |
| 服务口令 | 必须设置 `DIFY_API_TOKEN`，不能是 `change-me`，否则服务拒绝启动 |
| 跨域 | `CORS_ALLOW_ORIGINS` 默认为空。只有浏览器要直连时，才按来源白名单配置 |
| 身份头 | `ACL_TRUST_HEADERS` 保持 `0`。打开后任何人都能自称管理员读全库。只有网关会剥离客户端同名头时才开，并同时设 `ACL_TRUST_HEADERS_ACK=1` |
| 用户身份 | 多租户场景用 `AUTH_TOKEN_PRINCIPALS_FILE` 把口令映射到用户、租户、角色 |
| 健康检查 | 探针用 `/health`；`/health/detail` 含配置摘要，需要口令 |
| 并发和超时 | `RAG_MAX_CONCURRENT` / `RAG_MAX_QUEUED`（默认 8 / 16），`RAG_REQUEST_TIMEOUT`（默认 120 秒） |
| 费用上限 | `RAG_MAX_MODEL_CALLS`、`RERANK_TOKEN_BUDGET`、`RERANK_REQUEST_BUDGET` 可限制单请求或累计用量 |
| 密钥 | `.env`、原始数据和模型缓存不要提交到仓库 |

## 配置参考

最常改的几项：

| 变量 | 作用 |
| --- | --- |
| `DATA_DIR` | 文档目录，默认 `./data/clear_docs` |
| `COLLECTION` | 向量集合名，迁移后要改成新集合 |
| `KB_STATE_PATH` | 版本状态文件，服务和入库必须用同一个 |
| `MILVUS_URI` | Milvus 地址，默认 `http://127.0.0.1:19530` |
| `API_PORT` | 服务端口，默认 `8000` |
| `RERANK_TOP_N` | 精排后保留几条，默认 `10` |
| `RAG_CONTEXT_MAX_TOKENS` | 给 LLM 的证据上限，默认 `6000` token |

完整列表展开下面各组查看，也可以看 `.env.example` 和 [配置指南](docs/CONFIGURATION_GUIDE.md)。

<details>
<summary>模型后端</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_BACKEND` | `api`（配置模板） | `api`（OpenAI 兼容）/ `local`（Ollama，需关闭 API-only） |
| `EMBED_BACKEND` | `api`（配置模板） | `api`（OpenAI 兼容）/ `local`（Ollama，需关闭 API-only） |
| `RERANK_BACKEND` | `api`（配置模板） | `api` / `hf` / `qwen3reranker` / `sbert` / `none` |
| `RERANK_ENABLED` | `1` | 是否启用精排 |
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | 本地 Ollama 服务地址 |
| `OLLAMA_LLM_MODEL` | `qwen3.5:35b-a3b-q4_K_M` | 本地 LLM 模型 |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:8b-q8_0` | 本地 Embedding 模型 |
| `LLM_BASE_URL` | 空 | API 后端地址（如 `https://api.deepseek.com/v1`） |
| `LLM_API_KEY` | 空 | API 密钥 |
| `LLM_MODEL` | 空 | API 模型名 |
| `EMBED_BASE_URL` | 空 | Embedding API 地址（留空复用 `LLM_BASE_URL`） |
| `EMBED_MODEL` | 空 | Embedding 模型名（api 后端必填） |
| `EMBED_DIM` | 空 | 向量维度（留空自动探测） |
| `RERANK_API_URL` | 空 | API 精排地址 |
| `RERANK_API_MODEL` | 空 | API 精排模型名 |

</details>

<details>
<summary>存储与入库</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `./data/clear_docs` | 待入库文档目录 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 服务地址 |
| `COLLECTION` | `pcb_kb` | 向量集合名称 |
| `KB_STATE_PATH` | `./data/kb_state_<COLLECTION>.json` | API 与入库共享的发布指针、阶段状态及上一版本 |
| `INGEST_INCREMENTAL` | `1` | 增量入库：按内容哈希指纹只处理新增 / 变更文档 |
| `INGEST_PLAN_ONLY` | `0` | 只预览要处理的文件，不写任何数据 |
| `INGEST_OVERWRITE` | `0` | 对非空集合会报错；全量重建请用新集合 |
| `AUTO_FIX_ENCODING` | `1` | 入库前修复 `.txt` 文件编码（会改写原文件） |
| `LEXICAL_CACHE_TTL_HOURS` | `24` | 词法索引缓存有效期（小时） |
| `LEXICAL_RELOAD_STAMP` | `<词法缓存>.stamp` | 入库结束后写入的重载标记；长驻 API 按它轮询并热重载 BM25 |

</details>

<details>
<summary>检索与生成</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RECALL_TOP_K` | `200` | 初始召回数量 |
| `RERANK_TOP_N` | `10` | 重排返回条数，API 请求也使用这个值；不决定输入量 |
| `RERANK_MAX_CANDIDATES` | `50` | 真正送入重排 API 的候选上限，独立于初始召回 200 条 |
| `RERANK_MAX_DOCUMENT_CHARS` | `1200` | 每条重排正文的字符上限；不发送文件路径、ID 等技术元数据 |
| `RERANK_MIN_INTERVAL` | `0` | 主动限速间隔；0 关闭，非零时跨进程共享 |
| `RERANK_TOKEN_BUDGET` | `0` | 0 不限制累计 token；用量记录和未知用量估计仍保留 |
| `RERANK_REQUEST_BUDGET` | `0` | 0 不限制累计请求次数；正数时包含重试 |
| `RERANK_429_BACKOFF_SECONDS` | `60` | 仅收到服务端 429 且无 Retry-After 时退避；不是主动限速 |
| `CHUNK_EXPAND_MAX_EXTRA` | `5` | 上下文扩展可额外返回的条数 |
| `HYDE_ENABLED` | `1` | 是否启用 HyDE 查询增强 |
| `QUERY_UNDERSTANDING_MODE` | `hybrid` | 查询理解模式：`rule` / `llm` / `hybrid` |
| `QUERY_DECOMPOSE_ENABLED` | `1` | 是否对复合问题自动分解 |
| `QUERY_STEP_BACK_ENABLED` | `1` | 是否对具体问题自动 Step-back |
| `STEPBACK_ROUTE_WEIGHT` | `0.7` | Step-back 召回路线在 RRF 中的权重系数 |
| `CONTEXTUAL_RETRIEVAL_MODE` | `rule` | chunk 语境前缀：`off` / `rule` / `llm` |
| `CITATION_ENABLED` | `1` | 是否在答案中标注引用来源 |
| `RAG_CONTEXT_MAX_TOKENS` | `6000` | 上下文 token 预算；按 token 逐条填充，替代固定条数。`0` 退回旧逻辑 |
| `RAG_FACTS_BUDGET_RATIO` | `0.22` | 事实摘要段占预算比例，其余留给原文证据与引用表 |
| `RAG_MIN_CHUNK_TOKENS` | `120` | 单条证据最小配额，低于此值停止填充 |
| `RAG_DEMOTE_LOW_VALUE` | `1` | 入库标记的低价值块是否降权到末尾（只调顺序，不硬过滤） |

</details>

<details>
<summary>融合与 chunk 质量</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FUSION_QUOTA_ENABLED` | `1` | 配额保底 + RRF 补齐。`0` 退回纯加权 RRF |
| `FUSION_MIN_PER_ROUTE` | `1` | 每路保底条数（未配 `FUSION_QUOTAS` 时生效） |
| `FUSION_QUOTAS` | 空 | 显式配额，按路名写：`vector=3,bm25=2,graph=1`（路名：vector/bm25/graph/hyde/stepback）；旧的位置式 `3,2,1` 按 向量/HyDE/BM25 解释，两个值时为 向量/BM25；写错时告警并退回均分 |
| `FUSION_RRF_K` | `40` | 融合平滑参数，包含图来源 |
| `CHUNK_QUALITY_MODE` | `flag` | `off` 不评估 / `flag` 只打标记 / `drop` 丢弃低价值块 |

</details>

<details>
<summary>缓存与会话</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SEMANTIC_CACHE_ENABLED` | `1` | 是否启用语义缓存 |
| `SEMANTIC_CACHE_MAX_SIZE` | `512` | 缓存条目上限 |
| `SEMANTIC_CACHE_TTL_SECONDS` | `3600` | 缓存有效期（秒） |
| `SEMANTIC_CACHE_THRESHOLD` | `0.95` | 语义命中的相似度阈值 |
| `SESSION_STORE` | `memory` | 会话存储：`memory` / `sqlite`（重启后保留） |
| `SESSION_DB_PATH` | `./data/sessions.db` | SQLite 会话库路径 |
| `SESSION_EXPIRE_MINUTES` | `60` | 会话过期时间（分钟） |
| `SESSION_MAX_COUNT` | `1000` | 最大会话数 |
| `SESSION_REQUIRE_IDENTITY` | `0` | 要求会话具有已认证的 tenant/user；已有无归属会话不可被认领 |

</details>

<details>
<summary>服务、并发与用量</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DIFY_API_TOKEN` | 无（**未设置时服务拒绝启动**） | 业务接口鉴权口令；占位值 `change-me` 同样会被拒绝 |
| `CORS_ALLOW_ORIGINS` | 空（不下发跨域头） | 允许跨域的来源白名单，逗号分隔；需要浏览器直连时才配置 |
| `API_PORT` | `8000` | API 服务端口 |
| `SELF_RAG_ENABLED` | `0` | 是否启用 Agentic RAG 迭代检索 |
| `SELF_RAG_MAX_ITERATIONS` | `3` | 迭代检索的最大轮数 |
| `RAG_REQUEST_TIMEOUT` | `120` | 总请求 deadline（秒），覆盖排队、检索、生成及 SSE |
| `RAG_MAX_CONCURRENT` / `RAG_MAX_QUEUED` | `8` / `16` | 每个服务进程的并发与等待边界 |
| `RAG_REQUEST_MAX_TOKENS` | 模型窗口 | 完整序列化请求上限，包含历史、标题、引用和输出预留 |
| `RAG_MAX_MODEL_CALLS` | `0` | 单请求模型调用总次数上限；0 不限制此项 |
| `RAG_LLM_MAX_INPUT_TOKENS` / `RAG_EMBEDDING_MAX_INPUT_TOKENS` / `RAG_RERANK_MAX_INPUT_TOKENS` | `0` | 各类模型的单请求累计输入上限 |
| `RAG_USAGE_LEDGER` | 空 | 可选 JSONL 用量账本；实际值与保守估算分列 |
| `AUTH_TOKEN_PRINCIPALS_FILE` | 空 | 以 Token 的 SHA-256 为键映射用户、租户、角色的 JSON 文件 |

</details>

<details>
<summary>GraphRAG 图检索</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `GRAPH_RAG_ENABLED` | `0` | 是否启用图检索（图谱缺失时自动降级为无图模式） |
| `GRAPH_PATH` | `./data/graph/kg.json` | 尚未迁移的旧图文件路径；新版使用快照中的路径 |
| `GRAPH_SEMANTIC_ENABLED` | `1` | 建立实体/关系描述向量索引 |
| `GRAPH_SEMANTIC_BACKEND` | `milvus` | 独立 Milvus 集合；或 `local` 文件向量余弦检索 |
| `GRAPH_EXTRACT_BACKEND` | `rule` | 抽取方式：`rule`（零 LLM 成本）/ `llm` / `hybrid` |
| `GRAPH_HOP` | `1` | 实体邻域扩展跳数 |
| `GRAPH_TOP_K` | `5` | 图路径返回的候选数 |
| `GRAPH_WEIGHT` | `0.35` | 图路在 RRF 融合中的权重（独立一路，有独立 rank） |
| `GRAPH_LLM_GLEANING` | `1` | 图抽取的 gleaning 轮数（回灌已抽结果补漏） |
| `GRAPH_LLM_DESC_CHARS` | `120` | 单条实体 / 关系描述字符上限 |
| `GRAPH_NODE_DESC_CHARS` | `240` | 实体描述跨块合并后的总上限 |
| `GRAPH_EXPAND_MAX` | `2` | 旧 `_graph_augment` 接口的扩展上限；共享检索管道不执行此扩展 |
| `GRAPH_COMMUNITY_ENABLED` | `0` | 是否生成社区摘要（面向综述类问题） |

开启图检索前，先完成新版入库或图补建。图抽取失败时保留旧版本；图命中的实体和关系必须回查到有权限的原文，才会进入答案证据。

</details>

<details>
<summary>上下文压缩</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CONTEXT_COMPRESSION_ENABLED` | `0` | 是否启用压缩 |
| `COMPRESSION_MODE` | `extractive` | `extractive` / `llm` / `hybrid` |
| `COMPRESSION_TARGET_RATIO` | `0.6` | 目标保留比例 |
| `COMPRESSION_BUDGET_CHARS` | `12000` | 进入上下文的字符预算 |
| `CONTEXT_DEDUPE_ENABLED` | `1` | 是否做近似去重 |
| `CONTEXT_DEDUPE_THRESHOLD` | `0.85` | 去重相似度阈值 |

</details>

<details>
<summary>访问控制</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ACL_ENABLED` | `0` | 是否启用多租户过滤 |
| `ACL_DEFAULT_VISIBILITY` | `private` | 入库默认可见性 |
| `ACL_PUBLIC_VALUE` | `public` | 该 visibility 的文档对所有租户可见 |
| `ACL_ADMIN_ROLES` | `admin,root` | 可跨租户访问的角色 |
| `ACL_TRUST_HEADERS` | `0` | 是否信任客户端自报的身份头。默认关闭：打开后任何人都能自报 `X-User-Roles: admin` 读全库 |
| `ACL_TRUST_HEADERS_ACK` | `0` | 显式确认「网关已剥离客户端同名头」。`ACL_ENABLED=1` 且 `ACL_TRUST_HEADERS=1` 而未设此项时，服务拒绝启动 |
| `ACL_STRICT_LEGACY` | `0` | 未标注权限字段的历史数据是否拒绝访问 |
| `ACL_HEADER_TENANT` | `X-Tenant-Id` | 租户请求头，另有 `X-User-Id` / `X-User-Roles` / `X-User-Groups` |

</details>

<details>
<summary>可观测性与多模态</summary>

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OBSERVABILITY_ENABLED` | `1` | 观测总开关 |
| `TRACE_ENABLED` | `1` | 是否记录 tracing |
| `TRACE_LOG_PATH` | `./data/traces.jsonl` | trace 落盘路径 |
| `TRACE_SAMPLE_RATE` | `1.0` | 采样率（高 QPS 可降到 `0.1`） |
| `MULTIMODAL_ENABLED` | `0` | 是否启用 PDF 图片描述（需配置视觉模型） |
| `MULTIMODAL_VLM_MODEL` | 空 | 视觉模型名（留空复用 `LLM_MODEL`） |
| `TABLE_MARKDOWN_ENABLED` | `1` | 表格结构化为 Markdown（零成本） |

</details>

## 评测

用自己的文档，量一量检索和回答的质量：

```bash
# 1. 从已入库的文档生成测试题
python eval/build_golden_dataset.py --limit 100

# 2. 人工抽查 20~30 条后，运行评测
python eval/evaluate.py --dataset eval/datasets/golden.jsonl
```

| 指标 | 看什么 | 参考线 |
| --- | --- | --- |
| Faithfulness | 答案有没有编造，是否和原文一致 | ≥ 0.75 |
| Answer Relevancy | 是否真的回答了问题 | ≥ 0.80 |
| Context Precision | 有用的原文是否排在前面 | ≥ 0.70 |
| Context Recall | 需要的信息是否都找到了 | ≥ 0.80 |

加 `--retrieval-only` 只评检索，更快更省；加 `--no-rerank` 可以对比精排的效果。详见 [eval/README.md](eval/README.md)。

想让检索评测和线上服务用同一套参数，运行：

```bash
python eval/evaluate_recall.py --rrf-k 40 --recall-k 200 --rerank --rerank-top-n 10
```

<details>
<summary>评测口径和线上参数对照</summary>

| 口径 | CLI / eval | API 进程 |
| --- | --- | --- |
| `RECALL_TOP_K` | 200（`FUSION_RRF_K`=40） | 200 |
| `RERANK_TOP_N` | 10 | 10 |
| API 实际重排输入上限 | 50 | 50 |
| 实际喂给 LLM | `RAG_TOP_DOCS`=5 | 5 |
| `--soft-match` 默认 | `none`（`0.78` 是 **embed 模式**的阈值，非命中判定默认） | — |
| `--rrf-k` / `--recall-k` 默认 | 60 / 40 | 40 / 200 |

未显式传参时，评测脚本与线上并不同参，分数不能直接当作线上指标。指标只适合在同一个数据集内比较不同配置，原因见 [已知问题](docs/ARCHITECTURE.md#已知问题known-issues)。

</details>

## 常见问题

<details>
<summary>Milvus 连接失败怎么办？</summary>

```bash
# 查看容器状态
docker compose -f docker/milvus/docker-compose.yml ps

# 重启 Milvus
docker compose -f docker/milvus/docker-compose.yml restart
```

Milvus 首次启动约需 1-2 分钟，可用 `python scripts/check_env.py` 确认连通性。

</details>

<details>
<summary>服务启动时报缺少配置？</summary>

- `DIFY_API_TOKEN` 没设置或是 `change-me`：服务会拒绝启动
- `LLM_BACKEND=api`：需要 `LLM_BASE_URL` 与 `LLM_MODEL`
- `EMBED_BACKEND=api`：需要 `EMBED_MODEL`（`EMBED_BASE_URL` 留空时自动复用 `LLM_BASE_URL`）
- `RERANK_BACKEND=api`：需要 `RERANK_API_URL`

运行 `python scripts/check_env.py` 可直接看到缺失项。

</details>

<details>
<summary>网页提问返回 401？</summary>

点右上角「设置」，填入 `.env` 里的 `DIFY_API_TOKEN`。调 API 时，请求头要带 `Authorization: Bearer <DIFY_API_TOKEN>`。

</details>

<details>
<summary>Rerank 不可用或显存不足？</summary>

- `RERANK_BACKEND=api` 必须配置 `RERANK_API_URL`，否则精排初始化失败并回退；
- 本地精排（`hf` / `qwen3reranker`）需要足够显存，不足时会自动回退为「仅召回」；
- 不需要精排时设置 `RERANK_BACKEND=none`，或直接改用 API 精排。

</details>

<details>
<summary>入库后新文档检索不到 / 排得很靠后？</summary>

依次检查三件事：
1. **新版本是否已发布**：在 `/manage` 检查版本、待发布状态及失败阶段；
2. **服务和入库是否用同一个状态文件**：两边的 `KB_STATE_PATH` 应相同，响应里的 `revision` 应是新版本；
3. **是否只看了 Hit@K**：新文档缺一路投票时 Hit@K 可能不变，但 MRR/NDCG 会掉。

</details>

<details>
<summary>重新入库后 BM25 仍在检索旧内容？</summary>

服务在每个请求开始时检查已发布版本，同步切换快照、BM25、图谱和集合。先检查服务与入库是否使用同一 `KB_STATE_PATH`，以及待发布阶段是否失败。`LEXICAL_CACHE_PATH` 和缓存 TTL 主要用于尚未迁移的旧集合。

</details>

<details>
<summary>回答里出现 degraded 是什么意思？</summary>

表示有一路检索或某个组件没跑起来，比如 `rerank_unavailable`（精排不可用）、`graph_unavailable`（图谱没建好）。答案仍会返回，但可能不完整。看 `degraded` 里的名字，再查 `/health/detail` 和服务日志。

</details>

## 更多文档

| 文档 | 内容 |
| --- | --- |
| [原理与设计](docs/ARCHITECTURE.md) | 亮点、架构图、检索各阶段、已知问题、修复记录 |
| [配置指南](docs/CONFIGURATION_GUIDE.md) | 全部配置项的详细说明 |
| [Dify 接入指南](docs/DIFY_INTEGRATION_GUIDE.md) | 在 Dify 里接入的完整步骤 |
| [Web 界面说明](docs/WEB_UI.md) | 问答页与管理页 |
| [WSL 运行说明](docs/WSL_API.md) | 在 Windows 上通过 WSL 启动、停止服务 |
| [安装说明](docs/SETUP.md) | 环境安装细节 |
| [Agent 接入示例](examples/README.md) | Python 元件与 function calling |
| [评测说明](eval/README.md) | 数据集构建与评测脚本 |
| [修复与验证报告](docs/LIGHTRAG_FIXES_WSL_20261023.md) | 2026-10-23 修复记录与迁移操作顺序 |

## 公开仓库说明

请在本地通过 `.env` 配置真实服务地址和密钥，不要将 `.env`、原始数据或模型缓存提交到 GitHub。

## License

遵循 MIT 协议，任何人都有权使用，但是修改请务必再开源
