# PCB-RAG：面向 PCB 知识库的智能问答系统

PCB-RAG 是一个面向 PCB 设计规范、工艺资料与工程经验文档的检索增强生成（RAG）系统。项目支持文档入库、向量检索、BM25 词法召回、多路融合、Rerank 精排、元数据过滤和 Dify 外部知识库 API 集成，可用于构建 PCB 领域的智能问答助手。

博客文章：https://blog.eecs.top/index.php/archives/3/

## 项目亮点

- **本地 / API 双后端**：LLM、Embedding、Rerank 均支持「本地服务（Ollama / HuggingFace）」与「OpenAI 兼容 API」两种后端，通过 `*_BACKEND` 环境变量切换，不锁定厂商。
- **一键环境补全**：`scripts/setup.sh` 自动创建虚拟环境、安装依赖、生成 `.env`、启动 Milvus，并按后端检查模型可用性。
- **领域化文档处理**：针对 PCB 规范、EDA 工具文档、工艺参数等资料进行清洗、切块和元数据抽取。
- **混合检索架构**：结合 Milvus 向量检索、BM25 词法检索、HyDE 查询扩展和多路召回融合，提高专业问题召回率。
- **Rerank 精排**：支持 API 精排（Jina / 硅基流动等）与本地模型精排（Qwen3-Reranker / cross-encoder）。
- **Dify 集成**：提供 FastAPI 外部知识库接口，可接入 Dify 工作流或对话应用。
- **标准项目结构**：采用 `src/pcb_rag` 包结构，便于安装、导入和维护。

## 技术栈

- Python 3.10+
- LlamaIndex
- Milvus
- OpenAI 兼容 API / Ollama / HuggingFace Transformers
- FastAPI
- Dify External Knowledge API

## 目录结构

```text
.
├── .env.example                  # 公开安全的环境变量模板
├── requirements.txt              # Python 依赖
├── pyproject.toml                # Python 包配置
├── src/pcb_rag/                  # 核心源码包
│   ├── api_clients.py            # LLM / Embedding / Rerank 双后端工厂
│   ├── ingest.py                 # 文档入库
│   ├── query.py                  # 交互式问答
│   ├── preprocess_docs.py        # 文档预处理
│   └── dify_external_api.py      # Dify 外部知识库 API
├── scripts/                      # 一键安装与运行脚本
├── docker/milvus/docker-compose.yml
├── data/README.md                # 数据目录说明
└── docs/                         # 配置、集成和优化说明文档
```

## 快速开始

### 1. 一键补全环境

```bash
bash scripts/setup.sh
```

该脚本会执行：

- 创建 `.venv`
- 安装 `requirements.txt`
- 执行 `pip install -e .`
- 从 `.env.example` 生成 `.env`
- 创建 `data/clear_docs/`、`data/raw_docs/`、`logs/`
- 启动 `docker/milvus/docker-compose.yml`
- 按 `LLM_BACKEND` / `EMBED_BACKEND` 检查后端（local 时拉取 Ollama 模型，api 时跳过）
- 运行 `scripts/check_env.py`

前置要求：Python 3.10+、Docker、Docker Compose；本地后端需要 Ollama，API 后端只需可访问的 OpenAI 兼容服务。详细说明见 `docs/SETUP.md`。

### 2. 配置模型后端

编辑 `.env`，按需选择后端：

```bash
# 方式一：全本地（Ollama）
LLM_BACKEND=local
EMBED_BACKEND=local
OLLAMA_LLM_MODEL=qwen3.5:35b-a3b-q4_K_M
OLLAMA_EMBED_MODEL=qwen3-embedding:8b-q8_0

# 方式二：全 API（OpenAI 兼容，以硅基流动为例）
LLM_BACKEND=api
EMBED_BACKEND=api
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=Qwen/Qwen3-8B
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024

# 方式三：混搭（例如 LLM 走 API，Embedding 走本地）
```

Rerank 后端：

```bash
# API 精排（推荐，无需本地显存）
RERANK_BACKEND=api
RERANK_API_URL=https://api.jina.ai/v1/rerank
RERANK_API_KEY=xxxx
RERANK_API_MODEL=jina-reranker-v2-base-multilingual

# 本地精排（可选 hf / qwen3reranker / sbert），或 none 关闭
RERANK_BACKEND=qwen3reranker
```

### 3. 准备数据

公开仓库不包含任何原始语料。请将你有权使用的 PCB 文档放入：

```text
data/clear_docs/
```

### 4. 文档入库

```bash
bash scripts/run_ingest.sh
```

等价命令：

```bash
python -m pcb_rag.ingest
```

> 默认增量写入（`INGEST_OVERWRITE=0`）。如需清空重建集合，设置 `INGEST_OVERWRITE=1`。

### 5. 运行问答

```bash
bash scripts/run_query.sh
```

等价命令：

```bash
python -m pcb_rag.query
```

示例问题：

```text
4 层 PCB 的阻抗控制需要关注哪些参数？
Altium Designer 中如何处理高速差分线等长？
```

## Dify 外部知识库 API

启动服务：

```bash
bash scripts/serve_api.sh
```

等价命令：

```bash
uvicorn pcb_rag.dify_external_api:app --host 0.0.0.0 --port 8000
```

在 Dify 外部知识库中配置：

- URL：`http://<your-server-host>:8000/retrieval`
- API Key：`Bearer <your DIFY_API_TOKEN>`

更多步骤见 `docs/DIFY_INTEGRATION_GUIDE.md`。`/health` 端点会返回当前后端配置与检索参数，便于排查问题。

## 常用配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `./data/clear_docs` | 待入库文档目录 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 服务地址 |
| `COLLECTION` | `pcb_kb` | 向量集合名称 |
| `INGEST_OVERWRITE` | `0` | 入库时是否清空重建集合 |
| `LLM_BACKEND` | `local` | LLM 后端：`local`(Ollama) / `api`(OpenAI 兼容) |
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | 本地 Ollama 服务地址 |
| `OLLAMA_LLM_MODEL` | `qwen3.5:35b-a3b-q4_K_M` | 本地 LLM 模型 |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:8b-q8_0` | 本地 Embedding 模型 |
| `LLM_BASE_URL` | 空 | API 后端地址（如 `https://api.deepseek.com/v1`） |
| `LLM_API_KEY` | 空 | API 密钥 |
| `LLM_MODEL` | 空 | API 模型名 |
| `EMBED_BACKEND` | `local` | Embedding 后端：`local` / `api` |
| `EMBED_BASE_URL` | 空 | Embedding API 地址（留空复用 `LLM_BASE_URL`） |
| `EMBED_MODEL` | 空 | Embedding 模型名（api 后端必填） |
| `EMBED_DIM` | 空 | 向量维度（留空自动探测） |
| `RECALL_TOP_K` | `200` | 初始召回数量 |
| `LEXICAL_CACHE_TTL_HOURS` | `24` | 词法索引缓存有效期（小时） |
| `CHUNK_EXPAND_MAX_EXTRA` | `5` | 上下文扩展可额外返回的条数 |
| `RERANK_ENABLED` | `1` | 是否启用 Rerank |
| `RERANK_BACKEND` | `qwen3reranker` | Rerank 后端：`api` / `hf` / `qwen3reranker` / `sbert` / `none` |
| `RERANK_API_URL` | 空 | API 精排地址 |
| `RERANK_API_MODEL` | 空 | API 精排模型名 |
| `DIFY_API_TOKEN` | `change-me` | Dify 外部知识库鉴权 Token |

完整配置见 `.env.example` 和 `docs/CONFIGURATION_GUIDE.md`。

## 开发意义

- 设计并实现面向 PCB 领域文档的 RAG 问答系统，支持规范文档入库、向量检索、BM25 召回、多路融合、Rerank 精排与答案生成。
- 基于 LlamaIndex、Milvus、FastAPI 搭建知识库服务，模型层抽象出「本地 / OpenAI 兼容 API」双后端，可按部署环境自由切换。
- 针对 PCB 专业术语、工艺参数和 EDA 工具场景设计元数据抽取、查询过滤和混合检索策略，提升专业问题召回与回答相关性。
- 整理为可复现工程：提供标准 Python 包结构、环境变量模板、Docker Compose 与一键 setup/run 脚本。

## 公开仓库说明

请在本地通过 `.env` 配置真实服务地址和密钥，不要将 `.env`、原始数据或模型缓存提交到 GitHub。

## License

遵循 MIT 协议，任何人都有权使用，但是修改请务必再开源
