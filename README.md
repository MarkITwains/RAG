# PCB-RAG：面向 PCB 知识库的智能问答系统

PCB-RAG 是一个面向 PCB 设计规范、工艺资料与工程经验文档的检索增强生成（RAG）系统。项目支持文档入库、向量检索、BM25 词法召回、多路融合、Rerank 精排、元数据过滤和 Dify 外部知识库 API 集成，可用于构建 PCB 领域的智能问答助手。
博客文章：https://blog.eecs.top/index.php/archives/3/


## 项目亮点

- **一键环境补全**：`scripts/setup.sh` 自动创建虚拟环境、安装依赖、生成 `.env`、启动 Milvus 并检查 Ollama 模型。
- **领域化文档处理**：针对 PCB 规范、EDA 工具文档、工艺参数等资料进行清洗、切块和元数据抽取。
- **混合检索架构**：结合 Milvus 向量检索、BM25 词法检索、HyDE 查询扩展和多路召回融合，提高专业问题召回率。
- **Rerank 精排**：默认按效果优先启用 Qwen Reranker，可通过 `.env` 调整模型和检索参数。
- **Dify 集成**：提供 FastAPI 外部知识库接口，可接入 Dify 工作流或对话应用。
- **标准项目结构**：采用 `src/pcb_rag` 包结构，便于安装、导入和维护。

## 技术栈

- Python 3.10+
- LlamaIndex
- Milvus
- Ollama
- HuggingFace Transformers / SentenceTransformers
- FastAPI
- Dify External Knowledge API

## 目录结构

```text
.
├── .env.example                  # 公开安全的环境变量模板
├── requirements.txt              # Python 依赖
├── pyproject.toml                # Python 包配置
├── src/pcb_rag/                  # 核心源码包
│   ├── ingest.py                 # 文档入库
│   ├── query.py                  # 交互式问答
│   ├── preprocess_docs.py         # 文档预处理
│   └── dify_external_api.py       # Dify 外部知识库 API
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
- 检查 Ollama 并尝试拉取 `.env` 中配置的模型
- 运行 `scripts/check_env.py`

前置要求：Python 3.10+、Docker、Docker Compose、Ollama。详细说明见 `docs/SETUP.md`。

### 2. 准备数据

公开仓库不包含任何原始语料。请将你有权使用的 PCB 文档放入：

```text
data/clear_docs/
```

### 3. 文档入库

```bash
bash scripts/run_ingest.sh
```

等价命令：

```bash
python -m pcb_rag.ingest
```

### 4. 运行问答

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

更多步骤见 `docs/DIFY_INTEGRATION_GUIDE.md`。

## 常用配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATA_DIR` | `./data/clear_docs` | 待入库文档目录 |
| `MILVUS_URI` | `http://127.0.0.1:19530` | Milvus 服务地址 |
| `COLLECTION` | `pcb_kb` | 向量集合名称 |
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama 服务地址 |
| `OLLAMA_LLM_MODEL` | `qwen3.5:35b-a3b-q4_K_M` | LLM 模型 |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding 模型 |
| `RECALL_TOP_K` | `200` | 初始召回数量 |
| `RERANK_ENABLED` | `1` | 是否启用 Rerank |
| `RERANK_BACKEND` | `qwen3reranker` | Rerank 后端 |
| `RERANK_MODEL` | `Qwen/Qwen3-Reranker-4B` | Rerank 模型 |
| `DIFY_API_TOKEN` | `change-me` | Dify 外部知识库鉴权 Token |

完整配置见 `.env.example` 和 `docs/CONFIGURATION_GUIDE.md`。

## 开发意义

- 设计并实现面向 PCB 领域文档的 RAG 问答系统，支持规范文档入库、向量检索、BM25 召回、多路融合、Rerank 精排与答案生成。
- 基于 LlamaIndex、Milvus、Ollama 和 FastAPI 搭建私有化知识库服务，并通过 Dify External Knowledge API 对外提供问答能力。
- 针对 PCB 专业术语、工艺参数和 EDA 工具场景设计元数据抽取、查询过滤和混合检索策略，提升专业问题召回与回答相关性。
- 整理为可复现工程：提供标准 Python 包结构、环境变量模板、Docker Compose 与一键 setup/run 脚本。

## 公开仓库说明

请在本地通过 `.env` 配置真实服务地址和密钥，不要将 `.env`、原始数据或模型缓存提交到 GitHub。

## License

遵循 MIT 协议，任何人都有权使用，但是修改请务必再开源
