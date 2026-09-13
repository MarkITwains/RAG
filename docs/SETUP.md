# 环境搭建说明

本项目提供一键环境脚本，目标是在 Windows Git Bash、Linux 或 macOS 终端中完成依赖安装和运行环境准备。

## 前置依赖

请先安装：

- Python 3.10+
- Docker 与 Docker Compose
- Git Bash（Windows 用户推荐）
- 模型服务（二选一）：
  - `LLM_BACKEND=local` / `EMBED_BACKEND=local`：安装并启动 Ollama
  - `LLM_BACKEND=api` / `EMBED_BACKEND=api`：准备任意 OpenAI 兼容服务的 `BASE_URL` 与 `API_KEY`

## 一键补全环境

在项目根目录执行：

```bash
bash scripts/setup.sh
```

脚本会自动完成：

1. 创建 `.venv` 虚拟环境。
2. 安装 `requirements.txt`。
3. 执行 `pip install -e .`，将 `src/pcb_rag` 安装为本地可导入包。
4. 如果 `.env` 不存在，则从 `.env.example` 复制。
5. 创建运行目录：`data/clear_docs/`、`data/raw_docs/`、`logs/`。
6. 启动 Milvus：`docker/milvus/docker-compose.yml`。
7. 按 `LLM_BACKEND` / `EMBED_BACKEND` 检查后端：local 时尝试拉取 Ollama 模型，api 时跳过。
8. 运行 `scripts/check_env.py` 检查 Python 包、Milvus、模型后端和数据目录。

## 模型后端配置

LLM / Embedding / Rerank 各自独立选择后端，可自由混搭。

### 全本地（Ollama + 本地 Rerank）

```bash
LLM_BACKEND=local
EMBED_BACKEND=local
OLLAMA_LLM_MODEL=qwen3.5:35b-a3b-q4_K_M
OLLAMA_EMBED_MODEL=qwen3-embedding:8b-q8_0
RERANK_ENABLED=1
RERANK_BACKEND=qwen3reranker
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RECALL_TOP_K=200
```

### 全 API（OpenAI 兼容，示例：硅基流动）

```bash
LLM_BACKEND=api
EMBED_BACKEND=api
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=Qwen/Qwen3-8B
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024
RERANK_BACKEND=api
RERANK_API_URL=https://api.siliconflow.cn/v1/rerank
RERANK_API_KEY=sk-xxxx
RERANK_API_MODEL=BAAI/bge-reranker-v2-m3
RECALL_TOP_K=100
```

### 资源受限（本地轻量配置）

```bash
LLM_BACKEND=local
EMBED_BACKEND=local
OLLAMA_LLM_MODEL=qwen3:8b
RERANK_ENABLED=0
RECALL_TOP_K=40
```

## 常用命令

```bash
# 检查环境
python scripts/check_env.py

# 文档入库
bash scripts/run_ingest.sh

# 交互式问答
bash scripts/run_query.sh

# Dify 外部知识库 API
bash scripts/serve_api.sh
```

也可以直接使用 Python 模块命令：

```bash
python -m pcb_rag.ingest
python -m pcb_rag.query
uvicorn pcb_rag.dify_external_api:app --host 0.0.0.0 --port 8000
```

## 数据目录

公开仓库不包含语料。请将授权文档放入：

```text
data/clear_docs/
```

不要提交原始文档、缓存文件、模型文件或 `.env`。

## Milvus 排查

查看容器状态：

```bash
docker compose -f docker/milvus/docker-compose.yml ps
```

重启 Milvus：

```bash
docker compose -f docker/milvus/docker-compose.yml restart
```

停止 Milvus：

```bash
docker compose -f docker/milvus/docker-compose.yml down
```

## 模型后端排查

先确认当前使用的后端：

```bash
python scripts/check_env.py
```

**本地后端（local）**

```bash
ollama list                                   # 查看已安装模型
ollama pull qwen3.5:35b-a3b-q4_K_M            # 拉取 LLM
ollama pull qwen3-embedding:8b-q8_0           # 拉取 Embedding
```

如果提示 Ollama 无法连接，请确认服务已启动，并检查 `.env` 中的 `OLLAMA_BASE`。

**API 后端（api）**

- 确认 `LLM_BASE_URL` 指向 OpenAI 兼容地址（通常以 `/v1` 结尾）。
- 确认 `LLM_API_KEY` 有效，且 `LLM_MODEL` / `EMBED_MODEL` 是服务方支持的模型名。
- Embedding 维度不确定时，先留空 `EMBED_DIM` 让程序自动探测；若探测失败，请显式填写。
- API 精排需同时配置 `RERANK_API_URL`；未配置时请在 `.env` 中设 `RERANK_BACKEND=none`。
