# 环境搭建说明

本项目提供一键环境脚本，目标是在 Windows Git Bash、Linux 或 macOS 终端中完成依赖安装和运行环境准备。

## 前置依赖

请先安装：

- Python 3.10+
- Docker 与 Docker Compose
- Ollama
- Git Bash（Windows 用户推荐）

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
7. 检查 Ollama，并尝试拉取 `.env` 中配置的 LLM 与 Embedding 模型。
8. 运行 `scripts/check_env.py` 检查 Python 包、Milvus、Ollama 和数据目录。

## 默认模型配置

`.env.example` 默认偏效果优先：

```bash
OLLAMA_LLM_MODEL=qwen3.5:35b-a3b-q4_K_M
RERANK_ENABLED=1
RERANK_BACKEND=qwen3reranker
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RECALL_TOP_K=200
```

这类模型对显存、内存和下载时间要求较高。如果你的机器资源有限，可以在 `.env` 中改为较轻配置，例如：

```bash
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

## Ollama 排查

查看模型：

```bash
ollama list
```

手动拉取模型：

```bash
ollama pull qwen3.5:35b-a3b-q4_K_M
ollama pull nomic-embed-text
```

如果 `scripts/check_env.py` 提示 Ollama 无法连接，请确认 Ollama 服务已经启动，并检查 `.env` 中的 `OLLAMA_BASE`。
