# WSL 纯外部 API 运行环境

本机部署记录：2026-10-20，发行版 `pure2404`（Ubuntu 24.04），用户 `pi`。

LLM、Embedding、Rerank 使用项目 `.env` 中配置的外部接口。启动脚本强制三个后端为 `api`，并关闭 Hugging Face 在线模型下载。无需 Ollama、CUDA、PyTorch 或本地模型权重。

2026-10-20 重排已按用户最新要求恢复 SiliconFlow `BAAI/bge-reranker-v2-m3`，完整接口 `https://api.siliconflow.cn/v1/rerank`；见 [重排用量保护](RERANK_GUARDS_20261020.md)。候选上限 50、返回 10，主动限速为 0。LLM 和 NVIDIA Embedding 沿用原外部 API。

## 环境位置

- 项目：`/mnt/d/myProject/rag_v1-main/rag_v1`
- Python 环境：`/home/pi/.local/share/pcb-rag/venv`，项目 `.venv` 指向它。
- 原 Python 环境：保留在项目中的 `.venv.backup-*`。
- Milvus 数据：`/home/pi/.local/share/pcb-rag/milvus/volumes`，从原项目数据卷复制。
- Python 依赖快照：`/home/pi/.local/share/pcb-rag/requirements-api.lock.txt`。
- API 日志：项目 `logs/wsl-api.log`。
- 本次运行验证：项目 `logs/wsl-setup-verification.json`。

数据库运行的是 Milvus、etcd 和 MinIO，它们用于存储和检索；模型计算通过外部 API 完成。原项目 `docker/milvus/volumes` 保留为迁移前副本。Embedding 修复后，现用集合为 `pcb_kb_nv_query_passage_20261020`，有 5680 条记录，向量维度为 2048；旧集合 `pcb_kb` 保留。模式、评测口径和回退步骤见 [Embedding 修复记录](EMBEDDING_FIX.md)。

## 从 PowerShell 启动

先启动数据库，再启动问答服务：

```powershell
wsl -d pure2404 -u root -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_milvus.sh start
wsl -d pure2404 -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_api.sh start
```

数据库脚本用 root 访问 Docker；API 用普通用户 `pi` 运行。API 在 tmux 会话 `pcb-rag-api` 中运行，关闭当前终端后可继续工作。`start` 等待健康检查成功后才报告就绪。

Windows 访问地址：

- 健康检查：`http://localhost:8000/health`
- 交互接口文档：`http://localhost:8000/docs`
- 完整问答：`POST http://localhost:8000/api/ask`
- Dify 外部知识库：`POST http://localhost:8000/retrieval`

业务接口使用 `.env` 内的 `DIFY_API_TOKEN` 作为 Bearer Token。不要把密钥写进仓库或命令历史。

## 状态、日志与停止

```powershell
wsl -d pure2404 -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_api.sh status
wsl -d pure2404 -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_api.sh logs
wsl -d pure2404 -u root -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_milvus.sh status
```

停止 API 和数据库：

```powershell
wsl -d pure2404 -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_api.sh stop
wsl -d pure2404 -u root -- bash /mnt/d/myProject/rag_v1-main/rag_v1/scripts/wsl_milvus.sh stop
```

停止命令保留数据卷。重启 Windows 或执行 `wsl --shutdown` 后，用前面的两条启动命令恢复服务。

## 代理

Windows 代理端口为 `12345`。WSL 使用 NAT 网络，通过默认路由的网关访问该端口。

用户代理脚本是 `~/.config/pcb-rag/proxy.sh`，已接入 `~/.profile` 和 `~/.bashrc`；API 启动脚本也会主动读取它。该脚本动态计算网关，并设置 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 和相应小写变量。本机数据库地址加入 `NO_PROXY`。

Docker 单独配置代理：`/etc/systemd/system/docker.service.d/wsl-proxy.conf`。Docker 每次启动时运行 `/usr/local/sbin/wsl-docker-proxy-env`，重新计算网关并写入 `/run/wsl-docker-proxy.env`，无需固定 WSL 的内部 IP。

Windows 的 `.wslconfig` 已备份为同目录下的 `.wslconfig.backup-20261020-143352`。当前使用 NAT，关闭 WSL 自动代理注入，改用上面的显式代理配置。Windows 上的代理程序需要保持运行。

## 安装与验证依赖

在 WSL 项目目录中：

```bash
source "$HOME/.config/pcb-rag/proxy.sh"
uv pip install --python "$HOME/.local/share/pcb-rag/venv/bin/python" -r requirements-api.txt -e .
uv pip check --python "$HOME/.local/share/pcb-rag/venv/bin/python"
.venv/bin/python scripts/check_env.py
```

`requirements-api.txt` 使用 `llama-index-core`，并限制 `pymupdf<2`，避免新版本 PDF 库引入本地版面模型组件。Milvus 使用 Milvus 项目维护的同版本 MinIO 镜像。

运行测试时，显式隔离 `.env` 中的历史 ACL 信任头配置：

```bash
ACL_TRUST_HEADERS=0 REQUIRE_LLAMA_INDEX=1 .venv/bin/python -m pytest -q -rs
```

环境搭建沿用当前模型、向量和检索参数。Embedding 输入模式等检索优化应另建实验集合验证，避免改变现有冻结评测的基准。
