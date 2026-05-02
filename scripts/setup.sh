#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

info() { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

python_cmd=""
if command_exists python; then
  python_cmd="python"
elif command_exists python3; then
  python_cmd="python3"
else
  fail "未找到 Python，请先安装 Python 3.10+。"
fi

if ! "$python_cmd" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  fail "Python 版本需要 >= 3.10。"
fi

if [ ! -d ".venv" ]; then
  info "创建虚拟环境 .venv"
  "$python_cmd" -m venv .venv
fi

if [ -f ".venv/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  fail "未找到虚拟环境激活脚本。"
fi

info "安装 Python 依赖"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

if [ ! -f ".env" ]; then
  info "创建 .env"
  cp .env.example .env
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

mkdir -p "${DATA_DIR:-./data/clear_docs}" data/raw_docs logs

compose_cmd=()
if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command_exists docker-compose; then
  compose_cmd=(docker-compose)
else
  warn "未找到 Docker Compose，跳过 Milvus 启动。请安装 Docker Desktop 或 docker compose。"
fi

if [ ${#compose_cmd[@]} -gt 0 ]; then
  info "启动 Milvus"
  "${compose_cmd[@]}" -f docker/milvus/docker-compose.yml up -d
fi

if command_exists ollama; then
  if ollama list >/dev/null 2>&1; then
    if [ -n "${OLLAMA_LLM_MODEL:-}" ]; then
      info "检查/拉取 LLM 模型: ${OLLAMA_LLM_MODEL}"
      ollama pull "${OLLAMA_LLM_MODEL}" || warn "LLM 模型拉取失败，可稍后手动执行: ollama pull ${OLLAMA_LLM_MODEL}"
    fi
    if [ -n "${OLLAMA_EMBED_MODEL:-}" ]; then
      info "检查/拉取 Embedding 模型: ${OLLAMA_EMBED_MODEL}"
      ollama pull "${OLLAMA_EMBED_MODEL}" || warn "Embedding 模型拉取失败，可稍后手动执行: ollama pull ${OLLAMA_EMBED_MODEL}"
    fi
  else
    warn "Ollama 未响应，请先启动 Ollama 服务后再拉取模型。"
  fi
else
  warn "未找到 Ollama。请安装并启动 Ollama 后执行: ollama pull ${OLLAMA_LLM_MODEL:-qwen3.5:35b-a3b-q4_K_M}"
fi

info "运行环境检查"
python scripts/check_env.py || warn "环境检查发现问题，请根据上方提示处理。"

cat <<'EOF'

环境准备流程已执行。
下一步：
  1. 将授权文档放入 data/clear_docs/
  2. 执行 bash scripts/run_ingest.sh
  3. 执行 bash scripts/run_query.sh
  4. 如需 Dify API，执行 bash scripts/serve_api.sh
EOF
