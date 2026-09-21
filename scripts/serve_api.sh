#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".venv/Scripts/activate" ]; then
  source ".venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# --loop asyncio：uvloop 与 query.py 的 nest_asyncio/run_until_complete 不兼容，
  #   Linux + uvicorn[standard] 默认 uvloop 时启动即崩（2026-09-20 实测，见 TestRule.md）
  uvicorn pcb_rag.dify_external_api:app --host 0.0.0.0 --port "${API_PORT:-8000}" --loop asyncio
