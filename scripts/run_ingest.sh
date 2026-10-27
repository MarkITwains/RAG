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
  source "$(dirname "${BASH_SOURCE[0]}")/env_defaults.sh"
  pcb_rag_load_env_defaults .env
fi

python -m pcb_rag.ingest
