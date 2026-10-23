#!/usr/bin/env bash
# Start the API in WSL with external model providers only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${PCB_RAG_RUNTIME_DIR:-$HOME/.local/share/pcb-rag}"
PYTHON="$RUNTIME_DIR/venv/bin/python"
SESSION="pcb-rag-api"
LOG_FILE="$ROOT_DIR/logs/wsl-api.log"

if [ -r "$HOME/.config/pcb-rag/proxy.sh" ]; then
  source "$HOME/.config/pcb-rag/proxy.sh"
fi

cd "$ROOT_DIR"
export LLM_BACKEND=api EMBED_BACKEND=api RERANK_BACKEND=api
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

require_python() {
  if [ ! -x "$PYTHON" ]; then
    printf 'Missing WSL Python environment: %s\n' "$PYTHON" >&2
    exit 1
  fi
}

case "${1:-start}" in
  run)
    require_python
    exec "$PYTHON" -m uvicorn pcb_rag.dify_external_api:app \
      --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" --loop asyncio
    ;;
  start)
    require_python
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      printf 'API session already exists.\n'
    else
      mkdir -p "$ROOT_DIR/logs"
      printf -v command 'exec bash %q run >> %q 2>&1' "$ROOT_DIR/scripts/wsl_api.sh" "$LOG_FILE"
      tmux new-session -d -s "$SESSION" "$command"
    fi
    for ((attempt=1; attempt<=90; attempt++)); do
      if curl --noproxy '*' --connect-timeout 1 --max-time 1 --fail --silent \
          "http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/health" >/dev/null; then
        printf 'API ready at http://%s:%s; log: %s\n' "${API_HOST:-127.0.0.1}" "${API_PORT:-8000}" "$LOG_FILE"
        exit 0
      fi
      if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        printf 'API exited during startup.\n' >&2
        tail -n 30 "$LOG_FILE" >&2
        exit 1
      fi
      sleep 1
    done
    printf 'API did not become ready; inspect %s\n' "$LOG_FILE" >&2
    exit 1
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux send-keys -t "$SESSION" C-c
      printf 'Sent shutdown signal to API session.\n'
    else
      printf 'API session is not running.\n'
    fi
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      printf 'API session: running\n'
    else
      printf 'API session: stopped\n'
    fi
    curl --noproxy '*' --connect-timeout 2 --max-time 5 --fail --silent --show-error \
      "http://${API_HOST:-127.0.0.1}:${API_PORT:-8000}/health"
    printf '\n'
    ;;
  logs)
    tail -n "${LOG_LINES:-60}" "$LOG_FILE"
    ;;
  *)
    printf 'Usage: bash scripts/wsl_api.sh {start|stop|status|logs|run}\n' >&2
    exit 2
    ;;
esac
