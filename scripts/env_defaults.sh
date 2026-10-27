# Load the project's dotenv parser once, preserving explicit caller overrides.
pcb_rag_load_env_defaults() {
    local env_file="$1"
    [ -f "$env_file" ] || return 0
    export PCB_RAG_ENV_FILE="$env_file"
    local output key value
    output="$(PCB_RAG_LOAD_DOTENV=0 "${PYTHON:-python}" - "$env_file" <<'PY'
import os
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "src"))
from pcb_rag.env_loader import _unquote

for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    line = line.removeprefix("export ").lstrip()
    key, value = line.split("=", 1)
    key = key.strip()
    if not key.isidentifier() or key in os.environ:
        continue
    print(f"{key}={_unquote(value)}")
PY
)"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        key="${line%%=*}"
        value="${line#*=}"
        printf -v "$key" '%s' "$value"
        export "$key"
    done <<< "$output"
}
