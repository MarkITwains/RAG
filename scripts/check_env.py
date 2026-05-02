import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check_import(module: str) -> bool:
    try:
        __import__(module)
        print(f"[OK] import {module}")
        return True
    except Exception as exc:
        print(f"[FAIL] import {module}: {exc}")
        return False


def fetch_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    ok = True
    if sys.version_info < (3, 10):
        print(f"[FAIL] Python >= 3.10 required, current: {sys.version.split()[0]}")
        ok = False
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    for module in [
        "llama_index",
        "pymilvus",
        "fastapi",
        "pydantic",
        "transformers",
        "sentence_transformers",
        "jieba",
    ]:
        ok = check_import(module) and ok

    if (root / ".env").exists():
        print("[OK] .env exists")
    else:
        print("[FAIL] .env missing; copy .env.example to .env")
        ok = False

    data_dir = root / os.getenv("DATA_DIR", "./data/clear_docs")
    if data_dir.exists():
        print(f"[OK] data directory: {data_dir}")
    else:
        print(f"[FAIL] data directory missing: {data_dir}")
        ok = False

    ollama_base = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
    try:
        tags = fetch_json(f"{ollama_base}/api/tags")
        names = {m.get("name", "") for m in tags.get("models", [])}
        print(f"[OK] Ollama reachable: {ollama_base}")
        for env_key in ["OLLAMA_LLM_MODEL", "OLLAMA_EMBED_MODEL"]:
            model = os.getenv(env_key, "").strip()
            if model and model not in names:
                print(f"[WARN] Ollama model not listed: {model}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] Ollama not reachable at {ollama_base}: {exc}")

    milvus_uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    try:
        from pymilvus import connections

        connections.connect(alias="check_env", uri=milvus_uri)
        connections.disconnect("check_env")
        print(f"[OK] Milvus reachable: {milvus_uri}")
    except Exception as exc:
        print(f"[WARN] Milvus not reachable at {milvus_uri}: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
