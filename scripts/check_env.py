"""PCB-RAG 环境自检脚本。

按当前后端配置检查：
  - Python 版本与关键依赖
  - .env / 数据目录
  - LLM 后端（local=Ollama，api=OpenAI 兼容）
  - Embedding 后端（local=Ollama，api=OpenAI 兼容）
  - Rerank 后端（api=HTTP 接口，其余为本地模型，仅做配置校验）
  - Milvus 连通性
"""

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


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _auth_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_json(url: str, headers: dict = None, timeout: float = 8.0):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_llm() -> bool:
    backend = _env("LLM_BACKEND", "local").lower()

    if backend == "api":
        base = _env("LLM_BASE_URL").rstrip("/")
        model = _env("LLM_MODEL")
        if not base or not model:
            print("[FAIL] LLM_BACKEND=api 需要同时配置 LLM_BASE_URL 与 LLM_MODEL")
            return False
        try:
            fetch_json(f"{base}/models", headers=_auth_headers(_env("LLM_API_KEY")))
            print(f"[OK] LLM API 可访问: {base} (model={model})")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[WARN] LLM API 无法访问 {base}: {exc}")
        return True

    ollama_base = _env("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
    model = _env("OLLAMA_LLM_MODEL")
    try:
        tags = fetch_json(f"{ollama_base}/api/tags")
        names = {m.get("name", "") for m in tags.get("models", [])}
        print(f"[OK] Ollama 可访问: {ollama_base}")
        if model and model not in names:
            print(f"[WARN] Ollama 未找到 LLM 模型: {model}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] Ollama 无法访问 {ollama_base}: {exc}")
    return True


def check_embed() -> bool:
    backend = _env("EMBED_BACKEND", "local").lower()

    if backend == "api":
        base = (_env("EMBED_BASE_URL") or _env("LLM_BASE_URL")).rstrip("/")
        model = _env("EMBED_MODEL")
        if not base or not model:
            print("[FAIL] EMBED_BACKEND=api 需要配置 EMBED_MODEL（以及 EMBED_BASE_URL 或 LLM_BASE_URL）")
            return False
        dim = _env("EMBED_DIM")
        print(f"[OK] Embedding API 配置就绪: {base} (model={model}, dim={dim or '自动探测'})")
        return True

    ollama_base = _env("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
    model = _env("OLLAMA_EMBED_MODEL")
    try:
        tags = fetch_json(f"{ollama_base}/api/tags")
        names = {m.get("name", "") for m in tags.get("models", [])}
        print(f"[OK] Ollama 可访问: {ollama_base}")
        if model and model not in names:
            print(f"[WARN] Ollama 未找到 Embedding 模型: {model}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[WARN] Ollama 无法访问 {ollama_base}: {exc}")
    return True


def check_rerank() -> bool:
    backend = _env("RERANK_BACKEND", "qwen3reranker").lower()
    enabled = _env("RERANK_ENABLED", "1") not in {"0", "false", "False"}

    if not enabled or backend in {"none", "off", "disabled"}:
        print("[OK] Rerank 未启用")
        return True

    if backend == "api":
        url = _env("RERANK_API_URL")
        if not url:
            print("[FAIL] RERANK_BACKEND=api 需要配置 RERANK_API_URL（或设为 none 关闭精排）")
            return False
        print(f"[OK] Rerank API 配置就绪: {url} (model={_env('RERANK_API_MODEL') or '服务默认'})")
        return True

    print(f"[OK] Rerank 使用本地模型: backend={backend}, model={_env('HF_RERANK_MODEL') or _env('RERANK_MODEL')}")
    return True


def check_milvus() -> bool:
    milvus_uri = _env("MILVUS_URI", "http://127.0.0.1:19530")
    try:
        from pymilvus import connections

        connections.connect(alias="check_env", uri=milvus_uri)
        connections.disconnect("check_env")
        print(f"[OK] Milvus 可访问: {milvus_uri}")
        return True
    except Exception as exc:
        print(f"[WARN] Milvus 无法访问 {milvus_uri}: {exc}")
        return True


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    ok = True
    if sys.version_info < (3, 10):
        print(f"[FAIL] 需要 Python >= 3.10，当前: {sys.version.split()[0]}")
        ok = False
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    for module in [
        "llama_index",
        "pymilvus",
        "fastapi",
        "pydantic",
        "jieba",
    ]:
        ok = check_import(module) and ok

    # 按后端校验可选依赖
    if _env("LLM_BACKEND", "local").lower() == "api" or _env("EMBED_BACKEND", "local").lower() == "api":
        for module in ["llama_index.llms.openai_like", "llama_index.embeddings.openai"]:
            check_import(module)
    if _env("LLM_BACKEND", "local").lower() == "local" or _env("EMBED_BACKEND", "local").lower() == "local":
        check_import("llama_index.llms.ollama")
        check_import("llama_index.embeddings.ollama")

    if (root / ".env").exists():
        print("[OK] .env 存在")
    else:
        print("[FAIL] 缺少 .env；请先复制 .env.example 为 .env")
        ok = False

    data_dir = root / _env("DATA_DIR", "./data/clear_docs")
    if data_dir.exists():
        print(f"[OK] 数据目录: {data_dir}")
    else:
        print(f"[FAIL] 数据目录不存在: {data_dir}")
        ok = False

    check_llm()
    check_embed()
    check_rerank()
    check_milvus()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
