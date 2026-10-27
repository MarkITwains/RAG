"""PCB-RAG 环境自检脚本。

按当前后端配置检查：
  - Python 版本与关键依赖
  - .env / 数据目录
  - LLM 后端（local=Ollama，api=OpenAI 兼容）
  - Embedding 后端（local=Ollama，api=OpenAI 兼容）
  - Rerank 后端（api=HTTP 接口，其余为本地模型，仅做配置校验）
  - Milvus 连通性
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env(path: Path) -> None:
    src_dir = str(path.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from pcb_rag.env_loader import load_project_env

    load_project_env(path)


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
        provider = _env("EMBED_API_PROVIDER", "auto") or "auto"
        print(
            f"[OK] Embedding API 配置就绪: {base} "
            f"(model={model}, dim={dim or '自动探测'}, provider={provider})"
        )
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
        provider = _env("RERANK_API_PROVIDER", "auto") or "auto"
        print(
            f"[OK] Rerank API 配置就绪: {url} "
            f"(model={_env('RERANK_API_MODEL') or '服务默认'}, provider={provider})"
        )
        if url.rstrip("/").endswith("/reranking") and provider.lower() not in {"nvidia", "auto"}:
            print(
                "[WARN] 该地址是 NIM reranking 接口，请求体为 query/passages 对象且返回 rankings[].logit；"
                "provider 应设为 nvidia，否则按 /v1/rerank 兼容格式发送会被 400 拒绝"
            )
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


def required_modules():
    modules = [
        "llama_index.core",
        "llama_index.vector_stores.milvus",
        "llama_index.readers.file",
        "pymilvus",
        "fastapi",
        "pydantic",
        "httpx",
        "jieba",
        "fitz",
        "docx",
    ]
    modules.append(
        "llama_index.llms.openai_like" if _env("LLM_BACKEND", "local").lower() == "api" else "llama_index.llms.ollama"
    )
    modules.append(
        "llama_index.embeddings.openai"
        if _env("EMBED_BACKEND", "local").lower() == "api"
        else "llama_index.embeddings.ollama"
    )
    if _env("RERANK_ENABLED", "1").lower() not in {"0", "false"} and _env(
        "RERANK_BACKEND", "qwen3reranker"
    ).lower() not in {
        "api",
        "none",
        "off",
        "disabled",
    }:
        modules.extend(["torch", "transformers", "sentence_transformers"])
    return modules


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dependencies-only", action="store_true", help="check imports without network or corpus access"
    )
    parser.add_argument("--config-only", action="store_true", help="audit dotenv keys without network or model access")
    parser.add_argument("--strict-config", action="store_true", help="fail on unknown, duplicate, or overridden dotenv keys")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.config_only:
        from pcb_rag.config_audit import audit_dotenv

        report = audit_dotenv(root / ".env", root=root)
        print(f"[OK] dotenv entries={report['entries']}, known keys={report['known_keys']}")
        for key in report["duplicates"]:
            print(f"[FAIL] duplicate dotenv key: {key}")
        for key in report["unknown"]:
            print(f"[FAIL] dotenv key has no code consumer: {key}")
        for key in report["legacy"]:
            print(f"[WARN] legacy-only dotenv key: {key}")
        for key in report["overridden"]:
            print(f"[WARN] external environment overrides dotenv key: {key}")
        return int(bool(report["duplicates"] or report["unknown"] or (args.strict_config and report["overridden"])))
    load_env(root / ".env")

    ok = True
    if sys.version_info < (3, 10):
        print(f"[FAIL] 需要 Python >= 3.10，当前: {sys.version.split()[0]}")
        ok = False
    else:
        print(f"[OK] Python {sys.version.split()[0]}")

    for module in required_modules():
        ok = check_import(module) and ok

    if args.dependencies_only:
        return 0 if ok else 1

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

    ok = check_llm() and ok
    ok = check_embed() and ok
    ok = check_rerank() and ok
    ok = check_milvus() and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
