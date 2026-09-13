"""PCB-RAG 模型客户端工厂（本地 / API 双后端）。

统一管理 LLM、Embedding、Rerank 三类模型的构建，支持两种后端并可自由切换：

LLM
  - ``local`` ：Ollama 本地服务   -> ``OLLAMA_BASE`` / ``OLLAMA_LLM_MODEL``
  - ``api``   ：OpenAI 兼容接口   -> ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``

Embedding
  - ``local`` ：Ollama 本地服务   -> ``OLLAMA_BASE`` / ``OLLAMA_EMBED_MODEL``
  - ``api``   ：OpenAI 兼容接口   -> ``EMBED_BASE_URL`` / ``EMBED_API_KEY`` / ``EMBED_MODEL``

Rerank（后端分发在 ``query.py`` 的 ``_try_build_reranker`` 中完成）
  - ``api``            ：HTTP rerank 接口（Jina / 硅基流动 / DashScope 兼容）
  - ``hf``             ：本地 HuggingFace Transformers
  - ``qwen3reranker``  ：本地 Qwen3-Reranker 生成式模型
  - ``sbert``          ：本地 SentenceTransformer cross-encoder
  - ``none``           ：关闭精排

用法::

    from pcb_rag.api_clients import build_llm, build_embed_model, get_embedding_dim
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 0. 代理环境变量清理（必须在任何网络客户端 import 之前执行，幂等）
# ---------------------------------------------------------------------------
def _sanitize_proxy_env_for_httpx() -> None:
    """修正 httpx 不识别的代理 scheme，并将本地服务加入 NO_PROXY。"""

    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]

    for key in proxy_keys:
        val = os.environ.get(key)
        if not val:
            continue
        v = val.strip()
        if v.lower().startswith("socks://"):
            os.environ[key] = "socks5://" + v[len("socks://") :]

    no_proxy_key = (
        "NO_PROXY"
        if "NO_PROXY" in os.environ
        else "no_proxy"
        if "no_proxy" in os.environ
        else "NO_PROXY"
    )
    existing = os.environ.get(no_proxy_key, "")
    entries = [e.strip() for e in existing.split(",") if e.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in entries:
            entries.append(host)
    os.environ[no_proxy_key] = ",".join(entries)


_sanitize_proxy_env_for_httpx()

from llama_index.core.bridge.pydantic import Field  # noqa: E402
from llama_index.core.postprocessor.types import BaseNodePostprocessor  # noqa: E402
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle  # noqa: E402

# ---------------------------------------------------------------------------
# 1. 后端选择
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").strip().lower()
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local").strip().lower()

# ---- 本地后端（Ollama） ----
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:35b-a3b-q4_K_M").strip()
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b-q8_0").strip()
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
EMBED_NUM_CTX = int(os.getenv("EMBED_NUM_CTX", "2048"))

# ---- API 后端（OpenAI 兼容） ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "").strip().rstrip("/") or LLM_BASE_URL
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "").strip() or LLM_API_KEY
EMBED_MODEL = os.getenv("EMBED_MODEL", "").strip()
EMBED_DIM = int(os.getenv("EMBED_DIM", "0") or 0)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# ---- Rerank API（OpenAI 兼容之外的 /rerank 接口） ----
RERANK_API_URL = os.getenv("RERANK_API_URL", "").strip()
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "").strip() or LLM_API_KEY
RERANK_API_MODEL = os.getenv("RERANK_API_MODEL", "").strip()
RERANK_API_TIMEOUT = float(os.getenv("RERANK_API_TIMEOUT", "60"))

_PLACEHOLDER_KEY = "sk-no-key-required"


# ---------------------------------------------------------------------------
# 2. LLM 工厂
# ---------------------------------------------------------------------------
def build_llm(
    model: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    request_timeout: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Any:
    """构建 LLM 实例（``local`` = Ollama，``api`` = OpenAI 兼容）。"""

    backend = (backend or LLM_BACKEND).strip().lower()
    temp = 0.7 if temperature is None else float(temperature)

    if backend == "api":
        from llama_index.llms.openai_like import OpenAILike

        if not LLM_BASE_URL:
            raise RuntimeError(
                "LLM_BACKEND=api 需要配置 LLM_BASE_URL（OpenAI 兼容的 /v1 地址）"
            )
        model_name = (model or LLM_MODEL).strip()
        if not model_name:
            raise RuntimeError("LLM_BACKEND=api 需要配置 LLM_MODEL")

        return OpenAILike(
            model=model_name,
            api_base=LLM_BASE_URL,
            api_key=LLM_API_KEY or _PLACEHOLDER_KEY,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=LLM_CONTEXT_WINDOW,
            max_tokens=LLM_MAX_TOKENS,
            temperature=temp,
            timeout=request_timeout or LLM_TIMEOUT,
        )

    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=model or OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE,
        request_timeout=request_timeout or OLLAMA_TIMEOUT,
        context_window=OLLAMA_NUM_CTX,
        additional_kwargs={
            "num_ctx": OLLAMA_NUM_CTX,
            "temperature": temp,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
        },
        thinking=False,
    )


# ---------------------------------------------------------------------------
# 3. Embedding 工厂
# ---------------------------------------------------------------------------
def build_embed_model(model: Optional[str] = None, *, backend: Optional[str] = None) -> Any:
    """构建 Embedding 实例（``local`` = Ollama，``api`` = OpenAI 兼容）。"""

    backend = (backend or EMBED_BACKEND).strip().lower()

    if backend == "api":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not EMBED_BASE_URL:
            raise RuntimeError(
                "EMBED_BACKEND=api 需要配置 EMBED_BASE_URL（或复用 LLM_BASE_URL）"
            )
        model_name = (model or EMBED_MODEL).strip()
        if not model_name:
            raise RuntimeError("EMBED_BACKEND=api 需要配置 EMBED_MODEL")

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_base": EMBED_BASE_URL,
            "api_key": EMBED_API_KEY or _PLACEHOLDER_KEY,
            "embed_batch_size": EMBED_BATCH_SIZE,
        }
        if EMBED_DIM > 0:
            kwargs["dimensions"] = EMBED_DIM
        try:
            return OpenAIEmbedding(**kwargs)
        except TypeError:
            # 部分版本 / 服务不支持 dimensions 参数，降级重试
            kwargs.pop("dimensions", None)
            return OpenAIEmbedding(**kwargs)

    from llama_index.embeddings.ollama import OllamaEmbedding

    # 修复：原先硬编码模型名，导致 OLLAMA_EMBED_MODEL 环境变量完全失效
    return OllamaEmbedding(
        model_name=model or OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE,
        ollama_additional_kwargs={"num_ctx": EMBED_NUM_CTX},
    )


def get_embedding_dim(embed_model: Any = None) -> Optional[int]:
    """获取 embedding 维度：优先 ``EMBED_DIM``，否则实际请求一次探测。"""

    if EMBED_DIM > 0:
        return int(EMBED_DIM)

    model = embed_model if embed_model is not None else build_embed_model()
    try:
        return len(model.get_query_embedding("dimension probe"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Rerank API 客户端
# ---------------------------------------------------------------------------
class ApiReranker(BaseNodePostprocessor):
    """通过 HTTP ``/rerank`` 接口做精排。

    兼容主流返回格式：
      - ``{"results": [{"index": 0, "relevance_score": 0.9}, ...]}``  (Jina / 硅基流动)
      - ``{"output": {"results": [...]}}``                            (DashScope)
    """

    api_url: str = Field(description="Rerank API endpoint")
    api_key: str = Field(default="", description="Bearer token")
    model: str = Field(default="", description="Rerank 模型名")
    top_n: int = Field(default=10, description="保留的文档数")
    timeout: float = Field(default=60.0, description="请求超时（秒）")

    @classmethod
    def class_name(cls) -> str:
        return "ApiReranker"

    def _request_scores(self, query: str, documents: Sequence[str]) -> List[float]:
        import requests

        payload: dict[str, Any] = {
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        if self.model:
            payload["model"] = self.model

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results")
        if results is None and isinstance(data.get("output"), dict):
            results = data["output"].get("results")
        if results is None and isinstance(data.get("data"), list):
            results = data["data"]
        if not isinstance(results, list):
            raise RuntimeError(f"Rerank API 响应格式无法识别: {str(data)[:200]}")

        scores = [0.0] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if idx is None or score is None:
                continue
            idx = int(idx)
            if 0 <= idx < len(scores):
                scores[idx] = float(score)
        return scores

    def _postprocess_nodes(
        self,
        nodes: Optional[List[NodeWithScore]] = None,
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        nodes = nodes or []
        if not nodes:
            return []
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        documents = [
            n.node.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes
        ]

        try:
            scores = self._request_scores(query_bundle.query_str, documents)
        except Exception as exc:  # 网络/接口异常时保底，避免整条链路失败
            print(f"[Rerank] API 精排失败，回退原始排序: {exc}")
            return sorted(nodes, key=lambda x: -(x.score or 0.0))[: self.top_n]

        for node, score in zip(nodes, scores):
            node.score = float(score)
        return sorted(nodes, key=lambda x: -(x.score or 0.0))[: self.top_n]


def build_api_reranker(top_n: int) -> ApiReranker:
    """构建 API Reranker，缺少必要配置时抛出明确异常。"""

    if not RERANK_API_URL:
        raise RuntimeError(
            "RERANK_BACKEND=api 需要配置 RERANK_API_URL"
            "（例如 https://api.jina.ai/v1/rerank 或 https://api.siliconflow.cn/v1/rerank）"
        )
    return ApiReranker(
        api_url=RERANK_API_URL,
        api_key=RERANK_API_KEY,
        model=RERANK_API_MODEL,
        top_n=top_n,
        timeout=RERANK_API_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# 5. 诊断信息
# ---------------------------------------------------------------------------
def describe_backends() -> dict:
    """返回当前后端配置摘要，便于启动日志与 ``/health`` 展示。"""

    return {
        "llm_backend": LLM_BACKEND,
        "llm_model": LLM_MODEL if LLM_BACKEND == "api" else OLLAMA_LLM_MODEL,
        "llm_base_url": LLM_BASE_URL if LLM_BACKEND == "api" else OLLAMA_BASE,
        "llm_configured": bool(LLM_BASE_URL and LLM_MODEL) if LLM_BACKEND == "api" else bool(OLLAMA_BASE),
        "embed_backend": EMBED_BACKEND,
        "embed_model": EMBED_MODEL if EMBED_BACKEND == "api" else OLLAMA_EMBED_MODEL,
        "embed_base_url": EMBED_BASE_URL if EMBED_BACKEND == "api" else OLLAMA_BASE,
        "embed_configured": bool(EMBED_BASE_URL and EMBED_MODEL) if EMBED_BACKEND == "api" else bool(OLLAMA_BASE),
        "embed_dim": EMBED_DIM or None,
    }
