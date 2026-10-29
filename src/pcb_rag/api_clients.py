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
import logging
import math
import time
from typing import Any, List, Optional, Sequence
from urllib.parse import urlsplit

from pcb_rag.embedding_contract import embedding_profile, resolve_embedding_provider
from pcb_rag.rerank_control import RerankControl, RerankStopped, extract_usage, provider_identity

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
EMBED_API_PROVIDER = os.getenv("EMBED_API_PROVIDER", "auto").strip().lower()
EMBED_TRUNCATE = os.getenv("EMBED_TRUNCATE", "END").strip().upper()

# ---- Rerank API（OpenAI 兼容之外的 /rerank 接口） ----
RERANK_API_URL = os.getenv("RERANK_API_URL", "").strip()
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "").strip() or LLM_API_KEY
RERANK_API_MODEL = os.getenv("RERANK_API_MODEL", "").strip()
RERANK_API_TIMEOUT = float(os.getenv("RERANK_API_TIMEOUT", "60"))
#: ``auto`` / ``openai`` / ``nvidia``。``openai`` 指 Jina、硅基流动、DashScope 这类
#: ``/v1/rerank`` 兼容接口；``nvidia`` 指 NIM 的 ``/retrieval/.../reranking`` 接口，
#: 两者请求体与返回体都不同，不能混用。
RERANK_API_PROVIDER = os.getenv("RERANK_API_PROVIDER", "auto").strip().lower()
RERANK_TRUNCATE = os.getenv("RERANK_TRUNCATE", "END").strip().upper()

#: NIM reranking 接口的 passages 数量上限（服务端硬限制）。
NVIDIA_RERANK_MAX_PASSAGES = 512

_PLACEHOLDER_KEY = "sk-no-key-required"

#: ``dimensions`` 参数的发送策略：auto / 1 / 0
#:
#: ``dimensions`` 是 OpenAI 自家 text-embedding-3* 用来降维的参数，其它模型和自建网关
#: 多数不支持 —— 传了会被 400 拒绝，且报错信息通常与"维度"无关，很难定位。
#: 默认 ``auto``：仅当模型名形如 ``text-embedding-3*`` 时才发送，
#: 既保留 OpenAI 用户的降维能力，又不会把自定义网关的请求打挂。
EMBED_SEND_DIMENSIONS = os.getenv("EMBED_SEND_DIMENSIONS", "auto").strip().lower()


def _should_send_dimensions(model_name: str) -> bool:
    """判断是否随请求发送 ``dimensions`` 参数（见 ``EMBED_SEND_DIMENSIONS``）。"""
    if EMBED_SEND_DIMENSIONS in {"1", "true", "yes", "on"}:
        return True
    if EMBED_SEND_DIMENSIONS in {"0", "false", "no", "off"}:
        return False
    return model_name.lower().startswith("text-embedding-3")


def api_only_enabled() -> bool:
    return os.getenv("AI_API_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}


def assert_api_only(*, judge: bool = False) -> None:
    if not api_only_enabled():
        return
    backends = {"LLM_BACKEND": LLM_BACKEND, "EMBED_BACKEND": EMBED_BACKEND}
    rerank_enabled = os.getenv("RERANK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    rerank_backend = os.getenv("RERANK_BACKEND", "qwen3reranker").strip().lower()
    if rerank_enabled and rerank_backend not in {"api", "none", "off", "disabled"}:
        backends["RERANK_BACKEND"] = rerank_backend
    if os.getenv("COLBERT_RERANK_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
        backends["COLBERT_RERANK_ENABLED"] = "local"
    if judge:
        backends["JUDGE_BACKEND"] = os.getenv("JUDGE_BACKEND", "api").strip().lower()
    invalid = [f"{name}={value}" for name, value in backends.items() if value != "api"]
    if invalid:
        raise RuntimeError("AI_API_ONLY=1 要求所有 AI 调用使用 API: " + ", ".join(invalid))


# ---------------------------------------------------------------------------
# 2. LLM 工厂
# ---------------------------------------------------------------------------
def build_llm(
    model: Optional[str] = None,
    *,
    backend: Optional[str] = None,
    request_timeout: Optional[float] = None,
    temperature: Optional[float] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """构建 LLM 实例（``local`` = Ollama，``api`` = OpenAI 兼容）。"""

    backend = (backend or LLM_BACKEND).strip().lower()
    temp = LLM_TEMPERATURE if temperature is None else float(temperature)
    if api_only_enabled() and backend != "api":
        raise RuntimeError("AI_API_ONLY=1 禁止 LLM 使用本地后端")

    if backend == "api":
        from llama_index.llms.openai_like import OpenAILike

        base_url = (api_base or LLM_BASE_URL).strip().rstrip("/")
        credential = LLM_API_KEY if api_key is None else api_key.strip()
        if not base_url:
            raise RuntimeError(
                "LLM_BACKEND=api 需要配置 LLM_BASE_URL（OpenAI 兼容的 /v1 地址）"
            )
        model_name = (model or LLM_MODEL).strip()
        if not model_name:
            raise RuntimeError("LLM_BACKEND=api 需要配置 LLM_MODEL")

        return OpenAILike(
            model=model_name,
            api_base=base_url,
            api_key=credential or _PLACEHOLDER_KEY,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=LLM_CONTEXT_WINDOW,
            max_tokens=LLM_MAX_TOKENS,
            temperature=temp,
            timeout=request_timeout or LLM_TIMEOUT,
            max_retries=0,
        )

    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=model or OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE,
        request_timeout=request_timeout or OLLAMA_TIMEOUT,
        context_window=OLLAMA_NUM_CTX,
        additional_kwargs={
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": LLM_MAX_TOKENS or int(os.getenv("RAG_OUTPUT_RESERVE_TOKENS", "1024")),
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
    if api_only_enabled() and backend != "api":
        raise RuntimeError("AI_API_ONLY=1 禁止 Embedding 使用本地后端")

    if backend == "api":
        from llama_index.embeddings.openai import OpenAIEmbedding
        from pcb_rag.metered_embeddings import EmbeddingUsageMixin

        class MeteredOpenAIEmbedding(EmbeddingUsageMixin, OpenAIEmbedding):
            pass

        if not EMBED_BASE_URL:
            raise RuntimeError(
                "EMBED_BACKEND=api 需要配置 EMBED_BASE_URL（或复用 LLM_BASE_URL）"
            )
        model_name = (model or EMBED_MODEL).strip()
        if not model_name:
            raise RuntimeError("EMBED_BACKEND=api 需要配置 EMBED_MODEL")

        # 注意两个参数的区别（踩过的坑）：
        #   ``model``      —— llama-index 会拿它去构造 OpenAIEmbeddingModelType 枚举，
        #                     **不在枚举里的模型名会直接抛 ValueError**
        #                     （例如自建网关的 "nvidia/nemotron-3-embed-1b:free"，
        #                       报错：is not a valid OpenAIEmbeddingModelType）
        #   ``model_name`` —— 不上枚举校验，直接作为请求里的 model 发出去
        # 因此自定义 / 网关模型 id 必须走 model_name；OpenAI 官方模型两者等价。
        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "api_base": EMBED_BASE_URL,
            "api_key": EMBED_API_KEY or _PLACEHOLDER_KEY,
            "embed_batch_size": EMBED_BATCH_SIZE,
            "max_retries": 0,
            "timeout": 120.0,
        }
        provider = resolve_embedding_provider(EMBED_API_PROVIDER, EMBED_BASE_URL)
        if provider == "nvidia":
            from pcb_rag.nvidia_embeddings import NvidiaApiEmbedding

            # NIM embeddings accept only END / NONE; START is rejected with HTTP 400.
            if EMBED_TRUNCATE not in {"END", "NONE"}:
                raise ValueError("EMBED_TRUNCATE must be END or NONE for the NVIDIA embeddings API")
            if EMBED_DIM and not 1 <= EMBED_DIM <= 4096:
                raise ValueError("EMBED_DIM must be between 1 and the model's native 4096 dimensions")
            return NvidiaApiEmbedding(
                **kwargs,
                truncate=EMBED_TRUNCATE,
                expected_dimension=EMBED_DIM,
            )
        if EMBED_DIM > 0 and _should_send_dimensions(model_name):
            kwargs["dimensions"] = EMBED_DIM
        try:
            return MeteredOpenAIEmbedding(**kwargs)
        except TypeError:
            # 部分版本 / 服务不支持 dimensions 参数，降级重试
            kwargs.pop("dimensions", None)
            return MeteredOpenAIEmbedding(**kwargs)

    from llama_index.embeddings.ollama import OllamaEmbedding
    from pcb_rag.metered_embeddings import EmbeddingUsageMixin

    class MeteredOllamaEmbedding(EmbeddingUsageMixin, OllamaEmbedding):
        pass

    # 修复：原先硬编码模型名，导致 OLLAMA_EMBED_MODEL 环境变量完全失效
    import inspect

    kwargs = {}
    if "client_kwargs" in inspect.signature(OllamaEmbedding.__init__).parameters:
        kwargs["client_kwargs"] = {"timeout": OLLAMA_TIMEOUT}
    embed = MeteredOllamaEmbedding(
        model_name=model or OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE,
        ollama_additional_kwargs={"num_ctx": EMBED_NUM_CTX},
        **kwargs,
    )
    if not kwargs:
        # 0.4.x creates the Ollama clients before a subclass can pass client_kwargs.
        # httpx requests inherit this default; update both sync and async clients.
        import httpx

        for client in (embed._client, embed._async_client):
            client._client._timeout = httpx.Timeout(OLLAMA_TIMEOUT)
    return embed


def get_embedding_profile(dimension: Optional[int] = None, *, document_text="llama_index_metadata_embed_v1") -> dict:
    """Identity of the API vector space, without credentials."""
    return embedding_profile(
        provider=resolve_embedding_provider(EMBED_API_PROVIDER, EMBED_BASE_URL),
        api_base=EMBED_BASE_URL,
        model=EMBED_MODEL,
        dimension=dimension if dimension is not None else EMBED_DIM,
        truncate=EMBED_TRUNCATE,
        document_text=document_text,
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
# ---------------------------------------------------------------------------
# Rerank input, usage and budget controls. Rate reservations are shared across
# processes. A zero interval explicitly disables proactive throttling; provider
# 429 responses remain visible and retries are bounded.
# ---------------------------------------------------------------------------
RERANK_MIN_INTERVAL = max(0.0, float(os.getenv("RERANK_MIN_INTERVAL", "0")))
RERANK_429_RETRIES = max(0, int(os.getenv("RERANK_429_RETRIES", "2")))
RERANK_429_BACKOFF_SECONDS = max(0.0, float(os.getenv("RERANK_429_BACKOFF_SECONDS", "60")))
_RERANK_LOGGER = logging.getLogger("pcb_rag.rerank")


class ApiReranker(BaseNodePostprocessor):
    """通过 HTTP ``/rerank`` 接口做精排。

    兼容主流返回格式：
      - ``{"results": [{"index": 0, "relevance_score": 0.9}, ...]}``  (Jina / 硅基流动)
      - ``{"output": {"results": [...]}}``                            (DashScope)
    """

    api_url: str = Field(description="Rerank API endpoint")
    api_key: str = Field(default="", description="Bearer token")
    model: str = Field(default="", description="Rerank 模型名")
    top_n: int = Field(default=10, ge=1, description="保留的文档数")
    timeout: float = Field(default=60.0, description="请求超时（秒）")
    max_candidates: int = Field(default=50, ge=1, description="实际送入 API 的候选上限")
    max_document_chars: int = Field(default=1200, ge=1)
    max_query_chars: int = Field(default=1000, ge=1)

    @classmethod
    def class_name(cls) -> str:
        return "ApiReranker"

    def _wire_payload(self, query: str, documents: Sequence[str], wanted: int) -> dict[str, Any]:
        """Build the provider-specific request body (Jina / 硅基流动 / DashScope 兼容)."""
        payload: dict[str, Any] = {
            "query": query,
            "documents": list(documents),
            "top_n": wanted,
            "return_documents": False,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _parse_scores(
        self, data: dict[str, Any], documents: Sequence[str], wanted: int
    ) -> List[Optional[float]]:
        """Map a provider response onto document positions, rejecting malformed output."""
        results = data.get("results")
        if results is None and isinstance(data.get("output"), dict):
            results = data["output"].get("results")
        if results is None and isinstance(data.get("data"), list):
            results = data["data"]
        if not isinstance(results, list) or len(results) < wanted:
            raise ValueError("Missing rerank scores")
        scores: List[Optional[float]] = [None] * len(documents)
        for item in results:
            idx = item.get("index")
            score = float(item.get("relevance_score", item.get("score")))
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(scores):
                raise ValueError("Invalid rerank index")
            if scores[idx] is not None or not math.isfinite(score):
                raise ValueError("Duplicate index or nonfinite rerank score")
            scores[idx] = score
        return scores

    def _request_scores(self, query: str, documents: Sequence[str]) -> List[Optional[float]]:
        import requests

        if not documents:
            return []
        if len(documents) > self.max_candidates:
            raise ValueError("Rerank candidates exceed the configured input cap")
        query = query[:self.max_query_chars]
        documents = [document[:self.max_document_chars] for document in documents]
        wanted = min(self.top_n, len(documents))
        payload = self._wire_payload(query, documents, wanted)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        control = RerankControl.from_env()
        provider = provider_identity(self.api_url, self.api_key)
        phase = os.getenv("RERANK_PHASE", "production")
        activity = os.getenv("RERANK_ACTIVITY", "request")
        from pcb_rag.usage import finish_usage, reserve_usage
        usage_tickets = {}

        def record(ticket, **kwargs):
            event = control.finish(ticket, **kwargs)
            usage_ticket = usage_tickets.get(ticket.request_id)
            if usage_ticket:
                finish_usage(usage_ticket, {"usage": kwargs.get("usage", {})}, error=kwargs.get("status") != "success")
            event.update(
                model=self.model, phase=phase, activity=activity, candidates=len(documents), wire_top_n=wanted,
                estimated_tokens=ticket.estimated_tokens,
            )
            _RERANK_LOGGER.info("[RerankUsage] %s", event, extra={"rerank_usage": event})

        for attempt in range(RERANK_429_RETRIES + 1):
            ticket = control.reserve(
                provider=provider, model=self.model, query=query, documents=documents,
                interval=RERANK_MIN_INTERVAL, phase=phase, activity=activity, top_n=wanted,
            )
            control.wait(ticket)
            try:
                from pcb_rag.runtime import current_control
                request = current_control()
                timeout = min(self.timeout, request.remaining()) if request else self.timeout
                usage_tickets[ticket.request_id] = reserve_usage("rerank", [query, *documents], self.model)
            except BaseException:
                control.finish(ticket, status="cancelled", actual_tokens=0, usage_source="not_sent")
                raise
            try:
                control.mark_inflight(ticket)
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout, allow_redirects=False)
            except requests.RequestException as exc:
                record(ticket, status="transport_error", error_type=type(exc).__name__, stop_reason="transport_error_billing_unknown")
                raise RerankStopped("Rerank transport failed; billing is unknown and automatic retries are disabled") from exc
            try:
                data = resp.json()
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            tokens, source, usage = extract_usage(data)
            message = str(data.get("message", data.get("error", "")))[:500]
            if self.api_key:
                message = message.replace(self.api_key, "[REDACTED]")
            response_meta = {
                "provider_request_id": str(data.get("id", ""))[:120],
                "error_code": str(data.get("code", ""))[:80],
                "error_message": message,
                "retry_after": resp.headers.get("Retry-After"),
                "rate_limit_headers": {
                    key: str(value)[:256] for key, value in resp.headers.items()
                    if key.lower().startswith("x-ratelimit-")
                },
            }
            accounting = {
                "actual_tokens": tokens, "usage_source": source, "usage": usage,
                "http_status": resp.status_code, "response_meta": response_meta,
            }
            if not 200 <= resp.status_code < 300:
                fatal = not (resp.status_code == 429 and attempt < RERANK_429_RETRIES)
                record(ticket, status="http_error", stop_reason=f"http_{resp.status_code}" if fatal else None, **accounting)
                if resp.status_code == 429 and attempt < RERANK_429_RETRIES:
                    retry_after = resp.headers.get("Retry-After", "")
                    try:
                        backoff = max(0.0, min(60.0, float(retry_after))) if retry_after else min(60.0, RERANK_429_BACKOFF_SECONDS * 2.0 ** attempt)
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        try:
                            backoff = max(0.0, min(60.0, parsedate_to_datetime(retry_after).timestamp() - time.time()))
                        except (ValueError, TypeError, OverflowError):
                            backoff = min(60.0, RERANK_429_BACKOFF_SECONDS * 2.0 ** attempt)
                    print(f"[Rerank] 429 限流，重试 {attempt + 1}/{RERANK_429_RETRIES}，等待 {backoff:.1f}s", flush=True)
                    deadline = time.monotonic() + backoff
                    while time.monotonic() < deadline:
                        control.check()
                        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
                    continue
                if resp.status_code == 429:
                    print(f"[Rerank] 429 限流，无剩余重试，Retry-After={resp.headers.get('Retry-After', '')}", flush=True)
                raise RerankStopped(f"Rerank HTTP {resp.status_code}; circuit opened, no ranking fallback")
            try:
                scores = self._parse_scores(data, documents, wanted)
            except (ValueError, TypeError, AttributeError) as exc:
                record(ticket, status="protocol_error", error_type=type(exc).__name__, stop_reason="invalid_rerank_response", **accounting)
                raise RerankStopped("Rerank response invalid; circuit opened, no ranking fallback") from exc
            record(ticket, status="success", **accounting)
            return scores
        raise RerankStopped("Rerank retries exhausted")

    def _postprocess_nodes(
        self,
        nodes: Optional[List[NodeWithScore]] = None,
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        nodes = list(nodes or [])[:self.max_candidates]
        if not nodes:
            return []
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")

        documents = [
            n.node.get_content(metadata_mode=MetadataMode.NONE) for n in nodes
        ]
        scores = self._request_scores(query_bundle.query_str, documents)
        scored = []
        for node, score in zip(nodes, scores):
            if score is not None:
                node.score = float(score)
                scored.append(node)
        return sorted(scored, key=lambda x: -x.score)[: self.top_n]


class NvidiaApiReranker(ApiReranker):
    """NIM ``/retrieval/<model>/reranking`` 接口适配。

    与 Jina / 硅基流动兼容接口的三处差异（2026-10-23 实测确认）：

    - ``query`` 与每个 passage 都是对象（``{"text": ...}``），传纯字符串会被 400 拒绝；
    - 不接受 ``top_n`` / ``documents`` / ``return_documents``，传入即 ``extra_forbidden``；
      服务端返回全部 passage 的排序，因此裁剪发生在本地；
    - 返回 ``{"rankings": [{"index": 0, "logit": -1.955}, ...]}``，分值是 logit。
      为保持与旧 reranker 一致的 0~1 外显语义（sources[].score 直接对外），
      此处按 ``sigmoid(logit)`` 归一化；排序结果与 logit 完全一致。
    """

    truncate: str = Field(default="END")

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaApiReranker"

    def _wire_payload(self, query: str, documents: Sequence[str], wanted: int) -> dict[str, Any]:
        if len(documents) > NVIDIA_RERANK_MAX_PASSAGES:
            raise ValueError(f"NVIDIA reranking accepts at most {NVIDIA_RERANK_MAX_PASSAGES} passages")
        if self.truncate not in {"END", "NONE"}:
            raise ValueError("RERANK_TRUNCATE must be END or NONE for the NVIDIA reranking API")
        payload: dict[str, Any] = {
            "query": {"text": query},
            "passages": [{"text": document} for document in documents],
            "truncate": self.truncate,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _parse_scores(
        self, data: dict[str, Any], documents: Sequence[str], wanted: int
    ) -> List[Optional[float]]:
        rankings = data.get("rankings")
        if not isinstance(rankings, list) or len(rankings) < len(documents):
            raise ValueError("Missing rerank scores")
        scores: List[Optional[float]] = [None] * len(documents)
        for item in rankings:
            idx = item.get("index")
            logit = float(item.get("logit", item.get("score")))
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(scores):
                raise ValueError("Invalid rerank index")
            if scores[idx] is not None or not math.isfinite(logit):
                raise ValueError("Duplicate index or nonfinite rerank score")
            scores[idx] = 1.0 / (1.0 + math.exp(-logit))
        return scores


def resolve_rerank_provider(setting: str, api_url: str) -> str:
    """判定 rerank 线路格式（``auto`` / ``openai`` / ``nvidia``）。

    自动识别同时要求官方域名与 ``/reranking`` 路径后缀，避免伪造域名被误判。
    自建 NIM 网关请显式设置 ``RERANK_API_PROVIDER=nvidia``。
    """

    provider = setting.strip().lower()
    if provider not in {"auto", "openai", "nvidia"}:
        raise ValueError("RERANK_API_PROVIDER must be auto, openai, or nvidia")
    if provider != "auto":
        return provider
    parts = urlsplit(api_url)
    host = (parts.hostname or "").lower()
    if host in {"ai.api.nvidia.com", "integrate.api.nvidia.com"} and parts.path.rstrip("/").endswith("/reranking"):
        return "nvidia"
    return "openai"


def build_api_reranker(top_n: int) -> ApiReranker:
    """构建 API Reranker，缺少必要配置时抛出明确异常。"""

    if not RERANK_API_URL:
        raise RuntimeError(
            "RERANK_BACKEND=api 需要配置 RERANK_API_URL"
            "（例如 https://api.jina.ai/v1/rerank，或 NIM 的 "
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking）"
        )
    common: dict[str, Any] = {
        "api_url": RERANK_API_URL,
        "api_key": RERANK_API_KEY,
        "model": RERANK_API_MODEL,
        "top_n": top_n,
        "timeout": RERANK_API_TIMEOUT,
        "max_candidates": int(os.getenv("RERANK_MAX_CANDIDATES", "50")),
        "max_document_chars": int(os.getenv("RERANK_MAX_DOCUMENT_CHARS", "1200")),
        "max_query_chars": int(os.getenv("RERANK_MAX_QUERY_CHARS", "1000")),
    }
    if resolve_rerank_provider(RERANK_API_PROVIDER, RERANK_API_URL) == "nvidia":
        return NvidiaApiReranker(**common, truncate=RERANK_TRUNCATE)
    return ApiReranker(**common)


# ---------------------------------------------------------------------------
# 5. 诊断信息
# ---------------------------------------------------------------------------
def describe_backends() -> dict:
    """返回当前后端配置摘要，便于启动日志与 ``/health`` 展示。"""

    embed_provider = resolve_embedding_provider(EMBED_API_PROVIDER, EMBED_BASE_URL) if EMBED_BACKEND == "api" else "local"
    rerank_backend = os.getenv("RERANK_BACKEND", "qwen3reranker").strip().lower()
    rerank_provider = (
        resolve_rerank_provider(RERANK_API_PROVIDER, RERANK_API_URL) if rerank_backend == "api" and RERANK_API_URL else None
    )
    return {
        "llm_backend": LLM_BACKEND,
        "llm_model": LLM_MODEL if LLM_BACKEND == "api" else OLLAMA_LLM_MODEL,
        "llm_base_url": LLM_BASE_URL if LLM_BACKEND == "api" else OLLAMA_BASE,
        "llm_configured": bool(LLM_BASE_URL and LLM_MODEL) if LLM_BACKEND == "api" else bool(OLLAMA_BASE),
        "embed_backend": EMBED_BACKEND,
        "embed_api_provider": embed_provider,
        "embed_query_input_type": "query" if embed_provider == "nvidia" else None,
        "embed_document_input_type": "passage" if embed_provider == "nvidia" else None,
        "embed_model": EMBED_MODEL if EMBED_BACKEND == "api" else OLLAMA_EMBED_MODEL,
        "embed_base_url": EMBED_BASE_URL if EMBED_BACKEND == "api" else OLLAMA_BASE,
        "embed_configured": bool(EMBED_BASE_URL and EMBED_MODEL) if EMBED_BACKEND == "api" else bool(OLLAMA_BASE),
        "embed_dim": EMBED_DIM or None,
        "rerank_backend": rerank_backend,
        "rerank_api_provider": rerank_provider,
        "rerank_model": RERANK_API_MODEL if rerank_backend == "api" else None,
        "rerank_api_url": RERANK_API_URL if rerank_backend == "api" else None,
    }
