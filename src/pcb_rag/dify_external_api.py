"""
PCB-RAG Dify 外部知识库 API
================================
将项目中所有的优化检索策略（HyDE、多路召回、BM25+、BGE/Qwen3-Reranker-4B Rerank 等）
包装为符合 Dify 外部知识库 API 规范的 FastAPI 服务。

Dify 外部知识库 API 规范：
    POST /retrieval
    Authorization: Bearer <API_TOKEN>
    Content-Type: application/json
    请求体 → {"knowledge_id": "...", "query": "...", "retrieval_setting": {...}}
    响应体 → {"records": [{"content": "...", "score": 0.0, "title": "...", "metadata": {...}}]}

启动方式：
    python -m pcb_rag.dify_external_api
    或 uvicorn pcb_rag.dify_external_api:app --host 0.0.0.0 --port 8000
"""

# ---------------------------------------------------------------------------
# 0. 最先设置代理和镜像环境变量（必须在任何 HF 相关 import 之前执行）
# ---------------------------------------------------------------------------
import os

# 先加载仓库根目录的 .env：直接 `uvicorn pcb_rag.dify_external_api:app` 或
# `python -m pcb_rag.dify_external_api` 时也能读到 LLM / Embedding / Rerank /
# Token 配置（否则只有走 scripts/*.sh 才会有）
from pcb_rag.usage import model_stream
from pcb_rag.env_loader import load_project_env

load_project_env()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# 1. 从 query.py 导入所有需要的组件
#    query.py 中 if __name__ == "__main__" 内部才执行 main()，import 无副作用。
# ---------------------------------------------------------------------------
from pcb_rag.query import (
    # ── 配置常量 ──────────────────────────────────────────────────────────
    RECALL_TOP_K,
    RERANK_TOP_N,
    HYDE_ENABLED,
    FUSION_NUM_QUERIES,
    CHUNK_EXPAND_ENABLED,
    MetadataMode,
    Settings,
    # ── 答案生成函数 ──────────────────────────────────────────────────────
    format_answer_with_citations,
    CITATION_ENABLED,
    extract_citations,
    SELF_RAG_ENABLED,
    self_rag_query,
    format_self_rag_result,
)

from pcb_rag.api_clients import describe_backends
from pcb_rag.cache import all_cache_stats, embed_query_for_cache, get_cache

# P2：上下文压缩（去重 + 抽取式压缩 + 预算裁剪）
from pcb_rag.compression import (
    describe_compression,
)

# P1-2：GraphRAG 图检索
from pcb_rag.graph_rag import (
    describe_graph,
)

# P2：可观测性（指标快照 / trace / span）
from pcb_rag.observability import (
    describe_observability,
    metrics_snapshot,
)

# P2：访问控制（多租户 / 角色）
from pcb_rag.security import (
    ACL_ENABLED,
    current_principal,
    describe_acl,
    validate_acl_security,
)

# ---------------------------------------------------------------------------
# 2. 配置日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pcb-rag-dify")

# ---------------------------------------------------------------------------
# 3. API 鉴权 Token（需与 Dify 后台「外部知识库」配置的 API Key 保持一致）
# ---------------------------------------------------------------------------
#: 占位默认值。这是公开字符串，等于没有鉴权，因此启动时必须拒绝（见 _assert_token_configured）。
_DEFAULT_API_TOKEN = "change-me"
API_TOKEN: str = os.getenv("DIFY_API_TOKEN", _DEFAULT_API_TOKEN).strip()


def _assert_token_configured() -> None:
    """启动自检：Token 未配置 / 仍是占位值时直接拒绝启动。

    "默认不安全 + 文档里提醒"不算安全设计：判据是「配置错误时是否 fail-safe」。
    这里把最常见的配置错误（忘了设 DIFY_API_TOKEN）变成一次响亮的启动失败，
    而不是一个静默的全开放服务。
    """
    if not API_TOKEN or API_TOKEN == _DEFAULT_API_TOKEN:
        raise RuntimeError(
            "DIFY_API_TOKEN 未配置（当前值：" + (API_TOKEN or "<empty>") + "）。"
            "默认占位值等于无鉴权，服务拒绝启动。请在 .env 或环境变量中设置一个强随机 Token，"
            "并与 Dify「外部知识库 API Key」保持一致。"
        )
    if len(API_TOKEN) < 16:
        logger.warning(
            "[Security] DIFY_API_TOKEN 长度 < 16，建议使用至少 16 位的随机串。"
        )

# ---------------------------------------------------------------------------
# 3b. 会话管理器（服务端对话记忆）
# ---------------------------------------------------------------------------

SESSION_EXPIRE_MINUTES = int(os.getenv("SESSION_EXPIRE_MINUTES", "60"))
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "300"))
# 会话持久化：memory（默认，进程重启后丢失）| sqlite（落盘，重启后保留）
SESSION_STORE = os.getenv("SESSION_STORE", "memory").strip().lower()
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "./data/sessions.db")


from pcb_rag.sessions import SessionManager, principal_from_token


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

# ---------------------------------------------------------------------------
# 4. 全局单例：在应用启动时一次性完成耗时初始化，服务运行期间复用
# ---------------------------------------------------------------------------
from pcb_rag.retrieval import get_pipeline
from pcb_rag.conversation import (
    rewrite_query_with_history as _rewrite_query_with_history,
    generate_answer_with_history as _generate_answer_with_history,
    prepare_with_history,
)

_app_state: dict = get_pipeline().state


def _initialize_retrieval_engine():
    get_pipeline().initialize()


def _maybe_reload_lexical_index():
    get_pipeline().refresh()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时完成初始化，关闭时释放资源。"""
    # 先做配置自检（fail-fast），再做昂贵的模型 / Milvus 初始化
    _assert_token_configured()
    _acl_problem = validate_acl_security()
    if _acl_problem:
        raise RuntimeError(_acl_problem)
    if not CORS_ALLOW_ORIGINS:
        logger.info("[CORS] 未配置 CORS_ALLOW_ORIGINS，不下发跨域头（同源 / 服务端调用不受影响）")

    _initialize_retrieval_engine()

    get_pipeline().refresh()

    _session_cleanup_task = asyncio.create_task(_session_cleanup_worker())

    logger.info(f"[SessionManager] 后台清理任务已启动 (间隔={SESSION_CLEANUP_INTERVAL}s)")

    yield

    _session_cleanup_task.cancel()
    try:
        await _session_cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("PCB-RAG Dify API 服务关闭。")


async def _session_cleanup_worker():
    """后台任务：定期清理过期会话。"""
    while True:
        try:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
            sm = get_session_manager()
            cleaned = sm.cleanup_expired(all_owners=True)
            if cleaned > 0:
                logger.info(f"[SessionCleanup] 自动清理了 {cleaned} 个过期会话")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[SessionCleanup] 清理任务异常: {e}")


# ---------------------------------------------------------------------------
# 5. FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PCB-RAG Dify External Knowledge API",
    description="为 Dify 提供外部知识库接入的 PCB 专业文档检索服务",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Web 界面
#
# 单文件静态页，不含任何配置或密钥：页面本身公开可访问是安全的，数据一律经
# 带鉴权的 API 取。用户首次提问若收到 401，页面会提示填入 Bearer Token。
# ---------------------------------------------------------------------------
_WEBUI_DIR = Path(__file__).resolve().parent / "webui"
_WEBUI_INDEX = _WEBUI_DIR / "index.html"


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def web_ui():
    """返回 Web 问答界面。"""
    if not _WEBUI_INDEX.is_file():
        raise HTTPException(
            status_code=404,
            detail="Web 界面文件缺失：src/pcb_rag/webui/index.html",
        )
    return FileResponse(_WEBUI_INDEX, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """浏览器默认会请求 favicon，返回 204 避免日志里出现无意义的 404。"""
    return Response(status_code=204)


@app.get("/manage", include_in_schema=False)
def management_ui():
    return FileResponse(_WEBUI_DIR / "manage.html", media_type="text/html")


# CORS 中间件
#
# 默认不开通配。威胁模型：本服务是 Bearer 认证、不使用 cookie，浏览器不会自动
# 携带 Authorization，因此 allow_origins=["*"] 不构成直接的凭证泄露；但它没有任何
# 理由常开，而且一旦将来改成 cookie 认证（或打开 allow_credentials），通配会立刻
# 变成严重漏洞。所以这里改成显式白名单，并显式关闭 credentials。
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,   # 空列表 = 不下发任何 CORS 头（浏览器默认阻止跨域）
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Trace-Id"],
)


from pcb_rag.service_runtime import RequestBoundaryMiddleware
app.add_middleware(RequestBoundaryMiddleware)


# ---------------------------------------------------------------------------
# 6. 数据模型（严格遵循 Dify 外部知识库 API 规范）
# ---------------------------------------------------------------------------
class RetrievalSetting(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100, description="返回的最大文档数")
    score_threshold: float = Field(default=0.0, description="最低得分阈值")
    score_threshold_enabled: bool = Field(default=False)


class RetrievalReq(BaseModel):
    knowledge_id: str = Field(description="知识库 ID（Dify 生成，接收即可）")
    query: str = Field(description="用户查询文本", max_length=8000)
    retrieval_setting: RetrievalSetting = Field(default_factory=RetrievalSetting)


class RecordMetadata(BaseModel):
    source_path: str = ""
    source_type: str = ""
    vendor: str = ""
    eda: str = ""


class Record(BaseModel):
    content: str
    score: float
    title: str = ""
    metadata: dict = {}


class RetrievalRes(BaseModel):
    records: List[Record]


# ---------------------------------------------------------------------------
# 7. 鉴权依赖
# ---------------------------------------------------------------------------
def verify_token(authorization: str = Header(None)):
    """验证 Bearer Token，与 Dify 后台配置保持一致。

    用 ``secrets.compare_digest`` 做常量时间比较。说明威胁模型：跨网络的计时攻击
    在实践中极难成立（抖动远大于单次比较的差异），这条属于「成本极低、顺手就该修」，
    不是高危漏洞。
    """
    expected = f"Bearer {API_TOKEN}"
    if principal_from_token(authorization) is not None:
        return
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API token.")


# ---------------------------------------------------------------------------
# 8. 核心检索函数（完整复用 query.py 的优化管道）
# ---------------------------------------------------------------------------
def _retrieve_nodes(user_query: str, top_k: int = 5, score_threshold: float = 0.0, _no_business_filter=False):
    return get_pipeline().retrieve_nodes(user_query, top_k, score_threshold)




def _cache_scope_for_principal() -> str:
    """语义缓存的隔离域。

    缓存里存的是**已经过 ACL 过滤**的检索结果，必须按访问主体隔离：
    - ACL 关闭：所有请求共享一个域（原行为）
    - 管理员：单独一个域（能看到全部数据）
    - 普通用户：租户 + user_id + 角色/组指纹。带 user_id 是因为 owner 级放行
      会让同租户、同角色的两个人看到不同的私有文档。
    """
    if not ACL_ENABLED:
        return ""
    p = current_principal()
    if p.is_admin:
        return "admin"
    return f"{p.tenant_id.lower()}|{p.user_id.lower()}|{p.fingerprint()}"


def retrieve_chunks(user_query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Record]:
    """
    完整执行 PCB-RAG 检索管道，返回适配 Dify 的 Record 列表。
    内部调用 _retrieve_nodes 获取 NodeWithScore，再转换为 Record。
    带语义缓存：相同或语义相近的查询直接复用结果（按访问主体隔离，避免跨租户泄漏）。
    """
    cache = get_cache("retrieval")
    pipeline = get_pipeline()
    pipeline.refresh()
    revision = pipeline.state.get("revision", "legacy")
    config_key = pipeline.config.fingerprint
    cache_scope = f"{_cache_scope_for_principal()}|revision={revision}|pipeline={config_key}|top_k={top_k}|thr={score_threshold}"
    cache_key = f"{user_query}|top_k={top_k}|thr={score_threshold}"
    query_vec = embed_query_for_cache(user_query)

    cached = cache.get(cache_key, query_vec, scope=cache_scope)
    if isinstance(cached, dict) and cached.get("params") == [top_k, score_threshold] and cached.get("revision") == revision:
        logger.info(f"[Cache] 检索命中缓存（query={user_query[:40]}...）")
        return [Record(**item) for item in cached.get("records", [])]

    # GraphRAG 融合 / 邻域扩展与 ACL 兜底复核已下沉到 _retrieve_nodes 末尾，
    # 所有端点（/retrieval、/api/ask、/api/chat、/api/ask/stream）共享同一条链路，
    # 不会再出现"某个端点少了兜底或少了图增强"的分叉。
    from pcb_rag.query import build_context_bundle
    nodes = build_context_bundle(_retrieve_nodes(user_query, top_k=top_k, score_threshold=score_threshold)).nodes

    records: List[Record] = []
    for nws in nodes:
        score = float(nws.score) if nws.score is not None else 0.0
        node = nws.node
        content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        meta: dict = node.metadata if hasattr(node, "metadata") else {}
        source_path: str = meta.get("source_path", "")
        source_name: str = Path(source_path).stem if source_path else meta.get("source_type", "未知来源")
        records.append(Record(
            content=content,
            score=round(score, 6),
            title=source_name,
            metadata={
                "chunk_id": str(node.id_), "revision": meta.get("revision", ""),
                "section": meta.get("section") or meta.get("parent_title", ""),
                "page": str(meta.get("page_label") or meta.get("page", "")),
                "source_path": source_path,
                "source_type": meta.get("source_type", ""),
                "vendor": meta.get("vendor", ""),
                "eda": meta.get("eda", ""),
            },
        ))
    if pipeline.state.get("revision", "legacy") != revision:
        return records
    cache.set(
        cache_key,
        {"params": [top_k, score_threshold], "revision": revision, "pipeline": config_key, "records": [r.model_dump() for r in records]},
        query_vec,
        scope=cache_scope,
    )
    logger.info(f"[Result] 返回 {len(records)} 条记录（query={user_query[:40]}...）")
    return records


# ---------------------------------------------------------------------------
# 9. Dify 外部知识库接口：POST /retrieval
# ---------------------------------------------------------------------------
@app.post("/retrieval", response_model=RetrievalRes, summary="Dify 外部知识库检索接口")
def external_retrieval(
    req: RetrievalReq,
    _: str = Depends(verify_token),
):
    """
    符合 Dify 外部知识库 API 规范的检索端点（同步执行，不会阻塞事件循环）。

    Dify 将在每次用户提问时调用此端点，传入 query，
    返回的 records 将作为上下文注入 LLM 的 Prompt 中。
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    setting = req.retrieval_setting
    top_k           = max(1, min(setting.top_k, 20))   # 限制最多 20 条，防止过长
    score_threshold = setting.score_threshold if setting.score_threshold_enabled else 0.0

    try:
        records = retrieve_chunks(
            user_query=req.query.strip(),
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.error(f"[Error] 检索异常: {e}", exc_info=True)
        raise _api_error(e) from e

    return RetrievalRes(records=records)


# ---------------------------------------------------------------------------
# 10. /api/ask：全链路问答端点（检索 + LLM 合成答案）
#     供 Dify 工作流 HTTP Request 节点调用
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(default="user", description="角色：user 或 assistant")
    content: str = Field(default="", description="消息内容")

    class Config:
        extra = "ignore"  # 忽略 Dify 可能传入的额外字段（如 files, tool_calls 等）


class AskReq(BaseModel):
    query: str = Field(description="用户问题", max_length=8000)
    top_k: int = Field(default=5, ge=1, le=20, description="检索 chunk 数量")
    score_threshold: float = Field(default=0.0, ge=0.0, description="最低得分阈值")
    score_threshold_enabled: bool = Field(default=False)
    use_self_rag: bool = Field(default=False, description="是否启用 Self-RAG 自反思迭代")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="对话历史（按时间顺序排列的 user/assistant 消息列表），"
                    "用于多轮对话中解析指代和上下文。"
                    "Dify 工作流中可通过 {{#sys.conversation_history#}} 传入。"
    )


class AnswerDiagnostics(BaseModel):
    citations: List[dict] = Field(default_factory=list)
    invalid_citations: List[int] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)
    revision: str = "legacy"
    routes: dict = Field(default_factory=dict)
    timings: dict = Field(default_factory=dict)
    degraded: List[str] = Field(default_factory=list)
    usage: List[dict] = Field(default_factory=list)


def _pipeline_llm():
    return get_pipeline().llm or Settings.llm


def _api_error(error):
    from pcb_rag.runtime import DeadlineExceeded, WorkRejected
    from pcb_rag.sessions import SessionAccessDenied
    from pcb_rag.usage import UsageBudgetExceeded
    from pcb_rag.rerank_control import RerankStopped
    if isinstance(error, (DeadlineExceeded, SessionAccessDenied)):
        raise error
    if isinstance(error, HTTPException):
        return error
    if isinstance(error, (WorkRejected, UsageBudgetExceeded)):
        return HTTPException(status_code=429, detail=str(error))
    if isinstance(error, RerankStopped):
        return HTTPException(status_code=503, detail="重排服务已停止，请检查用量与服务状态")
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail="问答处理失败，请使用 trace ID 检查服务日志")


def _answer_request(raw_query, history, top_k, threshold, use_self_rag):
    from pcb_rag.runtime import current_control
    llm = _pipeline_llm()
    try:
        search_query = _rewrite_query_with_history(raw_query, history, llm) if history else raw_query
        if use_self_rag and SELF_RAG_ENABLED:
            class Retriever:
                def retrieve(self, value):
                    return _retrieve_nodes(search_query if value == raw_query else value, top_k, threshold)
            result = self_rag_query(raw_query, Retriever(), llm=llm)
            answer = format_self_rag_result(result, show_history=False)
        else:
            nodes = _retrieve_nodes(search_query, top_k, threshold)
            result = _generate_answer_with_history(raw_query, nodes, history, llm)
            answer = format_answer_with_citations(result, verbose=False)
        bundle = result["evidence_bundle"]
        control = current_control()
        return {"answer": answer, "sources": [entry["source_path"] for entry in bundle.sources],
            "records": [Record(content=entry["text"], score=entry["score"], title=entry["source_name"],
                metadata={"citation_id": entry["index"], "chunk_id": entry["chunk_id"],
                          "source_path": entry["source_path"], "section": entry["section"],
                          "revision": entry["metadata"].get("revision", ""),
                          "page": str(entry["metadata"].get("page_label", ""))}) for entry in bundle.entries],
            "rewritten_query": search_query if search_query != raw_query else None,
            "citations": bundle.citations(result["answer"]), "invalid_citations": bundle.invalid_citations(result["answer"]),
            "budget": bundle.budget, "revision": control.revision if control else get_pipeline().state.get("revision", "legacy"),
            "routes": dict(control.routes) if control else {}, "timings": dict(control.timings) if control else {},
            "degraded": list(dict.fromkeys(control.degraded)) if control else [], "usage": list(control.usage) if control else []}
    except Exception as error:
        logger.exception("answer request failed")
        raise _api_error(error) from error


class AskRes(AnswerDiagnostics):
    answer: str
    sources: List[str] = []
    records: List[Record] = []
    rewritten_query: Optional[str] = Field(
        default=None,
        description="经历史改写后的独立查询（仅在有历史改写时返回，便于调试）"
    )


@app.post("/api/ask", response_model=AskRes, summary="全链路问答接口（检索 + LLM 合成，支持多轮对话）")
def ask_rag(req: AskReq, _: str = Depends(verify_token)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    return AskRes(**_answer_request(req.query.strip(), [m.model_dump() for m in req.chat_history or []],
        req.top_k, req.score_threshold if req.score_threshold_enabled else 0.0, req.use_self_rag))


# ---------------------------------------------------------------------------
# 10a. /api/chat：带会话记忆的问答端点（推荐使用）
#      自动管理对话历史，客户端只需传递 session_id
# ---------------------------------------------------------------------------
class ChatReq(BaseModel):
    query: str = Field(description="用户问题", max_length=8000)
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID（首次对话不传，后续对话传入返回的 session_id）"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="检索 chunk 数量")
    score_threshold: float = Field(default=0.0, ge=0.0, description="最低得分阈值")
    score_threshold_enabled: bool = Field(default=False)
    use_self_rag: bool = Field(default=False, description="是否启用 Self-RAG 自反思迭代")


class ChatRes(AnswerDiagnostics):
    answer: str
    session_id: str = Field(description="会话ID（后续对话需传入此ID）")
    is_new_session: bool = Field(description="是否为新创建的会话")
    sources: List[str] = []
    records: List[Record] = []
    rewritten_query: Optional[str] = Field(
        default=None,
        description="经历史改写后的独立查询（仅在有历史改写时返回，便于调试）"
    )


@app.post("/api/chat", response_model=ChatRes, summary="带会话记忆的问答接口（推荐）")
def chat_rag(req: ChatReq, _: str = Depends(verify_token)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    sm = get_session_manager()
    session_id, is_new = sm.get_or_create(req.session_id)
    payload = _answer_request(req.query.strip(), sm.get_history(session_id), req.top_k,
        req.score_threshold if req.score_threshold_enabled else 0.0, req.use_self_rag)
    sm.add_message(session_id, "user", req.query.strip())
    sm.add_message(session_id, "assistant", payload["answer"])
    return ChatRes(**payload, session_id=session_id, is_new_session=is_new)


# ---------------------------------------------------------------------------
# 10b. /api/chat/plain：带会话记忆的纯文本问答端点
# ---------------------------------------------------------------------------
@app.post("/api/chat/plain", response_class=Response, summary="带会话记忆的纯文本问答接口")
def chat_rag_plain(req: ChatReq, _: str = Depends(verify_token)):
    result = chat_rag(req, _)
    return Response(content=result.answer, media_type="text/plain; charset=utf-8",
        headers={"X-Session-ID": result.session_id, "X-Is-New-Session": str(result.is_new_session).lower()})


# ---------------------------------------------------------------------------
# 10c. /api/ask/plain：纯文本问答端点（Dify 工作流首选）
#      直接返回 text/plain，{{http_request.body}} 即为答案，无需 JSON 解析
# ---------------------------------------------------------------------------
@app.post("/api/ask/plain", response_class=Response, summary="纯文本问答接口（Dify 工作流推荐，支持多轮对话）")
def ask_rag_plain(req: AskReq, _: str = Depends(verify_token)):
    return Response(content=ask_rag(req, _).answer, media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# 10d. /api/ask/stream：SSE 流式问答端点
#      先推送检索来源，再逐 token 推送答案，适合需要流式体验的前端
# ---------------------------------------------------------------------------
@app.post("/api/ask/stream", summary="流式问答接口（SSE）")
def ask_rag_stream(
    req: AskReq,
    _: str = Depends(verify_token),
):
    """SSE 流式问答。

    事件类型：

    - ``sources``  ：检索完成后立即推送，包含来源列表与改写后的查询
    - ``token``    ：逐段推送答案文本
    - ``citations``：答案生成完毕后的引用溯源
    - ``error``    ：生成过程中出错
    - ``done``     ：流结束
    """
    import json as _json

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    threshold = req.score_threshold if req.score_threshold_enabled else 0.0
    raw_query = req.query.strip()
    history_dicts = [m.model_dump() for m in req.chat_history] if req.chat_history else []

    def _sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {_json.dumps(payload, ensure_ascii=False)}\n\n"

    def _generate():
        try:
            # 1) 历史感知查询改写
            search_query = raw_query
            if history_dicts:
                search_query = _rewrite_query_with_history(raw_query, history_dicts, _pipeline_llm())

            # 2) 检索
            nodes = _retrieve_nodes(
                user_query=search_query,
                top_k=req.top_k,
                score_threshold=threshold,
            )

            prepared = prepare_with_history(raw_query, nodes, history_dicts, _pipeline_llm())
            bundle = prepared.evidence
            sources_info = bundle.sources

            yield _sse(
                "sources",
                {
                    "sources": [info.get("source_path", "") for info in sources_info],
                    # snippet / score / tokens 是给 Web UI 用的：只给文件名的话，
                    # 用户无法判断"这条证据到底说了什么、有多可信"，只能盲信答案。
                    "citations": [
                        {
                            "index": info.get("index"),
                            "source_name": info.get("source_name", ""),
                            "section": info.get("section", ""),
                            "chunk_id": info.get("chunk_id", ""),
                            "snippet": info.get("snippet", ""),
                            "score": info.get("score", 0.0),
                            "tokens": info.get("tokens", 0),
                            "truncated": info.get("truncated", False),
                        }
                        for info in sources_info
                    ],
                    "rewritten_query": search_query if search_query != raw_query else None,
                },
            )

            prompt = prepared.prompt

            if not bundle.entries:
                yield _sse("token", {"text": "根据提供的文档无法回答该问题，未找到可用证据。"})
                yield _sse("done", {"answer_length": 0, "budget": bundle.budget})
                return
            # 4) 流式生成
            answer_parts: List[str] = []
            try:
                for chunk in model_stream(_pipeline_llm(), prompt):
                    delta = getattr(chunk, "delta", None)
                    if delta is None:
                        delta = getattr(chunk, "text", "") or ""
                    if delta:
                        answer_parts.append(delta)
                        yield _sse("token", {"text": delta})
            except Exception as e:
                logger.error(f"[Error] 流式生成失败: {e}", exc_info=True)
                yield _sse("error", {"message": f"生成失败: {e}"})

            answer = "".join(answer_parts)
            if CITATION_ENABLED and answer:
                try:
                    yield _sse("citations", {"citations": extract_citations(answer, bundle), "invalid_citations": bundle.invalid_citations(answer)})
                except Exception:
                    pass

            from pcb_rag.runtime import current_control
            control = current_control()
            yield _sse("done", {"answer_length": len(answer), "budget": bundle.budget,
                                "degraded": list(dict.fromkeys(control.degraded)) if control else [],
                                "routes": control.routes if control else {}, "timings": control.timings if control else {},
                                "usage": control.usage if control else []})
        except Exception as e:
            logger.error(f"[Error] /api/ask/stream 异常: {e}", exc_info=True)
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 11. 会话管理 API 端点
# ---------------------------------------------------------------------------
class SessionCreateRes(BaseModel):
    session_id: str
    message: str = "会话创建成功"


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: str
    updated_at: str


class SessionHistoryRes(BaseModel):
    session_id: str
    history: List[dict]
    message_count: int


class SessionListRes(BaseModel):
    sessions: List[SessionInfo]
    total_count: int


@app.post("/api/session/create", response_model=SessionCreateRes, summary="创建新会话")
def create_session(_: str = Depends(verify_token)):
    """
    创建新的对话会话，返回 session_id。
    后续对话时将此 session_id 传入 /api/chat 即可保持上下文。
    """
    sm = get_session_manager()
    session_id = sm.create_session()
    return SessionCreateRes(session_id=session_id)


@app.get("/api/session/{session_id}", response_model=SessionHistoryRes, summary="获取会话历史")
def get_session_history(session_id: str, _: str = Depends(verify_token)):
    """
    获取指定会话的完整对话历史。
    """
    sm = get_session_manager()
    if not sm.exists(session_id):
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    history = sm.get_history(session_id)
    return SessionHistoryRes(
        session_id=session_id,
        history=history,
        message_count=len(history),
    )


@app.delete("/api/session/{session_id}", summary="删除会话")
def delete_session(session_id: str, _: str = Depends(verify_token)):
    """
    删除指定会话及其历史记录。
    """
    sm = get_session_manager()
    if sm.delete_session(session_id):
        return {"message": f"会话 {session_id} 已删除", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")


@app.post("/api/session/{session_id}/clear", summary="清空会话历史")
def clear_session_history(session_id: str, _: str = Depends(verify_token)):
    """
    清空指定会话的历史记录，但保留会话本身。
    """
    sm = get_session_manager()
    if sm.clear_session(session_id):
        return {"message": f"会话 {session_id} 历史已清空", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")


@app.get("/api/sessions", response_model=SessionListRes, summary="列出所有会话")
def list_sessions(_: str = Depends(verify_token)):
    """
    列出所有活跃会话的摘要信息。
    """
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return SessionListRes(
        sessions=[SessionInfo(**s) for s in sessions],
        total_count=len(sessions),
    )


@app.post("/api/sessions/cleanup", summary="清理过期会话")
def cleanup_sessions(_: str = Depends(verify_token)):
    """
    手动触发过期会话清理。
    返回清理的会话数量。
    """
    sm = get_session_manager()
    cleaned = sm.cleanup_expired()
    return {
        "message": f"已清理 {cleaned} 个过期会话",
        "cleaned_count": cleaned,
    }


def verify_admin(_: str = Depends(verify_token)):
    from pcb_rag import security
    identity_required = security.ACL_ENABLED or os.getenv("AUTH_TOKEN_PRINCIPALS_FILE") or os.getenv("SESSION_REQUIRE_IDENTITY") == "1"
    if identity_required and not security.current_principal().is_admin:
        raise HTTPException(status_code=403, detail="administrator identity required")


@app.get("/api/admin/status", summary="发布版本与处理状态")
def ingestion_status(_: str = Depends(verify_admin)):
    from pcb_rag.management import JOBS, processing_status
    return dict(processing_status(get_pipeline().revisions), job=JOBS.status())


@app.get("/api/admin/documents", summary="文档处理阶段")
def ingestion_documents(offset: int = 0, limit: int = 100, _: str = Depends(verify_admin)):
    from pcb_rag.management import document_status
    if offset < 0 or not 1 <= limit <= 200:
        raise HTTPException(status_code=400, detail="invalid pagination")
    return document_status(get_pipeline().revisions, offset=offset, limit=limit)


@app.post("/api/admin/retry", summary="重试未完成入库")
def retry_ingestion(_: str = Depends(verify_admin)):
    from pcb_rag.management import JOBS
    try:
        return JOBS.retry(get_pipeline().revisions)
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/graph/neighbors", summary="按原文权限查看图谱关系")
def graph_neighbors(entity: str, _: str = Depends(verify_token)):
    from pcb_rag.engine import KnowledgeBase
    if not entity.strip() or len(entity) > 200:
        raise HTTPException(status_code=400, detail="invalid entity")
    pipeline = get_pipeline()
    pipeline.refresh()
    return KnowledgeBase(auto_init=False, pipeline=pipeline).graph_neighbors(entity).to_dict()


# ---------------------------------------------------------------------------
# 12. 健康检查端点（供 Dify 或 AD 监控服务状态）
# ---------------------------------------------------------------------------
@app.get("/health", summary="健康检查（LB / 探针用，不含敏感配置）")
def health_check():
    """轻量健康检查：只回答「进程活着、依赖是否就绪」。

    早先这里返回 describe_acl() / describe_backends() / cache 统计等明细，而这些
    端点没有鉴权 —— 租户字段名、admin 角色名、匿名租户名、后端 URL、缓存命中数
    都会对匿名调用者暴露。现在拆成两个端点：

    - ``/health``：只给 LB / k8s 探针用，无敏感信息、无需鉴权
    - ``/health/detail``：完整配置摘要，需要 Bearer Token
    """
    return {
        "status": "ok",
        "index_ready": "index" in _app_state,
        "bm25_ready": _app_state.get("bm25_index") is not None,
        "rerank_ready": _app_state.get("rerank") is not None,
    }


@app.get("/health/detail", summary="健康检查明细（需鉴权）")
def health_detail(_: str = Depends(verify_token)):
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return {
        "status": "ok",
        "index_ready": "index" in _app_state,
        "bm25_ready": _app_state.get("bm25_index") is not None,
        "rerank_ready": _app_state.get("rerank") is not None,
        "recall_k": _app_state.get("recall_k"),
        "session_manager": {
            "active_sessions": len(sessions),
            "max_sessions": SESSION_MAX_COUNT,
            "expire_minutes": SESSION_EXPIRE_MINUTES,
        },
        "perf_params": {
            "RECALL_TOP_K": RECALL_TOP_K,
            "RERANK_TOP_N": RERANK_TOP_N,
            "FUSION_NUM_QUERIES": FUSION_NUM_QUERIES,
            "HYDE_ENABLED": HYDE_ENABLED,
            "CHUNK_EXPAND_ENABLED": CHUNK_EXPAND_ENABLED,
        },
        "backends": describe_backends(),
        "cache": all_cache_stats(),
        "graph_rag": describe_graph(_app_state.get("graph")),
        "acl": describe_acl(),
        "compression": describe_compression(),
        "observability": describe_observability(),
    }


@app.get("/metrics", summary="运行指标（计数器 + 延迟分位）")
def metrics():
    """返回进程内累计的运行指标。

    口径说明：
      - ``counters``：累计次数（缓存命中、图检索补充、压缩丢弃、ACL 拦截等）
      - ``histograms``：滑动窗口内的 count / mean / p50 / p95 / p99

    可直接接入 Prometheus 文本导出器，也便于压测时即时观测。
    """

    return metrics_snapshot()


# ---------------------------------------------------------------------------
# 12. 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "pcb_rag.dify_external_api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info",
        reload=False,  # 生产模式：禁用 reload（避免 GPU 模型被重复加载）
    )
