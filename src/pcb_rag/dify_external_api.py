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
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------------------
# 0b. API 模式性能参数覆盖（在 import query 之前执行，setdefault 不覆盖已有值）
#     大幅减少召回候选数、精排数量、查询变体，缩短单次请求耗时
# ---------------------------------------------------------------------------
os.environ.setdefault("OLLAMA_LLM_MODEL", "qwen3.5:35b-a3b-q4_K_M")  # 强制使用指定 LLM
os.environ.setdefault("RECALL_TOP_K", "200")
os.environ.setdefault("RERANK_TOP_N", "10")
os.environ.setdefault("FUSION_NUM_QUERIES", "3")
os.environ.setdefault("CHUNK_EXPAND_ENABLED", "1")
os.environ.setdefault("RAG_DOC_MAX_CHARS", "1000")
os.environ.setdefault("RAG_TOP_DOCS", "5")
os.environ.setdefault("HYDE_MAX_LENGTH", "200")
os.environ.setdefault("HF_RERANK_MAX_LENGTH", "1024")
os.environ.setdefault("OLLAMA_TIMEOUT", "180")         # qwen3.5:35b-a3b-q4_K_M 生成长答案可能需要 60-150s

import sys
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ---------------------------------------------------------------------------
# 1. 从 query.py 导入所有需要的组件
#    query.py 中 if __name__ == "__main__" 内部才执行 main()，import 无副作用。
# ---------------------------------------------------------------------------
from pcb_rag.query import (
    # ── 配置常量 ──────────────────────────────────────────────────────────
    OLLAMA_BASE,
    RECALL_TOP_K,
    RERANK_TOP_N,
    RERANK_ENABLED,
    HYDE_ENABLED,
    QUERY2DOC_ENABLED,
    QUERY_ENHANCE_ALL,
    SHORT_QUERY_THRESHOLD,
    MULTI_EXPAND_ENABLED,
    FUSION_NUM_QUERIES,
    MULTIPATH_ENABLED,
    COLBERT_RERANK_ENABLED,
    MULTIPATH_DENSE_TOP_K,
    MULTIPATH_SPARSE_TOP_K,
    MULTIPATH_RERANK_CANDIDATES,
    MULTIPATH_RRF_K,
    CHUNK_EXPAND_ENABLED,
    CHUNK_EXPAND_NEIGHBORS,
    CHUNK_EXPAND_PARENT,
    CHUNK_GROUP_BY_DOC,
    FUSION_WEIGHTS,
    FUSION_RRF_K,
    HYDE_ROUTE_WEIGHT,
    # ── 查询路由（动态权重） ──────────────────────────────────────────────
    QUERY_ROUTING_ENABLED,
    _classify_query,
    _get_query_strategy,
    # ── 工具函数 ──────────────────────────────────────────────────────────
    build_index,
    _try_build_reranker,
    _load_or_build_bm25,
    _try_build_colbert_reranker,
    _expand_query,
    _extract_query_filters,
    _is_short_query,
    _build_multi_expand_queries,
    _pick_llm_candidates,
    _configure_llm,
    DEFAULT_NUM_CTX,
    DEFAULT_TIMEOUT,
    # ── 查询增强函数 ──────────────────────────────────────────────────────
    hyde_expand_query,
    query2doc_expand,
    # ── RAG Prompt / 上下文构建 ──────────────────────────────────────────
    RAG_PROMPT_TEMPLATE,
    build_context_with_sources,
    # ── Retriever 类 ──────────────────────────────────────────────────────
    LocalBM25Retriever,
    MultiPathRetriever,
    ThreeWayHyDEFusionRetriever,
    expand_chunks_with_context,
    # ── 内部工具（并行 Fusion 检索需要）─────────────────────────────────
    _node_id_key,
    _weighted_rrf_fuse_three_routes,
    # ── LlamaIndex 核心类 ─────────────────────────────────────────────────
    QueryBundle,
    MetadataMode,
    NodeWithScore,
    Settings,
    # ── 答案生成函数 ──────────────────────────────────────────────────────
    generate_answer_with_citation,
    format_answer_with_citations,
    SELF_RAG_ENABLED,
    self_rag_query,
    format_self_rag_result,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama

# ---------------------------------------------------------------------------
# 1c. 持久 HyDE 线程池 + 第二路变体函数
#     - _HYDE_EXECUTOR 独立于主检索 pool，不会被 with 语句 shutdown 阻塞
#     - _hyde_expand_v2 从规范/标准引用角度生成假设文档，与 HyDE 路1 互补
# ---------------------------------------------------------------------------
_HYDE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hyde")

_HYDE_PROMPT_V2 = """你是 PCB 电路板领域的技术专家。请根据问题生成"规范引用型"检索片段。

内容应包括：相关国标/行标条款、测试方法名称、关键技术指标名称。
长度控制在 80-160 字。严禁编造具体数值或标准号。

问题：{query}

输出："""


def _hyde_expand_v2(query: str, llm=None) -> str:
    """HyDE 第二路变体：以规范/标准引用角度生成假设文档，增加检索多样性。"""
    import re
    try:
        if llm is None:
            llm = Settings.llm
        prompt = _HYDE_PROMPT_V2.format(query=query)
        response = llm.complete(prompt)
        text = (response.text if hasattr(response, "text") else str(response)).strip()
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        if not text or len(text) < 20:
            return query
        max_len = int(os.getenv("HYDE_MAX_LENGTH", "200"))
        if len(text) > max_len:
            text = text[:max_len] + "..."
        return f"{query}\n\n{text}"
    except Exception:
        return query


# ---------------------------------------------------------------------------
# 1b. 会话历史感知查询改写 Prompt
# ---------------------------------------------------------------------------
_HISTORY_REWRITE_PROMPT = """你是 PCB 领域的智能助手。用户的最新问题可能包含代词或省略，需要结合对话历史改写为一个独立、完整的检索查询。

规则：
1. 如果最新问题本身已经完整（不含代词/省略），直接原样输出。
2. 如果最新问题包含"它""这个""那个""上述"等指代词，或省略了主语，用对话历史中的具体实体替换。
3. 只输出改写后的查询，不要解释。
4. 保留原始问题的疑问意图。

对话历史（最近 {n} 轮）：
{history}

最新问题：{query}

改写后的查询："""

# 带历史的 RAG Prompt 模板
_RAG_PROMPT_WITH_HISTORY = """你是 PCB 电路板领域的技术专家。请根据以下参考文档和对话历史回答用户问题。

要求：
1. 只基于参考文档中的信息回答，不要编造
2. 如果文档中没有相关信息，明确说明"根据提供的文档无法回答"
3. 回答要准确、专业、简洁
4. 在回答末尾标注信息来源（如：[来源：GBT4588.4-2017]）
5. 如果用户的问题是对前一轮的追问，请结合对话历史进行回答

对话历史（最近 {n} 轮）：
{history}

参考文档：
{context}

用户问题：{query}

回答："""

# 最大保留历史轮数
_MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))


def _rewrite_query_with_history(
    query: str,
    chat_history: list,
    llm=None,
) -> str:
    """利用 LLM 将含有指代/省略的追问改写为完整的独立查询。

    如果历史为空或查询本身足够完整（>30字且无代词），直接返回原查询。
    """
    if not chat_history:
        return query

    # 快速判断：如果查询本身很长且没有明显代词，跳过改写
    _ANAPHORA_PATTERNS = ["它", "这个", "那个", "上述", "前面", "刚才", "上面",
                          "该", "其", "此", "这些", "那些", "还有呢", "继续",
                          "接着", "然后呢", "补充", "详细"]
    has_anaphora = any(p in query for p in _ANAPHORA_PATTERNS)
    if len(query) > 40 and not has_anaphora:
        return query

    if llm is None:
        llm = Settings.llm

    # 构建历史文本（最近 N 轮）
    recent = chat_history[-_MAX_HISTORY_TURNS:]
    history_lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_lines.append(f"用户：{content}")
        elif role == "assistant":
            # 截断过长的助手回复
            truncated = content[:200] + "..." if len(content) > 200 else content
            history_lines.append(f"助手：{truncated}")
    history_text = "\n".join(history_lines)

    prompt = _HISTORY_REWRITE_PROMPT.format(
        n=len(recent), history=history_text, query=query
    )

    try:
        response = llm.complete(prompt)
        rewritten = (response.text if hasattr(response, "text") else str(response)).strip()
        # 基本合理性检查
        if rewritten and 3 < len(rewritten) < 500:
            logger.info(f"[HistoryRewrite] '{query}' → '{rewritten[:80]}'")
            return rewritten
    except Exception as e:
        logger.warning(f"[HistoryRewrite] 改写失败，使用原始查询: {e}")

    return query


def _generate_answer_with_history(
    query: str,
    retrieved_docs: list,
    chat_history: list,
    llm=None,
) -> dict:
    """在 generate_answer_with_citation 基础上注入对话历史，使 LLM 能理解上下文。"""
    if not chat_history:
        # 无历史时回退到标准函数
        return generate_answer_with_citation(query, retrieved_docs, llm)

    if llm is None:
        llm = Settings.llm

    if not retrieved_docs:
        return {
            'answer': '根据提供的文档无法回答该问题，未找到相关文档。',
            'citations': [],
            'sources': [],
            'context_info': {'doc_count': 0},
        }

    # 构建 context
    context, sources_info = build_context_with_sources(retrieved_docs)

    # 构建历史文本
    recent = chat_history[-_MAX_HISTORY_TURNS:]
    history_lines = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_lines.append(f"用户：{content}")
        elif role == "assistant":
            truncated = content[:300] + "..." if len(content) > 300 else content
            history_lines.append(f"助手：{truncated}")
    history_text = "\n".join(history_lines)

    prompt = _RAG_PROMPT_WITH_HISTORY.format(
        n=len(recent), history=history_text, context=context, query=query
    )

    try:
        response = llm.complete(prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        logger.error(f"[RAG] 带历史答案生成失败: {e}")
        answer = f"答案生成过程中出现错误：{e}"

    from pcb_rag.query import CITATION_ENABLED, extract_citations
    citations = []
    if CITATION_ENABLED:
        citations = extract_citations(answer, retrieved_docs)

    sources = [info['source_path'] for info in sources_info]

    return {
        'answer': answer,
        'citations': citations,
        'sources': sources,
        'context_info': {
            'doc_count': len(sources_info),
            'total_chars': sum(info['char_count'] for info in sources_info),
            'sources_detail': sources_info,
        },
    }


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
API_TOKEN: str = os.getenv("DIFY_API_TOKEN", "change-me")

# ---------------------------------------------------------------------------
# 3b. 会话管理器（服务端对话记忆）
# ---------------------------------------------------------------------------
import uuid
import threading
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, List, Tuple

SESSION_EXPIRE_MINUTES = int(os.getenv("SESSION_EXPIRE_MINUTES", "60"))
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "300"))


class SessionManager:
    """
    服务端会话管理器：维护多轮对话历史，支持按 session_id 隔离。

    特性：
    - LRU 淘汰：超过 SESSION_MAX_COUNT 时淘汰最久未访问的会话
    - 过期清理：SESSION_EXPIRE_MINUTES 后自动清理
    - 线程安全：使用 threading.Lock 保护共享状态
    """

    def __init__(
        self,
        max_sessions: int = SESSION_MAX_COUNT,
        expire_minutes: int = SESSION_EXPIRE_MINUTES,
    ):
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._max_sessions = max_sessions
        self._expire_delta = timedelta(minutes=expire_minutes)
        self._lock = threading.Lock()
        self._last_cleanup = datetime.now()
        logger.info(f"[SessionManager] 初始化完成 (max={max_sessions}, expire={expire_minutes}min)")

    def create_session(self, session_id: str = None) -> str:
        """创建新会话，返回 session_id。"""
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = {
                    "history": [],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                self._sessions.move_to_end(sid)
                self._evict_if_needed()
        logger.debug(f"[SessionManager] 创建会话: {sid[:8]}...")
        return sid

    def get_or_create(self, session_id: str = None) -> Tuple[str, bool]:
        """获取或创建会话，返回 (session_id, is_new)。"""
        if session_id:
            with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id]["updated_at"] = datetime.now()
                    self._sessions.move_to_end(session_id)
                    return session_id, False
        new_sid = self.create_session(session_id)
        return new_sid, True

    def get_history(self, session_id: str) -> List[dict]:
        """获取会话历史（按时间顺序）。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["updated_at"] = datetime.now()
                self._sessions.move_to_end(session_id)
                return list(session["history"])
        return []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """向会话添加一条消息。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["history"].append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                })
                session["updated_at"] = datetime.now()
                self._sessions.move_to_end(session_id)
                logger.debug(f"[SessionManager] 会话 {session_id[:8]}... 添加消息 ({role})")

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史（保留会话）。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["history"] = []
                session["updated_at"] = datetime.now()
                return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话。"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"[SessionManager] 删除会话: {session_id[:8]}...")
                return True
        return False

    def list_sessions(self) -> List[dict]:
        """列出所有会话摘要。"""
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "message_count": len(s["history"]),
                    "created_at": s["created_at"].isoformat(),
                    "updated_at": s["updated_at"].isoformat(),
                }
                for sid, s in self._sessions.items()
            ]

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量。"""
        now = datetime.now()
        expired = []
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session["updated_at"] > self._expire_delta:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info(f"[SessionManager] 清理过期会话: {len(expired)} 个")
        return len(expired)

    def _evict_if_needed(self) -> None:
        """LRU 淘汰最久未访问的会话。"""
        while len(self._sessions) > self._max_sessions:
            oldest_sid = next(iter(self._sessions))
            del self._sessions[oldest_sid]
            logger.info(f"[SessionManager] LRU 淘汰会话: {oldest_sid[:8]}...")


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
_app_state: dict = {}


def _initialize_retrieval_engine() -> None:
    """
    在 FastAPI 启动阶段一次性初始化：
      - LLM + Embedding 配置
      - Milvus VectorStoreIndex
      - 本地 BM25 索引
      - Reranker（HF / Qwen3-Reranker-4B）
      - ColBERT Reranker（可选）
    """
    logger.info("=== PCB-RAG 初始化中 ===")

    # 选择 LLM（Ollama 本地）
    llm_candidates = _pick_llm_candidates(OLLAMA_BASE)
    if not llm_candidates:
        llm_candidates = ["qwen3.5:35b-a3b-q4_K_M"]
    # ── 关闭思考模式（thinking=False），大幅加速所有 LLM 调用 ───────────
    model_name = llm_candidates[0]
    Settings.llm = Ollama(
        model=model_name,
        base_url=OLLAMA_BASE,
        request_timeout=180.0,
        context_window=DEFAULT_NUM_CTX,
        additional_kwargs={
            "num_ctx": DEFAULT_NUM_CTX,
            # Qwen3.5 官方 Instruct (non-thinking) 模式推荐参数
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
        },
        thinking=False,   # ★ Ollama API: "think": false → 关闭思考模式
    )
    _app_state["llm_candidates"] = llm_candidates
    logger.info(f"[LLM] 已配置: {model_name} (thinking=False)")

    # 初始化向量索引（连接 Milvus + 加载 Embedding 模型）
    logger.info("[Index] 正在连接 Milvus 并加载 Embedding 模型...")
    index = build_index(llm_candidates[0])
    _app_state["index"] = index
    logger.info("[Index] VectorStoreIndex 初始化完成")

    # 初始化本地 BM25 索引
    bm25_index = None
    if RERANK_ENABLED or True:  # BM25 在 Fusion 模式下始终需要
        try:
            vector_store = index.storage_context.vector_store
            if isinstance(vector_store, MilvusVectorStore):
                logger.info("[BM25] 正在加载/构建本地词法索引...")
                bm25_index = _load_or_build_bm25(vector_store)
                if bm25_index:
                    logger.info(f"[BM25] 词法索引就绪，文档数={len(bm25_index.docs)}")
                else:
                    logger.warning("[BM25] 词法索引构建失败，将回退为纯向量检索")
        except Exception as e:
            logger.warning(f"[BM25] 未能初始化: {e}")
    _app_state["bm25_index"] = bm25_index

    # 初始化 Reranker（BGE / Qwen3-Reranker-4B）
    rerank = _try_build_reranker()
    _app_state["rerank"] = rerank
    if rerank:
        recall_k = max(RECALL_TOP_K, RERANK_TOP_N)
        logger.info(f"[Rerank] 精排模型就绪，top_n={RERANK_TOP_N}, recall_k={recall_k}")
    else:
        recall_k = 20
        logger.warning("[Rerank] 未启用或加载失败，使用 recall_k=20")
    _app_state["recall_k"] = recall_k

    # 初始化 ColBERT Reranker（MultiPath 模式可选）
    colbert_reranker = None
    if MULTIPATH_ENABLED and COLBERT_RERANK_ENABLED and bm25_index is not None:
        colbert_reranker = _try_build_colbert_reranker()
        if colbert_reranker:
            logger.info("[ColBERT] 多路召回精排模型就绪")
    _app_state["colbert_reranker"] = colbert_reranker

    # 打印最终检索模式
    if MULTIPATH_ENABLED and bm25_index:
        logger.info("[检索模式] MultiPath 多路召回 + Late Interaction")
    elif bm25_index:
        logger.info("[检索模式] 三路 Fusion（向量 + HyDE向量 + BM25+）+ Rerank")
    else:
        logger.info("[检索模式] 纯向量检索（BM25 不可用）")

    logger.info("=== PCB-RAG 初始化完成，开始服务 ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动时完成初始化，关闭时释放资源。"""
    _initialize_retrieval_engine()

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
            cleaned = sm.cleanup_expired()
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

# CORS 中间件：允许 Dify 等外部服务跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的方法、路径和耗时，便于排查 Dify 调用问题。"""
    start = time.time()
    method = request.method
    path = request.url.path
    logger.info(f"[REQ] {method} {path} - 开始处理")
    try:
        response = await call_next(request)
        elapsed = time.time() - start
        logger.info(f"[REQ] {method} {path} - {response.status_code} ({elapsed:.1f}s)")
        return response
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[REQ] {method} {path} - 异常 ({elapsed:.1f}s): {e}")
        raise


# ---------------------------------------------------------------------------
# 6. 数据模型（严格遵循 Dify 外部知识库 API 规范）
# ---------------------------------------------------------------------------
class RetrievalSetting(BaseModel):
    top_k: int = Field(default=5, description="返回的最大文档数")
    score_threshold: float = Field(default=0.0, description="最低得分阈值")
    score_threshold_enabled: bool = Field(default=False)


class RetrievalReq(BaseModel):
    knowledge_id: str = Field(description="知识库 ID（Dify 生成，接收即可）")
    query: str = Field(description="用户查询文本")
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
    """验证 Bearer Token，与 Dify 后台配置保持一致。"""
    if not authorization or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API token.")


# ---------------------------------------------------------------------------
# 8. 核心检索函数（完整复用 query.py 的优化管道）
# ---------------------------------------------------------------------------
def _retrieve_nodes(user_query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[NodeWithScore]:
    """
    完整执行 PCB-RAG 检索管道，返回原始 NodeWithScore 列表（供 LLM 合成使用）。

    管道顺序：
      1. 元数据过滤解析（source_type/vendor/eda 等）
      2. 查询扩展（PCB 同义词替换）
      3. HyDE / Query2Doc 查询增强（短查询或全量增强）
      4. 多路召回（MultiPath 或 Fusion 三路）
      5. BGE / Qwen3-VL Rerank 精排
      6. Chunk 上下文扩展（邻近 Chunk 合并）
      7. 按得分阈值过滤 + top_k 截断
    """
    index       = _app_state["index"]
    bm25_index  = _app_state["bm25_index"]
    rerank      = _app_state["rerank"]
    recall_k    = _app_state["recall_k"]
    colbert_reranker = _app_state["colbert_reranker"]

    t_start = time.time()

    # ── Step 1: 提取元数据过滤条件 ────────────────────────────────────────
    clean_q, filters = _extract_query_filters(user_query)
    if filters:
        filter_strs = [f"{f.key}={f.value}" for f in getattr(filters, "filters", [])]
        logger.info(f"[Filter] {'; '.join(filter_strs)}")

    # ── Step 2: 智能查询路由（动态权重）──────────────────────────────────
    query_type = "general"
    strategy = None
    dynamic_weights = FUSION_WEIGHTS
    if QUERY_ROUTING_ENABLED:
        query_type = _classify_query(clean_q)
        strategy = _get_query_strategy(query_type)
        dynamic_weights = (strategy["vector_weight"], strategy["bm25_weight"])
        logger.info(f"[QueryRoute] {query_type} ({strategy['description']}), "
                    f"weights(vec={dynamic_weights[0]:.2f}, bm25={dynamic_weights[1]:.2f})")

    # ── Step 3: 查询扩展（同义词替换）────────────────────────────────────
    expanded_q = _expand_query(clean_q)
    if expanded_q != clean_q:
        logger.info(f"[QueryExpand] {expanded_q[:80]}")

    # ── Step 4: HyDE / Query2Doc 查询增强 ────────────────────────────────
    #    Fusion 模式下 HyDE 将在 Step 5 中与检索并行执行，此处跳过
    hyde_q   = expanded_q
    q2doc_q  = expanded_q
    should_enhance = QUERY_ENHANCE_ALL or _is_short_query(clean_q)
    _use_parallel_fusion = (bm25_index is not None and not MULTIPATH_ENABLED)

    if should_enhance and not _use_parallel_fusion:
        # 仅 MultiPath / 纯向量模式：同步预计算 HyDE / Query2Doc
        if HYDE_ENABLED:
            try:
                hyde_q = hyde_expand_query(expanded_q)
            except Exception as e:
                logger.warning(f"[HyDE] 生成失败: {e}")
                hyde_q = expanded_q
        if QUERY2DOC_ENABLED:
            try:
                q2doc_q = query2doc_expand(expanded_q)
            except Exception as e:
                logger.warning(f"[Query2Doc] 失败: {e}")
                q2doc_q = expanded_q

    t_prep = time.time()
    logger.info(f"[Timing] 预处理: {t_prep - t_start:.2f}s")

    # ── Step 5: 构建检索器 ────────────────────────────────────────────────
    vector_retriever = index.as_retriever(similarity_top_k=recall_k, filters=filters)

    retrieved_nodes: List[NodeWithScore] = []

    if MULTIPATH_ENABLED and bm25_index is not None:
        # ── 5a. MultiPath 多路召回 + Late Interaction ──────────────────
        logger.info("[MultiPath] 执行多路 Dense+Sparse 召回...")
        lexical_retriever = LocalBM25Retriever(
            bm25_index,
            similarity_top_k=MULTIPATH_SPARSE_TOP_K,
            filters=filters,
        )

        # 多扩展查询变体（用于 BM25）- 使用路由策略的 num_queries
        target_num_queries = strategy["num_queries"] if strategy else FUSION_NUM_QUERIES
        if MULTI_EXPAND_ENABLED and len(clean_q) <= SHORT_QUERY_THRESHOLD:
            expand_queries = _build_multi_expand_queries(q2doc_q, num_queries=target_num_queries)
            logger.info(f"[MultiExpand] 生成 {len(expand_queries)} 个变体查询")
        else:
            expand_queries = [q2doc_q]

        multipath_retriever = MultiPathRetriever(
            dense_retriever=vector_retriever,
            sparse_retriever=lexical_retriever,
            colbert_reranker=colbert_reranker,
            dense_top_k=MULTIPATH_DENSE_TOP_K,
            sparse_top_k=MULTIPATH_SPARSE_TOP_K,
            rerank_candidates=MULTIPATH_RERANK_CANDIDATES,
            rrf_k=MULTIPATH_RRF_K,
            enable_colbert=COLBERT_RERANK_ENABLED and colbert_reranker is not None,
        )
        retrieved_nodes = multipath_retriever.retrieve(
            hyde_q,
            top_k=RERANK_TOP_N if rerank else recall_k,
            expand_queries=expand_queries,
        )

        # 标准 Rerank（ColBERT 未启用时）
        if rerank is not None and (colbert_reranker is None or not COLBERT_RERANK_ENABLED):
            logger.info(f"[Rerank] 精排中，top_n={RERANK_TOP_N}...")
            retrieved_nodes = rerank._postprocess_nodes(
                retrieved_nodes,
                query_bundle=QueryBundle(query_str=expanded_q),
            )

    elif bm25_index is not None:
        # ── 5b. 并行 Fusion 召回（HyDE LLM + 向量 + BM25 同时执行）──────
        logger.info("[Fusion] 并行三路 RRF 召回（向量 + HyDE + BM25+）...")
        lexical_retriever = LocalBM25Retriever(
            bm25_index, similarity_top_k=recall_k, filters=filters,
        )

        vec_weight  = float(dynamic_weights[0]) if len(dynamic_weights) >= 1 else 1.0
        bm25_weight = float(dynamic_weights[1]) if len(dynamic_weights) >= 2 else 1.0
        hyde_weight = vec_weight * float(HYDE_ROUTE_WEIGHT)

        # 多扩展查询变体（规则生成，<0.01s）
        target_nq = strategy["num_queries"] if strategy else FUSION_NUM_QUERIES
        if MULTI_EXPAND_ENABLED:
            raw_queries = _build_multi_expand_queries(expanded_q, num_queries=target_nq)
            bm25_queries = _build_multi_expand_queries(expanded_q, num_queries=target_nq)
        else:
            raw_queries = [expanded_q]
            bm25_queries = [expanded_q]

        # ★ 核心优化：
        #   HyDE 2路提前提交到专用 _HYDE_EXECUTOR（不阻塞主检索）
        #   主检索 pool 仅跑向量+BM25，完成后非阻塞 check HyDE
        hyde_futs: list = []
        if should_enhance and HYDE_ENABLED:
            hyde_futs.append(_HYDE_EXECUTOR.submit(hyde_expand_query, expanded_q))  # 路1: 事实型
            hyde_futs.append(_HYDE_EXECUTOR.submit(_hyde_expand_v2, expanded_q))    # 路2: 规范型

        # 主检索 pool：向量 + BM25（与 HyDE LLM 完全并行）
        n_workers = len(raw_queries) + len(bm25_queries)
        with ThreadPoolExecutor(max_workers=min(n_workers, 8)) as pool:
            # ① 所有向量检索并行提交
            vec_futs = [pool.submit(vector_retriever.retrieve, q) for q in raw_queries]
            # ② 所有 BM25 检索并行提交
            bm25_futs = [pool.submit(lexical_retriever.retrieve, q) for q in bm25_queries]

            # 收集向量结果（去重）
            all_raw_vec: List[NodeWithScore] = []
            seen_raw: set = set()
            for f in vec_futs:
                try:
                    for n in f.result(timeout=60)[:recall_k]:
                        nid = _node_id_key(n)
                        if nid not in seen_raw:
                            seen_raw.add(nid)
                            all_raw_vec.append(n)
                except Exception as e:
                    logger.warning(f"[Vec] 检索失败: {e}")
            raw_vec_nodes = all_raw_vec[:recall_k * 2]

            # 收集 BM25 结果（去重）
            all_bm25: List[NodeWithScore] = []
            seen_bm25: set = set()
            for f in bm25_futs:
                try:
                    for n in f.result(timeout=60)[:recall_k]:
                        nid = _node_id_key(n)
                        if nid not in seen_bm25:
                            seen_bm25.add(nid)
                            all_bm25.append(n)
                except Exception as e:
                    logger.warning(f"[BM25] 检索失败: {e}")
            bm25_nodes = all_bm25[:recall_k * 2]

        t_main = time.time()
        logger.info(f"[Timing] 主检索: {t_main - t_prep:.2f}s "
                   f"(vec={len(raw_vec_nodes)}, bm25={len(bm25_nodes)})")

        # ③ 收集 HyDE：给予最多 HYDE_WAIT_BUDGET 秒的等待时间
        #    qwen3.5:35b-a3b-q4_K_M 关闭思考后 HyDE 一般 5-20s，首次冷启动可能更久
        HYDE_WAIT_BUDGET = float(os.getenv("HYDE_WAIT_BUDGET", "30"))
        hyde_elapsed = t_main - t_prep  # HyDE 已经跑了这么久
        hyde_remain = max(0, HYDE_WAIT_BUDGET - hyde_elapsed)

        hyde_vec_nodes: List[NodeWithScore] = []
        seen_hyde: set = set()
        n_hit, n_skip = 0, 0

        if hyde_futs and hyde_remain > 0:
            from concurrent.futures import wait as _futures_wait, FIRST_COMPLETED
            done_set, not_done = _futures_wait(hyde_futs, timeout=hyde_remain)
            for hf in done_set:
                try:
                    _hq = hf.result()
                    if _hq and _hq != expanded_q:
                        for n in vector_retriever.retrieve(_hq)[:recall_k]:
                            nid = _node_id_key(n)
                            if nid not in seen_hyde:
                                seen_hyde.add(nid)
                                hyde_vec_nodes.append(n)
                        n_hit += 1
                except Exception as e:
                    logger.warning(f"[HyDE] 结果取失败: {e}")
            n_skip = len(not_done)
            for hf in not_done:
                hf.cancel()
                logger.info("[HyDE] 生成超时，取消")
        elif hyde_futs:
            n_skip = len(hyde_futs)
            for hf in hyde_futs:
                hf.cancel()
            logger.info(f"[HyDE] 主检索已耗时 {hyde_elapsed:.1f}s，无剩余预算，跳过 HyDE")

        t_hyde = time.time()
        logger.info(f"[Timing] HyDE: {t_hyde - t_prep:.2f}s "
                   f"(命中={n_hit}/{len(hyde_futs)}, 跳过={n_skip}/{len(hyde_futs)}, "
                   f"nodes={len(hyde_vec_nodes)})")
        if n_hit == 0 and hyde_futs:
            logger.warning("[HyDE] 本次全部未命中，可能原因: LLM 响应慢/模型加载中。"
                          f" 预算={HYDE_WAIT_BUDGET}s, 主检索耗时={hyde_elapsed:.1f}s")

        # RRF 三路融合
        routes: list = [
            (raw_vec_nodes, vec_weight),
            (bm25_nodes, bm25_weight),
        ]
        if hyde_vec_nodes:
            routes.insert(1, (hyde_vec_nodes, hyde_weight))

        retrieved_nodes = _weighted_rrf_fuse_three_routes(
            routes, top_n=recall_k, rrf_k=FUSION_RRF_K,
        )

        # Rerank 精排
        if rerank is not None:
            t_pre_rerank = time.time()
            logger.info(f"[Rerank] 精排 {len(retrieved_nodes)} 候选...")
            retrieved_nodes = rerank._postprocess_nodes(
                retrieved_nodes,
                query_bundle=QueryBundle(query_str=expanded_q),
            )
            logger.info(f"[Timing] 精排: {time.time() - t_pre_rerank:.2f}s "
                       f"→ {len(retrieved_nodes)} nodes")

    else:
        # ── 5c. 纯向量检索（BM25 不可用）──────────────────────────────
        logger.info("[VectorOnly] 纯向量检索...")
        retrieved_nodes = vector_retriever.retrieve(
            QueryBundle(query_str=hyde_q)
        )
        if rerank is not None:
            retrieved_nodes = rerank._postprocess_nodes(
                retrieved_nodes,
                query_bundle=QueryBundle(query_str=expanded_q),
            )

    # ── Step 6: Chunk 上下文扩展（合并邻近 Chunk）──────────────────────
    if CHUNK_EXPAND_ENABLED and retrieved_nodes:
        original_count = len(retrieved_nodes)
        try:
            retrieved_nodes = expand_chunks_with_context(
                retrieved_nodes,
                index=index,
                expand_neighbors=CHUNK_EXPAND_NEIGHBORS,
                expand_parent=CHUNK_EXPAND_PARENT,
                group_by_doc=CHUNK_GROUP_BY_DOC,
            )
            if len(retrieved_nodes) > original_count:
                logger.info(f"[ChunkExpand] {original_count} → {len(retrieved_nodes)} chunks")
        except Exception as e:
            logger.warning(f"[ChunkExpand] 扩展失败，跳过: {e}")

    # ── Step 7: 按得分阈值过滤 + top_k 截断 ──────────────────────────────
    filtered: List[NodeWithScore] = []
    for nws in retrieved_nodes:
        score = float(nws.score) if nws.score is not None else 0.0
        if score_threshold > 0.0 and score < score_threshold:
            continue
        node = nws.node
        content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
        if not content:
            continue
        filtered.append(nws)
        if len(filtered) >= top_k:
            break

    t_total = time.time()
    logger.info(f"[Timing] 检索总耗时: {t_total - t_start:.2f}s → {len(filtered)} 条 "
               f"(query={user_query[:40]}...)")
    return filtered


def retrieve_chunks(user_query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Record]:
    """
    完整执行 PCB-RAG 检索管道，返回适配 Dify 的 Record 列表。
    内部调用 _retrieve_nodes 获取 NodeWithScore，再转换为 Record。
    """
    nodes = _retrieve_nodes(user_query, top_k=top_k, score_threshold=score_threshold)
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
                "source_path": source_path,
                "source_type": meta.get("source_type", ""),
                "vendor": meta.get("vendor", ""),
                "eda": meta.get("eda", ""),
            },
        ))
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
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

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
    query: str = Field(description="用户问题")
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


class AskRes(BaseModel):
    answer: str
    sources: List[str] = []
    records: List[Record] = []
    rewritten_query: Optional[str] = Field(
        default=None,
        description="经历史改写后的独立查询（仅在有历史改写时返回，便于调试）"
    )


@app.post("/api/ask", response_model=AskRes, summary="全链路问答接口（检索 + LLM 合成，支持多轮对话）")
def ask_rag(
    req: AskReq,
    _: str = Depends(verify_token),
):
    """
    全链路 RAG 问答端点：执行完整检索管道后，调用本地 LLM 合成答案。
    使用 def（非 async def）确保同步 LLM 调用在线程池中执行，不阻塞事件循环。
    **支持多轮对话**：通过 chat_history 传入对话历史，自动解析指代/省略。

    返回：
    - answer: LLM 生成的答案（含引用标注）
    - sources: 引用的来源文件路径列表
    - records: 检索到的原始 chunk 列表（可选用于调试）
    - rewritten_query: 改写后的查询（仅在有历史改写时返回）

    Dify 工作流 HTTP Request 节点示例配置：
      Method: POST
      URL: http://<服务器IP>:8000/api/ask
      Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
      Body: {"query": "{{#sys.query#}}", "top_k": 5,
             "chat_history": {{#sys.conversation_history#}} }
    提取输出：{{#http_request.body.answer#}}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    threshold = req.score_threshold if req.score_threshold_enabled else 0.0
    raw_query = req.query.strip()
    history_dicts = [m.model_dump() for m in req.chat_history] if req.chat_history else []

    try:
        t_ask_start = time.time()

        # ── 历史感知查询改写：解析代词/省略 → 独立查询 ─────────────────
        search_query = raw_query
        rewritten = None
        if history_dicts:
            search_query = _rewrite_query_with_history(raw_query, history_dicts)
            if search_query != raw_query:
                rewritten = search_query
            logger.info(f"[Timing] 历史改写: {time.time() - t_ask_start:.2f}s")

        # ── 检索阶段：使用改写后的查询检索 ───────────────────────────────
        t_ret = time.time()
        nodes = _retrieve_nodes(
            user_query=search_query,
            top_k=req.top_k,
            score_threshold=threshold,
        )
        logger.info(f"[Timing] 检索阶段: {time.time() - t_ret:.2f}s")

        # ── 合成阶段：LLM 生成带引用的答案（注入对话历史）───────────────
        t_synth = time.time()
        use_sr = req.use_self_rag and SELF_RAG_ENABLED
        if use_sr:
            logger.info("[Self-RAG] 启动自反思问答迭代...")
            class _NodeListRetriever:
                def __init__(self, ns):
                    self._ns = ns
                def retrieve(self, _query_str):
                    return self._ns

            sr_result = self_rag_query(
                query=raw_query,
                retriever=_NodeListRetriever(nodes),
                llm=Settings.llm,
            )
            answer_text = format_self_rag_result(sr_result, show_history=False)
            sources = []
        else:
            result = _generate_answer_with_history(
                query=raw_query,
                retrieved_docs=nodes,
                chat_history=history_dicts,
                llm=Settings.llm,
            )
            answer_text = format_answer_with_citations(result, verbose=False)
            sources = result.get("sources", [])

        logger.info(f"[Timing] 答案合成: {time.time() - t_synth:.2f}s")

        # ── 同时打包 records（供调试）────────────────────────────────────
        records_out: List[Record] = []
        for nws in nodes:
            score = float(nws.score) if nws.score is not None else 0.0
            node = nws.node
            content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
            meta: dict = node.metadata if hasattr(node, "metadata") else {}
            source_path: str = meta.get("source_path", "")
            source_name: str = Path(source_path).stem if source_path else meta.get("source_type", "未知来源")
            records_out.append(Record(
                content=content, score=round(score, 6), title=source_name,
                metadata={"source_path": source_path,
                          "source_type": meta.get("source_type", ""),
                          "vendor": meta.get("vendor", ""),
                          "eda": meta.get("eda", "")},
            ))

    except Exception as e:
        logger.error(f"[Error] /api/ask 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

    logger.info(f"[Ask] 答案生成完毕（总耗时={time.time() - t_ask_start:.1f}s, query={raw_query[:40]}...）")
    return AskRes(answer=answer_text, sources=sources, records=records_out, rewritten_query=rewritten)


# ---------------------------------------------------------------------------
# 10a. /api/chat：带会话记忆的问答端点（推荐使用）
#      自动管理对话历史，客户端只需传递 session_id
# ---------------------------------------------------------------------------
class ChatReq(BaseModel):
    query: str = Field(description="用户问题")
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID（首次对话不传，后续对话传入返回的 session_id）"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="检索 chunk 数量")
    score_threshold: float = Field(default=0.0, ge=0.0, description="最低得分阈值")
    score_threshold_enabled: bool = Field(default=False)
    use_self_rag: bool = Field(default=False, description="是否启用 Self-RAG 自反思迭代")


class ChatRes(BaseModel):
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
def chat_rag(
    req: ChatReq,
    _: str = Depends(verify_token),
):
    """
    带服务端会话记忆的 RAG 问答端点。

    **核心特性**：
    - 自动管理对话历史，无需客户端维护
    - 首次对话不传 session_id，返回新会话ID
    - 后续对话传入 session_id 即可延续上下文
    - 支持指代消解（"它是什么？"→ 自动补全为完整问题）

    **使用流程**：
    1. 首次请求：{"query": "什么是HDI板？"}
       响应：{"answer": "...", "session_id": "abc123", "is_new_session": true}
    2. 后续请求：{"query": "它的工艺要求是什么？", "session_id": "abc123"}
       响应：{"answer": "...", "session_id": "abc123", "is_new_session": false}
       （"它"会被自动解析为"HDI板"）

    **Dify 工作流配置**：
      Method: POST
      URL: http://<服务器IP>:8000/api/chat
      Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
      Body: {"query": "{{#sys.query#}}", "session_id": "{{#自定义变量.session_id#}}"}
    提取输出：
      - 答案：{{#http_request.body.answer#}}
      - 会话ID：{{#http_request.body.session_id#}}（存入变量供下次使用）
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    sm = get_session_manager()
    session_id, is_new = sm.get_or_create(req.session_id)
    threshold = req.score_threshold if req.score_threshold_enabled else 0.0
    raw_query = req.query.strip()

    try:
        t_chat_start = time.time()

        history = sm.get_history(session_id)

        history_dicts = history if history else []

        search_query = raw_query
        rewritten = None
        if history_dicts:
            search_query = _rewrite_query_with_history(raw_query, history_dicts)
            if search_query != raw_query:
                rewritten = search_query
            logger.info(f"[Timing] 历史改写: {time.time() - t_chat_start:.2f}s")

        t_ret = time.time()
        nodes = _retrieve_nodes(
            user_query=search_query,
            top_k=req.top_k,
            score_threshold=threshold,
        )
        logger.info(f"[Timing] 检索阶段: {time.time() - t_ret:.2f}s")

        t_synth = time.time()
        use_sr = req.use_self_rag and SELF_RAG_ENABLED
        if use_sr:
            logger.info("[Self-RAG] 启动自反思问答迭代...")
            class _NodeListRetriever:
                def __init__(self, ns):
                    self._ns = ns
                def retrieve(self, _query_str):
                    return self._ns

            sr_result = self_rag_query(
                query=raw_query,
                retriever=_NodeListRetriever(nodes),
                llm=Settings.llm,
            )
            answer_text = format_self_rag_result(sr_result, show_history=False)
            sources = []
        else:
            result = _generate_answer_with_history(
                query=raw_query,
                retrieved_docs=nodes,
                chat_history=history_dicts,
                llm=Settings.llm,
            )
            answer_text = format_answer_with_citations(result, verbose=False)
            sources = result.get("sources", [])

        logger.info(f"[Timing] 答案合成: {time.time() - t_synth:.2f}s")

        sm.add_message(session_id, "user", raw_query)
        sm.add_message(session_id, "assistant", answer_text)

        records_out: List[Record] = []
        for nws in nodes:
            score = float(nws.score) if nws.score is not None else 0.0
            node = nws.node
            content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
            meta: dict = node.metadata if hasattr(node, "metadata") else {}
            source_path: str = meta.get("source_path", "")
            source_name: str = Path(source_path).stem if source_path else meta.get("source_type", "未知来源")
            records_out.append(Record(
                content=content, score=round(score, 6), title=source_name,
                metadata={"source_path": source_path,
                          "source_type": meta.get("source_type", ""),
                          "vendor": meta.get("vendor", ""),
                          "eda": meta.get("eda", "")},
            ))

    except Exception as e:
        logger.error(f"[Error] /api/chat 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

    logger.info(f"[Chat] 答案生成完毕（总耗时={time.time() - t_chat_start:.1f}s, "
                f"session={session_id[:8]}..., new={is_new}, query={raw_query[:40]}...）")
    return ChatRes(
        answer=answer_text,
        session_id=session_id,
        is_new_session=is_new,
        sources=sources,
        records=records_out,
        rewritten_query=rewritten,
    )


# ---------------------------------------------------------------------------
# 10b. /api/chat/plain：带会话记忆的纯文本问答端点
# ---------------------------------------------------------------------------
@app.post("/api/chat/plain", response_class=Response, summary="带会话记忆的纯文本问答接口")
def chat_rag_plain(
    req: ChatReq,
    _: str = Depends(verify_token),
):
    """
    与 /api/chat 逻辑相同，但以 text/plain 返回纯答案字符串。
    响应头 X-Session-ID 包含会话ID，X-Is-New-Session 标识是否为新会话。

    Dify 工作流配置：
      Method: POST
      URL: http://<服务器IP>:8000/api/chat/plain
      Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
      Body: {"query": "{{#sys.query#}}", "session_id": "{{#自定义变量.session_id#}}"}
    结束节点直接引用：{{#http_request.body#}}
    从响应头提取会话ID：{{#http_request.headers.X-Session-ID#}}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    sm = get_session_manager()
    session_id, is_new = sm.get_or_create(req.session_id)
    threshold = req.score_threshold if req.score_threshold_enabled else 0.0
    raw_query = req.query.strip()

    try:
        t_plain_start = time.time()

        history = sm.get_history(session_id)
        history_dicts = history if history else []

        search_query = raw_query
        if history_dicts:
            search_query = _rewrite_query_with_history(raw_query, history_dicts)

        nodes = _retrieve_nodes(
            user_query=search_query,
            top_k=req.top_k,
            score_threshold=threshold,
        )

        use_sr = req.use_self_rag and SELF_RAG_ENABLED
        if use_sr:
            class _NodeListRetriever:
                def __init__(self, ns):
                    self._ns = ns
                def retrieve(self, _q):
                    return self._ns
            sr_result = self_rag_query(
                query=raw_query,
                retriever=_NodeListRetriever(nodes),
                llm=Settings.llm,
            )
            answer_text = format_self_rag_result(sr_result, show_history=False)
        else:
            result = _generate_answer_with_history(
                query=raw_query,
                retrieved_docs=nodes,
                chat_history=history_dicts,
                llm=Settings.llm,
            )
            answer_text = format_answer_with_citations(result, verbose=False)

        sm.add_message(session_id, "user", raw_query)
        sm.add_message(session_id, "assistant", answer_text)

    except Exception as e:
        logger.error(f"[Error] /api/chat/plain 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

    logger.info(f"[ChatPlain] 答案生成完毕（总耗时={time.time() - t_plain_start:.1f}s, "
                f"session={session_id[:8]}..., new={is_new}, query={raw_query[:40]}...）")
    return Response(
        content=answer_text,
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Session-ID": session_id,
            "X-Is-New-Session": str(is_new).lower(),
        },
    )


# ---------------------------------------------------------------------------
# 10c. /api/ask/plain：纯文本问答端点（Dify 工作流首选）
#      直接返回 text/plain，{{http_request.body}} 即为答案，无需 JSON 解析
# ---------------------------------------------------------------------------
@app.post("/api/ask/plain", response_class=Response, summary="纯文本问答接口（Dify 工作流推荐，支持多轮对话）")
def ask_rag_plain(
    req: AskReq,
    _: str = Depends(verify_token),
):
    """
    与 /api/ask 逻辑完全相同，但以 text/plain 返回纯答案字符串。
    使用 def（非 async def）确保同步 LLM 调用在线程池中执行。
    **支持多轮对话**：通过 chat_history 传入对话历史。

    Dify 工作流 HTTP Request 节点配置：
      Method: POST
      URL: http://<服务器IP>:8000/api/ask/plain
      Headers: Authorization: Bearer <API_TOKEN>, Content-Type: application/json
      Body: {"query": "{{#sys.query#}}", "top_k": 5,
             "chat_history": {{#sys.conversation_history#}} }
    结束节点直接引用：{{#http_request.body#}}
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    threshold = req.score_threshold if req.score_threshold_enabled else 0.0
    raw_query = req.query.strip()
    history_dicts = [m.model_dump() for m in req.chat_history] if req.chat_history else []

    try:
        t_plain_start = time.time()

        # ── 历史感知查询改写 ──────────────────────────────────────────────
        search_query = raw_query
        if history_dicts:
            search_query = _rewrite_query_with_history(raw_query, history_dicts)

        nodes = _retrieve_nodes(
            user_query=search_query,
            top_k=req.top_k,
            score_threshold=threshold,
        )
        use_sr = req.use_self_rag and SELF_RAG_ENABLED
        if use_sr:
            class _NodeListRetriever:
                def __init__(self, ns):
                    self._ns = ns
                def retrieve(self, _q):
                    return self._ns
            sr_result = self_rag_query(
                query=raw_query,
                retriever=_NodeListRetriever(nodes),
                llm=Settings.llm,
            )
            answer_text = format_self_rag_result(sr_result, show_history=False)
        else:
            result = _generate_answer_with_history(
                query=raw_query,
                retrieved_docs=nodes,
                chat_history=history_dicts,
                llm=Settings.llm,
            )
            answer_text = format_answer_with_citations(result, verbose=False)
    except Exception as e:
        logger.error(f"[Error] /api/ask/plain 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

    logger.info(f"[AskPlain] 答案生成完毕（总耗时={time.time() - t_plain_start:.1f}s, query={raw_query[:40]}...）")
    return Response(content=answer_text, media_type="text/plain; charset=utf-8")


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
    history = sm.get_history(session_id)
    if not history and session_id not in sm._sessions:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
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


# ---------------------------------------------------------------------------
# 12. 健康检查端点（供 Dify 或 AD 监控服务状态）
# ---------------------------------------------------------------------------
@app.get("/health", summary="健康检查")
def health_check():
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return {
        "status": "ok",
        "index_ready": "index" in _app_state,
        "bm25_ready":  _app_state.get("bm25_index") is not None,
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
    }


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
