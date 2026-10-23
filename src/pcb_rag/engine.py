"""面向 Agent 的知识库元件。

定位
----
本模块是**程序接口**，不是 HTTP 服务。两者分工：

- :mod:`pcb_rag.dify_external_api` —— HTTP 层。FastAPI 端点、Dify 协议适配、
  Bearer 鉴权、会话管理。
- 本模块 —— 元件层。供自研 agent 直接 ``import`` 调用，返回结构化对象。

二者共用同一套检索实现与同一个索引实例（``_app_state``），因此不会重复建索引、
不会出现两份词法索引互相不一致。

为什么返回值是 dataclass 而不是字符串
--------------------------------------
agent 需要自己做决策：这条证据够不够、要不要再检索一轮、引用要不要展示给用户。
把上下文拼成一个字符串交给 agent，等于把"是否足够"的判断锁死在元件内部。
因此 :meth:`KnowledgeBase.retrieve` 返回逐条证据，``context`` 作为可选字段单独提供。

``degraded`` 字段是刻意保留的：检索链路有多处降级（图缺失、精排失败、词法索引
不可用），静默降级会让 agent 误以为"没有更多证据"，而实际是"某条路没跑起来"。

与 HTTP 模块的耦合
------------------
本模块复用 ``dify_external_api`` 的 ``_app_state`` 与 ``_retrieve_nodes``，这是
**过渡态而非终态**：检索编排有 400 余行（多路召回、融合、精排、chunk 扩展、图增强、
权限复核），整体搬迁会在没有集成测试的环境里引入难以定位的回归。

演进方向：把 ``_retrieve_nodes`` 及其依赖下沉到独立的 retrieval 层，``engine`` 与
HTTP 层都只依赖那一层。届时本模块的 ``import`` 全部改为指向新的 retrieval 层，
对外接口不变。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结构化结果
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    """一条检索证据。"""

    index: int
    source_path: str
    source_name: str
    section: str = ""
    chunk_id: str = ""
    score: float = 0.0
    tokens: int = 0
    snippet: str = ""
    text: str = ""
    truncated: bool = False
    doc_type: str = ""
    source_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """检索结果。

    Attributes:
        query: 原始查询
        rewritten_query: 历史感知改写后的检索式；未改写时为 None
        evidence: 逐条证据（已按相关度降序）
        context: 拼装好的上下文（三段式）；不需要时为 None
        elapsed_ms: 端到端耗时
        degraded: 降级说明列表，空列表表示全链路正常
    """

    query: str
    evidence: List[Evidence] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    context: Optional[str] = None
    elapsed_ms: int = 0
    degraded: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "evidence": [e.to_dict() for e in self.evidence],
            "context": self.context,
            "elapsed_ms": self.elapsed_ms,
            "degraded": list(self.degraded),
            "is_empty": self.is_empty,
        }


@dataclass
class AnswerResult:
    """问答结果。"""

    query: str
    answer: str
    evidence: List[Evidence] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: int = 0
    degraded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": list(self.citations),
            "elapsed_ms": self.elapsed_ms,
            "degraded": list(self.degraded),
        }


@dataclass
class GraphNeighborhood:
    """实体的图邻域。"""

    entity: str
    found: bool
    node_type: str = ""
    description: str = ""
    degree: int = 0
    relations: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    degraded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 元件
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """PCB 知识库元件。

    典型用法::

        kb = KnowledgeBase()
        kb.ensure_ready()                     # 首次调用会连接 Milvus、加载模型

        hit = kb.retrieve("4 层板阻抗控制关注什么", top_k=8)
        for ev in hit.evidence:
            print(ev.source_name, ev.section, ev.score)

        ans = kb.answer("4 层板阻抗控制关注什么")

    线程安全：``ensure_ready`` 用锁保护，重复调用是廉价的空操作；检索本身
    复用底层索引的只读结构，可并发调用。
    """

    def __init__(
        self,
        *,
        auto_init: bool = True,
        include_full_text: bool = True,
        include_context: bool = False,
    ) -> None:
        """
        Args:
            auto_init: 构造时是否立即初始化（False 则首次调用时惰性初始化）
            include_full_text: 证据里是否带完整正文。agent 需要自行摘录时开启；
                只做展示时关掉可显著减少内存与序列化开销。
            include_context: ``retrieve`` 是否同时返回拼装好的上下文。
                只有当 agent 想把整块上下文直接喂给模型时才需要。
        """
        self.include_full_text = include_full_text
        self.include_context = include_context
        self._ready = False
        self._init_lock = threading.Lock()
        if auto_init:
            self.ensure_ready()

    # ---- 生命周期 --------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready and self._state_ready()

    @staticmethod
    def _state() -> Dict[str, Any]:
        """取底层共享状态（延迟导入，避免在无 HTTP 需求的进程里拉起 FastAPI）。"""
        from pcb_rag import dify_external_api as _api

        return _api._app_state

    def _state_ready(self) -> bool:
        try:
            return "index" in self._state()
        except Exception:
            return False

    def ensure_ready(self) -> None:
        """确保索引与模型已就绪。幂等。

        若同进程内 HTTP 服务已初始化过，这里直接复用，不会重复建索引。
        """
        if self.ready:
            return

        with self._init_lock:
            if self.ready:
                return

            from pcb_rag import dify_external_api as _api

            if "index" not in _api._app_state:
                logger.info("[KB] 初始化检索组件（首次调用，耗时较长）")
                _api._initialize_retrieval_engine()
            self._ready = True

    def health(self) -> Dict[str, Any]:
        """返回各组件就绪状态与降级原因。agent 可据此决定是否先提示用户。"""
        try:
            state = self._state()
        except Exception as exc:
            return {"ready": False, "error": str(exc), "degraded": ["init_failed"]}

        index_ok = "index" in state
        bm25_ok = state.get("bm25_index") is not None
        rerank_ok = state.get("rerank") is not None
        graph_ok = state.get("graph") is not None

        degraded: List[str] = []
        if not index_ok:
            degraded.append("vector_index_missing")
        if not bm25_ok:
            degraded.append("lexical_index_missing")
        if not rerank_ok:
            degraded.append("rerank_unavailable")
        if not graph_ok:
            degraded.append("graph_unavailable")

        return {
            "ready": index_ok,
            "vector_index": index_ok,
            "lexical_index": bm25_ok,
            "rerank": rerank_ok,
            "graph": graph_ok,
            "recall_k": state.get("recall_k"),
            "degraded": degraded,
        }

    def _collect_degraded(self) -> List[str]:
        return list(self.health().get("degraded", []))

    # ---- 内部：NodeWithScore → Evidence --------------------------------
    def _to_evidence(self, item: Any, index: int) -> Optional[Evidence]:
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", None) or {}

        source_path = metadata.get("source_path", "") or ""
        source_name = Path(source_path).stem if source_path else ""

        section = (
            metadata.get("parent_title")
            or metadata.get("chunk_section")
            or metadata.get("section")
            or ""
        )

        text = ""
        try:
            from llama_index.core.schema import MetadataMode

            text = node.get_content(metadata_mode=MetadataMode.NONE) or ""
        except Exception:
            text = str(getattr(node, "text", "") or "")

        if not text:
            return None

        # token 数按正文实算：metadata 里只有 char_count，字符数不能直接当
        # token 用（中英混排差异很大），而 agent 需要它来估算自己的预算。
        from pcb_rag.context_budget import estimate_tokens

        return Evidence(
            index=index,
            source_path=source_path,
            source_name=source_name or source_path or "未知来源",
            section=str(section),
            chunk_id=str(getattr(node, "id_", "") or metadata.get("chunk_id", "")),
            score=float(getattr(item, "score", 0.0) or 0.0),
            tokens=estimate_tokens(text),
            snippet=text[:400],
            text=text if self.include_full_text else "",
            # evidence 来自原始 node，未经上下文裁剪；是否被裁由 context 组装决定，
            # 与单条证据无关。
            truncated=False,
            doc_type=str(metadata.get("doc_type", "") or ""),
            source_type=str(metadata.get("source_type", "") or ""),
        )

    # ---- 检索 -----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        score_threshold: float = 0.0,
        with_context: Optional[bool] = None,
    ) -> RetrievalResult:
        """检索证据，不生成答案。

        Args:
            query: 用户查询
            top_k: 返回证据条数
            score_threshold: 精排分数阈值，0 表示不过滤
            with_context: 是否同时返回拼装好的上下文；默认取构造参数
        """
        started = time.time()
        if not query or not query.strip():
            return RetrievalResult(query=query or "", elapsed_ms=0, degraded=["empty_query"])

        self.ensure_ready()

        from pcb_rag.dify_external_api import _retrieve_nodes

        degraded = self._collect_degraded()

        try:
            nodes = _retrieve_nodes(
                user_query=query.strip(),
                top_k=int(top_k),
                score_threshold=float(score_threshold),
            )
        except Exception as exc:
            logger.exception("[KB] 检索失败")
            return RetrievalResult(
                query=query,
                elapsed_ms=int((time.time() - started) * 1000),
                degraded=degraded + [f"retrieval_error:{exc}"],
            )

        evidence: List[Evidence] = []
        for i, item in enumerate(nodes or [], 1):
            ev = self._to_evidence(item, i)
            if ev is not None:
                evidence.append(ev)

        want_context = self.include_context if with_context is None else bool(with_context)
        context: Optional[str] = None
        if want_context and nodes:
            from pcb_rag.query import build_context_with_sources

            try:
                context, _ = build_context_with_sources(nodes)
            except Exception as exc:
                degraded.append(f"context_build_error:{exc}")

        return RetrievalResult(
            query=query,
            evidence=evidence,
            context=context,
            elapsed_ms=int((time.time() - started) * 1000),
            degraded=degraded,
        )

    def answer(
        self,
        query: str,
        *,
        top_k: int = 10,
        score_threshold: float = 0.0,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> AnswerResult:
        """检索并生成答案，返回答案与引用。

        Args:
            history: 形如 ``[{"role": "user", "content": ...}, ...]`` 的对话历史。
                元件本身不保存会话状态 —— 多轮上下文由 agent 自己管理并传入。
        """
        started = time.time()
        if not query or not query.strip():
            return AnswerResult(query=query or "", answer="", degraded=["empty_query"])

        self.ensure_ready()

        from pcb_rag.dify_external_api import (
            _MAX_HISTORY_TURNS,
            _RAG_PROMPT_WITH_HISTORY,
            _retrieve_nodes,
            _rewrite_query_with_history,
        )
        from pcb_rag.query import RAG_PROMPT_TEMPLATE, build_context_with_sources, extract_citations

        degraded = self._collect_degraded()
        raw_query = query.strip()

        history_list = [dict(m) for m in (history or [])]
        search_query = raw_query
        if history_list:
            try:
                search_query = _rewrite_query_with_history(raw_query, history_list)
            except Exception as exc:
                degraded.append(f"query_rewrite_failed:{exc}")

        try:
            nodes = _retrieve_nodes(
                user_query=search_query,
                top_k=int(top_k),
                score_threshold=float(score_threshold),
            )
        except Exception as exc:
            logger.exception("[KB] 检索失败")
            return AnswerResult(
                query=raw_query,
                answer="",
                elapsed_ms=int((time.time() - started) * 1000),
                degraded=degraded + [f"retrieval_error:{exc}"],
            )

        evidence: List[Evidence] = []
        for i, item in enumerate(nodes or [], 1):
            ev = self._to_evidence(item, i)
            if ev is not None:
                evidence.append(ev)

        if not nodes:
            return AnswerResult(
                query=raw_query,
                answer="根据现有文档未检索到相关内容。",
                evidence=[],
                elapsed_ms=int((time.time() - started) * 1000),
                degraded=degraded + ["no_evidence"],
            )

        context, _sources_info = build_context_with_sources(nodes)

        if history_list:
            recent = history_list[-_MAX_HISTORY_TURNS:]
            lines = []
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"用户：{content}")
                elif role == "assistant":
                    truncated = content[:300] + "..." if len(content) > 300 else content
                    lines.append(f"助手：{truncated}")
            prompt = _RAG_PROMPT_WITH_HISTORY.format(
                n=len(recent),
                history="\n".join(lines),
                context=context,
                query=raw_query,
            )
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=raw_query)

        from llama_index.core import Settings

        try:
            response = Settings.llm.complete(prompt)
            answer = getattr(response, "text", None) or str(response)
        except Exception as exc:
            logger.exception("[KB] 生成失败")
            return AnswerResult(
                query=raw_query,
                answer="",
                evidence=evidence,
                elapsed_ms=int((time.time() - started) * 1000),
                degraded=degraded + [f"generation_error:{exc}"],
            )

        citations: List[Dict[str, Any]] = []
        try:
            citations = extract_citations(answer, nodes)
        except Exception as exc:
            degraded.append(f"citation_error:{exc}")

        return AnswerResult(
            query=raw_query,
            answer=answer,
            evidence=evidence,
            citations=citations,
            elapsed_ms=int((time.time() - started) * 1000),
            degraded=degraded,
        )

    # ---- 图 -------------------------------------------------------------
    def graph_neighbors(self, entity: str, *, max_edges: int = 12) -> GraphNeighborhood:
        """查询实体在图中的邻域（实体描述 + 关系描述 + 原文依据）。"""
        if not entity or not entity.strip():
            return GraphNeighborhood(entity=entity or "", found=False, degraded=["empty_query"])

        try:
            state = self._state()
        except Exception as exc:
            return GraphNeighborhood(entity=entity, found=False, degraded=[f"init_failed:{exc}"])

        graph = state.get("graph")
        if graph is None:
            return GraphNeighborhood(entity=entity, found=False, degraded=["graph_unavailable"])

        from pcb_rag.graph_rag import build_neighborhood_text, normalize_entity

        name = normalize_entity(entity)
        node = graph.nodes.get(name)
        if node is None:
            # 退化为模糊链接：查询词可能只是实体名的一部分
            from pcb_rag.graph_rag import link_entities

            hits = link_entities(entity, graph, max_entities=1)
            if not hits:
                return GraphNeighborhood(entity=entity, found=False, degraded=[])
            name = hits[0][0]
            node = graph.nodes.get(name)

        if node is None:
            return GraphNeighborhood(entity=entity, found=False, degraded=[])

        relations = []
        for edge in graph.edges_of(name)[:max_edges]:
            relations.append(
                {
                    "source": edge.get("h", ""),
                    "relation": edge.get("r", ""),
                    "target": edge.get("t", ""),
                    "relation_description": edge.get("rd", ""),
                    "weight": float(edge.get("w", 0.0) or 0.0),
                    "evidence": edge.get("e", ""),
                }
            )

        return GraphNeighborhood(
            entity=name,
            found=True,
            node_type=getattr(node, "type", ""),
            description=getattr(node, "desc", ""),
            degree=int(getattr(node, "count", 0) or 0),
            relations=relations,
            text=build_neighborhood_text(graph, name),
        )

    # ---- Agent 工具 ------------------------------------------------------
    def tool_specs(self) -> List[Dict[str, Any]]:
        """返回 OpenAI function calling 格式的工具定义。

        直接塞进 agent 的 tools 列表即可；配合 :meth:`call_tool` 分发。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": (
                        "在 PCB 知识库中检索原文证据（标准条款、工艺参数、EDA 操作）。"
                        "只返回证据片段与出处，不生成答案——需要自己判断是否足够。"
                        "适合需要精确引用、或想先看有哪些材料再决定下一步的场景。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索问题，尽量具体"},
                            "top_k": {
                                "type": "integer",
                                "description": "返回证据条数，默认 8，上限 30",
                                "minimum": 1,
                                "maximum": 30,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_knowledge",
                    "description": (
                        "在 PCB 知识库中检索并生成带引用的答案。"
                        "答案严格基于检索到的文档，文档没有依据时会明确说明。"
                        "适合直接回答用户问题的场景。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "用户问题"},
                            "top_k": {
                                "type": "integer",
                                "description": "参考证据条数，默认 10",
                                "minimum": 1,
                                "maximum": 30,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explore_entity",
                    "description": (
                        "查询某个 PCB 概念在知识图谱中的邻域：它是什么、与哪些实体相连、"
                        "关系说明了什么。适合探索概念之间的关联、或需要多跳推理的场景。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity": {
                                "type": "string",
                                "description": "实体名，如「镀金层厚度」「GB/T 4677-2002」",
                            }
                        },
                        "required": ["entity"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_status",
                    "description": "查询知识库各组件的就绪状态。首次调用前可用它确认服务是否可用。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """按工具名分发调用，返回可直接序列化给模型的结果。"""
        args = dict(arguments or {})

        if name == "search_knowledge":
            result = self.retrieve(
                str(args.get("query", "")),
                top_k=min(int(args.get("top_k", 8) or 8), 30),
            )
            payload = result.to_dict()
            if not self.include_full_text:
                for ev in payload["evidence"]:
                    ev.pop("text", None)
            payload.pop("context", None)
            return payload

        if name == "ask_knowledge":
            result = self.answer(
                str(args.get("query", "")),
                top_k=min(int(args.get("top_k", 10) or 10), 30),
            )
            payload = result.to_dict()
            if not self.include_full_text:
                for ev in payload["evidence"]:
                    ev.pop("text", None)
            return payload

        if name == "explore_entity":
            return self.graph_neighbors(str(args.get("entity", ""))).to_dict()

        if name == "knowledge_status":
            return self.health()

        return {"error": f"未知工具: {name}"}

    # ---- 异步包装 --------------------------------------------------------
    async def aretrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        """``retrieve`` 的异步版本（检索是阻塞 IO，放到线程池执行）。"""
        return await asyncio.to_thread(self.retrieve, query, **kwargs)

    async def aanswer(self, query: str, **kwargs: Any) -> AnswerResult:
        """``answer`` 的异步版本。"""
        return await asyncio.to_thread(self.answer, query, **kwargs)

    async def acall_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await asyncio.to_thread(self.call_tool, name, arguments)


_default_kb: Optional[KnowledgeBase] = None
_default_lock = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    """进程级单例。多处以元件身份使用时共享同一个索引实例。"""
    global _default_kb
    if _default_kb is None:
        with _default_lock:
            if _default_kb is None:
                _default_kb = KnowledgeBase()
    return _default_kb
