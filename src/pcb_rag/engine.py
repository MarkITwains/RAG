"""Structured knowledge-base SDK backed by the independent retrieval pipeline."""

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

    revision: str = "legacy"
    routes: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    usage: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "routes": self.routes,
            "timings": self.timings,
            "budget": self.budget,
            "usage": self.usage,
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

    invalid_citations: List[int] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    revision: str = "legacy"
    routes: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    usage: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invalid_citations": self.invalid_citations,
            "budget": self.budget,
            "revision": self.revision,
            "routes": self.routes,
            "timings": self.timings,
            "usage": self.usage,
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
        pipeline=None,
    ) -> None:
        """
        Args:
            auto_init: 构造时是否立即初始化（False 则首次调用时惰性初始化）
            include_full_text: 证据里是否带完整正文。agent 需要自行摘录时开启；
                只做展示时关掉可显著减少内存与序列化开销。
            include_context: ``retrieve`` 是否同时返回拼装好的上下文。
                只有当 agent 想把整块上下文直接喂给模型时才需要。
        """
        from pcb_rag.retrieval import get_pipeline

        self.pipeline = pipeline or get_pipeline()
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

    def _state(self) -> Dict[str, Any]:
        return self.pipeline.state

    def _state_ready(self) -> bool:
        try:
            return "index" in self._state()
        except Exception:
            return False

    def ensure_ready(self) -> None:
        if self.ready:
            return
        with self._init_lock:
            if not self.ready:
                self.pipeline.initialize()
                self._ready = True

    def health(self) -> Dict[str, Any]:
        """返回各组件就绪状态与降级原因。agent 可据此决定是否先提示用户。"""
        try:
            state = self._state()
        except Exception as exc:
            return {"ready": False, "error": str(exc), "degraded": ["init_failed"]}

        index_ok = state.get("index") is not None
        bm25_ok = state.get("bm25_index") is not None
        rerank_ok = state.get("rerank") is not None
        graph_ok = state.get("graph") is not None

        degraded: List[str] = []
        if not index_ok:
            degraded.append("vector_index_missing")
        if self.pipeline.config.lexical_enabled and not bm25_ok:
            degraded.append("lexical_index_missing")
        if self.pipeline.config.rerank_enabled and not rerank_ok:
            degraded.append("rerank_unavailable")
        if self.pipeline.config.graph_enabled and not graph_ok:
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

    # ---- 内部：NodeWithScore → Evidence --------------------------------
    def _to_evidence(self, item: Any, index: int) -> Optional[Evidence]:
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", None) or {}

        source_path = metadata.get("source_path", "") or ""
        source_name = Path(source_path).stem if source_path else ""

        section = metadata.get("parent_title") or metadata.get("chunk_section") or metadata.get("section") or ""

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
        self, query: str, *, top_k: int = 10, score_threshold: float = 0.0, with_context: Optional[bool] = None
    ) -> RetrievalResult:
        from pcb_rag.query import build_context_bundle
        from pcb_rag.runtime import request_scope

        started = time.monotonic()
        if not query or not query.strip():
            return RetrievalResult(query=query or "", degraded=["empty_query"])
        self.ensure_ready()
        degraded: List[str] = []
        want_context = self.include_context if with_context is None else bool(with_context)
        try:
            with request_scope(self.pipeline.config.timeout) as control:
                nodes = self.pipeline.retrieve_nodes(
                    user_query=query.strip(), top_k=int(top_k), score_threshold=float(score_threshold)
                )
                bundle = build_context_bundle(nodes)
                # Every SDK evidence index matches the optional prompt context.
                evidence = self._bundle_evidence(bundle)
                degraded.extend(control.degraded)
        except Exception as exc:
            return RetrievalResult(
                query=query,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                revision=control.revision,
                routes=dict(control.routes),
                timings=dict(control.timings),
                usage=list(control.usage),
                degraded=list(dict.fromkeys(control.degraded + [f"retrieval_error:{exc}"])),
            )
        return RetrievalResult(
            query=query,
            evidence=evidence,
            context=bundle.context if want_context else None,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            degraded=list(dict.fromkeys(degraded)),
            revision=control.revision,
            routes=dict(control.routes),
            timings=dict(control.timings),
            budget=bundle.budget,
            usage=list(control.usage),
        )

    def _bundle_evidence(self, bundle):
        evidence = []
        for row in bundle.entries:
            metadata = row["metadata"]
            evidence.append(
                Evidence(
                    index=row["index"],
                    source_path=row["source_path"],
                    source_name=row["source_name"],
                    section=row["section"],
                    chunk_id=row["chunk_id"],
                    score=row["score"],
                    tokens=row["tokens"],
                    snippet=row["snippet"],
                    text=row["text"] if self.include_full_text else "",
                    truncated=row["truncated"],
                    doc_type=str(metadata.get("doc_type", "")),
                    source_type=str(metadata.get("source_type", "")),
                )
            )
        return evidence

    def answer(
        self,
        query: str,
        *,
        top_k: int = 10,
        score_threshold: float = 0.0,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> AnswerResult:
        from pcb_rag.conversation import generate_answer_with_history, rewrite_query_with_history
        from pcb_rag.runtime import request_scope

        started = time.monotonic()
        if not query or not query.strip():
            return AnswerResult(query=query or "", answer="", degraded=["empty_query"])
        self.ensure_ready()
        degraded: List[str] = []
        raw_query = query.strip()
        history = [dict(message) for message in history or []]
        with request_scope(self.pipeline.config.timeout) as control:
            try:
                search_query = (
                    rewrite_query_with_history(raw_query, history, self.pipeline.llm) if history else raw_query
                )
                nodes = self.pipeline.retrieve_nodes(
                    user_query=search_query, top_k=int(top_k), score_threshold=float(score_threshold)
                )
            except Exception as exc:
                return AnswerResult(
                    query=raw_query,
                    answer="",
                    revision=control.revision,
                    routes=dict(control.routes),
                    timings=dict(control.timings),
                    usage=list(control.usage),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    degraded=list(dict.fromkeys(control.degraded + [f"retrieval_error:{exc}"])),
                )
            if not nodes:
                return AnswerResult(
                    query=raw_query,
                    answer="根据现有文档未检索到相关内容。",
                    revision=control.revision,
                    routes=dict(control.routes),
                    timings=dict(control.timings),
                    usage=list(control.usage),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    degraded=list(dict.fromkeys(control.degraded + ["no_evidence"])),
                )
            try:
                result = generate_answer_with_history(raw_query, nodes, history, self.pipeline.llm)
                bundle = result["evidence_bundle"]
            except Exception as exc:
                return AnswerResult(
                    query=raw_query,
                    answer="",
                    revision=control.revision,
                    routes=dict(control.routes),
                    timings=dict(control.timings),
                    usage=list(control.usage),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    degraded=list(dict.fromkeys(control.degraded + [f"generation_error:{exc}"])),
                )
            degraded.extend(control.degraded)
            return AnswerResult(
                query=raw_query,
                answer=result["answer"],
                evidence=self._bundle_evidence(bundle),
                citations=result["citations"],
                invalid_citations=result["invalid_citations"],
                budget=bundle.budget,
                revision=control.revision,
                routes=dict(control.routes),
                timings=dict(control.timings),
                usage=list(control.usage),
                elapsed_ms=int((time.monotonic() - started) * 1000),
                degraded=list(dict.fromkeys(degraded)),
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

        from pcb_rag import security
        from pcb_rag.graph_rag import build_neighborhood_text, link_entities, normalize_entity

        normalized = normalize_entity(entity)
        candidates = [normalized] if normalized in graph.nodes else []
        candidates.extend(name for name, _ in link_entities(entity, graph) if name not in candidates)
        if not candidates:
            return GraphNeighborhood(entity=entity, found=False, degraded=[])

        if security.ACL_ENABLED or state.get("active"):
            from pcb_rag.graph_semantic import authorized_graph
            from pcb_rag.runtime import request_scope, run_stage

            try:
                with request_scope(self.pipeline.config.timeout):
                    graph = authorized_graph(
                        graph,
                        lambda ids: run_stage("graph_sources", self.pipeline._fetch, ids, state),
                        lambda nodes: self.pipeline._allowed(nodes, state, parents=True),
                        entities=candidates,
                    )
            except Exception as exc:
                return GraphNeighborhood(
                    entity=entity, found=False, degraded=["graph_source_error:" + type(exc).__name__]
                )

        name = next((candidate for candidate in candidates if candidate in graph.nodes), "")
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
                    "sources": [
                        {
                            key: source.get(key, "")
                            for key in ("doc_id", "chunk_id", "revision", "source_path", "qualifiers")
                        }
                        for source in edge.get("contributions", [])
                    ],
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
        from pcb_rag.runtime import request_scope

        with request_scope(self.pipeline.config.timeout) as control:
            return await asyncio.wait_for(asyncio.to_thread(self.retrieve, query, **kwargs), control.remaining())

    async def aanswer(self, query: str, **kwargs: Any) -> AnswerResult:
        """``answer`` 的异步版本。"""
        from pcb_rag.runtime import request_scope

        with request_scope(self.pipeline.config.timeout) as control:
            return await asyncio.wait_for(asyncio.to_thread(self.answer, query, **kwargs), control.remaining())

    async def acall_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from pcb_rag.runtime import request_scope

        with request_scope(self.pipeline.config.timeout) as control:
            return await asyncio.wait_for(asyncio.to_thread(self.call_tool, name, arguments), control.remaining())


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
