"""Shared retrieval pipeline for HTTP, SDK, CLI and production evaluation.

Models, stores and configuration belong to a pipeline instance. This module
never imports the HTTP adapter and can be used without constructing an app.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from pcb_rag.context_budget import dedupe_by_key
from pcb_rag.fusion import (
    merge_scored_runs,
    normalize_route_quotas,
    parse_route_quotas,
    quota_rrf_fuse,
    quotas_for_routes,
    rrf_merge_runs,
    weighted_rrf_fuse,
)
from pcb_rag.revisions import RevisionStore, fingerprint, node_visible, state_path
from pcb_rag.runtime import IO_EXECUTOR, DeadlineExceeded, await_work, request_scope, run_stage
from pcb_rag.security import current_principal, filter_nodes


def _key(item):
    node = getattr(item, "node", item)
    return str(getattr(node, "node_id", None) or getattr(node, "id_", None) or id(node))


@dataclass(frozen=True)
class PipelineConfig:
    recall_k: int = 200
    rerank_top_n: int = 10
    rerank_enabled: bool = True
    lexical_enabled: bool = True
    graph_enabled: bool = False
    graph_top_k: int = 5
    graph_weight: float = 0.35
    routing: bool = True
    hyde: bool = False
    query2doc: bool = False
    multi_expand: bool = False
    num_queries: int = 3
    enhance_all: bool = False
    multipath: bool = False
    colbert: bool = False
    fusion_weights: tuple[float, float] = (1.0, 1.0)
    rrf_k: int = 40
    quota_enabled: bool = True
    quota_min: int = 1
    # 按路名绑定的保底配额：((路名, 条数), ...)；也接受 dict 或旧的位置式整数
    quotas: tuple = ()
    dense_top_k: int = 200
    sparse_top_k: int = 200
    multipath_candidates: int = 200
    multipath_rrf_k: int = 60
    hyde_weight: float = 0.7
    stepback_weight: float = 0.5
    expand: bool = True
    expand_neighbors: int = 1
    expand_parent: bool = True
    expand_max_extra: int = 5
    compression: bool = False
    timeout: float = 120.0
    collection: str = "pcb_kb"
    mode: str = "production"

    @classmethod
    def from_env(cls, **overrides):
        from pcb_rag import query as q

        names = {
            "recall_k": "RECALL_TOP_K",
            "rerank_top_n": "RERANK_TOP_N",
            "rerank_enabled": "RERANK_ENABLED",
            "lexical_enabled": "LEXICAL_ENABLED",
            "graph_enabled": "GRAPH_RAG_ENABLED",
            "graph_top_k": "GRAPH_TOP_K",
            "graph_weight": "GRAPH_WEIGHT",
            "routing": "QUERY_ROUTING_ENABLED",
            "hyde": "HYDE_ENABLED",
            "query2doc": "QUERY2DOC_ENABLED",
            "multi_expand": "MULTI_EXPAND_ENABLED",
            "num_queries": "FUSION_NUM_QUERIES",
            "enhance_all": "QUERY_ENHANCE_ALL",
            "multipath": "MULTIPATH_ENABLED",
            "colbert": "COLBERT_RERANK_ENABLED",
            "fusion_weights": "FUSION_WEIGHTS",
            "rrf_k": "FUSION_RRF_K",
            "quota_enabled": "FUSION_QUOTA_ENABLED",
            "quota_min": "FUSION_MIN_PER_ROUTE",
            "hyde_weight": "HYDE_ROUTE_WEIGHT",
            "dense_top_k": "MULTIPATH_DENSE_TOP_K",
            "sparse_top_k": "MULTIPATH_SPARSE_TOP_K",
            "multipath_candidates": "MULTIPATH_RERANK_CANDIDATES",
            "multipath_rrf_k": "MULTIPATH_RRF_K",
            "stepback_weight": "STEPBACK_ROUTE_WEIGHT",
            "expand": "CHUNK_EXPAND_ENABLED",
            "expand_neighbors": "CHUNK_EXPAND_NEIGHBORS",
            "expand_parent": "CHUNK_EXPAND_PARENT",
            "compression": "COMPRESSION_ENABLED",
            "collection": "COLLECTION",
        }
        values = {key: getattr(q, name) for key, name in names.items() if hasattr(q, name)}
        values.update(
            timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", "120")),
            expand_max_extra=int(os.getenv("CHUNK_EXPAND_MAX_EXTRA", "5")),
        )
        raw_quotas = str(getattr(q, "FUSION_QUOTAS_RAW", ""))
        if raw_quotas:
            # 容错解析：空项跳过，写错只告警并退回均分，不让 get_pipeline() 起不来
            values["quotas"] = tuple(parse_route_quotas(raw_quotas).items())
        values.update(overrides)
        return cls(**values)

    @property
    def fingerprint(self):
        return fingerprint(asdict(self))[:24]


@dataclass
class PipelineRun:
    nodes: list[Any] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    routes: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    revision: str = "legacy"


def read_nodes(path):
    from llama_index.core.schema import TextNode

    with Path(path).open(encoding="utf-8") as stream:
        return [
            TextNode(
                text=row["text"],
                id_=row["id"],
                metadata=row.get("metadata", {}),
                excluded_embed_metadata_keys=list(row.get("metadata", {})),
                excluded_llm_metadata_keys=list(row.get("metadata", {})),
            )
            for line in stream
            if line.strip()
            for row in [json.loads(line)]
        ]


def build_lexical(nodes):
    from collections import Counter
    from pcb_rag.query import _Bm25Index, _tokenize

    docs = [node for node in nodes if node.metadata.get("node_level") != "parent"]
    tfs, df, inverted, total = [], Counter(), {}, 0
    for index, node in enumerate(docs):
        tf = Counter(_tokenize(node.text))
        tfs.append(tf)
        total += sum(tf.values())
        for term in tf:
            df[term] += 1
            inverted.setdefault(term, []).append(index)
    return _Bm25Index(docs=docs, tfs=tfs, df=df, avgdl=total / max(1, len(docs)), inverted=inverted)


class RetrievalPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        state: dict | None = None,
        source_loader: Callable | None = None,
        revision_store: RevisionStore | None = None,
    ):
        self.config = config or PipelineConfig.from_env()
        self.state = state if state is not None else {}
        self.source_loader = source_loader
        self.revisions = revision_store or RevisionStore(state_path(self.config.collection))
        self._lock = threading.RLock()

    @property
    def llm(self):
        return self.state.get("llm")

    def initialize(self):
        with self._lock:
            if self.state.get("index") is not None:
                return
            from pcb_rag import query as q
            from pcb_rag.api_clients import assert_api_only, build_llm

            assert_api_only()
            llm = self.state.get("llm") or build_llm()
            index = q.build_index(
                getattr(llm, "model", "default"),
                llm=llm,
                collection=self.config.collection,
                embed_model=self.state.get("embed_model"),
            )
            self.state.update(
                index=index,
                llm=llm,
                embed_model=getattr(index, "_embed_model", None),
                recall_k=self.config.recall_k,
                graph=None,
                graph_retriever=None,
            )
            self.state["rerank"] = q._try_build_reranker() if self.config.rerank_enabled else None
            self.state["colbert_reranker"] = q._try_build_colbert_reranker() if self.config.colbert else None
            self.state["bm25_index"] = (
                q._load_or_build_bm25(index.storage_context.vector_store) if self.config.lexical_enabled else None
            )
            self.refresh(force=True)

    def refresh(self, *, force=False):
        """Atomically switch all immutable artifacts to one published revision."""
        with self._lock:
            active = self.revisions.read().get("active", {})
            revision = active.get("revision", "legacy")
            if not force and self.state.get("revision") == revision:
                # Legacy graph-only deployments also need hot reload. With the
                # graph route disabled the stamp is never stored, so comparing it
                # would force a full refresh on every request.
                if active or not self.config.graph_enabled:
                    return
                from pcb_rag.graph_rag import GRAPH_PATH

                stamp = Path(GRAPH_PATH).stat().st_mtime_ns if Path(GRAPH_PATH).exists() else None
                if stamp == self.state.get("graph_stamp"):
                    return
            from pcb_rag.graph_rag import GRAPH_PATH, KnowledgeGraph

            updated = dict(self.state, active=active, revision=revision)
            artifacts = active.get("artifacts", {})
            if artifacts.get("chunks"):
                nodes = read_nodes(artifacts["chunks"])
                updated["source_nodes"] = {node.id_: node for node in nodes}
                updated["bm25_index"] = build_lexical(nodes) if self.config.lexical_enabled else None
            if active.get("collection") and active["collection"] != updated.get("collection", self.config.collection):
                from pcb_rag.query import build_index

                updated["index"] = build_index(
                    getattr(self.llm, "model", "default"),
                    llm=self.llm,
                    collection=active["collection"],
                    embed_model=updated.get("embed_model"),
                )
            updated["collection"] = active.get("collection", self.config.collection)
            if self.config.graph_enabled:
                path = artifacts.get("graph") or GRAPH_PATH
                updated["graph"] = KnowledgeGraph.load(path)
                updated["graph_stamp"] = Path(path).stat().st_mtime_ns if Path(path).exists() else None
                from pcb_rag.graph_semantic import GraphEvidenceRetriever

                updated["graph_retriever"] = (
                    GraphEvidenceRetriever.load(
                        updated["graph"],
                        artifacts.get("graph_vectors") or str(path) + ".vectors.json",
                        updated.get("embed_model"),
                        client=getattr(
                            getattr(getattr(updated.get("index"), "storage_context", None), "vector_store", None),
                            "client",
                            None,
                        ),
                    )
                    if updated["graph"]
                    else None
                )
            # ``updated`` starts as a copy of the current state, so a plain update
            # replaces every key in one step. clear() + update() left a window in
            # which a concurrent request saw an empty state (KeyError: 'index').
            self.state.update(updated)

    def _fetch(self, ids, state):
        if self.source_loader:
            return self.source_loader(ids)
        if "source_nodes" in state:
            return [copy.deepcopy(state["source_nodes"][key]) for key in ids if key in state["source_nodes"]]
        from pcb_rag.sources import fetch_source_nodes

        return fetch_source_nodes(state["index"].storage_context.vector_store, ids)

    def _allowed(self, nodes, state, *, parents=False):
        result = filter_nodes(nodes, current_principal())
        return [
            item
            for item in result
            if (parents or getattr(item, "node", item).metadata.get("node_level") != "parent")
            and node_visible(getattr(item, "node", item).metadata, state.get("active", {}))
        ]

    def _retrieve(self, user_query, top_k, score_threshold, control, *, fallback=False):
        from pcb_rag import query as q
        from pcb_rag.security import build_acl_filters, merge_filters

        self.refresh()
        with self._lock:
            # Snapshot under the lock so one request never mixes artifacts from
            # two revisions while another thread is publishing.
            state = dict(self.state)
        control.revision = state.get("revision", "legacy")
        cfg = self.config
        clean, business_filters = q._extract_query_filters(user_query, include_acl=False)
        filters = merge_filters(business_filters, build_acl_filters(current_principal()))
        if fallback:
            business_filters = None
            filters = build_acl_filters(current_principal())
        # Push published revision and child constraints down to Milvus when the
        # new metadata schema exists. The in-memory check still runs before any
        # external reranker and again before returning evidence.
        if state.get("active"):
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

            revisions = list({doc["revision"] for doc in state["active"]["docs"].values()})
            filters = (
                merge_filters(
                    filters,
                    MetadataFilters(
                        filters=[
                            MetadataFilter(key="revision", value=revisions, operator=FilterOperator.IN),
                            MetadataFilter(key="node_level", value="child"),
                        ]
                    ),
                )
                if revisions
                else filters
            )
        weights, queries, lexical_queries = cfg.fusion_weights, [clean], [q._expand_query(clean)]
        understanding = q.understand_query(clean, llm=state.get("llm")) if cfg.routing else {}
        if understanding:
            weights = (understanding["vector_weight"], understanding["bm25_weight"])
        if cfg.multi_expand:
            n = understanding.get("num_queries", cfg.num_queries)
            queries = q._build_multi_expand_queries(clean, num_queries=n, llm=state.get("llm"))
            lexical_queries = q._build_multi_expand_queries(lexical_queries[0], num_queries=n, llm=state.get("llm"))
        if understanding.get("need_decompose"):
            extra = run_stage("decompose", q.decompose_query, clean, llm=self.llm)
            queries = list(dict.fromkeys(queries + extra))
            lexical_queries = list(dict.fromkeys(lexical_queries + extra))
        routes, jobs, groups = [], [], {}
        vector = state["index"].as_retriever(
            similarity_top_k=cfg.dense_top_k if cfg.multipath else cfg.recall_k, filters=filters
        )
        lexical = (
            q.LocalBM25Retriever(
                state["bm25_index"],
                similarity_top_k=cfg.sparse_top_k if cfg.multipath else cfg.recall_k,
                filters=filters,
            )
            if state.get("bm25_index")
            else None
        )

        def submit(name, weight, function, *args):
            started = time.monotonic()
            try:
                jobs.append((name, weight, IO_EXECUTOR.submit(function, *args), started))
            except Exception as error:
                control.routes[name] = "failed:" + type(error).__name__
                control.degraded.append(name + "_failed:" + type(error).__name__)

        multipath = cfg.multipath and lexical is not None and cfg.mode == "production"
        if multipath:
            retriever = q.MultiPathRetriever(
                dense_retriever=vector,
                sparse_retriever=lexical,
                dense_top_k=cfg.dense_top_k,
                sparse_top_k=cfg.sparse_top_k,
                enable_colbert=False,
                rerank_candidates=cfg.multipath_candidates,
                rrf_k=cfg.multipath_rrf_k,
                dense_weight=float(weights[0]),
                sparse_weight=float(weights[1]),
            )
            submit(
                "multipath",
                1.0,
                lambda: retriever.retrieve(
                    clean,
                    top_k=cfg.multipath_candidates,
                    expand_queries=queries,
                    sparse_expand_queries=lexical_queries,
                ),
            )
        elif cfg.mode != "bm25":
            for index, value in enumerate(queries):
                submit(f"vector:{index}", float(weights[0]), vector.retrieve, value)
        if not multipath and cfg.lexical_enabled and cfg.mode != "vector":
            if lexical:
                for index, value in enumerate(lexical_queries):
                    submit(f"bm25:{index}", float(weights[1]), lexical.retrieve, value)
            else:
                control.routes["bm25"] = "unavailable"
                control.degraded.append("lexical_index_missing")
        elif not multipath:
            control.routes["bm25"] = "disabled"
        if cfg.graph_enabled:
            graph = state.get("graph_retriever")
            if graph:

                def graph_candidates():
                    return graph.retrieve_sources(
                        clean,
                        lambda ids: self._fetch(ids, state),
                        top_k=cfg.graph_top_k,
                        allowed=lambda nodes: self._allowed(nodes, state),
                    )

                submit("graph", cfg.graph_weight, graph_candidates)
                if not graph.semantic_ready:
                    control.degraded.append("graph_semantic_unavailable")
            else:
                control.routes["graph"] = "unavailable"
                control.degraded.append("graph_unavailable")
        else:
            control.routes["graph"] = "disabled"

        # Enhanced routes use the explicitly injected LLM and participate before
        # reranking. Query2Doc, decomposition and Step-back retain separate flags.
        def enhanced_nodes(retriever, expander):
            # The pipeline already decided from cfg.hyde / cfg.query2doc /
            # cfg.enhance_all that this route runs, so the module-level env
            # switches must not veto it (force=True). strict=True lets LLM
            # failures reach the route status instead of looking like a no-op.
            enhanced = expander(clean, llm=self.llm, force=True, strict=True)
            if not enhanced or enhanced.strip() == clean:
                return None
            return retriever.retrieve(enhanced)

        enhance = cfg.enhance_all or q._is_short_query(clean)
        if cfg.hyde and enhance:
            submit(
                "hyde",
                float(weights[0]) * cfg.hyde_weight,
                lambda: enhanced_nodes(vector, q.hyde_expand_query),
            )
            hyde_deadline = time.monotonic() + max(0.0, float(os.getenv("HYDE_WAIT_BUDGET", "30")))
        if cfg.query2doc and enhance:
            if lexical is not None and cfg.lexical_enabled and cfg.mode != "vector":
                submit(
                    "query2doc",
                    float(weights[1]),
                    lambda: enhanced_nodes(lexical, q.query2doc_expand),
                )
            else:
                control.routes["query2doc"] = "unavailable"
                control.degraded.append("query2doc_lexical_unavailable")
        if understanding.get("need_step_back"):
            submit(
                "stepback",
                float(weights[0]) * cfg.stepback_weight,
                lambda: vector.retrieve(q.step_back_query(clean, llm=self.llm)),
            )
        for name, weight, future, started in jobs:
            try:
                if name == "hyde":
                    try:
                        nodes = await_work(future, control, timeout=max(0.0, hyde_deadline - time.monotonic()))
                    except DeadlineExceeded:
                        if control.deadline - time.monotonic() <= 0:
                            raise
                        control.routes[name] = "wait_timeout"
                        control.degraded.append("hyde_wait_timeout")
                        continue
                else:
                    nodes = await_work(future, control)
                if nodes is None and name in {"hyde", "query2doc"}:
                    control.routes[name] = "skipped_noop"
                    control.degraded.append(name + "_noop")
                    continue
                nodes = self._allowed(copy.deepcopy(nodes), state)
                nodes = dedupe_by_key(nodes, _key)
                group_name = "bm25" if name == "query2doc" else name.split(":")[0]
                group = groups.setdefault(group_name, {"runs": [], "weight": weight})
                group["runs"].append(nodes)
                control.routes[name] = "ok" if nodes else "empty"
            except DeadlineExceeded:
                for _, _, pending, _ in jobs:
                    pending.cancel()
                raise
            except Exception as error:
                from pcb_rag.usage import UsageBudgetExceeded

                if isinstance(error, UsageBudgetExceeded):
                    # Same contract as rerank: an exhausted budget stops the
                    # request instead of being recorded as one failed route.
                    for _, _, pending, _ in jobs:
                        pending.cancel()
                    raise
                control.routes[name] = "failed:" + type(error).__name__
                control.degraded.append(name + "_failed:" + type(error).__name__)
            finally:
                control.timings[name] = (time.monotonic() - started) * 1000
        # Raw BM25 scores grow with query length, so runs from different
        # queries (multi-expand, decomposition, Query2Doc) are merged by rank.
        # Graph evidence already carries rank-fused scores.
        rank_scored = {"graph"}
        ranked_groups = []
        for name, group in groups.items():
            if name == "bm25" and len(group["runs"]) > 1:
                merged = rrf_merge_runs(group["runs"], top_n=cfg.recall_k * 2, rrf_k=cfg.rrf_k)
                rank_scored.add(name)
            else:
                merged = merge_scored_runs(group["runs"], top_n=cfg.recall_k * 2)
            ranked_groups.append((name, merged, group["weight"]))
        ranked_groups = [group for group in ranked_groups if group[1]]
        routes = [(nodes, weight) for _name, nodes, weight in ranked_groups]
        if not routes:
            candidates = []
            score_kind = "native"
        elif len(routes) == 1:
            candidates = routes[0][0][: cfg.recall_k]
            group_name = ranked_groups[0][0]
            if group_name == "multipath":
                score_kind = getattr(retriever, "last_score_kind", "rrf")
            else:
                score_kind = "rrf" if group_name in rank_scored else "native"
        else:
            fuse = quota_rrf_fuse if cfg.quota_enabled else weighted_rrf_fuse
            options = {"min_per_route": cfg.quota_min} if cfg.quota_enabled else {}
            if cfg.quota_enabled:
                # Bind quotas by route name: empty or failed routes are dropped
                # above, so positions shift between requests.
                quotas = quotas_for_routes(
                    [name for name, _nodes, _weight in ranked_groups],
                    normalize_route_quotas(cfg.quotas),
                    cfg.quota_min,
                )
                if quotas is not None:
                    options["quotas"] = quotas
            candidates = fuse(routes, cfg.recall_k, rrf_k=cfg.rrf_k, **options)
            score_kind = "rrf"
        if not candidates and business_filters and not fallback:
            control.degraded.append("business_filter_relaxed")
            return self._retrieve(user_query, top_k, score_threshold, control, fallback=True)
        rerank = state.get("rerank") if cfg.rerank_enabled else None
        if candidates and cfg.colbert and state.get("colbert_reranker") is not None:
            try:
                ranked = run_stage(
                    "colbert", state["colbert_reranker"].rerank, clean, [(_key(item), item.node) for item in candidates]
                )
                candidates = [
                    q.NodeWithScore(node=node, score=score) for _, node, score in ranked[: max(top_k, cfg.rerank_top_n)]
                ]
                score_kind = "rerank"
                control.routes["colbert"] = "ok"
            except DeadlineExceeded:
                raise
            except Exception as error:
                control.routes["colbert"] = "failed:" + type(error).__name__
                control.degraded.append("colbert_failed:" + type(error).__name__)
        elif candidates and rerank:
            try:
                candidates = run_stage(
                    "rerank", rerank._postprocess_nodes, candidates, query_bundle=q.QueryBundle(query_str=clean)
                )
                # API rerank adapters may return fallback nodes without raising.
                if any(getattr(item, "node", item).metadata.get("rerank_fallback") for item in candidates):
                    control.degraded.append("rerank_fallback")
                    control.routes["rerank"] = "fallback"
                else:
                    score_kind = "rerank"
                    control.routes["rerank"] = "ok"
            except DeadlineExceeded:
                raise
            except Exception as error:
                from pcb_rag.rerank_control import RerankStopped
                from pcb_rag.usage import UsageBudgetExceeded

                if isinstance(error, (RerankStopped, UsageBudgetExceeded)):
                    raise
                control.routes["rerank"] = "failed:" + type(error).__name__
                control.degraded.append("rerank_failed:" + type(error).__name__)
        elif cfg.rerank_enabled and not rerank:
            control.routes["rerank"] = "unavailable"
            control.degraded.append("rerank_unavailable")
        else:
            control.routes["rerank"] = "skipped_empty" if cfg.rerank_enabled else "disabled"
        threshold = 0 if score_kind == "rrf" else score_threshold
        if score_kind == "rrf" and score_threshold:
            control.degraded.append("score_threshold_ignored_for_rrf")
        selected = [item for item in candidates if float(item.score or 0) >= threshold][:top_k]
        if cfg.expand and selected:
            from pcb_rag.sources import expand_sources

            try:
                selected = expand_sources(
                    selected,
                    lambda ids: self._fetch(ids, state),
                    neighbors=cfg.expand_neighbors,
                    parents=cfg.expand_parent,
                    max_extra=cfg.expand_max_extra,
                    allowed=lambda nodes: self._allowed(nodes, state, parents=True),
                )
            except Exception as error:
                control.degraded.append("context_expansion_failed:" + type(error).__name__)
        if cfg.compression:
            from pcb_rag import compression

            if compression.COMPRESSION_MODE in {"llm", "hybrid"} and self.llm is None:
                control.degraded.append("compression_llm_unavailable")
            selected = compression.compress_pipeline(copy.deepcopy(selected), clean, llm=self.llm)
        selected = self._allowed(selected, state, parents=True)
        control.remaining()
        return PipelineRun(
            selected,
            list(dict.fromkeys(control.degraded)),
            dict(control.routes),
            dict(control.timings),
            state.get("revision", "legacy"),
        )

    def retrieve(self, user_query: str, top_k: int = 5, score_threshold: float = 0.0) -> PipelineRun:
        if not user_query.strip() or top_k <= 0:
            return PipelineRun(degraded=["empty_query"])
        self.initialize()
        with request_scope(self.config.timeout) as control:
            return self._retrieve(user_query.strip(), top_k, score_threshold, control)

    def retrieve_nodes(self, user_query: str, top_k: int = 5, score_threshold: float = 0.0):
        return self.retrieve(user_query, top_k, score_threshold).nodes

    def with_config(self, **overrides):
        return RetrievalPipeline(
            replace(self.config, **overrides),
            state=dict(self.state),
            source_loader=self.source_loader,
            revision_store=self.revisions,
        )


_default: RetrievalPipeline | None = None
_default_lock = threading.Lock()


def get_pipeline() -> RetrievalPipeline:
    global _default
    with _default_lock:
        if _default is None:
            _default = RetrievalPipeline()
        return _default
