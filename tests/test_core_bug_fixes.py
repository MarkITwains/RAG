"""Regression tests for the BM25 / HyDE / GraphRAG / fusion bug fixes."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.schema import NodeWithScore, TextNode

from pcb_rag import fusion, query, security
from pcb_rag.graph_rag import GraphRetriever, KnowledgeGraph, Triple, link_entities
from pcb_rag.graph_semantic import GraphEvidenceRetriever
from pcb_rag.retrieval import PipelineConfig, RetrievalPipeline, build_lexical
from pcb_rag.revisions import RevisionStore


def _item(key, score):
    return NodeWithScore(node=TextNode(id_=key, text=key), score=score)


# ---------------------------------------------------------------- 1. lexical ACL
def _lexical_rows():
    return [
        {
            "id": "private",
            "text": "铜厚 35 μm",
            "tenant_id": "t-a",
            "visibility": "private",
            "acl_groups": "|eng|",
            "revision": "r1",
            "node_level": "child",
            "doc_node_id": "d1",
        },
        {"id": "parent", "text": "铜厚 章节全文", "node_level": "parent", "revision": "r1", "doc_node_id": "d1"},
    ]


def test_lexical_export_keeps_acl_and_revision_fields(tmp_path, monkeypatch):
    cache = tmp_path / "lexical.jsonl"
    monkeypatch.setattr(query, "LEXICAL_ENABLED", True)
    monkeypatch.setattr(query, "LEXICAL_CACHE_PATH", str(cache))
    monkeypatch.setattr(query, "_iter_milvus_rows", lambda _: iter(_lexical_rows()))

    index = query._load_or_build_bm25(object())

    assert [doc.id_ for doc in index.docs] == ["private"]  # parent chunks stay out of BM25
    metadata = index.docs[0].metadata
    for key in ("tenant_id", "visibility", "acl_groups", "revision", "node_level", "doc_node_id"):
        assert key in metadata
    assert query._lexical_cache_schema_ok(cache)


def test_old_lexical_cache_is_rebuilt_when_acl_is_enabled(tmp_path, monkeypatch):
    cache = tmp_path / "lexical.jsonl"
    cache.write_text(json.dumps({"id": "private", "text": "铜厚", "metadata": {}}) + "\n", encoding="utf-8")
    calls = []
    monkeypatch.setenv("LEXICAL_CACHE_TTL_HOURS", "0")
    monkeypatch.setattr(query, "LEXICAL_ENABLED", True)
    monkeypatch.setattr(query, "ACL_ENABLED", True)
    monkeypatch.setattr(query, "LEXICAL_CACHE_PATH", str(cache))
    monkeypatch.setattr(query, "_iter_milvus_rows", lambda _: calls.append(1) or iter(_lexical_rows()))

    index = query._load_or_build_bm25(object())

    assert calls == [1]
    assert index.docs[0].metadata["visibility"] == "private"


def test_lexical_private_chunk_is_denied_to_other_tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(query, "LEXICAL_ENABLED", True)
    monkeypatch.setattr(query, "LEXICAL_CACHE_PATH", str(tmp_path / "lexical.jsonl"))
    monkeypatch.setattr(query, "_iter_milvus_rows", lambda _: iter(_lexical_rows()))
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    index = query._load_or_build_bm25(object())

    outsider = security.Principal(user_id="u", tenant_id="t-b")
    insider = security.Principal(user_id="u", tenant_id="t-a")

    assert security.filter_nodes(index.docs, outsider) == []
    assert len(security.filter_nodes(index.docs, insider)) == 1


# ---------------------------------------------------------------- 2. BM25 run merge
def test_rrf_merge_runs_ignores_raw_score_scale():
    original = [_item("exact", 5.0), _item("other", 4.0)]
    long_query = [_item("noise", 50.0), _item("exact", 40.0)]

    assert [i.node.id_ for i in fusion.merge_scored_runs([original, long_query])][0] == "noise"
    assert [i.node.id_ for i in fusion.rrf_merge_runs([original, long_query])] == ["exact", "noise", "other"]


def test_rrf_merge_single_run_keeps_native_scores():
    out = fusion.rrf_merge_runs([[_item("a", 7.5)]])
    assert out[0].score == 7.5


def test_query2doc_run_does_not_outrank_original_bm25_hit(offline_pipeline, monkeypatch):
    from dataclasses import replace

    fixture = offline_pipeline
    real, low = fixture.docs[1], fixture.docs[0]

    class Lexical:
        def retrieve(self, text):
            if text.startswith("Q2D"):
                return [NodeWithScore(node=low, score=50.0), NodeWithScore(node=real, score=40.0)]
            return [NodeWithScore(node=real, score=5.0)]

    fixture.pipeline.state["bm25_index"] = object()
    fixture.pipeline.config = replace(
        fixture.pipeline.config, lexical_enabled=True, query2doc=True, enhance_all=True, mode="bm25"
    )
    monkeypatch.setattr(query, "LocalBM25Retriever", lambda *args, **kwargs: Lexical())
    monkeypatch.setattr(query, "_expand_query", lambda text: text)
    monkeypatch.setattr(query, "query2doc_expand", lambda text, llm=None, **_: "Q2D " + text)

    result = fixture.pipeline.retrieve("铜厚", top_k=2)

    assert [item.node.id_ for item in result.nodes] == ["real", "low"]


# ---------------------------------------------------------------- 3. bigram adjacency
def test_english_bigrams_only_join_adjacent_words():
    tokens = query._tokenize("IPC-6012 规定焊盘 annular ring")

    assert "annular_ring" in tokens
    assert "ipc-6012_annular" not in tokens


# ---------------------------------------------------------------- 4. DF filter / operators
def test_df_filter_skipped_on_tiny_corpus():
    index = build_lexical([TextNode(id_="a", text="阻抗 控制"), TextNode(id_="b", text="阻抗 测试")])

    result = query.LocalBM25Retriever(index).retrieve("阻抗")

    assert {item.node.id_ for item in result} == {"a", "b"}


def test_unknown_filter_operator_fails_closed():
    retriever = query.LocalBM25Retriever(build_lexical([TextNode(text="x")]))
    unknown = SimpleNamespace(key="k", operator="not-a-real-operator", value=1)

    assert retriever._match_filter({"k": 1}, unknown) is False


def test_any_operator_is_implemented():
    from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter

    if not hasattr(FilterOperator, "ANY"):
        pytest.skip("installed llama-index has no ANY operator")
    retriever = query.LocalBM25Retriever(build_lexical([TextNode(text="x")]))
    condition = MetadataFilter(key="tags", value=["b", "c"], operator=FilterOperator.ANY)

    assert retriever._match_filter({"tags": ["a", "b"]}, condition) is True
    assert retriever._match_filter({"tags": ["a"]}, condition) is False


# ---------------------------------------------------------------- 5/6. HyDE config and errors
def test_pipeline_hyde_flag_overrides_module_switch(offline_pipeline, monkeypatch):
    from dataclasses import replace

    fixture = offline_pipeline
    monkeypatch.setattr(query, "HYDE_ENABLED", False)
    monkeypatch.setattr(query, "QUERY_ENHANCE_ALL", False)
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)

    result = fixture.pipeline.retrieve("铜厚")

    assert result.routes["hyde"] == "ok"
    assert len(fixture.index.calls) == 2 and fixture.index.calls[1].startswith("铜厚\n")


def test_hyde_llm_error_is_reported_as_route_failure(offline_pipeline, monkeypatch):
    from dataclasses import replace

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    monkeypatch.setattr(query, "model_complete", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    result = fixture.pipeline.retrieve("铜厚")

    assert result.routes["hyde"] == "failed:RuntimeError"
    assert "hyde_noop" not in result.degraded


def test_hyde_budget_exhaustion_stops_the_request(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag.usage import UsageBudgetExceeded

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    monkeypatch.setattr(
        query, "model_complete", lambda *args, **kwargs: (_ for _ in ()).throw(UsageBudgetExceeded("budget"))
    )

    with pytest.raises(UsageBudgetExceeded):
        fixture.pipeline.retrieve("铜厚")


def test_legacy_hyde_still_falls_back_on_plain_errors_but_not_on_budget(monkeypatch):
    from pcb_rag.usage import UsageBudgetExceeded

    monkeypatch.setattr(query, "HYDE_ENABLED", True)
    monkeypatch.setattr(query, "model_complete", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    assert query.hyde_expand_query("铜厚", llm=object()) == "铜厚"

    monkeypatch.setattr(query, "model_complete", lambda *args, **kwargs: (_ for _ in ()).throw(UsageBudgetExceeded("b")))
    with pytest.raises(UsageBudgetExceeded):
        query.hyde_expand_query("铜厚", llm=object())


# ---------------------------------------------------------------- 7. short query
def test_long_chinese_query_is_not_short(monkeypatch):
    monkeypatch.setattr(query, "SHORT_QUERY_THRESHOLD", 120)
    long_query = "阻抗控制" * 20 + "，" + "差分走线" * 10 + "，" + "层叠设计" * 5

    assert len(long_query) > 120
    assert query._is_short_query(long_query) is False
    assert query._is_short_query("铜厚") is True


# ---------------------------------------------------------------- 8. entity linking
def test_entity_alias_needs_word_boundary():
    graph = KnowledgeGraph()
    graph.add_triple(Triple("电源完整性", "包含", "去耦电容", aliases=["PI"]))

    assert link_entities("SPI 总线走线要求", graph) == []
    assert "电源完整性" in {name for name, _ in link_entities("PI 设计要求", graph)}


def test_numeric_entity_needs_word_boundary():
    graph = KnowledgeGraph()
    graph.add_triple(Triple("线宽", "最小值", "0.1"))

    assert link_entities("10.1 版本说明", graph) == []
    assert "0.1" in {name for name, _ in link_entities("间距 0.1 是否够", graph)}


# ---------------------------------------------------------------- 9. graph scores
def test_graph_sub_routes_are_fused_by_rank():
    graph = KnowledgeGraph()
    graph.add_triple(Triple("铜厚", "要求", "30 μm", chunk_id="c0", weight=1.0, revision="r1"))
    graph.add_triple(Triple("铜厚", "要求", "35 μm", chunk_id="c1", weight=3.0, revision="r1"))
    graph.add_triple(Triple("铜厚", "测试", "荧光法", chunk_id="c2", weight=2.0, revision="r1"))
    nodes = {key: TextNode(id_=key, text=key, metadata={"revision": "r1"}) for key in ("c0", "c1", "c2", "s1")}

    class Entities:
        records = [{"key": "x"}]

        def search(self, vector, top_k):
            return [{"score": 0.3, "sources": [{"chunk_id": "s1", "doc_id": "", "revision": "r1", "qualifiers": {}}]}]

    embed = SimpleNamespace(get_query_embedding=lambda text: [1.0])
    retriever = GraphEvidenceRetriever(graph, embed, entities=Entities())

    results = retriever.retrieve_sources("铜厚要求", lambda ids: [nodes[i] for i in ids if i in nodes], top_k=2)

    # Heaviest edge first, then the semantic route's best hit instead of more same-entity edges.
    assert [item.node.id_ for item in results] == ["c1", "s1"]


# ---------------------------------------------------------------- 10. aggregated graph nodes
def test_aggregated_graph_nodes_are_not_treated_as_legacy(monkeypatch):
    graph = KnowledgeGraph()
    graph.add_triple(Triple("镀金层", "厚度", "0.05 μm", evidence="私有文档原文", chunk_id="c1"))
    nodes = GraphRetriever(graph).retrieve("镀金层厚度")
    assert nodes
    principal = security.Principal(user_id="u", tenant_id="t-b")

    monkeypatch.setattr(security, "ACL_ENABLED", True)
    assert security.filter_nodes(nodes, principal) == []

    monkeypatch.setattr(security, "ACL_ENABLED", False)
    assert len(security.filter_nodes(nodes, principal)) == len(nodes)


# ---------------------------------------------------------------- 11/12. quotas
def test_quota_parsing_is_tolerant_and_name_bound():
    assert fusion.parse_route_quotas("3,2,") == {"vector": 3, "bm25": 2}
    assert fusion.parse_route_quotas("1,0,4") == {"vector": 1, "hyde": 0, "bm25": 4}
    assert fusion.parse_route_quotas("vector=3, graph=1") == {"vector": 3, "graph": 1}
    assert fusion.parse_route_quotas("3,x") == {}
    assert fusion.quotas_for_routes(["vector", "bm25", "stepback"], {"vector": 1, "hyde": 0, "bm25": 4}, 1) == [
        1,
        4,
        1,
    ]


def test_bad_fusion_quotas_do_not_break_pipeline_config(monkeypatch):
    monkeypatch.setattr(query, "FUSION_QUOTAS_RAW", "3,2,oops")

    assert PipelineConfig.from_env().quotas == ()


def test_pipeline_quotas_follow_route_names(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import retrieval

    fixture = offline_pipeline
    captured = {}

    def fake_fuse(routes, top_n, **options):
        captured.update(options)
        return [node for nodes, _weight in routes for node in nodes][:top_n]

    class Lexical:
        def retrieve(self, text):
            return [NodeWithScore(node=fixture.docs[1], score=1.0)]

    fixture.pipeline.state["bm25_index"] = object()
    fixture.pipeline.config = replace(
        fixture.pipeline.config, lexical_enabled=True, quotas=(("bm25", 5),), quota_min=1
    )
    monkeypatch.setattr(query, "LocalBM25Retriever", lambda *args, **kwargs: Lexical())
    monkeypatch.setattr(retrieval, "quota_rrf_fuse", fake_fuse)

    fixture.pipeline.retrieve("铜厚")

    assert captured["quotas"] == [1, 5]  # vector, bm25


# ---------------------------------------------------------------- 13. refresh
class _WatchedState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sizes = []

    def clear(self):
        super().clear()
        self.sizes.append(len(self))

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.sizes.append(len(self))


def test_refresh_never_exposes_an_empty_state(tmp_path):
    state = _WatchedState(index=object(), revision="legacy")
    pipeline = RetrievalPipeline(
        PipelineConfig(graph_enabled=False), state=state, revision_store=RevisionStore(tmp_path / "s.json")
    )

    pipeline.refresh(force=True)

    assert state.sizes and 0 not in state.sizes
    assert state["index"] is not None


def test_refresh_skips_graph_stamp_when_graph_is_disabled(tmp_path, monkeypatch):
    from pcb_rag import graph_rag

    graph_file = tmp_path / "graph.json"
    graph_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(graph_rag, "GRAPH_PATH", str(graph_file))
    state = _WatchedState(index=object(), revision="legacy")
    pipeline = RetrievalPipeline(
        PipelineConfig(graph_enabled=False), state=state, revision_store=RevisionStore(tmp_path / "s.json")
    )

    pipeline.refresh()
    pipeline.refresh()

    assert state.sizes == []
