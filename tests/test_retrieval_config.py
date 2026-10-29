"""Environment-controlled retrieval limits must reach final results."""

import pytest

from pcb_rag.graph_rag import KnowledgeGraph, Triple
from pcb_rag.graph_semantic import GraphEvidenceRetriever

pytest.importorskip("llama_index.core")
from llama_index.core.schema import TextNode


def test_graph_top_k_limits_entity_name_hits():
    graph = KnowledgeGraph()
    for index in range(3):
        graph.add_triple(Triple("铜厚", "要求", f"{index + 30} μm", chunk_id=f"c{index}", revision="r1"))
    nodes = [TextNode(id_=f"c{index}", text=f"铜厚 {index + 30} μm", metadata={"revision": "r1"}) for index in range(3)]
    retriever = GraphEvidenceRetriever(graph)

    results = retriever.retrieve_sources("铜厚要求", lambda ids: nodes, top_k=1)

    # Entity-name hits share a score; chunk_id provides deterministic tie-breaking.
    assert [item.node.id_ for item in results] == ["c0"]


def test_graph_hop_two_keeps_second_hop_evidence():
    from pcb_rag.graph_rag import GraphRetriever

    graph = KnowledgeGraph()
    graph.add_triple(Triple("铜厚", "要求", "35 μm", evidence="first"))
    graph.add_triple(Triple("35 μm", "测试方法", "荧光法", evidence="second"))
    graph.add_triple(Triple("测试标准", "规定", "铜厚", evidence="incoming"))
    retriever = GraphRetriever(graph, hops=2)

    passages = retriever.build_passages("铜厚要求")
    text = "\n".join(passage["text"] for passage in passages)

    assert "35 μm --测试方法--> 荧光法" in text
    assert "35 μm --测试方法--> 荧光法（反向）" not in text
    assert "测试标准 --规定--> 铜厚（反向）" in text


def test_shared_graph_evidence_uses_configured_hops(monkeypatch):
    from pcb_rag import graph_semantic

    graph = KnowledgeGraph()
    graph.add_triple(Triple("铜厚", "要求", "35 μm", chunk_id="first", revision="r1"))
    graph.add_triple(Triple("35 μm", "测试方法", "荧光法", chunk_id="second", revision="r1"))
    nodes = [
        TextNode(id_="first", text="铜厚 35 μm", metadata={"revision": "r1"}),
        TextNode(id_="second", text="35 μm 荧光法", metadata={"revision": "r1"}),
    ]
    monkeypatch.setattr(graph_semantic, "GRAPH_HOP", 2)

    results = GraphEvidenceRetriever(graph).retrieve_sources("铜厚要求", lambda ids: nodes, top_k=5)

    assert "second" in {item.node.id_ for item in results}


def test_custom_milvus_embedding_field_is_used_for_ingestion(monkeypatch):
    from pcb_rag import ingest

    monkeypatch.setenv("MILVUS_EMBEDDING_FIELD", "custom_vector")
    assert ingest.embedding_field_name() == "custom_vector"


def _ingest_main_harness(monkeypatch):
    from types import SimpleNamespace

    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    from pcb_rag import ingest

    captured = {}
    monkeypatch.setattr(ingest, "_load_ingest_documents", lambda: [])
    monkeypatch.setattr(ingest, "build_llm", lambda: MockLLM())
    monkeypatch.setattr(ingest, "build_embed_model", lambda: MockEmbedding(embed_dim=8))
    monkeypatch.setattr(ingest, "get_embedding_dim", lambda _: 8)
    monkeypatch.setattr(ingest, "EMBED_BACKEND", "local")
    monkeypatch.setattr(ingest, "GRAPH_RAG_ENABLED", True)
    monkeypatch.setattr(ingest, "COLLECTION", "test")
    monkeypatch.setenv("KB_STATE_PATH", ":memory:")

    class Store:
        def __init__(self, **kwargs):
            captured["store"] = kwargs
            self.client = SimpleNamespace(get_collection_stats=lambda _: {"row_count": 0})

    monkeypatch.setattr(ingest, "MilvusVectorStore", Store)
    monkeypatch.setattr(ingest.VectorStoreIndex, "from_vector_store", lambda **_: SimpleNamespace())
    monkeypatch.setattr(ingest.incremental, "save_manifest", lambda *args, **kwargs: None)
    from pcb_rag import ingestion

    def run(*args, **kwargs):
        captured["run"] = kwargs
        return {"docs": {}, "revision": "test"}

    monkeypatch.setattr(ingestion, "run_ingestion", run)
    return captured


def test_ingest_uses_custom_embedding_field_and_disables_semantic_vectors(monkeypatch):
    from pcb_rag import ingest

    captured = _ingest_main_harness(monkeypatch)
    monkeypatch.setenv("MILVUS_EMBEDDING_FIELD", "custom_vector")
    monkeypatch.setenv("GRAPH_SEMANTIC_ENABLED", "false")

    ingest.main()

    assert captured["store"]["embedding_field"] == "custom_vector"
    assert captured["run"]["graph_vectors"] is None


def test_ingest_incremental_off_requests_safe_full_rebuild(monkeypatch):
    from pcb_rag import ingest

    captured = _ingest_main_harness(monkeypatch)
    monkeypatch.setattr(ingest, "INGEST_INCREMENTAL", False)

    ingest.main()

    assert captured["run"]["force_rebuild"] is True


def test_encoding_repair_can_be_disabled_before_document_loading(tmp_path, monkeypatch):
    from pcb_rag import ingest

    (tmp_path / "document.txt").write_text("text", encoding="utf-8")
    calls = []
    monkeypatch.setattr(ingest, "AUTO_FIX_ENCODING", False)
    monkeypatch.setattr(ingest, "_fix_file_encodings", lambda path: calls.append(path))
    monkeypatch.setattr(ingest.SimpleDirectoryReader, "load_data", lambda self: [])

    assert ingest._load_ingest_documents(str(tmp_path)) == []
    assert calls == []


def test_ingest_plan_only_never_rewrites_source_files(tmp_path, monkeypatch):
    from pcb_rag import ingest

    source = tmp_path / "corpus"
    source.mkdir()
    original = "测试内容".encode("gb18030")
    (source / "document.txt").write_bytes(original)
    monkeypatch.setattr(ingest, "DATA_DIR", str(source))
    monkeypatch.setattr(ingest, "AUTO_FIX_ENCODING", True)
    monkeypatch.setenv("INGEST_PLAN_ONLY", "1")

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def load_data(self):
            from types import SimpleNamespace

            return [SimpleNamespace(id_="doc", text="测试内容", hash="hash", metadata={"file_path": "document.txt"})]

    monkeypatch.setattr(ingest, "SimpleDirectoryReader", Reader)
    ingest.main()

    assert (source / "document.txt").read_bytes() == original


def test_graph_failure_tolerance_accepts_true(monkeypatch):
    from pcb_rag import graph_rag
    from pcb_rag.revisions import pipeline_fingerprints

    monkeypatch.setenv("GRAPH_TOLERATE_CHUNK_FAILURES", "true")
    monkeypatch.setattr(
        graph_rag, "extract_triples_llm", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad JSON"))
    )
    monkeypatch.setattr(graph_rag, "GRAPH_LLM_WORKERS", 1)

    assert graph_rag._extract_triples_concurrent([TextNode(text="broken")], None, mode="llm", strict=True) == []
    assert (
        pipeline_fingerprints({})["graph"] != pipeline_fingerprints({"GRAPH_TOLERATE_CHUNK_FAILURES": "true"})["graph"]
    )


def test_pipeline_uses_configured_fusion_rrf_k(monkeypatch):
    from pcb_rag import query
    from pcb_rag.retrieval import PipelineConfig

    monkeypatch.setattr(query, "FUSION_RRF_K", 73)

    assert PipelineConfig.from_env().rrf_k == 73


def test_hyde_wait_budget_skips_only_slow_hyde_route(offline_pipeline, monkeypatch):
    import threading
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    started, release = threading.Event(), threading.Event()

    def slow_hyde(text, llm=None, **_):
        started.set()
        release.wait(1)
        return text

    monkeypatch.setattr(query, "hyde_expand_query", slow_hyde)
    monkeypatch.setenv("HYDE_WAIT_BUDGET", "0.01")
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    try:
        result = fixture.pipeline.retrieve("铜厚")
    finally:
        release.set()

    assert {item.node.id_ for item in result.nodes} == {"real", "low"}
    assert result.routes["hyde"] == "wait_timeout"
    assert "hyde_wait_timeout" in result.degraded
    assert started.is_set()


def test_hyde_wait_budget_keeps_fast_hyde_route(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    monkeypatch.setenv("HYDE_WAIT_BUDGET", "30")
    monkeypatch.setattr(query, "hyde_expand_query", lambda text, llm=None, **_: text + "\n关键事实：测试")

    result = fixture.pipeline.retrieve("铜厚")

    assert result.routes["hyde"] == "ok"
    assert "hyde_wait_timeout" not in result.degraded


def test_hyde_noop_does_not_duplicate_original_vector_route(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    monkeypatch.setattr(query, "hyde_expand_query", lambda text, llm=None, **_: text)

    result = fixture.pipeline.retrieve("铜厚")

    assert fixture.index.calls == ["铜厚"]
    assert result.routes["hyde"] == "skipped_noop"
    assert "hyde_noop" in result.degraded


def test_hyde_wait_budget_accepts_route_completed_before_earlier_job(offline_pipeline, monkeypatch):
    import time
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline

    def rows(text):
        if text == "铜厚":
            time.sleep(0.05)
            return []
        return [fixture.score(fixture.docs[1])]

    fixture.index.rows = rows
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True)
    monkeypatch.setenv("HYDE_WAIT_BUDGET", "0.01")
    monkeypatch.setattr(query, "hyde_expand_query", lambda text, llm=None, **_: text + "\nHYPOTHETICAL")

    result = fixture.pipeline.retrieve("铜厚")

    assert fixture.index.calls == ["铜厚", "铜厚\nHYPOTHETICAL"]
    assert result.routes["hyde"] == "ok"
    assert [item.node.id_ for item in result.nodes] == ["real"]
    assert "hyde_wait_timeout" not in result.degraded


def test_hyde_wait_budget_does_not_mask_request_deadline(offline_pipeline, monkeypatch):
    import threading
    from dataclasses import replace
    from pcb_rag import query
    from pcb_rag.runtime import DeadlineExceeded

    fixture = offline_pipeline
    release = threading.Event()
    fixture.pipeline.config = replace(fixture.pipeline.config, hyde=True, timeout=0.01)
    monkeypatch.setenv("HYDE_WAIT_BUDGET", "30")

    def slow_hyde(text, llm=None, **_):
        release.wait(1)
        return text

    monkeypatch.setattr(query, "hyde_expand_query", slow_hyde)
    try:
        with pytest.raises(DeadlineExceeded):
            fixture.pipeline.retrieve("铜厚")
    finally:
        release.set()


def test_multi_query_merges_runs_by_best_score(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, multi_expand=True)
    monkeypatch.setattr(query, "_build_multi_expand_queries", lambda text, num_queries=3, llm=None: [text, "变体"])
    fixture.index.rows = lambda text: [fixture.score(fixture.node(text), 0.99 if text == "变体" else 0.1)]

    result = fixture.pipeline.retrieve("原始", top_k=2)

    assert [item.node.id_ for item in result.nodes] == ["变体", "原始"]


def test_query2doc_is_a_lexical_expansion(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    lexical = type("Lexical", (), {"calls": []})()
    lexical.retrieve = lambda text: lexical.calls.append(text) or []
    fixture.pipeline.state["bm25_index"] = object()
    fixture.pipeline.config = replace(fixture.pipeline.config, lexical_enabled=True, query2doc=True, enhance_all=True)
    monkeypatch.setattr(query, "LocalBM25Retriever", lambda *args, **kwargs: lexical)
    monkeypatch.setattr(query, "query2doc_expand", lambda text, llm=None, **_: "Q2D " + text)

    fixture.pipeline.retrieve("铜厚")

    assert "Q2D 铜厚" in lexical.calls
    assert all(not text.startswith("Q2D") for text in fixture.index.calls)


def test_multipath_receives_distinct_queries_and_route_weights(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import query

    fixture = offline_pipeline
    captured = {}

    class MultiPath:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def retrieve(self, query, top_k, expand_queries, sparse_expand_queries):
            captured["dense_queries"] = expand_queries
            captured["sparse_queries"] = sparse_expand_queries
            return []

    fixture.pipeline.state["bm25_index"] = object()
    fixture.pipeline.config = replace(
        fixture.pipeline.config, multipath=True, lexical_enabled=True, routing=True, mode="production"
    )
    monkeypatch.setattr(query, "MultiPathRetriever", MultiPath)
    monkeypatch.setattr(query, "_expand_query", lambda text: text + " SYNONYM")
    monkeypatch.setattr(
        query,
        "understand_query",
        lambda *args, **kwargs: {
            "vector_weight": 0.2,
            "bm25_weight": 0.8,
            "num_queries": 3,
            "need_decompose": False,
            "need_step_back": False,
        },
    )

    fixture.pipeline.retrieve("铜厚")

    assert captured["dense_queries"] == ["铜厚"]
    assert captured["sparse_queries"] == ["铜厚 SYNONYM"]
    assert captured["dense_weight"] == 0.2
    assert captured["sparse_weight"] == 0.8


def test_multipath_dense_only_applies_native_score_threshold(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from llama_index.core.schema import NodeWithScore

    fixture = offline_pipeline

    class Dense:
        def _retrieve(self, bundle):
            return [NodeWithScore(node=fixture.docs[1], score=0.5)]

    fixture.index.as_retriever = lambda **kwargs: Dense()
    fixture.pipeline.state["bm25_index"] = fixture.build_lexical([])
    fixture.pipeline.config = replace(fixture.pipeline.config, multipath=True, lexical_enabled=True)

    result = fixture.pipeline.retrieve("语义查询", score_threshold=0.8)

    assert result.nodes == []
    assert "score_threshold_ignored_for_rrf" not in result.degraded


def test_acl_only_empty_result_does_not_relax_business_filters(offline_pipeline, monkeypatch):
    from pcb_rag import security

    fixture = offline_pipeline
    fixture.index.rows = []
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    monkeypatch.setenv("ACL_STRICT_LEGACY", "1")

    result = fixture.pipeline.retrieve("无匹配")

    assert fixture.index.calls == ["无匹配"]
    assert "business_filter_relaxed" not in result.degraded


def test_empty_business_filter_result_is_relaxed_once(offline_pipeline):
    fixture = offline_pipeline
    fixture.index.rows = lambda text: [] if len(fixture.index.calls) == 1 else [fixture.score(fixture.docs[1])]

    result = fixture.pipeline.retrieve("vendor=missing 铜厚")

    assert fixture.index.calls == ["铜厚", "铜厚"]
    assert "business_filter_relaxed" in result.degraded
    assert [item.node.id_ for item in result.nodes] == ["real"]


def test_pipeline_llm_compression_uses_pipeline_llm(offline_pipeline, monkeypatch):
    from dataclasses import replace
    from pcb_rag import compression

    fixture = offline_pipeline
    fixture.pipeline.config = replace(fixture.pipeline.config, compression=True)
    monkeypatch.setattr(compression, "COMPRESSION_ENABLED", True)
    monkeypatch.setattr(compression, "COMPRESSION_MODE", "llm")

    fixture.pipeline.retrieve("铜厚")

    assert len(fixture.llm._prompts) == 2
    assert all("资料压缩助手" in prompt for prompt in fixture.llm._prompts)


@pytest.mark.local_model
def test_colbert_maxsim_excludes_padding():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from pcb_rag.query import ColBERTReranker

    scorer = SimpleNamespace()
    query_embedding = torch.tensor([[1.0, 0.0]])
    document_embedding = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    query_mask = torch.tensor([1])
    document_mask = torch.tensor([1, 0])

    assert ColBERTReranker._maxsim_score(scorer, query_embedding, document_embedding) == 1.0
    assert (
        ColBERTReranker._maxsim_score(
            scorer, query_embedding, document_embedding, query_mask=query_mask, doc_mask=document_mask
        )
        == 0.0
    )
    assert (
        ColBERTReranker._maxsim_score(scorer, query_embedding, document_embedding, doc_mask=torch.tensor([0, 0])) == 0.0
    )
