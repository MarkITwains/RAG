"""Configuration edge cases that affect artifact integrity and recovery."""

from types import SimpleNamespace

import os
import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.schema import TextNode

from pcb_rag import ingest


def _chunks(splitter, docs):
    original = ingest.HierarchicalStructureSplitter
    ingest.HierarchicalStructureSplitter = lambda **kwargs: splitter
    try:
        return ingest._parent_child_parse(docs)
    finally:
        ingest.HierarchicalStructureSplitter = original


def _doc():
    return SimpleNamespace(id_="doc", text="source", metadata={"source_path": "a.txt"})


def test_drop_quality_rebuilds_parent_and_adjacency(monkeypatch):
    chunks = [
        ("keep A", {"parent_idx": 0, "_parent_text": "parent A"}),
        ("drop B", {"parent_idx": 0, "_parent_text": "parent A"}),
        ("keep C", {"parent_idx": 0, "_parent_text": "parent A"}),
        ("drop D", {"parent_idx": 1, "_parent_text": "parent B"}),
    ]
    monkeypatch.setattr(ingest, "CHUNK_QUALITY_MODE", "drop")
    monkeypatch.setattr(
        ingest,
        "assess_chunk_quality",
        lambda text: {
            "is_low_value": text.startswith("drop"),
            "score": 0,
            "flags": [],
        },
    )
    nodes = _chunks(SimpleNamespace(split_document=lambda text, metadata: chunks), [_doc()])
    children = [node for node in nodes if node.metadata["node_level"] == "child"]
    ids = {node.id_ for node in nodes}

    assert [node.text for node in children] == ["keep A", "keep C"]
    assert len([node for node in nodes if node.metadata["node_level"] == "parent"]) == 1
    assert all(
        not node.metadata.get(field) or node.metadata[field] in ids
        for node in nodes
        for field in ("prev_id", "next_id")
    )
    assert all(node.metadata.get("parent_id") in ids for node in children)
    assert children[0].metadata["next_id"] == children[1].id_
    assert children[1].metadata["prev_id"] == children[0].id_
    assert children[0].metadata["is_first_in_parent"] and children[1].metadata["is_last_in_parent"]


def test_generic_parser_honors_drop_mode(monkeypatch):
    from pcb_rag import ingest

    class Parser:
        def get_nodes_from_documents(self, documents, show_progress=False):
            return [
                TextNode(id_="keep", text="keep", metadata={}),
                TextNode(id_="drop", text="drop", metadata={}),
            ]

    monkeypatch.setattr(ingest, "_build_node_parser", lambda model: Parser())
    monkeypatch.setattr(
        ingest.Settings,
        "_embed_model",
        __import__("llama_index.core.embeddings", fromlist=["MockEmbedding"]).MockEmbedding(embed_dim=2),
    )
    monkeypatch.setattr(ingest, "augment_documents_with_multimodal", lambda docs, progress=True: docs)
    monkeypatch.setattr(ingest, "apply_contextual_retrieval", lambda nodes, docs: nodes)
    monkeypatch.setattr(ingest, "CHUNK_QUALITY_MODE", "drop")
    monkeypatch.setattr(
        ingest,
        "assess_chunk_quality",
        lambda text: {
            "is_low_value": text == "drop",
            "score": 0,
            "flags": [],
        },
    )

    nodes = ingest._parse_ingest_documents([_doc()])

    assert [node.text for node in nodes] == ["keep"]


def test_generic_drop_removes_childless_parent(monkeypatch):
    from pcb_rag import ingest

    class Parser:
        def get_nodes_from_documents(self, documents, show_progress=False):
            return [
                TextNode(id_="parent", text="parent", metadata={"node_level": "parent"}),
                TextNode(id_="drop", text="drop", metadata={"parent_id": "parent"}),
            ]

    monkeypatch.setattr(ingest, "_build_node_parser", lambda model: Parser())
    monkeypatch.setattr(
        ingest.Settings,
        "_embed_model",
        __import__("llama_index.core.embeddings", fromlist=["MockEmbedding"]).MockEmbedding(embed_dim=2),
    )
    monkeypatch.setattr(ingest, "augment_documents_with_multimodal", lambda docs, progress=True: docs)
    monkeypatch.setattr(ingest, "apply_contextual_retrieval", lambda nodes, docs: nodes)
    monkeypatch.setattr(ingest, "CHUNK_QUALITY_MODE", "drop")
    monkeypatch.setattr(ingest, "assess_chunk_quality", lambda text: {"is_low_value": True, "score": 0, "flags": []})

    assert ingest._parse_ingest_documents([_doc()]) == []


def test_related_edges_respect_exact_hop_boundary():
    from pcb_rag.graph_rag import KnowledgeGraph, Triple

    graph = KnowledgeGraph()
    graph.add_triple(Triple("铜厚", "要求", "35 μm"))
    graph.add_triple(Triple("35 μm", "测试方法", "荧光法"))

    assert len(graph.related_edges(["铜厚"], hops=1)) == 1
    assert len(graph.related_edges(["铜厚"], hops=2)) == 2


def test_attention_drop_rebuilds_hierarchy(monkeypatch):
    chunks = [("keep A", {}), ("drop B", {}), ("keep C", {})]
    splitter = SimpleNamespace(split_text=lambda text: chunks)
    original = ingest.AttentionSemanticSplitter
    ingest.AttentionSemanticSplitter = lambda **kwargs: splitter
    monkeypatch.setattr(ingest, "CHUNK_QUALITY_MODE", "drop")
    monkeypatch.setattr(
        ingest,
        "assess_chunk_quality",
        lambda text: {
            "is_low_value": text.startswith("drop"),
            "score": 0,
            "flags": [],
        },
    )
    try:
        nodes = ingest._attention_semantic_parse([_doc()])
    finally:
        ingest.AttentionSemanticSplitter = original
    ids = {node.id_ for node in nodes}

    assert [node.text for node in nodes] == ["keep A", "keep C"]
    assert all(
        not node.metadata.get(field) or node.metadata[field] in ids
        for node in nodes
        for field in ("prev_id", "next_id")
    )
    assert nodes[0].metadata["next_id"] == nodes[1].id_
    assert nodes[1].metadata["prev_id"] == nodes[0].id_


def test_zero_split_settings_are_not_replaced_by_defaults():
    splitter = ingest.HierarchicalStructureSplitter(child_overlap=0, semantic_variance_threshold=0)
    assert splitter.child_overlap == 0
    assert splitter.semantic_variance_threshold == 0
    attention = ingest.AttentionSemanticSplitter(threshold=0, overlap=0, window_size=0, breakpoint_percentile=0)
    assert attention.threshold == 0
    assert attention.overlap == 0
    assert attention.window_size == 0
    assert attention.breakpoint_percentile == 0


def test_graph_expansion_uses_configured_hops():
    from pcb_rag.graph_rag import KnowledgeGraph, Triple, expand_nodes_with_graph

    graph = KnowledgeGraph()
    graph.add_triple(Triple("铜厚", "要求", "35 μm", evidence="first"))
    graph.add_triple(Triple("35 μm", "测试方法", "荧光法", evidence="second"))
    original = TextNode(id_="base", text="铜厚要求", metadata={})
    item = SimpleNamespace(node=original, score=1.0)

    expanded = expand_nodes_with_graph([item], graph, max_extra=1, hops=2)

    assert "35 μm --测试方法--> 荧光法" in expanded[-1].node.text


def test_llm_evidence_respects_configured_character_limit(monkeypatch):
    from pcb_rag import graph_rag

    monkeypatch.setattr(graph_rag, "GRAPH_EVIDENCE_CHARS", 5)
    monkeypatch.setattr(graph_rag, "GRAPH_LLM_GLEANING", 0)
    monkeypatch.setattr(graph_rag, "GRAPH_MAX_TRIPLES_PER_CHUNK", 1)
    text = "铜厚要求为35μm"
    raw = '[{"head":"铜厚","relation":"要求","tail":"35 μm","evidence":"铜厚要求为35μm"}]'
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=raw))

    triples = graph_rag.extract_triples_llm(text, llm, strict=True)

    assert triples[0].evidence == "铜厚要求为"


def test_expired_reservations_are_reclaimed_after_process_restart(tmp_path):
    from pcb_rag.rerank_control import RerankControl

    path = tmp_path / "budget.sqlite3"
    control = RerankControl(path, "scope", 130, 100, reservation_ttl=1)
    ticket = control.reserve(
        provider="account",
        model="rerank",
        query="",
        documents=["a"],
        interval=0,
        phase="test",
        activity="measured",
        current_time=100,
    )
    control = RerankControl(path, "scope", 130, 100, reservation_ttl=1)
    control._reap_expired_reservations(current_time=ticket.scheduled_at + 2)

    assert control.snapshot()["attempts"] == 0
    assert control.snapshot()["reserved_tokens"] == 0
    assert control.records()[0]["status"] == "expired_unsent"


def test_expired_inflight_request_keeps_attempt_and_conservative_accounting(tmp_path):
    from pcb_rag.rerank_control import RerankControl

    control = RerankControl(tmp_path / "budget.sqlite3", "scope", 130, 100, reservation_ttl=1)
    ticket = control.reserve(
        provider="account",
        model="rerank",
        query="",
        documents=["a"],
        interval=0,
        phase="test",
        activity="measured",
        current_time=100,
    )
    control.mark_inflight(ticket)
    with control.connect() as db:
        db.execute("UPDATE requests SET expires_at=101 WHERE request_id=?", (ticket.request_id,))
    control._reap_expired_reservations(current_time=102)

    state = control.snapshot()
    assert state["attempts"] == 1
    assert state["reserved_tokens"] == 0
    assert state["accounted_tokens"] == ticket.estimated_tokens
    assert state["reported_tokens"] == 0
    assert control.records()[0]["status"] == "expired_inflight"


def test_reclaiming_expired_reservation_preserves_circuit_reason(tmp_path):
    from pcb_rag.rerank_control import RerankControl

    control = RerankControl(tmp_path / "budget.sqlite3", "scope", 130, 100, reservation_ttl=1)
    control.reserve(
        provider="account",
        model="rerank",
        query="",
        documents=["a"],
        interval=0,
        phase="test",
        activity="measured",
        current_time=100,
    )
    control.trip("http_402")
    control._reap_expired_reservations(current_time=102)

    assert control.snapshot()["circuit_reason"] == "http_402"
    assert control.snapshot()["reserved_tokens"] == 0


def test_unexpired_reservations_are_not_reclaimed(tmp_path):
    from pcb_rag.rerank_control import RerankControl

    control = RerankControl(tmp_path / "budget.sqlite3", "scope", 130, 100, reservation_ttl=10)
    control.reserve(
        provider="account",
        model="rerank",
        query="",
        documents=["a"],
        interval=0,
        phase="test",
        activity="measured",
        current_time=100,
    )
    control._reap_expired_reservations(current_time=105)

    assert control.snapshot()["attempts"] == 1
    assert control.snapshot()["reserved_tokens"] > 0


def test_frozen_eval_does_not_backfill_production_dotenv(monkeypatch):
    from pcb_rag import env_loader

    monkeypatch.setenv("PCB_RAG_ENV_FILE", ".env.eval")
    monkeypatch.setenv("PCB_RAG_LOAD_DOTENV", "1")
    monkeypatch.setattr(env_loader, "project_root", lambda: __import__("pathlib").Path("/tmp/test-root"))
    assert env_loader.load_project_env() == 0


def test_frozen_worker_blocks_default_dotenv_before_package_isolation(tmp_path, monkeypatch):
    from pcb_rag import env_loader

    monkeypatch.setenv("PCB_RAG_EXPERIMENT_FROZEN", "1")
    monkeypatch.delenv("PCB_RAG_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("RECALL_TOP_K=999\n", encoding="utf-8")
    monkeypatch.setattr(env_loader, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RECALL_TOP_K", raising=False)

    assert env_loader.load_project_env() == 0
    assert "RECALL_TOP_K" not in os.environ


def test_frozen_base_env_drops_unfrozen_secret_keys(monkeypatch):
    from scripts.experiment_queue import frozen_base_env

    monkeypatch.setenv("DIFY_API_TOKEN", "host-secret")
    monkeypatch.setenv("LLM_API_KEY", "host-key")
    monkeypatch.setenv("RECALL_TOP_K", "999")

    env = frozen_base_env({"LLM_MODEL": "frozen"})

    assert "DIFY_API_TOKEN" not in env
    assert "LLM_API_KEY" not in env
    assert "RECALL_TOP_K" not in env
