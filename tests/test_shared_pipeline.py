import json
from dataclasses import replace


from pcb_rag import security
from pcb_rag.engine import KnowledgeBase
from pcb_rag.graph_rag import KnowledgeGraph, Triple
from pcb_rag.graph_semantic import GraphEvidenceRetriever, build_graph_vectors
from pcb_rag.retrieval import read_nodes


class Embedding:
    model_name = "test-space"
    api_base = "test-local"
    truncate = "END"

    def get_text_embedding_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def get_query_embedding(self, text):
        return [1.0, 0.0]


def graph_route(fixture, tmp_path, nodes):
    graph = KnowledgeGraph()
    for node in nodes:
        graph.add_triple(
            Triple(
                "镀层厚度",
                "采用",
                "荧光法",
                chunk_id=node.id_,
                doc_id=node.metadata.get("doc_node_id", ""),
                evidence=node.text,
                head_desc=node.text,
                relation_desc=node.text,
            )
        )
    path = tmp_path / "semantic.json"
    embed = Embedding()
    build_graph_vectors(graph, embed, path)
    retriever = GraphEvidenceRetriever.load(graph, path, embed)
    fixture.pipeline.config = replace(fixture.pipeline.config, graph_enabled=True)
    fixture.pipeline.state.update(
        graph=graph, graph_retriever=retriever, source_nodes={node.id_: node for node in nodes}
    )
    return graph, path


def test_graph_recovers_zero_base_recall_without_entity_name(offline_pipeline, tmp_path):
    f = offline_pipeline
    f.index.rows = []
    graph_route(f, tmp_path, [f.node("evidence", "该方法能在不破坏样品的情况下测定金层的厚薄。")])
    result = f.pipeline.retrieve("如何无损检查覆盖层的尺寸？")
    assert [node.node.id_ for node in result.nodes] == ["evidence"]
    assert result.routes["vector:0"] == "empty"
    assert result.routes["graph"] == "ok"
    assert result.nodes[0].node.metadata["graph_route"] in {"entity_description", "relation_description"}


def test_community_summary_recalls_original_evidence(offline_pipeline, tmp_path):
    class CommunityEmbedding(Embedding):
        def get_text_embedding_batch(self, texts):
            return [[1.0, 0.0] if "无损测量" in text else [0.0, 1.0] for text in texts]

    f = offline_pipeline
    f.index.rows = []
    evidence = f.node("evidence", "该方法能在不破坏样品的情况下测定金层的厚薄。")
    graph = KnowledgeGraph()
    graph.add_triples(
        [
            Triple("镀层厚度", "采用", "荧光法", chunk_id="evidence", evidence=evidence.text),
            Triple("荧光法", "属于", "无损检测", chunk_id="evidence", evidence=evidence.text),
        ]
    )
    graph.communities = {
        "测厚主题": {
            "entities": ["镀层厚度", "荧光法", "无损检测"],
            "size": 3,
            "summary": "无损测量覆盖层厚度的方法",
        }
    }
    path = tmp_path / "community.json"
    embed = CommunityEmbedding()
    payload = build_graph_vectors(graph, embed, path)
    retriever = GraphEvidenceRetriever.load(graph, path, embed)
    f.pipeline.config = replace(f.pipeline.config, graph_enabled=True)
    f.pipeline.state.update(graph=graph, graph_retriever=retriever, source_nodes={"evidence": evidence})

    result = f.pipeline.retrieve("如何检查覆盖层尺寸？")

    assert len(payload["communities"]) == 1
    assert [item.node.id_ for item in result.nodes] == ["evidence"]
    assert result.nodes[0].node.metadata["graph_route"] == "community_summary"
    assert result.nodes[0].node.text == evidence.text
    assert "无损测量" not in result.nodes[0].node.text


def test_graph_acl_filters_before_candidate_limit(offline_pipeline, tmp_path, monkeypatch):
    f = offline_pipeline
    f.index.rows = []
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    private = [
        f.node(
            f"private-{index:02d}",
            "secret coating process",
            tenant_id="other",
            visibility="private",
            doc_node_id=f"private-doc-{index}",
        )
        for index in range(20)
    ]
    public = f.node("public", "公开测试方法", tenant_id="acme", visibility="public", doc_node_id="public-doc")
    graph_route(f, tmp_path, [*private, public])
    token = security.set_request_principal(security.Principal(tenant_id="acme", user_id="alice", authenticated=True))
    try:
        result = f.pipeline.retrieve("如何检查覆盖层？")
    finally:
        security.reset_request_principal(token)

    assert [item.node.id_ for item in result.nodes] == ["public"]
    assert result.routes["graph"] == "ok"


def test_private_graph_text_never_reaches_reranker_or_neighborhood(offline_pipeline, tmp_path, monkeypatch):
    f = offline_pipeline
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    private = f.node("private", "secret coating process", tenant_id="other", visibility="private", doc_node_id="d2")
    public = f.node("public", "公开测试方法", tenant_id="acme", visibility="public", doc_node_id="d1")
    graph_route(f, tmp_path, [private, public])
    seen = []

    class Reranker:
        def _postprocess_nodes(self, nodes, **kwargs):
            seen.extend(nodes)
            return nodes

    f.index.rows = [f.score(private), f.score(public)]
    f.pipeline.state["rerank"] = Reranker()
    f.pipeline.config = replace(f.pipeline.config, rerank_enabled=True)
    token = security.set_request_principal(security.Principal(tenant_id="acme", user_id="alice", authenticated=True))
    try:
        result = f.pipeline.retrieve("如何检查覆盖层？")
        assert {item.node.id_ for item in seen} == {"public"}
        assert {item.node.id_ for item in result.nodes} == {"public"}
        neighborhood = KnowledgeBase(auto_init=False, pipeline=f.pipeline).graph_neighbors("镀层厚度")
        assert neighborhood.found
        assert "secret" not in json.dumps(neighborhood.to_dict())
        assert neighborhood.relations[0]["sources"][0]["chunk_id"] == "public"
    finally:
        security.reset_request_principal(token)


def test_neighborhood_authorization_fetches_only_candidate_sources(offline_pipeline, tmp_path, monkeypatch):
    f = offline_pipeline
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    public = f.node("public", "公开测试方法", tenant_id="acme", visibility="public", doc_node_id="d1")
    graph, _ = graph_route(f, tmp_path, [public])
    unrelated = f.node("unrelated", "secret", tenant_id="other", visibility="private", doc_node_id="d2")
    graph.add_triple(Triple("秘密实体", "采用", "秘密工艺", chunk_id="unrelated", doc_id="d2"))
    f.pipeline.state["source_nodes"]["unrelated"] = unrelated
    requested = []
    original_fetch = f.pipeline._fetch

    def fetch(ids, state):
        requested.extend(ids)
        return original_fetch(ids, state)

    monkeypatch.setattr(f.pipeline, "_fetch", fetch)
    token = security.set_request_principal(security.Principal(tenant_id="acme", user_id="alice", authenticated=True))
    try:
        neighborhood = KnowledgeBase(auto_init=False, pipeline=f.pipeline).graph_neighbors("镀层厚度")
    finally:
        security.reset_request_principal(token)

    assert neighborhood.found
    assert requested == ["public"]
    assert "secret" not in json.dumps(neighborhood.to_dict())


def test_neighborhood_falls_back_to_authorized_fuzzy_entity(offline_pipeline, monkeypatch):
    f = offline_pipeline
    monkeypatch.setattr(security, "ACL_ENABLED", True)
    private = f.node("private", "secret", tenant_id="other", visibility="private", doc_node_id="d2")
    public = f.node("public", "公开测试方法", tenant_id="acme", visibility="public", doc_node_id="d1")
    graph = KnowledgeGraph()
    graph.add_triples(
        [
            Triple("镀层厚度", "采用", "秘密工艺", chunk_id="private", doc_id="d2"),
            Triple("镀层厚度要求", "采用", "荧光法", chunk_id="public", doc_id="d1"),
        ]
    )
    f.pipeline.state.update(graph=graph, source_nodes={"private": private, "public": public})
    token = security.set_request_principal(security.Principal(tenant_id="acme", user_id="alice", authenticated=True))
    try:
        neighborhood = KnowledgeBase(auto_init=False, pipeline=f.pipeline).graph_neighbors("镀层厚度")
    finally:
        security.reset_request_principal(token)

    assert neighborhood.found
    assert neighborhood.entity == "镀层厚度要求"
    assert neighborhood.relations[0]["sources"][0]["chunk_id"] == "public"
    assert "secret" not in json.dumps(neighborhood.to_dict())


def test_embedding_mode_mismatch_disables_semantic_index(offline_pipeline, tmp_path):
    graph, path = graph_route(offline_pipeline, tmp_path, [offline_pipeline.node("c")])
    embed = Embedding()
    embed.truncate = "NONE"
    assert not GraphEvidenceRetriever.load(graph, path, embed).semantic_ready
    assert not GraphEvidenceRetriever(KnowledgeGraph(), Embedding()).semantic_ready


def test_sdk_cli_eval_select_identical_final_evidence(offline_pipeline):
    from pcb_rag.cli import retrieve_for_cli
    from eval.evaluate import EvalEngine

    f = offline_pipeline
    sdk = KnowledgeBase(auto_init=False, pipeline=f.pipeline).retrieve("铜厚", top_k=5)
    cli = retrieve_for_cli("铜厚", pipeline=f.pipeline, top_k=5)
    evaluator = EvalEngine(200, 5, False, pipeline=f.pipeline)
    evaluation = evaluator.retrieve("铜厚")
    assert (
        [entry.chunk_id for entry in sdk.evidence]
        == [entry.chunk_id for entry in cli.evidence]
        == [node.node.id_ for node in evaluation]
        == ["real", "low"]
    )
    assert [entry.text for entry in sdk.evidence] == [node.node.text for node in evaluation]


def test_sdk_legacy_budget_fallback_returns_evidence(offline_pipeline, monkeypatch):
    from pcb_rag import query

    monkeypatch.setattr(query, "RAG_CONTEXT_MAX_TOKENS", 0)
    result = KnowledgeBase(auto_init=False, pipeline=offline_pipeline.pipeline).retrieve("铜厚", with_context=True)

    assert [entry.chunk_id for entry in result.evidence] == ["real", "low"]
    assert result.context
    assert all(entry.tokens > 0 for entry in result.evidence)


def test_disabled_routes_do_not_report_degradation(offline_pipeline):
    kb = KnowledgeBase(auto_init=False, pipeline=offline_pipeline.pipeline)
    result = kb.retrieve("铜厚")
    assert not result.degraded and not kb.health()["degraded"]
    assert result.routes["graph"] == result.routes["bm25"] == result.routes["rerank"] == "disabled"


def test_empty_answer_preserves_failed_route_diagnostics(offline_pipeline):
    f = offline_pipeline
    f.index.rows = []
    f.pipeline.config = replace(f.pipeline.config, graph_enabled=True)
    result = KnowledgeBase(auto_init=False, pipeline=f.pipeline).answer("铜厚")
    assert result.routes["vector:0"] == "empty" and result.routes["graph"] == "unavailable"
    assert set(result.degraded) == {"no_evidence", "graph_unavailable"}
    assert result.timings["vector:0"] >= 0


def test_multi_query_uses_pipeline_llm_not_global_settings(offline_pipeline, monkeypatch):
    from pcb_rag import query as q
    from llama_index.core import Settings

    f = offline_pipeline
    monkeypatch.setattr(q, "LLM_QUERY_REWRITE_ENABLED", True)
    monkeypatch.setattr(Settings, "_llm", None)
    f.pipeline.config = replace(f.pipeline.config, multi_expand=True, num_queries=2)
    result = f.pipeline.retrieve("铜厚")
    assert result.nodes and len(f.index.calls) == 2
    assert len(f.llm._prompts) == 2
    assert all("原始查询" in prompt for prompt in f.llm._prompts)


def test_self_rag_uses_distinct_followup_retrieval(offline_pipeline, monkeypatch):
    from pcb_rag import query as q

    f = offline_pipeline
    monkeypatch.setattr(q, "SELF_RAG_ENABLED", True)
    monkeypatch.setattr(
        q,
        "_agentic_plan",
        lambda query, docs, llm, used: {
            "sufficient": len(used) == 2,
            "next_query": "铜厚测量条件",
            "reason": "需要补充条件",
        },
    )
    monkeypatch.setattr(q, "self_rag_evaluate", lambda *args: {"overall": 4})
    f.index.rows = lambda query: [f.score(f.node("second" if "条件" in query else "first"))]

    class Adapter:
        def retrieve(self, query):
            return f.pipeline.retrieve_nodes(query)

    result = q.self_rag_query("铜厚", Adapter(), f.llm, max_iterations=4, verbose=False)
    assert f.index.calls == ["铜厚", "铜厚测量条件"]
    assert result["self_rag_info"]["iterations"] == 2
    assert {entry["chunk_id"] for entry in result["evidence_bundle"].entries} == {"first", "second"}


def test_self_rag_low_final_score_triggers_threshold_retrieval(offline_pipeline, monkeypatch):
    from pcb_rag import query as q

    f = offline_pipeline
    monkeypatch.setattr(q, "SELF_RAG_ENABLED", True)
    monkeypatch.setattr(q, "_agentic_plan", lambda *args: {"sufficient": True, "next_query": "", "reason": "规划完成"})
    scores = iter([2, 4])

    def evaluate(*args):
        score = next(scores)
        return {
            "overall": score,
            "retrieval_suggestion": "铜厚测量条件" if score < 3 else "",
            "reason": "需要补充条件",
        }

    monkeypatch.setattr(q, "self_rag_evaluate", evaluate)
    f.index.rows = lambda query: [f.score(f.node("second" if "条件" in query else "first"))]

    class Adapter:
        def retrieve(self, query):
            return f.pipeline.retrieve_nodes(query)

    result = q.self_rag_query("铜厚", Adapter(), f.llm, max_iterations=3, threshold=3, verbose=False)

    assert f.index.calls == ["铜厚", "铜厚测量条件"]
    assert result["self_rag_info"]["final_score"] == 4
    assert result["self_rag_info"]["iterations"] == 2
    assert any(row.get("phase") == "threshold_retrieval" for row in result["self_rag_info"]["history"])


def test_self_rag_threshold_stops_on_duplicate_query(offline_pipeline, monkeypatch):
    from pcb_rag import query as q

    f = offline_pipeline
    monkeypatch.setattr(q, "SELF_RAG_ENABLED", True)
    monkeypatch.setattr(q, "_agentic_plan", lambda *args: {"sufficient": True, "next_query": "", "reason": "规划完成"})
    monkeypatch.setattr(
        q,
        "self_rag_evaluate",
        lambda *args: {"overall": 1, "retrieval_suggestion": "铜厚要求", "reason": "重复查询"},
    )

    class Adapter:
        def retrieve(self, query):
            return f.pipeline.retrieve_nodes(query)

    result = q.self_rag_query("铜厚要求", Adapter(), f.llm, max_iterations=4, threshold=3, verbose=False)

    assert f.index.calls == ["铜厚要求"]
    assert result["self_rag_info"]["iterations"] == 1


def test_published_snapshot_refreshes_lexical_graph_and_source_visibility(tmp_path, offline_pipeline):
    from pcb_rag.ingestion import run_ingestion
    from pcb_rag.revisions import pipeline_fingerprints
    from llama_index.core.schema import Document

    f = offline_pipeline
    f.pipeline.config = replace(f.pipeline.config, collection="test", lexical_enabled=True, graph_enabled=True)
    rows = {}

    def publish(text):
        return run_ingestion(
            [Document(text=text, id_="document", metadata={"doc_node_id": "doc", "source_path": "A.txt"})],
            pipeline=pipeline_fingerprints({}),
            collection="test",
            store=f.pipeline.revisions,
            parse=lambda docs: [f.node("child", doc.text, doc_node_id="doc") for doc in docs],
            insert=lambda batch: rows.update({node.id_: node for node in batch}),
            verify=lambda ids: all(node_id in rows for node_id in ids),
            graph_build=lambda nodes: KnowledgeGraph(),
        )

    old = publish("旧版铜厚参数")
    f.index.rows = []
    f.pipeline.refresh()
    assert f.pipeline.state["revision"] == old["revision"]
    new = publish("新版阻抗测试参数")
    f.index.rows = [f.score(node) for node in rows.values()]
    result = f.pipeline.retrieve("新版阻抗测试参数")
    assert result.revision == new["revision"]
    assert all("旧版" not in item.node.text for item in result.nodes)
    assert f.pipeline.state["graph"].revision == new["revision"]
    assert {node.id_ for node in f.pipeline.state["bm25_index"].docs} == {
        node.id_ for node in read_nodes(new["artifacts"]["lexical"])
    }
