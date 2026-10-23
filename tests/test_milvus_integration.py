"""Opt-in real Milvus test. Only uniquely named temporary collections are touched."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("PCB_RAG_TEST_MILVUS") != "1", reason="set PCB_RAG_TEST_MILVUS=1 for a local Milvus service"
    ),
]


def test_real_ingestion_recovery_graph_indexes_and_collection_switch(tmp_path, monkeypatch, offline_pipeline):
    from fastapi.testclient import TestClient
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from pymilvus import MilvusClient
    from pcb_rag import ingest, query, retrieval, dify_external_api as api
    from pcb_rag.graph_rag import KnowledgeGraph
    from pcb_rag.migrate import activate_migration, migration_report
    from pcb_rag.retrieval import PipelineConfig, RetrievalPipeline, read_nodes
    from pcb_rag.revisions import RevisionStore

    uri = os.getenv("PCB_RAG_TEST_MILVUS_URI", "http://127.0.0.1:19530")
    # A freshly resumed WSL distro may expose the port before its proxy is ready.
    deadline = time.monotonic() + 90
    while True:
        try:
            client = MilvusClient(uri=uri, timeout=5)
            client.list_collections(timeout=5)
            break
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    prefix = "codex_test_" + uuid.uuid4().hex[:12]
    collection, migrated = prefix, prefix + "_next"
    before = set(client.list_collections())
    assert collection not in before and migrated not in before
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    a, b = source_dir / "IPC-6012E-A.txt", source_dir / "IPC-6012E-B.txt"
    content = "IPC-6012E\n\n1 铜厚测试\n\n铜厚不小于 35 μm。应按规定方法测量。" * 12
    a.write_text(content, encoding="utf-8")
    b.write_text(content + "\n采用荧光法无损测量。", encoding="utf-8")
    state, target = RevisionStore(tmp_path / "live.json"), RevisionStore(tmp_path / "migration.json")

    class CountingEmbedding(MockEmbedding):
        _passages: list = PrivateAttr(default_factory=list)

        def _get_text_embedding(self, text):
            self._passages.append(text)
            return super()._get_text_embedding(text)

        def _get_text_embeddings(self, texts):
            self._passages.extend(texts)
            return [[0.5] * self.embed_dim for _ in texts]

    embed, llm = CountingEmbedding(embed_dim=8), offline_pipeline.llm
    monkeypatch.setattr(Settings, "_embed_model", embed)
    monkeypatch.setattr(Settings, "_llm", llm)
    for key, value in {
        "DATA_DIR": str(source_dir),
        "COLLECTION": collection,
        "MILVUS_URI": uri,
        "INGEST_MANIFEST_PATH": str(tmp_path / "manifest.json"),
        "EMBED_BACKEND": "local",
        "NODE_PARSER_MODE": "parent_child",
        "PARENT_MAX_SIZE": 1600,
        "PARENT_MIN_SIZE": 20,
        "CHILD_MAX_SIZE": 200,
        "CHILD_MIN_SIZE": 30,
        "SEMANTIC_SPLIT_THRESHOLD": 100000,
        "GRAPH_RAG_ENABLED": True,
        "GRAPH_EXTRACT_BACKEND": "rule",
        "GRAPH_COMMUNITY_ENABLED": False,
        "CONTEXTUAL_RETRIEVAL_ENABLED": True,
        "CONTEXTUAL_RETRIEVAL_MODE": "rule",
    }.items():
        monkeypatch.setattr(ingest, key, value)
    monkeypatch.setattr(ingest, "build_llm", lambda: llm)
    monkeypatch.setattr(ingest, "build_embed_model", lambda: embed)
    monkeypatch.setattr(ingest, "get_embedding_dim", lambda model: 8)
    monkeypatch.setenv("KB_STATE_PATH", str(state.path))
    monkeypatch.setenv("GRAPH_SEMANTIC_BACKEND", "milvus")
    monkeypatch.setenv("GRAPH_SEMANTIC_ENABLED", "1")
    monkeypatch.setenv("INGEST_OVERWRITE", "0")
    monkeypatch.setattr(query, "MILVUS_URI", uri)
    monkeypatch.setattr(query, "EMBED_BACKEND", "local")
    monkeypatch.setattr(query, "get_embedding_dim", lambda model: 8)
    report = {"passed": False, "kind": "deterministic_models_real_milvus", "collections": [collection, migrated]}
    started = time.monotonic()
    try:
        ingest.main()
        first = state.read()["active"]
        children = [node for node in read_nodes(first["artifacts"]["chunks"]) if node.metadata["node_level"] == "child"]
        ids = [node.id_ for node in children]
        assert client.get(collection_name=collection, ids=ids, output_fields=["id"], consistency_level="Strong")
        assert all(node.metadata["source_type"] != "unknown" for node in children)
        assert any(node.metadata.get("pcb_facts") for node in children)
        assert all("doc_node_id:" not in text and "tenant_id:" not in text for text in embed._passages)
        semantic = json.loads(Path(first["artifacts"]["graph_vectors"]).read_text())
        assert set(semantic["collections"]) == {"entities", "relations"}
        for name in semantic["collections"].values():
            hits = client.search(collection_name=name, data=[[0.5] * 8], limit=1, output_fields=["key"])
            assert hits[0]

        async def create_store():
            return MilvusVectorStore(
                uri=uri, collection_name=collection, dim=8, embedding_field="embedding", consistency_level="Strong"
            )

        vector_store = asyncio.run(create_store())
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed)
        pipeline = RetrievalPipeline(
            PipelineConfig(
                collection=collection,
                graph_enabled=True,
                routing=False,
                lexical_enabled=True,
                rerank_enabled=False,
                expand=True,
            ),
            state={"index": index, "llm": llm, "embed_model": embed},
            revision_store=state,
        )
        result = pipeline.retrieve("铜厚测试", top_k=2)
        assert result.nodes and result.routes["graph"] == "ok"
        assert any(item.node.metadata["node_level"] == "parent" for item in result.nodes)
        count = len(embed._passages)
        ingest.main()
        assert len(embed._passages) == count
        assert state.read()["active"]["revision"] == first["revision"]

        a.write_text(content.replace("35", "70"), encoding="utf-8")
        original = ingest.build_graph_from_nodes

        def fail_graph(*args, **kwargs):
            raise RuntimeError("injected graph extraction failure")

        monkeypatch.setattr(ingest, "build_graph_from_nodes", fail_graph)
        with pytest.raises(RuntimeError, match="injected"):
            ingest.main()
        assert state.read()["active"]["revision"] == first["revision"]
        assert state.read()["pending"]["stages"]["graph"]["status"] == "failed"
        assert len(
            client.get(collection_name=collection, ids=ids, output_fields=["id"], consistency_level="Strong")
        ) == len(ids)
        old_result = pipeline.retrieve("铜厚测试", top_k=2)
        assert all("70 μm" not in item.node.text for item in old_result.nodes)
        pending_chunks = state.read()["pending"]["stages"]["chunks"]["result"]["path"]
        pending_passages = {
            node.text for node in read_nodes(pending_chunks) if node.metadata.get("node_level") == "child"
        }
        count = len(embed._passages)
        monkeypatch.setattr(ingest, "build_graph_from_nodes", original)
        ingest.main()
        current = state.read()["active"]
        # Only entity/relation descriptions are embedded during graph retry.
        assert pending_passages.isdisjoint(embed._passages[count:])
        assert current["revision"] != first["revision"]
        assert pipeline.retrieve("铜厚测试", 2).revision == current["revision"]
        b.unlink()
        ingest.main()
        current = state.read()["active"]
        assert len(current["docs"]) == 1
        graph = KnowledgeGraph.load(current["artifacts"]["graph"])
        assert all(
            source["doc_id"] in current["docs"] for edge in graph.edges.values() for source in edge["contributions"]
        )

        monkeypatch.setattr(ingest, "COLLECTION", migrated)
        monkeypatch.setenv("KB_STATE_PATH", str(target.path))
        monkeypatch.setenv("INGEST_REQUIRE_EMPTY_COLLECTION", "1")
        ingest.main()
        built = target.read()["active"]
        validation = migration_report(built, old_nodes=read_nodes(current["artifacts"]["chunks"]))
        assert validation["valid"], validation["errors"]
        activate_migration(state, target, validation, expected_revision=current["revision"])
        switched = pipeline.retrieve("铜厚测试", 2)
        assert switched.revision == built["revision"]
        assert pipeline.state["collection"] == migrated
        monkeypatch.setattr(retrieval, "_default", pipeline)
        monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
        monkeypatch.setattr(api, "API_TOKEN", "integration-token")
        with TestClient(api.app) as http:
            response = http.post(
                "/api/ask", headers={"authorization": "Bearer integration-token"}, json={"query": "铜厚测试"}
            )
            assert response.status_code == 200, response.text
            assert response.json()["revision"] == built["revision"]
            assert response.json()["citations"][0]["chunk_id"] in {
                row["metadata"]["chunk_id"] for row in response.json()["records"]
            }

        monkeypatch.setenv("KB_STATE_PATH", str(state.path))
        a.unlink()
        ingest.main()
        empty = state.read()["active"]
        assert not KnowledgeGraph.load(empty["artifacts"]["graph"]).edges
        assert not pipeline.retrieve("铜厚测试").nodes
        report.update(
            passed=True,
            elapsed_seconds=round(time.monotonic() - started, 3),
            initial_children=len(children),
            initial_graph_indexes=semantic["collections"],
            metadata_coverage=validation["metadata_coverage"],
            mapping_counts=validation["mapping_counts"],
            checks=[
                "real_ingest_main",
                "child_vectors",
                "real_parents",
                "entity_relation_indexes",
                "unchanged_skip",
                "failed_stage_old_revision",
                "resume_without_chunk_embedding",
                "shared_relation_delete",
                "new_collection_validation",
                "live_collection_switch",
                "http_citation",
                "empty_graph_restart",
            ],
        )
    finally:
        created = set(client.list_collections()) - before
        temporary = [name for name in created if name == prefix or name.startswith(prefix + "_")]
        for name in temporary:
            client.drop_collection(collection_name=name)
        report["temporary_collections_removed"] = len(temporary)
        report["production_collections_preserved"] = before.issubset(set(client.list_collections()))
        destination = os.getenv("RAG_VALIDATION_REPORT")
        if destination:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
