from types import SimpleNamespace

import pytest

pytest.importorskip("llama_index.core")
from llama_index.core.schema import TextNode

from pcb_rag import incremental
from pcb_rag.graph_rag import KnowledgeGraph, Triple
from pcb_rag.ingestion import run_ingestion
from pcb_rag.revisions import RevisionStore, node_visible, pipeline_fingerprints
from pcb_rag.retrieval import read_nodes


def document(text="original", name="A"):
    return SimpleNamespace(id_=name, hash=text, text=text, metadata={"file_path": name + ".txt"})


class Harness:
    def __init__(self, tmp_path):
        self.store = RevisionStore(tmp_path / "state.json")
        self.rows = {}
        self.parses = self.inserts = self.graphs = 0
        self.fail = ""

    def parse(self, docs):
        self.parses += 1
        if self.fail == "chunks":
            raise RuntimeError("parse failed")
        return [
            TextNode(
                id_=doc.id_ + "-chunk",
                text=doc.text,
                metadata={"doc_node_id": incremental.doc_node_id_of(doc), "source_path": doc.id_ + ".txt"},
            )
            for doc in docs
        ]

    def insert(self, nodes):
        self.inserts += 1
        for node in nodes:
            self.rows[node.id_] = node
        if self.fail == "vectors":
            raise RuntimeError("partial write failed")

    def verify(self, ids):
        assert set(ids).issubset(self.rows)

    def graph(self, nodes):
        self.graphs += 1
        if self.fail == "graph":
            raise RuntimeError("graph failed")
        graph = KnowledgeGraph()
        for node in nodes:
            graph.add_triple(
                Triple(
                    "铜厚",
                    "要求",
                    "35 μm",
                    chunk_id=node.id_,
                    doc_id=node.metadata["doc_node_id"],
                    revision=node.metadata["revision"],
                    evidence=node.text,
                )
            )
        return graph

    def run(self, docs, config=None, graph_vectors=None):
        return run_ingestion(
            docs,
            pipeline=pipeline_fingerprints(config or {}),
            collection="test",
            parse=self.parse,
            insert=self.insert,
            verify=self.verify,
            graph_build=self.graph,
            graph_vectors=graph_vectors,
            store=self.store,
        )


@pytest.mark.parametrize("failed_stage", ["chunks", "vectors", "graph"])
def test_stage_failure_preserves_active_revision_and_resumes(tmp_path, failed_stage):
    harness = Harness(tmp_path)
    old = harness.run([document()])
    harness.fail = failed_stage
    with pytest.raises(RuntimeError):
        harness.run([document("changed")])
    state = harness.store.read()
    assert state["active"]["revision"] == old["revision"]
    assert state["pending"]["stages"][failed_stage]["status"] == "failed"
    visible = [row for row in harness.rows.values() if node_visible(row.metadata, state["active"])]
    assert [row.text for row in visible] == ["original"]
    inserted_before_retry = harness.inserts
    harness.fail = ""
    current = harness.run([document("changed")])
    assert current["revision"] != old["revision"]
    assert harness.store.read()["pending"] == {}
    if failed_stage == "graph":
        assert harness.inserts == inserted_before_retry
    assert [row.text for row in harness.rows.values() if node_visible(row.metadata, current)] == ["changed"]
    assert set(current["storage_revisions"].values()) == {current["revision"]}


def test_graph_only_backfill_does_not_parse_or_embed_unchanged_content(tmp_path):
    harness = Harness(tmp_path)
    old = harness.run([document()])
    counts = harness.parses, harness.inserts, harness.graphs
    assert harness.run([document()])["revision"] == old["revision"]
    assert (harness.parses, harness.inserts, harness.graphs) == counts
    current = harness.run([document()], {"GRAPH_LLM_GLEANING": "2"})
    assert harness.parses == counts[0] and harness.inserts == counts[1]
    assert harness.graphs == counts[2] + 1
    assert current["revision"] != old["revision"]


def test_parser_change_rebuilds_content_and_metadata(tmp_path):
    harness = Harness(tmp_path)
    old = harness.run([document()])
    current = harness.run([document()], {"CHILD_MAX_SIZE": "500"})
    assert harness.parses == 2 and harness.inserts == 2
    assert list(old["docs"].values())[0]["revision"] != list(current["docs"].values())[0]["revision"]


def test_acl_change_reparses_existing_documents(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    old = harness.run([document()])
    monkeypatch.setattr("pcb_rag.security.ACL_ENABLED", True)
    monkeypatch.setattr("pcb_rag.ingest.ACL_ENABLED", True)
    original_parse = harness.parse

    def parse_with_acl(docs):
        from pcb_rag.security import acl_metadata_for_ingest

        nodes = original_parse(docs)
        for node in nodes:
            node.metadata.update(acl_metadata_for_ingest())
        return nodes

    harness.parse = parse_with_acl
    current = harness.run([document()], {"ACL_ENABLED": "1"})

    assert current["revision"] != old["revision"]
    assert harness.parses == 2
    assert list(current["docs"].values())[0]["revision"] != list(old["docs"].values())[0]["revision"]
    assert harness.rows[list(harness.rows)[-1]].metadata["visibility"] == "private"


def test_force_rebuild_reparses_and_reembeds_unchanged_documents(tmp_path):
    harness = Harness(tmp_path)
    old = harness.run([document()])
    current = run_ingestion(
        [document()],
        pipeline=pipeline_fingerprints({}),
        collection="test",
        parse=harness.parse,
        insert=harness.insert,
        verify=harness.verify,
        graph_build=harness.graph,
        store=harness.store,
        force_rebuild=True,
    )

    assert current["revision"] != old["revision"]
    assert list(current["docs"].values())[0]["revision"] != list(old["docs"].values())[0]["revision"]
    assert harness.parses == 2 and harness.inserts == 2
    assert current["storage_revisions"]["vectors"] == current["revision"]


def test_force_rebuild_twice_does_not_reuse_published_revision(tmp_path):
    harness = Harness(tmp_path)
    revisions = []
    for _ in range(2):
        revisions.append(
            run_ingestion(
                [document()],
                pipeline=pipeline_fingerprints({}),
                collection="test",
                parse=harness.parse,
                insert=harness.insert,
                verify=harness.verify,
                graph_build=harness.graph,
                store=harness.store,
                force_rebuild=True,
            )["revision"]
        )

    assert len(set(revisions)) == 2
    assert harness.parses == 2 and harness.inserts == 2 and harness.graphs == 2


def test_delete_last_source_publishes_empty_snapshot_and_graph(tmp_path):
    harness = Harness(tmp_path)
    harness.run([document()])
    empty = harness.run([])
    assert read_nodes(empty["artifacts"]["chunks"]) == []
    assert not KnowledgeGraph.load(empty["artifacts"]["graph"]).edges
    assert not any(node_visible(row.metadata, empty) for row in harness.rows.values())


def test_lexical_failure_retries_without_reembedding(tmp_path, monkeypatch):
    from pcb_rag import retrieval

    harness = Harness(tmp_path)
    harness.run([document()])
    original = retrieval.build_lexical
    monkeypatch.setattr(retrieval, "build_lexical", lambda _: (_ for _ in ()).throw(RuntimeError("lexical failed")))
    with pytest.raises(RuntimeError):
        harness.run([document("next")])
    assert harness.store.read()["pending"]["stages"]["lexical"]["status"] == "failed"
    inserts = harness.inserts
    monkeypatch.setattr(retrieval, "build_lexical", original)
    harness.run([document("next")])
    assert harness.inserts == inserts


@pytest.mark.parametrize("missing", ["graph", "graph_vectors"])
def test_resume_rebuilds_missing_graph_artifacts_before_publish(tmp_path, monkeypatch, missing):
    from pathlib import Path

    harness = Harness(tmp_path)
    original = harness.store.publish

    def interrupted(artifacts):
        Path(artifacts[missing]).unlink()
        raise RuntimeError("publication interrupted")

    def semantic(graph, path):
        path.write_text('{"revision": "' + graph.revision + '"}', encoding="utf-8")

    monkeypatch.setattr(harness.store, "publish", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        harness.run([document()], graph_vectors=semantic)
    assert harness.store.read()["pending"]["stages"]["graph"]["status"] == "complete"
    counts = harness.parses, harness.inserts, harness.graphs
    monkeypatch.setattr(harness.store, "publish", original)
    active = harness.run([document()], graph_vectors=semantic)
    assert all(Path(path).is_file() for path in active["artifacts"].values())
    assert KnowledgeGraph.load(active["artifacts"]["graph"]).edges
    assert (harness.parses, harness.inserts, harness.graphs) == counts
