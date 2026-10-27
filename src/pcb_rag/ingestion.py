"""Recoverable ingestion: stage immutable data, verify, then publish one pointer.

Old rows and snapshots remain readable until an explicit retention operation.
Updating a document never deletes its currently published evidence first.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

from pcb_rag import incremental
from pcb_rag.graph_rag import KnowledgeGraph
from pcb_rag.revisions import RevisionStore, fingerprint


def write_nodes(path, nodes):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for node in nodes:
                stream.write(
                    json.dumps({"id": node.id_, "text": node.text, "metadata": node.metadata}, ensure_ascii=False)
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target.resolve())


def stamp_nodes(nodes, documents):
    """Version physical IDs while preserving stable logical IDs and references."""
    result = copy.deepcopy(nodes)
    id_map = {}
    for node in result:
        metadata = node.metadata
        doc_id = str(metadata.get("doc_node_id") or "")
        if doc_id not in documents:
            raise ValueError(f"chunk has no source document: {node.id_}")
        logical = str(metadata.get("logical_chunk_id") or node.id_)
        revision = documents[doc_id]["revision"]
        physical = fingerprint([logical, revision])[:32]
        id_map[node.id_] = physical
        id_map[logical] = physical
        metadata.update(logical_chunk_id=logical, revision=revision, node_level=metadata.get("node_level") or "child")
        node.id_ = physical
        # Embedding metadata must not include adjacency IDs, ACLs or large fact
        # structures. Source context is already present in the passage text.
        if hasattr(node, "excluded_embed_metadata_keys"):
            node.excluded_embed_metadata_keys = list(metadata)
            node.excluded_llm_metadata_keys = list(metadata)
    if len({node.id_ for node in result}) != len(result):
        raise ValueError("duplicate chunk IDs; source/page identity is ambiguous")
    for node in result:
        for key in ("parent_id", "prev_id", "next_id"):
            value = node.metadata.get(key)
            if value:
                node.metadata[key] = id_map.get(value, "")
    return result


def run_ingestion(
    documents,
    *,
    pipeline,
    collection,
    parse,
    insert,
    verify,
    graph_build=None,
    graph_vectors=None,
    store: RevisionStore | None = None,
    force_rebuild: bool = False,
):
    """Callbacks separate model/database IO from the recoverable state machine."""
    from pcb_rag.retrieval import read_nodes

    store = store or RevisionStore()
    with store.writer_lock():
        state = store.read()
        active = state.get("active") or {}
        same_collection = active.get("collection", collection) == collection
        old_docs = active.get("docs", {}) if same_collection else {}
        _, _, current = incremental.diff_documents(documents, {"docs": old_docs}, pipeline=pipeline)
        if force_rebuild:
            current = copy.deepcopy(current)
            rebuild_id = active.get("revision", "initial")
            for doc_id, doc in current.items():
                doc["revision"] = fingerprint([doc_id, doc["hash"], pipeline["chunks"], pipeline["vectors"], rebuild_id])[:24]
        expected = os.getenv("INGEST_EXPECTED_REVISION", "")
        candidate = fingerprint({"docs": current, "pipeline": pipeline, "collection": collection})[:24]
        if expected and candidate != expected:
            raise ValueError("source/configuration changed since the failed run; start a new ingestion explicitly")
        pending = store.begin(current, pipeline, collection)
        revision = pending["revision"]
        if active.get("revision") == revision:
            state = store.read()
            state["pending"] = {}
            store.write(state)
            return active
        directory = store.path.parent / (store.path.stem + "_snapshots") / revision
        directory.mkdir(parents=True, exist_ok=True)
        chunk_path, graph_path = directory / "chunks.jsonl", directory / "graph.json"
        lexical_path, semantic_path = directory / "lexical.jsonl", directory / "graph.vectors.json"
        previous_nodes = (
            read_nodes(active["artifacts"]["chunks"])
            if same_collection and active.get("artifacts", {}).get("chunks")
            else []
        )
        changed = {
            doc_id
            for doc_id, doc in current.items()
            if force_rebuild
            or old_docs.get(doc_id, {}).get("hash") != doc["hash"]
            or old_docs.get(doc_id, {}).get("pipeline", {}).get("chunks") != pipeline["chunks"]
        }

        def parse_stage():
            retained = [
                node
                for node in previous_nodes
                if node.metadata.get("doc_node_id") in current and node.metadata.get("doc_node_id") not in changed
            ]
            selected = [doc for doc in documents if incremental.doc_node_id_of(doc) in changed]
            new_nodes = parse(selected) if selected else []
            nodes = stamp_nodes(retained + list(new_nodes), current)
            # Every nonempty source must have at least one child before publish.
            covered = {node.metadata["doc_node_id"] for node in nodes if node.metadata["node_level"] == "child"}
            missing = {
                incremental.doc_node_id_of(doc) for doc in documents if str(getattr(doc, "text", "")).strip()
            } - covered
            if missing:
                raise ValueError(f"parsed sources contain no child evidence: {sorted(missing)}")
            return {"path": write_nodes(chunk_path, nodes), "count": len(nodes)}

        store.stage("chunks", parse_stage)
        nodes = read_nodes(chunk_path)
        existing = {node.id_ for node in previous_nodes}
        new_children = [
            node for node in nodes if node.metadata.get("node_level") == "child" and node.id_ not in existing
        ]
        all_children = [node for node in nodes if node.metadata.get("node_level") == "child"]

        def vector_stage():
            for offset in range(0, len(new_children), 128):
                batch = new_children[offset : offset + 128]
                key = "vectors:" + fingerprint([node.id_ for node in batch])[:16]

                def insert_batch(batch=batch):
                    insert(batch)
                    verify([node.id_ for node in batch])
                    return {"count": len(batch)}

                store.stage(key, insert_batch)
            verify([node.id_ for node in all_children])
            return {"count": len(all_children), "inserted": len(new_children)}

        store.stage("vectors", vector_stage)

        def lexical_stage():
            from pcb_rag.retrieval import build_lexical

            built = build_lexical(all_children)
            if len(built.docs) != len(all_children):
                raise ValueError("lexical/source count mismatch")
            return {"path": write_nodes(lexical_path, all_children), "count": len(built.docs)}

        store.stage("lexical", lexical_stage)

        def graph_stage():
            if graph_build is None:
                return {"enabled": False}
            previous_graph = active.get("artifacts", {}).get("graph")
            graph = KnowledgeGraph.load(previous_graph) if previous_graph else None
            graph_changed = active.get("pipeline", {}).get("graph") != pipeline["graph"]
            rebuild = graph is None or graph_changed or any(edge.get("needs_backfill") for edge in graph.edges.values())
            if rebuild:
                graph = KnowledgeGraph()
                targets = set(current)
            else:
                targets = {
                    key for key, doc in current.items() if old_docs.get(key, {}).get("revision") != doc["revision"]
                }
                graph.remove_docs(targets | (set(old_docs) - set(current)))
            for doc_id in sorted(targets):
                part_path = directory / "graph_parts" / (doc_id + ".json")

                def build_part(doc_id=doc_id, part_path=part_path):
                    part = graph_build([node for node in all_children if node.metadata["doc_node_id"] == doc_id])
                    part.revision = revision
                    part.save(str(part_path))
                    return {"path": str(part_path.resolve()), "relations": len(part.edges)}

                store.stage("graph:" + doc_id, build_part)
                part = KnowledgeGraph.load(str(part_path))
                if part is None:
                    raise ValueError("missing completed graph partition")
                graph.merge(part)
            graph.revision = revision
            graph.save(str(graph_path))  # Empty graphs must also be persisted.
            if graph_vectors:
                graph_vectors(graph, semantic_path)
            return {
                "enabled": True,
                "path": str(graph_path.resolve()),
                "paths": [str(semantic_path.resolve())] if graph_vectors else [],
                "entities": len(graph.nodes),
                "relations": len(graph.edges),
            }

        store.stage("graph", graph_stage)
        artifacts = {"chunks": str(chunk_path.resolve()), "lexical": str(lexical_path.resolve())}
        if graph_build:
            artifacts["graph"] = str(graph_path.resolve())
        if graph_build and graph_vectors:
            artifacts["graph_vectors"] = str(semantic_path.resolve())
        verify([node.id_ for node in all_children])
        # Publish is a single atomic file replacement. Failures above leave the
        # old revision active and retain resumable stage results.
        return store.publish(artifacts)
