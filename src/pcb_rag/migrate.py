"""Build a dedicated collection, validate source mappings, then publish it.

Nothing changes by default. --execute builds an isolated revision; --publish
additionally changes the live pointer after validation. Existing collections
and snapshots are retained. Graph backfill reuses published chunk vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pcb_rag.revisions import RevisionStore, atomic_json, state_path


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_key(text):
    return re.sub(r"\s+", "", text)


def migration_report(active, *, old_nodes=(), required_ids=()):
    from pcb_rag.retrieval import read_nodes
    from pcb_rag.graph_rag import KnowledgeGraph

    nodes = read_nodes(active["artifacts"]["chunks"])
    children = [node for node in nodes if node.metadata.get("node_level") == "child"]
    by_id = {node.id_: node for node in nodes}
    errors = []
    if len(by_id) != len(nodes):
        errors.append("duplicate physical chunk IDs")
    docs = active.get("docs", {})
    seen_docs = set()
    for node in nodes:
        md = node.metadata
        document = docs.get(md.get("doc_node_id"))
        if not document or md.get("revision") != document["revision"]:
            errors.append("source/revision mismatch: " + node.id_)
        if md.get("node_level") == "child":
            seen_docs.add(md.get("doc_node_id"))
        for field in ("parent_id", "prev_id", "next_id"):
            if not md.get(field):
                continue
            target = by_id.get(md[field])
            if target is None or any(target.metadata.get(key) != md.get(key) for key in ("doc_node_id", "revision")):
                errors.append("invalid " + field + ": " + node.id_)
            elif field == "parent_id" and target.metadata.get("node_level") != "parent":
                errors.append("parent link points to a child: " + node.id_)
    if set(docs) - seen_docs:
        errors.append("documents without child evidence: " + ",".join(sorted(set(docs) - seen_docs)))
    fields = (
        "source_path",
        "source_type",
        "doc_node_id",
        "revision",
        "node_level",
        "section",
        "page_label",
        "pcb_facts",
    )
    coverage = {
        field: sum(bool(node.metadata.get(field)) and node.metadata.get(field) != "unknown" for node in children)
        for field in fields
    }
    if any(
        coverage[field] < len(children)
        for field in ("source_path", "source_type", "doc_node_id", "revision", "node_level")
    ):
        errors.append("required source metadata is missing or unknown")
    graph_path = active["artifacts"].get("graph")
    graph = KnowledgeGraph.load(graph_path) if graph_path else None
    graph_sources = 0
    if graph is not None:
        if graph.revision != active["revision"]:
            errors.append("graph publication revision mismatch")
        for edge in graph.edges.values():
            for source in edge.get("contributions", []):
                graph_sources += 1
                node = by_id.get(source.get("chunk_id"))
                if (
                    source.get("needs_backfill")
                    or node is None
                    or any(
                        source.get(source_key) != node.metadata.get(node_key)
                        for source_key, node_key in (("doc_id", "doc_node_id"), ("revision", "revision"))
                    )
                ):
                    errors.append("invalid graph source contribution")
    logical = defaultdict(list)
    exact = defaultdict(list)
    for node in children:
        logical[node.metadata.get("logical_chunk_id", node.id_)].append(node.id_)
        exact[(str(node.metadata.get("source_path", "")).replace("\\", "/"), _text_key(node.text))].append(node.id_)
    mapping = {}
    for old in old_nodes:
        candidates = logical.get(old.metadata.get("logical_chunk_id", old.id_)) or exact.get(
            (str(old.metadata.get("source_path", "")).replace("\\", "/"), _text_key(old.text)), []
        )
        mapping[old.id_] = {
            "status": "mapped" if len(candidates) == 1 else "ambiguous" if candidates else "unmapped",
            "new_ids": list(candidates),
        }
    for old_id in required_ids:
        if old_id in by_id:
            mapping.setdefault(old_id, {"status": "mapped", "new_ids": [old_id]})
        if mapping.get(old_id, {}).get("status") != "mapped":
            errors.append("unresolved evaluation truth ID: " + old_id)
    return {
        "version": 1,
        "revision": active["revision"],
        "collection": active["collection"],
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "documents": len(docs),
        "children": len(children),
        "parents": len(nodes) - len(children),
        "metadata_coverage": coverage,
        "graph_source_contributions": graph_sources,
        "truth_mapping": mapping,
        "mapping_counts": dict(Counter(item["status"] for item in mapping.values())),
        "artifact_sha256": {name: _digest(path) for name, path in active["artifacts"].items()},
    }


def activate_migration(source: RevisionStore, target: RevisionStore, report, *, expected_revision):
    with source.writer_lock():
        state = source.read()
        if state.get("active", {}).get("revision") != expected_revision:
            raise RuntimeError("live revision changed during migration; regenerate the validation report")
        if state.get("pending"):
            raise RuntimeError("live ingestion is pending; complete it before switching collections")
        active = target.read().get("active", {})
        if not report.get("valid") or report.get("revision") != active.get("revision"):
            raise ValueError("migration validation is incomplete or failed")
        if any(
            report.get("artifact_sha256", {}).get(name) != _digest(path)
            for name, path in active.get("artifacts", {}).items()
        ):
            raise ValueError("migration artifacts changed after validation")
        state["previous"] = state.get("active", {})
        state["active"] = active
        state["pending"] = {}
        source.write(state)
    return active


def main(argv=None):
    from pcb_rag.env_loader import load_project_env

    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-collection")
    parser.add_argument("--state", type=Path, default=state_path())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--stage", choices=("all", "graph"), default="all")
    parser.add_argument("--graph-generation", default="backfill-v1")
    parser.add_argument("--mapping-from", type=Path, help="old chunks/lexical JSONL for exact ID mapping")
    parser.add_argument(
        "--truth-ids", type=Path, help="JSON array of evaluation chunk IDs which must map unambiguously"
    )
    args = parser.parse_args(argv)
    if args.publish and not args.execute:
        parser.error("--publish requires --execute")
    source = RevisionStore(args.state)
    previous = source.read().get("active", {})
    source_collection = previous.get("collection") or os.getenv("COLLECTION", "pcb_kb")
    env = dict(os.environ, INGEST_OVERWRITE="0")
    env.pop("INGEST_EXPECTED_REVISION", None)
    if args.data_dir:
        env["DATA_DIR"] = str(args.data_dir.resolve())
    if args.stage == "graph":
        if not previous:
            parser.error("graph backfill needs a published chunk snapshot; migrate legacy data first")
        target = source
        env.update(COLLECTION=source_collection, GRAPH_RAG_ENABLED="1", KB_GRAPH_GENERATION=args.graph_generation)
    else:
        name = args.target_collection or ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,179}", name) or name == source_collection:
            parser.error("choose a new, dedicated --target-collection")
        target = RevisionStore(source.path.parent / "migrations" / name / "state.json")
        env.update(COLLECTION=name, INGEST_REQUIRE_EMPTY_COLLECTION="1")
    env.update(
        KB_STATE_PATH=str(target.path.resolve()),
        INGEST_PLAN_ONLY="0" if args.execute else "1",
        INGEST_MANIFEST_PATH=str(target.path.with_suffix(".manifest.json").resolve()),
    )
    completed = subprocess.run([sys.executable, "-m", "pcb_rag.ingest"], env=env, check=False)
    if completed.returncode or not args.execute:
        return completed.returncode
    from pcb_rag.retrieval import read_nodes

    mapping_path = args.mapping_from or previous.get("artifacts", {}).get("chunks")
    old_nodes = read_nodes(mapping_path) if mapping_path else []
    required = json.loads(args.truth_ids.read_text(encoding="utf-8")) if args.truth_ids else []
    if not isinstance(required, list) or any(not isinstance(value, str) for value in required):
        parser.error("--truth-ids must contain a JSON array of strings")
    active = target.read()["active"]
    report = migration_report(active, old_nodes=old_nodes, required_ids=required)
    report_path = target.path.with_suffix(".validation.json")
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "validation": str(report_path),
                "valid": report["valid"],
                "errors": report["errors"],
                "revision": active["revision"],
                "metadata_coverage": report["metadata_coverage"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["valid"]:
        return 2
    if args.publish and args.stage == "all":
        activate_migration(source, target, report, expected_revision=previous.get("revision"))
        print("Published collection: " + active["collection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
