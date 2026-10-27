#!/usr/bin/env python
"""Re-embed existing Milvus rows into a separate profiled NVIDIA collection.

Keeps IDs, text, dynamic metadata and source vectors intact while the embedding
model, and its dimension, change. Checkpoints permit resuming after an API
failure. A paired vector-recall comparison is reported only when the source and
target dimensions match; across dimensions only the target baseline is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_rows(client, collection: str) -> list[dict]:
    iterator = client.query_iterator(
        collection_name=collection, filter="id != ''", output_fields=["*"], batch_size=128, timeout=60
    )
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    return sorted(rows, key=lambda row: row["id"])


def corpus_digest(rows: list[dict], vector_field: str) -> str:
    digest = hashlib.sha256()
    for row in rows:
        body = {key: value for key, value in row.items() if key != vector_field}
        digest.update(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def embedding_text(row: dict) -> str:
    from llama_index.core.schema import MetadataMode
    from llama_index.core.vector_stores.utils import metadata_dict_to_node

    node = metadata_dict_to_node(row)
    node.set_content(row["text"])
    return node.get_content(metadata_mode=MetadataMode.EMBED)


def summarize_ranks(ranks: list[int]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "mrr@20": round(sum(1 / rank for rank in ranks if 0 < rank <= 20) / n, 6),
        "mrr@10": round(sum(1 / rank for rank in ranks if 0 < rank <= 10) / n, 6),
        **{f"hit@{k}": round(sum(0 < rank <= k for rank in ranks) / n, 6) for k in [1, 5, 10, 20, 200]},
    }


def evaluate(vectors, queries, ids: list[str], dataset: list[dict]) -> dict:
    import numpy as np

    scores = np.asarray(queries, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    ranks = []
    details = []
    for row, ranking in zip(dataset, np.argsort(-scores, axis=1, kind="stable")):
        expected = set(row["ground_truth_ids"])
        rank = next((position for position, idx in enumerate(ranking, 1) if ids[int(idx)] in expected), 0)
        ranks.append(rank)
        details.append({"query": row["query"], "rank": rank})
    return {"summary": summarize_ranks(ranks), "results": details}


def main() -> int:
    import numpy as np
    from llama_index.embeddings.openai import OpenAIEmbedding
    from pymilvus import DataType, MilvusClient

    from pcb_rag.api_clients import (
        EMBED_API_KEY,
        EMBED_BASE_URL,
        EMBED_DIM,
        EMBED_MODEL,
        build_embed_model,
        get_embedding_profile,
    )
    from pcb_rag.embedding_contract import PROFILE_KEY, profile_digest, validate_embedding_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="pcb_kb")
    parser.add_argument("--target", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--dataset", default=str(ROOT / "eval/datasets/regression_v1.json"))
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.target == args.source or args.batch_size < 1 or args.min_interval < 0:
        parser.error("Use a separate target, a positive batch size, and a nonnegative interval")

    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    source_info = client.describe_collection(args.source)
    vector_field = os.getenv("MILVUS_EMBEDDING_FIELD", "embedding")
    source_dim = next(int(field["params"]["dim"]) for field in source_info["fields"] if field["name"] == vector_field)
    model = build_embed_model(backend="api")
    model.embed_batch_size = args.batch_size
    # 目标维度由 EMBED_DIM 决定，允许与源集合不同：换 embedding 模型时维度往往一起变，
    # 强行要求相等会让"只替换向量、保留 id/正文/图谱"这条迁移路径失效。
    target_dim = EMBED_DIM if EMBED_DIM > 0 else len(model.get_query_embedding("dimension probe"))
    if target_dim < 1:
        raise RuntimeError("Could not determine the target embedding dimension")
    profile = get_embedding_profile(target_dim)
    if profile["provider"] != "nvidia":
        raise RuntimeError("This migration requires the NVIDIA query/passage adapter")
    rows = read_rows(client, args.source)
    if not rows:
        raise RuntimeError("Source collection is empty")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Source collection contains duplicate IDs")
    corpus_hash = corpus_digest(rows, vector_field)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not dataset or any(not set(row["ground_truth_ids"]).intersection(ids) for row in dataset):
        raise RuntimeError("Regression labels are absent from the source corpus")

    checkpoint = ROOT / "data/embedding_migrations" / args.target
    checkpoint.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint / "state.json"
    identity = {"source": args.source, "target": args.target, "source_sha256": corpus_hash, "profile": profile}
    state = {**identity, "completed": 0}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in identity.items()):
            raise RuntimeError("Checkpoint does not match the source data or embedding configuration")
        if not client.has_collection(args.target):
            raise RuntimeError("Checkpoint target collection is missing")
        validate_embedding_profile(client.describe_collection(args.target).get("description", ""), profile)
    else:
        if client.has_collection(args.target):
            raise RuntimeError("Target already exists without this migration's checkpoint; refusing to overwrite it")
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=source_info["enable_dynamic_field"],
            description=json.dumps({PROFILE_KEY: profile, "source": args.source, "source_sha256": corpus_hash}),
        )
        for field in source_info["fields"]:
            params = dict(field.get("params", {}))
            if "max_length" in params:
                params["max_length"] = int(params["max_length"])
            if "dim" in params:
                params["dim"] = target_dim
            schema.add_field(
                field_name=field["name"],
                datatype=DataType(field["type"]),
                is_primary=field.get("is_primary", False),
                **params,
            )
        index = client.prepare_index_params()
        index.add_index(field_name=vector_field, index_type="FLAT", metric_type="IP")
        client.create_collection(collection_name=args.target, schema=schema, index_params=index)
        write_json(state_path, state)
    print(f"[snapshot] {len(rows)} rows; {len(dataset)} regression questions; sha256={corpus_hash}", flush=True)

    vector_path = checkpoint / "passage_vectors.npy"
    if state["completed"] and not vector_path.exists():
        raise RuntimeError("Checkpoint vectors are missing")
    vectors = np.lib.format.open_memmap(
        vector_path, mode="r+" if vector_path.exists() else "w+", dtype="float32", shape=(len(rows), target_dim)
    )
    if vectors.shape != (len(rows), target_dim) or not 0 <= state["completed"] <= len(rows):
        raise RuntimeError("Checkpoint shape or completed count is invalid")
    start_time = time.monotonic()
    previous_request = 0.0
    for start in range(state["completed"], len(rows), args.batch_size):
        wait = args.min_interval - (time.monotonic() - previous_request)
        if wait > 0:
            time.sleep(wait)
        batch = rows[start : start + args.batch_size]
        previous_request = time.monotonic()
        batch_vectors = model.get_text_embedding_batch([embedding_text(row) for row in batch], show_progress=False)
        if len(batch_vectors) != len(batch):
            raise RuntimeError("Embedding batch size mismatch")
        stop = start + len(batch)
        vectors[start:stop] = batch_vectors
        vectors.flush()
        target_rows = [{**row, vector_field: vector} for row, vector in zip(batch, batch_vectors)]
        client.upsert(collection_name=args.target, data=target_rows)
        state["completed"] = stop
        write_json(state_path, state)
        print(f"[passage] {stop}/{len(rows)} elapsed={time.monotonic() - start_time:.1f}s", flush=True)

    client.flush(args.target)
    target_rows = read_rows(client, args.target)
    if len(target_rows) != len(rows) or corpus_digest(target_rows, vector_field) != corpus_hash:
        raise RuntimeError("Target IDs, text, or metadata differ from the source")
    if corpus_digest(read_rows(client, args.source), vector_field) != corpus_hash:
        raise RuntimeError("Source corpus changed during the migration")
    print("[verified] source and target IDs, text, metadata match exactly", flush=True)

    same_space = target_dim == source_dim
    query_path = checkpoint / "queries.json"
    questions = [row["query"] for row in dataset]
    question_hash = hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()
    if query_path.exists():
        query_cache = json.loads(query_path.read_text(encoding="utf-8"))
        if query_cache["question_sha256"] != question_hash:
            raise RuntimeError("Query checkpoint belongs to a different regression dataset")
        if same_space and "legacy" not in query_cache:
            raise RuntimeError("Query checkpoint lacks the legacy vectors required for a paired comparison")
    else:
        legacy = OpenAIEmbedding(model_name=EMBED_MODEL, api_base=EMBED_BASE_URL, api_key=EMBED_API_KEY)
        legacy_queries = [] if same_space else None
        fixed_queries = []
        for start in range(0, len(questions), args.batch_size):
            batch = questions[start : start + args.batch_size]
            if legacy_queries is not None:
                legacy_queries.extend(legacy._get_text_embeddings(batch))
                time.sleep(args.min_interval)
            fixed_queries.extend(model._embed(batch, "query"))
            time.sleep(args.min_interval)
        query_cache = {"question_sha256": question_hash, "fixed": fixed_queries}
        if legacy_queries is not None:
            query_cache["legacy"] = legacy_queries
        write_json(query_path, query_cache)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "target": args.target,
        "rows": len(rows),
        "dimensions": {"source": source_dim, "target": target_dim},
        "source_sha256": corpus_hash,
        "profile": profile,
        "profile_sha256": profile_digest(profile),
        "dataset": str(Path(args.dataset).name),
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "after": evaluate(vectors, query_cache["fixed"], ids, dataset),
    }
    if same_space:
        old_vectors = np.asarray([row[vector_field] for row in rows], dtype=np.float32)
        report["evaluation_scope"] = (
            "paired exact IP vector ranking; no metadata filters, reranker, query rewriting, or HyDE"
        )
        report["before"] = evaluate(old_vectors, query_cache["legacy"], ids, dataset)
        report["query_only"] = evaluate(old_vectors, query_cache["fixed"], ids, dataset)
        before = report["before"]["summary"]["mrr@20"]
        after = report["after"]["summary"]["mrr@20"]
        report["delta_mrr@20"] = round(after - before, 6)
        report["candidate_improved"] = after > before
    else:
        # 维度不同即向量空间不同，跨空间点积没有意义，只留目标侧基线供后续对比。
        report["evaluation_scope"] = (
            f"target-only exact IP vector ranking; source and target dimensions differ "
            f"({source_dim} -> {target_dim}), so no paired baseline is computed"
        )
    write_json(Path(args.report), report)
    state["verified"] = True
    state["report"] = str(Path(args.report))
    write_json(state_path, state)
    print(
        json.dumps(
            {key: report[key]["summary"] for key in ["before", "query_only", "after"] if key in report},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if "delta_mrr@20" in report:
        print(f"[done] delta_mrr@20={report['delta_mrr@20']:+.6f}; production settings unchanged", flush=True)
    else:
        print(
            f"[done] target-only baseline written (dimensions {source_dim} -> {target_dim}); "
            "production settings unchanged",
            flush=True,
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
