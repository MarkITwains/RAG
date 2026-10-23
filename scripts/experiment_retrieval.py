#!/usr/bin/env python
"""Run one retrieval configuration, sequentially, with durable per-query data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.experiment_common import (  # noqa: E402
    append_jsonl,
    digest,
    now,
    public_config,
    read_jsonl,
    summarize_retrieval,
    write_json,
    assert_frozen_inputs,
    isolate_project_env,
)


class Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.reset()

    def reset(self):
        self.timing = {}
        self.route = None
        self.warnings = []
        self.rerank_calls = []
        self.rerank_failures = 0
        self.rerank_usage = []

    def emit(self, record):
        event = getattr(record, "rerank_usage", None)
        if event is not None:
            self.rerank_usage.append(dict(event))
        message = record.getMessage()
        for label, seconds in re.findall(r"\[Timing\]\s*([^:：]+?)[:：]\s*([\d.]+)\s*s", message):
            self.timing[label.strip()] = self.timing.get(label.strip(), 0) + float(seconds)
        route = re.search(r"\[QueryRoute\]\s+(\w+)", message)
        if route:
            self.route = route.group(1)
        if record.levelno >= logging.WARNING:
            self.warnings.append(message[:1500])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--path", choices=["evalscript", "production", "bm25"], required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    isolate_project_env()
    from pcb_rag.env_loader import load_project_env

    load_project_env()
    output = Path(args.out)
    assert_frozen_inputs(output.parent.parent, dict(os.environ))
    partial = output.with_suffix(".jsonl")
    meta_path = output.with_suffix(".meta.json")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]
    if not dataset:
        raise RuntimeError("Empty dataset")
    collector = Collector()
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger().addHandler(collector)
    initialized_at = time.perf_counter()
    if args.path == "production":
        import pcb_rag.dify_external_api as api

        api._initialize_retrieval_engine()

        def retrieve(query):
            return api._retrieve_nodes(query, top_k=args.top_k)

        actual_recall = api._app_state["recall_k"]
    from eval import evaluate_recall as er
    import pcb_rag.query as rag
    from pcb_rag.api_clients import ApiReranker, describe_backends
    from llama_index.core.schema import MetadataMode, QueryBundle

    if args.path != "production":
        index = rag.build_index(os.getenv("LLM_MODEL", ""))
        if args.path == "evalscript":
            rag.HYDE_ENABLED = True
            rag.SHORT_QUERY_THRESHOLD = 100
        actual_recall = int(os.getenv("RECALL_TOP_K", "200"))
        retriever = er._build_retriever(
            index,
            recall_k=actual_recall,
            mode="bm25" if args.path == "bm25" else "fusion_hyde_rerank",
            use_colbert=False,
            use_hyde=args.path == "evalscript",
            use_query2doc=False,
            colbert_candidates=50,
            rrf_k=40,
            llm=rag.Settings.llm,
        )
        rerank = rag._try_build_reranker() if args.path == "evalscript" else None
        if args.path == "evalscript" and rerank is None:
            raise RuntimeError("Required API reranker did not initialize")

        def retrieve(query):
            nodes = retriever.retrieve(query)
            if rerank is not None:
                nodes = rerank._postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
            return nodes[: args.top_k]

    original_rerank = ApiReranker._request_scores

    def measured_rerank(self, query, documents):
        start = time.perf_counter()
        try:
            return original_rerank(self, query, documents)
        except Exception:
            collector.rerank_failures += 1
            raise
        finally:
            collector.rerank_calls.append(
                {"candidates": len(documents), "wall_s": round(time.perf_counter() - start, 4)}
            )

    ApiReranker._request_scores = measured_rerank
    identity = {
        "dataset_sha256": digest(Path(args.dataset)),
        "path": args.path,
        "top_k": args.top_k,
        "dataset_n": len(dataset),
        "config": public_config(dict(os.environ)),
    }
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["identity"] != identity:
        raise RuntimeError("Partial results belong to another dataset or configuration")
    if not meta_path.exists():
        write_json(
            meta_path,
            {
                "identity": identity,
                "started_at": now(),
                "backends": describe_backends(),
                "actual_recall_k": actual_recall,
                "initialization_s": round(time.perf_counter() - initialized_at, 4),
                "execution_context": {
                    "max_parallel_experiments": int(os.getenv("PCB_RAG_EXPERIMENT_MAX_PARALLEL", "1")),
                    "timing_mode": os.getenv("PCB_RAG_EXPERIMENT_TIMING_MODE", "exclusive"),
                    "note": "Shared-resource ablation timings are not isolated latency measurements.",
                },
                "effective_module_settings": {
                    key: getattr(rag, key, None)
                    for key in [
                        "HYDE_ENABLED",
                        "HYDE_MAX_LENGTH",
                        "SHORT_QUERY_THRESHOLD",
                        "RERANK_TOP_N",
                        "FUSION_NUM_QUERIES",
                        "FUSION_RRF_K",
                    ]
                },
                "timing_note": "Parallel stages overlap. Stage durations must not be summed to infer wall time.",
            },
        )
    results = read_jsonl(partial)
    if any(row["query"] != dataset[i]["query"] for i, row in enumerate(results)):
        raise RuntimeError("Partial query order differs from the frozen dataset")
    warmup_sizes = []
    os.environ["RERANK_ACTIVITY"] = "warmup"
    for item in dataset[: args.warmup]:
        nodes = retrieve(item["query"])
        warmup_sizes.append(len(nodes))
        if not nodes:
            print("[warmup] empty result", flush=True)
        else:
            print(f"[warmup] {len(nodes)} nodes", flush=True)
    if warmup_sizes and not any(warmup_sizes):
        raise RuntimeError("All warmup retrievals were empty; refusing a potentially broken experiment")
    os.environ["RERANK_ACTIVITY"] = "measured"
    for position in range(len(results), len(dataset)):
        item = dataset[position]
        collector.reset()
        start = time.perf_counter()
        error = None
        try:
            nodes = retrieve(item["query"])
        except Exception as exc:
            nodes = []
            error = f"{type(exc).__name__}: {exc}"
        wall = time.perf_counter() - start
        got = [er._node_id(node) or "" for node in nodes]
        gt = set(item.get("ground_truth_ids", []))
        rank = next((i for i, node_id in enumerate(got, 1) if node_id in gt), 0)
        record = {
            "index": position,
            "query": item["query"],
            "gt_ids": sorted(gt),
            "gt_texts": item.get("ground_truth_text", []),
            "got_ids": got,
            "first_hit_rank": rank,
            "mrr": 1 / rank if rank else 0,
            "ap": er._average_precision(got, gt),
            "ndcg@10": er._ndcg_at_k(got, gt, 10),
            "hits": {f"hit@{k}": float(any(node_id in gt for node_id in got[:k])) for k in [1, 3, 5, 10]},
            "wall_s": round(wall, 4),
            "timing": dict(collector.timing),
            "route": collector.route,
            "rerank_calls": list(collector.rerank_calls),
            "rerank_failures": collector.rerank_failures,
            "rerank_usage": list(collector.rerank_usage),
            "warnings": list(collector.warnings),
            "error": error,
            "got": [
                {
                    "id": er._node_id(node),
                    "score": node.score,
                    "text": node.node.get_content(metadata_mode=MetadataMode.NONE),
                    "metadata": node.node.metadata,
                }
                for node in nodes
            ],
        }
        record["timing"]["rerank_request_including_throttle"] = round(
            sum(call["wall_s"] for call in collector.rerank_calls), 4
        )
        results.append(record)
        append_jsonl(partial, record)
        write_json(
            output.with_suffix(".progress.json"),
            {"completed": len(results), "total": len(dataset), "updated_at": now(), "last_wall_s": record["wall_s"]},
        )
        print(f"[query {position + 1}/{len(dataset)}] rank={rank} wall={wall:.2f}s", flush=True)
        if error or collector.rerank_failures:
            write_json(
                output.with_suffix(".failure.json"),
                {
                    "at": now(),
                    "index": position,
                    "error": error,
                    "rerank_failures": collector.rerank_failures,
                    "note": "Stopped on the first failed request; degraded rankings are not valid measurements.",
                },
            )
            return 2
    summary = summarize_retrieval(results)
    summary["ranking_metrics_available"] = all(bool(row["gt_ids"]) for row in results)
    if not summary["ranking_metrics_available"]:
        for key in ["mrr", "mrr@10", "map", "ndcg@10", "hit@1", "hit@3", "hit@5", "hit@10"]:
            summary[key] = None
    summary["route_counter"] = dict(Counter(row["route"] for row in results))
    write_json(
        output,
        {
            "meta": json.loads(meta_path.read_text(encoding="utf-8")),
            "summary": summary,
            "results": results,
            "finished_at": now(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 2 if summary["invalid"] or summary["n"] != len(dataset) else 0


if __name__ == "__main__":
    raise SystemExit(main())
