"""Small, dependency-free helpers for reproducible experiment records."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def isolate_project_env() -> None:
    """Keep workers on exported frozen values, without changing production code."""
    if os.getenv("PCB_RAG_EXPERIMENT_FROZEN") != "1":
        return
    from pcb_rag import env_loader

    original = env_loader.load_project_env
    if getattr(original, "_experiment_isolated", False):
        return

    def frozen_load(path=None):
        return 0 if path is None else original(path)

    frozen_load._experiment_isolated = True
    env_loader.load_project_env = frozen_load


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def assert_frozen_inputs(run: Path, env: dict) -> None:
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest(run / "datasets/regression_v1.json") != manifest["dataset_sha256"]:
        raise RuntimeError("Frozen regression dataset changed")
    for relative, expected in manifest.get("code_fingerprints", {}).items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Core implementation changed during the run: {relative}")
    for relative, expected in manifest.get("orchestration_fingerprints", {}).items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Experiment implementation changed during the run: {relative}")
    cache = Path(env["LEXICAL_CACHE_PATH"])
    if not cache.is_absolute():
        cache = ROOT / cache
    if digest(cache) != manifest["lexical_sha256"]:
        raise RuntimeError("Frozen lexical corpus changed")
    # Prevent the production loader's TTL from re-exporting/reordering frozen
    # rows if an upstream outage makes this run last more than 24 hours.
    cache.touch()


def percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))], 4)


def public_config(env: dict) -> dict:
    prefixes = (
        "LLM_",
        "EMBED_",
        "RERANK_",
        "FUSION_",
        "QUERY_",
        "HYDE_",
        "CHUNK_",
        "RAG_",
        "LEXICAL_",
        "BM25_",
        "MULTI_",
        "NODE_",
        "CHILD_",
        "PARENT_",
        "SEMANTIC_",
        "CONTEXTUAL_",
        "GRAPH_",
        "SELF_",
        "API_",
        "ACL_",
        "JUDGE_",
        "GENERATOR_",
        "RECALL_",
        "MILVUS_",
        "OLLAMA_",
        "HF_",
        "QWEN3_",
        "COMPRESSION_",
        "CONTEXT_",
        "MULTIMODAL_",
        "TABLE_",
        "INGEST_",
        "OBSERVABILITY_",
        "TRACE_",
        "METRIC_",
        "SLOW_SPAN_MS",
        "SESSION_",
        "CITATION_",
        "STEPBACK_",
        "DECOMPOSE_",
    )
    secret_names = ("API_KEY", "API_TOKEN", "AUTH_TOKEN", "PASSWORD", "SECRET", "CREDENTIAL")
    return {
        key: value
        for key, value in sorted(env.items())
        if (key.startswith(prefixes) or key in {"COLLECTION", "DATA_DIR", "INGEST_MANIFEST_PATH"})
        and not any(part in key for part in secret_names)
    }


def summarize_retrieval(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        return {"n": 0, "invalid": True}
    timing_keys = set().union(*(row.get("timing", {}) for row in rows))
    return {
        "n": count,
        "mrr": round(statistics.fmean(row["mrr"] for row in rows), 6),
        "mrr@10": round(statistics.fmean(row["mrr"] if 0 < row["first_hit_rank"] <= 10 else 0 for row in rows), 6),
        "map": round(statistics.fmean(row["ap"] for row in rows), 6),
        **{f"hit@{k}": round(statistics.fmean(row["hits"][f"hit@{k}"] for row in rows), 6) for k in [1, 3, 5, 10]},
        "ndcg@10": round(statistics.fmean(row["ndcg@10"] for row in rows), 6),
        "wall_s_mean": round(statistics.fmean(row["wall_s"] for row in rows), 4),
        "wall_s_p50": percentile([row["wall_s"] for row in rows], 50),
        "wall_s_p95": percentile([row["wall_s"] for row in rows], 95),
        "wall_s_p99": percentile([row["wall_s"] for row in rows], 99),
        "empty_results": sum(not row["got_ids"] for row in rows),
        "request_errors": sum(bool(row.get("error")) for row in rows),
        "rerank_fallbacks": sum(row.get("rerank_failures", 0) for row in rows),
        "rerank_reported_tokens": sum(
            usage["actual_tokens"]
            for row in rows
            for usage in row.get("rerank_usage", [])
            if usage.get("actual_tokens") is not None
        ),
        "rerank_unknown_usage_requests": sum(
            usage.get("actual_tokens") is None for row in rows for usage in row.get("rerank_usage", [])
        ),
        "invalid": any(row.get("error") or row.get("rerank_failures") for row in rows),
        "timing_mean": {
            key: round(statistics.fmean(row["timing"][key] for row in rows if key in row["timing"]), 4)
            for key in sorted(timing_keys)
        },
    }
