#!/usr/bin/env python
"""Checkpointed auxiliary stages for the six documented experiment groups."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts.experiment_common import (  # noqa: E402
    append_jsonl,
    digest,
    now,
    read_jsonl,
    write_json,
    assert_frozen_inputs,
    isolate_project_env,
)


def corpus() -> dict:
    path = Path(os.environ["LEXICAL_CACHE_PATH"])
    return {row["id"]: row for row in read_jsonl(path)}


def gate(run: Path):
    path_a = json.loads((run / "retrieval/r1_baseline_evalscript.json").read_text(encoding="utf-8"))
    path_b = json.loads((run / "retrieval/r1_baseline_production.json").read_text(encoding="utf-8"))
    if path_a["summary"]["invalid"] or path_b["summary"]["invalid"]:
        raise RuntimeError("Baseline contains request errors or rerank fallback; cannot pass the gate")
    if path_a["summary"]["n"] != 80 or path_b["summary"]["n"] != 80:
        raise RuntimeError("Baseline is not the complete frozen 80-question set")
    differences = []
    for left, right in zip(path_a["results"], path_b["results"]):
        if left["query"] != right["query"]:
            raise RuntimeError("Baseline query order mismatch")
        if left["first_hit_rank"] != right["first_hit_rank"]:
            differences.append(
                {
                    "query": left["query"],
                    "path_a_rank": left["first_hit_rank"],
                    "path_b_rank": right["first_hit_rank"],
                    "path_b_route": right["route"],
                    "eda_term": any(word in left["query"].lower() for word in ["altium", "allegro"]),
                }
            )
    delta = path_b["summary"]["mrr"] - path_a["summary"]["mrr"]
    result = {
        "path_a": path_a["summary"],
        "path_b": path_b["summary"],
        "delta_mrr_B_minus_A": delta,
        "threshold": 0.03,
        "threshold_exceeded": abs(delta) > 0.03,
        "decision": "continue within-production ablation; do not treat the two retrieval implementations as interchangeable"
        if abs(delta) > 0.03
        else "baseline paths within threshold; continue",
        "known_structural_differences": [
            "Path A forces HyDE and short-query threshold=100 and uses _FusionEnhancedRetriever",
            "Path B extracts business metadata filters, applies intent-specific retrieval weights, and has a HyDE wait budget",
            "effective HyDE max length and API defaults are preserved in each path meta file",
        ],
        "attribution_limit": "The rank comparison identifies differences but does not establish a causal contribution for each mechanism. The production ablation quantifies those mechanisms separately.",
        "different_rank_questions": differences,
    }
    write_json(run / "diagnostics/r1_gate.json", result)


def summary(run: Path):
    from scripts.repair_embeddings import read_rows, corpus_digest
    from pymilvus import MilvusClient

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    client = MilvusClient(uri=os.environ["MILVUS_URI"])
    rows = read_rows(client, manifest["collection"])
    after = corpus_digest(rows, "embedding")
    client.close()
    if after != manifest["corpus_nonvector_sha256"]:
        raise RuntimeError("Frozen corpus changed during experiments")
    write_json(
        run / "diagnostics/final_freeze_check.json",
        {"checked_at": now(), "rows": len(rows), "corpus_nonvector_sha256": after, "unchanged": True},
    )
    retrieval = []
    stage_rows = []
    for path in sorted((run / "retrieval").glob("*.json")):
        if path.name.endswith((".meta.json", ".progress.json")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" in data:
            execution = data.get("meta", {}).get("execution_context", {})
            job = next((item for item in manifest["steps"] if item["name"] == path.stem), {})
            context = {
                "timing_mode": execution.get("timing_mode", "exclusive"),
                "max_parallel_experiments": execution.get("max_parallel_experiments", 1),
                "max_observed_parallel_steps": job.get("max_observed_parallel_steps", 1),
            }
            retrieval.append({"configuration": path.stem, **context, **data["summary"]})
            timing_keys = set().union(*(row.get("timing", {}) for row in data.get("results", [])))
            for stage in sorted(timing_keys):
                values = [row["timing"][stage] for row in data["results"] if stage in row.get("timing", {})]
                from scripts.experiment_common import percentile

                stage_rows.append(
                    {
                        "configuration": path.stem,
                        "stage": stage,
                        "samples": len(values),
                        "mean_s": statistics.fmean(values),
                        "p50_s": percentile(values, 50),
                        "p95_s": percentile(values, 95),
                        **context,
                        "note": "Parallel and nested stage times overlap; do not sum them.",
                    }
                )
    fields = [
        "configuration",
        "timing_mode",
        "max_parallel_experiments",
        "max_observed_parallel_steps",
        "n",
        "mrr",
        "mrr@10",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "ndcg@10",
        "wall_s_mean",
        "wall_s_p50",
        "wall_s_p95",
        "wall_s_p99",
        "empty_results",
        "request_errors",
        "rerank_fallbacks",
        "rerank_reported_tokens",
        "rerank_unknown_usage_requests",
        "invalid",
    ]
    write_csv(run / "retrieval_summary.csv", retrieval, fields)
    write_csv(
        run / "stage_timing_summary.csv",
        stage_rows,
        [
            "configuration",
            "stage",
            "samples",
            "mean_s",
            "p50_s",
            "p95_s",
            "timing_mode",
            "max_parallel_experiments",
            "max_observed_parallel_steps",
            "note",
        ],
    )
    ablation = [row.copy() for row in retrieval if row["configuration"][:2].isdigit()]
    by_prefix = {row["configuration"][:2]: row for row in ablation}
    for row in ablation:
        number = int(row["configuration"][:2])
        control = f"{number - 1:02d}" if 2 <= number <= 8 else "08" if number >= 9 else None
        previous = by_prefix.get(control)
        if previous:
            row["compared_to"] = previous["configuration"]
            row["delta_mrr@10"] = row["mrr@10"] - previous["mrr@10"]
            row["delta_hit@5"] = row["hit@5"] - previous["hit@5"]
            if row["timing_mode"] == previous["timing_mode"] == "exclusive":
                row["delta_mean_s"] = row["wall_s_mean"] - previous["wall_s_mean"]
            else:
                row["delta_mean_s"] = None
                row["latency_comparison_note"] = (
                    "Unavailable: shared-resource timings do not isolate the mechanism cost."
                )
    write_csv(
        run / "ablation_summary.csv",
        ablation,
        fields + ["compared_to", "delta_mrr@10", "delta_hit@5", "delta_mean_s", "latency_comparison_note"],
    )
    latencies = []
    for path in sorted((run / "load").glob("g3_*_c*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" in data:
            latencies.append(data["summary"])
    write_csv(
        run / "load_summary.csv",
        latencies,
        [
            "endpoint",
            "concurrency",
            "total",
            "ok",
            "errors",
            "error_rate",
            "throughput_rps",
            "wall_s",
            "latency_p50",
            "latency_p95",
            "latency_p99",
        ],
    )
    output = {
        "run_id": manifest["run_id"],
        "created_at": now(),
        "collection": manifest["collection"],
        "corpus_unchanged": True,
        "scheduling": manifest.get("scheduling", {}),
        "retrieval": retrieval,
        "load": latencies,
    }
    if os.getenv("RERANK_CONTROL_DB"):
        from pcb_rag.rerank_control import RerankControl

        control = RerankControl.from_env()
        output["rerank_accounting"] = control.snapshot()
        records = control.records()
        write_json(run / "rerank_usage.json", output["rerank_accounting"])
        write_json(run / "diagnostics/rerank_usage_records.json", records)
        write_csv(
            run / "rerank_usage.csv",
            records,
            [
                "request_id",
                "phase",
                "activity",
                "model",
                "candidates",
                "requested_top_n",
                "query_chars",
                "document_chars",
                "actual_tokens",
                "estimated_tokens",
                "accounted_tokens",
                "usage_source",
                "status",
                "http_status",
                "started_at",
                "finished_at",
                "response_meta_json",
            ],
        )
    for key, path in [
        ("baseline_gate", "diagnostics/r1_gate.json"),
        ("generation_quality", "quality/g4_gen_quality.json"),
        ("softmatch", "diagnostics/g5_softmatch_scan.json"),
        ("golden_audit", "reviews/golden_audit_summary.json"),
        ("soft_review", "reviews/g5_review_summary.json"),
        ("scenarios", "reviews/g6_summary.json"),
        ("corpus_stats", "diagnostics/corpus_stats.json"),
    ]:
        target = run / path
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            output[key] = {k: v for k, v in data.items() if k not in ["rows", "per_query", "different_rank_questions"]}
    quality_data = output.get("generation_quality", {})
    quality_rows = [
        {"metric": key, **value, "judge_model": quality_data.get("config", {}).get("judge_model")}
        for key, value in quality_data.get("summary", {}).items()
        if isinstance(value, dict) and key != "overall"
    ]
    write_csv(
        run / "generation_quality_summary.csv",
        quality_rows,
        ["metric", "avg", "threshold", "passed", "evaluated", "failed", "failure_rate", "judge_model"],
    )
    soft = output.get("softmatch", {})
    soft_rows = []
    if soft:
        soft_rows.append({"method": "hard_id", "threshold": "", **soft["hard"]})
        soft_rows.extend(
            {"method": "passage_cosine", "threshold": threshold, **values} for threshold, values in soft["soft"].items()
        )
    write_csv(run / "softmatch_summary.csv", soft_rows, ["method", "threshold", "n", "mrr", "hit@1", "hit@5", "hit@10"])
    output["execution_steps"] = [
        {
            key: job[key]
            for key in [
                "name",
                "stage",
                "status",
                "exit_code",
                "reason",
                "timing_mode",
                "max_observed_parallel_steps",
            ]
            if key in job
        }
        for job in manifest["steps"]
        if job["name"] != "final_summary"
    ]
    output["incomplete_or_failed_steps"] = [job["name"] for job in output["execution_steps"] if job["status"] != "done"]
    output["manual_work_provenance"] = (
        "AI-authored and AI-reviewed substitutes explicitly authorized by user; no human-origin test claims."
    )
    lines = [
        "# RAG experiment rerun",
        "",
        f"Run: {manifest['run_id']}",
        f"Collection: {manifest['collection']}",
        "",
        manifest.get("scheduling", {}).get("latency_note", "All experiment steps executed sequentially."),
        "",
        "## Retrieval configurations",
        "",
    ]
    for row in retrieval:
        lines.append(
            f"- {row['configuration']}: n={row['n']}; MRR@10={row.get('mrr@10')}; Hit@5={row.get('hit@5')}; p50={row.get('wall_s_p50')} s; p95={row.get('wall_s_p95')} s; timing={row['timing_mode']}; invalid={row.get('invalid')}"
        )
    lines.extend(
        [
            "",
            "## Completeness and provenance",
            "",
            output["manual_work_provenance"],
            f"Incomplete/failed steps: {output['incomplete_or_failed_steps']}",
            "",
            "Detailed data: summary.json, ablation_summary.csv, load_summary.csv, stage_timing_summary.csv, generation_quality_summary.csv, softmatch_summary.csv, reviews/.",
        ]
    )
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(run / "summary.json", output)


def model(name: str):
    from pcb_rag.api_clients import build_llm

    return build_llm(name, temperature=0.2)


def ask_json(llm, prompt: str) -> dict:
    from eval.metrics import _ask_json

    for attempt in range(3):
        result = _ask_json(llm, prompt)
        if isinstance(result, dict):
            return result
        if attempt < 2:
            time.sleep(2)
    raise RuntimeError("AI review returned no usable JSON in three attempts")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stats(run: Path):
    import psutil
    from pcb_rag.query import build_index, _load_or_build_bm25

    rows = list(corpus().values())
    lengths = sorted(len(row["text"]) for row in rows)
    process = psutil.Process()
    index = build_index(os.environ["LLM_MODEL"])
    before = process.memory_info().rss
    start = time.perf_counter()
    bm25 = _load_or_build_bm25(index.storage_context.vector_store)
    elapsed = time.perf_counter() - start
    report = {
        "created_at": now(),
        "collection": os.environ["COLLECTION"],
        "chunks": len(rows),
        "documents": len({(row.get("metadata") or {}).get("source_path") for row in rows}),
        "chunk_chars_min": min(lengths),
        "chunk_chars_max": max(lengths),
        "chunk_chars_mean": statistics.fmean(lengths),
        "chunk_chars_median": statistics.median(lengths),
        "chunk_length_scope": "stored text, including contextual prefix; no new chunking",
        "bm25_docs": len(bm25.docs),
        "bm25_build_from_frozen_cache_s": elapsed,
        "bm25_rss_delta_bytes": max(0, process.memory_info().rss - before),
        "full_ingest_elapsed_s": None,
        "full_ingest_note": "Not re-ingested: corpus is frozen; existing input-mode migration measured separately.",
        "gpu_inference": "external APIs; remote GPU utilization and memory unavailable",
    }
    write_json(run / "diagnostics/corpus_stats.json", report)


def generate(run: Path):
    from eval.build_golden_dataset import _GEN_PROMPT

    llm = model(os.environ.get("GENERATOR_MODEL", "deepseek-v4-1-261017"))
    path = run / "datasets/golden_raw.jsonl"
    completed = read_jsonl(path)
    used = {row["source_chunk_id"] for row in completed}
    chunks = [row for row in corpus().values() if len(row["text"]) >= 200]
    random.Random(42).shuffle(chunks)
    for chunk in chunks:
        if len(completed) >= 100:
            break
        if chunk["id"] in used:
            continue
        try:
            result = ask_json(llm, _GEN_PROMPT.format(n=1, chunk=chunk["text"][:2500]))
            item = next(
                item
                for item in result["items"]
                if len(str(item.get("question", ""))) >= 6 and len(str(item.get("ground_truth", ""))) >= 4
            )
            record = {
                "id": f"new-{len(completed) + 1:04d}",
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "source_chunk_id": chunk["id"],
                "source_path": chunk.get("metadata", {}).get("source_path", ""),
                "generator_model": llm.model,
                "authored_by": "AI",
            }
            append_jsonl(path, record)
            completed.append(record)
            print(f"[generated {len(completed)}/100]", flush=True)
        except Exception as exc:
            append_jsonl(
                run / "quality/generation_errors.jsonl", {"chunk_id": chunk["id"], "error": str(exc), "at": now()}
            )
    if len(completed) != 100:
        raise RuntimeError("Could not generate 100 valid questions")
    write_json(
        run / "datasets/golden_provenance.json",
        {
            "generator_model": llm.model,
            "answer_model": os.environ["LLM_MODEL"],
            "judge_model": os.environ.get("JUDGE_MODEL", "glm-5-3-260814"),
            "raw_count": len(completed),
            "review_provenance": "AI review replacing the requested manual review, explicitly authorized by user",
        },
    )


def audit_golden(run: Path):
    raw = read_jsonl(run / "datasets/golden_raw.jsonl")
    chunks = corpus()
    sample = random.Random(42).sample(raw, 30)
    path = run / "reviews/golden_audit.jsonl"
    completed = read_jsonl(path)
    known = {row["id"] for row in completed}
    llm = model(os.environ.get("JUDGE_MODEL", "glm-5-3-260814"))
    for row in sample:
        if row["id"] in known:
            continue
        prompt = (
            "逐条审核下面的 RAG 测试样本。判断问题是否自然、能否仅依赖原文回答、参考答案是否被原文支持。"
            "不要因为原文噪声或标题就默认通过。只输出 JSON："
            '{"natural":true,"answerable":true,"reference_supported":true,"keep":true,"reason":"具体依据"}。\n'
            + json.dumps(
                {
                    "question": row["question"],
                    "reference": row["ground_truth"],
                    "source_chunk": chunks[row["source_chunk_id"]]["text"],
                },
                ensure_ascii=False,
            )
        )
        result = ask_json(llm, prompt)
        flags = [result.get(key) for key in ["natural", "answerable", "reference_supported", "keep"]]
        if not all(isinstance(flag, bool) for flag in flags):
            raise RuntimeError("Audit booleans missing")
        keep = all(flags)
        audit = {
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "source_chunk_id": row["source_chunk_id"],
            "reviewer_kind": "AI",
            "reviewer_model": llm.model,
            "keep": keep,
            "review": result,
        }
        append_jsonl(path, audit)
        completed.append(audit)
        print(f"[golden reviewed {len(completed)}/30] keep={keep}", flush=True)
    rejected = {row["id"] for row in completed if not row["keep"]}
    audited = [row for row in raw if row["id"] not in rejected]
    target = run / "datasets/golden_audited.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audited), encoding="utf-8")
    write_json(
        run / "reviews/golden_audit_summary.json",
        {
            "reviewer_kind": "AI",
            "reviewer_model": llm.model,
            "raw": len(raw),
            "reviewed": len(completed),
            "rejected": len(rejected),
            "rejection_rate": len(rejected) / len(completed),
            "evaluation_rows": len(audited),
            "unreviewed_rows": len(raw) - len(completed),
        },
    )


def parallel_judge(question: str, answer: str, contexts: list[str], reference: str, llm) -> dict:
    """Same prompts and formulas as eval.metrics; independent judge calls overlap."""
    from eval import metrics as metrics

    def relevant(context):
        result = metrics._ask_json(
            llm,
            metrics._CONTEXT_RELEVANT_PROMPT.format(
                question=metrics._short(question, 1000), context=metrics._short(context, 2000)
            ),
        )
        return bool(result["relevant"]) if isinstance(result, dict) and "relevant" in result else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            "faithfulness": pool.submit(metrics.faithfulness, answer, contexts, llm),
            "answer_relevancy": pool.submit(metrics.answer_relevancy, question, answer, llm),
            "context_recall": pool.submit(metrics.context_recall, reference, contexts, llm),
        }
        flags = list(pool.map(relevant, contexts))
        result = {key: future.result() for key, future in futures.items()}
    if not contexts:
        precision = 0.0
    elif any(flag is None for flag in flags):
        precision = None
    elif not any(flags):
        precision = 0.0
    else:
        hit = 0
        terms = []
        for index, flag in enumerate(flags, 1):
            if flag:
                hit += 1
                terms.append(hit / index)
        precision = sum(terms) / sum(flags)
    result["context_precision"] = precision
    values = [value for value in result.values() if isinstance(value, (int, float))]
    if values:
        result["overall"] = round(statistics.fmean(values), 4)
    result["judge_failed"] = [key for key, value in result.items() if value is None]
    return result


def quality(run: Path):
    from pcb_rag.rerank_control import RerankStopped
    from eval.evaluate import EvalEngine, _node_text
    from eval.metrics import summarize
    from pcb_rag.query import generate_answer_with_citation

    dataset = read_jsonl(run / "datasets/golden_audited.jsonl")
    rows_path = run / "quality/g4_rows.jsonl"
    completed = read_jsonl(rows_path)
    known = {row["id"] for row in completed}
    engine = EvalEngine(recall_k=200, top_k=5)
    judge = model(os.environ.get("JUDGE_MODEL", "glm-5-3-260814"))
    generator = os.environ.get("GENERATOR_MODEL", "deepseek-v4-1-261017")
    if len({engine.llm.model, judge.model, generator}) != 3:
        raise RuntimeError("Generation, answering and judging must use three distinct models")
    for row in dataset:
        if row["id"] in known:
            continue
        started = time.perf_counter()
        result = {"id": row["id"], "question": row["question"], "ground_truth": row["ground_truth"]}
        try:
            nodes = engine.retrieve(row["question"])
            contexts = [_node_text(node.node) for node in nodes]
            retrieved_at = time.perf_counter()
            answer = generate_answer_with_citation(row["question"], nodes, engine.llm)
            answered_at = time.perf_counter()
            scores = parallel_judge(row["question"], answer["answer"], contexts, row["ground_truth"], judge)
            result.update(
                answer=answer["answer"],
                citations=answer.get("citations", []),
                contexts=contexts,
                scores=scores,
                retrieved_sources=[node.node.metadata.get("source_path") for node in nodes],
                retrieval_s=retrieved_at - started,
                generation_s=answered_at - retrieved_at,
                judge_s=time.perf_counter() - answered_at,
                wall_s=time.perf_counter() - started,
            )
        except RerankStopped as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(rows_path, result)
            raise
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        append_jsonl(rows_path, result)
        completed.append(result)
        print(
            f"[quality {len(completed)}/{len(dataset)}] {result.get('scores', {}).get('overall', result.get('error'))}",
            flush=True,
        )
        write_json(
            run / "quality/g4_progress.json", {"completed": len(completed), "total": len(dataset), "updated_at": now()}
        )
    summary = summarize([row["scores"] for row in completed if row.get("scores")])
    invalid_metrics = [
        key for key, value in summary.items() if isinstance(value, dict) and value.get("failure_rate", 0) > 0.2
    ]
    report = {
        "config": {
            "dataset": "golden_audited.jsonl",
            "dataset_sha256": digest(run / "datasets/golden_audited.jsonl"),
            "samples": len(completed),
            "top_k": 5,
            "recall_k": 200,
            "retrieval_path": "eval.evaluate.EvalEngine (documented legacy quality-evaluation path)",
            "generator_model": generator,
            "answer_model": engine.llm.model,
            "judge_model": judge.model,
            "judge_parallel_workers": 8,
            "judge_protocol": "original prompts and AP/statement formulas; independent calls concurrent",
        },
        "summary": summary,
        "invalid_metrics": invalid_metrics,
        "request_errors": sum(bool(row.get("error")) for row in completed),
        "rows": completed,
    }
    write_json(run / "quality/g4_gen_quality.json", report)


def softmatch(run: Path):
    import numpy as np
    from pcb_rag.api_clients import build_embed_model, get_embedding_profile

    data = json.loads((run / "retrieval/r1_baseline_evalscript.json").read_text(encoding="utf-8"))
    rows = data["results"]
    texts = []
    for row in rows:
        gt = row["gt_texts"] if isinstance(row["gt_texts"], list) else [row["gt_texts"]]
        texts.extend(str(text)[:2000] for text in gt if text)
        texts.extend(hit["text"][:2000] for hit in row["got"][:10])
    unique = list(dict.fromkeys(texts))
    cache_path = run / "diagnostics/soft_vectors.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        embedder = build_embed_model(backend="api")
        embedder.embed_batch_size = 64
        vectors = []
        for start in range(0, len(unique), 64):
            vectors.extend(embedder.get_text_embedding_batch(unique[start : start + 64]))
            print(f"[soft embedded {min(start + 64, len(unique))}/{len(unique)}]", flush=True)
            time.sleep(2)
        cache = dict(zip(unique, vectors))
        write_json(cache_path, cache)
    per_query = []
    differences = []
    for row in rows:
        gt = row["gt_texts"] if isinstance(row["gt_texts"], list) else [row["gt_texts"]]
        got = row["got"][:10]
        sims = []
        for hit in got:
            vector = np.asarray(cache[hit["text"][:2000]])
            scores = []
            for text in gt:
                if not text:
                    continue
                reference = np.asarray(cache[str(text)[:2000]])
                scores.append(float(vector @ reference / (np.linalg.norm(vector) * np.linalg.norm(reference))))
            sims.append(max(scores, default=0.0))
        hard = next((i for i, hit in enumerate(got, 1) if hit["id"] in row["gt_ids"]), 0)
        ranks = {
            str(threshold): next((i for i, sim in enumerate(sims, 1) if sim >= threshold), 0)
            for threshold in [0.7, 0.78, 0.85, 0.9]
        }
        per_query.append({"query": row["query"], "hard_rank": hard, "soft_ranks": ranks, "similarities": sims})
        if ranks["0.78"] and not hard:
            hit = got[ranks["0.78"] - 1]
            differences.append(
                {
                    "id": f"diff-{len(differences) + 1:03d}",
                    "query": row["query"],
                    "gt_ids": row["gt_ids"],
                    "gt_texts": gt,
                    "got": hit,
                    "soft_rank": ranks["0.78"],
                    "cosine": sims[ranks["0.78"] - 1],
                }
            )

    def metrics(ranks):
        n = len(ranks)
        return {
            "n": n,
            "mrr": sum(1 / rank for rank in ranks if rank) / n,
            **{f"hit@{k}": sum(0 < rank <= k for rank in ranks) / n for k in [1, 5, 10]},
        }

    report = {
        "retrieval_source": "r1_baseline_evalscript.json; actual results reused without repeating retrieval",
        "embedding_profile": get_embedding_profile(),
        "hard": metrics([row["hard_rank"] for row in per_query]),
        "soft": {
            str(threshold): metrics([row["soft_ranks"][str(threshold)] for row in per_query])
            for threshold in [0.7, 0.78, 0.85, 0.9]
        },
        "softdiff_count": len(differences),
        "per_query": per_query,
    }
    write_json(run / "diagnostics/g5_softmatch_scan.json", report)
    write_json(run / "reviews/g5_softdiff_samples.json", differences)


def review_soft(run: Path):
    data = json.loads((run / "reviews/g5_softdiff_samples.json").read_text(encoding="utf-8"))
    sample = random.Random(42).sample(data, min(20, len(data)))
    chunks = corpus()
    path = run / "reviews/g5_softdiff_reviewed.jsonl"
    completed = read_jsonl(path)
    known = {row["id"] for row in completed}
    llm = model(os.environ.get("JUDGE_MODEL", "glm-5-3-260814"))
    for row in sample:
        if row["id"] in known:
            continue
        gt_rows = [chunks[node_id] for node_id in row["gt_ids"] if node_id in chunks]
        got = row["got"]
        adjacent = any(
            got["id"] in [chunk.get("metadata", {}).get("prev_id"), chunk.get("metadata", {}).get("next_id")]
            for chunk in gt_rows
        )
        prompt = (
            "审核软匹配但没有精确ID命中的检索样本。只因语义相近或共享标题不能算等价。"
            "分类：A=同文档相邻切块且答案信息等价；B=不同文档重复条款/重复讲解且答案信息等价；"
            "C=来源不同但实质语义等价；D=错误软匹配、缺少关键答案或矛盾。"
            '输出 JSON：{"category":"A/B/C/D","equivalent":true,"reason":"指出具体答案信息","confidence":"high/medium/low"}。\n'
            + json.dumps(
                {"question": row["query"], "retrieved": got, "references": gt_rows, "adjacent_ids": adjacent},
                ensure_ascii=False,
            )
        )
        result = ask_json(llm, prompt)
        if result.get("category") not in ["A", "B", "C", "D"]:
            raise RuntimeError("Invalid soft-difference category")
        record = {"id": row["id"], "query": row["query"], "reviewer_kind": "AI", "reviewer_model": llm.model, **result}
        append_jsonl(path, record)
        completed.append(record)
        print(f"[soft reviewed {len(completed)}/{len(sample)}] {result['category']}", flush=True)
    counts = dict(Counter(row["category"] for row in completed))
    write_json(
        run / "reviews/g5_review_summary.json",
        {
            "reviewer_kind": "AI",
            "available_differences": len(data),
            "reviewed": len(completed),
            "categories": counts,
            "A_plus_B_share": (counts.get("A", 0) + counts.get("B", 0)) / len(completed) if completed else None,
        },
    )


def scenarios(run: Path):
    source = json.loads((ROOT / "scripts/experiment_scenarios.json").read_text(encoding="utf-8"))
    dataset = [
        {"query": row["question"], "ground_truth_ids": [], "scenario_type": row["type"], "scenario_id": row["id"]}
        for row in source["questions"]
    ]
    write_json(run / "datasets/scenarios.json", dataset)
    write_json(run / "datasets/scenarios_provenance.json", source)


def review_scenarios(run: Path):
    report = json.loads((run / "retrieval/g6_scenarios.json").read_text(encoding="utf-8"))
    dataset = json.loads((run / "datasets/scenarios.json").read_text(encoding="utf-8"))
    path = run / "reviews/g6_scenario_review.jsonl"
    completed = read_jsonl(path)
    known = {row["index"] for row in completed}
    llm = model(os.environ.get("JUDGE_MODEL", "glm-5-3-260814"))
    for row, source in zip(report["results"], dataset):
        if row["index"] in known:
            continue
        contexts = row["got"][:5]
        prompt = (
            "审核一个拟真工程提问的前5条检索资料。只有这些资料足以回答问题的核心需求，才将hit_at_5设为true；"
            "仅主题相同、缺少版本/参数/关键步骤不能通过。多段联合可以通过，指出用到的资料编号。"
            '输出JSON：{"hit_at_5":true,"supporting_indices":[1],"reason":"具体依据","missing_information":"没有则空串"}。\n'
            + json.dumps({"question": row["query"], "contexts": contexts}, ensure_ascii=False)
        )
        result = ask_json(llm, prompt)
        if not isinstance(result.get("hit_at_5"), bool):
            raise RuntimeError("Invalid scenario hit label")
        record = {
            "index": row["index"],
            "scenario_type": source["scenario_type"],
            "query": row["query"],
            "reviewer_kind": "AI",
            "reviewer_model": llm.model,
            **result,
        }
        append_jsonl(path, record)
        completed.append(record)
        print(f"[scenario reviewed {len(completed)}/{len(dataset)}] hit={result['hit_at_5']}", flush=True)
    summary = {
        "provenance": "AI-authored realistic scenarios and AI review; not a human-origin test set",
        "n": len(completed),
        "ai_reviewed_hit@5": statistics.fmean(row["hit_at_5"] for row in completed),
        "by_intended_type": {
            kind: {
                "n": sum(row["scenario_type"] == kind for row in completed),
                "hit@5": statistics.fmean(row["hit_at_5"] for row in completed if row["scenario_type"] == kind),
            }
            for kind in sorted({row["scenario_type"] for row in completed})
        },
    }
    write_json(run / "reviews/g6_summary.json", summary)
    write_csv(
        run / "reviews/g6_review.csv",
        completed,
        [
            "index",
            "scenario_type",
            "query",
            "hit_at_5",
            "reason",
            "missing_information",
            "reviewer_kind",
            "reviewer_model",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--task",
        choices=[
            "gate",
            "summary",
            "stats",
            "generate",
            "audit_golden",
            "quality",
            "softmatch",
            "review_soft",
            "scenarios",
            "review_scenarios",
        ],
        required=True,
    )
    args = parser.parse_args()
    isolate_project_env()
    from pcb_rag.env_loader import load_project_env

    load_project_env()
    run = Path(args.run_dir)
    assert_frozen_inputs(run, dict(os.environ))
    globals()[args.task](run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
