#!/usr/bin/env python
"""Restore SiliconFlow privately and prepare/pilot/launch a guarded frozen run."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_common import digest, now, public_config, write_json  # noqa: E402
from scripts.experiment_queue import export_progress  # noqa: E402

DEFAULT_RUN = ROOT / "eval/reports/rerun_20261020_siliconflow_guarded"


def write_private(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def update_env(path: Path, updates: dict, backup: Path):
    backup.write_bytes(path.read_bytes())
    backup.chmod(0o600)
    remaining = dict(updates)
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=", line)
        if match and match[1] in updates:
            key = match[1]
            if key in remaining:
                lines.append(f"{key}={remaining.pop(key)}")
        else:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in remaining.items())
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def scaffold(run: Path, manifest: dict, env: dict, source: Path):
    private = Path.home() / ".config/pcb-rag/experiments" / (manifest["run_id"] + ".json")
    if run.exists() or private.exists():
        raise RuntimeError("The run or its private environment already exists")
    for folder in ["retrieval", "logs", "datasets", "quality", "reviews", "load", "diagnostics"]:
        (run / folder).mkdir(parents=True)
    for path in (source / "datasets").glob("regression_v1*"):
        shutil.copy2(path, run / "datasets" / path.name)
    write_private(private, env)
    manifest.update(private_env_path=str(private), env=public_config(env))
    write_json(run / "manifest.json", manifest)
    write_json(run / "heartbeat.json", {"updated_at": now(), "status": "ready", "active_steps": []})
    export_progress(run, manifest)


def prepare(run: Path):
    if subprocess.run(["tmux", "has-session", "-t", "pcb-rag-experiments"], capture_output=True).returncode == 0:
        raise RuntimeError("An experiment queue is already running")
    source = ROOT / "eval/reports/rerun_20261020_api_nvidia_fixed"
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    env = json.loads(Path(original["private_env_path"]).read_text(encoding="utf-8"))
    config = json.loads((Path.home() / ".config/pcb-rag/rerank-siliconflow.json").read_text(encoding="utf-8"))
    assert config["RERANK_API_URL"] == "https://api.siliconflow.cn/v1/rerank"
    assert config["RERANK_API_MODEL"] == "BAAI/bge-reranker-v2-m3"
    cache = Path(env["LEXICAL_CACHE_PATH"])
    if not cache.is_absolute():
        cache = ROOT / cache
    if digest(cache) != original["lexical_sha256"]:
        raise RuntimeError("The lexical corpus changed")
    if digest(source / "datasets/regression_v1.json") != original["dataset_sha256"]:
        raise RuntimeError("The regression dataset changed")
    from pymilvus import MilvusClient
    from scripts.repair_embeddings import corpus_digest, read_rows

    client = MilvusClient(uri=env["MILVUS_URI"])
    try:
        rows = read_rows(client, original["collection"])
    finally:
        client.close()
    if corpus_digest(rows, "embedding") != original["corpus_nonvector_sha256"]:
        raise RuntimeError("The frozen collection changed")
    limits = {
        "RERANK_BACKEND": "api",
        "RERANK_ENABLED": "1",
        **config,
        "RERANK_MODEL": config["RERANK_API_MODEL"],
        "RERANK_TOP_N": "10",
        "RERANK_MAX_CANDIDATES": "50",
        "RERANK_MAX_DOCUMENT_CHARS": "1200",
        "RERANK_MAX_QUERY_CHARS": "1000",
        "RERANK_MIN_INTERVAL": "0",
        "RERANK_429_RETRIES": "2",
        "RERANK_429_BACKOFF_SECONDS": "60",
        "RERANK_CONTROL_DB": str(Path.home() / ".local/share/pcb-rag/rerank/control.sqlite3"),
    }
    env.update(limits)
    env.update(
        RERANK_BUDGET_SCOPE=run.name,
        RERANK_TOKEN_BUDGET="0",
        RERANK_REQUEST_BUDGET="0",
        PCB_RAG_EXPERIMENT_FROZEN="1",
        PCB_RAG_EXPERIMENT_MAX_PARALLEL="3",
    )
    for key in ["RERANK_ACTIVITY", "RERANK_PHASE"]:
        env.pop(key, None)
    manifest = copy.deepcopy(original)
    for key in [
        "stopped_at",
        "finished_at",
        "superseded_by",
        "supersession_reason",
        "preflight_archive",
        "inference_preflight",
        "upstream_availability",
        "execution_order_note",
    ]:
        manifest.pop(key, None)
    manifest.update(
        run_id=run.name,
        started_at=now(),
        formal_started_at=now(),
        status="ready",
        max_parallel_steps=3,
        predecessor_run="rerun_20261020_api_nvidia_voyage25",
        restored_config_from=source.name,
        code_fingerprints={
            str(path.relative_to(ROOT)): digest(path)
            for folder in [ROOT / "src/pcb_rag", ROOT / "eval"]
            for path in sorted(folder.glob("*.py"))
        },
        orchestration_fingerprints={
            str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "scripts").glob("experiment_*.py"))
        },
        protocol_sha256=digest(ROOT / "TestRule.md"),
        rerank_protocol={
            "recall_before_rerank": 200,
            "sent_candidate_cap": 50,
            "returned_top_n": 10,
            "document_serialization": "MetadataMode.NONE; contextual text prefixes preserved",
            "max_document_chars": 1200,
            "max_query_chars": 1000,
            "proactive_rate_limit_seconds": 0,
            "bounded_429_retries": 2,
            "formal_token_budget": 0,
            "request_budget": 0,
            "unknown_usage_policy": "Conservative UTF-8 pair reservation, separately labeled from reported tokens",
            "protocol_change": "User requested cost controls, restoring SiliconFlow, and an initially unthrottled test. Old ranking results are not reused.",
        },
        scheduling={
            "max_parallel_steps": 3,
            "parallel_stages": ["r1_baselines", "r2"],
            "exclusive_stages": "Gate, corpus statistics, g3 load testing and auxiliary stages",
            "latency_note": "Baselines/ablations share resources by user request. Use the exclusive g3 load tests for isolated latency claims.",
        },
    )
    runtime = {
        "started_at",
        "finished_at",
        "exit_code",
        "wall_s",
        "command",
        "log",
        "deferred_at",
        "deferral_reason",
        "interrupted_at",
        "interruption_note",
        "reason",
        "max_observed_parallel_steps",
        "timing_mode",
    }
    for job in manifest["steps"]:
        for key in runtime:
            job.pop(key, None)
        job.update(status="pending", parallel_safe=job["stage"] == "r2" or job["name"].startswith("r1_baseline_"))
    pilot = run / "diagnostics/pilot"
    manifest["preflight_pilot_path"] = str(pilot)
    scaffold(run, manifest, env, source)
    pilot_env = {
        **env,
        "RERANK_BUDGET_SCOPE": run.name + "_pilot",
        "RERANK_TOKEN_BUDGET": "0",
        "RERANK_REQUEST_BUDGET": "0",
        "RERANK_429_RETRIES": "0",
    }
    pilot_manifest = copy.deepcopy(manifest)
    pilot_manifest.update(run_id=run.name + "_pilot", started_at=now(), formal_started_at=now(), status="ready")
    pilot_manifest.pop("preflight_pilot_path", None)
    pilot_manifest["steps"] = []
    for old_name in ["02_vector_rerank", "03_fusion_rerank", "04_expansion"]:
        old_job = next(job for job in manifest["steps"] if job["name"] == old_name)
        name = "pilot_" + old_name
        pilot_manifest["steps"].append(
            {
                "name": name,
                "stage": "pilot",
                "status": "pending",
                "parallel_safe": True,
                "required_models": [],
                "overrides": old_job["overrides"],
                "argv": [
                    "scripts/experiment_retrieval.py",
                    "--dataset",
                    "{run}/datasets/regression_v1.json",
                    "--out",
                    "{run}/retrieval/" + name + ".json",
                    "--path",
                    "production",
                    "--top-k",
                    "10",
                    "--warmup",
                    "0",
                    "--limit",
                    "3",
                ],
            }
        )
    scaffold(pilot, pilot_manifest, pilot_env, source)
    backup_dir = Path.home() / ".config/pcb-rag/experiments" / ("guarded-backup-" + time.strftime("%Y%m%d-%H%M%S"))
    backup_dir.mkdir(mode=0o700)
    production = {**limits, "RERANK_TOKEN_BUDGET": "0", "RERANK_REQUEST_BUDGET": "0"}
    update_env(ROOT / ".env", production, backup_dir / ".env.before")
    update_env(
        ROOT / ".env.eval",
        {
            **limits,
            "RERANK_BUDGET_SCOPE": env["RERANK_BUDGET_SCOPE"],
            "RERANK_TOKEN_BUDGET": env["RERANK_TOKEN_BUDGET"],
            "RERANK_REQUEST_BUDGET": env["RERANK_REQUEST_BUDGET"],
        },
        backup_dir / ".env.eval.before",
    )
    preflight = ROOT / "logs/siliconflow-guarded-preflight.json"
    if preflight.exists():
        shutil.copy2(preflight, run / "diagnostics/initial_small_provider_probe.json")
    write_json(
        run / "diagnostics/preparation.json",
        {
            "at": now(),
            "corpus_rows": len(rows),
            "corpus_unchanged": True,
            "regression_rows": original["regression_rows"],
            "old_results_reused": False,
            "private_backups": str(backup_dir),
        },
    )
    (run / "HANDOFF.md").write_text(
        "# Guarded SiliconFlow rerun\n\n"
        "The user requested all rerank cost controls, restoration of SiliconFlow BAAI/bge-reranker-v2-m3, and a fresh test.\n"
        "Latest steering: proactive rate limiting is disabled (interval=0); first validate real provider behavior with a small, budgeted parallel pilot.\n"
        "50 input candidates maximum; API/local top_n=10 (top50 ablation explicitly changes output); plain text only; no technical metadata.\n"
        "Total token and request limits are disabled (0 means unlimited), including production and pilots; usage accounting remains enabled.\n"
        "Provider usage, warmups, retries, unknown billing, and input counts are durable in a WSL SQLite ledger.\n"
        "402/auth/budget/transport/protocol errors open a shared circuit; no silent ranking fallback; the queue halts all workers.\n"
        "Only a successful pilot permits formal launch. Existing Voyage/old SiliconFlow rows are never mixed into this batch.\n"
        "Keep all src/pcb_rag, eval and experiment scripts frozen after preparation; fingerprints are enforced.\n"
        "All inference remains external. WSL proxy port 12345, NVIDIA query/passage vector collection unchanged.\n"
        "AI substitute reviews/scenarios were explicitly authorized; never label them human-reviewed or human-authored.\n"
        "Read manifest.json/heartbeat.json/progress.json/rerank_usage.json for actual current state. Never print private environment files or keys.\n",
        encoding="utf-8",
    )
    return {
        "run": str(run),
        "pilot": str(pilot),
        "model": config["RERANK_API_MODEL"],
        "candidate_cap": 50,
        "top_n": 10,
        "proactive_interval": 0,
        "formal_budget": 0,
    }


def run_pilot(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    pilot = Path(manifest["preflight_pilot_path"])
    manifest["status"] = "validating"
    write_json(run / "manifest.json", manifest)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment_queue.py"), "--run-dir", str(pilot)], cwd=ROOT
    )
    pilot_manifest = json.loads((pilot / "manifest.json").read_text())
    manifest["status"] = (
        "ready" if result.returncode == 0 and pilot_manifest["status"] == "complete" else "validation_failed"
    )
    manifest["pilot_status"] = pilot_manifest["status"]
    manifest["pilot_finished_at"] = now()
    write_json(run / "manifest.json", manifest)
    write_json(run / "heartbeat.json", {"updated_at": now(), "status": manifest["status"], "active_steps": []})
    export_progress(run, manifest)
    return {"pilot_status": pilot_manifest["status"], "exit_code": result.returncode}


def launch(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "ready" or manifest.get("pilot_status") != "complete":
        raise RuntimeError("The parallel, unthrottled pilot has not passed; formal launch is refused")
    if subprocess.run(["tmux", "has-session", "-t", "pcb-rag-experiments"], capture_output=True).returncode == 0:
        raise RuntimeError("An experiment queue is already running")
    pilot = Path(manifest["preflight_pilot_path"])
    reports = []
    for path in (pilot / "retrieval").glob("pilot_*.json"):
        if path.name.endswith((".meta.json", ".progress.json", ".failure.json")):
            continue
        data = json.loads(path.read_text())
        if data["summary"]["invalid"] or data["summary"]["n"] != 3:
            raise RuntimeError("Pilot contains failed/incomplete measurements")
        reports.append({"configuration": path.stem, "summary": data["summary"]})
    if len(reports) != 3:
        raise RuntimeError("Expected three completed pilot configurations")
    write_json(run / "diagnostics/pilot_gate.json", {"at": now(), "passed": True, "reports": reports})
    command = (
        "cd "
        + shlex.quote(str(ROOT))
        + " && . /home/pi/.config/pcb-rag/proxy.sh && exec "
        + shlex.join([sys.executable, str(ROOT / "scripts/experiment_queue.py"), "--run-dir", str(run)])
        + " >> "
        + shlex.quote(str(run / "queue.log"))
        + " 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", "pcb-rag-experiments", command], check=True)
    return {"started_at": now(), "run": run.name, "max_parallel": 3, "proactive_interval": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "pilot", "launch"])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = {"prepare": prepare, "pilot": run_pilot, "launch": launch}[args.action](args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
