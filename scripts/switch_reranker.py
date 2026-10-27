#!/usr/bin/env python
"""Validate a private reranker configuration and create a separate frozen run.

Credentials are read from a mode-0600 file, never from command-line arguments.
The existing experiment supervisor must be stopped before applying a switch.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()
sys.path.insert(0, str(ROOT))
from scripts.experiment_common import (  # noqa: E402
    assert_frozen_inputs,
    digest,
    now,
    public_config,
    write_json,
)


def check_call(config: dict, documents: list[str]) -> dict:
    from pcb_rag.api_clients import ApiReranker

    started = time.perf_counter()
    limit = int(os.getenv("RERANK_MAX_CANDIDATES", "50"))
    original_count = len(documents)
    documents = documents[:limit]
    wanted = min(10, len(documents))
    result = {"candidate_count": len(documents), "original_candidate_count": original_count, "started_at": now()}
    try:
        client = ApiReranker(
            api_url=config["RERANK_API_URL"],
            api_key=config["RERANK_API_KEY"],
            model=config["RERANK_API_MODEL"],
            top_n=wanted,
            max_candidates=limit,
        )
        scores = client._request_scores("PCB差分信号走线时，如何保证阻抗连续性？", documents)
        indices = sorted((i for i, score in enumerate(scores) if score is not None), key=lambda i: -scores[i])
        result.update(
            requested_model=config["RERANK_API_MODEL"],
            returned_results=len(indices),
            status_code=200,
            finite_scores=all(math.isfinite(scores[i]) for i in indices),
            top_indices=indices[:3],
        )
        result["ok"] = len(indices) == wanted and result["finite_scores"]
    except Exception as exc:
        result.update(ok=False, error_type=type(exc).__name__)
    result["wall_s"] = round(time.perf_counter() - started, 4)
    return result


def preflight(config: dict, previous: Path, parallel: int, output: Path) -> dict:
    manifest = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    corpus_path = Path(manifest["env"]["LEXICAL_CACHE_PATH"])
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    documents = []
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            documents.append(row["text"])
            if len(documents) == 200:
                break
    if len(documents) != 200:
        raise RuntimeError("Expected 200 frozen candidates for reranker validation")
    small = [
        "香蕉富含钾，适合做水果沙拉。",
        "PCB差分信号应保持线宽、线距和参考平面连续，减少过孔与支路，并按叠层计算差分阻抗。",
        "印制板丝印用于标记元件位号和装配信息。",
    ]
    semantic = check_call(config, small)
    semantic["expected_document_first"] = semantic.get("top_indices", [])[:1] == [1]
    semantic["ok"] = semantic["ok"] and semantic["expected_document_first"]
    large = check_call(config, documents)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        simultaneous = list(pool.map(lambda _: check_call(config, documents[:20]), range(parallel)))
    report = {
        "created_at": now(),
        "endpoint": config["RERANK_API_URL"],
        "model": config["RERANK_API_MODEL"],
        "semantic_probe": semantic,
        "candidate_cap_probe": large,
        "concurrency": parallel,
        "parallel_probes": simultaneous,
        "ok": semantic["ok"] and large["ok"] and all(row["ok"] for row in simultaneous),
        "note": "All probes use the guarded client, input cap, usage ledger and budget.",
    }
    write_json(output, report)
    return report


def private_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def apply_switch(config: dict, previous: Path, run: Path, parallel: int, probe: Path) -> dict:
    report = json.loads(probe.read_text(encoding="utf-8"))
    if not report.get("ok") or report["model"] != config["RERANK_API_MODEL"]:
        raise RuntimeError("A successful matching preflight is required")
    if report["endpoint"] != config["RERANK_API_URL"] or report["concurrency"] < parallel:
        raise RuntimeError("Preflight endpoint or concurrency does not match")
    if previous == run or run.exists():
        raise RuntimeError("The new run must have a fresh, separate directory")
    session = subprocess.run(["tmux", "has-session", "-t", "pcb-rag-experiments"], capture_output=True)
    if session.returncode == 0:
        raise RuntimeError("Stop the previous experiment tmux session before switching")
    with (previous / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
        env_path = Path(old["private_env_path"])
        frozen = json.loads(env_path.read_text(encoding="utf-8"))
        assert_frozen_inputs(previous, frozen)
        private_env = env_path.parent / (run.name + ".json")
        if private_env.exists():
            raise RuntimeError("A private environment for the new run already exists")
        changed = {
            "RERANK_BACKEND": "api",
            "RERANK_ENABLED": "1",
            **config,
            "RERANK_MIN_INTERVAL": frozen.get("RERANK_MIN_INTERVAL", "6.2"),
            "RERANK_429_RETRIES": frozen.get("RERANK_429_RETRIES", "6"),
        }
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = env_path.parent / ("rerank-switch-backup-" + stamp)
        backup_dir.mkdir(mode=0o700)
        for name in [".env", ".env.eval"]:
            source = ROOT / name
            backup = backup_dir / (name + ".before")
            backup.write_bytes(source.read_bytes())
            backup.chmod(0o600)
            remaining = dict(changed)
            lines = []
            for line in source.read_text(encoding="utf-8").splitlines():
                match = re.match(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=", line)
                if match and match[1] in changed:
                    key = match[1]
                    if key in remaining:
                        lines.append(f"{key}={remaining.pop(key)}")
                else:
                    lines.append(line)
            lines.extend(f"{key}={value}" for key, value in remaining.items())
            temporary = source.with_suffix(source.suffix + ".tmp")
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temporary.replace(source)
        frozen.update(changed)
        frozen.update(
            PCB_RAG_EXPERIMENT_FROZEN="1",
            PCB_RAG_EXPERIMENT_MAX_PARALLEL=str(parallel),
        )
        private_json(private_env, frozen)
        new = copy.deepcopy(old)
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
            new.pop(key, None)
        new.update(
            run_id=run.name,
            started_at=now(),
            formal_started_at=now(),
            status="ready",
            private_env_path=str(private_env),
            private_env_sha256=digest(private_env),
            env=public_config(frozen),
            protocol_sha256=digest(ROOT / "TestRule.md"),
            max_parallel_steps=parallel,
            predecessor_run=previous.name,
            configuration_change={
                "old_rerank_model": old["env"].get("RERANK_API_MODEL"),
                "new_rerank_model": config["RERANK_API_MODEL"],
                "endpoint": config["RERANK_API_URL"],
                "old_results_reused": False,
                "corpus_and_regression_unchanged": True,
            },
            scheduling={
                "max_parallel_steps": parallel,
                "parallel_stages": ["r2"],
                "exclusive_stages": "All other stages, including both baselines and g3 load testing",
                "latency_note": "Parallel ablation timings include shared provider/local resource contention; use exclusive baselines and g3 for isolated latency claims.",
                "user_authorized_parallel_experiments": True,
            },
            orchestration_fingerprints={
                str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "scripts").glob("experiment_*.py"))
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
        }
        for job in new["steps"]:
            for key in runtime:
                job.pop(key, None)
            job.update(status="pending", parallel_safe=job["stage"] == "r2")
        for directory in ["retrieval", "logs", "datasets", "quality", "reviews", "load", "diagnostics"]:
            (run / directory).mkdir(parents=True)
        for path in (previous / "datasets").glob("regression_v1*"):
            shutil.copy2(path, run / "datasets" / path.name)
        shutil.copy2(probe, run / "diagnostics/reranker_preflight.json")
        write_json(run / "manifest.json", new)
        write_json(run / "heartbeat.json", {"updated_at": now(), "status": "ready", "active_steps": []})
        old.update(status="superseded", stopped_at=now(), superseded_by=run.name)
        old["supersession_reason"] = (
            "User requested a different reranker and parallel ablations; old-provider results retained separately."
        )
        for job in old["steps"]:
            if job["status"] == "running":
                job.update(
                    status="interrupted",
                    interrupted_at=now(),
                    interruption_note="Reranker changed by user; partial rows retained.",
                )
        write_json(previous / "manifest.json", old)
        write_json(
            previous / "heartbeat.json", {"updated_at": now(), "status": "superseded", "superseded_by": run.name}
        )
        from scripts.experiment_queue import export_progress

        export_progress(previous, old)
        export_progress(run, new)
        return {
            "new_run": str(run),
            "old_run_status": "superseded",
            "steps": len(new["steps"]),
            "max_parallel_ablations": parallel,
            "model": config["RERANK_API_MODEL"],
            "endpoint": config["RERANK_API_URL"],
            "private_backups": str(backup_dir),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["check", "apply"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--probe-report", type=Path, default=ROOT / "logs/rerank-voyage-preflight.json")
    args = parser.parse_args()
    if args.parallel < 1 or args.parallel > 3:
        parser.error("This validated switch supports 1 to 3 simultaneous experiments")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(config) != {"RERANK_API_URL", "RERANK_API_MODEL", "RERANK_API_KEY"}:
        raise RuntimeError("The private file must contain exactly the three reranker configuration values")
    if any(not isinstance(value, str) or not value or "\n" in value for value in config.values()):
        raise RuntimeError("Invalid private reranker configuration")
    if args.phase == "check":
        import uuid

        os.environ.setdefault("RERANK_BUDGET_SCOPE", "rerank-preflight-" + uuid.uuid4().hex)
        os.environ.setdefault("RERANK_TOKEN_BUDGET", "0")
        os.environ.setdefault("RERANK_REQUEST_BUDGET", "0")
        result = preflight(config, args.previous_run.resolve(), args.parallel, args.probe_report)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required for apply")
        result = apply_switch(
            config, args.previous_run.resolve(), args.run_dir.resolve(), args.parallel, args.probe_report
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
