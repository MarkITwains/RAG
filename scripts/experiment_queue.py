#!/usr/bin/env python
"""Run frozen experiments with bounded parallelism and exclusive timing stages."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_common import now, write_json  # noqa: E402


_PROXY_KEYS = {
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
}


def frozen_base_env(frozen_env):
    """Keep host runtime essentials, never unfrozen project configuration."""
    from scripts.experiment_common import public_config

    project_keys = set(public_config(os.environ)) | {
        key for key in os.environ
        if key.startswith(("PCB_RAG_", "INGEST_", "SESSION_", "TRACE_", "OBSERVABILITY_"))
        or any(part in key for part in ("API_KEY", "API_TOKEN", "AUTH_TOKEN", "PASSWORD", "SECRET", "CREDENTIAL"))
    }
    base = {key: value for key, value in os.environ.items() if key not in project_keys or key in _PROXY_KEYS}
    base.update(frozen_env)
    base["PCB_RAG_EXPERIMENT_FROZEN"] = "1"
    return base


def load_frozen_env(path, expected_digest):
    from scripts.experiment_common import digest

    if not expected_digest:
        raise RuntimeError("Frozen environment digest is required; refusing to inherit project defaults")
    if digest(path) != expected_digest:
        raise RuntimeError("Frozen environment changed; refusing to resume with different configuration")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Frozen environment must be a JSON object")
    return value


def export_progress(run, manifest):
    output = {
        "run_id": manifest["run_id"],
        "updated_at": now(),
        "status": manifest.get("status"),
        "max_parallel_steps": manifest.get("max_parallel_steps", 1),
        "steps": [],
    }
    for job in manifest["steps"]:
        item = {
            key: job[key]
            for key in [
                "name",
                "stage",
                "status",
                "started_at",
                "finished_at",
                "exit_code",
                "wall_s",
                "parallel_safe",
                "max_observed_parallel_steps",
                "timing_mode",
            ]
            if key in job
        }
        path = run / "retrieval" / (job["name"] + ".json")
        if path.exists():
            item["summary"] = json.loads(path.read_text(encoding="utf-8")).get("summary")
        progress = path.with_suffix(".progress.json")
        if progress.exists():
            item["query_progress"] = json.loads(progress.read_text(encoding="utf-8"))
        output["steps"].append(item)
    write_json(run / "progress.json", output)


def can_start(job, active_jobs, maximum):
    """Only explicitly parallel-safe jobs may share experiment resources."""
    if len(active_jobs) >= maximum:
        return False
    if not active_jobs:
        return True
    return job.get("parallel_safe", False) and all(other.get("parallel_safe", False) for other in active_jobs)


def ready_jobs(manifest):
    states = {job["name"]: job["status"] for job in manifest["steps"]}
    return [
        job
        for job in manifest["steps"]
        if job["status"] in {"pending", "interrupted"}
        and all(
            states.get(dependency)
            in ({"done", "failed", "skipped"} if job.get("run_after_failed_dependencies") else {"done"})
            for dependency in job.get("depends_on", [])
        )
    ]


def probe_model(run, name, env, cache):
    previous = cache.get(name)
    if previous and time.monotonic() - previous["checked"] < 300:
        return previous["available"]
    started = time.monotonic()
    payload = {"model": name, "messages": [{"role": "user", "content": "Reply exactly OK."}], "max_tokens": 256}
    request = urllib.request.Request(
        env["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + env["LLM_API_KEY"], "Content-Type": "application/json"},
    )
    available = False
    entry = {"model": name, "at": now(), "max_tokens": 256}
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read())
            choice = data.get("choices", [{}])[0]
            content_ready = bool(choice.get("message", {}).get("content"))
            reasoning_ready = (
                choice.get("finish_reason") == "length" and data.get("usage", {}).get("completion_tokens", 0) > 0
            )
            available = content_ready or reasoning_ready
            entry.update(
                status=response.status, nonempty_content=content_ready, truncated_reasoning_output=reasoning_ready
            )
    except Exception as exc:
        entry.update(error_type=type(exc).__name__)
    entry.update(seconds=round(time.monotonic() - started, 3), available=available)
    with (run / "diagnostics/model_availability.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    cache[name] = {"checked": time.monotonic(), "available": available}
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--register-frozen-env", action="store_true", help="record the digest of an existing frozen env")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    manifest_path = run / "manifest.json"
    if args.register_frozen_env:
        from scripts.experiment_common import digest

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["private_env_sha256"] = digest(Path(manifest["private_env_path"]))
        write_json(manifest_path, manifest)
        print("Registered the existing frozen environment digest; configuration was not changed")
        return 0
    lock = (run / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "superseded":
        raise RuntimeError("This run was superseded; use its replacement run instead")
    if manifest.get("preflight_pilot_path") and manifest.get("pilot_status") != "complete":
        raise RuntimeError("The required small-sample pilot has not passed; formal queue refused")
    frozen_env = load_frozen_env(Path(manifest["private_env_path"]), manifest.get("private_env_sha256"))
    for job in manifest["steps"]:
        if job["status"] == "running":
            job.update(status="interrupted", interruption_note="Supervisor restarted; per-query checkpoint retained.")
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    base_env = frozen_base_env(frozen_env)
    availability = {}
    active = {}
    maximum = max(1, int(manifest.get("max_parallel_steps", 1)))
    from pcb_rag.rerank_control import RerankControl

    control = RerankControl.from_env(base_env) if base_env.get("RERANK_CONTROL_DB") else None

    def cleanup():
        for worker in active.values():
            process = worker["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def interrupted(signum, frame):
        cleanup()
        raise SystemExit(128 + signum)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # WSL NAT gateways may change across a resume; keep the current shell proxy.
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]:
        if key in os.environ:
            base_env[key] = os.environ[key]
    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if control is not None:
            usage = control.snapshot()
            write_json(run / "rerank_usage.json", usage)
            if usage["circuit_reason"]:
                cleanup()
                for name, worker in list(active.items()):
                    process = worker["process"]
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    worker["log"].close()
                    job = next(item for item in manifest["steps"] if item["name"] == name)
                    job.update(
                        status="done" if process.returncode == 0 else "interrupted",
                        exit_code=process.returncode,
                        interrupted_at=now(),
                        interruption_note="Shared rerank circuit opened: " + usage["circuit_reason"],
                    )
                    del active[name]
                manifest.update(status="halted", halted_at=now(), halt_reason=usage["circuit_reason"])
                write_json(manifest_path, manifest)
                export_progress(run, manifest)
                write_json(run / "diagnostics/rerank_usage_records.json", control.records())
                write_json(
                    run / "heartbeat.json",
                    {
                        "updated_at": now(),
                        "status": "halted",
                        "active_steps": [],
                        "reason": usage["circuit_reason"],
                    },
                )
                print("[halt] shared rerank circuit: " + usage["circuit_reason"], flush=True)
                return 3
        for name, worker in list(active.items()):
            process = worker["process"]
            if process.poll() is None and time.monotonic() - worker["started"] > worker["max_wall_s"]:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if process.poll() is not None:
                worker["log"].close()
                job = next(item for item in manifest["steps"] if item["name"] == name)
                job.update(
                    status="done" if process.returncode == 0 else "failed",
                    exit_code=process.returncode,
                    finished_at=now(),
                    wall_s=round(time.monotonic() - worker["started"], 3),
                    log=str(worker["log_path"]),
                )
                del active[name]
                print(f"[finish] {name} rc={process.returncode}", flush=True)
        states_before = {job["name"]: job["status"] for job in manifest["steps"]}
        for job in manifest["steps"]:
            if (
                not job.get("run_after_failed_dependencies")
                and job["status"] in {"pending", "interrupted"}
                and any(
                    states_before.get(dependency) in {"failed", "skipped"} for dependency in job.get("depends_on", [])
                )
            ):
                job.update(status="skipped", reason="A required predecessor failed; no result was fabricated.")
        write_json(manifest_path, manifest)
        states = {job["name"]: job["status"] for job in manifest["steps"]}
        for selected in ready_jobs(manifest):
            active_jobs = [job for job in manifest["steps"] if job["name"] in active]
            if not can_start(selected, active_jobs, maximum):
                continue
            if not all(probe_model(run, name, base_env, availability) for name in selected.get("required_models", [])):
                continue
            output = run / "retrieval" / (selected["name"] + ".json")
            if selected.get("argv"):
                command = [sys.executable, *[str(arg).replace("{run}", str(run)) for arg in selected["argv"]]]
            else:
                command = [
                    sys.executable,
                    str(ROOT / "scripts/experiment_retrieval.py"),
                    "--dataset",
                    str(run / "datasets/regression_v1.json"),
                    "--out",
                    str(output),
                    "--path",
                    selected["path"],
                    "--top-k",
                    str(selected["top_k"]),
                    "--warmup",
                    "3",
                ]
            timing_mode = "shared_resources" if selected.get("parallel_safe", False) and maximum > 1 else "exclusive"
            selected.update(status="running", started_at=now(), command=command, timing_mode=timing_mode)
            log_path = run / "logs" / (selected["name"] + ".log")
            log = log_path.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={
                        **base_env,
                        **selected.get("overrides", {}),
                        "PCB_RAG_EXPERIMENT_TIMING_MODE": timing_mode,
                        "RERANK_PHASE": selected["name"],
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                log.close()
                raise
            active[selected["name"]] = {
                "process": process,
                "started": time.monotonic(),
                "max_wall_s": selected.get("max_wall_s", 14400),
                "log": log,
                "log_path": log_path,
            }
            for job in manifest["steps"]:
                if job["name"] in active:
                    job["max_observed_parallel_steps"] = max(job.get("max_observed_parallel_steps", 1), len(active))
            write_json(manifest_path, manifest)
            print(f"[start] {selected['name']} active={len(active)}/{maximum} timing={timing_mode}", flush=True)
        write_json(manifest_path, manifest)
        export_progress(run, manifest)
        if not active:
            remaining = [job for job in manifest["steps"] if job["status"] not in {"done", "failed", "skipped"}]
            if manifest.get("plan_complete") and not remaining:
                manifest["status"] = (
                    "complete"
                    if not any(job["status"] == "failed" for job in manifest["steps"])
                    else "complete_with_failures"
                )
                manifest["finished_at"] = now()
                write_json(manifest_path, manifest)
                export_progress(run, manifest)
                write_json(
                    run / "heartbeat.json",
                    {
                        "updated_at": now(),
                        "status": manifest["status"],
                        "active_steps": [],
                        "finished_at": manifest["finished_at"],
                    },
                )
                if control is not None:
                    write_json(run / "rerank_usage.json", control.snapshot())
                    write_json(run / "diagnostics/rerank_usage_records.json", control.records())
                return 0 if manifest["status"] == "complete" else 2
            write_json(
                run / "heartbeat.json",
                {
                    "updated_at": now(),
                    "status": "awaiting_upstream_or_dependencies",
                    "steps": states,
                    "pid": os.getpid(),
                },
            )
        else:
            workers = [
                {
                    "name": name,
                    "child_pid": worker["process"].pid,
                    "elapsed_s": round(time.monotonic() - worker["started"], 1),
                    "log": str(worker["log_path"]),
                }
                for name, worker in active.items()
            ]
            write_json(
                run / "heartbeat.json",
                {
                    "updated_at": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "active_step": workers[0]["name"],
                    "active_steps": workers,
                    "max_parallel_steps": maximum,
                },
            )
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
