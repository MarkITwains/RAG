#!/usr/bin/env python
"""Run six isolated fixed-concurrency load tests with metrics and resource data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_common import append_jsonl, now, write_json, assert_frozen_inputs, isolate_project_env  # noqa: E402


def main() -> int:
    import psutil
    import requests

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    isolate_project_env()
    run = Path(args.run_dir)
    assert_frozen_inputs(run, dict(os.environ))
    base = "http://127.0.0.1:18000"
    env = {
        **os.environ,
        "SEMANTIC_CACHE_ENABLED": "0",
        "EXPERIMENT_PORT": "18000",
        "TRACE_LOG_PATH": str(run / "load/traces.jsonl"),
    }
    for endpoint in ["retrieval", "ask"]:
        for concurrency in [1, 4, 8]:
            name = f"g3_{endpoint}_c{concurrency}"
            out = run / "load" / (name + ".json")
            if out.exists():
                continue
            if out.with_suffix(".jsonl").exists():
                attempt = str(int(time.time()))
                out.with_suffix(".jsonl").rename(out.with_suffix(".interrupted-" + attempt + ".jsonl"))
            log_path = run / "logs" / (name + "_server.log")
            with log_path.open("a", encoding="utf-8") as log:
                server = subprocess.Popen(
                    [sys.executable, str(ROOT / "scripts/experiment_server.py")],
                    cwd=ROOT,
                    env={**env, "RERANK_PHASE": name, "RERANK_ACTIVITY": "load_including_warmup"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                stop_monitor = threading.Event()

                def monitor():
                    process = psutil.Process(server.pid)
                    process.cpu_percent(None)
                    while not stop_monitor.wait(1):
                        try:
                            append_jsonl(
                                run / "load" / (name + "_resources.jsonl"),
                                {
                                    "at": now(),
                                    "pid": server.pid,
                                    "cpu_percent": process.cpu_percent(None),
                                    "rss_bytes": process.memory_info().rss,
                                },
                            )
                        except psutil.Error:
                            return

                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                try:
                    for _ in range(120):
                        if server.poll() is not None:
                            raise RuntimeError("Experiment server exited during startup")
                        try:
                            if requests.get(base + "/health", timeout=2).status_code == 200:
                                break
                        except requests.RequestException:
                            pass
                        time.sleep(1)
                    else:
                        raise RuntimeError("Experiment server did not become ready")
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/load_test.py"),
                        "--base-url",
                        base,
                        "--endpoint",
                        endpoint,
                        "--concurrency",
                        str(concurrency),
                        "--total",
                        "60",
                        "--sample",
                        "30",
                        "--warmup",
                        "3",
                        "--dataset",
                        str(run / "datasets/regression_v1.json"),
                        "--out",
                        str(out),
                        "--reset-metrics",
                    ]
                    print(f"[load start] {name}", flush=True)
                    result = subprocess.run(command, cwd=ROOT, env=env)
                    if result.returncode == 2 or not out.exists():
                        raise RuntimeError(f"Load test did not produce a complete report: {name}")
                    metrics = requests.get(base + "/metrics", timeout=10).json()
                    write_json(run / "load" / (name + "_metrics.json"), metrics)
                    report = json.loads(out.read_text(encoding="utf-8"))
                    report["metrics_crosscheck"] = {
                        "http_completed": metrics.get("counters", {}).get(
                            "http.POST." + ("retrieval" if endpoint == "retrieval" else "api.ask") + ".count", 0
                        ),
                        "note": "Client timeouts can leave an in-flight server request; phase timers overlap.",
                    }
                    report["gpu"] = {
                        "inference_location": "external APIs",
                        "local_gpu_inference": False,
                        "remote_utilization": None,
                        "remote_memory_bytes": None,
                    }
                    report["server_pid"] = server.pid
                    report["server_log"] = str(log_path)
                    write_json(out, report)
                    if report.get("summary", {}).get("errors", 0):
                        raise RuntimeError(f"Load test contains failed requests: {name}")
                finally:
                    stop_monitor.set()
                    thread.join(timeout=3)
                    server.terminate()
                    try:
                        server.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
            text = log_path.read_text(encoding="utf-8", errors="replace")
            report = json.loads(out.read_text(encoding="utf-8"))
            report["server_events"] = {
                "hyde_skips": text.count("生成超时") + text.count("无剩余预算"),
                "hyde_all_missed": text.count("本次全部未命中"),
                "rerank_429": text.count("429 限流"),
                "rerank_fallbacks": text.count("回退原始排序"),
            }
            write_json(out, report)
            print(f"[load finished] {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
