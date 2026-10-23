#!/usr/bin/env python
"""Print experiment progress without loading credentials or making API calls."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Defaults to the newest non-superseded run")
    args = parser.parse_args()
    if args.run_dir:
        run = Path(args.run_dir)
    else:
        reports = Path(__file__).resolve().parents[1] / "eval/reports"
        candidates = []
        for path in reports.glob("*/manifest.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") != "superseded":
                candidates.append((data.get("formal_started_at", data.get("started_at", "")), path.parent))
        if not candidates:
            parser.error("No experiment run found; supply --run-dir")
        run = max(candidates, key=lambda item: item[0])[1]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    heartbeat = json.loads((run / "heartbeat.json").read_text(encoding="utf-8"))
    print("Run:", manifest["run_id"])
    print("State:", manifest["status"], "heartbeat:", heartbeat["updated_at"])
    active = heartbeat.get("active_steps", [])
    print(
        "Active:", ", ".join(item["name"] for item in active) or heartbeat.get("active_step", heartbeat.get("status"))
    )
    print("Parallel capacity:", manifest.get("max_parallel_steps", 1))
    print(
        "Rerank:",
        manifest.get("env", {}).get("RERANK_API_MODEL"),
        "candidate cap:",
        manifest.get("env", {}).get("RERANK_MAX_CANDIDATES", "legacy"),
        "output top_n:",
        manifest.get("env", {}).get("RERANK_TOP_N"),
        "proactive interval:",
        manifest.get("env", {}).get("RERANK_MIN_INTERVAL", "legacy"),
    )
    usage_path = run / "rerank_usage.json"
    if usage_path.exists():
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
        print(
            "Rerank usage:",
            usage["reported_tokens"],
            "reported tokens;",
            usage["accounted_tokens"],
            "/",
            usage["token_limit"] if usage["token_limit"] else "unlimited",
            "budget-accounted tokens;",
            usage["reserved_tokens"],
            "reserved;",
            usage["attempts"],
            "attempts; circuit:",
            usage["circuit_reason"],
        )
    print("Completed:", sum(job["status"] == "done" for job in manifest["steps"]), "/", len(manifest["steps"]))
    for job in manifest["steps"]:
        path = run / "retrieval" / (job["name"] + ".progress.json")
        detail = ""
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
            detail = f" {progress['completed']}/{progress['total']} queries"
        print(f"{job['status']:12} {job['name']}{detail}")


if __name__ == "__main__":
    main()
