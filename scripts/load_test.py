#!/usr/bin/env python
"""第 3 组并发压测（TestRule.md §二.3）。

对指定端点做固定并发压测，输出 p50/p95/p99、吞吐、错误率，写 JSON 报告。

用法（通常由 scripts/run_g3_loadtest.sh 编排）::

    .venv/bin/python scripts/load_test.py --endpoint retrieval --concurrency 8 \
        --total 60 --token "$DIFY_API_TOKEN" --out eval/reports/g3_retrieval_c8.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def _pctl(vals, p):
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", choices=["retrieval", "ask"], required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--total", type=int, default=60)
    ap.add_argument("--dataset", default=str(ROOT / "eval/datasets/regression_v1.json"))
    ap.add_argument("--sample", type=int, default=30, help="从冻结集抽 N 题（固定种子）")
    ap.add_argument("--token", default="")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    random.Random(42).shuffle(rows)
    queries = [r["query"] for r in rows[: args.sample]]

    headers = {"Authorization": f"Bearer {args.token}", "Content-Type": "application/json"}
    if args.endpoint == "retrieval":
        url = f"{args.base_url}/retrieval"
        payloads = [{
            "knowledge_id": "pcb_kb",
            "query": q,
            "retrieval_setting": {"top_k": args.top_k, "score_threshold": 0.0,
                                  "score_threshold_enabled": False},
        } for q in queries]
    else:
        url = f"{args.base_url}/api/ask"
        payloads = [{"query": q} for q in queries]

    # 预检：第一发 422 之类直接把响应体打出来，避免整场白跑
    pre = requests.post(url, json=payloads[0], headers=headers, timeout=300)
    if pre.status_code != 200:
        print(f"[FATAL] 预检失败 HTTP {pre.status_code}: {pre.text[:300]}")
        return 2

    results: list[tuple[float, object]] = []
    t_start = time.time()

    def one(i: int) -> None:
        p = payloads[i % len(payloads)]
        t0 = time.time()
        try:
            r = requests.post(url, json=p, headers=headers, timeout=300)
            results.append((time.time() - t0, r.status_code))
        except Exception as e:  # noqa: BLE001 —— 压测必须把错误当数据点
            results.append((time.time() - t0, f"ERR:{type(e).__name__}"))

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, range(args.total)))
    wall = time.time() - t_start

    dts = [d for d, _ in results]
    codes = {}
    for _, c in results:
        codes[str(c)] = codes.get(str(c), 0) + 1
    ok = sum(1 for _, c in results if c == 200)
    summary = {
        "endpoint": args.endpoint,
        "concurrency": args.concurrency,
        "total": args.total,
        "sample_queries": len(queries),
        "ok": ok,
        "errors": args.total - ok,
        "error_rate": round((args.total - ok) / args.total, 4),
        "throughput_rps": round(args.total / wall, 3),
        "wall_s": round(wall, 2),
        "latency_p50": _pctl(dts, 50),
        "latency_p95": _pctl(dts, 95),
        "latency_p99": _pctl(dts, 99),
        "status_codes": codes,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "latencies": dts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if ok == args.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
