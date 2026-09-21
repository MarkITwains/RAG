#!/usr/bin/env python
"""第 1 组第二条路径：生产路径基线（TestRule.md §二.1）。

与 evaluate_recall.py（评测脚本路径）的差异：直接在 .env.eval 配置下 import
``dify_external_api``，调用生产检索函数 :func:`_retrieve_nodes`，取 node id 序列算
指标 —— 回答「评测脚本和线上是不是同一个系统」（两条路径 MRR 差 > 0.03 即为要查的 bug）。

用法::

    set -a; source .env.eval; set +a
    CHUNK_EXPAND_ENABLED=0 .venv/bin/python scripts/eval_production_path.py \
        --top-k 20 --out eval/reports/r1_baseline_production.json

要点：
- ``CHUNK_EXPAND_ENABLED=0``：按 TestRule 取纯排名（唯一偏离生产的开关，由运行脚本设置）；
- 指标函数复用 ``eval/evaluate_recall.py`` 的 ``_ndcg_at_k`` / ``_average_precision``；
- 分段耗时从 dify logger 的 ``[Timing]`` 行抓取；``[QueryRoute]`` 抓查询类型；
- ``--warmup N`` 先跑 N 题丢弃（HyDE 30s 预算 + LLM 冷启动，见 TestRule §五）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "eval"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

TIMING_RE = re.compile(r"\[Timing\]\s*([^:：]+?)[:：]\s*([\d.]+)\s*s")
ROUTE_RE = re.compile(r"\[QueryRoute\]\s+(\w+)")
ENV_KEYS = (
    "RECALL_TOP_K", "RERANK_TOP_N", "FUSION_NUM_QUERIES", "FUSION_RRF_K",
    "HYDE_ENABLED", "CHUNK_EXPAND_ENABLED", "MULTI_EXPAND_ENABLED",
    "QUERY_ROUTING_ENABLED", "QUERY_DECOMPOSE_ENABLED", "QUERY_STEP_BACK_ENABLED",
    "QUERY_UNDERSTANDING_MODE",
)
KS = (1, 3, 5, 10)


class _Collector(logging.Handler):
    """抓取 dify logger 里的 [Timing] 分段秒数与 [QueryRoute] 查询类型。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.timing: dict[str, float] = {}
        self.route: str | None = None

    def reset(self) -> None:
        self.timing = {}
        self.route = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for label, sec in TIMING_RE.findall(msg):
            key = label.strip()
            self.timing[key] = self.timing.get(key, 0.0) + float(sec)
        m = ROUTE_RE.search(msg)
        if m:
            self.route = m.group(1)


def _node_ids(nodes) -> list[str]:
    out = []
    for n in nodes:
        nid = getattr(n, "id_", None)
        if not nid:
            nid = getattr(getattr(n, "node", None), "node_id", None)
        out.append(str(nid or ""))
    return out


def _pctl(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(ROOT / "eval/datasets/regression_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "eval/reports/r1_baseline_production.json"))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0, help="先跑 N 题热身并丢弃")
    args = ap.parse_args()

    # 注意顺序：先 import api（按当前 env 定型），再 import evaluate_recall
    # （后者模块级会强改部分 env，见 TestRule §五 的坑）。
    import pcb_rag.dify_external_api as api
    from evaluate_recall import _average_precision, _ndcg_at_k

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("数据集为空")
        return 1

    print(f"[init] _initialize_retrieval_engine() ...", flush=True)
    api._initialize_retrieval_engine()

    collector = _Collector()
    api.logger.addHandler(collector)

    results: list[dict] = []

    def run_one(row: dict, tag: str = "") -> None:
        collector.reset()
        t0 = time.time()
        nodes = api._retrieve_nodes(row["query"], top_k=args.top_k)
        wall = time.time() - t0
        got = _node_ids(nodes)
        gt = set(map(str, row.get("ground_truth_ids") or []))
        rank = next((i + 1 for i, g in enumerate(got) if g in gt), 0)
        rec = {
            "query": row["query"],
            "got_ids": got,
            "gt_ids": sorted(gt),
            "first_hit_rank": rank,
            "mrr": 1.0 / rank if rank else 0.0,
            "hits": {f"hit@{k}": 1.0 if any(g in gt for g in got[:k]) else 0.0 for k in KS},
            "ndcg@10": _ndcg_at_k(got, gt, 10),
            "ap": _average_precision(got, gt),
            "wall_s": round(wall, 3),
            "timing": dict(collector.timing),
            "route": collector.route,
        }
        if not tag:
            results.append(rec)
            print(f"[{len(results)}/{len(rows)}] rank={rank} wall={wall:.1f}s "
                  f"route={rec['route']} {row['query'][:36]}", flush=True)
        else:
            print(f"[warmup] rank={rank} wall={wall:.1f}s got={len(got)} {row['query'][:36]}", flush=True)
        return rec

    warm_recs = [run_one(row, tag="warmup") for row in rows[: args.warmup]]
    if warm_recs and all(not r["got_ids"] for r in warm_recs):
        # 2026-09-20 事故教训：路由全挂（NameError 被兜底吞掉）时 _retrieve_nodes
        # 返回空列表，脚本会"成功"产出全零废报告。热身必须在此时拒绝继续。
        print("[FATAL] 热身查询的检索结果全部为空 —— 生产链路故障（路由全挂/配置错误），"
              "拒绝继续以避免废报告", flush=True)
        return 2

    t_start = time.time()
    for row in rows:
        run_one(row)
    print(f"[done] {len(results)} 题耗时 {time.time() - t_start:.0f}s", flush=True)

    n = max(len(results), 1)
    timing_keys: set[str] = set()
    for r in results:
        timing_keys.update(r["timing"])
    summary = {
        "dataset": str(Path(args.dataset).name),
        "n": len(results),
        "top_k": args.top_k,
        "mrr": round(statistics.mean(r["mrr"] for r in results), 4),
        "map": round(statistics.mean(r["ap"] for r in results), 4),
        **{f"hit@{k}": round(statistics.mean(r["hits"][f"hit@{k}"] for r in results), 4) for k in KS},
        "ndcg@10": round(statistics.mean(r["ndcg@10"] for r in results), 4),
        "wall_s_p50": _pctl([r["wall_s"] for r in results], 50),
        "wall_s_p95": _pctl([r["wall_s"] for r in results], 95),
        "timing_mean": {
            k: round(statistics.mean(r["timing"][k] for r in results if k in r["timing"]), 3)
            for k in sorted(timing_keys)
        },
        "route_counter": dict(Counter(r["route"] for r in results)),
        "env": {k: os.getenv(k) for k in ENV_KEYS},
    }
    empty_n = sum(1 for r in results if not r["got_ids"])
    summary["empty_results"] = empty_n
    if results and empty_n == len(results):
        print("[FATAL] 正式跑 80 题结果全为空 —— 数据不可用（已写出报告供排查，退出码 2）", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n== summary ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if (results and empty_n == len(results)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
