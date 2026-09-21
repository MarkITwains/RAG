#!/usr/bin/env python
"""G5 软匹配诊断：检索只跑一次 + 多阈值离线扫描（TestRule §二.5）。

动机：evaluate_recall.py 的报告只存汇总指标、不存逐题检索结果，
每换一个 --embed-threshold 就要全量重跑一遍检索（80 题 × HyDE + 精排限速
≈ 1 小时）。本脚本把流程拆成两段：

  1. 检索段：与 r1 Path A 完全同配置（fusion_hyde_rerank / recall 200 /
     rerank top-n 10 / rrf-k 40），逐题保存 top-10 的 id / score / 原文；
  2. 扫描段：纯离线，对 0.70 / 0.78 / 0.85 / 0.90 四个阈值分别计算
     软 MRR / Hit@1/5/10（embedding 带缓存，只调一次 API），
     同时输出"软命中但硬未命中"样本清单，供人工做 A/B/C/D 分类。

输出：
  eval/reports/g5_retrieval_raw.json    检索原始结果（可复用，勿删）
  eval/reports/g5_softmatch_scan.json   四阈值指标 + 硬指标对照
  eval/reports/g5_softdiff_samples.json 软硬差异样本（人工分类输入）

用法：在 .env.eval 环境下运行
  set -a; source .env.eval; set +a
  CHUNK_EXPAND_ENABLED=0 python scripts/softmatch_scan.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

from llama_index.core.schema import MetadataMode, QueryBundle  # noqa: E402

import pcb_rag.query as rag_query  # noqa: E402
from eval import evaluate_recall as er  # noqa: E402

# ── 配置（与 r1_baseline_evalscript 同源可比）───────────────────────────────
DATASET = Path(os.getenv("G5_DATASET", str(ROOT / "eval/datasets/regression_v1.json")))
RAW_OUT = Path(os.getenv("G5_RAW_OUT", str(ROOT / "eval/reports/g5_retrieval_raw.json")))
SCAN_OUT = Path(os.getenv("G5_SCAN_OUT", str(ROOT / "eval/reports/g5_softmatch_scan.json")))
DIFF_OUT = Path(os.getenv("G5_DIFF_OUT", str(ROOT / "eval/reports/g5_softdiff_samples.json")))
RECALL_K = int(os.getenv("G5_RECALL_K", "200"))
TOP_N = int(os.getenv("G5_TOP_N", "10"))
THRESHOLDS = [float(x) for x in os.getenv("G5_THRESHOLDS", "0.70,0.78,0.85,0.90").split(",")]
TOP_KS = [1, 5, 10]


def _node_text(nws) -> str:
    try:
        return nws.node.get_content(metadata_mode=MetadataMode.NONE).strip()
    except Exception:
        return ""


def _cos(a, b) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ── 阶段 1：检索（结果落盘，可断点续跑）────────────────────────────────────
def run_retrieval() -> list[dict]:
    if RAW_OUT.is_file():
        print(f"[G5] 发现已有检索结果 {RAW_OUT}，跳过检索段（删除该文件可强制重跑）")
        return json.loads(RAW_OUT.read_text(encoding="utf-8"))

    print(f"[G5] 数据集: {DATASET}")
    dataset = er._load_dataset(DATASET)
    print(f"[G5] 有效题数: {len(dataset)}")

    index = rag_query.build_index("eval")

    # 与 evaluate_recall.py --mode fusion_hyde_rerank 的强制开关保持一致
    rag_query.HYDE_ENABLED = True
    rag_query.SHORT_QUERY_THRESHOLD = 100
    print("[G5] 强制启用 HyDE，短查询阈值=100（与 Path A 口径一致）")

    llm_instance = getattr(rag_query.Settings, "llm", None)
    retriever = er._build_retriever(
        index,
        recall_k=RECALL_K,
        mode="fusion_hyde_rerank",
        use_colbert=False,
        use_hyde=True,
        use_query2doc=False,
        colbert_candidates=50,
        rrf_k=40,
        llm=llm_instance,
    )

    rag_query.RERANK_ENABLED = True
    rag_query.RERANK_TOP_N = TOP_N
    reranker = rag_query._try_build_reranker()
    print(f"[G5] Reranker: {'OK' if reranker else 'None（硬指标将不可比）'}")

    rows: list[dict] = []
    started = time.time()
    for i, item in enumerate(dataset, 1):
        q = str(item["query"]).strip()
        gt_ids = [str(x) for x in (item.get("ground_truth_ids") or []) if x]
        gt_texts = [t for t in (item.get("ground_truth_text") or []) if isinstance(t, str) and t.strip()]
        if not gt_ids:
            continue
        try:
            nodes = retriever.retrieve(q)
            nodes = er._maybe_rerank(reranker, nodes, q)
            got = [
                {"id": er._node_id(n) or "", "score": float(getattr(n, "score", 0.0) or 0.0),
                 "text": _node_text(n)[:2000]}
                for n in nodes[:TOP_N]
            ]
        except Exception as exc:
            print(f"  [{i}/{len(dataset)}] 检索失败: {exc}")
            got = []
        rows.append({"query": q, "gt_ids": gt_ids, "gt_texts": gt_texts, "got": got})
        hit = any(g["id"] in set(gt_ids) for g in got)
        print(f"  [{i}/{len(dataset)}] {q[:36]}... top1={'✓' if got and got[0]['id'] in set(gt_ids) else '✗'} "
              f"硬命中={'有' if hit else '无'} ({time.time()-started:.0f}s)")
        if i % 10 == 0:
            RAW_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    RAW_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[G5] 检索段完成（{time.time()-started:.0f}s），原始结果 → {RAW_OUT}")
    return rows


# ── 阶段 2：离线阈值扫描 ───────────────────────────────────────────────────
def scan_thresholds(rows: list[dict]) -> dict:
    embed_model = getattr(rag_query.Settings, "embed_model", None)
    if embed_model is None:
        from pcb_rag.api_clients import build_embed_model

        embed_model = build_embed_model()
    print(f"[G5] Embedding 模型: {getattr(embed_model, 'model_name', 'unknown')}")

    # 预计算：每题每个位置的 最大cosine(gt_texts, got_text)
    per_q: list[dict] = []
    for r in rows:
        sims = []
        for g in r["got"]:
            best = 0.0
            for gt in r["gt_texts"]:
                try:
                    best = max(best, _cos(er._embed_cached(embed_model, gt[:2000]),
                                          er._embed_cached(embed_model, g["text"][:2000])))
                except Exception:
                    pass
            sims.append(round(best, 4))
        got_ids = [g["id"] for g in r["got"]]
        hard_rank = next((i + 1 for i, gid in enumerate(got_ids) if gid in set(r["gt_ids"])), 0)
        per_q.append({"query": r["query"], "gt_ids": r["gt_ids"], "gt_texts": r["gt_texts"],
                      "got_ids": got_ids, "got_texts": [g["text"] for g in r["got"]],
                      "sims": sims, "hard_rank": hard_rank})
    print(f"[G5] 相似度预计算完成（embedding 缓存 {len(er._EMBED_CACHE)} 条）")

    def metrics(ranks: list[int]) -> dict:
        n = len(per_q)
        out = {"n": n}
        out["mrr"] = round(sum(1.0 / r for r in ranks if r > 0) / n, 4)
        for k in TOP_KS:
            out[f"hit@{k}"] = round(sum(1 for r in ranks if 0 < r <= k) / n, 4)
        return out

    report: dict = {
        "config": {"dataset": str(DATASET), "recall_k": RECALL_K, "top_n": TOP_N,
                   "thresholds": THRESHOLDS,
                   "embed_model": str(getattr(embed_model, "model_name", "unknown"))},
        "hard": metrics([p["hard_rank"] for p in per_q]),
        "soft": {},
    }
    for thr in THRESHOLDS:
        ranks = []
        for p in per_q:
            ranks.append(next((i + 1 for i, s in enumerate(p["sims"]) if s >= thr), 0))
        report["soft"][str(thr)] = metrics(ranks)
        print(f"[G5] thr={thr:.2f}: {report['soft'][str(thr)]}")

    # 软硬差异样本（按基准阈值 0.78 导出，供人工 A/B/C/D 分类）
    base = str(os.getenv("G5_BASE_THRESHOLD", "0.78"))
    diffs = []
    for p in per_q:
        soft_rank = next((i + 1 for i, s in enumerate(p["sims"]) if s >= float(base)), 0)
        if soft_rank and not p["hard_rank"]:
            diffs.append({
                "query": p["query"],
                "soft_rank": soft_rank,
                "cosine": p["sims"][soft_rank - 1],
                "got_id": p["got_ids"][soft_rank - 1],
                "got_text": p["got_texts"][soft_rank - 1],
                "gt_text": "\n---\n".join(p["gt_texts"])[:1500],
                "category": "",  # 人工填写：A=相邻chunk / B=跨文档重复 / C=语义等价 / D=误判
            })
    report["softdiff_count"] = len(diffs)
    report["note"] = "soft 列为对应阈值下的软匹配指标；hard 为 id 精确匹配。差异样本见 g5_softdiff_samples.json"

    SCAN_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    DIFF_OUT.write_text(json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[G5] 指标 → {SCAN_OUT}")
    print(f"[G5] 软硬差异样本 {len(diffs)} 条 → {DIFF_OUT}（请人工填写 category 字段）")
    return report


def main() -> int:
    rows = run_retrieval()
    scan_thresholds(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
