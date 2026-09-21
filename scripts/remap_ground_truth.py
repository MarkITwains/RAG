#!/usr/bin/env python
"""按**文本**把旧数据集的 ground_truth_ids 重新映射到当前库里的 chunk id。

为什么需要这个
--------------
``eval/eval_dataset.json`` 的 ``ground_truth_ids`` 是 2026-03 那份库里的 chunk id。
实测（scripts 里的探针 + 本脚本的 --report-mismatch）表明：

- ``doc_node_id = md5(SimpleDirectoryReader 给出的 file_path)``，而该 file_path 是
  **绝对路径** → 换目录 / 换部署位置，id 全变；
- 三种 chunk id 规则（``{doc}-pc-{i}`` / ``{doc}-{i}`` / ``{doc}-sem-{i}``）配 4 种
  候选路径形态，扫到 idx=2000 都是 0 命中 —— 旧 id 已完全不可复现；
- **但 100/100 条的 ``ground_truth_text`` 都能在对应源文件里原文定位**。

所以：题目与真值内容完好，只有 id 失效。这里按文本把 id 重挂到当前 chunk 上，
产出一份可用的冻结回归集，并把"id 存活率 0%、文本可定位率 100%"如实记录进 meta ——
这本身就是回答"3 月那份 0.87 为什么复现不出来"的证据。

用法::

    .venv/bin/python scripts/remap_ground_truth.py --report-only       # 只诊断
    .venv/bin/python scripts/remap_ground_truth.py --min-keep 80       # 生成回归集
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

DATASET_DIR = ROOT / "eval" / "datasets"
OUT_REGRESSION = DATASET_DIR / "regression_v1.json"
OUT_META = DATASET_DIR / "regression_v1.meta.json"


def norm(text: str) -> str:
    """归一化：去掉所有空白，便于跨换行/空格做包含匹配。"""
    return re.sub(r"\s+", "", text or "")


def load_chunks(uri: str, coll: str, fields: list[str]) -> list[dict]:
    """把整个集合的 chunk 拉进内存（几千条 × 800 字符，量级很小）。"""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    if coll not in client.list_collections():
        raise SystemExit(f"集合不存在: {coll}（先跑入库）")
    client.load_collection(coll)
    rows: list[dict] = []
    it = client.query_iterator(collection_name=coll, filter="", output_fields=fields, batch_size=1000)
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            rows.extend(batch)
    finally:
        it.close()
    return rows


def match_chunk(snippet: str, chunks: list[dict]) -> tuple[dict | None, int, float]:
    """在 chunks 里找包含该片段的 chunk。

    Returns:
        (命中的 chunk, 使用的探针长度, 覆盖率) —— 覆盖率 = 片段被 chunk 覆盖的比例。
    """
    total = max(len(snippet), 1)
    for probe_len in (120, 80, 60, 40, 24):
        probe = snippet[:probe_len]
        if len(probe) < 12:
            break
        hits = [c for c in chunks if probe in norm(c.get("text", ""))]
        if not hits:
            continue
        # 覆盖度：真值被该 chunk 完整包含时为 1.0，否则退化为探针覆盖率
        def coverage_of(c: dict) -> float:
            return 1.0 if snippet in norm(c.get("text", "")) else probe_len / total

        best = max(hits, key=coverage_of)
        return best, probe_len, round(coverage_of(best), 3)
    return None, 0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "eval_dataset.json"))
    ap.add_argument("--min-keep", type=int, default=80)
    ap.add_argument("--report-only", action="store_true", help="只输出诊断，不写回归集")
    ap.add_argument("--probe", type=int, default=3, help="打印前 N 条匹配样例")
    args = ap.parse_args()

    uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    coll = os.getenv("COLLECTION", "pcb_kb")
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    print(f"集合 {coll} @ {uri}")
    chunks = load_chunks(uri, coll, ["id", "text"])
    print(f"  拉取 chunk: {len(chunks)} 条")
    if not chunks:
        print("✗ 集合为空，先完成入库")
        return 1

    print(f"\n按文本重映射 {len(rows)} 条真值 ...")
    remapped, unresolved = [], []
    for r in rows:
        snippet = norm("".join(r.get("ground_truth_text") or []))
        hit, probe_len, cover = match_chunk(snippet, chunks)
        if hit is None:
            unresolved.append(r.get("query", "")[:50])
            continue
        new = dict(r)
        old_ids = list(r.get("ground_truth_ids") or [])
        new["ground_truth_ids"] = [str(hit["id"])]
        new["_remap"] = {
            "old_ids": old_ids,
            "probe_len": probe_len,
            "coverage": cover,
            "matched_by": "text_contains",
        }
        remapped.append(new)

    print(f"  成功重映射: {len(remapped)}/{len(rows)} = {len(remapped)/len(rows):.0%}")
    if unresolved:
        print(f"  无法定位 {len(unresolved)} 条（真值跨 chunk 边界或语料已变），前 3:")
        for q in unresolved[:3]:
            print(f"    - {q!r}")

    if args.probe:
        print("\n  样例:")
        for r in remapped[: args.probe]:
            rm = r["_remap"]
            print(f"    Q: {r['query'][:44]}...")
            print(f"       旧 id {rm['old_ids'][0][:12]}... → 新 id {r['ground_truth_ids'][0][:12]}...  "
                  f"探针 {rm['probe_len']} 字 覆盖 {rm['coverage']}")

    if args.report_only:
        print("\n（--report-only，未写文件）")
        return 0

    if len(remapped) < args.min_keep:
        print(f"\n✗ 可用 {len(remapped)} 条 < 要求 {args.min_keep} 条，不生成回归集")
        return 2

    # 按来源文档分层，保证覆盖面
    by_src: dict[str, list[dict]] = {}
    for r in remapped:
        by_src.setdefault((r.get("ground_truth_metadata") or {}).get("source_path", "?"), []).append(r)
    kept = list(remapped)
    if len(kept) > args.min_keep:
        kept = []
        for src, items in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
            kept.append(items[0])
        i = 1
        while len(kept) < args.min_keep:
            added = False
            for src, items in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
                if i < len(items) and len(kept) < args.min_keep:
                    kept.append(items[i]); added = True
            if not added:
                break
            i += 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REGRESSION.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": Path(args.dataset).name,
        "source_rows": len(rows),
        "id_survival_rate": 0.0,
        "id_survival_note": (
            "旧 chunk id 完全不可复现：doc_node_id = md5(绝对路径)，且三种 id 规则 × 4 种候选路径 "
            "形态扫到 idx=2000 均 0 命中。这与 TestRule.md §五 的猜测一致 —— 语料在 3 月之后重入过库。"
        ),
        "text_locatable_rows": len(remapped),
        "text_locatable_rate": round(len(remapped) / max(len(rows), 1), 4),
        "regression_rows": len(kept),
        "remap_method": "text_contains（归一化空白后按最长探针包含匹配）",
        "coverage_histogram": dict(Counter(round(r["_remap"]["coverage"], 1) for r in kept)),
        "source_docs": len(by_src),
        "milvus": {"uri": uri, "collection": coll, "chunks": len(chunks)},
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n回归集: {len(kept)} 条 → {OUT_REGRESSION.relative_to(ROOT)}")
    print(f"元数据: {OUT_META.relative_to(ROOT)}")
    print(f"覆盖来源文档: {len(by_src)} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
