#!/usr/bin/env python
"""切块质量体检：用真实落库的 chunk 回答「切得好不好」。

不依赖任何主观判断，只算可量化的指标：

1. **长度分布** —— p50/p90/p99/max，以及超过 ``CHILD_MAX_SIZE``(800) 的条数与占比。
   配置里 child 上限是 800，若大量 chunk 远大于它，说明子块切分没生效。
2. **句中断裂率** —— 末尾（去空白后）不是句末标点的 chunk 占比。按长度切分时
   实现是"按句子组装"，这个比例应当很低；若很高，说明是在字符中间硬切。
3. **结构起始率** —— 以章节编号（``1.2.3`` / ``第X章`` / ``附录``）开头的 chunk 占比，
   用于验证父层是否真的按文档结构切（结构感知）。
4. **overlap 生效性** —— 相邻 chunk（``next_id``）之间是否真存在配置的 80 字重叠。
5. **真值包含率** —— 可选：给定数据集，统计其 ground_truth_text 能否被**单个** chunk
   完整覆盖。这是与检索指标最直接相关的一条：真值被切断，就永远召不回来。

用法::

    .venv/bin/python scripts/chunk_quality.py
    .venv/bin/python scripts/chunk_quality.py --dataset eval/eval_dataset.json --limit 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

SENT_END = "。！？.!?…\"'）)》】"
HEADING = re.compile(r"^\s*(?:第[一二三四五六七八九十百]+[章节条款]|附录\s*[A-Z]|\d+(?:\.\d+)*[\s、．.]|[（(]\d+[)）])")
#: Contextual Retrieval 注入的语境前缀（``【上下文：…】``）。
#: 度量结构起始率之前必须先剥掉它，否则 100% 的 chunk 都以它开头、命中率恒为 0
#: （这是度量本身的坑，不是切块的问题）。
CTX_PREFIX = re.compile(r"^【上下文[:：][^】]*】\s*")


def strip_ctx(t: str) -> str:
    return CTX_PREFIX.sub("", t or "")


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


def pull_chunks(uri: str, coll: str, limit: int = 0) -> list[dict]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=uri)
    if coll not in client.list_collections():
        raise SystemExit(f"集合不存在: {coll}")
    out: list[dict] = []
    it = client.query_iterator(
        collection_name=coll,
        filter="",
        output_fields=["id", "text", "next_id", "parent_id", "is_last_in_parent", "total_chunks_in_doc"],
        batch_size=1000,
    )
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            out.extend(batch)
            if limit and len(out) >= limit:
                break
    finally:
        it.close()
    return out[: limit or None]


def pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=os.getenv("COLLECTION", "pcb_kb"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--child-max", type=int, default=int(os.getenv("CHILD_MAX_SIZE", "800")))
    ap.add_argument("--overlap", type=int, default=int(os.getenv("CHILD_OVERLAP", "80")))
    ap.add_argument("--dataset", default="", help="可选：统计真值是否被单个 chunk 完整覆盖")
    args = ap.parse_args()

    uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    rows = pull_chunks(uri, args.collection, args.limit)
    if not rows:
        print("集合为空")
        return 1

    # 度量正文长度 / 结尾 / 结构前先剥掉 Contextual Retrieval 注入的【上下文：…】前缀：
    # 前缀约 100 字且每块都有，混进长度统计会把"配置 800"的口径整体抬高
    # （此前 ① 的 34.3% 超限有相当部分来自这里，属度量口径问题）。
    texts = [strip_ctx(r.get("text", "") or "") for r in rows]
    lengths = sorted(len(t) for t in texts)
    by_id = {str(r["id"]): (r.get("text", "") or "") for r in rows}

    over = [n for n in lengths if n > args.child_max]
    print("=" * 74)
    print(f"切块质量体检  集合={args.collection}  样本={len(rows)} 条")
    print("=" * 74)

    print("\n① 长度分布（正文长度，已剥【上下文】前缀；配置 child 上限 = %d）" % args.child_max)
    print(f"   min={lengths[0]}  p50={pct(lengths,50)}  p90={pct(lengths,90)}  "
          f"p99={pct(lengths,99)}  max={lengths[-1]}  均值={statistics.mean(lengths):.0f}")
    print(f"   超过上限的条数: {len(over)} ({len(over)/len(rows):.1%})"
          + (f"  最大超出: {max(over)} 字" if over else ""))

    bad_end = [t for t in texts if norm(t) and norm(t)[-1] not in SENT_END]
    print(f"\n② 句中断裂率（末尾非句末标点）: {len(bad_end)/len(rows):.1%}  ({len(bad_end)}/{len(rows)})")
    for t in bad_end[:3]:
        print(f"     例: …{norm(t)[-40:]!r}")

    head_start = [t for t in texts if HEADING.match(strip_ctx(t))]
    print(f"\n③ 结构起始率（剥掉【上下文】前缀后，以章节编号/第X章/附录/列表编号开头）: "
          f"{len(head_start)/len(rows):.1%}  ({len(head_start)}/{len(rows)})")

    # ④ overlap：只在**同一父块内**的相邻 chunk 之间检查。
    # 父块之间的边界本来就不该有 overlap，混进来会把分母灌水。
    checked = hit = skipped_cross_parent = 0
    for r in rows[:2000]:
        nid = str(r.get("next_id") or "")
        cur, nxt = r.get("text") or "", by_id.get(nid, "")
        if not nid or not nxt or len(cur) < args.overlap:
            continue
        if r.get("is_last_in_parent"):
            skipped_cross_parent += 1
            continue
        checked += 1
        if cur[-args.overlap:] in nxt:
            hit += 1
    if checked or skipped_cross_parent:
        print(f"\n④ 同父块内相邻 chunk 的 overlap 命中率: "
              f"{hit/max(checked,1):.1%}  ({hit}/{checked})"
              f"   [跳过跨父块对 {skipped_cross_parent} 组]")

    # ⑤ 真值包含率
    if args.dataset:
        ds_path = Path(args.dataset)
        if ds_path.is_file():
            ds = json.loads(ds_path.read_text(encoding="utf-8"))
            corpus = [norm(t) for t in texts]
            ok = 0
            for item in ds:
                snip = norm("".join(item.get("ground_truth_text") or []))
                if snip and any(snip in c for c in corpus):
                    ok += 1
            print(f"\n⑤ 真值被单个 chunk 完整覆盖: {ok}/{len(ds)} = {ok/len(ds):.1%}")
            print("   （这是与检索最相关的一条：真值被切断，就永远召不回来）")

    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
