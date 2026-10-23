# -*- coding: utf-8 -*-
"""语料体检：重复、残缺、分类覆盖率。

用法
----
    python scripts/audit_corpus.py                 # 体检 data/pcb_docs_clean
    python scripts/audit_corpus.py --dir <路径>     # 指定其它语料目录
    python scripts/audit_corpus.py --json out.json # 同时输出机器可读结果

判定口径
--------
* 近重复：8-gram shingle + Jaccard ≥ 0.10。阈值取 0.10 而不是 0.8，是因为
  同一本书的两个提取版本重排后 shingle 重合度会很低（实测 0.10~0.30），
  而不同标准之间因共享「前言/规范性引用文件」套话，重合度也能到 0.22。
  所以这个指标只用来**提示人工确认**，不足以单独判定重复。
* 残缺：字符数 < 2000 且章节号覆盖不完整。字符数本身不可靠 —— 有些标准
  确实短，有些是表格提取失败（表格内容丢失后只剩标题，字符数骤降）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "pcb_docs_clean")


def shingles(text: str, n: int = 8) -> set:
    t = re.sub(r"\s+", "", text)
    return {hashlib.md5(t[i:i + n].encode()).digest()[:8] for i in range(0, max(0, len(t) - n), 3)}


def read(p: str) -> str:
    return open(p, encoding="utf-8", errors="ignore").read()


def audit(root: str) -> dict:
    files = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".txt"):
                p = os.path.join(r, f)
                files.append((os.path.relpath(p, root).replace(os.sep, "/"), read(p)))

    # 非 txt 会被 SimpleDirectoryReader 一并读入（无 required_exts），单独列出
    strays = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".txt"):
                strays.append(os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/"))

    big = [(n, t) for n, t in files if len(t) > 5000]
    sh = {n: shingles(t) for n, t in big}
    pairs = []
    names = [n for n, _ in big]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sh[names[i]], sh[names[j]]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= 0.10:
                pairs.append({"jaccard": round(jac, 3), "a": names[i], "b": names[j]})
    pairs.sort(key=lambda x: -x["jaccard"])

    # 内容级重复：20-gram Jaccard ≥ 0.5。与上面的「近重复提示」分工不同 ——
    # 上面的 8-gram 阈值低，噪声大（不同标准共享前言也能到 0.22）；
    # 这一档基本可以断定「同一份内容存了两次」，本仓库曾据此发现一份
    # 挂错标准号的副本（文件名 GB/T 45713.4，正文实为 GB/T 13557）。
    grams = {}
    for n, t in files:
        tt = re.sub(r"\s+", "", t)
        grams[n] = {tt[i:i + 20] for i in range(0, max(0, len(tt) - 20))}
    content_dups = []
    gn = list(grams)
    for i in range(len(gn)):
        for j in range(i + 1, len(gn)):
            a, b = grams[gn[i]], grams[gn[j]]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= 0.5:
                content_dups.append({"jaccard": round(jac, 3), "a": gn[i], "b": gn[j]})
    content_dups.sort(key=lambda x: -x["jaccard"])

    # 残缺：字符数 + 章号覆盖
    short = []
    for n, t in files:
        if len(t) >= 2000:
            continue
        heads = re.findall(r"^\s*(\d{1,2})(?:\.\d{1,2}){0,2}\s+\S", t, re.M)
        chs = sorted({int(h) for h in heads})
        short.append({
            "file": n,
            "chars": len(t),
            "chapters_found": chs,
            "gap": bool(chs) and chs != list(range(1, max(chs) + 1)),
        })

    # 分类
    sys.path.insert(0, os.path.join(os.path.dirname(root), "..", "src"))
    try:
        from pcb_rag.ingest import _infer_source_type  # type: ignore
        dist = Counter(_infer_source_type("/" + n.replace(os.sep, "/") + "/") for n, _ in files)
    except Exception as e:  # 依赖不全时降级，不让体检失败
        dist = Counter({f"(不可用: {e})": len(files)})

    return {
        "root": root,
        "n_files": len(files),
        "total_chars": sum(len(t) for _, t in files),
        "stray_non_txt": strays,
        "near_duplicates": pairs,
        "content_duplicates": content_dups,
        "short_files": sorted(short, key=lambda x: x["chars"]),
        "source_type": dict(dist),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--json")
    args = ap.parse_args()

    res = audit(args.dir)
    print(f"目录: {res['root']}")
    print(f"文件数: {res['n_files']}   总字符: {res['total_chars']:,}")
    print(f"非 txt 文件（会干扰入库）: {res['stray_non_txt'] or '无'}")
    print(f"来源分类: {res['source_type']}")

    print(f"\n内容级重复 (20-gram Jaccard ≥ 0.5，基本可断定同一份内容存了两次): "
          f"{len(res['content_duplicates'])} 对")
    for p in res["content_duplicates"]:
        print(f"  {p['jaccard']:.2f}  {os.path.basename(p['a'])}")
        print(f"        {os.path.basename(p['b'])}")

    print(f"\n近重复提示 (8-gram Jaccard ≥ 0.10，仅提示，需人工确认): {len(res['near_duplicates'])} 对")
    for p in res["near_duplicates"][:10]:
        print(f"  {p['jaccard']:.3f}  {os.path.basename(p['a'])}  <->  {os.path.basename(p['b'])}")

    print(f"\n不足 2000 字符的文件: {len(res['short_files'])} 个")
    for s in res["short_files"]:
        print(f"  {s['chars']:6d}  章号{s['chapters_found']}  缺章={s['gap']}  {s['file']}")

    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
