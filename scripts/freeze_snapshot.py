#!/usr/bin/env python
"""冻结语料快照 + 冻结回归集（TestRule.md §一「开测前的三个前提」）。

做两件事：

1. **语料快照**：记录 Milvus 集合行数、语料文件数与总字节、词法缓存行数与 md5。
   写进 ``eval/datasets/baseline_snapshot.json``。此后整个重测期间不再重新入库；
   需要验证增量入库等改动时用独立 collection（``COLLECTION=pcb_kb_test``）。

2. **回归集存活校验**：对 ``eval/eval_dataset.json`` 的每条 ground_truth_id 去
   Milvus 里确认是否还在，剔除已失效的，再按来源文档分层保留不少于 ``--min-keep``
   条，另存为 ``eval/datasets/regression_v1.json``，并把语料指纹写进
   ``regression_v1.meta.json``。此后不再改动，作为所有版本横向对比的唯一基准。

用法::

    .venv/bin/python scripts/freeze_snapshot.py                     # 仅快照
    .venv/bin/python scripts/freeze_snapshot.py --check-survival    # 快照 + 存活校验
    .venv/bin/python scripts/freeze_snapshot.py --check-survival --min-keep 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()

DATASET_DIR = ROOT / "eval" / "datasets"
OUT_SNAPSHOT = DATASET_DIR / "baseline_snapshot.json"
OUT_REGRESSION = DATASET_DIR / "regression_v1.json"
OUT_META = DATASET_DIR / "regression_v1.meta.json"


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_snapshot() -> dict:
    """采集语料侧指纹：文件数、总字节、词法缓存行数与 md5。"""
    data_dir = Path(os.getenv("DATA_DIR", "./data/clear_docs"))
    files = sorted(p for p in data_dir.glob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)

    lex_path = Path(os.getenv("LEXICAL_CACHE_PATH", "./data/lexical_corpus.jsonl"))
    lex_lines = lex_md5 = None
    if lex_path.is_file():
        lex_lines = sum(1 for line in lex_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        lex_md5 = _md5_of_file(lex_path)

    manifest_path = Path(os.getenv("INGEST_MANIFEST_PATH", "./data/ingest_manifest.json"))
    manifest_docs = None
    if manifest_path.is_file():
        try:
            manifest_docs = len(json.loads(manifest_path.read_text(encoding="utf-8")).get("docs", {}))
        except Exception:
            manifest_docs = None

    return {
        "data_dir": str(data_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "lexical_cache_path": str(lex_path),
        "lexical_cache_lines": lex_lines,
        "lexical_cache_md5": lex_md5,
        "manifest_docs": manifest_docs,
    }


def milvus_snapshot() -> dict:
    """采集向量库侧指纹：集合名、行数、embedding 维度。"""
    from pymilvus import MilvusClient

    uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    coll = os.getenv("COLLECTION", "pcb_kb")
    client = MilvusClient(uri=uri)

    if coll not in client.list_collections():
        return {"uri": uri, "collection": coll, "exists": False}

    client.flush(coll)
    stats = client.get_collection_stats(coll)
    dim = None
    try:
        for f in client.describe_collection(coll).get("fields", []):
            if f.get("name") == os.getenv("MILVUS_EMBEDDING_FIELD", "embedding"):
                dim = (f.get("params") or {}).get("dim")
    except Exception:
        pass
    return {
        "uri": uri,
        "collection": coll,
        "exists": True,
        "row_count": stats.get("row_count"),
        "embedding_dim": dim,
    }


def id_survival(ids: list[str]) -> set[str]:
    """返回在 Milvus 中确实存在的 id 集合。"""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    coll = os.getenv("COLLECTION", "pcb_kb")
    alive: set[str] = set()
    batch = 200
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        expr = "id in [" + ", ".join(f'"{x}"' for x in chunk) + "]"
        try:
            rows = client.query(coll, filter=expr, output_fields=["id"])
            alive.update(str(r["id"]) for r in rows)
        except Exception as exc:
            print(f"  ⚠️  存活校验查询失败（{len(chunk)} 条）: {exc}")
    return alive


def stratified_keep(rows: list[dict], min_keep: int) -> tuple[list[dict], dict]:
    """按来源文档分层保留，确保覆盖面不因剔除而坍缩到少数几篇文档。"""
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        src = (r.get("ground_truth_metadata") or {}).get("source_path", "?")
        by_src[src].append(r)

    # 先每篇文档保底 1 条，再按各文档原有条数从多到少补齐
    kept: list[dict] = []
    for src, items in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        kept.append(items[0])
    order = sorted(by_src.items(), key=lambda kv: -len(kv[1]))
    idx = {src: 1 for src, _ in order}
    while len(kept) < min_keep and any(idx[src] < len(items) for src, items in order):
        for src, items in order:
            if len(kept) >= min_keep:
                break
            if idx[src] < len(items):
                kept.append(items[idx[src]])
                idx[src] += 1

    kept_ids = {id(r) for r in kept}
    kept = [r for r in rows if id(r) in kept_ids]
    stats = {
        "source_docs_total": len(by_src),
        "source_docs_kept": len({(r.get("ground_truth_metadata") or {}).get("source_path") for r in kept}),
        "per_doc_kept": dict(Counter((r.get("ground_truth_metadata") or {}).get("source_path", "?") for r in kept)),
    }
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "eval_dataset.json"))
    ap.add_argument("--check-survival", action="store_true", help="做 ground_truth_ids 存活校验并生成回归集")
    ap.add_argument("--min-keep", type=int, default=80, help="回归集最少保留条数（默认 80）")
    args = ap.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("① 语料快照")
    corpus = corpus_snapshot()
    for k, v in corpus.items():
        print(f"   {k}: {v}")

    print("\n② 向量库快照")
    milvus = milvus_snapshot()
    for k, v in milvus.items():
        print(f"   {k}: {v}")

    snapshot = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "milvus": milvus,
    }
    OUT_SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n   已写入 {OUT_SNAPSHOT.relative_to(ROOT)}")

    if not args.check_survival:
        print("\n（未开启 --check-survival，跳过回归集生成）")
        return 0

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        print(f"\n✗ 数据集不存在: {ds_path}")
        return 1
    rows = json.loads(ds_path.read_text(encoding="utf-8"))
    ids = [i for r in rows for i in (r.get("ground_truth_ids") or [])]
    print("\n" + "=" * 70)
    print(f"③ 存活校验（{ds_path.name}: {len(rows)} 条 / {len(ids)} 个 id）")

    alive = id_survival(ids)
    total = len(ids)
    print(f"   存活: {len(alive)}/{total} = {len(alive)/max(total,1):.1%}")

    kept_rows = [
        r for r in rows
        if r.get("ground_truth_ids") and all(i in alive for i in r["ground_truth_ids"])
    ]
    print(f"   整条可用（全部 id 存活）: {len(kept_rows)}")

    if not kept_rows:
        print("\n✗ 没有任何一条的 ground_truth_id 还在库里 —— 该数据集已完全失效。")
        print("  最可能的原因：重入过库导致 chunk id 变化（切块参数/语料/embedding 模型变了），")
        print("  或入库时的绝对路径与生成数据集时不同（doc_node_id = md5(绝对路径)）。")
        print("  此时不能拿它当回归集，需要重新生成 ground truth。")
        return 2

    if len(kept_rows) >= args.min_keep:
        regression = kept_rows
        strat = {
            "source_docs_total": len({(r.get("ground_truth_metadata") or {}).get("source_path") for r in kept_rows}),
            "source_docs_kept": len({(r.get("ground_truth_metadata") or {}).get("source_path") for r in kept_rows}),
            "per_doc_kept": dict(Counter((r.get("ground_truth_metadata") or {}).get("source_path", "?") for r in kept_rows)),
        }
    else:
        print(f"   可用不足 {args.min_keep} 条 → 按来源文档分层保留（保底每篇 1 条）")
        regression, strat = stratified_keep(kept_rows, args.min_keep)

    for r in regression:
        r["_frozen"] = {"dataset": "regression_v1", "source": ds_path.name}

    OUT_REGRESSION.write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "frozen_at": snapshot["frozen_at"],
        "source_dataset": ds_path.name,
        "source_rows": len(rows),
        "survived_rows": len(kept_rows),
        "survival_rate": round(len(kept_rows) / max(len(rows), 1), 4),
        "regression_rows": len(regression),
        "corpus_fingerprint": corpus,
        "milvus_fingerprint": milvus,
        "stratification": strat,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n   回归集: {len(regression)} 条 → {OUT_REGRESSION.relative_to(ROOT)}")
    print(f"   元数据: {OUT_META.relative_to(ROOT)}")
    print(f"   覆盖来源文档: {strat['source_docs_kept']}/{strat['source_docs_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
