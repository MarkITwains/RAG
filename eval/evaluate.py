"""PCB-RAG 评测脚本。

对黄金数据集逐题执行「检索 → 生成 → 打分」，输出四指标与明细报告。

用法::

    # 全量评测
    python eval/evaluate.py --dataset eval/datasets/golden.jsonl

    # 只跑前 20 题，指定 recall_k 与保留数
    python eval/evaluate.py --dataset eval/datasets/golden.jsonl --limit 20 --recall-k 100 --top-k 5

    # 只评检索（不调生成，速度快、成本低）
    python eval/evaluate.py --dataset eval/datasets/golden.jsonl --retrieval-only

    # 关闭精排，对比重排带来的收益
    python eval/evaluate.py --dataset eval/datasets/golden.jsonl --no-rerank

数据集格式（JSONL）::

    {"id": "q-0001", "question": "...", "ground_truth": "...", "source_path": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 允许直接运行脚本时导入 pcb_rag 与 eval 包
ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT / "src"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()  # 直接 `python eval/evaluate.py` 时也能读到 .env 里的 API 配置

from pcb_rag.api_clients import build_llm  # noqa: E402
from eval.metrics import (  # noqa: E402
    ALL_METRICS,
    JUDGE_FAILURE_RATE_THRESHOLD,
    evaluate_case,
    format_summary,
    summarize,
)


def _load_dataset(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {path}\n"
            f"可用 `python eval/build_golden_dataset.py` 生成，或按 README 手工准备。"
        )

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("question") and obj.get("ground_truth"):
                rows.append(obj)

    if limit and limit > 0:
        rows = rows[:limit]
    return rows


class EvalEngine:
    """Quality evaluation uses the production pipeline with explicit overrides."""
    def __init__(self, recall_k: int, top_k: int, use_rerank: bool = True, *, pipeline=None):
        from pcb_rag.retrieval import PipelineConfig, RetrievalPipeline
        self.recall_k, self.top_k = recall_k, top_k
        self.pipeline = pipeline or RetrievalPipeline(PipelineConfig.from_env(recall_k=recall_k, rerank_enabled=use_rerank))
        self.pipeline.initialize()
        self.llm = self.pipeline.llm
        self.index = self.pipeline.state.get("index")
        self.rerank = self.pipeline.state.get("rerank")
        self.bm25 = self.pipeline.state.get("bm25_index")
        self.last_run = None

    def retrieve(self, query: str) -> list:
        from pcb_rag.query import build_context_bundle
        self.last_run = self.pipeline.retrieve(query, top_k=self.top_k)
        self.last_bundle = build_context_bundle(self.last_run.nodes)
        return self.last_bundle.nodes

    def configuration(self):
        from dataclasses import asdict
        return {"pipeline": asdict(self.pipeline.config), "pipeline_fingerprint": self.pipeline.config.fingerprint,
                "revision": self.pipeline.state.get("revision", "legacy")}


def _node_text(node) -> str:
    try:
        from llama_index.core.schema import MetadataMode

        return node.get_content(metadata_mode=MetadataMode.NONE).strip()
    except Exception:
        return str(node)


def run(
    dataset_path: Path,
    out_path: Path,
    limit: int,
    top_k: int,
    recall_k: int,
    use_rerank: bool,
    retrieval_only: bool,
    metrics: list[str],
) -> dict:
    rows = _load_dataset(dataset_path, limit)
    if not rows:
        print("数据集为空。")
        return {}

    print(f"[Eval] 共 {len(rows)} 条样本，top_k={top_k}, recall_k={recall_k}, "
          f"rerank={'on' if use_rerank else 'off'}, mode={'retrieval' if retrieval_only else 'full'}")

    engine = EvalEngine(recall_k=recall_k, top_k=top_k, use_rerank=use_rerank)
    # 异源 judge：通过 JUDGE_MODEL / JUDGE_BACKEND 注入与答题模型不同的判卷模型
    # （如 JUDGE_MODEL=glm-5-3-260814 JUDGE_BACKEND=api）。留空则与答题同模型（不推荐）。
    judge = build_llm(
        os.getenv("JUDGE_MODEL") or None,
        backend=os.getenv("JUDGE_BACKEND") or None,
    )
    judge_model = getattr(judge, "model", "unknown")
    print(f"[Eval] Judge: {judge_model}"
          f"（答题: {getattr(engine.llm, 'model', 'unknown')}）")

    from pcb_rag.query import generate_answer_with_citation

    results: list[dict] = []
    started = time.time()

    for idx, row in enumerate(rows, 1):
        question = row["question"]
        ground_truth = row["ground_truth"]
        item: dict[str, Any] = {"id": row.get("id", f"#{idx}"), "question": question}

        try:
            nodes = engine.retrieve(question)
            contexts = [_node_text(n.node) for n in nodes]
            item["retrieved_count"] = len(contexts)
            item["retrieved_sources"] = [
                (n.node.metadata or {}).get("source_path", "") for n in nodes
            ]

            if retrieval_only:
                answer = ""
            else:
                result = generate_answer_with_citation(question, nodes, engine.llm)
                contexts = [node.node.text for node in result["evidence_bundle"].nodes]
                answer = result.get("answer", "")
                item["answer"] = answer
                item["citations"] = result.get("citations", [])

            scores = evaluate_case(
                question=question,
                answer=answer or ground_truth,  # 仅评检索时，用参考答案代替答案以跳过生成类指标
                contexts=contexts,
                ground_truth=ground_truth,
                llm=judge,
                metrics=(["context_precision", "context_recall"] if retrieval_only else metrics),
            )
            item["scores"] = scores
            item["ground_truth"] = ground_truth
            results.append(item)

            # 指标可能是 None（判卷失败 = 无法判定），不能直接按 float 格式化
            score_brief = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}=n/a"
                for k, v in scores.items()
                if k not in {"overall", "judge_failed"}
            )
            print(f"  [{idx}/{len(rows)}] {question[:36]}...  {score_brief}")
            if scores.get("judge_failed"):
                print(f"      ↳ 判卷失败（无法判定，不计入均值）: {', '.join(scores['judge_failed'])}")
        except Exception as exc:
            item["error"] = str(exc)
            results.append(item)
            print(f"  [{idx}/{len(rows)}] 失败: {exc}")

    elapsed = time.time() - started
    summary = summarize([r["scores"] for r in results if r.get("scores")])

    report = {
        "config": {
            **engine.configuration(),
            "dataset": str(dataset_path),
            "samples": len(results),
            "top_k": top_k,
            "recall_k": recall_k,
            "rerank": use_rerank,
            "mode": "retrieval" if retrieval_only else "full",
            "metrics": ["context_precision", "context_recall"] if retrieval_only else metrics,
            "elapsed_seconds": round(elapsed, 1),
            # 判卷可用性：数字只有在 judge 正常工作时才有意义，因此把状态写进报告
            "judge": {
                "model": judge_model,
                "answer_model": getattr(engine.llm, "model", "unknown"),
                "failures": summary.get("judge_failures", 0),
                "failure_rate": summary.get("judge_failure_rate", 0.0),
                "failure_rate_threshold": JUDGE_FAILURE_RATE_THRESHOLD,
                "invalid": summary.get("invalid", False),
            },
        },
        "summary": summary,
        "rows": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + format_summary(summary))
    print(f"\n报告已写入: {out_path}（耗时 {elapsed:.1f}s）")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PCB-RAG 评测")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "eval" / "datasets" / "golden.jsonl"),
        help="黄金数据集（JSONL）",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "eval" / "reports" / "report.json"),
        help="报告输出路径",
    )
    parser.add_argument("--limit", type=int, default=0, help="评测条数上限（0=全部）")
    parser.add_argument("--top-k", type=int, default=5, help="最终返回的上下文条数")
    parser.add_argument("--recall-k", type=int, default=50, help="精排前召回条数")
    parser.add_argument("--no-rerank", action="store_true", help="关闭精排")
    parser.add_argument("--retrieval-only", action="store_true", help="只评检索，不评生成")
    parser.add_argument(
        "--metrics",
        default=",".join(ALL_METRICS),
        help=f"要评估的指标，逗号分隔（默认全部: {','.join(ALL_METRICS)}）",
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    invalid = [m for m in metrics if m not in ALL_METRICS]
    if invalid:
        print(f"未知指标: {invalid}，可选: {ALL_METRICS}")
        return 1

    run(
        dataset_path=Path(args.dataset),
        out_path=Path(args.out),
        limit=args.limit,
        top_k=args.top_k,
        recall_k=args.recall_k,
        use_rerank=not args.no_rerank,
        retrieval_only=args.retrieval_only,
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
