"""构建评测用黄金数据集。

从已入库的语料（词法缓存 ``data/lexical_corpus.jsonl``）中抽样 chunk，
用 LLM 反推出「问题 + 参考答案 + 出处」，生成评测数据集。

用法::

    # 生成 100 道题（每个 chunk 1 个问题）
    python eval/build_golden_dataset.py --limit 100 --per-chunk 1

    # 指定语料与输出
    python eval/build_golden_dataset.py \
        --corpus data/lexical_corpus.jsonl \
        --out eval/datasets/golden.jsonl

输出格式（JSONL，每行一条）::

    {"id": "q-0001", "question": "...", "ground_truth": "...",
     "source_chunk_id": "...", "source_path": "..."}

注意事项：

- 自动生成的数据集**必须人工抽检**（建议至少复核 20~30 条），
  低质量黄金集会让后续所有评测结论都不可信。
- 也可以完全不用本脚本，手工按上述格式准备数据集。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

# 允许直接运行脚本时导入 pcb_rag
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pcb_rag.env_loader import load_project_env  # noqa: E402

load_project_env()  # 直接运行本脚本时也能读到 .env 里的 API 配置

from pcb_rag.api_clients import build_llm  # noqa: E402


_GEN_PROMPT = """你是 PCB 领域的出题专家。下面是一段来自规范/工艺文档的片段，请基于它出 {n} 道考察性问题并给出参考答案。

要求：
1. 问题必须**能且仅能**依据该片段回答，不要引入片段外知识
2. 问题要像真实工程师会问的问题，避免"根据上文"这类指代
3. 参考答案要简洁准确，控制在 150 字以内
4. 只输出 JSON，不要解释

片段：
{chunk}

输出 JSON 格式：
{{"items": [{{"question": "问题", "ground_truth": "参考答案"}}]}}"""


def _extract_json(text: str) -> Optional[Any]:
    import re

    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def load_corpus(path: Path) -> list[dict]:
    """读取词法缓存语料（每行一个 chunk）。"""
    if not path.exists():
        raise FileNotFoundError(
            f"语料文件不存在: {path}\n"
            f"请先执行文档入库（bash scripts/run_ingest.sh），"
            f"或运行一次问答以生成词法缓存；也可用 --corpus 指定其他 JSONL。"
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
            text = obj.get("text")
            if isinstance(text, str) and len(text) >= 120:
                rows.append(obj)
    return rows


def build_dataset(
    corpus_path: Path,
    out_path: Path,
    limit: int,
    per_chunk: int,
    min_chars: int,
    seed: int,
) -> int:
    rows = load_corpus(corpus_path)
    rows = [r for r in rows if len(r.get("text", "")) >= min_chars]
    if not rows:
        print("语料中没有满足长度要求的 chunk。")
        return 0

    random.seed(seed)
    random.shuffle(rows)
    rows = rows[: max(1, limit)]

    llm = build_llm()
    print(f"使用 LLM: {getattr(llm, 'model', 'unknown')}")
    print(f"待处理 chunk: {len(rows)}（每个生成 {per_chunk} 题）")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, 1):
            chunk = row.get("text", "")
            meta = row.get("metadata") or {}
            try:
                response = llm.complete(_GEN_PROMPT.format(n=per_chunk, chunk=chunk[:2500]))
                text = response.text if hasattr(response, "text") else str(response)
                data = _extract_json(text)
            except Exception as exc:
                print(f"  [{idx}/{len(rows)}] 生成失败: {exc}")
                continue

            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list):
                print(f"  [{idx}/{len(rows)}] 解析失败，跳过")
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                ground_truth = str(item.get("ground_truth", "")).strip()
                if len(question) < 6 or len(ground_truth) < 4:
                    continue

                written += 1
                record = {
                    "id": f"q-{written:04d}",
                    "question": question,
                    "ground_truth": ground_truth,
                    "source_chunk_id": row.get("id", ""),
                    "source_path": meta.get("source_path", ""),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 10 == 0:
                print(f"  进度 {idx}/{len(rows)}，已生成 {written} 题")

    print(f"完成：{written} 条写入 {out_path}")
    print("请人工抽检 20~30 条，剔除不可回答或答案有误的样本后再用于评测。")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 评测黄金数据集")
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "data" / "lexical_corpus.jsonl"),
        help="语料文件（JSONL，默认 data/lexical_corpus.jsonl）",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "eval" / "datasets" / "golden.jsonl"),
        help="输出数据集路径",
    )
    parser.add_argument("--limit", type=int, default=100, help="抽样的 chunk 数量")
    parser.add_argument("--per-chunk", type=int, default=1, help="每个 chunk 生成的问题数")
    parser.add_argument("--min-chars", type=int, default=200, help="chunk 最小字符数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    build_dataset(
        corpus_path=Path(args.corpus),
        out_path=Path(args.out),
        limit=args.limit,
        per_chunk=args.per_chunk,
        min_chars=args.min_chars,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
