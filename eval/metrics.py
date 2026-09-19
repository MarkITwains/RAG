"""PCB-RAG 评测指标（LLM-as-Judge 实现，不引入额外依赖）。

四个核心指标：

- **Faithfulness（忠实度）**：答案中的每条陈述是否能被检索上下文支持
- **Answer Relevancy（答案相关性）**：答案是否真正回应了问题
- **Context Precision（上下文精确率）**：相关上下文是否排在前面（用平均精度 AP）
- **Context Recall（上下文召回率）**：参考答案所需信息是否都被召回

生产参考阈值（可按业务与风险容忍度调整）：

===============  =========
指标              参考阈值
===============  =========
Faithfulness      >= 0.75
Answer Relevancy  >= 0.80
Context Precision >= 0.70
Context Recall    >= 0.80
===============  =========

诊断建议：指标要组合起来读，而不是单独看。

- 高忠实度 + 低上下文相关性 → 生成正常，是**检索问题**
- 低忠实度 + 上下文正确     → **生成漂移**（收紧提示 / 降温 / 换模型）
- 低忠实度 + 答案却正确     → 最危险：模型绕开检索、直接用训练数据作答
- 高召回 + 低精确率         → 信息都在但被噪声淹没，应加强重排
- 低召回 + 高精确率         → 召回太窄，应增大 top-k 或改进查询改写
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Sequence

THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.80,
}

ALL_METRICS = list(THRESHOLDS.keys())

#: 判卷失败率超过该比例时，整份报告标记 ``invalid``（数值不可信）。
JUDGE_FAILURE_RATE_THRESHOLD = float(os.getenv("JUDGE_FAILURE_RATE_THRESHOLD", "0.2"))

#: ``summarize`` 输出里保留给判卷状态的两个键（不是指标名）
_RESERVED_KEYS = ("judge_failures", "invalid", "judge_failure_rate")

# ---------------------------------------------------------------------------
# 三态约定（很重要，别改回去）
#
# 各指标函数的返回值有三种含义：
#   - ``float``：评测成功，该分数有效
#   - ``None`` ：**评测失败**（judge 调用异常 / 返回不可解析 / judge 表示没有可判定陈述）
#   - ``0.0``  ：评测成功，且结果确实是零分（例如根本没召回上下文）
#
# 早先的实现把 ``None`` 情况静默降级为 ``0.0``，于是"judge 挂了"与"答得差"
# 混为一谈，所有指标被系统性**低估**，而且没有任何地方能看到这件事发生过。
# 现在由 `summarize` 分列 evaluated / failed，并在失败率过高时把报告标记为
# invalid —— 宁可让报告看起来"不可用"，也不要给出一个看不出问题的漂亮数字。
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)


def _extract_json(text: str) -> Optional[Any]:
    """从模型输出中提取第一个 JSON 对象或数组。"""
    text = _strip_think(text)
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group())
        except Exception:
            continue
    return None


def _ask_json(llm, prompt: str) -> Optional[Any]:
    """调用 LLM 并解析 JSON。

    **评测失败返回 ``None``**（而不是降级成某个默认值）—— 见上方"三态约定"。
    调用方必须显式处理 None，不能当成 0 分。
    """
    try:
        response = llm.complete(prompt)
        text = response.text if hasattr(response, "text") else str(response)
        return _extract_json(text)
    except Exception:
        return None


def _clip01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def _short(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit]


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """你是严格的答案审核员。请判断"回答"中的每条事实性陈述能否从"参考资料"中得到支持。

参考资料：
{context}

回答：
{answer}

请输出 JSON（每条陈述一项）：
{{"statements": [{{"text": "陈述原文", "supported": true}}]}}

只输出 JSON，不要解释。"""


def faithfulness(answer: str, contexts: Sequence[str], llm) -> Optional[float]:
    """答案中能被上下文支持的陈述占比。

    Returns:
        0.0 表示"确实没有可支持的陈述 / 没有上下文"；``None`` 表示**无法判定**
        （judge 调用失败，或 judge 表示答案里没有可判定的陈述）。
    """
    if not answer or not contexts:
        return 0.0
    context = "\n\n".join(contexts)
    data = _ask_json(
        llm,
        _FAITHFULNESS_PROMPT.format(context=_short(context, 8000), answer=_short(answer, 3000)),
    )
    if data is None:
        return None
    statements = data.get("statements") if isinstance(data, dict) else None
    if not isinstance(statements, list) or not statements:
        # judge 认为没有可判定陈述："无法判定" ≠ "全不被支持"
        return None
    supported = sum(1 for s in statements if isinstance(s, dict) and s.get("supported"))
    return _clip01(supported / len(statements))


# ---------------------------------------------------------------------------
# 2. Answer Relevancy
# ---------------------------------------------------------------------------
_RELEVANCY_PROMPT = """你是严格的答案审核员。请判断"回答"是否真正回应了"问题"（只判断是否切题，不判断事实是否正确）。

问题：{question}

回答：{answer}

请输出 JSON：{{"score": 0.0}}

其中 score 为 0~1 的数字：1 表示完全切题且有实质内容，0 表示答非所问。
只输出 JSON。"""


def answer_relevancy(question: str, answer: str, llm) -> Optional[float]:
    """答案与问题的相关程度（0~1）。``None`` 表示无法判定。"""
    if not question or not answer:
        return 0.0
    data = _ask_json(
        llm,
        _RELEVANCY_PROMPT.format(question=_short(question, 1000), answer=_short(answer, 3000)),
    )
    if not isinstance(data, dict) or "score" not in data:
        return None
    return _clip01(data.get("score"))


# ---------------------------------------------------------------------------
# 3. Context Precision（平均精度 AP）
# ---------------------------------------------------------------------------
_CONTEXT_RELEVANT_PROMPT = """请判断下面这段参考资料是否有助于回答该问题。

问题：{question}

参考资料：
{context}

只输出 JSON：{{"relevant": true}}"""


def context_precision(question: str, contexts: Sequence[str], llm) -> Optional[float]:
    """相关上下文是否排在前面；用平均精度（AP）度量位置惩罚。

    AP 依赖**每条上下文的位置**，因此只要有任意一条判卷失败，位置序列就被破坏、
    AP 无法可靠计算 —— 此时返回 ``None``（而不是把失败项当作"不相关"）。
    """
    if not contexts:
        return 0.0

    flags: list[bool] = []
    for ctx in contexts:
        data = _ask_json(
            llm,
            _CONTEXT_RELEVANT_PROMPT.format(question=_short(question, 1000), context=_short(ctx, 2000)),
        )
        if not isinstance(data, dict) or "relevant" not in data:
            return None  # 位置信息不完整 → 无法计算 AP
        flags.append(bool(data.get("relevant")))

    total_relevant = sum(flags)
    if total_relevant == 0:
        return 0.0

    hits = 0
    acc = 0.0
    for i, flag in enumerate(flags, 1):
        if flag:
            hits += 1
            acc += hits / i  # precision@i
    return _clip01(acc / total_relevant)


# ---------------------------------------------------------------------------
# 4. Context Recall
# ---------------------------------------------------------------------------
_RECALL_PROMPT = """你是严格的审核员。请判断"参考答案"中的每条关键信息能否在"参考资料"中找到。

参考资料：
{context}

参考答案：
{ground_truth}

请输出 JSON（每条关键信息一项）：
{{"statements": [{{"text": "信息点", "found": true}}]}}

只输出 JSON，不要解释。"""


def context_recall(ground_truth: str, contexts: Sequence[str], llm) -> Optional[float]:
    """参考答案中的信息点被上下文覆盖的比例。``None`` 表示无法判定。"""
    if not ground_truth or not contexts:
        return 0.0
    context = "\n\n".join(contexts)
    data = _ask_json(
        llm,
        _RECALL_PROMPT.format(context=_short(context, 8000), ground_truth=_short(ground_truth, 3000)),
    )
    if data is None:
        return None
    statements = data.get("statements") if isinstance(data, dict) else None
    if not isinstance(statements, list) or not statements:
        return None
    found = sum(1 for s in statements if isinstance(s, dict) and s.get("found"))
    return _clip01(found / len(statements))


# ---------------------------------------------------------------------------
# 批量入口
# ---------------------------------------------------------------------------
def evaluate_case(
    question: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: str,
    llm,
    metrics: Optional[Sequence[str]] = None,
) -> dict:
    """评估单条样本，返回各指标得分、overall 均值与判卷失败清单。

    返回值中：
    - 各指标为 ``float``（有效）或 ``None``（**判卷失败，无法判定**）；
    - ``judge_failed``：本条样本里无法判定的指标名列表；
    - ``overall``：仅对**有效**分数求均值（没有任何有效分数时不输出 overall）。
    """
    names = list(metrics or ALL_METRICS)
    result: dict[str, Any] = {}

    if "faithfulness" in names:
        result["faithfulness"] = faithfulness(answer, contexts, llm)
    if "answer_relevancy" in names:
        result["answer_relevancy"] = answer_relevancy(question, answer, llm)
    if "context_precision" in names:
        result["context_precision"] = context_precision(question, contexts, llm)
    if "context_recall" in names:
        result["context_recall"] = context_recall(ground_truth, contexts, llm)

    evaluated = [v for v in result.values() if isinstance(v, (int, float))]
    if evaluated:
        result["overall"] = round(sum(evaluated) / len(evaluated), 4)
    result["judge_failed"] = [k for k, v in result.items() if v is None]
    return result


def summarize(rows: Sequence[dict]) -> dict:
    """按指标聚合均值，并对照阈值给出通过情况与**判卷失败统计**。

    每个指标的条目里包含：

    - ``avg``：仅对**有效**分数求均值（判卷失败的样本不计入也不补零）
    - ``evaluated`` / ``failed``：有效与失败的样本数
    - ``failure_rate``：失败占比
    - ``passed``：avg 与阈值比较

    顶层额外给出 ``judge_failures``（失败总数）与 ``invalid``（失败率超过
    ``JUDGE_FAILURE_RATE_THRESHOLD`` 时为 True —— 此时报告里的数字不应被采信）。
    """
    rows = [r for r in rows if r]
    if not rows:
        return {}

    summary: dict[str, Any] = {}
    total_evaluated = 0
    total_failed = 0

    for name in ALL_METRICS:
        values = [r[name] for r in rows if isinstance(r.get(name), (int, float))]
        failed = sum(1 for r in rows if name in r and r.get(name) is None)
        if not values and not failed:
            continue
        threshold = THRESHOLDS.get(name)
        entry: dict[str, Any] = {
            "evaluated": len(values),
            "failed": failed,
        }
        total_evaluated += len(values)
        total_failed += failed
        if values:
            avg = sum(values) / len(values)
            entry["avg"] = round(avg, 4)
            entry["threshold"] = threshold
            entry["passed"] = (avg >= threshold) if threshold is not None else None
        else:
            # 全部失败：不给 avg，避免下游把它当成 0 分
            entry["avg"] = None
            entry["threshold"] = threshold
            entry["passed"] = None
        judged = len(values) + failed
        if judged:
            entry["failure_rate"] = round(failed / judged, 4)
        summary[name] = entry

    overall_values = [r["overall"] for r in rows if isinstance(r.get("overall"), (int, float))]
    if overall_values:
        summary["overall"] = {"avg": round(sum(overall_values) / len(overall_values), 4)}

    if not summary:
        return {}

    judged_total = total_evaluated + total_failed
    failure_rate = (total_failed / judged_total) if judged_total else 0.0
    summary["judge_failures"] = total_failed
    summary["judge_failure_rate"] = round(failure_rate, 4)
    summary["invalid"] = bool(failure_rate > JUDGE_FAILURE_RATE_THRESHOLD)
    return summary


def format_summary(summary: dict) -> str:
    """把汇总结果格式化为可读文本。"""
    if not summary:
        return "（无评测结果）"

    lines = ["指标                    平均分   阈值    是否达标   有效/失败", "-" * 68]
    for name, info in summary.items():
        if name == "overall" or name in _RESERVED_KEYS:
            continue
        threshold = info.get("threshold")
        passed = info.get("passed")
        flag = "达标" if passed else "未达标"
        if threshold is None:
            flag = "-"
        avg_text = f"{info['avg']:.4f}" if isinstance(info.get("avg"), (int, float)) else "   n/a"
        judged = f"{info.get('evaluated', 0)}/{info.get('failed', 0)}"
        lines.append(f"{name:<22} {avg_text:<8} {str(threshold or '-'):<7} {flag:<10} {judged}")

    if "overall" in summary:
        lines.append("-" * 68)
        lines.append(f"{'overall':<22} {summary['overall']['avg']:<8.4f}")

    failures = summary.get("judge_failures", 0)
    if failures:
        rate = summary.get("judge_failure_rate", 0.0)
        lines.append("-" * 68)
        lines.append(
            f"判卷失败: {failures} 次（失败率 {rate:.1%}，阈值 {JUDGE_FAILURE_RATE_THRESHOLD:.0%}）"
        )
        lines.append(
            "提示：'无法判定'与'答得差'是两回事 —— 失败样本不计入均值，也不补零。"
        )
    if summary.get("invalid"):
        lines.append("")
        lines.append("⚠️  判卷失败率超过阈值，本报告的指标不可信，请先排查 judge 服务后重跑。")

    return "\n".join(lines)
