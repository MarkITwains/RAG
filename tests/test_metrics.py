"""``eval.metrics`` 评测指标单元测试。

用假 LLM 替换真实模型调用，覆盖 JSON 提取、指标计算与阈值聚合逻辑，
使评测体系的核心算法可以在无模型服务的情况下被 CI 持续验证。
"""

import json

import pytest

from eval.metrics import (
    ALL_METRICS,
    THRESHOLDS,
    _clip01,
    _extract_json,
    _strip_think,
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_case,
    faithfulness,
    format_summary,
    summarize,
)


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeLLM:
    """按预设脚本依次返回响应，脚本用尽后重复最后一条。"""

    def __init__(self, *payloads):
        self._payloads = [
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False) for payload in payloads
        ]
        self.calls = 0

    def complete(self, prompt: str) -> _Response:
        index = min(self.calls, len(self._payloads) - 1)
        self.calls += 1
        return _Response(self._payloads[index])


class BrokenLLM:
    def complete(self, prompt: str) -> _Response:
        raise RuntimeError("模型服务不可用")


class TestJsonHelpers:
    def test_strip_think_removes_reasoning_block(self):
        assert _strip_think("<think>思考过程</think>最终答案") == "最终答案"

    def test_extract_json_finds_object_after_think(self):
        assert _extract_json('<think>hmm</think>{"score": 0.5}') == {"score": 0.5}

    def test_extract_json_finds_array(self):
        assert _extract_json("前缀 [1, 2] 后缀") == [1, 2]

    def test_extract_json_returns_none_for_plain_text(self):
        assert _extract_json("没有 JSON 的纯文本") is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(1.5, 1.0), (-1, 0.0), (0.42, 0.42), ("bad", 0.0), (None, 0.0)],
    )
    def test_clip01_bounds(self, raw, expected):
        assert _clip01(raw) == expected


class TestFaithfulness:
    def test_half_of_statements_supported(self):
        llm = FakeLLM({"statements": [{"text": "a", "supported": True}, {"text": "b", "supported": False}]})
        assert faithfulness("答案", ["上下文"], llm) == pytest.approx(0.5)

    def test_all_statements_supported(self):
        llm = FakeLLM({"statements": [{"text": "a", "supported": True}]})
        assert faithfulness("答案", ["上下文"], llm) == pytest.approx(1.0)

    def test_missing_statements_yields_zero(self):
        llm = FakeLLM({"statements": []})
        assert faithfulness("答案", ["上下文"], llm) == 0.0

    def test_empty_inputs_yield_zero(self):
        llm = FakeLLM({"statements": [{"text": "a", "supported": True}]})
        assert faithfulness("", ["上下文"], llm) == 0.0
        assert faithfulness("答案", [], llm) == 0.0

    def test_malformed_statement_entries_are_ignored(self):
        llm = FakeLLM({"statements": [{"text": "a", "supported": True}, "噪声"]})
        assert faithfulness("答案", ["上下文"], llm) == pytest.approx(0.5)

    def test_broken_llm_falls_back_to_zero(self):
        assert faithfulness("答案", ["上下文"], BrokenLLM()) == 0.0


class TestAnswerRelevancy:
    def test_score_is_passed_through(self):
        assert answer_relevancy("问题", "回答", FakeLLM({"score": 0.9})) == pytest.approx(0.9)

    def test_out_of_range_score_is_clipped(self):
        assert answer_relevancy("问题", "回答", FakeLLM({"score": 5})) == 1.0

    def test_missing_score_yields_zero(self):
        assert answer_relevancy("问题", "回答", FakeLLM({"unexpected": 1})) == 0.0

    def test_empty_inputs_yield_zero(self):
        assert answer_relevancy("", "回答", FakeLLM({"score": 1})) == 0.0
        assert answer_relevancy("问题", "", FakeLLM({"score": 1})) == 0.0


class TestContextPrecision:
    def test_average_precision_penalizes_late_hits(self):
        # 命中位置 1 与 3：AP = (1/1 + 2/3) / 2
        llm = FakeLLM({"relevant": True}, {"relevant": False}, {"relevant": True})
        assert context_precision("问题", ["c1", "c2", "c3"], llm) == pytest.approx(0.8333, abs=1e-4)

    def test_all_contexts_relevant(self):
        llm = FakeLLM({"relevant": True}, {"relevant": True})
        assert context_precision("问题", ["c1", "c2"], llm) == pytest.approx(1.0)

    def test_no_relevant_context_yields_zero(self):
        llm = FakeLLM({"relevant": False}, {"relevant": False})
        assert context_precision("问题", ["c1", "c2"], llm) == 0.0

    def test_empty_contexts_yield_zero(self):
        assert context_precision("问题", [], FakeLLM({"relevant": True})) == 0.0


class TestContextRecall:
    def test_partial_coverage(self):
        llm = FakeLLM({"statements": [{"found": True}, {"found": True}, {"found": False}]})
        assert context_recall("参考答案", ["上下文"], llm) == pytest.approx(2 / 3, abs=1e-4)

    def test_empty_inputs_yield_zero(self):
        llm = FakeLLM({"statements": [{"found": True}]})
        assert context_recall("", ["上下文"], llm) == 0.0
        assert context_recall("参考答案", [], llm) == 0.0


class TestAggregation:
    def test_evaluate_case_averages_all_metrics(self):
        payloads = [
            {"statements": [{"text": "a", "supported": True}, {"text": "b", "supported": False}]},
            {"score": 1.0},
            {"relevant": True},
            {"relevant": False},
            {"statements": [{"found": True}, {"found": False}]},
        ]
        result = evaluate_case("问题", "回答", ["c1", "c2"], "参考答案", FakeLLM(*payloads))

        assert set(result) == {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "overall",
        }
        assert result["faithfulness"] == pytest.approx(0.5)
        assert result["answer_relevancy"] == pytest.approx(1.0)
        assert result["context_precision"] == pytest.approx(1.0)
        assert result["context_recall"] == pytest.approx(0.5)
        assert result["overall"] == pytest.approx(0.75)

    def test_evaluate_case_supports_metric_subset(self):
        result = evaluate_case(
            "问题", "回答", ["c1"], "参考答案", FakeLLM({"relevant": True}), metrics=["context_precision"]
        )
        assert set(result) == {"context_precision", "overall"}

    def test_summarize_marks_threshold_pass_and_fail(self):
        summary = summarize([{"faithfulness": 0.9, "context_recall": 0.1, "overall": 0.5}])
        assert summary["faithfulness"]["passed"] is True
        assert summary["context_recall"]["passed"] is False
        assert summary["overall"]["avg"] == pytest.approx(0.5)

    def test_summarize_ignores_empty_rows(self):
        assert summarize([]) == {}
        assert summarize([{}]) == {}

    def test_format_summary_marks_pass_and_fail(self):
        text = format_summary(summarize([{"faithfulness": 0.9, "context_recall": 0.1}]))
        assert "达标" in text
        assert "未达标" in text

    def test_format_summary_without_data(self):
        assert format_summary({}) == "（无评测结果）"

    def test_thresholds_cover_all_metrics(self):
        assert set(THRESHOLDS) == set(ALL_METRICS)
