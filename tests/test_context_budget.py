"""上下文预算与三段式组装的单元测试。

覆盖的语义点：
- token 估算对中英混排的方向性（中文按 1.5 字/token，英文按 4 字符/token）
- 事实句识别：数值+单位、标准号、规范性措辞；无命中时退回首句
- 预算切分：原文证据拿剩余额度；预算过小时放弃事实段
- 逐条填充：超预算按比例截断、剩余不足则停止
- 三段渲染：编号在事实段/证据段/引用表三处一致，事实段受预算约束
"""

from pcb_rag.context_budget import (
    dedupe_by_key,
    estimate_tokens,
    extract_fact_sentences,
    fit_evidence,
    plan_budget,
    render_sections,
)


# ---------------------------------------------------------------------------
# token 估算
# ---------------------------------------------------------------------------
class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_chinese_denser_than_english(self):
        """同样字符数，中文占的 token 更多。"""
        zh = estimate_tokens("印制电路板阻抗控制")
        en = estimate_tokens("abcdefghij")
        assert zh > en

    def test_scales_with_length(self):
        short = estimate_tokens("镀金层厚度")
        long = estimate_tokens("镀金层厚度" * 20)
        assert long > short

    def test_minimum_one_token_for_nonempty(self):
        assert estimate_tokens("a") >= 1


# ---------------------------------------------------------------------------
# 事实句识别
# ---------------------------------------------------------------------------
class TestExtractFactSentences:
    def test_picks_numeric_spec(self):
        text = "本节介绍镀金工艺。镀金层厚度应不小于 0.05 μm，测量方法见附录。"
        out = extract_fact_sentences(text)
        assert "0.05" in out

    def test_picks_standard_number(self):
        text = "依据 GB/T 4677-2002 的规定。测试前需静置 24 小时。"
        out = extract_fact_sentences(text)
        assert "GB" in out

    def test_picks_normative_wording(self):
        text = "焊盘不得有氧化层。其余按常规执行。"
        out = extract_fact_sentences(text)
        assert "不得" in out

    def test_falls_back_to_first_sentence(self):
        text = "阻抗控制是高速板设计的核心。后面还有别的内容。"
        out = extract_fact_sentences(text)
        assert out.startswith("阻抗控制")

    def test_respects_limit(self):
        text = (
            "镀金层厚度应不小于 0.05 μm。"
            "镍层厚度应为 3~6 μm。"
            "金层纯度必须达到 99.9%。"
        )
        out = extract_fact_sentences(text, limit=1)
        assert out.count(" / ") == 0

    def test_empty_input(self):
        assert extract_fact_sentences("") == ""


# ---------------------------------------------------------------------------
# 预算切分
# ---------------------------------------------------------------------------
class TestPlanBudget:
    def test_total_is_preserved(self):
        plan = plan_budget(6000)
        consumed = plan["ref"] + plan["facts"] + plan["chunks"] + plan["overhead"]
        assert consumed == 6000

    def test_chunks_take_the_remainder(self):
        """原文证据拿剩余额度，不受固定比例约束。"""
        plan = plan_budget(10000, facts_ratio=0.2)
        assert plan["chunks"] > plan["facts"]

    def test_zero_budget(self):
        plan = plan_budget(0)
        assert plan["total"] == 0 and plan["chunks"] == 0

    def test_tiny_budget_drops_facts(self):
        """预算太小（装不下两条最小证据）时放弃事实段。

        300 tokens：facts=66、ref=18、overhead=30 → 原文仅剩 186，
        低于 2×min_chunk_tokens(240)，应把事实段让出来。
        """
        plan = plan_budget(300, min_chunk_tokens=120)
        assert plan["facts"] == 0
        assert plan["chunks"] > 0

    def test_ref_budget_capped(self):
        small = plan_budget(1000)
        big = plan_budget(100000)
        assert big["ref"] == small["ref"] or big["ref"] <= 200


# ---------------------------------------------------------------------------
# 逐条填充
# ---------------------------------------------------------------------------
class TestFitEvidence:
    def test_fits_all_when_within_budget(self):
        items = ["短内容一。", "短内容二。"]
        out = fit_evidence(items, lambda x: x, chunk_budget=5000)
        assert len(out) == 2
        assert all(not r["truncated"] for r in out)

    def test_truncates_last_item(self):
        items = ["印制电路板" * 200] * 3
        out = fit_evidence(items, lambda x: x, chunk_budget=400, min_chunk_tokens=50)
        assert out
        assert out[-1]["truncated"] is True
        assert out[-1]["text"].endswith("…")

    def test_stops_when_remaining_too_small(self):
        """剩余额度不足一条最小证据时停止，不塞半句话。"""
        items = ["x" * 2000 for _ in range(5)]
        out = fit_evidence(items, lambda x: x, chunk_budget=300, min_chunk_tokens=200)
        assert len(out) <= 1

    def test_zero_budget(self):
        assert fit_evidence(["a"], lambda x: x, chunk_budget=0) == []

    def test_skips_empty_text(self):
        out = fit_evidence(["", "  ", "有效内容"], lambda x: x, chunk_budget=1000)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 三段渲染
# ---------------------------------------------------------------------------
def _entry(idx: int, text: str, section: str = "5.2 镀层", doc_type: str = "national_standard"):
    return {
        "text": text,
        "source_name": f"标准{idx}",
        "source_path": f"{idx}.txt",
        "section": section,
        "metadata": {"doc_type": doc_type, "standards": ["GB/T 4677"]},
    }


class TestRenderSections:
    def test_three_sections_present(self):
        entries = [_entry(1, "镀金层厚度应不小于 0.05 μm。")]
        out = render_sections(entries, facts_budget=500)
        assert "【事实摘要】" in out
        assert "【原文证据】" in out
        assert "【引用来源】" in out

    def test_numbering_aligned_across_sections(self):
        entries = [_entry(1, "甲。"), _entry(2, "乙。")]
        out = render_sections(entries, facts_budget=500)
        assert "[2] 标准2" in out
        assert "[2] 2.txt" in out

    def test_facts_section_omitted_when_budget_zero(self):
        out = render_sections([_entry(1, "甲。")], facts_budget=0)
        assert "【事实摘要】" not in out
        assert "【原文证据】" in out

    def test_facts_section_respects_budget(self):
        entries = [_entry(i, "镀金层厚度应不小于 0.05 μm。") for i in range(1, 20)]
        out = render_sections(entries, facts_budget=80)
        fact_part = out.split("【原文证据】")[0]
        # 预算只够一两条，不应把 19 条全塞进事实段
        assert fact_part.count("[") <= 4

    def test_locator_includes_standard_number(self):
        out = render_sections([_entry(1, "甲。")], facts_budget=500)
        assert "GB/T 4677" in out

    def test_empty_entries(self):
        assert render_sections([], facts_budget=500) == ""


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------
class TestDedupe:
    def test_keeps_first_occurrence(self):
        items = [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "a", "v": 3}]
        out = dedupe_by_key(items, lambda x: x["id"])
        assert [x["v"] for x in out] == [1, 2]

    def test_none_key_falls_back_to_object_identity(self):
        items = [{"id": None}, {"id": None}]
        out = dedupe_by_key(items, lambda x: x["id"])
        assert len(out) == 2
