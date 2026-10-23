"""上下文构建的集成测试（需要 llama-index-core）。

与 ``test_context_budget.py`` 的分工：
- 后者测纯逻辑（预算切分、截断、渲染），不依赖任何第三方库；
- 本文件测适配层 ``build_context_with_budget``：NodeWithScore → 三段式上下文，
  验证 metadata 取值、编号对齐、去重与预算耗尽后的行为。
"""

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

from pcb_rag import query as rag_query  # noqa: E402


def _make_doc(
    text: str,
    node_id: str,
    *,
    score: float = 0.9,
    source: str = "GBT2036-1994印制电路术语.txt",
    section: str = "5.2.6 照相缩小尺寸",
    doc_type: str = "national_standard",
):
    node = TextNode(
        text=text,
        id_=node_id,
        metadata={
            "source_path": source,
            "parent_title": section,
            "doc_type": doc_type,
            "standards": ["GB/T 2036"],
        },
    )
    return NodeWithScore(node=node, score=score)


SPEC_TEXT = (
    "5.2.6 照相缩小尺寸 photographic reduction dimension\n"
    "照相底图上给照相操作者指示照相原版缩小程度的尺寸。"
    "该尺寸值是 1:1 比例的精确数值，并且必须在照相底图上标注。"
)

NORMAL_TEXT = "镀金层厚度应不小于 0.05 μm，镍层厚度应为 3~6 μm，测试方法按 GB/T 4677 执行。"


class TestBuildContextWithBudget:
    def test_returns_three_sections(self):
        docs = [_make_doc(SPEC_TEXT, "a"), _make_doc(NORMAL_TEXT, "b")]
        context, sources = rag_query.build_context_with_budget(docs)

        assert "【事实摘要】" in context
        assert "【原文证据】" in context
        assert "【引用来源】" in context
        assert len(sources) == 2

    def test_source_numbers_match_context_labels(self):
        docs = [_make_doc(SPEC_TEXT, "a"), _make_doc(NORMAL_TEXT, "b")]
        context, sources = rag_query.build_context_with_budget(docs)

        for info in sources:
            idx = info["index"]
            assert f"[{idx}]" in context
        assert [s["index"] for s in sources] == [1, 2]

    def test_metadata_section_is_used(self):
        docs = [_make_doc(NORMAL_TEXT, "a", section="6.3 镀金层要求")]
        context, sources = rag_query.build_context_with_budget(docs)
        assert "6.3 镀金层要求" in context
        assert sources[0]["section"] == "6.3 镀金层要求"

    def test_doc_type_appears_in_fact_locator(self):
        docs = [_make_doc(NORMAL_TEXT, "a")]
        context, _ = rag_query.build_context_with_budget(docs)
        assert "national_standard" in context

    def test_dedupe_same_node_id(self):
        docs = [_make_doc(NORMAL_TEXT, "dup"), _make_doc(NORMAL_TEXT, "dup")]
        _, sources = rag_query.build_context_with_budget(docs)
        assert len(sources) == 1

    def test_zero_budget_returns_empty_for_fallback(self):
        """预算为 0 时返回空，调用方据此回退到固定条数逻辑。"""
        docs = [_make_doc(NORMAL_TEXT, "a")]
        context, sources = rag_query.build_context_with_budget(docs, max_total_tokens=0)
        assert context == ""
        assert sources == []

    def test_small_budget_still_produces_evidence(self):
        docs = [_make_doc(NORMAL_TEXT, "a")]
        context, sources = rag_query.build_context_with_budget(docs, max_total_tokens=400)
        assert sources
        assert "【原文证据】" in context

    def test_more_docs_than_old_fixed_limit(self):
        """旧实现固定 5 条；预算模式应能吃下更多短块。"""
        docs = [_make_doc(f"第{i}条：镀金层厚度应不小于 0.05 μm。", f"n{i}", score=1.0 - i * 0.01) for i in range(12)]
        _, sources = rag_query.build_context_with_budget(docs, max_total_tokens=6000)
        assert len(sources) > 5

    def test_long_doc_is_truncated_not_dropped(self):
        docs = [_make_doc("镀金层厚度应不小于 0.05 μm。" * 400, "long")]
        _, sources = rag_query.build_context_with_budget(docs, max_total_tokens=600)
        assert len(sources) == 1
        assert sources[0]["truncated"] is True

    def test_empty_input(self):
        assert rag_query.build_context_with_budget([]) == ("", [])


class TestLegacyPathStillAvailable:
    def test_explicit_max_docs_uses_old_logic(self):
        """显式传 max_docs/max_chars 时保持旧的固定截断语义（评测脚本依赖）。"""
        docs = [_make_doc(NORMAL_TEXT, f"n{i}") for i in range(10)]
        context, sources = rag_query.build_context_with_sources(docs, max_docs=3, max_chars=100)
        assert len(sources) == 3
        assert "【事实摘要】" not in context

    def test_default_uses_budget_mode(self):
        docs = [_make_doc(NORMAL_TEXT, "a")]
        context, _ = rag_query.build_context_with_sources(docs)
        if rag_query.RAG_CONTEXT_MAX_TOKENS > 0:
            assert "【原文证据】" in context
