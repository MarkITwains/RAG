"""上下文压缩与去重单元测试。

覆盖：分句、词元抽取、句子打分、抽取式压缩、相似度与去重、字符预算裁剪。
"""

from types import SimpleNamespace

from pcb_rag.compression import (
    _shingles,
    extract_terms,
    split_sentences,
    similarity,
)
from pcb_rag.compression import dedupe_nodes as _dedupe_nodes
from pcb_rag.compression import compress_text_extractive as _compress_text
from pcb_rag.compression import score_sentence as _score_sentence


def _node(text: str, node_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata={}, node_id=node_id)


class TestSplitSentences:
    def test_splits_on_chinese_punctuation(self):
        sentences = split_sentences("第一句说明镀层的适用范围。第二句给出厚度要求！第三句描述试验方法？")
        assert len(sentences) == 3

    def test_splits_on_newlines(self):
        sentences = split_sentences("第一行内容。\n第二行内容。")
        assert len(sentences) == 2

    def test_drops_short_fragments(self):
        assert all(len(s) >= 6 for s in split_sentences("好。这是一句完整的话。"))

    def test_empty_input(self):
        assert split_sentences("") == []

    def test_long_sentence_falls_back_to_comma(self):
        # 单行超过 MAX_SENTENCE_LEN（400 字符）且无句末标点，应退化为按逗号切分
        long_text = "这是一句用于验证长句回退切分的说明文字，" * 40
        sentences = split_sentences(long_text)
        assert len(sentences) > 1


class TestExtractTerms:
    def test_extracts_chinese_ngrams(self):
        terms = extract_terms("镀金层厚度要求")
        assert "镀金" in terms
        assert "厚度" in terms

    def test_extracts_ascii_words_and_standards(self):
        terms = extract_terms("GB/T 4677 与 IPC-6012 的差异")
        assert any("gb/t" in t for t in terms)
        assert "4677" in terms or any("4677" in t for t in terms)

    def test_long_terms_first(self):
        terms = extract_terms("镀金层厚度测试方法")
        assert len(terms[0]) >= len(terms[-1])

    def test_empty_query(self):
        assert extract_terms("") == []


class TestScoreSentence:
    def test_query_overlap_raises_score(self):
        terms = extract_terms("镀金层厚度")
        hit = _score_sentence("镀金层厚度应不小于 0.8 μm。", terms, 0, 5)
        miss = _score_sentence("本文件规定了试验的一般要求。", terms, 0, 5)
        assert hit > miss

    def test_position_bonus_favours_leading_sentence(self):
        terms = extract_terms("阻抗控制")
        leading = _score_sentence("阻抗控制要求如下。", terms, 0, 5)
        trailing = _score_sentence("阻抗控制要求如下。", terms, 4, 5)
        assert leading > trailing

    def test_numeric_sentence_gains_weight(self):
        terms = extract_terms("厚度")
        numeric = _score_sentence("厚度为 0.8 mm。", terms, 0, 5)
        plain = _score_sentence("厚度由供需双方协商确定。", terms, 0, 5)
        assert numeric > plain


class TestExtractiveCompression:
    def test_keeps_relevant_sentence(self):
        text = (
            "本文件规定了印制板的通用要求。\n"
            "镀金层厚度应不小于 0.8 μm，测试方法按 GB/T 4677 执行。\n"
            "本文件由全国印制电路标准化技术委员会提出。\n"
            "目 录\n"
            "1 范围 ……………………………………………………… 1\n"
            "2 规范性引用文件 …………………………………………… 2\n"
        )
        compressed = _compress_text(text, "镀金层厚度要求", max_chars=120, min_chars=40)
        assert "0.8 μm" in compressed
        assert len(compressed) < len(text)

    def test_short_text_untouched(self):
        text = "镀层厚度不小于 0.8 μm。"
        assert _compress_text(text, "镀层厚度", min_chars=200) == text

    def test_empty_text(self):
        assert _compress_text("", "查询") == ""

    def test_preserves_original_order(self):
        text = "甲：镀金层厚度要求。乙：无关内容段落。丙：镀金层测试方法。"
        compressed = _compress_text(text, "镀金层", max_chars=60, min_chars=10)
        assert compressed.index("镀金层厚度") < compressed.index("镀金层测试") if "镀金层测试" in compressed else True


class TestSimilarityAndDedupe:
    def test_identical_texts(self):
        assert similarity("镀金层厚度 0.8 μm", "镀金层厚度 0.8 μm") == 1.0

    def test_unrelated_texts(self):
        assert similarity("镀金层厚度要求", "阻焊层丝印颜色规定") < 0.3

    def test_near_duplicate_detected(self):
        a = "镀金层厚度应不小于 0.8 μm，测试方法按 GB/T 4677 执行。"
        b = "镀金层厚度应不小于 0.8 μm，测试方法按 GB/T 4677 执行！"
        assert similarity(a, b) > 0.85

    def test_shingles_of_short_text(self):
        assert _shingles("短") == {"短"}

    def test_dedupe_removes_near_duplicate(self):
        # 文本需超过 DEDUPE_MIN_CHARS（默认 80），否则按「过短不比较」直接保留
        text = (
            "镀金层厚度应不小于 0.8 μm，测试方法按 GB/T 4677 执行，取样位置为板面中心区域。"
            "试样应在 260 ℃ 下保持 10 s 后自然冷却，任何分层、起泡或变色均判定为不合格，"
            "复检规则与判定依据由供需双方在质量协议中另行约定。"
        )
        nodes = [
            _node(text, "a"),
            _node(text + "！", "b"),
            _node("阻焊层颜色应为绿色或黑色，具体由供需双方约定并记录于装配图纸中。", "c"),
        ]
        kept = _dedupe_nodes(nodes)
        assert len(kept) == 2
        assert kept[0].node_id == "a"
        assert kept[1].node_id == "c"

    def test_dedupe_keeps_short_nodes(self):
        nodes = [_node("短文本", "a"), _node("短文本", "b")]
        assert len(_dedupe_nodes(nodes)) == 2

    def test_dedupe_single_node(self):
        nodes = [_node("唯一内容", "a")]
        assert _dedupe_nodes(nodes) == nodes


class TestNodeLevelCompression:
    def test_compress_nodes_marks_metadata(self, monkeypatch):
        import pcb_rag.compression as compression

        monkeypatch.setattr(compression, "COMPRESSION_ENABLED", True)
        text = "".join([f"第{i}条 本文件规定了通用要求。" for i in range(10)])
        text += "镀金层厚度应不小于 0.8 μm，按 GB/T 4677 执行。"
        nodes = [_node(text, "a")]

        result = compression.compress_nodes(nodes, "镀金层厚度", max_chars=80)
        assert len(result) == 1
        assert len(result[0].text) < len(text)
        assert result[0].metadata.get("compressed") is True

    def test_compress_disabled_returns_input(self):
        import pcb_rag.compression as compression

        nodes = [_node("内容" * 200, "a")]
        assert compression.compress_nodes(nodes, "内容")[0].text == nodes[0].text

    def test_budget_truncates_tail(self, monkeypatch):
        import pcb_rag.compression as compression

        nodes = [_node("甲" * 100, "a"), _node("乙" * 100, "b"), _node("丙" * 100, "c")]
        kept = compression.apply_context_budget(nodes, budget_chars=150)
        assert len(kept) == 1
        assert kept[0].node_id == "a"

    def test_summarize_reports_ratio(self, monkeypatch):
        import pcb_rag.compression as compression

        monkeypatch.setattr(compression, "COMPRESSION_ENABLED", True)
        text = "".join([f"第{i}条 通用要求。" for i in range(12)]) + "镀金层厚度应不小于 0.8 μm。"
        nodes = compression.compress_nodes([_node(text, "a")], "镀金层厚度", max_chars=60)
        summary = compression.summarize_compression(nodes)
        assert summary["nodes"] == 1
        assert summary["ratio"] <= 1.0
