"""chunk 信息密度评估的单元测试。

样例取自真实语料：``GBT43863-2024`` 的 8.2.5.3 节是一个典型的
「裸标签罗列块」，而 ``GBT2036-1994`` 的 5.2.6 节是正常的技术定义段。
判定必须能把前者挑出来、又不误伤后者。
"""

from pcb_rag.chunk_quality import assess_chunk_quality, annotate_metadata


BARE_TAG_CHUNK = """8.2.5.3 语句组成

<socket>语句包含以下内容：

<default>
<port>
<portgroup>
<powerdomain_group>
<swappable_port>
<swappable_group>
<port_assignement>
<frequency>
<constraint>
<extensions>

8.2.5.4 default语句"""

NORMAL_CHUNK = (
    "5.2.6 照相缩小尺寸 photographic reduction dimension\n"
    "照相底图上给照相操作者指示照相原版缩小程度的尺寸。例如两条直线之间或两个规定点"
    "之间的距离。该尺寸值是 1:1 比例的精确数值，并且必须在照相底图上标注。"
)

SPEC_CHUNK = (
    "6.3 镀金层要求\n"
    "镀金层厚度应不小于 0.05 μm，镍层厚度应为 3~6 μm。"
    "金层纯度必须达到 99.9%。测试方法按 GB/T 4677-2002 执行。"
)


class TestAssessChunkQuality:
    def test_bare_tag_list_is_low_value(self):
        result = assess_chunk_quality(BARE_TAG_CHUNK)
        assert result["is_low_value"] is True
        assert "bare_tag_list" in result["flags"]

    def test_normal_paragraph_is_not_low_value(self):
        result = assess_chunk_quality(NORMAL_CHUNK)
        assert result["is_low_value"] is False
        assert result["flags"] == []

    def test_spec_chunk_is_high_value(self):
        result = assess_chunk_quality(SPEC_CHUNK)
        assert result["is_low_value"] is False
        assert result["score"] >= 0.7

    def test_empty_text(self):
        result = assess_chunk_quality("")
        assert result["is_low_value"] is True
        assert "empty" in result["flags"]

    def test_too_short(self):
        result = assess_chunk_quality("见图 3。")
        assert "too_short" in result["flags"]

    def test_repetitive(self):
        result = assess_chunk_quality("—" * 200)
        assert "repetitive" in result["flags"]

    def test_table_rule_only(self):
        text = "|---|---|---|\n|:--|:--:|--:|\n" * 2 + "表 1 参数"
        result = assess_chunk_quality(text)
        assert "table_rule_only" in result["flags"]

    def test_metrics_exposed_for_tuning(self):
        result = assess_chunk_quality(BARE_TAG_CHUNK)
        metrics = result["metrics"]
        assert metrics["bare_tag_lines"] >= 3
        assert 0.0 <= metrics["density"] <= 1.0
        assert metrics["chars"] > 0

    def test_score_within_range(self):
        for text in (BARE_TAG_CHUNK, NORMAL_CHUNK, "", "—" * 200):
            score = assess_chunk_quality(text)["score"]
            assert 0.0 <= score <= 1.0


class TestAnnotateMetadata:
    def test_low_value_gets_marked(self):
        md = {}
        out = annotate_metadata(md, BARE_TAG_CHUNK)
        assert out["low_value"] is True
        assert "chunk_quality" in out
        assert "bare_tag_list" in out["quality_flags"]

    def test_high_value_gets_no_extra_keys(self):
        md = {"source_path": "a.txt"}
        out = annotate_metadata(md, NORMAL_CHUNK)
        assert out == {"source_path": "a.txt"}
