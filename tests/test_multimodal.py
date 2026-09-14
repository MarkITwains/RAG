"""多模态模块单元测试。

覆盖：表格结构化成 Markdown、列切分、图片抽取的容错、文档增强的降级路径。
不发起真实 VLM 调用（未配置时 `describe_image` 直接返回空串）。
"""

from types import SimpleNamespace

import pcb_rag.multimodal as multimodal
from pcb_rag.multimodal import (
    _looks_like_table_row,
    _split_columns,
    _to_markdown_table,
    augment_documents_with_multimodal,
    describe_image,
    describe_multimodal,
    extract_pdf_images,
    extract_pdf_tables,
    is_vision_configured,
    text_tables_to_markdown,
)


class TestSplitColumns:
    def test_splits_on_multiple_spaces(self):
        assert _split_columns("镀层厚度    ≥0.8 μm    GB/T 4677") == ["镀层厚度", "≥0.8 μm", "GB/T 4677"]

    def test_splits_on_tabs(self):
        assert _split_columns("项目\t要求\t方法") == ["项目", "要求", "方法"]

    def test_splits_on_pipes(self):
        assert _split_columns("| 项目 | 要求 |") == ["项目", "要求"]

    def test_plain_sentence_returns_empty(self):
        assert _split_columns("本文件规定了印制板的通用要求。") == []
        assert _split_columns("") == []
        assert _split_columns("   ") == []


class TestLooksLikeTableRow:
    def test_numeric_row_is_table(self):
        assert _looks_like_table_row(["镀层厚度", "0.8", "μm"]) is True

    def test_short_text_columns_are_table(self):
        assert _looks_like_table_row(["项目", "要求", "方法"]) is True

    def test_single_column_is_not_table(self):
        assert _looks_like_table_row(["只有一列"]) is False

    def test_long_text_row_is_not_table(self):
        row = ["这是一段很长的说明文字，用来确保不会被误判为表格的一个单元格"] * 2
        assert _looks_like_table_row(row) is False


class TestMarkdownTable:
    def test_renders_header_and_separator(self):
        markdown = _to_markdown_table([["项目", "要求"], ["厚度", "0.8"]])
        lines = markdown.split("\n")
        assert lines[0] == "| 项目 | 要求 |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 厚度 | 0.8 |"

    def test_pads_ragged_rows(self):
        markdown = _to_markdown_table([["a", "b", "c"], ["d"]])
        body_row = markdown.split("\n")[2]
        assert body_row.startswith("| d |")
        # 3 列的行应有 4 个竖线分隔符
        assert body_row.count("|") == 4

    def test_escapes_pipe(self):
        markdown = _to_markdown_table([["a|b", "c"]])
        assert "a\\|b" in markdown

    def test_empty_input(self):
        assert _to_markdown_table([]) == ""


class TestTextTablesToMarkdown:
    def test_converts_aligned_block(self):
        raw = "项目        要求        试验方法\n镀层厚度    ≥0.8 μm     GB/T 4677\n附着力      无脱落      IPC-TM-650"
        converted = text_tables_to_markdown(raw)
        assert "| 项目 | 要求 | 试验方法 |" in converted
        assert "| 镀层厚度 | ≥0.8 μm | GB/T 4677 |" in converted

    def test_keeps_plain_text_untouched(self):
        raw = "本文件规定了印制板镀金层的技术要求。\n试验方法按 GB/T 4677 执行。"
        assert text_tables_to_markdown(raw) == raw

    def test_single_row_block_is_not_converted(self):
        raw = "项目        要求\n正文继续说明其它内容。"
        assert "| --- |" not in text_tables_to_markdown(raw)

    def test_mixed_content_preserved(self):
        raw = "引言段落。\n项目    数值\n厚度    0.8\n结语段落。"
        converted = text_tables_to_markdown(raw)
        assert "引言段落。" in converted
        assert "结语段落。" in converted
        assert "| 厚度 | 0.8 |" in converted

    def test_empty_input(self):
        assert text_tables_to_markdown("") == ""


class TestImageExtraction:
    def test_missing_file_returns_empty(self, tmp_path):
        assert extract_pdf_images(str(tmp_path / "nope.pdf")) == []

    def test_non_pdf_returns_empty(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("plain text", encoding="utf-8")
        assert extract_pdf_images(str(path)) == []

    def test_pdf_tables_missing_file(self, tmp_path):
        assert extract_pdf_tables(str(tmp_path / "nope.pdf")) == []


class TestDescribeImage:
    def test_returns_empty_when_disabled(self, monkeypatch):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", False)
        assert describe_image(b"fake-bytes") == ""

    def test_returns_empty_without_bytes(self):
        assert describe_image(b"") == ""

    def test_vision_not_configured_by_default(self, monkeypatch):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", False)
        monkeypatch.setattr(multimodal, "MULTIMODAL_VLM_BASE_URL", "")
        monkeypatch.setattr(multimodal, "MULTIMODAL_VLM_MODEL", "")
        assert is_vision_configured() is False


class TestAugmentDocuments:
    def test_tables_converted_when_multimodal_disabled(self, monkeypatch):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", False)
        monkeypatch.setattr(multimodal, "TABLE_MARKDOWN_ENABLED", True)

        text = "项目        要求\n镀层厚度    ≥0.8 μm\n附着力      无脱落"
        docs = [SimpleNamespace(text=text, metadata={"source_path": "a.pdf"})]

        result = augment_documents_with_multimodal(docs)
        assert len(result) == 1
        assert "| 镀层厚度 | ≥0.8 μm |" in result[0].text
        assert result[0].metadata.get("has_markdown_table") is True

    def test_disabled_tables_keep_text(self, monkeypatch):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", False)
        monkeypatch.setattr(multimodal, "TABLE_MARKDOWN_ENABLED", False)

        text = "项目        要求\n镀层厚度    ≥0.8 μm"
        docs = [SimpleNamespace(text=text, metadata={})]
        result = augment_documents_with_multimodal(docs)
        assert result[0].text == text

    def test_non_pdf_does_not_extract_images(self, monkeypatch, tmp_path):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", True)
        monkeypatch.setattr(multimodal, "TABLE_MARKDOWN_ENABLED", False)

        txt = tmp_path / "a.txt"
        txt.write_text("纯文本", encoding="utf-8")
        docs = [SimpleNamespace(text="纯文本", metadata={"source_path": str(txt)})]
        result = augment_documents_with_multimodal(docs)
        assert len(result) == 1

    def test_empty_input(self):
        assert augment_documents_with_multimodal([]) == []
        assert augment_documents_with_multimodal(None) == []

    def test_missing_pdf_file_is_tolerated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(multimodal, "MULTIMODAL_ENABLED", True)
        monkeypatch.setattr(multimodal, "TABLE_MARKDOWN_ENABLED", False)
        docs = [SimpleNamespace(text="x", metadata={"source_path": str(tmp_path / "gone.pdf")})]
        result = augment_documents_with_multimodal(docs)
        assert len(result) == 1


class TestDescribeMultimodal:
    def test_reports_configuration(self):
        info = describe_multimodal()
        assert "enabled" in info
        assert "table_markdown" in info
        assert "vision_configured" in info
