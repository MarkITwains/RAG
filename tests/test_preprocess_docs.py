"""``pcb_rag.preprocess_docs`` 编码修复与乱码清理单元测试。

该模块只依赖标准库（chardet 为可选依赖），测试因此可以在 CI 中完全离线运行。
"""

import pytest

from pcb_rag import preprocess_docs
from pcb_rag.preprocess_docs import (
    add_section_markers,
    calculate_chinese_ratio,
    calculate_garbage_ratio,
    calculate_mojibake_ratio,
    calculate_replacement_ratio,
    clean_garbage_chars,
    decode_file_content,
    detect_encoding,
    detect_encoding_simple,
    identify_sections,
    is_valid_char,
    print_summary,
    process_directory,
    process_file,
    score_text_quality,
    should_try_mojibake_repair,
    try_mojibake_repair,
)

GBK_SAMPLE = "中华人民共和国国家标准印制电路板设计要求与试验方法说明。"


class TestDetectEncodingSimple:
    def test_utf8_bom(self):
        assert detect_encoding_simple(b"\xef\xbb\xbfhello") == "utf-8-sig"

    def test_utf16_bom(self):
        assert detect_encoding_simple(b"\xff\xfe" + "中文".encode("utf-16-le")) == "utf-16-le"
        assert detect_encoding_simple(b"\xfe\xff" + "中文".encode("utf-16-be")) == "utf-16-be"

    def test_plain_utf8(self):
        assert detect_encoding_simple("PCB 阻抗控制".encode()) == "utf-8"

    def test_gbk_chinese(self):
        assert detect_encoding_simple(GBK_SAMPLE.encode("gbk")) == "gbk"


class TestDetectEncoding:
    def test_utf8_chinese_is_detected(self):
        encoding, confidence = detect_encoding(GBK_SAMPLE.encode("utf-8"))
        assert encoding in ("utf-8", "utf-8-sig")
        assert confidence > 0.0


class TestDecodeFileContent:
    def test_reads_gbk_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preprocess_docs, "HAS_CHARDET", False)
        path = tmp_path / "gbk.txt"
        path.write_bytes((GBK_SAMPLE * 3).encode("gbk"))

        text, encoding = decode_file_content(str(path))

        assert "印制电路板" in text
        assert encoding in ("gbk", "gb2312", "gb18030")


class TestIsValidChar:
    def test_ascii_and_cjk_are_valid(self):
        assert is_valid_char("A") is True
        assert is_valid_char("中") is True

    def test_private_use_area_is_invalid(self):
        assert is_valid_char("\ue000") is False

    def test_latin_extended_a_is_invalid(self):
        assert is_valid_char("\u0100") is False


class TestRatioHelpers:
    def test_garbage_ratio_of_clean_text(self):
        assert calculate_garbage_ratio("中文文本") == 0.0

    def test_garbage_ratio_counts_invalid_and_replacement_chars(self):
        assert calculate_garbage_ratio("中文\ue000") == pytest.approx(1 / 3)
        assert calculate_garbage_ratio("\ufffd") == pytest.approx(1.0)

    def test_garbage_ratio_of_empty_text(self):
        assert calculate_garbage_ratio("") == 0.0

    def test_chinese_ratio(self):
        assert calculate_chinese_ratio("中文ab") == pytest.approx(0.5)
        assert calculate_chinese_ratio("") == 0.0

    def test_replacement_ratio(self):
        assert calculate_replacement_ratio("a\ufffdb") == pytest.approx(1 / 3)
        assert calculate_replacement_ratio("") == 0.0

    def test_mojibake_ratio(self):
        assert calculate_mojibake_ratio("\u00c0\u00c1") == pytest.approx(1.0)
        assert calculate_mojibake_ratio("") == 0.0


class TestCleanGarbageChars:
    def test_removes_control_and_replacement_chars(self):
        assert clean_garbage_chars("a\x00b\ufffdc\x0b") == "abc"

    def test_removes_invalid_unicode_chars(self):
        assert clean_garbage_chars("正常\ue000文本") == "正常文本"

    def test_collapses_spaces_and_removes_blank_lines(self):
        assert clean_garbage_chars("a   b") == "a b"
        # 连续换行先折叠为两个，随后被 MULTILINE 的 ^\s+ 进一步移除，最终只剩一个换行
        assert clean_garbage_chars("a\n\n\n\nb") == "a\nb"

    def test_keeps_chinese_content(self):
        assert clean_garbage_chars("  PCB 阻抗控制 ") == "PCB 阻抗控制"

    def test_empty_input(self):
        assert clean_garbage_chars("") == ""


class TestSectionIdentification:
    def test_identifies_chapter_section_appendix_and_table(self):
        text = "\n".join(["1 范围", "1.1 一般要求", "附录A 补充说明", "表1 参数表"])
        found = {(number, kind) for _, _, number, _, kind in identify_sections(text)}
        assert ("1", "chapter") in found
        assert ("1.1", "section") in found
        assert ("附录A", "appendix") in found
        assert ("表1", "table") in found

    def test_plain_text_yields_no_sections(self):
        assert identify_sections("这是一段普通说明文字，没有章节编号。") == []

    def test_add_section_markers_injects_marker(self):
        marked = add_section_markers("1 范围\n正文内容")
        assert "<!-- SECTION: chapter | 1 | 范围 -->" in marked


class TestTextQualityScore:
    def test_clean_chinese_scores_better_than_garbage(self):
        clean = "印制电路板的阻抗控制需要综合考虑线宽、介质厚度与铜箔厚度等参数。"
        garbage = "\ue000\ue000\ue000\ufffd\ufffd印制"
        assert score_text_quality(clean) < score_text_quality(garbage)

    def test_long_ascii_text_gets_penalty(self):
        assert score_text_quality("a" * 60) >= 0.05


class TestMojibakeRepair:
    def test_short_text_is_skipped(self):
        assert should_try_mojibake_repair("短文本") is False
        assert try_mojibake_repair("短文本") is None

    def test_replacement_chars_trigger_repair_attempt(self):
        assert should_try_mojibake_repair("印制电路板" * 20 + "\ufffd") is True

    def test_clean_long_text_is_not_repaired(self):
        text = "印制电路板阻抗控制需要综合考虑线宽与介质厚度。" * 5
        assert should_try_mojibake_repair(text) is False
        assert try_mojibake_repair(text) is None

    def test_mojibake_text_can_be_repaired_back(self):
        # 模拟 UTF-8 字节被按 latin-1 解码产生的典型乱码
        original = "印制电路板阻抗控制需要综合考虑线宽与介质厚度等参数。" * 3
        mojibake = original.encode("utf-8").decode("latin-1")

        assert should_try_mojibake_repair(mojibake) is True
        repaired = try_mojibake_repair(mojibake)
        assert repaired is not None
        text, method, _ = repaired
        assert method.startswith("mojibake:")
        assert text == original


class TestProcessFile:
    def test_clean_utf8_file_is_skipped(self, tmp_path):
        source = tmp_path / "clean.txt"
        source.write_text(GBK_SAMPLE * 3, encoding="utf-8")
        target = tmp_path / "out" / "clean.txt"

        result = process_file(str(source), str(target), dry_run=True)

        assert result["status"] == "skip"
        assert result["original_encoding"] in ("utf-8", "utf-8-sig")
        assert not target.exists()

    def test_gbk_file_is_converted_to_utf8(self, tmp_path, monkeypatch):
        # 固定为简化编码检测分支，避免 chardet 版本差异导致断言不稳定
        monkeypatch.setattr(preprocess_docs, "HAS_CHARDET", False)
        source = tmp_path / "gbk.txt"
        source.write_bytes((GBK_SAMPLE * 3).encode("gbk"))
        target = tmp_path / "out" / "gbk.txt"

        result = process_file(str(source), str(target), dry_run=False)

        assert result["status"] == "success"
        assert result["original_encoding"] in ("gbk", "gb2312", "gb18030")
        assert target.exists()
        assert "印制电路板" in target.read_text(encoding="utf-8")

    def test_empty_file_is_handled(self, tmp_path):
        source = tmp_path / "empty.txt"
        source.write_text("", encoding="utf-8")

        result = process_file(str(source), str(tmp_path / "out" / "empty.txt"), dry_run=True)

        assert result["status"] in ("skip", "success")
        assert result["original_encoding"] is not None


class TestProcessDirectory:
    def test_processes_supported_extensions_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preprocess_docs, "HAS_CHARDET", False)
        source_dir = tmp_path / "in"
        source_dir.mkdir()
        (source_dir / "doc.txt").write_bytes((GBK_SAMPLE * 3).encode("gbk"))
        (source_dir / "binary.bin").write_bytes(b"\x00\x01\x02")

        out_dir = tmp_path / "out"
        results = process_directory(str(source_dir), str(out_dir), dry_run=False)

        assert len(results) == 1
        assert (out_dir / "doc.txt").exists()
        assert "印制电路板" in (out_dir / "doc.txt").read_text(encoding="utf-8")


class TestPrintSummary:
    def test_prints_overview(self, capsys):
        print_summary(
            [
                {
                    "status": "success",
                    "original_encoding": "gbk",
                    "garbage_ratio_before": 0.2,
                    "input": "a.txt",
                }
            ]
        )
        output = capsys.readouterr().out
        assert "处理摘要" in output
        assert "gbk" in output
