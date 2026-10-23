"""入库元数据的单元测试。

重点覆盖 ``_infer_source_type`` 的文件名兜底 —— 语料常以平铺方式放在
``clear_docs/`` 下，路径里没有「标准/国家标准」这类层级，只靠目录判断会
把全部文档归为 unknown，进而让元数据过滤与 ``by_source_type`` 统计同时失效。

同时锁住一个已修复的坑：调用方传入的 rel_norm 末尾带斜杠，
直接 ``rsplit("/")[-1]`` 会取到空串，兜底逻辑静默失效。
"""

import pytest

pytest.importorskip("llama_index.core")

from pcb_rag.ingest import (  # noqa: E402
    _infer_source_type,
    enhance_chunk_metadata,
)


class TestInferSourceTypeByDirectory:
    def test_national_standard_dir(self):
        assert _infer_source_type("/标准/国家标准/x.txt/") == "standard_national"

    def test_international_standard_dir(self):
        assert _infer_source_type("/标准/国际标准/x.txt/") == "standard_international"

    def test_directory_takes_priority_over_filename(self):
        """显式目录分类不应被文件名规则覆盖。"""
        assert _infer_source_type("/文献/GBT2036-1994.txt/") == "literature"


class TestInferSourceTypeByFilename:
    def test_gb_prefix(self):
        assert _infer_source_type("/GBT2036-1994印制电路术语.txt/") == "standard_national"

    def test_gb_slash_t(self):
        assert _infer_source_type("/GB/T 4677-2002 印制板测试方法.txt/") == "standard_national"

    def test_ipc_prefix(self):
        assert _infer_source_type("/IPC-2581C.txt/") == "standard_international"

    def test_sj_prefix(self):
        assert _infer_source_type("/SJT 11042-1996.txt/") == "standard_industry"

    def test_iso_prefix(self):
        assert _infer_source_type("/ISO 9001.txt/") == "standard_international"

    def test_eda_tool_name(self):
        assert _infer_source_type("/Altium Designer 实用教程.txt/") == "pcb_design"
        assert _infer_source_type("/Cadence Allegro 布线指南.txt/") == "pcb_design"

    def test_material_keyword(self):
        assert _infer_source_type("/覆铜板材料手册.txt/") == "pcb_material"

    def test_unknown_when_nothing_matches(self):
        assert _infer_source_type("/随便一个文档.txt/") == "unknown"

    def test_trailing_slash_does_not_break_fallback(self):
        """回归：rel_norm 末尾带斜杠时，文件名兜底必须仍然生效。"""
        with_slash = _infer_source_type("/GBT2036-1994印制电路术语.txt/")
        without_slash = _infer_source_type("/GBT2036-1994印制电路术语.txt")
        assert with_slash == "standard_national"
        assert with_slash == without_slash

    def test_empty_input(self):
        assert _infer_source_type("") == "unknown"


class TestEnhanceChunkMetadata:
    def test_fills_source_type_when_missing(self):
        meta = enhance_chunk_metadata(
            chunk_text="镀金层厚度应不小于 0.05 μm。",
            chunk_meta={},
            doc_text="GB/T 4677-2002 印制板测试方法",
            file_path="data/clear_docs/GBT4677-2002印制板测试方法.txt",
        )
        assert meta["source_type"] == "standard_national"

    def test_fills_source_type_when_unknown(self):
        meta = enhance_chunk_metadata(
            chunk_text="镀金层厚度应不小于 0.05 μm。",
            chunk_meta={"source_type": "unknown"},
            doc_text="",
            file_path="data/clear_docs/IPC-2581C.txt",
        )
        assert meta["source_type"] == "standard_international"

    def test_keeps_explicit_source_type(self):
        meta = enhance_chunk_metadata(
            chunk_text="任意内容",
            chunk_meta={"source_type": "literature"},
            doc_text="",
            file_path="data/clear_docs/GBT2036-1994.txt",
        )
        assert meta["source_type"] == "literature"

    def test_doc_type_is_filled(self):
        meta = enhance_chunk_metadata(
            chunk_text="依据 GB/T 4677 执行。",
            chunk_meta={},
            doc_text="按 GB/T 4677 检测",
            file_path="a.txt",
        )
        assert meta["doc_type"] == "national_standard"

    def test_existing_section_not_overwritten(self):
        """结构解析给出的 parent_title 应优先于正文首行猜测。"""
        meta = enhance_chunk_metadata(
            chunk_text="5.2.6 照相缩小尺寸\n正文内容。",
            chunk_meta={"section": "3.1.16 预制内层覆铜板"},
            doc_text="",
            file_path="a.txt",
        )
        assert meta["section"] == "3.1.16 预制内层覆铜板"
