"""CI 依赖自检：确保关键回归测试不会「静默跳过」。

背景
----
`tests/test_security.py::TestMergeFilters` 是 ACL 合并 bug 的回归测试，它用
``pytest.importorskip("llama_index.core")`` 守卫。如果 CI 不装 llama-index，
这些用例会被 **skip** —— 而 skipped 不会让流水线变红。于是"260 passed" 里
可能掺着"其实没跑"，制造虚假安全感。

本文件把这件事变成显式信号：
- 设置 ``REQUIRE_LLAMA_INDEX=1``（CI 里设置）时，缺依赖 = 失败；
- 未设置时（本地开发环境）自动跳过，不干扰日常开发。
"""

import ast
import importlib
import os

import pytest

REQUIRE = os.getenv("REQUIRE_LLAMA_INDEX", "") not in {"", "0", "false", "False"}

#: 关键回归测试所需的模块 —— 缺任何一个都会让对应测试被静默跳过
REQUIRED_FOR_REGRESSION = [
    "llama_index.core",
    "llama_index.core.vector_stores.types",
]


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not REQUIRE, reason="仅 CI 环境校验（设置 REQUIRE_LLAMA_INDEX=1 启用）")
class TestRegressionDependencies:
    def test_llama_index_core_is_installed(self):
        for name in REQUIRED_FOR_REGRESSION:
            assert _importable(name), (
                f"{name} 缺失 → test_security.py::TestMergeFilters 会被 importorskip "
                "静默跳过。请安装 `pip install -e '.[dev,test_llama]'`。"
            )

    def test_merge_filters_does_not_skip(self):
        """真正验证：TestMergeFilters 的 fixture 能解析出 llama-index 类型。"""
        pytest.importorskip("llama_index.core")
        from llama_index.core.vector_stores.types import MetadataFilters  # noqa: F401


HEAVY_PREFIXES = ("torch", "llama_index", "pymilvus", "transformers", "sentence_transformers")


def _toplevel_imports(source: str) -> list:
    """只取**顶层** import；函数内的惰性 import 不算（那是刻意的轻量引用策略）。"""
    tree = ast.parse(source)
    names: list = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_pure_logic_modules_have_no_heavy_dependency():
    """反向约束：纯逻辑模块的**顶层**不允许引入重型依赖。

    否则这些模块只能靠 pytest.importorskip 度日 —— 而 skip 不会让 CI 变红，
    等于没测。它们的测试恰恰是本仓库里最该稳定跑起来的那批。
    """
    for name in ("pcb_rag.fusion", "pcb_rag.incremental", "pcb_rag.cache", "eval.metrics"):
        module = importlib.import_module(name)
        with open(module.__file__, "r", encoding="utf-8") as fh:
            imports = _toplevel_imports(fh.read())
        offenders = [imp for imp in imports if imp.split(".")[0] in HEAVY_PREFIXES]
        assert not offenders, f"{name} 顶层不应依赖 {offenders}（可改为函数内惰性导入）"
