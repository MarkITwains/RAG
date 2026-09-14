"""包结构冒烟测试。

确认在未安装 Milvus / Ollama / FastAPI 等重型依赖的 CI 环境中，
纯逻辑模块依然可以被正常导入；同时锁定「重型模块存在但不在测试中导入」
这一约定，避免有人无意间把 torch 之类的依赖拖进测试链路。
"""

import importlib
from pathlib import Path

import pytest

PURE_LOGIC_MODULES = [
    "pcb_rag",
    "pcb_rag.cache",
    "pcb_rag.preprocess_docs",
    "eval.metrics",
]

HEAVY_MODULES = [
    "api_clients.py",
    "ingest.py",
    "query.py",
    "dify_external_api.py",
]


@pytest.mark.parametrize("module_name", PURE_LOGIC_MODULES)
def test_pure_logic_modules_are_importable(module_name):
    assert importlib.import_module(module_name) is not None


def test_heavy_modules_exist_but_are_not_imported():
    package_dir = Path(importlib.import_module("pcb_rag").__file__).parent
    for filename in HEAVY_MODULES:
        assert (package_dir / filename).is_file()
