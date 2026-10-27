import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pcb_rag.config import env_bool
from pcb_rag.env_loader import _unquote


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("'quoted value' # comment", "quoted value"),
        ('"quoted value" # comment', "quoted value"),
        ("plain # comment", "plain"),
        ("'https://example.invalid/v1'", "https://example.invalid/v1"),
    ],
)
def test_dotenv_unquote_handles_comments_and_quotes(raw, expected):
    assert _unquote(raw) == expected


@pytest.mark.parametrize("value", ["0", "false", "FALSE", " False ", "no", "OFF", "off"])
def test_boolean_false_spellings_are_consistent(value, monkeypatch):
    monkeypatch.setenv("TEST_SWITCH", value)
    assert env_bool("TEST_SWITCH", True) is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "ON", "on"])
def test_boolean_true_spellings_are_consistent(value, monkeypatch):
    monkeypatch.setenv("TEST_SWITCH", value)
    assert env_bool("TEST_SWITCH", False) is True


def test_missing_or_empty_boolean_uses_default(monkeypatch):
    monkeypatch.delenv("TEST_SWITCH", raising=False)
    assert env_bool("TEST_SWITCH", True) is True
    monkeypatch.setenv("TEST_SWITCH", "")
    assert env_bool("TEST_SWITCH", False) is False


def test_startup_boolean_values_reach_shared_pipeline():
    env = dict(
        os.environ,
        PCB_RAG_LOAD_DOTENV="0",
        CHUNK_EXPAND_ENABLED="OFF",
        HYDE_ENABLED="FALSE",
        LEXICAL_ENABLED="off",
        SEMANTIC_CACHE_ENABLED="False",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    script = """
import json
from pcb_rag import cache, query
from pcb_rag.retrieval import PipelineConfig
config = PipelineConfig.from_env()
print(json.dumps([
    query.CHUNK_EXPAND_ENABLED,
    query.HYDE_ENABLED,
    query.LEXICAL_ENABLED,
    cache.SEMANTIC_CACHE_ENABLED,
    config.expand,
    config.lexical_enabled,
]))
"""

    result = subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)

    assert json.loads(result.stdout) == [False] * 6


def test_direct_submodule_import_loads_project_env_before_constants(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RERANK_API_URL=https://example.invalid/rerank\nGRAPH_RAG_ENABLED=1\n", encoding="utf-8")
    package = tmp_path / "src" / "pcb_rag"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(Path("src/pcb_rag/__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
    (package / "env_loader.py").write_text(
        Path("src/pcb_rag/env_loader.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (package / "api_clients.py").write_text(
        "RERANK_API_URL = __import__('os').getenv('RERANK_API_URL')\n", encoding="utf-8"
    )
    (package / "graph_rag.py").write_text(
        "GRAPH_RAG_ENABLED = __import__('os').getenv('GRAPH_RAG_ENABLED') == '1'\n", encoding="utf-8"
    )
    script = "from pcb_rag.api_clients import RERANK_API_URL; from pcb_rag.graph_rag import GRAPH_RAG_ENABLED; print(RERANK_API_URL, GRAPH_RAG_ENABLED)"
    isolated_path = (tmp_path / "src").as_posix()
    isolated_script = f"import sys; sys.path.insert(0, {isolated_path!r}); {script}"
    isolated_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "RERANK_API_URL", "GRAPH_RAG_ENABLED"}
    }
    isolated_env["PCB_RAG_LOAD_DOTENV"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", isolated_script],
        cwd=tmp_path,
        env=isolated_env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "https://example.invalid/rerank True"
