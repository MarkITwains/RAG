"""Secret-free dotenv coverage checks shared by diagnostics and tests."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from pcb_rag.env_loader import _unquote

LEGACY_KEYS = {
    "FUSION_MODE": "only a compatibility/display setting; it does not select the fusion algorithm",
    "CHUNK_GROUP_BY_DOC": "legacy context grouping is not used by the shared evidence pipeline",
    "GRAPH_EXPAND_MAX": "only the legacy _graph_augment path; not used by the shared pipeline",
}
DYNAMIC_KEYS = {
    "RAG_LLM_MAX_INPUT_TOKENS",
    "RAG_EMBEDDING_MAX_INPUT_TOKENS",
    "RAG_RERANK_MAX_INPUT_TOKENS",
}


def env_file_values(path):
    values = {}
    duplicates = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if key in values:
            duplicates.append(key)
        values[key] = _unquote(value)
    return values, duplicates


def known_environment_keys(root):
    keys = set(DYNAMIC_KEYS) | set(LEGACY_KEYS)
    for folder in ("src", "scripts", "eval"):
        for path in (root / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if re.fullmatch(r"[A-Z][A-Z0-9_]*", first.value):
                        keys.add(first.value)
    return keys


def audit_dotenv(path, *, root, environ=None):
    environ = os.environ if environ is None else environ
    values, duplicates = env_file_values(path)
    known = known_environment_keys(root)
    unknown = sorted(set(values) - known)
    legacy = sorted(set(values) & set(LEGACY_KEYS))
    overridden = sorted(key for key in values if key in environ and environ[key] != values[key])
    return {
        "entries": len(values),
        "duplicates": duplicates,
        "unknown": unknown,
        "legacy": legacy,
        "overridden": overridden,
        "known_keys": len(known),
    }
