"""Processing fingerprints, atomic published snapshots and resumable stage state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


STAGE_VERSIONS = {"chunks": 5, "vectors": 2, "lexical": 3, "graph": 5}
STAGE_KEYS = {
    "chunks": (
        "NODE_PARSER_MODE",
        "PARENT_MAX_SIZE",
        "PARENT_MIN_SIZE",
        "PARENT_MIN_SECTION_KEEP",
        "CHILD_MAX_SIZE",
        "CHILD_MIN_SIZE",
        "CHILD_OVERLAP",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "CHUNK_HIERARCHY_ENABLED",
        "HIERARCHICAL_CHUNK_SIZES",
        "STRUCTURE_MAX_CHUNK_SIZE",
        "STRUCTURE_MIN_CHUNK_SIZE",
        "STRUCTURE_OVERLAP",
        "SEMANTIC_BUFFER_SIZE",
        "SEMANTIC_BREAKPOINT_THRESHOLD",
        "SEMANTIC_CHUNK_SIZE",
        "ATTENTION_MODEL",
        "ATTENTION_USE_OLLAMA",
        "ATTENTION_MIN_CHUNK_SIZE",
        "ATTENTION_MAX_CHUNK_SIZE",
        "ATTENTION_THRESHOLD",
        "ATTENTION_BREAKPOINT_PERCENTILE",
        "ATTENTION_OVERLAP",
        "ATTENTION_WINDOW_SIZE",
        "CHUNK_EMBEDDING_PROFILE",
        "NORMALIZE_TEXT",
        "AUTO_FIX_ENCODING",
        "STRUCTURED_FLATTEN",
        "STRUCTURED_FLATTEN_MIN_RUN",
        "ANCHOR_FIGURE_TABLE",
        "ANCHOR_FIGURE_TABLE_LINES",
        "ANCHOR_FIGURE_TABLE_MAX_CHARS",
        "NORMALIZE_TEXT",
        "CLEAN_OCR_GARBAGE",
        "ACL_ENABLED",
        "ACL_TENANT_FIELD",
        "ACL_GROUPS_FIELD",
        "ACL_VISIBILITY_FIELD",
        "ACL_OWNER_FIELD",
        "ACL_DEFAULT_TENANT",
        "ACL_ANONYMOUS_TENANT",
        "ACL_DEFAULT_VISIBILITY",
        "ACL_DEFAULT_OWNER",
        "ACL_PUBLIC_VALUE",
        "CONTEXTUAL_RETRIEVAL_ENABLED",
        "CONTEXTUAL_RETRIEVAL_BACKEND",
        "CONTEXTUAL_MAX_CHARS",
        "TABLE_MARKDOWN_ENABLED",
        "MULTIMODAL_ENABLED",
        "MULTIMODAL_VLM_BASE_URL",
        "MULTIMODAL_VLM_MODEL",
        "MULTIMODAL_MAX_TOKENS",
        "MULTIMODAL_MAX_IMAGES_PER_DOC",
        "MULTIMODAL_MIN_IMAGE_PX",
        "MULTIMODAL_MIN_IMAGE_BYTES",
        "MULTIMODAL_DESC_MAX_CHARS",
        "MULTIMODAL_IMAGE_PROMPT",
        "TABLE_MIN_ROWS",
        "TABLE_MIN_COLS",
        "MULTIMODAL_IMPLEMENTATION",
        "CHUNK_QUALITY_IMPLEMENTATION",
        "CHUNK_QUALITY_MODE",
        "CONTEXTUAL_RETRIEVAL_MODE",
        "CONTEXTUAL_LLM_MAX_CHARS",
        "CONTEXTUAL_LLM_BATCH",
        "SEMANTIC_SPLIT_THRESHOLD",
        "SEMANTIC_VARIANCE_THRESHOLD",
        "PARSER_IMPLEMENTATION",
        "METADATA_IMPLEMENTATION",
        "CONTEXTUAL_IMPLEMENTATION",
    ),
    "vectors": (
        "EMBED_BACKEND",
        "EMBED_MODEL",
        "EMBED_DIM",
        "EMBED_QUERY_PREFIX",
        "EMBED_TEXT_PREFIX",
        "EMBED_INPUT_TYPE",
        "NVIDIA_QUERY_INPUT_TYPE",
        "NVIDIA_TEXT_INPUT_TYPE",
        "EMBED_API_PROVIDER",
        "EMBED_TRUNCATE",
        "EMBED_SEND_DIMENSIONS",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDING_PROFILE",
    ),
    "lexical": ("BM25_K1", "BM25_B", "LEXICAL_MAX_BYTES", "LEXICAL_IMPLEMENTATION"),
    "graph": (
        "GRAPH_RAG_ENABLED",
        "GRAPH_EXTRACT_BACKEND",
        "GRAPH_LLM_GLEANING",
        "GRAPH_LLM_MAX_CHARS",
        "GRAPH_TOLERATE_CHUNK_FAILURES",
        "GRAPH_LLM_DESC_CHARS",
        "GRAPH_NODE_DESC_CHARS",
        "GRAPH_EVIDENCE_CHARS",
        "GRAPH_COMMUNITY_MIN_SIZE",
        "GRAPH_COMMUNITY_MAX_SUMMARY_CHARS",
        "GRAPH_MAX_TRIPLES_PER_CHUNK",
        "GRAPH_SEMANTIC_BACKEND",
        "GRAPH_SEMANTIC_ENABLED",
        "LLM_MODEL",
        "LLM_BACKEND",
        "OLLAMA_LLM_MODEL",
        "GRAPH_COMMUNITY_ENABLED",
        "GRAPH_PROMPT_IMPLEMENTATION",
        "GRAPH_IMPLEMENTATION",
        "KB_GRAPH_GENERATION",
    ),
}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def pipeline_fingerprints(config: dict[str, Any] | None = None, *, versions=None) -> dict[str, str]:
    config = dict(os.environ if config is None else config)
    versions = versions or STAGE_VERSIONS
    result = {}
    for stage, keys in STAGE_KEYS.items():
        values = {key: config.get(key, "") for key in keys}
        if stage != "chunks":
            values["chunks"] = result["chunks"]
        if stage == "graph":
            values["vectors"] = result["vectors"]
        result[stage] = fingerprint({"version": versions[stage], "config": values})
    return result


def state_path(collection: str | None = None) -> Path:
    name = collection or os.getenv("COLLECTION", "pcb_kb")
    return Path(os.getenv("KB_STATE_PATH", f"./data/kb_state_{name}.json"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RevisionStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else state_path()

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "active": {}, "pending": {}}
        # Corrupt state must fail closed, never silently select an old corpus.
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "active" not in value:
            raise ValueError("invalid knowledge base revision state")
        return value

    def write(self, value):
        atomic_json(self.path, value)

    @contextmanager
    def writer_lock(self):
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def begin(self, docs: dict, pipeline: dict, collection: str) -> dict:
        state = self.read()
        revision = fingerprint({"docs": docs, "pipeline": pipeline, "collection": collection})[:24]
        pending = state.get("pending") or {}
        if pending.get("revision") != revision:
            pending = {
                "revision": revision,
                "docs": docs,
                "pipeline": pipeline,
                "collection": collection,
                "stages": {},
                "started_at": time.time(),
            }
        state["pending"] = pending
        self.write(state)
        return pending

    def stage(self, name: str, operation, *, revision: str | None = None):
        state = self.read()
        pending = state["pending"]
        if revision and pending.get("revision") != revision:
            raise RuntimeError("ingestion revision changed during processing")
        existing = pending["stages"].get(name, {})
        if existing.get("status") == "complete":
            result = existing.get("result")
            artifacts = list(result.get("paths", [])) if isinstance(result, dict) else []
            if isinstance(result, dict) and result.get("path"):
                artifacts.append(result["path"])
            if all(Path(path).is_file() for path in artifacts):
                return result
        pending["stages"][name] = {"status": "running", "started_at": time.time()}
        self.write(state)
        try:
            result = operation()
        except BaseException as error:
            state = self.read()
            pending = state["pending"]
            pending["stages"][name].update(
                status="failed", error_type=type(error).__name__, error=str(error)[:500], finished_at=time.time()
            )
            self.write(state)
            raise
        state = self.read()
        pending = state["pending"]
        pending["stages"][name].update(status="complete", result=result, finished_at=time.time())
        self.write(state)
        return result

    def publish(self, artifacts: dict, *, required=("chunks", "vectors", "lexical", "graph")) -> dict:
        state = self.read()
        pending = state["pending"]
        missing = [name for name in required if pending["stages"].get(name, {}).get("status") != "complete"]
        if missing:
            raise RuntimeError("cannot publish incomplete stages: " + ", ".join(missing))
        for name, path in artifacts.items():
            if path and not Path(path).is_file():
                raise RuntimeError(f"missing {name} artifact")
        active = dict(pending, artifacts=artifacts, published_at=time.time())
        for doc in active.get("docs", {}).values():
            for stage, status in doc.get("stages", {}).items():
                status["status"] = (
                    "complete" if pending["stages"].get(stage, {}).get("status") == "complete" else "pending"
                )
        active["storage_revisions"] = {stage: active["revision"] for stage in required}
        state["previous"] = state.get("active", {})
        state["active"] = active
        state["pending"] = {}
        self.write(state)
        return active


def node_visible(metadata: dict, active: dict) -> bool:
    """Reject staged, removed and old document revisions in every recall route."""
    revision = str(metadata.get("revision") or "")
    if not active:
        return not revision
    document = active.get("docs", {}).get(str(metadata.get("doc_node_id") or ""))
    if document is None:
        return False
    return revision == str(document.get("revision") or "")
