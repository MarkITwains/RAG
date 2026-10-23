"""Operator views and one bounded, resumable ingestion subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from pcb_rag.revisions import RevisionStore


def processing_status(store: RevisionStore):
    state = store.read()
    active, pending = state.get("active", {}), state.get("pending", {})
    return {
        "revision": active.get("revision", "legacy"),
        "collection": active.get("collection", ""),
        "published_at": active.get("published_at"),
        "documents": len(active.get("docs", {})),
        "storage_revisions": active.get("storage_revisions", {}),
        "pipeline": active.get("pipeline", {}),
        "pending_revision": pending.get("revision"),
        "stages": pending.get("stages", {}),
        "retry_available": bool(pending),
        "previous_revision": state.get("previous", {}).get("revision"),
    }


def document_status(store: RevisionStore, *, offset=0, limit=100):
    state = store.read()
    active, pending = state.get("active", {}), state.get("pending", {})
    docs = []
    for doc_id in sorted(set(active.get("docs", {})) | set(pending.get("docs", {}))):
        published = active.get("docs", {}).get(doc_id, {})
        staged = pending.get("docs", {}).get(doc_id)
        stage_names = ("chunks", "vectors", "lexical", "graph")
        stages = {
            name: (pending if staged else active).get("stages", {}).get(name, {}).get("status", "pending")
            for name in stage_names
        }
        graph_stage = pending.get("stages", {}).get("graph:" + doc_id)
        if staged and graph_stage:
            stages["graph"] = graph_stage["status"]
        docs.append(
            {
                "doc_id": doc_id,
                "source_path": (staged or published).get("path", ""),
                "revision": published.get("revision"),
                "pending_revision": (staged or {}).get("revision"),
                "status": "removed_on_publish" if pending and not staged else "processing" if staged else "published",
                "stages": stages,
            }
        )
    return {"total": len(docs), "offset": offset, "documents": docs[offset : offset + limit]}


class IngestionJobs:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._job = None

    def status(self):
        with self._lock:
            if not self._job:
                return {"status": "idle"}
            code = self._process.poll()
            return dict(
                self._job, status="running" if code is None else "complete" if code == 0 else "failed", exit_code=code
            )

    def retry(self, store: RevisionStore):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("an ingestion job is already running")
            with store.writer_lock():
                pending = store.read().get("pending", {})
                if not pending:
                    raise ValueError("no pending ingestion to retry")
            environment = dict(
                os.environ,
                KB_STATE_PATH=str(store.path.resolve()),
                COLLECTION=pending["collection"],
                INGEST_EXPECTED_REVISION=pending["revision"],
                INGEST_PLAN_ONLY="0",
                INGEST_OVERWRITE="0",
            )
            log_path = store.path.with_suffix(".retry.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as output:
                self._process = subprocess.Popen(
                    [sys.executable, "-m", "pcb_rag.ingest"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            self._job = {
                "revision": pending["revision"],
                "started_at": time.time(),
                "pid": self._process.pid,
                "log_path": str(Path(log_path).resolve()),
            }
        return self.status()


JOBS = IngestionJobs()
