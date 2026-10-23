"""Durable, cross-process accounting, throttling and circuit breaking for rerank.

Provider usage is authoritative when returned. Unknown usage remains charged
against the local budget using a conservative UTF-8 byte estimate; it is never
reported as observed billing. Reservations include waiting/in-flight requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class RerankStopped(RuntimeError):
    """No further rerank calls may be issued in this accounting scope."""


class RerankBudgetExceeded(RerankStopped):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_usage(data: dict) -> tuple[int | None, str, dict]:
    """Recognize SiliconFlow billing/tokens and OpenAI-style usage responses."""
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for label, value in [
        ("meta.billed_units", meta.get("billed_units")),
        ("usage", data.get("usage")),
        ("meta.tokens", meta.get("tokens")),
    ]:
        if not isinstance(value, dict):
            continue
        clean = {
            key: number
            for key, number in value.items()
            if key in {"total_tokens", "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}
            and isinstance(number, int)
            and not isinstance(number, bool)
            and number >= 0
        }
        if "total_tokens" in clean:
            return clean["total_tokens"], label, clean
        for first, second in [("input_tokens", "output_tokens"), ("prompt_tokens", "completion_tokens")]:
            if first in clean:
                return clean[first] + clean.get(second, 0), label, clean
    return None, "unavailable", {}


def estimate_tokens(query: str, documents: list[str]) -> int:
    """Conservative reservation, including a repeated query and framing per pair."""
    query_bytes = len(query.encode("utf-8"))
    return sum(query_bytes + len(document.encode("utf-8")) + 64 for document in documents)


@dataclass(frozen=True)
class Ticket:
    request_id: str
    scheduled_at: float
    estimated_tokens: int


class RerankControl:
    def __init__(self, path: str | Path, scope: str, token_budget: int, request_budget: int):
        if token_budget < 0 or request_budget < 0:
            raise ValueError("Rerank budgets cannot be negative; zero means unlimited")
        self.path = Path(path).expanduser()
        self.scope = scope
        self.token_budget = token_budget
        self.request_budget = request_budget
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS budgets (
                    scope TEXT PRIMARY KEY, token_limit INTEGER NOT NULL, request_limit INTEGER NOT NULL,
                    accounted_tokens INTEGER NOT NULL DEFAULT 0, reserved_tokens INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0, circuit_reason TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rate_limits (provider TEXT PRIMARY KEY, next_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY, scope TEXT NOT NULL, phase TEXT NOT NULL, activity TEXT NOT NULL,
                    model TEXT NOT NULL, candidates INTEGER NOT NULL, query_chars INTEGER NOT NULL, requested_top_n INTEGER,
                    document_chars INTEGER NOT NULL, estimated_tokens INTEGER NOT NULL,
                    actual_tokens INTEGER, accounted_tokens INTEGER, usage_source TEXT, usage_json TEXT,
                    status TEXT NOT NULL, http_status INTEGER, error_type TEXT, response_meta_json TEXT,
                    started_at TEXT NOT NULL, finished_at TEXT, scheduled_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS requests_scope ON requests(scope);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
            if "response_meta_json" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN response_meta_json TEXT")
            if "requested_top_n" not in columns:
                db.execute("ALTER TABLE requests ADD COLUMN requested_top_n INTEGER")
            db.execute(
                "INSERT OR IGNORE INTO budgets(scope,token_limit,request_limit,updated_at) VALUES(?,?,?,?)",
                (scope, token_budget, request_budget, utc_now()),
            )
            stored = db.execute("SELECT token_limit,request_limit FROM budgets WHERE scope=?", (scope,)).fetchone()
            if stored != (token_budget, request_budget):
                raise ValueError("Rerank scope budgets differ from the frozen limits; choose a new scope")
        self.path.chmod(0o600)

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        default = Path.home() / ".local/share/pcb-rag/rerank/control.sqlite3"
        return cls(
            env.get("RERANK_CONTROL_DB", str(default)),
            env.get("RERANK_BUDGET_SCOPE", "production:" + utc_now()[:10]),
            int(env.get("RERANK_TOKEN_BUDGET", "0")),
            int(env.get("RERANK_REQUEST_BUDGET", "0")),
        )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        try:
            with db:
                yield db
        finally:
            db.close()

    def snapshot(self) -> dict:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            state = dict(db.execute("SELECT * FROM budgets WHERE scope=?", (self.scope,)).fetchone())
            state["token_limit_enabled"] = state["token_limit"] > 0
            state["request_limit_enabled"] = state["request_limit"] > 0
            state["by_status"] = [
                dict(row)
                for row in db.execute(
                    "SELECT status,COUNT(*) AS requests,SUM(candidates) AS candidates,"
                    "SUM(actual_tokens) AS actual_tokens,SUM(accounted_tokens) AS accounted_tokens "
                    "FROM requests WHERE scope=? GROUP BY status",
                    (self.scope,),
                )
            ]
            state["reported_tokens"] = db.execute(
                "SELECT COALESCE(SUM(actual_tokens),0) FROM requests WHERE scope=?",
                (self.scope,),
            ).fetchone()[0]
            state["by_http_status"] = [
                dict(row)
                for row in db.execute(
                    "SELECT http_status,COUNT(*) AS requests,SUM(actual_tokens) AS actual_tokens "
                    "FROM requests WHERE scope=? AND http_status IS NOT NULL GROUP BY http_status",
                    (self.scope,),
                )
            ]
            state["unknown_usage_requests"] = db.execute(
                "SELECT COUNT(*) FROM requests WHERE scope=? AND actual_tokens IS NULL AND status!='cancelled'",
                (self.scope,),
            ).fetchone()[0]
        return state

    def check(self) -> None:
        with self.connect() as db:
            reason = db.execute("SELECT circuit_reason FROM budgets WHERE scope=?", (self.scope,)).fetchone()[0]
        if reason:
            raise RerankStopped("Rerank stopped: " + reason)

    def trip(self, reason: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE budgets SET circuit_reason=COALESCE(circuit_reason,?),updated_at=? WHERE scope=?",
                (reason, utc_now(), self.scope),
            )

    def reserve(
        self,
        *,
        provider: str,
        model: str,
        query: str,
        documents: list[str],
        interval: float,
        phase: str,
        activity: str,
        current_time: float | None = None,
        top_n: int | None = None,
    ) -> Ticket:
        estimate = estimate_tokens(query, documents)
        timestamp = time.time() if current_time is None else current_time
        request_id = uuid.uuid4().hex
        failure = None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            limit, requests, used, reserved, attempts, reason = db.execute(
                "SELECT token_limit,request_limit,accounted_tokens,reserved_tokens,attempts,circuit_reason "
                "FROM budgets WHERE scope=?",
                (self.scope,),
            ).fetchone()
            if reason:
                failure = RerankStopped("Rerank stopped: " + reason)
            elif (limit > 0 and used + reserved + estimate > limit) or (requests > 0 and attempts >= requests):
                reason = (
                    "token_budget_exhausted"
                    if limit > 0 and used + reserved + estimate > limit
                    else "request_budget_exhausted"
                )
                db.execute(
                    "UPDATE budgets SET circuit_reason=?,updated_at=? WHERE scope=?", (reason, utc_now(), self.scope)
                )
                failure = RerankBudgetExceeded("Rerank stopped before sending: " + reason)
            else:
                previous = db.execute("SELECT next_at FROM rate_limits WHERE provider=?", (provider,)).fetchone()
                scheduled = timestamp if interval <= 0 else max(timestamp, previous[0] if previous else timestamp)
                db.execute(
                    "INSERT OR REPLACE INTO rate_limits(provider,next_at) VALUES(?,?)", (provider, scheduled + interval)
                )
                db.execute(
                    "UPDATE budgets SET reserved_tokens=reserved_tokens+?,attempts=attempts+1,updated_at=? WHERE scope=?",
                    (estimate, utc_now(), self.scope),
                )
                db.execute(
                    "INSERT INTO requests(request_id,scope,phase,activity,model,candidates,query_chars,document_chars,"
                    "estimated_tokens,status,started_at,scheduled_at,requested_top_n) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        request_id,
                        self.scope,
                        phase,
                        activity,
                        model,
                        len(documents),
                        len(query),
                        sum(map(len, documents)),
                        estimate,
                        "reserved",
                        utc_now(),
                        scheduled,
                        top_n,
                    ),
                )
        if failure:
            raise failure
        return Ticket(request_id, scheduled, estimate)

    def wait(self, ticket: Ticket) -> None:
        try:
            while True:
                self.check()
                remaining = ticket.scheduled_at - time.time()
                if remaining <= 0:
                    return
                time.sleep(min(remaining, 0.25))
        except RerankStopped:
            self.finish(ticket, status="cancelled", actual_tokens=0, usage_source="not_sent")
            raise

    def finish(
        self,
        ticket: Ticket,
        *,
        status: str,
        actual_tokens: int | None = None,
        usage_source: str = "unavailable",
        usage: dict | None = None,
        http_status: int | None = None,
        error_type: str | None = None,
        response_meta: dict | None = None,
        stop_reason: str | None = None,
    ) -> dict:
        accounted = ticket.estimated_tokens if actual_tokens is None else actual_tokens
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stored = db.execute(
                "SELECT status FROM requests WHERE request_id=? AND scope=?", (ticket.request_id, self.scope)
            ).fetchone()
            if stored is None or stored[0] != "reserved":
                raise RuntimeError("Rerank request was already settled or belongs to another scope")
            db.execute(
                "UPDATE requests SET status=?,actual_tokens=?,accounted_tokens=?,usage_source=?,usage_json=?,"
                "http_status=?,error_type=?,response_meta_json=?,finished_at=? WHERE request_id=?",
                (
                    status,
                    actual_tokens,
                    accounted,
                    usage_source,
                    json.dumps(usage or {}),
                    http_status,
                    error_type,
                    json.dumps(response_meta or {}),
                    utc_now(),
                    ticket.request_id,
                ),
            )
            db.execute(
                "UPDATE budgets SET reserved_tokens=reserved_tokens-?,accounted_tokens=accounted_tokens+?,updated_at=? WHERE scope=?",
                (ticket.estimated_tokens, accounted, utc_now(), self.scope),
            )
            if stop_reason:
                db.execute(
                    "UPDATE budgets SET circuit_reason=COALESCE(circuit_reason,?) WHERE scope=?",
                    (stop_reason, self.scope),
                )
            used, limit = db.execute(
                "SELECT accounted_tokens,token_limit FROM budgets WHERE scope=?", (self.scope,)
            ).fetchone()
            if limit > 0 and used >= limit:
                db.execute(
                    "UPDATE budgets SET circuit_reason=COALESCE(circuit_reason,'token_budget_exhausted') WHERE scope=?",
                    (self.scope,),
                )
        return {
            "request_id": ticket.request_id,
            "scope": self.scope,
            "status": status,
            "actual_tokens": actual_tokens,
            "accounted_tokens": accounted,
            "usage_source": usage_source,
            "usage": usage or {},
            "http_status": http_status,
            "response_meta": response_meta or {},
        }

    def records(self) -> list[dict]:
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in db.execute("SELECT * FROM requests WHERE scope=? ORDER BY started_at", (self.scope,))
            ]


def provider_identity(url: str, key: str) -> str:
    return hashlib.sha256((url.rstrip("/") + "\0" + key).encode()).hexdigest()
