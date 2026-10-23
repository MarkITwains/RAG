"""Tenant/user-owned sessions with the same checks in memory and SQLite."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

from pcb_rag import security


class SessionAccessDenied(PermissionError):
    pass


def owner_key(principal=None):
    principal = principal or security.current_principal()
    if not principal.authenticated:
        if security.ACL_ENABLED or os.getenv("SESSION_REQUIRE_IDENTITY", "0") == "1":
            raise SessionAccessDenied("authenticated user identity required for sessions")
        return "local-single-user"
    return hashlib.sha256(json.dumps([principal.tenant_id, principal.user_id]).encode()).hexdigest()


class SessionManager:
    def __init__(self, max_sessions=None, expire_minutes=None, *, store=None, db_path=None):
        self._sessions = OrderedDict()
        self._max_sessions = max_sessions or int(os.getenv("SESSION_MAX_COUNT", "1000"))
        self._expire_delta = timedelta(minutes=expire_minutes or int(os.getenv("SESSION_EXPIRE_MINUTES", "60")))
        self._lock = threading.RLock()
        self._db = None
        if (store or os.getenv("SESSION_STORE", "memory")) == "sqlite":
            path = Path(db_path or os.getenv("SESSION_DB_PATH", "./data/sessions.db"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT, owner TEXT NOT NULL)"
            )
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(sessions)")}
            if "owner" not in columns:
                # Keep legacy history, but never assign it to the first visitor.
                self._db.execute("ALTER TABLE sessions ADD COLUMN owner TEXT NOT NULL DEFAULT 'legacy_unowned'")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)"
            )
            self._db.commit()
            for sid, created, updated, owner in self._db.execute(
                "SELECT session_id, created_at, updated_at, owner FROM sessions ORDER BY updated_at"
            ):
                self._sessions[sid] = {
                    "history": [],
                    "owner": owner,
                    "created_at": datetime.fromisoformat(created),
                    "updated_at": datetime.fromisoformat(updated),
                }
            for sid, role, content, timestamp in self._db.execute(
                "SELECT session_id, role, content, timestamp FROM messages ORDER BY id"
            ):
                if sid in self._sessions:
                    self._sessions[sid]["history"].append({"role": role, "content": content, "timestamp": timestamp})
            self.cleanup_expired(all_owners=True)

    def _get(self, sid, owner):
        session = self._sessions.get(sid)
        if session is not None and session["owner"] != owner:
            raise SessionAccessDenied("session not found")
        if session and datetime.now() - session["updated_at"] > self._expire_delta:
            self._drop(sid)
            return None
        return session

    def _persist(self, sid):
        if self._db is None:
            return
        session = self._sessions[sid]
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at, owner) VALUES (?, ?, ?, ?)",
                (sid, session["created_at"].isoformat(), session["updated_at"].isoformat(), session["owner"]),
            )
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            self._db.executemany(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                [(sid, message["role"], message["content"], message["timestamp"]) for message in session["history"]],
            )

    def _drop(self, sid):
        self._sessions.pop(sid, None)
        if self._db is not None:
            with self._db:
                self._db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                self._db.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))

    def create_session(self, session_id=None, *, owner=None):
        owner = owner or owner_key()
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if self._get(sid, owner) is None:
                self._sessions[sid] = {
                    "owner": owner,
                    "history": [],
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                while len(self._sessions) > self._max_sessions:
                    self._drop(next(iter(self._sessions)))
                self._persist(sid)
        return sid

    def get_or_create(self, session_id=None, *, owner=None):
        owner = owner or owner_key()
        with self._lock:
            if session_id and self._get(session_id, owner):
                self._sessions[session_id]["updated_at"] = datetime.now()
                self._sessions.move_to_end(session_id)
                self._persist(session_id)
                return session_id, False
            return self.create_session(session_id, owner=owner), True

    def get_history(self, session_id, *, owner=None):
        with self._lock:
            session = self._get(session_id, owner or owner_key())
            if not session:
                return []
            session["updated_at"] = datetime.now()
            self._sessions.move_to_end(session_id)
            self._persist(session_id)
            return copy.deepcopy(session["history"])

    def exists(self, session_id, *, owner=None):
        try:
            with self._lock:
                return self._get(session_id, owner or owner_key()) is not None
        except SessionAccessDenied:
            return False

    def add_message(self, session_id, role, content, *, owner=None):
        if role not in {"user", "assistant"}:
            raise ValueError("invalid conversation role")
        with self._lock:
            session = self._get(session_id, owner or owner_key())
            if session is None:
                raise SessionAccessDenied("session not found")
            session["history"].append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
            session["history"] = session["history"][-int(os.getenv("SESSION_MAX_MESSAGES", "100")) :]
            session["updated_at"] = datetime.now()
            self._sessions.move_to_end(session_id)
            self._persist(session_id)

    def clear_session(self, session_id, *, owner=None):
        with self._lock:
            session = self._get(session_id, owner or owner_key())
            if session is None:
                return False
            session["history"] = []
            session["updated_at"] = datetime.now()
            self._persist(session_id)
            return True

    def delete_session(self, session_id, *, owner=None):
        with self._lock:
            if self._get(session_id, owner or owner_key()) is None:
                return False
            self._drop(session_id)
            return True

    def list_sessions(self, *, owner=None):
        owner = owner or owner_key()
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "message_count": len(session["history"]),
                    "created_at": session["created_at"].isoformat(),
                    "updated_at": session["updated_at"].isoformat(),
                }
                for sid, session in self._sessions.items()
                if session["owner"] == owner and datetime.now() - session["updated_at"] <= self._expire_delta
            ]

    def cleanup_expired(self, *, owner=None, all_owners=False):
        owner = None if all_owners else (owner or owner_key())
        with self._lock:
            targets = [
                sid
                for sid, session in self._sessions.items()
                if (all_owners or session["owner"] == owner)
                and datetime.now() - session["updated_at"] > self._expire_delta
            ]
            for sid in targets:
                self._drop(sid)
            return len(targets)

    def close(self):
        if self._db is not None:
            self._db.close()


def principal_from_token(authorization):
    """Optional individual bearer credentials, stored as SHA-256 keys on disk."""
    path = os.getenv("AUTH_TOKEN_PRINCIPALS_FILE", "")
    if not path or not authorization or not authorization.startswith("Bearer "):
        return None
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
    record = json.loads(Path(path).read_text(encoding="utf-8")).get(digest)
    if not isinstance(record, dict) or not record.get("user_id") or not record.get("tenant_id"):
        return None
    return security.resolve_principal(
        user_id=record["user_id"],
        tenant_id=record["tenant_id"],
        roles=record.get("roles", []),
        groups=record.get("groups", []),
    )
