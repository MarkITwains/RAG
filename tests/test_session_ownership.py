import hashlib
import json
import sqlite3

import pytest

from pcb_rag.sessions import SessionAccessDenied, SessionManager, principal_from_token


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_every_session_operation_enforces_owner(tmp_path, backend):
    path = tmp_path / "sessions.db"
    manager = SessionManager(store=backend, db_path=path)
    sid = manager.create_session(owner="tenantA:alice")
    manager.add_message(sid, "user", "private history", owner="tenantA:alice")
    for stranger in ("tenantA:bob", "tenantB:alice"):
        assert manager.list_sessions(owner=stranger) == []
        assert not manager.exists(sid, owner=stranger)
        for method in (
            manager.get_history,
            manager.clear_session,
            manager.delete_session,
            manager.get_or_create,
            manager.create_session,
        ):
            with pytest.raises(SessionAccessDenied):
                method(sid, owner=stranger)
        with pytest.raises(SessionAccessDenied):
            manager.add_message(sid, "user", "injection", owner=stranger)
    if backend == "sqlite":
        manager.close()
        manager = SessionManager(store=backend, db_path=path)
    assert manager.get_history(sid, owner="tenantA:alice")[0]["content"] == "private history"
    assert manager.list_sessions(owner="tenantA:bob") == []
    manager.close()


def test_legacy_sqlite_history_not_claimed_by_first_user(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)")
        db.execute("INSERT INTO sessions VALUES ('old', '2026-10-23T12:00:00', '2026-10-23T12:00:00')")
    manager = SessionManager(store="sqlite", db_path=path, expire_minutes=10**9)
    with pytest.raises(SessionAccessDenied):
        manager.get_or_create("old", owner="first-visitor")
    assert manager.list_sessions(owner="first-visitor") == []
    manager.close()


def test_individual_bearer_credentials_bind_identity_without_trusting_headers(tmp_path, monkeypatch):
    path = tmp_path / "principals.json"
    path.write_text(
        json.dumps({hashlib.sha256(b"test-only-token").hexdigest(): {"user_id": "alice", "tenant_id": "A"}})
    )
    monkeypatch.setenv("AUTH_TOKEN_PRINCIPALS_FILE", str(path))
    assert principal_from_token("Bearer invalid") is None
    principal = principal_from_token("Bearer test-only-token")
    assert principal.user_id == "alice" and principal.tenant_id == "A"
