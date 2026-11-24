"""Payload-level edges of the resume window: the window each ``action='list'`` row reports, and which rows the
listing reads a durable record for.

``tests/tools/test_delegate_resume_window.py`` pins ``list_entry`` directly; these two go through
``delegate_task(action='list')`` so a mutant in ``_list_payload``'s wiring (the viewer's window instead of the
row's, or a record read for every status) cannot survive behind that unit-level cover.
"""

import json
import sqlite3
import time
from unittest.mock import patch

from tests.tools.test_delegate_continue_kept_children import (
    _KeptModel, _forget_process_state, _kept_parent, _kept_rows, _spawn_kept,
)
from tests.tools.test_delegate_resume_window import (
    _Roles, _configure, _set_record, _write_profile,
)
from tests.waiting import wait_until
from tools import delegate_children as dc
from tools import delegate_tool_resumable as resumable


def _wait_idle(parent, count):
    """Wait until *count* kept rows of *parent* read ``idle``."""
    def ready():
        rows = [r for r in _kept_rows(parent) if r["status"] == "idle"]
        return rows if len(rows) == count else None
    return wait_until(ready, what=f"{count} kept rows idle")


def _set_kept_status(parent, sid, status):
    """Rewrite a child's kept row (``delegate_children``) — the row ``action='list'`` reads its status from."""
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute(f"UPDATE {dc.TABLE} SET status=? WHERE subagent_id=?", (status, sid))
    conn.close()


def test_list_payload_uses_the_rows_own_window(tmp_path, monkeypatch):
    """Two kept children at the same age (1000 s): the ``reviewer`` child reports the reviewer window (3540 s, warm)
    and the ``implementer`` child the ``default`` one (270 s, cold). The viewer's own window gives 270 to both, so
    passing it through ``_list_payload`` would show the reviewer cold."""
    launch = tmp_path / "os-home" / ".hermes"
    for role in ("reviewer", "implementer"):
        _write_profile(launch / "profiles" / role, host=f"{role}.invalid", model=f"{role}-model", key=f"{role}-key")
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key",
                   delegation={"resume_window_seconds": {"default": 270, "reviewer": 3540}})
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROFILE_KEY", "launch-key")

    parent = _kept_parent(tmp_path / "parent-db")
    model = _KeptModel()
    with patch("tools.delegate_tool_routes._active_provider", return_value=_Roles(("reviewer", "implementer"))), \
            patch("run_agent.AIAgent", side_effect=model):
        review, _a = _spawn_kept(parent, "review the change", profile="reviewer")
        impl, _b = _spawn_kept(parent, "build it", profile="implementer")
    review_sid, impl_sid = review["subagent_ids"][0], impl["subagent_ids"][0]
    _wait_idle(parent, 2)
    aged = time.time() - 1000  # 16 minutes: past 270 s, inside 3540 s
    _set_record(parent, review_sid, stopped_at=aged)
    _set_record(parent, impl_sid, stopped_at=aged)

    from tools.delegate_tool import delegate_task
    listed = json.loads(delegate_task(action="list", parent_agent=parent))
    entries = {e["subagent_id"]: e for e in listed["children"]}
    assert set(entries) == {review_sid, impl_sid}, listed
    assert entries[review_sid]["idle_seconds"] >= 1000 and entries[impl_sid]["idle_seconds"] >= 1000, entries
    assert entries[review_sid]["resume_window_seconds"] == 3540, entries[review_sid]
    assert entries[review_sid]["warm"] is True, entries[review_sid]
    assert entries[impl_sid]["resume_window_seconds"] == 270, entries[impl_sid]
    assert entries[impl_sid]["warm"] is False, entries[impl_sid]


def test_list_payload_reads_records_only_for_idle_rows(tmp_path, monkeypatch):
    """The listing reads a durable record for the IDLE child it will show, and only for that one: a running kept
    row is listed from the kept row alone. Monkeypatched ``read_records`` records the ids it is asked for."""
    _configure({})
    parent = _kept_parent(tmp_path)
    model = _KeptModel()
    with patch("run_agent.AIAgent", side_effect=model):
        first, _a = _spawn_kept(parent, "write the parser")
        second, _b = _spawn_kept(parent, "write the lexer")
    idle_sid, running_sid = first["subagent_ids"][0], second["subagent_ids"][0]
    _wait_idle(parent, 2)
    _set_kept_status(parent, running_sid, "running")
    _forget_process_state()  # a restart: no live child, so the running row is listed from its kept row

    calls = []
    real = resumable.read_records

    def spy(parent_agent, ids):
        calls.append(list(ids))
        return real(parent_agent, ids)

    monkeypatch.setattr(resumable, "read_records", spy)
    from tools.delegate_tool import delegate_task
    listed = json.loads(delegate_task(action="list", parent_agent=parent))
    assert {e["subagent_id"] for e in listed["children"]} == {idle_sid, running_sid}, listed
    assert calls == [[idle_sid]], calls  # one read, for the idle child only
