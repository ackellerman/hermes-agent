"""Live operator-note injection into a running kanban worker.

``tools.kanban_tools.inject_new_comments_from_env`` polls the worker's task
for comments added *after* the run started and folds them into the live turn
via the agent's OUT-OF-BAND steer channel — so a user can talk to a running
task without the block→comment→unblock dance or a restart.

Verifies: no-op off a worker, watermark seeding (history isn't re-injected),
new comments steer (only while the task is ``running`` under the poller's own
dispatcher run — SPEC-0034 status+run gate), own-authored comments are skipped,
and the steer wrapper names the real author (never "from the operator").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
import tools.kanban_tools as kt


class FakeAgent:
    def __init__(self):
        self.steers: list[str] = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return True


@pytest.fixture
def worker_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    # Reset module-level poll state so tests don't leak into each other.
    kt._comment_watermark.clear()
    kt._comment_poll_last_attempt = 0.0
    return home


def _unthrottle():
    """Bypass the inter-poll rate limit for deterministic tests."""
    kt._comment_poll_last_attempt = 0.0


def _set_run_env(monkeypatch, tid, run_id):
    """Env of a dispatcher-owned worker scoped to ``tid`` under run ``run_id``."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _claim(conn, tid, run_id):
    """ready -> running with ``current_run_id = run_id`` (the poller's own run)."""
    claimed = kb.claim_task(conn, tid, claimer="test-claimer")
    assert claimed is not None, "claim failed"
    conn.execute(
        "UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, tid)
    )
    conn.commit()
    return claimed


def test_noop_without_worker_env(worker_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = FakeAgent()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


def test_seed_then_inject_new_comment(worker_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="live task")
        kb.add_comment(conn, tid, author="desktop", body="pre-existing note")
    finally:
        conn.close()

    conn = kbc.connect()
    try:
        _claim(conn, tid, run_id=751)
    finally:
        conn.close()
    _set_run_env(monkeypatch, tid, 751)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    # First poll seeds the watermark past the existing thread — no injection.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="desktop", body="actually use the v2 API")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert len(agent.steers) == 1
    assert "v2 API" in agent.steers[0]

    # Watermark advanced — a re-poll with no new comments injects nothing.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert len(agent.steers) == 1


def test_skips_own_authored_comments(worker_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="echo guard")
    finally:
        conn.close()

    conn = kbc.connect()
    try:
        _claim(conn, tid, run_id=751)
    finally:
        conn.close()
    _set_run_env(monkeypatch, tid, 751)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="worker-bot", body="i did a thing")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


# --- SPEC-0034 falsifiers (AC1/AC2/AC4) ---

def test_refuses_non_running_task_ac1(worker_home, monkeypatch):
    """AC1: no comment steer after the card leaves ``running``.

    Falsifies the 2026-09-12 incident shape: implementer had called
    ``kanban_request_review`` (task now ``review``), reviewer posted a fresh
    comment, the still-alive implementer's poller steered it into the
    finished run. The gate must return False and leave ``agent.steers`` empty
    even though a fresh foreign comment would otherwise be steered.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="reviewed task")
        _claim(conn, tid, run_id=759)
        kb.add_comment(conn, tid, author="desktop", body="pre-run note")  # seed thread
    finally:
        conn.close()

    conn = kbc.connect()
    try:
        # The run terminally ended: implementer's own request_review handoff
        # moved the card to review and closed run 759.
        ok = kb.request_review(conn, tid, expected_run_id=759, summary="done")
        assert ok, "request_review failed"
    finally:
        conn.close()

    _set_run_env(monkeypatch, tid, 759)  # stale process still thinks it owns run 759
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seeds watermark past "pre-run note"

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="worf-reviewer", body="blocked: see verdict above")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []

    # Watermark still advanced: a later poll of the same process re-injects nothing.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


def test_refuses_stale_run_id_ac2(worker_home, monkeypatch):
    """AC2: a poller whose run id mismatches ``current_run_id`` never steers."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="reclaimed task")
        _claim(conn, tid, run_id=760)
        kb.add_comment(conn, tid, author="desktop", body="pre-run note")
    finally:
        conn.close()

    _set_run_env(monkeypatch, tid, 759)  # dispatcher re-claimed under a NEW run
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="operator-seat", body="please continue with the v2 approach")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


def test_label_names_real_author_ac4(worker_home, monkeypatch):
    """AC4: a peer-authored comment is never wrapped as "from the operator"."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="label guard")
        _claim(conn, tid, run_id=751)
        kb.add_comment(conn, tid, author="desktop", body="pre-run note")
    finally:
        conn.close()

    _set_run_env(monkeypatch, tid, 751)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="worf-reviewer", body="verdict: approve with notes")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert len(agent.steers) == 1
    note = agent.steers[0]
    # The author-prefixed line carries the real author...
    assert "worf-reviewer" in note
    # ...and the wrapper never claims operator origin for a peer note.
    assert "from the operator" not in note
