"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Overlay K1: a dependency block names its parents (the edge already
        # exists here; INSERT OR IGNORE keeps it idempotent).
        kb.block_task(conn, child, reason="wait", kind="dependency",
                      depends_on=[parent])
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------




def test_unblock_survives_kind_deliberately(kanban_home, tmp_path):
    """block_kind/block_recurrences SURVIVE unblock by design: resetting them is
    the amnesia that let a cron unblock<->re-block loop unbounded (the breaker
    at BLOCK_RECURRENCE_LIMIT depends on the memory). Status is the authority
    on 'is it blocked now'; consumers must read status, not block_kind."""
    conn = kb.connect(tmp_path / "b.db")
    t = kb.create_task(conn, title="x", assignee="w")
    assert kb.block_task(conn, t, reason="waiting on human", kind="needs_input")
    assert kb.unblock_task(conn, t)
    row = kb.get_task(conn, t)
    assert row.status in ("todo", "ready")          # not blocked any more
    assert row.block_kind == "needs_input"          # memory survives, on purpose
    # and the breaker still counts across the unblock:
    assert kb.block_task(conn, t, reason="again", kind="needs_input")
    ev = [e for e in kb.list_events(conn, t) if e.kind == "block_loop_detected"]
    assert ev and ev[-1].payload["recurrences"] == 2


def test_refusal_prints_once_with_repair(kanban_home, capsys) -> None:
    """A refused block prints ONE message: the specific reason + the edge repair.

    The generic bulk 'cannot block <id>' line under it was a second, vaguer
    message for the same failure — suppressed via fail_msg -> None.
    """
    import argparse
    from hermes_cli import kanban as kcli
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="p", assignee="w")
        child = kb.create_task(conn, title="c", assignee="w", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"
    args = argparse.Namespace(task_id=child, reason=["waiting"], kind=None,
                              ids=None, depends_on=None)
    rc = kcli._cmd_block(args)
    err = capsys.readouterr().err
    assert rc == 1
    assert err.count("cannot block") == 1, err
    assert "status='todo'" in err and "kanban link" in err, err
