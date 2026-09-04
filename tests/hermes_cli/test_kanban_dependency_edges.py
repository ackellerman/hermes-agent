"""K1: a dependency block is an edge or it does not exist (overlay).

Today (2026-09-04) t_1bf1216b named its real dependency in 94 block reasons
and had zero ``task_links`` rows to it; ``recompute_ready`` promoted it every
60 s tick and the dispatcher cold-spawned a worker 97 times. These tests pin
the contract that closes that hole: ``block_task(kind="dependency")`` MUST
carry ``depends_on=[...]``, the kernel writes the edges, and the card cannot
be promoted until those parents are done.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _finish(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    kb.claim_task(conn, tid, claimer="worker")
    kb.complete_task(conn, tid, result="done")


def _parents(conn, child):
    return {
        r["parent_id"]
        for r in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id=?", (child,)
        ).fetchall()
    }


def test_dependency_block_without_depends_on_is_refused(kanban_home):
    """The t_1bf1216b shape: prose names the parent, no edge. Must refuse
    and name the argument so the worker self-heals in one turn."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, "child")
        with pytest.raises(ValueError, match="depends_on"):
            kb.block_task(conn, child, reason="await t_deadbeef", kind="dependency")
        # Nothing moved: still running, no synthetic dependency_wait event.
        assert kb.get_task(conn, child).status == "running"
        assert not [e for e in kb.list_events(conn, child) if e.kind == "dependency_wait"]


def test_dependency_block_writes_edge_and_parks_until_parent_done(kanban_home):
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, "child")
        assert _parents(conn, child) == set()
        ok = kb.block_task(
            conn, child, reason="await parent", kind="dependency",
            depends_on=[parent],
        )
        assert ok is True
        assert _parents(conn, child) == {parent}
        assert kb.get_task(conn, child).status == "todo"
        # The 60 s tick: with the parent still open, NO promotion.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Parent finishes → promoted exactly then.
        _finish(conn, parent)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_on_already_done_parent_is_refused(kanban_home):
    """Nothing to wait on → the worker is wrong; refuse rather than park."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        _finish(conn, parent)
        child = _running_task(conn, "child")
        with pytest.raises(ValueError, match="already done"):
            kb.block_task(
                conn, child, reason="await", kind="dependency", depends_on=[parent],
            )
        assert kb.get_task(conn, child).status == "running"


def test_dependency_block_unknown_parent_is_refused(kanban_home):
    with kb.connect_closing() as conn:
        child = _running_task(conn, "child")
        with pytest.raises(ValueError, match="unknown task"):
            kb.block_task(
                conn, child, reason="await", kind="dependency",
                depends_on=["t_00000000"],
            )
        assert kb.get_task(conn, child).status == "running"


def test_dependency_block_self_or_cycle_is_refused(kanban_home):
    with kb.connect_closing() as conn:
        a = kb.create_task(conn, title="a", assignee="worker")
        b = _running_task(conn, "b")
        kb.link_tasks(conn, parent_id=b, child_id=a)  # b -> a
        with pytest.raises(ValueError, match="cycle"):
            kb.block_task(conn, b, reason="await a", kind="dependency", depends_on=[a])
        with pytest.raises(ValueError, match="itself"):
            kb.block_task(conn, b, reason="await me", kind="dependency", depends_on=[b])


def test_dependency_block_event_carries_depends_on(kanban_home):
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, "child")
        kb.block_task(conn, child, reason="await", kind="dependency", depends_on=[parent])
        ev = [e for e in kb.list_events(conn, child) if e.kind == "dependency_wait"][-1]
        assert ev.payload.get("depends_on") == [parent]


def test_non_dependency_kinds_ignore_depends_on(kanban_home):
    """depends_on is only meaningful for kind=dependency; other kinds are
    unchanged (they still route to blocked/triage as before)."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, "child")
        assert kb.block_task(conn, child, reason="need a human", kind="needs_input")
        assert kb.get_task(conn, child).status == "blocked"
