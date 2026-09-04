"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)






# ---------------------------------------------------------------------------
# Overlay K3 (2026-09-04): decomposition is for FRESH cards only.
#
# Block-loop escalation routes a worked card to `triage`; the auto-decomposer
# only checked status, so on 2026-09-04 it re-planned t_2b796e9a (31 runs,
# worf-approved children, a landing card) into six duplicate children and
# overwrote the implementer assignee with the orchestrator. A card with any
# run history or any existing graph edge is not a rough idea — refuse, and
# say what to do instead.
# ---------------------------------------------------------------------------


def _children():
    return [
        {"title": "research", "body": "b", "assignee": "researcher", "parents": []},
        {"title": "build", "body": "b", "assignee": "engineer", "parents": [0]},
    ]


def _to_triage(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))


def test_decompose_refuses_card_with_prior_runs(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="worked", assignee="laforge")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="laforge") is not None
        kb.block_task(conn, tid, reason="stuck", kind="needs_input")
        _to_triage(conn, tid)  # what block-loop escalation does
        with pytest.raises(ValueError, match="run history"):
            kb.decompose_triage_task(
                conn, tid, root_assignee="orchestrator",
                children=_children(), author="auto-decomposer",
            )
        t = kb.get_task(conn, tid)
        assert t.status == "triage"           # untouched
        assert t.assignee == "laforge"        # not overwritten
        assert conn.execute(
            "SELECT count(*) FROM task_links WHERE parent_id=?", (tid,)
        ).fetchone()[0] == 0                  # no duplicate children


def test_decompose_refuses_card_with_existing_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", assignee="laforge")
        child = kb.create_task(conn, title="existing child", assignee="worf")
        kb.link_tasks(conn, parent_id=tid, child_id=child)
        _to_triage(conn, tid)
        with pytest.raises(ValueError, match="already has"):
            kb.decompose_triage_task(
                conn, tid, root_assignee="orchestrator",
                children=_children(), author="auto-decomposer",
            )
        assert kb.get_task(conn, tid).assignee == "laforge"


def test_decompose_keeps_existing_assignee_when_root_assignee_given(kanban_home):
    """A fresh triage card that already names an owner keeps it; the
    orchestrator becomes owner only when the card had none."""
    with kb.connect() as conn:
        owned = _create_triage(conn, title="owned", assignee="laforge")
        unowned = _create_triage(conn, title="unowned")
        assert kb.decompose_triage_task(
            conn, owned, root_assignee="orchestrator",
            children=_children(), author="d",
        )
        assert kb.decompose_triage_task(
            conn, unowned, root_assignee="orchestrator",
            children=_children(), author="d",
        )
        assert kb.get_task(conn, owned).assignee == "laforge"
        assert kb.get_task(conn, unowned).assignee == "orchestrator"
