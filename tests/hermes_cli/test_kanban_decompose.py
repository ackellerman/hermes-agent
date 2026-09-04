"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_board_orchestrator_profile_overrides_global_and_active_fallback(kanban_home, monkeypatch):
    """A fan-out root belongs to its board's coordinator, not another board's."""
    kb.create_board("fleet")
    kb.write_board_metadata("fleet", orchestrator_profile="fleet-orchestrator")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "fleet")

    patches = _patch_list_profiles([
        "fleet-orchestrator", "global-orchestrator", "active-profile",
    ])
    for p in patches:
        p.start()
    try:
        resolved = decomp._resolve_orchestrator_profile(
            {"kanban": {"orchestrator_profile": "global-orchestrator"}}
        )
    finally:
        for p in patches:
            p.stop()

    assert resolved == "fleet-orchestrator"


def test_decompose_fanout_assigns_root_to_board_orchestrator(kanban_home, monkeypatch):
    """The real fan-out path must persist the board owner on its root task."""
    kb.create_board("fleet")
    kb.write_board_metadata("fleet", orchestrator_profile="fleet-orchestrator")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "fleet")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="coordinate", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "one child is enough to exercise root ownership",
        "tasks": [{"title": "implement", "body": "do it", "assignee": "implementer", "parents": []}],
    })
    patches = _patch_list_profiles([
        "fleet-orchestrator", "global-orchestrator", "implementer",
    ])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"orchestrator_profile": "global-orchestrator"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
    assert root is not None
    assert root.assignee == "fleet-orchestrator"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason




def test_decompose_refuses_worked_card_before_calling_llm(kanban_home, monkeypatch):
    """Overlay K3: a card that reached triage by block-loop escalation (it has
    runs) must be refused BEFORE any LLM planning call — otherwise the tick
    re-plans and re-fails it every 60 s."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="worked", assignee="laforge")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="laforge") is not None
        kb.block_task(conn, tid, reason="stuck", kind="needs_input")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))

    import sys, types

    def _no_llm(*a, **k):
        raise AssertionError("LLM planner must not be called for a worked card")

    # decompose_task imports agent.auxiliary_client.call_llm lazily; plant a
    # module whose call_llm explodes so any planning attempt is loud.
    fake = types.ModuleType("agent.auxiliary_client")
    fake.call_llm = _no_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake)
    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not a fresh card" in outcome.reason
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.get_task(conn, tid).assignee == "laforge"
