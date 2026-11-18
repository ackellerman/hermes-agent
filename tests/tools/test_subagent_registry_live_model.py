"""list_active_subagents() — the child's live route, not only the route stamped at spawn.

The registry copy has never carried the live agent (``agent`` is one of
``_PRIVATE_RECORD_KEYS``), so a parent-side status line could only ever show the
model a child was spawned with.  These tests pin the additive keys the copy gains:
``live_model`` / ``live_provider`` (read off the live agent while it runs) and
``fallback_from`` (the record's spawn model, present only while the agent reports a
provider fallback in force).  ``model`` stays the spawn model, because
``_resume_spec()`` rebuilds a continued child from it.

The stub agent is a plain class, not a ``MagicMock``: a missing attribute must be
really missing (a MagicMock answers anything, and its unset attributes are truthy).
"""

from unittest.mock import MagicMock

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.delegate_tool import (
    _register_subagent,
    _unregister_subagent,
    list_active_subagents,
)

_MISSING = object()

SPAWN_MODEL = "claude-opus-5-5"
LIVE_MODEL = "glm-5.3"
LIVE_PROVIDER = "ollama-cloud"


class _StubAgent:
    """A child stand-in whose attributes are really present or really absent."""

    def __init__(self, model, provider, *, fallback_active=_MISSING):
        self.model = model
        self.provider = provider
        if fallback_active is not _MISSING:
            self._provider_fallback_active = fallback_active


def _with_registered(sid: str, *, model: str, agent) -> None:
    _register_subagent(
        {
            "subagent_id": sid,
            "parent_id": "root",
            "depth": 1,
            "goal": "test goal",
            "status": "running",
            "model": model,
            "agent": agent,
        }
    )


def _snapshot(sid: str) -> dict:
    return next(item for item in list_active_subagents() if item["subagent_id"] == sid)


def test_a_child_on_a_fallback_reports_the_route_it_runs_on():
    agent = _StubAgent(LIVE_MODEL, LIVE_PROVIDER, fallback_active=True)
    _with_registered("sid-live-fallback", model=SPAWN_MODEL, agent=agent)
    try:
        snapshot = _snapshot("sid-live-fallback")
        assert snapshot["live_model"] == LIVE_MODEL
        assert snapshot["live_provider"] == LIVE_PROVIDER
        assert snapshot["fallback_from"] == SPAWN_MODEL
        assert snapshot["model"] == SPAWN_MODEL
    finally:
        _unregister_subagent("sid-live-fallback", agent=agent)


def test_a_child_on_its_primary_reports_no_fallback_from():
    stated = _StubAgent(LIVE_MODEL, LIVE_PROVIDER, fallback_active=False)
    _with_registered("sid-live-primary", model=SPAWN_MODEL, agent=stated)
    try:
        snapshot = _snapshot("sid-live-primary")
        assert snapshot["live_model"] == LIVE_MODEL
        assert snapshot["live_provider"] == LIVE_PROVIDER
        assert "fallback_from" not in snapshot
    finally:
        _unregister_subagent("sid-live-primary", agent=stated)

    # Nothing sets the flag on a fresh agent, so the attribute is absent on the
    # common path; the reader's getattr default has to cover it.
    unstated = _StubAgent(SPAWN_MODEL, "anthropic")
    _with_registered("sid-live-primary-unstated", model=SPAWN_MODEL, agent=unstated)
    try:
        snapshot = _snapshot("sid-live-primary-unstated")
        assert snapshot["live_model"] == SPAWN_MODEL
        assert snapshot["live_provider"] == "anthropic"
        assert "fallback_from" not in snapshot
    finally:
        _unregister_subagent("sid-live-primary-unstated", agent=unstated)


def test_a_non_string_route_is_not_reported():
    agent = MagicMock()
    agent.model = MagicMock()
    agent.provider = ""
    # A MagicMock's unset attributes are truthy, so the flag must be stated.
    agent._provider_fallback_active = False
    _with_registered("sid-live-non-string", model=SPAWN_MODEL, agent=agent)
    try:
        snapshot = _snapshot("sid-live-non-string")
        assert "live_model" not in snapshot
        assert "live_provider" not in snapshot
        assert "fallback_from" not in snapshot
    finally:
        _unregister_subagent("sid-live-non-string", agent=agent)


def test_a_record_without_an_agent_gains_no_new_keys():
    _with_registered("sid-live-no-agent", model=SPAWN_MODEL, agent=None)
    try:
        snapshot = _snapshot("sid-live-no-agent")
        assert snapshot["model"] == SPAWN_MODEL
        assert "live_model" not in snapshot
        assert "live_provider" not in snapshot
        assert "fallback_from" not in snapshot
    finally:
        _unregister_subagent("sid-live-no-agent", agent=None)


def test_the_live_route_is_readable_under_a_home_override(tmp_path):
    agent = _StubAgent(LIVE_MODEL, LIVE_PROVIDER, fallback_active=True)
    _with_registered("sid-live-home-override", model=SPAWN_MODEL, agent=agent)
    token = set_hermes_home_override(str(tmp_path))
    try:
        snapshot = _snapshot("sid-live-home-override")
        assert snapshot["live_model"] == LIVE_MODEL
        assert snapshot["live_provider"] == LIVE_PROVIDER
        assert snapshot["fallback_from"] == SPAWN_MODEL
        assert snapshot["model"] == SPAWN_MODEL
    finally:
        reset_hermes_home_override(token)
        _unregister_subagent("sid-live-home-override", agent=agent)
