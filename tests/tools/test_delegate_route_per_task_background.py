"""Every spawned task keeps its own delegation route, on both dispatch legs.

``delegate_task`` resolves one route per task through ``resolve_task_routes``/``_route_one``
(``tools/delegate_tool.py`` ``_routes_for`` -> ``_build_children``) and the batch-wide ``creds`` is
a label only, blanked when the tasks' routes differ (``delegate_tool.py``'s ``runnable``/``creds``
line; that blanking is not asserted here — this test guards the per-task model, endpoint and effort
handoff). This is the regression guard for that handoff: a backend that routes task 0 to its own
model and task 1 to nothing (the parent's route) must hand each child the model, endpoint, effort
and route report of ITS task, both when the call runs sync and when it is dispatched with
``background=True``. Children share one ``_build_children`` call before
``_run_batch``/``_dispatch_background``, so one batch-wide model would show up on both legs.

Child construction and the child run are stubbed where this test reads what they were handed, and
``background=True`` stops at the async registry's admission boundary: no provider call, worker
thread or real delegation runs. The backend itself is injected at ``_active_provider`` and the
``delegation`` config read is pinned (``_load_config``), so what runs for real is the per-task route
machinery down to ``_build_children``, with the backend's route answers supplied by the fixture.
"""

import json
import threading
import types
from unittest.mock import MagicMock

import pytest

from agent.delegation_provider import DelegationProvider

ROUTE_MODEL = "route-x-model"
ROUTE = {
    "base_url": "https://route.example.invalid/v1", "api_key": "k-route-x", "model": ROUTE_MODEL,
    "reasoning_effort": "high", "label": "x",
}


class _PerTaskBackend(DelegationProvider):
    """Task 0 gets its own route; task 1 gets ``dict(config)``, i.e. nothing — the parent's route."""

    @property
    def name(self):
        return "per-task-probe"

    def route_task(self, task, config):
        return dict(ROUTE) if task["index"] == 0 else dict(config)


def _parent():
    parent = MagicMock()
    parent.base_url, parent.provider = "https://openrouter.ai/api/v1", "openrouter"
    parent.api_key, parent.api_mode = "parent-key", "chat_completions"
    parent.model, parent.platform = "parent-model", "cli"
    parent.session_id = "root"
    parent.request_overrides = {}
    parent.reasoning_config = {"enabled": True, "effort": "medium"}
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._delegate_spinner = None
    parent._safe_print = lambda line: None
    parent.tools = None  # not the joined delegate_task variant: background really dispatches
    parent._oneshot_session = False
    parent._interrupt_requested = False
    return parent


@pytest.fixture
def built(monkeypatch):
    """``([child construction kwargs], [built children])`` for one delegate_task call."""
    import tools.delegate_tool as dt

    calls, children = [], []

    def build(**kwargs):
        calls.append(kwargs)
        # ``model=None`` is the inherit contract: that child runs on the parent's model.
        child = types.SimpleNamespace(tool_progress_callback=None,
                                      model=kwargs.get("model") or kwargs["parent_agent"].model)
        children.append(child)
        return child

    def run_child(task_index, goal, child=None, parent_agent=None, **kw):
        return {"task_index": task_index, "status": "completed", "summary": goal,
                "api_calls": 1, "duration_seconds": 0.0}

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(dt, "_run_single_child", run_child)
    return calls, children


def _spawn(background, monkeypatch):
    """Run one two-task call; the background leg stops at async admission. ``(payload, admitted units)``."""
    import tools.delegate_tool as dt
    from tools import async_delegation

    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr("tools.delegate_tool_routes._active_provider", lambda config: _PerTaskBackend())
    admitted = []

    def admit(**kwargs):
        admitted.append(kwargs)
        return {"status": "dispatched", "delegation_id": kwargs["delegation_id"] or "unit-id"}

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", admit)
    if background:
        monkeypatch.setattr("tools.delegate_tool_config._get_independent_completions", lambda: False)
        monkeypatch.setattr("tools.delegate_tool_dispatch._resolve_async_wake_sid", lambda *a, **k: "")

    payload = json.loads(dt.delegate_task(
        tasks=[{"goal": "the routed one"}, {"goal": "the inherited one"}],
        background=background, parent_agent=_parent(),
    ))
    return payload, admitted


@pytest.mark.parametrize("background", [False, True], ids=["sync", "background"])
def test_each_task_takes_its_own_route(built, background, monkeypatch):
    calls, children = built
    payload, admitted = _spawn(background, monkeypatch)

    # The dispatch leg really is the one under test (never a silent sync fallback).
    if background:
        assert payload["status"] == "dispatched" and payload["mode"] == "background"
        assert len(admitted) == 1 and admitted[0]["task_indexes"] is None
    else:
        assert [r["status"] for r in payload["results"]] == ["completed", "completed"]

    routed, inherited = calls
    # Task 0: the backend's own model, endpoint, key and effort — not a batch-wide value.
    assert routed["model"] == ROUTE_MODEL
    assert (routed["override_base_url"], routed["override_api_key"]) == (ROUTE["base_url"], ROUTE["api_key"])
    assert routed["override_reasoning_effort"] == "high"
    # Task 1: nothing routed, so it inherits the parent's model and endpoint.
    assert inherited["model"] is None
    assert inherited["override_base_url"] is None and inherited["override_provider"] is None
    assert inherited["override_api_key"] is None
    assert inherited["override_reasoning_effort"] is None

    # What each child actually runs on: route X, and the parent's own model.
    assert [child.model for child in children] == [ROUTE_MODEL, "parent-model"]
    # The per-task route objects reached the children (labels only carry the backend here).
    assert children[0]._delegate_route == {"backend": "per-task-probe", "label": "x"}
    assert children[1]._delegate_route == {"backend": "per-task-probe"}
