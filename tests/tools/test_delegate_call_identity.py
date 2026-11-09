"""Every relayed subagent event names its ``delegate_task`` call (``delegate_call_id``) and, when the
task ran on a route, the route's label: the gateway groups one status message per call on them."""

import types

import tools.delegate_tool as dt
from tools.delegate_tool import _build_child_progress_callback
from tools.delegate_tool_routes import TaskRoute


def _route(label=None, backend="default", error=None):
    return TaskRoute(creds={"model": "claude-opus-5-5", "provider": "anthropic", "base_url": None, "api_key": None,
                            "api_mode": None}, routing_cfg={}, label=label, backend=backend, error=error)


def _parent(events):
    return types.SimpleNamespace(
        session_id="parent-sess", _delegate_spinner=None, _subagent_id=None,
        tool_progress_callback=lambda event, tool=None, preview=None, args=None, **kw: events.append((event, kw)),
    )


def _run_call(monkeypatch, task_list, routes):
    """One ``_build_children`` call; each stub child carries a real relay to the parent callback."""
    events: list = []
    parent = _parent(events)

    def fake_build(*, task_index, goal, parent_agent, task_count, **_kw):
        ref: dict = {}
        relay = _build_child_progress_callback(task_index, goal, parent_agent, task_count,
                                               subagent_id=f"sa-{task_index}-{id(ref)}", depth=0, session_ref=ref)
        return types.SimpleNamespace(_progress_identity_ref=ref, tool_progress_callback=relay)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
    children, err = dt._build_children(
        task_list, [None] * len(task_list), routes, top_role="leaf", max_iterations=5, parent_agent=parent,
        live_deleg_id=None, live_writers=[],
    )
    assert err is None
    for _i, task, child in children:
        child.tool_progress_callback("subagent.start", preview=task["goal"])
    return [kw for event, kw in events if event == "subagent.start"]


def test_children_of_one_call_share_a_call_id_and_calls_differ(monkeypatch):
    first = _run_call(monkeypatch, [{"goal": "a"}, {"goal": "b"}], [_route(), _route()])
    second = _run_call(monkeypatch, [{"goal": "c"}], [_route()])
    ids_first = {kw.get("delegate_call_id") for kw in first}
    assert len(first) == 2 and len(ids_first) == 1 and None not in ids_first
    (id_second,) = {kw.get("delegate_call_id") for kw in second}
    assert id_second and id_second not in ids_first


def test_route_label_only_when_the_task_has_a_route(monkeypatch):
    events = _run_call(monkeypatch, [{"goal": "routed"}, {"goal": "plain"}], [_route(label="flash", backend="ollama"), _route()])
    by_goal = {kw["goal"]: kw for kw in events}
    assert by_goal["routed"]["route_label"] == "flash"
    assert "route_label" not in by_goal["plain"]


def test_events_carry_the_number_of_children_the_call_built(monkeypatch):
    """A route-refused task builds no child: the call's events count only the built ones."""
    events = _run_call(monkeypatch, [{"goal": "a"}, {"goal": "refused"}, {"goal": "b"}],
                       [_route(), _route(backend="ollama", error="no route"), _route()])
    assert sorted(kw["goal"] for kw in events) == ["a", "b"]
    assert {kw.get("delegate_call_children") for kw in events} == {"2"}
    assert {kw["task_count"] for kw in events} == {3}, "the batch's i/N slot is unchanged"
