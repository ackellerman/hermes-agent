"""The opt-in ``pre_final_stop`` plugin seam on the real stop-gate path.

A plugin registered for ``pre_final_stop`` is asked once per text stop, after every existing gate
(late LSP, verify-on-stop, ``pre_verify``, kanban), whether the turn should get one more model call.
Hermes hands it identity and per-call outcomes only (never tool arguments or results), grants at
most one continuation per turn, and treats anything but an explicit ``continue`` as "finish".

Imports of the seam's own names live inside each test so a tree without the seam fails the test
(rc 1), not collection.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent

NUDGE = "Resume the follow-up, or state the real blocker and stop."
SECRET_ARG = "SECRET-ARGUMENT-TEXT-7f3a"


def _text(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model", usage=None,
    )


def _tool_calls(*calls):
    tool_calls = [
        SimpleNamespace(id=call_id, type="function",
                        function=SimpleNamespace(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
        for call_id, name, args in calls
    ]
    message = SimpleNamespace(content="", tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], model="test/model", usage=None,
    )


_TODO_DEF = {
    "type": "function",
    "function": {"name": "todo_list", "description": "todo",
                 "parameters": {"type": "object", "properties": {}}},
}


_READ_FILE_DEF = {
    "type": "function",
    "function": {"name": "read_file", "description": "read",
                 "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
}


def _make_agent(session_id, max_iterations, tool_defs=(_TODO_DEF,)):
    with (
        patch("model_tools.get_tool_definitions", return_value=list(tool_defs)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id=session_id, api_key="test-key", base_url="https://example.invalid/v1",
            provider="openai-compat", model="test/model", max_iterations=max_iterations,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance.diagnostics = []
    instance._emit_diagnostic_status = lambda message: instance.diagnostics.append(str(message))
    return instance


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return _make_agent


@pytest.fixture
def plugins(monkeypatch):
    """A real PluginManager carrying only the callbacks a test registers."""
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    def register(hook_name, callback):
        manager._hooks.setdefault(hook_name, []).append(callback)
        return callback

    return SimpleNamespace(manager=manager, register=register)


def _script(agent, *responses):
    """Serve ``responses`` in order; record the messages each model call was sent."""
    queue = list(responses)
    sent = []

    def call(api_kwargs):
        sent.append([dict(m) for m in api_kwargs.get("messages", [])])
        item = queue.pop(0)
        return item(agent) if callable(item) else item

    agent._interruptible_api_call = call
    return sent


def _nudges_in(messages):
    return sum(1 for m in messages if m.get("role") == "user" and m.get("content") == NUDGE)


def _always_continue(seen):
    def hook(**kwargs):
        seen.append(kwargs)
        return {"action": "continue", "message": NUDGE}
    return hook


def test_no_registered_plugin_changes_nothing(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK
    from hermes_cli.plugins import VALID_HOOKS

    assert PRE_FINAL_STOP_HOOK in VALID_HOOKS
    agent = make_agent("seam-off", 4)
    sent = _script(agent, _text("only answer"))

    result = agent.run_conversation("plan the thing")

    assert result["final_response"] == "only answer"
    assert result["completed"] is True
    assert len(sent) == 1
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]
    assert result["messages"][-1].get("finish_reason") != "verification_required"


def test_no_edit_turn_gets_exactly_one_continuation_then_exhaustion_is_visible(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen = []
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))
    agent = make_agent("seam-once", 6)
    sent = _script(agent, _text("premature stop"), _text("second stop"), _text("never requested"))

    result = agent.run_conversation("plan the thing")

    assert not agent._turn_file_mutation_paths  # no edit: pre_verify would never have fired
    assert len(sent) == 2
    assert _nudges_in(sent[-1]) == 1
    assert result["final_response"] == "second stop"
    assert result["completed"] is True
    assert [kw["continuation_available"] for kw in seen] == [True, False]
    assert any("exhausted" in line for line in agent.diagnostics), agent.diagnostics


def test_payload_is_immutable_identity_and_raw_outcomes_without_arguments_or_results(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen, turn_ids, events = [], [], []
    plugins.register("post_tool_call", lambda **kw: events.append(kw["tool_call_id"]))
    plugins.register("pre_tool_call", lambda **kw: {"action": "block", "message": "denied by policy"}
                     if kw.get("tool_call_id") == "call-denied" else None)
    plugins.register(PRE_FINAL_STOP_HOOK, lambda **kw: seen.append(kw) or turn_ids.append(agent._current_turn_id))
    agent = make_agent("seam-payload", 6)
    _script(
        agent,
        _tool_calls(("call-ok", "todo_list", {"todos": [{"id": "1", "content": SECRET_ARG, "status": "pending"}]})),
        _tool_calls(("call-denied", "todo_list", {"todos": [{"id": "2", "content": SECRET_ARG}]})),
        _text("done"),
    )

    result = agent.run_conversation("track it")

    assert result["final_response"] == "done"
    assert events == ["call-ok", "call-denied"]
    [payload] = seen
    assert payload["session_id"] == "seam-payload"
    assert payload["turn_id"] == turn_ids[0] and payload["turn_id"]
    assert payload["complete"] is True
    calls = payload["calls"]
    assert isinstance(calls, tuple)
    assert [(c.tool_name, c.call_id, c.turn_id, c.status) for c in calls] == [
        ("todo_list", "call-ok", payload["turn_id"], "ok"),
        ("todo_list", "call-denied", payload["turn_id"], "blocked"),
    ]
    assert calls[0].error_type is None and calls[0].error_message is None
    assert calls[1].error_type == "plugin_block" and "denied by policy" in calls[1].error_message
    with pytest.raises(AttributeError):
        calls[0].status = "ok"  # frozen
    assert SECRET_ARG not in repr(payload)
    assert not {"args", "result", "arguments", "messages"} & set(payload)


def test_missed_post_tool_event_is_reported_incomplete(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK
    from model_tools import suppress_post_tool_call_hook

    seen = []
    plugins.register("post_tool_call", lambda **kw: None)
    plugins.register(PRE_FINAL_STOP_HOOK, lambda **kw: seen.append(kw))
    agent = make_agent("seam-missed", 6)
    _script(agent, _tool_calls(("call-1", "todo_list", {"todos": []})), _text("done"))

    with suppress_post_tool_call_hook():
        agent.run_conversation("track it")

    [payload] = seen
    assert payload["complete"] is False
    [call] = payload["calls"]
    assert (call.call_id, call.tool_name, call.status) == ("call-1", "todo_list", None)


@pytest.mark.parametrize("outcome", ["none", "raise", "bare-string", "no-message", "unknown-action"])
def test_anything_but_an_explicit_continue_finishes_the_turn(make_agent, plugins, outcome):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    results = {
        "none": None, "bare-string": "continue", "no-message": {"action": "continue"},
        "unknown-action": {"action": "maybe", "message": NUDGE},
    }

    def hook(**_kw):
        if outcome == "raise":
            raise RuntimeError("domain plugin broke")
        return results[outcome]

    plugins.register(PRE_FINAL_STOP_HOOK, hook)
    agent = make_agent("seam-unknown", 6)
    sent = _script(agent, _text("final"), _text("never requested"))

    result = agent.run_conversation("plan the thing")

    assert len(sent) == 1
    assert result["final_response"] == "final"
    if outcome == "raise":
        assert any("raised or timed out" in line for line in agent.diagnostics), agent.diagnostics
    elif outcome != "none":
        assert any("pre_final_stop" in line for line in agent.diagnostics), agent.diagnostics


def test_plugin_diagnostic_is_surfaced_without_continuing(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    plugins.register(PRE_FINAL_STOP_HOOK,
                     lambda **_kw: {"action": "stop", "diagnostic": "Q-1 round 2: continuation skipped; evidence unknown"})
    agent = make_agent("seam-diag", 6)
    sent = _script(agent, _text("final"))

    agent.run_conversation("plan the thing")

    assert len(sent) == 1
    assert any("continuation skipped; evidence unknown" in line for line in agent.diagnostics)


def test_guardrail_halt_suppresses_the_seam(make_agent, plugins):
    from agent.tool_guardrails import ToolGuardrailDecision
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen = []
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))
    agent = make_agent("seam-guardrail", 6)

    def halted_answer(agent_):
        agent_._tool_guardrail_halt_decision = ToolGuardrailDecision(action="halt", code="loop", tool_name="todo_list")
        return _text("halted answer")

    sent = _script(agent, halted_answer, _text("never requested"))

    result = agent.run_conversation("plan the thing")

    assert seen == []
    assert len(sent) == 1
    assert result["final_response"] == "halted answer"


def test_cap_resets_per_turn_and_calls_never_leak_across_turns_or_sessions(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen = []
    plugins.register("post_tool_call", lambda **kw: None)
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))
    first = make_agent("seam-session-a", 8)
    _script(first, _tool_calls(("call-a", "todo_list", {"todos": []})), _text("a1"), _text("a2"))
    first.run_conversation("turn one")
    sent = _script(first, _text("b1"), _text("b2"), _text("never requested"))
    second_turn = first.run_conversation("turn two")
    other = make_agent("seam-session-b", 8)
    _script(other, _text("c1"), _text("c2"))
    other.run_conversation("other session")

    assert len(sent) == 2 and second_turn["final_response"] == "b2"
    by_turn = {}
    for kw in seen:
        by_turn.setdefault((kw["session_id"], kw["turn_id"]), []).append(kw)
    assert [len(v) for v in by_turn.values()] == [2, 2, 2]
    turn_one, turn_two, turn_other = by_turn.values()
    assert [c.call_id for c in turn_one[0]["calls"]] == ["call-a"]
    assert turn_two[0]["calls"] == () and turn_two[0]["complete"] is True
    assert turn_other[0]["calls"] == () and turn_other[0]["session_id"] == "seam-session-b"
    for kws in by_turn.values():
        assert [kw["continuation_available"] for kw in kws] == [True, False]


def test_candidate_is_a_provisional_verification_row_superseded_by_the_final(make_agent, plugins):
    from agent.agent_runtime_helpers import _merge_consecutive_assistants
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue([]))
    agent = make_agent("seam-supersede", 6)
    interims = []
    agent._emit_interim_assistant_message = lambda msg: interims.append(dict(msg))
    _script(agent, _text("premature stop"), _text("real final"))

    result = agent.run_conversation("plan the thing")

    assert [m["content"] for m in interims] == ["premature stop"]
    assert interims[0]["finish_reason"] == "verification_required"
    assert _nudges_in(result["messages"]) == 0  # the nudge is scaffolding, never transcript
    candidate = next(m for m in result["messages"] if m.get("content") == "premature stop")
    assert candidate["finish_reason"] == "verification_required"
    merged, _ = _merge_consecutive_assistants(result["messages"])
    assert [m.get("content") for m in merged if m.get("role") == "assistant"] == ["real final"]


def test_budget_fallback_keeps_the_composed_candidate(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue([]))
    agent = make_agent("seam-budget", 1)
    emitted = []
    agent.interim_assistant_callback = lambda text, **_kw: emitted.append(text)
    agent._handle_max_iterations = lambda *_a, **_kw: pytest.fail("no summary call: the candidate is kept")
    _script(agent, _text("composed report"))

    result = agent.run_conversation("plan the thing")

    assert emitted == ["composed report"]  # shown as an interim, like a verify-on-stop candidate
    assert result["final_response"] == "composed report"
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_with_verify_on_stop_the_seam_runs_after_it_and_only_once(make_agent, plugins, monkeypatch):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen = []
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    agent = make_agent("seam-verify", 8)
    sent = _script(agent, _text("v1"), _text("v2"), _text("v3"), _text("never requested"))

    with patch("agent.verification_stop.build_verify_on_stop_nudge", side_effect=["verify it", None, None]):
        result = agent.run_conversation("edit and plan")

    assert len(sent) == 3
    assert _nudges_in(sent[-1]) == 1
    assert len(seen) == 2  # v1 was taken by verify-on-stop, which runs first
    assert [kw["continuation_available"] for kw in seen] == [True, False]
    assert result["final_response"] == "v3"


def test_pre_verify_and_kanban_interleave_without_a_second_seam_opportunity(make_agent, plugins, monkeypatch):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen, pre_verify_calls = [], []
    stop = {"n": 0}
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))

    def pre_verify(**kw):
        pre_verify_calls.append(stop["n"])
        return {"action": "continue", "message": "run the tests"} if stop["n"] == 3 else None

    plugins.register("pre_verify", pre_verify)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_seam")

    def answer(text, *, edit=False):
        def produce(agent_):
            stop["n"] += 1
            if edit:
                agent_._turn_file_mutation_paths = {"changed.py"}
            return _text(text)
        return produce

    agent = make_agent("seam-kanban", 10)
    sent = _script(agent, answer("k1"), answer("k2"), answer("k3", edit=True), answer("k4"),
                   answer("never requested"))

    def kanban(messages, attempts=0):
        return "call kanban_complete" if stop["n"] == 2 else None

    with patch("agent.kanban_stop.build_kanban_stop_nudge", side_effect=kanban):
        result = agent.run_conversation("work the card")

    # k1: seam (one opportunity) · k2: kanban · k3: pre_verify · k4: seam spent → final.
    assert len(sent) == 4
    assert _nudges_in(sent[-1]) == 1
    assert [kw["continuation_available"] for kw in seen] == [True, False]
    assert 3 in pre_verify_calls
    assert result["final_response"] == "k4"


def test_shell_hooks_are_refused_and_the_hooks_test_payload_matches_runtime(make_agent, plugins):
    from agent.shell_hooks import _parse_hooks
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK
    from hermes_cli.hooks import _DEFAULT_PAYLOADS
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS

    assert PRE_FINAL_STOP_HOOK in SHELL_UNSUPPORTED_HOOKS
    specs, complete = _parse_hooks({PRE_FINAL_STOP_HOOK: [{"command": "/bin/true"}]})
    assert specs == [] and complete is True

    seen = []
    plugins.register("post_tool_call", lambda **kw: None)
    plugins.register(PRE_FINAL_STOP_HOOK, lambda **kw: seen.append(kw))
    agent = make_agent("seam-hooks-test", 6)
    _script(agent, _tool_calls(("call-1", "todo_list", {"todos": []})), _text("done"))
    agent.run_conversation("track it")

    [runtime] = seen
    sample = _DEFAULT_PAYLOADS[PRE_FINAL_STOP_HOOK]
    json.dumps(sample)
    assert set(sample) == set(runtime) - {"telemetry_schema_version"}
    [runtime_call] = runtime["calls"]
    assert set(sample["calls"][0]) == {f.name for f in dataclasses.fields(runtime_call)}


def test_a_hung_hook_is_bounded_and_finishes_the_turn_visibly(make_agent, plugins, monkeypatch):
    import hermes_cli.plugins as plugins_mod
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.2)
    never_set = threading.Event()

    def hung(**_kw):
        never_set.wait()  # never returns: only the hook bound can release the turn
        return {"action": "continue", "message": NUDGE}

    plugins.register(PRE_FINAL_STOP_HOOK, hung)
    agent = make_agent("seam-timeout", 6)
    sent = _script(agent, _text("final"), _text("never requested"))
    outcome = {}
    turn = threading.Thread(target=lambda: outcome.update(agent.run_conversation("plan the thing")), daemon=True)

    turn.start()
    turn.join(timeout=60)  # returns as soon as the turn ends; an unbounded hook holds it forever

    assert not turn.is_alive(), "a hung pre_final_stop callback held the final answer"
    assert len(sent) == 1 and outcome["final_response"] == "final"
    assert any("raised or timed out" in line for line in agent.diagnostics), agent.diagnostics


def test_a_failing_seam_dispatch_finishes_the_turn_visibly(make_agent, plugins, monkeypatch):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue([]))
    real_invoke = plugins.manager.invoke_hook

    def invoke(hook_name, **kwargs):
        if hook_name == PRE_FINAL_STOP_HOOK:
            raise RuntimeError("dispatch broke")
        return real_invoke(hook_name, **kwargs)

    monkeypatch.setattr(plugins.manager, "invoke_hook", invoke)
    agent = make_agent("seam-dispatch-broke", 6)
    sent = _script(agent, _text("final"), _text("never requested"))

    result = agent.run_conversation("plan the thing")

    assert len(sent) == 1 and result["final_response"] == "final"
    assert any("pre_final_stop: hook check failed" in line for line in agent.diagnostics), agent.diagnostics


def test_interrupt_suppresses_the_seam(make_agent, plugins):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    seen = []
    plugins.register(PRE_FINAL_STOP_HOOK, _always_continue(seen))
    agent = make_agent("seam-interrupt", 6)

    def interrupted_answer(agent_):
        agent_._interrupt_requested = True
        return _text("interrupted answer")

    sent = _script(agent, interrupted_answer, _text("never requested"))

    agent.run_conversation("plan the thing")

    assert seen == []
    assert len(sent) == 1


def test_the_event_ledger_stays_empty_while_no_pre_final_stop_callback_is_registered(make_agent, plugins):
    from hermes_cli.plugins_dispatch import post_tool_events

    turns = []
    plugins.register("post_tool_call", lambda **kw: turns.append((kw["session_id"], kw["turn_id"])))
    agent = make_agent("seam-ledger-off", 6)
    _script(agent, _tool_calls(("call-1", "todo_list", {"todos": []})), _text("done"))

    agent.run_conversation("track it")

    [(session_id, turn_id)] = turns
    assert post_tool_events(session_id, turn_id) == ((), False)


def test_parallel_calls_through_the_concurrent_executor_are_witnessed(make_agent, plugins, tmp_path):
    from agent.turn_stop_gates import PRE_FINAL_STOP_HOOK

    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("a")
    second.write_text("b")
    seen = []
    plugins.register("post_tool_call", lambda **kw: None)
    plugins.register(PRE_FINAL_STOP_HOOK, lambda **kw: seen.append(kw))
    agent = make_agent("seam-parallel", 6, tool_defs=(_READ_FILE_DEF,))
    _script(agent, _tool_calls(("p1", "read_file", {"path": str(first)}), ("p2", "read_file", {"path": str(second)})),
            _text("done"))

    agent.run_conversation("read both")

    [payload] = seen
    assert payload["complete"] is True
    assert [(c.call_id, c.status) for c in payload["calls"]] == [("p1", "ok"), ("p2", "ok")]


@pytest.fixture
def ledger(monkeypatch):
    """A fresh post-tool event ledger, plus ``complete(dispatched, events)`` for one session turn."""
    import hermes_cli.plugins_dispatch as dispatch
    from agent.turn_stop_gates import _pre_final_call_outcomes

    monkeypatch.setattr(dispatch, "_tool_events", {})
    monkeypatch.setattr(dispatch, "_tool_events_overflowed", set())

    def note(turn_id, call_id, **fields):
        dispatch._note_post_tool_event(
            {"session_id": "S", "turn_id": turn_id, "tool_call_id": call_id, "tool_name": "t", "status": "ok", **fields})

    def complete(dispatched, events, turn_id="T"):
        for call_id in events:
            note(turn_id, call_id)
        agent = SimpleNamespace(_pre_final_dispatched_calls=tuple((c, "t", turn_id) for c in dispatched))
        return _pre_final_call_outcomes(agent, "S", turn_id)[1]

    return SimpleNamespace(module=dispatch, note=note, complete=complete)


@pytest.mark.parametrize("dispatched,events,expected", [
    (["c1", "c2"], ["c1", "c2"], True),
    (["c1", "c2"], ["c1"], False),        # a missed event
    (["c1"], ["c1", "c1"], False),        # two events for one call
    (["c1"], ["c1", "c-other"], False),   # an event naming a call this turn did not dispatch
    ([""], [""], False),                  # a call without an id cannot be witnessed
], ids=["exact", "missed", "duplicate", "unknown-call", "empty-id"])
def test_complete_needs_exactly_one_event_per_dispatched_call(ledger, dispatched, events, expected):
    assert ledger.complete(dispatched, events) is expected


def test_a_turn_whose_events_overflowed_the_ledger_is_incomplete(ledger, monkeypatch):
    monkeypatch.setattr(ledger.module, "_TOOL_EVENT_MAX_PER_TURN", 2)

    # The third event is dropped; without the overflow flag the two kept events would match.
    assert ledger.complete(["c1", "c2"], ["c1", "c2", "c1"]) is False
    assert ledger.module.post_tool_events("S", "T")[1] is True


def test_a_turn_evicted_from_the_ledger_reads_incomplete(ledger, monkeypatch):
    monkeypatch.setattr(ledger.module, "_TOOL_EVENT_MAX_TURNS", 3)

    assert ledger.complete(["c1"], ["c1"], turn_id="T0") is True
    for n in range(4):
        ledger.note(f"later-{n}", "z")

    assert ledger.complete(["c1"], [], turn_id="T0") is False


def test_a_stored_error_message_is_capped(ledger):
    ledger.note("T", "c-long", status="error", error_message="x" * 50_000)
    ledger.note("T", "c-short", status="error", error_message="old_string not found")

    long_event, short_event = ledger.module.post_tool_events("S", "T")[0]
    assert long_event["error_message"].startswith("x" * 2000)
    assert len(long_event["error_message"]) <= 2100 and "truncated" in long_event["error_message"]
    assert short_event["error_message"] == "old_string not found"
