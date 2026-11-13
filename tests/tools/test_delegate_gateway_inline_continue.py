"""A gateway session that joins its delegations in the turn still offers ``action='continue'``.

Such a session lives on after the call that joined a child, so a finished child can be given a follow-up (the
handler already serves it and the joined result already carries the child's ``subagent_id``). Only the controls
that need a RUNNING child (list/steer/stop) have nothing to reach there. A one-shot session exits after its
reply and keeps the bare schema.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _built(inline: bool, monkeypatch, *, oneshot: bool = False):
    from agent.agent_init import _load_tools
    from tools.delegate_tool_config import gateway_delegation_inline_scope

    if oneshot:
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    else:
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    agent = SimpleNamespace(platform="whatsapp", quiet_mode=True)
    with gateway_delegation_inline_scope(inline):
        _load_tools(agent, ["hermes-cli"], None)
    return agent


def _delegate(agent) -> dict:
    return next(t["function"] for t in agent.tools if t["function"]["name"] == "delegate_task")


def test_a_joined_gateway_session_is_offered_continue_and_nothing_that_needs_a_running_child(monkeypatch):
    from tools.delegate_tool import session_joins_delegation

    agent = _built(True, monkeypatch)
    function = _delegate(agent)
    properties = function["parameters"]["properties"]
    assert properties["action"]["enum"] == ["spawn", "continue"]
    assert {"subagent_id", "message"} <= set(properties)
    assert not {"child_id", "scope", "query"} & set(properties)
    control_text = json.dumps([properties[name] for name in ("action", "subagent_id", "message")])
    assert not any(word in control_text for word in ("steer", "'stop'", "'list'"))
    assert "returns in this tool call" in properties["action"]["description"]
    assert "END YOUR TURN" not in function["description"]
    assert session_joins_delegation(agent) is True  # dispatch still reads this schema as the joined variant


def test_a_one_shot_session_is_still_offered_no_action(monkeypatch):
    properties = _delegate(_built(False, monkeypatch, oneshot=True))["parameters"]["properties"]
    assert not {"action", "subagent_id", "message"} & set(properties)


def test_the_joined_schema_survives_a_pin_restore_and_a_tools_refresh(monkeypatch):
    from tools.delegate_tool import session_joins_delegation
    from tools.mcp_tool_agent import refresh_agent_mcp_tools, restore_agent_tool_prefix, tool_pin_version

    first = _built(True, monkeypatch)
    fresh = _built(False, monkeypatch)
    restore_agent_tool_prefix(fresh, {"version": tool_pin_version(), "tools": first.tools})
    refresh_agent_mcp_tools(fresh, quiet_mode=True, content_aware=True)
    assert _delegate(fresh) == _delegate(first) and session_joins_delegation(fresh) is True
    assert _delegate(fresh)["parameters"]["properties"]["action"]["enum"] == ["spawn", "continue"]


class _ScriptedModel:
    def __init__(self):
        self.turns = []

    def __call__(self, **kw):
        child = MagicMock()
        child.session_id = kw.get("session_id") or f"child-sess-{len(self.turns)}"
        child.model, child.provider = kw.get("model"), kw.get("provider")
        child._parent_session_id = kw.get("parent_session_id")
        child.get_activity_summary.return_value = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 0}
        db = kw["session_db"]

        def run_conversation(user_message, conversation_history=None, **_):
            self.turns.append({"session_id": child.session_id, "message": user_message,
                               "history": conversation_history})
            db.create_session(child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"))
            db.append_message(child.session_id, "user", user_message)
            db.append_message(child.session_id, "assistant", f"done: {user_message}")
            return {"final_response": f"done: {user_message}", "completed": True, "api_calls": 1}

        child.run_conversation.side_effect = run_conversation
        return child


def test_a_call_built_from_the_joined_schema_continues_the_child_inside_the_call(monkeypatch):
    """The model's view end to end: every argument it sends is one the joined schema offers."""
    from tools import delegate_tool

    tools = _built(True, monkeypatch).tools
    offered = set(_delegate(SimpleNamespace(tools=tools))["parameters"]["properties"])
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "whatsapp"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.session_id, parent._delegate_depth, parent._interrupt_requested = "parent-sess", 0, False
    parent.enabled_toolsets, parent.disabled_toolsets = ["terminal", "file"], []
    parent._session_db = SessionDB()
    parent._session_db.create_session("parent-sess", "whatsapp")
    parent._active_children, parent._active_children_lock = [], threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._oneshot_session, parent.tools = False, tools

    model = _ScriptedModel()
    follow_up = {"action": "continue", "subagent_id": None, "message": "add tests"}
    assert set(follow_up) <= offered
    with patch("run_agent.AIAgent", side_effect=model):
        first = json.loads(delegate_tool.delegate_task(
            tasks=[{"goal": "write parser"}], background=True, parent_agent=parent))
        follow_up["subagent_id"] = first["results"][0]["subagent_id"]
        again = json.loads(delegate_tool.delegate_task(**follow_up, background=True, parent_agent=parent))
    assert again["results"][0]["subagent_id"] == follow_up["subagent_id"]
    assert "done: add tests" in again["results"][0]["summary"]
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]
    assert model.turns[1]["history"][-1]["content"] == "done: write parser"


@pytest.mark.parametrize("action", ["list", "stop"])
def test_unadvertised_live_control_still_answers_plainly(action, monkeypatch):
    from tools import delegate_tool

    parent = MagicMock(_delegate_depth=0, session_id="sess", _oneshot_session=False, _active_children=[],
                       _active_children_lock=None, tools=_built(True, monkeypatch).tools)
    assert isinstance(json.loads(delegate_tool.delegate_task(action=action, subagent_id="nope", parent_agent=parent)),
                      dict)
