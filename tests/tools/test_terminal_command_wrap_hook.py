"""The ``wrap_terminal_command`` hook: guards see the raw command, the hook decides what runs.

Harness notes: the environment is a mock, ``_check_all_guards`` is a recorder that approves
(or returns the test's verdict), ``spawn_background_process`` is a recorder, and
``iter_hook_callbacks`` yields only the test's callbacks for ``wrap_terminal_command``.
"""
import json
from unittest.mock import MagicMock

import pytest

import hermes_cli.plugins as plugins_mod
import tools.terminal_tool as terminal_tool_module

RAW = "touch /x/probe"
WRAPPED = "/usr/bin/bwrap --ro-bind / / --new-session -- /bin/bash -c 'touch /x/probe'"


def _env_config(tmp_path):
    return {
        "env_type": "local",
        "timeout": 30,
        "cwd": str(tmp_path),
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = {"guarded": [], "hook_calls": [], "spawned": []}
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: _env_config(tmp_path))
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)

    def guard(command, *_args, **_kwargs):
        state["guarded"].append(command)
        return state.get("approval") or {"approved": True}

    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", guard)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", mock_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    def spawn(**kwargs):
        state["spawned"].append(kwargs["command"])
        return json.dumps({"output": "", "exit_code": 0, "error": None, "session_id": "proc_x"})

    monkeypatch.setattr(terminal_tool_module, "spawn_background_process", spawn)

    def use(*callbacks):
        monkeypatch.setattr(
            plugins_mod,
            "iter_hook_callbacks",
            lambda name: tuple(callbacks) if name == "wrap_terminal_command" else (),
        )

    state["env"] = mock_env
    state["use"] = use
    return state


def _record(state, value):
    def callback(**kwargs):
        state["hook_calls"].append(kwargs)
        if isinstance(value, Exception):
            raise value
        return value

    return callback


def _run(**kwargs):
    return json.loads(terminal_tool_module.terminal_tool(command=RAW, **kwargs))


def test_the_hook_name_is_a_plugin_hook_and_not_a_shell_hook():
    assert "wrap_terminal_command" in plugins_mod.VALID_HOOKS
    assert "wrap_terminal_command" in plugins_mod.SHELL_UNSUPPORTED_HOOKS


def test_without_a_callback_the_raw_command_runs(harness):
    harness["use"]()

    _run()

    assert harness["env"].execute.call_args.args[0] == RAW


def test_the_callback_wraps_what_runs_and_the_guard_sees_only_the_raw_command(harness):
    harness["use"](_record(harness, WRAPPED))

    _run()

    assert harness["guarded"] == [RAW]
    assert harness["env"].execute.call_args.args[0] == WRAPPED


def test_a_guard_that_does_not_approve_never_reaches_the_callback(harness):
    harness["approval"] = {"approved": False, "description": "denied by test"}
    harness["use"](_record(harness, WRAPPED))

    _run()

    assert harness["hook_calls"] == []
    assert not harness["env"].execute.called


def test_a_raising_callback_refuses_the_command(harness):
    harness["use"](_record(harness, RuntimeError("boom")))

    result = _run()

    assert not harness["env"].execute.called
    assert "wrap_terminal_command" in json.dumps(result)


def test_a_block_directive_refuses_the_command_with_its_message(harness):
    harness["use"](_record(harness, {"action": "block", "message": "sandbox is down"}))

    result = _run()

    assert not harness["env"].execute.called
    assert "sandbox is down" in json.dumps(result)


def test_two_callbacks_that_disagree_refuse_the_command(harness):
    harness["use"](_record(harness, WRAPPED), _record(harness, WRAPPED + " "))

    _run()

    assert not harness["env"].execute.called


def test_a_callback_that_returns_a_non_string_refuses_the_command(harness):
    harness["use"](_record(harness, 42))

    _run()

    assert not harness["env"].execute.called


def test_a_callback_that_returns_an_empty_string_refuses_the_command(harness):
    harness["use"](_record(harness, ""))

    _run()

    assert not harness["env"].execute.called


def test_a_background_call_spawns_the_wrapped_command(harness):
    harness["use"](_record(harness, WRAPPED))

    _run(background=True, notify_on_complete=True)

    assert harness["spawned"] == [WRAPPED]


def test_the_callback_receives_the_raw_command_and_the_call_context(harness):
    harness["use"](_record(harness, WRAPPED))

    _run()

    assert harness["hook_calls"][0]["command"] == RAW
    assert harness["hook_calls"][0]["env_type"] == "local"
    assert harness["hook_calls"][0]["background"] is False
