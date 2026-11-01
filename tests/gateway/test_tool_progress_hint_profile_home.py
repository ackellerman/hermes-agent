"""The one-shot tool-progress (/verbose) hint latches in the home its read consults.

Upstream #119493 changes this call site but ships tests for the busy-input site only.

``TurnRunner._progress_onboarding_hint`` reads ``onboarding.seen.tool_progress_prompt`` through
``_load_gateway_config()`` (the context-local profile home when a multiplexer sets the override,
else the global home). The flag must be written to that same file; writing it to the global home
while the read consults the profile home means the hint repeats for that profile on every long
tool once the global file carries the flag.

The callback runs on the agent's worker thread, which the gateway starts under
``copy_context().run`` (so the profile-home override is visible there); the tests run it the
same way.
"""

import contextvars
import queue
import threading
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from agent.onboarding import TOOL_PROGRESS_FLAG, tool_progress_hint_gateway
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_HINT = tool_progress_hint_gateway()


def _write_cfg(home, data):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(data))


def _read_cfg(home):
    path = home / "config.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else None


def _gate_on(extra=None):
    data = {"display": {"tool_progress_command": True}}
    if extra:
        data.update(extra)
    return data


def _flagged():
    return {"onboarding": {"seen": {TOOL_PROGRESS_FLAG: True}}}


def _long_tool_turn() -> list:
    """One turn whose tool exceeded the long-tool threshold; returns what it queued. Runs the
    callback on a worker thread under a copied context, as the gateway's agent thread does."""
    from gateway.run_turn_runner import TurnRunner

    ctx = SimpleNamespace(
        _LONG_TOOL_THRESHOLD_S=10.0, progress_mode="all",
        long_tool_hint_fired=[False], progress_queue=queue.Queue(),
    )
    turn = TurnRunner(runner=None, ctx=ctx)
    worker = threading.Thread(
        target=contextvars.copy_context().run, args=(turn._progress_onboarding_hint, {"duration": 30.0}))
    worker.start()
    worker.join(10)
    out = []
    while not ctx.progress_queue.empty():
        out.append(ctx.progress_queue.get_nowait())
    return out


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A global home and a profile home; the profile home is the active override."""
    import gateway.run as gr

    global_home = tmp_path / "global"
    profile_home = tmp_path / "global" / "profiles" / "p1"
    global_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(gr, "_hermes_home", global_home)
    token = set_hermes_home_override(profile_home)
    try:
        yield SimpleNamespace(global_home=global_home, profile_home=profile_home)
    finally:
        reset_hermes_home_override(token)


class TestToolProgressHintLatch:

    def test_flag_absent_in_override_home_hint_once_then_not_again(self, homes):
        _write_cfg(homes.profile_home, _gate_on())
        assert _long_tool_turn() == [_HINT]
        assert _read_cfg(homes.profile_home)["onboarding"]["seen"][TOOL_PROGRESS_FLAG] is True
        assert _long_tool_turn() == []

    def test_flag_present_in_override_home_no_hint(self, homes):
        _write_cfg(homes.profile_home, _gate_on(_flagged()))
        assert _long_tool_turn() == []

    def test_global_flagged_profile_clean_latches_in_profile_home(self, homes):
        _write_cfg(homes.global_home, _gate_on(_flagged()))
        before = (homes.global_home / "config.yaml").read_bytes()
        _write_cfg(homes.profile_home, _gate_on())
        assert _long_tool_turn() == [_HINT]
        assert _read_cfg(homes.profile_home)["onboarding"]["seen"][TOOL_PROGRESS_FLAG] is True
        assert (homes.global_home / "config.yaml").read_bytes() == before
        # The repeat the bug produced: every later long tool showed the hint again.
        assert _long_tool_turn() == []


def test_no_override_latches_in_global_home(tmp_path, monkeypatch):
    """Single-profile gateway (no override): read and write are both the global home."""
    import gateway.run as gr

    global_home = tmp_path / "global"
    _write_cfg(global_home, _gate_on())
    monkeypatch.setattr(gr, "_hermes_home", global_home)
    assert _long_tool_turn() == [_HINT]
    assert _read_cfg(global_home)["onboarding"]["seen"][TOOL_PROGRESS_FLAG] is True
    assert _long_tool_turn() == []
