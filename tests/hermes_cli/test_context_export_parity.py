"""One-shot callers (``--format stream-json``, ``-z --usage-file``) get the TUI/desktop context gauge's
end-of-turn reading: the exported ``context`` object equals what ``tui_gateway.server._get_usage`` puts in the
gauge for the same agent, and is ``None`` exactly when the gauge has no reading."""

from types import SimpleNamespace

import pytest

from agent import conversation_loop
from agent.context_compressor import ContextCompressor
from hermes_cli.stream_json import result_context
from tui_gateway.server import _get_usage

_GAUGE_KEYS = {"used": "context_used", "max": "context_max", "percent": "context_percent",
               "source": "context_source", "estimated": "context_estimated"}


def _comp(last_prompt, last_real, length):
    return SimpleNamespace(last_prompt_tokens=last_prompt, last_real_prompt_tokens=last_real, context_length=length,
                           compression_count=0)


def _seeded_compressor():
    comp = ContextCompressor(model="fixture", config_context_length=100_000, quiet_mode=True)
    comp.maybe_seed_preflight_display_tokens(4321)
    return comp


_STATES = {
    "provider_reading": lambda: _comp(12_345, 12_345, 200_000),
    "preflight_seed": lambda: _comp(4_321, 0, 200_000),
    "preflight_seed_real_compressor": _seeded_compressor,
    "minus_one_sentinel": lambda: _comp(-1, 12_345, 200_000),
    "unknown_window_zero": lambda: _comp(12_345, 12_345, 0),
    "unknown_window_none": lambda: _comp(12_345, 12_345, None),
    "no_compressor": lambda: None,
}


def _agent(comp):
    return SimpleNamespace(context_compressor=comp, model="fixture")


@pytest.mark.parametrize("state", sorted(_STATES))
def test_exported_context_equals_the_tui_gauge(state):
    agent = _agent(_STATES[state]())
    gauge = _get_usage(agent)
    exported = result_context({"context_usage": conversation_loop._context_usage_snapshot(agent)})
    if "context_used" not in gauge:
        assert exported is None
    else:
        assert exported == {key: gauge[gauge_key] for key, gauge_key in _GAUGE_KEYS.items()}


@pytest.mark.parametrize("envelope", [
    {"final_response": "ok", "completed": True, "failed": False, "messages": []},
    {"final_response": "", "completed": False, "failed": True, "error": "HTTP 400", "messages": []},
], ids=["finalized", "early_return_failure"])
def test_every_run_result_carries_the_gauge_reading(monkeypatch, envelope):
    """Early-return verdicts never reach the finalizer; the reading must still ride every envelope."""
    agent = _agent(_comp(12_345, 12_345, 200_000))
    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", lambda *_a, **_k: dict(envelope))
    result = conversation_loop.run_conversation(agent, "say ok")
    gauge = _get_usage(agent)
    assert result_context(result) == {key: gauge[gauge_key] for key, gauge_key in _GAUGE_KEYS.items()}


def test_a_failing_context_engine_never_fails_the_turn(monkeypatch):
    """Third-party context engines own their occupancy figure; one that raises yields no reading, not a crash."""
    class _Raising:
        @property
        def last_prompt_tokens(self):
            raise RuntimeError("engine bug")

    agent = _agent(_Raising())
    monkeypatch.setattr(conversation_loop, "_run_conversation_turn", lambda *_a, **_k: {"final_response": "ok", "messages": []})
    result = conversation_loop.run_conversation(agent, "say ok")
    assert result["final_response"] == "ok"
    assert result["context_usage"] is None
