"""A ROUTED background-review fork honors ``auxiliary.background_review.fallback_chain`` (#93592).

The review fork is a full AIAgent running ``run_conversation``, so a 429 or outage on the routed
primary is recovered by the AIAgent fallback chain (``_fallback_chain``), never by the auxiliary
client's ``_try_configured_fallback_chain``. Before this, ``_fork_init_kwargs`` never passed
``fallback_model``: the fork started with an empty chain and a failing route lost the review.

The same-model (unrouted) fork must stay byte-identical to today: its constructor kwargs carry no
``fallback_model`` whatever the task block says, so nothing about the cache-parity path moves.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import run_agent
import agent.background_review as bg_review
from agent.background_review import build_cache_parity_fork

from tests.agent.test_background_review_cache_parity import _make_agent_stub, _make_recorder_class

ROUTED_RUNTIME = {
    "provider": "ollama-cloud", "model": "deepseek-v4.1-flash:cloud", "api_key": "test-key",
    "base_url": "https://ollama.com/v1", "api_mode": "chat_completions", "credential_pool": None,
    "request_overrides": {}, "max_tokens": None, "command": None, "args": [], "routed": True,
}
UNROUTED_RUNTIME = {**ROUTED_RUNTIME, "provider": "openai", "model": "test-model", "routed": False}
CHAIN = [{"provider": "openrouter", "model": "~deepseek/deepseek-flash-latest"}]


def _fork_kwargs(task_cfg, runtime):
    captured = {}
    agent = _make_agent_stub(run_agent.AIAgent)
    with patch.object(run_agent, "AIAgent", _make_recorder_class(captured)), \
            patch.object(bg_review, "_resolve_review_runtime", return_value=dict(runtime)):
        _fork, _rt, routed = build_cache_parity_fork(agent, task_cfg, max_iterations=5)
    assert routed is runtime["routed"]
    return captured["init_kwargs"]


def test_routed_fork_gets_the_task_fallback_chain():
    kwargs = _fork_kwargs({"fallback_chain": CHAIN}, ROUTED_RUNTIME)
    assert kwargs["fallback_model"] == CHAIN
    # A copy: the fork's failover bookkeeping must never write into the loaded config.
    assert kwargs["fallback_model"] is not CHAIN
    assert kwargs["fallback_model"][0] is not CHAIN[0]


def test_routed_fork_keeps_entry_order_and_extra_keys():
    chain = [
        {"provider": "openrouter", "model": "~deepseek/deepseek-flash-latest", "reasoning_effort": "low"},
        {"provider": "nous", "model": "deepseek/deepseek-v4"},
    ]
    kwargs = _fork_kwargs({"fallback_chain": chain}, ROUTED_RUNTIME)
    assert [(e["provider"], e["model"]) for e in kwargs["fallback_model"]] == [
        ("openrouter", "~deepseek/deepseek-flash-latest"), ("nous", "deepseek/deepseek-v4")]
    assert kwargs["fallback_model"][0]["reasoning_effort"] == "low"


def test_routed_fork_without_a_chain_is_unchanged():
    for task_cfg in ({}, {"fallback_chain": None}, {"fallback_chain": []}):
        assert "fallback_model" not in _fork_kwargs(task_cfg, ROUTED_RUNTIME), task_cfg


def test_unrouted_fork_never_gets_a_chain_and_its_kwargs_are_identical():
    base = _fork_kwargs({}, UNROUTED_RUNTIME)
    with_chain = _fork_kwargs({"fallback_chain": CHAIN}, UNROUTED_RUNTIME)
    assert "fallback_model" not in with_chain
    assert with_chain == base


def test_bad_shape_warns_and_is_ignored(caplog):
    for bad in ({"provider": "openrouter", "model": "x"}, "openrouter", 3, ["openrouter"],
                [{"provider": "openrouter"}], [{"model": "x"}]):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="agent.background_review"):
            kwargs = _fork_kwargs({"fallback_chain": bad}, ROUTED_RUNTIME)
        assert "fallback_model" not in kwargs, bad
        assert "auxiliary.background_review.fallback_chain" in caplog.text, bad


def test_partly_bad_chain_keeps_the_usable_entries_and_warns(caplog):
    chain = ["junk", {"provider": "openrouter"}, CHAIN[0]]
    with caplog.at_level(logging.WARNING, logger="agent.background_review"):
        kwargs = _fork_kwargs({"fallback_chain": chain}, ROUTED_RUNTIME)
    assert kwargs["fallback_model"] == CHAIN
    assert "auxiliary.background_review.fallback_chain" in caplog.text


def test_routed_side_question_fork_reads_and_names_its_own_block(caplog):
    """``build_cache_parity_fork`` is shared with a routed /btw: the chain comes from the block the
    caller passed (``auxiliary.side_question``), and the warning names that block, not the review's."""
    captured = {}
    agent = _make_agent_stub(run_agent.AIAgent)
    with patch.object(run_agent, "AIAgent", _make_recorder_class(captured)), \
            patch.object(bg_review, "_resolve_review_runtime", return_value=dict(ROUTED_RUNTIME)), \
            caplog.at_level(logging.WARNING, logger="agent.background_review"):
        build_cache_parity_fork(agent, {"fallback_chain": [CHAIN[0], {"model": "x"}]}, max_iterations=1,
                                write_origin="side_question")
    assert captured["init_kwargs"]["fallback_model"] == CHAIN
    assert "auxiliary.side_question.fallback_chain" in caplog.text
    assert "auxiliary.background_review" not in caplog.text


def test_real_routed_fork_arms_the_chain(monkeypatch, tmp_path):
    """Delivery: through the real AIAgent constructor, the routed fork's ``_fallback_chain`` (what
    ``try_activate_fallback`` walks on a 429) holds the configured entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _make_agent_stub(run_agent.AIAgent)
    agent.tools = []
    with patch.object(bg_review, "_resolve_review_runtime", return_value=dict(ROUTED_RUNTIME)):
        fork, _rt, routed = build_cache_parity_fork(agent, {"fallback_chain": CHAIN}, max_iterations=2)
    try:
        assert routed
        assert fork.provider == "ollama-cloud"
        assert [(e["provider"], e["model"]) for e in fork._fallback_chain] == [
            ("openrouter", "~deepseek/deepseek-flash-latest")]
        assert fork._fallback_index == 0
    finally:
        fork.close()
