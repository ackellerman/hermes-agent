"""OpenRouter provider_routing general passthrough.

Unknown keys in the ``provider_routing`` config block are forwarded verbatim
into the request's provider preference object, while the typed keys
(sort/only/ignore/order/require_parameters/data_collection) keep their
attribute-based semantics and win on conflict. Config is read via
``hermes_cli.config.load_config_readonly`` (deferred import inside the helper,
so the monkeypatch below resolves), following the sibling per-model test.
"""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch


def _agent(model, **flat):
    base = dict(providers_allowed=None, providers_ignored=None, providers_order=None, provider_sort="price",
                provider_require_parameters=False, provider_data_collection=None)
    base.update(flat)
    return SimpleNamespace(model=model, **base)


@pytest.fixture
def monkeypatch_config(monkeypatch):
    import hermes_cli.config as config_mod

    def _set(cfg):
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
        return cfg

    return _set


# --- _provider_routing_extra unit tests -------------------------------------

def test_extra_drops_typed_and_models_forwards_unknown_verbatim():
    pr = {
        "sort": "throughput",
        "only": ["DeepInfra"],
        "models": {"openai/gpt-6-astra": {"preferred_min_throughput": {"p90": 50}}},
        "zdr": True,
        "max_price": {"prompt": 0.5, "completion": 1.5},
        "quantizations": ["fp8"],
    }
    extra = cch._provider_routing_extra(pr, exclude=("models",))
    assert extra == {"zdr": True, "max_price": {"prompt": 0.5, "completion": 1.5},
                     "quantizations": ["fp8"]}


def test_extra_non_string_empty_keys_skipped():
    extra = cch._provider_routing_extra({1: "bad", "": "empty", "zdr": True})
    assert extra == {"zdr": True}


# --- _provider_preferences_for_agent contract tests -------------------------

def test_top_level_passthrough_carried_into_prefs(monkeypatch_config):
    monkeypatch_config({"provider_routing": {
        "sort": "price",
        "zdr": True,
        "allow_fallbacks": False,
        "preferred_min_throughput": {"p90": 50},
    }})
    prefs = cch._provider_preferences_for_agent(_agent("openrouter/deepseek/deepseek-v4-flash-0731"))
    assert prefs == {"sort": "price", "zdr": True, "allow_fallbacks": False,
                     "preferred_min_throughput": {"p90": 50}}


def test_per_model_passthrough_for_that_model_only_and_beats_flat(monkeypatch_config):
    monkeypatch_config({"provider_routing": {
        "sort": "price",
        "preferred_min_throughput": {"p90": 10},
        "models": {
            "openai/gpt-6-astra": {"preferred_min_throughput": {"p90": 50}},
            "anthropic/claude-fable-5.1": {"preferred_max_latency": 1.0},
        },
    }})
    # Per-model passthrough overrides flat for that model.
    assert cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra")) == {
        "sort": "price", "preferred_min_throughput": {"p90": 50}}
    # A model with only an extra per-model key keeps flat passthrough too.
    assert cch._provider_preferences_for_agent(_agent("anthropic/claude-fable-5.1")) == {
        "sort": "price", "preferred_min_throughput": {"p90": 10}, "preferred_max_latency": 1.0}
    # Unlisted model: no per-model passthrough leaks across models.
    assert cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6")) == {
        "sort": "price", "preferred_min_throughput": {"p90": 10}}


def test_typed_keys_win_over_passthrough(monkeypatch_config):
    monkeypatch_config({"provider_routing": {
        "sort": "price",
        "only": ["config-typo-provider"],
        "zdr": True,
    }})
    agent = _agent("openrouter/openai/gpt-6-astra", providers_allowed=["real-provider"])
    prefs = cch._provider_preferences_for_agent(agent)
    # Attribute-based typed "only" wins; the config typed value never leaks as passthrough.
    assert prefs["only"] == ["real-provider"]
    assert prefs["zdr"] is True


def test_falsy_passthrough_preserved_typed_falsy_still_dropped(monkeypatch_config):
    monkeypatch_config({"provider_routing": {
        "sort": "price",
        "require_parameters": False,
        "allow_fallbacks": False,
    }})
    prefs = cch._provider_preferences_for_agent(_agent("openrouter/openai/gpt-6-astra"))
    # passthrough falsy survives verbatim...
    assert prefs["allow_fallbacks"] is False
    # ...while the typed falsy value is still dropped by the pre-existing filter.
    assert "require_parameters" not in prefs


def test_empty_and_none_config_noop_regression(monkeypatch_config):
    for cfg in ({}, {"provider_routing": {}}, {"provider_routing": None}):
        monkeypatch_config(cfg)
        # Byte-identical to the prior typed-only behavior (sort defaults to "price").
        assert cch._provider_preferences_for_agent(_agent("openrouter/openai/gpt-6-astra")) == {"sort": "price"}


def test_non_dict_truthy_provider_routing_no_crash(monkeypatch_config):
    # A malformed (list/string) provider_routing config must not raise AttributeError.
    for bad in (["not-a-dict"], "provider_routing-as-string"):
        monkeypatch_config({"provider_routing": bad})
        assert cch._provider_preferences_for_agent(_agent("openrouter/openai/gpt-6-astra")) == {"sort": "price"}
    # Non-dict models block inside an otherwise-valid dict is also tolerated.
    monkeypatch_config({"provider_routing": {"sort": "price", "models": ["bad"]}})
    assert cch._provider_preferences_for_agent(_agent("openrouter/openai/gpt-6-astra")) == {"sort": "price"}