"""A profile route whose only Anthropic credential is benched for its model (a per-model 429 cooldown) is
not a missing provider: a bench that lifts within ``delegation.model_bench_wait_seconds`` is waited out and
the route resolved again, a longer one takes the profile's own fallback chain, and with neither the refusal
says when the bench lifts."""

import json
import time
from unittest.mock import MagicMock

import hermes_yaml as yaml
import pytest

KEY = "sk-ant-api03-profile-bench-0000"
MODEL = "claude-sonnet-4-5"
FALLBACKS = [
    {"provider": "custom", "model": "fb-one", "base_url": "https://fb-one.invalid/v1", "api_key": "fb-one-key"},
    {"provider": "custom", "model": "fb-two", "base_url": "https://fb-two.invalid/v1", "api_key": "fb-two-key"},
]


@pytest.fixture
def reviewer(tmp_path, monkeypatch):
    """A launch profile and a ``reviewer`` profile on native Anthropic with one pooled API key."""
    launch = tmp_path / "os-home" / ".hermes"
    home = launch / "profiles" / "reviewer"
    home.mkdir(parents=True)
    (launch / "config.yaml").write_text(yaml.safe_dump({"model": {
        "provider": "custom", "base_url": "https://launch.invalid/v1", "default": "launch-model", "api_key": "k"}}))
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {"provider": "anthropic", "default": MODEL}}))
    (home / ".env").write_text(f"ANTHROPIC_API_KEY={KEY}\n")
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "seat", "label": "seat", "auth_type": "api_key", "priority": 0, "source": "manual", "access_token": KEY,
    }]}}))
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    return home


def _bench(home, seconds):
    """What a 429 for MODEL records: the pool benches the key for that model until ``reset_at``."""
    from agent.credential_pool import load_pool
    from tools.delegate_tool_routes import delegated_profile_scope
    with delegated_profile_scope(home):
        load_pool("anthropic").mark_exhausted_and_rotate(
            status_code=429, error_context={"reset_at": time.time() + seconds}, api_key_hint=KEY,
            failure_reason="rate_limit", model=MODEL)


def _set_fallbacks(home, entries):
    cfg = yaml.safe_load((home / "config.yaml").read_text())
    cfg["fallback_providers"] = entries
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _route(delegation_cfg, parent=None):
    from tools.delegate_tool_routes import _route_to_task_route
    return _route_to_task_route({"profile": "reviewer"}, parent or MagicMock(_interrupt_requested=False), {},
                                delegation_cfg)


@pytest.fixture
def naps(monkeypatch):
    """Every wait the router takes, in seconds (the waits still happen)."""
    import tools.delegate_tool_routes as routes
    taken, real = [], time.sleep

    def nap(seconds):
        taken.append(seconds)
        real(seconds)

    monkeypatch.setattr(routes, "_bench_nap", nap, raising=False)
    return taken


def test_unbenched_profile_resolves_without_waiting(reviewer, naps):
    route = _route({"model_bench_wait_seconds": 30})
    assert (route.creds["model"], route.creds["api_key"]) == (MODEL, KEY) and not naps


def test_bench_lifting_within_the_limit_is_waited_out_and_the_profile_runs_on_its_own_model(reviewer, naps):
    _bench(reviewer, 1.5)
    _set_fallbacks(reviewer, FALLBACKS)
    started = time.time()
    route = _route({"model_bench_wait_seconds": 30})
    assert (route.creds["model"], route.creds["provider"], route.creds["api_key"]) == (MODEL, "anthropic", KEY)
    assert route.routing_cfg["fallback_providers"] == FALLBACKS  # the primary runs: its chain is untouched
    assert naps and 1.0 <= time.time() - started < 10


def test_bench_longer_than_the_limit_takes_the_profiles_fallback_chain_at_once(reviewer, naps):
    _bench(reviewer, 600)
    _set_fallbacks(reviewer, FALLBACKS)
    route = _route({"model_bench_wait_seconds": 5})
    assert not naps
    assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"]) == (
        "fb-one", "https://fb-one.invalid/v1", "fb-one-key")
    # The child's mid-run chain is what follows the entry it runs on.
    assert route.routing_cfg["fallback_providers"] == FALLBACKS[1:]


def test_long_bench_without_a_fallback_is_refused_naming_when_it_lifts(reviewer, naps):
    from agent.delegation_provider import DelegationRouteError
    _bench(reviewer, 600)
    with pytest.raises(DelegationRouteError) as refusal:
        _route({"model_bench_wait_seconds": 5})
    message = str(refusal.value)
    assert not naps
    assert f"rate-limited for {MODEL} until " in message and "UTC" in message
    assert "other Claude models remain available" in message and "no usable model provider" in message


def test_zero_limit_never_waits(reviewer, naps):
    from agent.delegation_provider import DelegationRouteError
    _bench(reviewer, 1.5)
    with pytest.raises(DelegationRouteError, match="rate-limited for"):
        _route({"model_bench_wait_seconds": 0})
    assert not naps


def test_the_default_limit_waits_out_a_short_bench(reviewer, naps):
    _bench(reviewer, 1.5)
    assert _route({}).creds["model"] == MODEL and naps


def test_a_parent_interrupt_ends_the_wait(reviewer, naps):
    from agent.delegation_provider import DelegationRouteError
    _bench(reviewer, 20)
    started = time.time()
    with pytest.raises(DelegationRouteError, match="rate-limited for"):
        _route({"model_bench_wait_seconds": 30}, MagicMock(_interrupt_requested=True))
    assert time.time() - started < 5


def test_missing_credentials_are_refused_as_before_and_never_fall_back(reviewer, naps):
    """Only a rate limit takes the chain: a profile without credentials is a setup error to show, not to mask."""
    from agent.delegation_provider import DelegationRouteError
    (reviewer / ".env").write_text("")
    (reviewer / "auth.json").write_text("{}")
    _set_fallbacks(reviewer, FALLBACKS)
    with pytest.raises(DelegationRouteError, match="no usable model provider"):
        _route({"model_bench_wait_seconds": 30})
    assert not naps


def test_benched_model_auth_error_carries_the_remaining_wait(reviewer):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.delegate_tool_routes import delegated_profile_scope
    _bench(reviewer, 600)
    with delegated_profile_scope(reviewer), pytest.raises(AuthError) as raised:
        resolve_runtime_provider(requested="anthropic", target_model=MODEL)
    assert raised.value.retryable is True and 590 < raised.value.retry_after <= 600
    assert raised.value.provider == "anthropic"
