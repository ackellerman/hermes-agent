"""A ``profile_routes`` seat can put a named model option on a profile route: the child keeps the profile's
home, SOUL, skills and toolsets and runs on the route's model, provider and fallback chain, with credentials
from the profile's own scope. A route that would carry the launching seat's key is refused before any child."""

import json
import threading
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest


def _write_profile(home, *, host, model, key, secret, effort, soul=None):
    # A hosted (non-loopback) endpoint: the key must come from the profile's own secrets.
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"},
        "agent": {"reasoning_effort": effort},
    }), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\nPROBE_SECRET={secret}\n", encoding="utf-8")
    if soul:
        (home / "SOUL.md").write_text(soul, encoding="utf-8")


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    launch = tmp_path / "os-home" / ".hermes"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key", secret="launch-secret",
                   effort="medium")
    coder = launch / "profiles" / "coder"
    _write_profile(coder, host="coder.invalid", model="coder-model", key="coder-key", secret="coder-secret",
                   effort="low", soul="I am the coder profile.")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # The launch profile's secrets live in the process env, as after a normal dotenv load.
    monkeypatch.setenv("PROFILE_KEY", "launch-key")
    monkeypatch.setenv("PROBE_SECRET", "launch-secret")
    # No OAuth account on this host: a provider key must come from the profile's own .env.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "os-home"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return launch, coder


def _snapshot():
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_constants import get_hermes_home
    return {"home": str(get_hermes_home()), "secret": get_secret("PROBE_SECRET"),
            "model": load_config_readonly()["model"]["default"]}


def _parent():
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._credential_pool = MagicMock(name="launch-pool")
    return parent


def _set_config(home, **sections):
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    cfg.update(sections)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _build(route, parent):
    from tools.delegate_tool import _build_child_agent, _route_overrides
    seen = {}

    def construct(**kwargs):
        seen.update(kwargs)
        return MagicMock(session_id="child-1")

    with patch("run_agent.AIAgent", side_effect=construct):
        child = _build_child_agent(task_index=0, goal="work as coder", context=None, toolsets=None,
                                   model=route.creds["model"], max_iterations=5, task_count=1,
                                   parent_agent=parent, **_route_overrides(route))
    return child, seen


class _RouteProvider:
    """A backend that hands core one fixed route for every task, as ``model_map``/``profile_routes`` does."""

    name = "profile_routes"

    def __init__(self, route):
        self.route = route

    def task_parameters(self, config):
        return {}

    def route_task(self, task, config):
        return dict(self.route)


def _resolve(route, parent=None, config=None):
    from tools.delegate_tool_routes import resolve_task_routes
    with patch("tools.delegate_tool_routes._active_provider", return_value=_RouteProvider(route)):
        return resolve_task_routes([{"goal": "work as coder"}], config or {}, parent or _parent())


def test_option_on_profile_runs_route_model_with_profile_identity(two_profiles):
    from tools.delegate_tool_routes import _route_to_task_route

    launch, coder = two_profiles
    parent = _parent()
    # A route's speed tier is applied on a profile+model route too, the way the non-profile branch applies it
    # (the offer gate itself is tested in tests/tools/test_delegation_route_fields.py).
    with patch("hermes_cli.models.resolve_fast_mode_overrides", return_value={"service_tier": "priority"}):
        route = _route_to_task_route({"profile": "coder", "model": "route-model", "provider": "custom",
                                      "service_tier": "fast"}, parent, {})
    assert route.creds["service_tier"] == "priority" and route.creds["task_tier_overrides"] == {"service_tier": "priority"}
    assert route.creds["request_overrides"]["service_tier"] == "priority"
    assert route.profile_home == coder
    assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"]) == (
        "route-model", "https://coder.invalid/v1", "coder-key")
    # No route effort: the profile's own resolved reasoning, never the launching seat's.
    assert route.reasoning_effort.config == {"enabled": True, "effort": "low"}
    _, built = _build(route, parent)
    assert (built["model"], built["base_url"], built["api_key"]) == (
        "route-model", "https://coder.invalid/v1", "coder-key")
    assert built["service_tier"] == "priority"
    assert built["load_soul_identity"] is True
    assert _snapshot()["home"] == str(launch)


def test_option_on_profile_uses_profile_scope_credentials(two_profiles, monkeypatch):
    from tools.delegate_tool_routes import _route_to_task_route

    launch, coder = two_profiles
    (launch / ".env").write_text("PROFILE_KEY=launch-key\nPROBE_SECRET=launch-secret\n"
                                 "ANTHROPIC_API_KEY=launch-anthropic-key\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "launch-anthropic-key")
    parent = _parent()
    option = {"profile": "coder", "model": "claude-sonnet-4-5", "provider": "anthropic"}

    with pytest.raises(ValueError, match="Cannot resolve delegation provider 'anthropic'"):
        _route_to_task_route(dict(option), parent, {})

    (coder / ".env").write_text("PROFILE_KEY=coder-key\nPROBE_SECRET=coder-secret\n"
                                "ANTHROPIC_API_KEY=profile-anthropic-key\n", encoding="utf-8")
    route = _route_to_task_route(dict(option), parent, {})
    assert (route.creds["model"], route.creds["provider"], route.creds["api_key"]) == (
        "claude-sonnet-4-5", "anthropic", "profile-anthropic-key")


def test_option_fallback_explicit_and_default(two_profiles):
    from tools.delegate_tool_routes import _route_to_task_route

    _, coder = two_profiles
    parent = _parent()
    chain = [{"provider": "anthropic", "model": "fb-one"}, {"provider": "openrouter", "model": "fb-two"}]
    _set_config(coder, fallback_providers=chain)
    explicit = [{"provider": "ollama-cloud", "model": "deepseek-v4.1-flash:cloud"}]
    route = _route_to_task_route({"profile": "coder", "model": "route-model", "provider": "custom",
                                  "fallback_providers": explicit}, parent, {})
    assert route.routing_cfg["fallback_providers"] == explicit
    # With none of its own, the option falls back to the profile's own model, then the profile's chain.
    route = _route_to_task_route({"profile": "coder", "model": "route-model", "provider": "custom"}, parent, {})
    assert route.routing_cfg["fallback_providers"] == [{"provider": "custom", "model": "coder-model"}, *chain]


def test_option_base_url_without_provider_refused(two_profiles):
    (route,), error = _resolve({"profile": "coder", "model": "route-model",
                                "base_url": "https://elsewhere.invalid/v1"})
    assert error is None
    assert "takes provider, not base_url" in (route.error or "")
    assert route.creds == {}  # refused before any child was built


def test_option_base_url_with_provider_refused(two_profiles):
    (route,), error = _resolve({"profile": "coder", "model": "route-model", "provider": "custom",
                                "base_url": "https://elsewhere.invalid/v1", "api_key": "stand-in-key"})
    assert error is None
    assert "takes provider, not base_url" in (route.error or "")
    assert route.creds == {}


def test_option_without_provider_refused(two_profiles):
    (route,), error = _resolve({"profile": "coder", "model": "route-model"})
    assert error is None
    assert "must name its provider" in (route.error or "")
    assert route.creds == {}


def test_option_bad_reasoning_effort_refused(two_profiles):
    route = {"profile": "coder", "model": "route-model", "provider": "custom", "reasoning_effort": "turbo"}
    (refused,), error = _resolve(dict(route))
    assert error is None
    assert "unknown reasoning_effort 'turbo'" in (refused.error or "")
    assert refused.creds == {}
    # A known level is the route's own, even when the launching delegation block says something else.
    (ok,), error = _resolve({**route, "reasoning_effort": "high"}, config={"reasoning_effort": "medium"})
    assert error is None and ok.error is None
    assert ok.reasoning_effort.config == {"enabled": True, "effort": "high"}


class _OptionProvider:
    """``profile_routes`` with one option: ``profile`` + ``option`` -> that option's model on that profile."""

    name = "profile_routes"

    def task_parameters(self, config):
        return {"profile": {"type": "string"}, "option": {"type": "string", "enum": ["alt"]}}

    def route_task(self, task, config):
        return {"profile": task["profile"], "model": "route-model", "provider": "custom",
                "label": f"{task['profile']}/{task['option']}"}


def test_option_continue_reroutes_to_same_model(two_profiles):
    from tools.delegate_tool_routes import resolve_task_routes

    _, coder = two_profiles
    task = {"goal": "work as coder", "profile": "coder", "option": "alt"}
    parent = _parent()
    with patch("tools.delegate_tool_routes._active_provider", return_value=_OptionProvider()):
        (first,), error = resolve_task_routes([dict(task)], {}, parent)
        assert error is None and first.error is None
        assert first.creds["model"] == "route-model" and first.profile_home == coder
        assert first.request == {"profile": "coder", "option": "alt"}
        (again,), error = resolve_task_routes([dict(first.request)], {}, parent)
    assert error is None and again.error is None
    assert again.creds["model"] == "route-model" and again.profile_home == coder
