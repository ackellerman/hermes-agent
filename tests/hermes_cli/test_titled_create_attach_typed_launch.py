"""Element e1: a resume compares the launch the user typed, not the route credentials resolved.

`hermes chat -c <new title> --create-if-missing -m X --provider P` creates its session from the
typed flags and then resumes it in the same invocation. The resume validates the requested flags
against the route frozen at creation, so the two sides must be the same identity:

1. **The typed launch is captured before credentials rewrite it.** `_init_model_and_provider`
   records `cli._typed_launch` right after the startup route resolution;
   `_ensure_runtime_credentials` then normalizes `cli.model` for the resolved provider
   (`_normalize_model_for_provider`: dot→hyphen for `anthropic`) and, on a primary `AuthError`,
   the fallback ladder replaces `cli.model`/`cli.requested_provider` with the fallback entry
   (`_resolve_fallback_runtime`). `_gateway_launch` reports the captured launch, so
   `_check_resume_launch` — the classic path every board round takes — compares typed-vs-frozen
   instead of fallback-vs-frozen and stops refusing the session the command itself just created.
   The runtime route is untouched: the turn still runs, and the row still records, whatever route
   the ladder resolved.

2. **A row frozen before the capture is accepted by its provider spelling.** A plain (untitled)
   create on the pre-fix build froze the post-credentials `cli.model` with no `model_requested` and
   no provider, so its frozen id is the provider's dot→hyphen spelling of the typed one
   (`deepseek-v4.1-flash:cloud` frozen as `deepseek-v4-1-flash:cloud`). The comparison accepts
   `typed.replace('.', '-') == frozen` — provider-free, consulting no config, strictly narrower
   than normalizing — so the same command line still resumes such a row, while a genuinely
   different model still refuses.

Every module under test is imported inside its test function: on the base tree this file fails on
the behaviour, not on a collection error.
"""

import pytest

# The `-m`/`--provider`/`--api-key`/`--base-url` keys the launch policy carries as a route.
_ROUTE_FLAGS = ("model", "provider", "api_key", "base_url")


def _cli_host(monkeypatch, model, provider, *, config=None, **attrs):
    """A CLI stand-in carrying the constructor phases this element reads.

    `_init_model_and_provider` is the real one — it is where the typed launch is captured — and the
    two methods `_ensure_runtime_credentials` calls back into the CLI are bound to their shipped
    implementations, so the fallback ladder and the provider normalizer under test are the real
    ones. `cli.CLI_CONFIG["model"]` is pinned so no test reads the profile on this machine.
    """
    import types

    import cli as cli_mod
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

    monkeypatch.setitem(cli_mod.CLI_CONFIG, "model",
                        dict(config or {"default": "cfg-model", "provider": "openrouter"}))
    host = types.SimpleNamespace(_console_print=lambda *_a, **_k: None, agent=None, _model_is_default=False,
                                 _active_agent_route_signature=None, _resumed=True, session_id="local-old",
                                 api_mode="chat_completions", acp_command=None, acp_args=[],
                                 api_key=None, base_url=None)
    for key, value in attrs.items():
        setattr(host, key, value)
    CLIInitMixin._init_model_and_provider(host, model, provider, None, None)
    host._resolve_fallback_runtime = lambda exc: CLIAgentSetupMixin._resolve_fallback_runtime(host, exc)
    host._normalize_model_for_provider = (
        lambda resolved: CLIModelSwitchMixin._normalize_model_for_provider(host, resolved))
    return host


def _script_resolver(monkeypatch, *, auth_error_for=(), answer=None):
    """Point `resolve_runtime_provider` at a scripted resolver: an `AuthError` for the named
    providers, `answer` (re-labelled with the requested provider) otherwise. No network, no
    keychain. The fallback notice the ladder prints is silenced."""
    from gateway import warning_notifications
    from hermes_cli import runtime_provider
    from hermes_cli.auth import AuthError

    def resolve(requested=None, explicit_api_key=None, explicit_base_url=None, target_model=None, **_kw):
        if requested in auth_error_for:
            raise AuthError("primary unavailable", provider=requested, code="missing_api_key")
        return {**(answer or {}), "provider": requested, "requested_provider": requested}

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    monkeypatch.setattr(warning_notifications, "render_notification", lambda present, platform=None: None)


class _ScriptedGateway:
    """A gateway with no socket: `session.create` mints `local-new`, `session.resume` answers the
    snapshot this test froze (the launched row's `launch_request`)."""

    final = False
    launch_api_key = None
    launch_api_key_typed = False

    def __init__(self, *, launch_request=None, known=("local-old",)):
        self.calls = []
        self.known = set(known)
        self.launch_request = dict(launch_request or {})

    def state(self, session_id):
        import types
        return types.SimpleNamespace(adopt_snapshot=lambda snapshot: None)

    def snapshot(self, session_id):
        info = {"model": "wire-model", "cwd": "/w", "enabled_toolsets": ["file"]}
        if self.launch_request:
            info["launch_request"] = dict(self.launch_request)
        return {"session_id": session_id, "stored_session_id": session_id, "messages": [], "revision": 1,
                "execution_generation": 0, "subscription_id": "sub-" + session_id, "info": info}

    def call(self, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        self.calls.append((method, params))
        if method == "session.create":
            self.known.add("local-new")
            return self.snapshot("local-new")
        if method == "session.resume":
            if params.get("session_id") not in self.known:
                raise GatewayClientError("not_found")
            return self.snapshot(params["session_id"])
        return {}

    def resume(self, call, session_id, *, typed=False):
        self.calls.append(("session.resume", {"session_id": session_id, "typed": typed}))
        return self.snapshot(session_id)

    def created(self):
        """The last `session.create` parameters (what the row was frozen from)."""
        return [params for method, params in self.calls if method == "session.create"][-1]


def test_a_launch_time_auth_failure_keeps_the_typed_launch_for_the_resume(monkeypatch, tmp_path):
    """The board's own shape: `-m <primary> --provider <p>` on a profile whose primary raises
    `AuthError` at launch. The ladder stays on the fallback — the turn runs there and the row
    records it — while the resume compares the launch the command typed, so the session this
    invocation just created is not refused with `-m/--model differs`."""
    import types

    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin, _current_runtime
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import (
        GatewaySessionStore, _check_resume_launch, _creation_policy, _titled_launch)

    typed_model, typed_provider = "claude-opus-5-5", "anthropic"
    fallback_model, fallback_provider = "deepseek-v4.1-flash:cloud", "ollama-cloud"

    host = _cli_host(monkeypatch, typed_model, typed_provider,
                     _fallback_model=[{"provider": fallback_provider, "model": fallback_model}])
    _script_resolver(monkeypatch, auth_error_for={typed_provider},
                     answer={"base_url": "https://relay.example/v1", "api_key": "sk-fallback",
                             "api_mode": "chat_completions"})
    assert CLIAgentSetupMixin._ensure_runtime_credentials(host)

    # The runtime route is the fallback: that is what the turn sends and what the row records.
    assert (host.model, host.provider, host.requested_provider) == (
        fallback_model, fallback_provider, fallback_provider)
    assert host.base_url == "https://relay.example/v1" and host.api_key == "sk-fallback"

    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("local-new", source="cli")
        # The agent the gateway builds for the turn, from the runtime this CLI resolved.
        runtime = _current_runtime(host)
        turn_agent = types.SimpleNamespace(model=host.model, provider=runtime["provider"],
                                           requested_provider=runtime["requested_provider"],
                                           base_url=runtime["base_url"], api_mode=runtime["api_mode"],
                                           _fallback_activated=True)
        GatewayTurnPrepareMixin._sync_session_model_from_agent(
            types.SimpleNamespace(_session_db=types.SimpleNamespace(_db=db)), "local-new", turn_agent)
        row = db.get_session("local-new")
        route = SessionDB.session_gateway_runtime(row)
        assert row["model"] == fallback_model
        assert route["provider"] == fallback_provider and route["fallback_active"] is True
    finally:
        db.close()

    # The create froze the typed flags: a titled create reads `_titled_launch(args)`.
    frozen = _creation_policy(_titled_launch(types.SimpleNamespace(model=typed_model,
                                                                  provider=typed_provider)))
    assert frozen == {"model": typed_model, "provider": typed_provider}

    # The comparison is handed the launch the command typed, not the route the ladder resolved.
    launch = _gateway_launch(host)
    assert (launch["model"], launch["provider"]) == (typed_model, typed_provider)
    snapshot = {"info": {"launch_request": frozen}}
    _check_resume_launch(launch, snapshot)  # the same flags again: no refusal

    # The classic call site, end to end: main sets args.resume to the created id and the client
    # resumes it through GatewaySessionStore.open_session.
    gateway = _ScriptedGateway(launch_request=frozen, known=("local-new",))
    store = GatewaySessionStore(gateway)
    host.session_id = "local-new"
    assert store.open_session(host, launch) == "local-new"

    # Control: the same existing row with a genuinely different -m still refuses.
    with pytest.raises(GatewayClientError, match="-m/--model differs"):
        _check_resume_launch({**launch, "model": "claude-opus-5-6"}, snapshot)


def test_a_legacy_plain_create_row_frozen_with_the_provider_spelling_resumes(monkeypatch, tmp_path):
    """A row created by a *plain* (untitled) create before this fix froze the post-credentials
    `cli.model` — the provider's dot→hyphen spelling — with neither `model_requested` nor a
    provider. The same command line must still resume it.

    The control is that plain-create snapshot, built by hand here because a current build's plain
    create records the typed id (see the test below) and a titled create already froze the typed
    args, so neither would exercise the rule."""
    import argparse

    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_chat import check_resume_policy
    from hermes_cli.gateway_session_store import _check_resume_launch
    from hermes_cli.model_normalize import normalize_model_for_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # no client config: the comparison resolves nothing
    typed = "deepseek-v4.1-flash:cloud"
    frozen_id = normalize_model_for_provider(typed, "anthropic")
    assert frozen_id == "deepseek-v4-1-flash:cloud"  # `_dots_to_hyphens`: the pre-fix frozen id
    legacy = {"info": {"launch_request": {"model": frozen_id}}}

    _check_resume_launch({"model": typed}, legacy)  # the classic path the board takes
    check_resume_policy(argparse.Namespace(model=typed), legacy)  # the thin client shares the rule

    # No rule needed for a titled create: it froze the typed id itself.
    _check_resume_launch({"model": typed, "provider": "anthropic"},
                         {"info": {"launch_request": {"model": typed, "provider": "anthropic"}}})

    for other in ("deepseek-v4-flash:cloud", "deepseek-v4.1-flash"):
        with pytest.raises(GatewayClientError, match="-m/--model differs"):
            _check_resume_launch({"model": other}, legacy)


def test_the_spelling_rule_is_strictly_narrower_than_normalizing(monkeypatch, tmp_path):
    """Only dots become hyphens. A model that differs in anything else — even one a normalizer
    might rewrite into something else — keeps refusing, so the rule cannot pass a genuine change."""
    import argparse

    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_chat import check_resume_policy

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    frozen = {"info": {"launch_request": {"model": "claude-sonnet-4-6"}}}
    check_resume_policy(argparse.Namespace(model="claude-sonnet-4.6"), frozen)
    for other in ("claude-sonnet-4-7", "claude-sonnet-4.6.1", "anthropic/claude-sonnet-4.6",
                  "CLAUDE-SONNET-4.6"):
        with pytest.raises(GatewayClientError, match="-m/--model differs"):
            check_resume_policy(argparse.Namespace(model=other), frozen)


def test_a_plain_create_on_a_dot_normalizing_provider_records_the_typed_id(monkeypatch):
    """A plain (untitled) create still runs on a dot-normalizing provider: the wire keeps the
    provider's spelling, and the route the row records is the id the user typed. That is the row a
    pre-fix build froze the other way round, which is what the rule above serves."""
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_session_store import GatewaySessionStore

    typed, wire = "deepseek-v4.1-flash:cloud", "deepseek-v4-1-flash:cloud"
    host = _cli_host(monkeypatch, typed, "anthropic", _resumed=False,
                     session_id="20261012_000000_abcdef")
    _script_resolver(monkeypatch, answer={"base_url": "https://api.anthropic.com/v1",
                                         "api_key": "sk-ant", "api_mode": "chat_completions"})
    assert CLIAgentSetupMixin._ensure_runtime_credentials(host)
    assert host.model == wire  # what the turn sends

    launch = _gateway_launch(host)
    assert launch["model"] == typed  # what the row records
    gateway = _ScriptedGateway()
    assert GatewaySessionStore(gateway).open_session(host, launch) == "local-new"
    created = gateway.created()
    assert created["model"] == typed and created["provider"] == "anthropic"


def test_a_url_bearing_alias_keeps_its_key_and_endpoint_in_the_launch(monkeypatch):
    """The capture is taken after the startup route resolution, so a URL-bearing alias's own key
    and endpoint are still in the launch — `open_session` binds them as `launch_api_key`, and
    `_typed_api_key` (a typed `--api-key` the gateway must accept or refuse) stays as it was."""
    from hermes_cli import model_switch
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_session_store import GatewaySessionStore

    alias = "https://relay.example/v1"
    route = model_switch.StartupModelRoute(model="alias-model", provider="alias-provider",
                                           base_url=alias, api_key="sk-alias")
    monkeypatch.setattr(model_switch, "resolve_startup_model_route", lambda *_a, **_k: route)
    host = _cli_host(monkeypatch, "fast-alias", None)
    assert host.model == "alias-model"
    # An alias's key is not a typed --api-key: the resume's refusal rule is unchanged.
    assert host._explicit_api_key == "sk-alias" and host._typed_api_key is None

    launch = _gateway_launch(host)
    assert launch["api_key"] == "sk-alias" and launch["base_url"] == alias
    gateway = _ScriptedGateway(launch_request={"model": "alias-model", "provider": "alias-provider",
                                               "base_url": alias})
    store = GatewaySessionStore(gateway)
    assert store.open_session(host, launch) == "local-old"
    assert store.connection.launch_api_key == "sk-alias"
    assert store.connection.launch_api_key_typed is False


def test_a_genuine_change_of_a_launch_flag_still_refuses(monkeypatch):
    """An existing session, frozen by an earlier invocation: a changed `-m`, `--provider` or
    `--base-url` still refuses, and the refusal still names that flag."""
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import _check_resume_launch

    host = _cli_host(monkeypatch, "claude-opus-5-5", "anthropic",
                     _fallback_model=[{"provider": "ollama-cloud", "model": "deepseek-v4.1-flash:cloud"}])
    _script_resolver(monkeypatch, auth_error_for={"anthropic"},
                     answer={"base_url": "https://relay.example/v1", "api_key": "sk-fallback",
                             "api_mode": "chat_completions"})
    assert CLIAgentSetupMixin._ensure_runtime_credentials(host)
    launch = _gateway_launch(host)
    snapshot = {"info": {"launch_request": {"model": "claude-opus-5-5", "provider": "anthropic",
                                            "base_url": "https://relay.example/v1"}}}
    _check_resume_launch(launch, snapshot)  # the same flags again are fine

    for changed, flag in (({"model": "claude-opus-5-6"}, "-m/--model"),
                          ({"provider": "openrouter"}, "--provider"),
                          ({"base_url": "https://other.example/v1"}, "--base-url")):
        with pytest.raises(GatewayClientError, match=flag):
            _check_resume_launch({**launch, **changed}, snapshot)


def test_a_typed_provider_auto_stays_no_pick(monkeypatch):
    """`--provider auto` is the client's "no pick" and is not launch policy on either side of the
    comparison: the classic launch carries no provider (this element), and the thin client's resume
    drops the same word through the create's own sentinel (element e3), so re-running the identical
    argv attaches instead of refusing `--provider differs`."""
    import argparse

    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.gateway_chat import check_resume_policy
    from hermes_cli.gateway_session_store import _check_resume_launch

    host = _cli_host(monkeypatch, "claude-opus-5-5", "auto")
    assert host._explicit_provider == "auto"
    assert _gateway_launch(host)["provider"] is None
    _check_resume_launch(_gateway_launch(host),
                         {"info": {"launch_request": {"model": "claude-opus-5-5"}}})

    # The workboard's argv re-typed on a resume: the sentinel is no pick, so it is not compared.
    check_resume_policy(argparse.Namespace(model="claude-opus-5-5", provider="auto"),
                        {"info": {"launch_request": {"model": "claude-opus-5-5"}}})


def test_new_session_drops_the_typed_route_and_rederives_from_config(monkeypatch):
    """`/new` still clears the typed `-m` and re-derives the model from config: the captured launch
    does not outlive the session it was captured for, so `/new`'s create carries no model route."""
    from hermes_cli.cli_init_mixin import _gateway_launch
    from hermes_cli.cli_session_mixin import CLISessionMixin

    host = _cli_host(monkeypatch, "claude-opus-5-5", "anthropic", conversation_history=[],
                     _session_db=None, _pending_title=None, service_tier=None, _credential_pool=None,
                     reasoning_config=None, _notify_session_boundary=lambda *_a, **_k: None)
    assert _gateway_launch(host)["model"] == "claude-opus-5-5"

    CLISessionMixin.new_session(host, silent=True)
    assert host._explicit_model_override is False
    assert host.model == "cfg-model"  # re-derived from config, not the previous session's -m
    assert _gateway_launch(host)["model"] is None and _gateway_launch(host)["provider"] is None
