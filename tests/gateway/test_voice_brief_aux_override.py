"""A pinned brief is absolute: ``task_chain_only`` stops all four escape doors, and failure bypasses.

``auxiliary.<task>.task_chain_only`` is read at the four doors a pinned auxiliary task can take
out of its own route — the ``openrouter``/``custom`` auto re-probe and the vision ``auto``
re-probe in ``_resolve_call_client``, the request-time main-agent net in
``_ladder_provider_fallback`` and the quarantine walk's discovery rung in
``_next_fallback_after_quarantine`` — through one helper,
``_task_stops_at_chain(task, resolved_provider)``, which also requires a non-``auto`` resolved
provider. So the shipped ``provider: auto`` + ``task_chain_only: true`` block on
``auxiliary.voice_brief`` stays inert: an auto brief keeps the e2 route (session pair, the
profile's ``fallback_providers`` chain, the discovery rung).

On a pinned brief the seam decides. ``deliver_preturn_brief`` reads its own task route and raises
``VoiceBriefBypassed(provider, reason)`` for both failure classes — the resolution-time
``AuxiliaryClientUnavailable`` and the request-time ladder error — and ``_voice_brief_before_turn``
posts one short message naming the failed provider and lets the turn run. Every other failure keeps
``VoiceBriefError``/``FAILED_TEXT``, and an auto brief whose main route and whole chain fail still
refuses the turn.

Real seams: ``GatewayRunner._run_agent`` → ``_voice_brief_before_turn``, the real
``deliver_preturn_brief``/``_ask_model``, the real ``agent.auxiliary_client`` route resolution,
recovery ladder and quarantine walk, and the real ``_get_auxiliary_task_config`` /
``_resolve_task_provider_model`` config reads (``load_config_readonly`` is stubbed with a
``DEFAULT_CONFIG`` copy where a test pins a block; the two-profile tests write real ``config.yaml``
files and read them through the real loader under ``_profile_runtime_scope``).

Faked, at the edges only: provider client construction (``resolve_provider_client`` returns a
recording fake per provider, and ``None`` for a provider whose credential is missing), the
discovery-provider roster (``_get_provider_chain``), the TTS provider and the platform adapter
(playback / text post). The session-pair lane is reached the way the auto route reaches it —
through ``_try_main_provider_route`` — and the main-agent net is reached the way it resolves —
through config ``model.provider`` / ``model.default``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent import auxiliary_client
from gateway import voice_brief
from gateway.config import Platform
from gateway.platforms.event import MessageType
from gateway.session import SessionSource

# Imported at collection, not inside a test: ``agent.auxiliary_client`` imports these wire helpers
# lazily, and ``openai``'s own import reads install metadata (pydantic's entry-point scan), which
# the home-I/O guard refuses once a test is running.
from agent import auxiliary_wire  # side-effect import; see the comment above

ORDER: list = []

CHAT_ID = "123"
GUILD_ID = 111
SESSION_PROVIDER = "session-provider"
SESSION_MODEL = "session-model"
SESSION_BASE_URL = "http://127.0.0.1:1/v1"
SESSION_API_KEY = "session-key"
SESSION_API_MODE = "chat_completions"
BRIEF_OUTPUT = "You want me to check the deployment logs and report the error."
MAIN_ANSWER = "main model answer"
PIN_KEY = "pin-key"


def _session_runtime(api_mode: str = SESSION_API_MODE) -> dict:
    return {
        "provider": SESSION_PROVIDER, "api_mode": api_mode,
        "base_url": SESSION_BASE_URL, "api_key": SESSION_API_KEY,
    }


def _route(api_mode: str = SESSION_API_MODE) -> dict:
    """The route dict ``_voice_brief_before_turn`` builds for one turn."""
    return {
        "provider": SESSION_PROVIDER, "model": SESSION_MODEL, "base_url": SESSION_BASE_URL,
        "api_key": SESSION_API_KEY, "api_mode": api_mode,
    }


def _bare_route() -> dict:
    """A process with no turn route: the door tests are about the client, not the seam.

    An empty main provider is what makes the built-in discovery chain reachable
    (``_discovery_chain_allowed``), which is the only way a door that was not read can show itself.
    """
    return {"provider": "", "model": "", "base_url": None, "api_key": None, "api_mode": None}


# ---------------------------------------------------------------------------
# Config: the real shipped blocks plus whatever this test pins
# ---------------------------------------------------------------------------

def _config(*, brief=True, aux=None, fallback_providers=(), main_pair=None):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    for task, block in (aux or {}).items():
        merged = dict(config["auxiliary"].get(task) or {})
        merged.update(block)
        config["auxiliary"][task] = merged
    if fallback_providers:
        config["fallback_providers"] = [dict(entry) for entry in fallback_providers]
    if main_pair is not None:
        provider, model = main_pair
        main_cfg = config.get("model")
        if not isinstance(main_cfg, dict):
            main_cfg = {}
        main_cfg["provider"], main_cfg["default"] = provider, model
        config["model"] = main_cfg
    config["discord"]["voice_brief"]["enabled"] = brief
    return config


def _pin(provider: str, model: str, *, chain=None, task_chain_only=None) -> dict:
    block = {"provider": provider, "model": model, "api_key": PIN_KEY}
    if chain is not None:
        block["fallback_chain"] = [dict(entry) for entry in chain]
    if task_chain_only is not None:
        block["task_chain_only"] = task_chain_only
    return block


TWO_FALLBACKS = (
    {"provider": "lane-a", "model": "modelA", "base_url": "http://127.0.0.1:2/v1"},
    {"provider": "lane-b", "model": "modelB", "base_url": "http://127.0.0.1:3/v1"},
)


# ---------------------------------------------------------------------------
# Provider edges: one fake client per endpoint; routing above them stays real
# ---------------------------------------------------------------------------

def _ok_response(text: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].message.tool_calls = None
    return response


def _exhausted_429(lane: str) -> Exception:
    exc = Exception(
        f"Error code: 429 - {{'error': {{'message': 'Weekly usage limit reached', "
        f"'type': 'usage_limit_reached', 'code': 'rate_limit_exceeded', 'lane': '{lane}'}}}}"
    )
    exc.status_code = 429
    return exc


def _auth_401(lane: str) -> Exception:
    exc = Exception(f"Error code: 401 - {{'error': {{'message': 'invalid api key', 'lane': '{lane}'}}}}")
    exc.status_code = 401
    return exc


def _lane(name: str, base_url: str, *, text: str | None = None, error: Exception | None = None):
    """A provider endpoint that records every request it receives."""

    async def create(**kwargs):
        ORDER.append(("model", name, kwargs.get("model")))
        if error is not None:
            raise error
        return _ok_response(text)

    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = create
    return client


def _response_text(response) -> str:
    return response.choices[0].message.content


class _Lanes:
    """The provider clients the real router resolves to, and what it asked for."""

    def __init__(self):
        self.by_provider: dict[str, tuple] = {}
        self.unbuildable: set[str] = set()
        self.session = None
        self.built: list[str] = []
        self.main_route_calls: list[tuple] = []

    def install(self, monkeypatch) -> None:
        real_resolve = auxiliary_client.resolve_provider_client

        def _resolve(provider, model=None, async_mode=False, raw_codex=False, *,
                     explicit_base_url=None, explicit_api_key=None, api_mode=None,
                     main_runtime=None, is_vision=False, task=None):
            label = str(provider or "").strip().lower()
            self.built.append(label)
            if label in self.unbuildable:
                return None, None
            if label in self.by_provider:
                return self.by_provider[label]
            return real_resolve(
                provider, model, async_mode, raw_codex=raw_codex,
                explicit_base_url=explicit_base_url, explicit_api_key=explicit_api_key,
                api_mode=api_mode, main_runtime=main_runtime, is_vision=is_vision, task=task,
            )

        def _main_route(provider, model, base_url, api_key, api_mode):
            self.main_route_calls.append((provider, model, base_url, api_key, api_mode))
            if self.session is None:
                return None
            return self.session, model, provider

        monkeypatch.setattr(auxiliary_client, "resolve_provider_client", _resolve)
        monkeypatch.setattr(auxiliary_client, "_try_main_provider_route", _main_route)
        monkeypatch.setattr(
            auxiliary_client, "_to_async_client", lambda client, model, **kwargs: (client, model),
        )
        # No real discovery probe ever runs: a test that needs the discovery rung installs its own.
        monkeypatch.setattr(auxiliary_client, "_get_provider_chain", list)


@pytest.fixture
def lanes(monkeypatch):
    """Install the provider edges; reset the aux client's caches so each test is its own process."""
    auxiliary_client._client_cache.clear()
    auxiliary_client._reset_aux_unhealthy_cache()
    installed = _Lanes()
    installed.install(monkeypatch)
    yield installed
    auxiliary_client._client_cache.clear()
    auxiliary_client._reset_aux_unhealthy_cache()


@pytest.fixture(autouse=True)
def _reset_order():
    ORDER.clear()
    yield
    ORDER.clear()


def _call(task: str, *, main_runtime: dict) -> object:
    """One real auxiliary call: resolution, ladder and quarantine walk are all the shipped ones."""
    async def _go():
        return await auxiliary_client.async_call_llm(
            task=task, main_runtime=main_runtime, messages=[{"role": "user", "content": "hi"}],
            temperature=0.2, route_info={},
        )

    return asyncio.new_event_loop().run_until_complete(_go())


def _ask(route: dict) -> str:
    return asyncio.new_event_loop().run_until_complete(
        voice_brief._ask_model(route, voice_brief.brief_messages("check the deploy logs"))
    )


# ---------------------------------------------------------------------------
# Gateway seam scaffolding
# ---------------------------------------------------------------------------

class _NoopAgent:
    """Agent stub: records that the turn started, returns immediately."""

    def __init__(self, *args, **kwargs):
        self.tools = []
        self.model = kwargs.get("model", "test-model")
        self.provider = kwargs.get("provider", "test-provider")
        self.session_id = kwargs.get("session_id", "session-1")
        self.context_compressor = None
        self.is_interrupted = False

    def run_conversation(self, user_message, conversation_history=None, task_id=None,
                         persist_user_message=None, persist_user_timestamp=None):
        ORDER.append(("turn",))
        return {"final_response": "Done.", "messages": [], "api_calls": 1, "completed": True}


class _EdgeAdapter:
    """Platform edge: playback + text post, recording the observed order."""

    def __init__(self):
        self._voice_text_channels = {GUILD_ID: CHAT_ID}
        self.posted: list[str] = []
        self.played: list[tuple] = []

    def voice_mixer_active(self, guild_id):
        return True

    def is_in_voice_channel(self, guild_id):
        return True

    def _should_auto_tts_for_chat(self, chat_id):
        return False

    def get_pending_message(self, session_key):
        return None

    async def play_in_voice_channel(self, guild_id, audio_path, strict=False):
        self.played.append((guild_id, audio_path, strict, os.path.isfile(audio_path)))
        ORDER.append(("play", guild_id, os.path.isfile(audio_path)))
        return True

    async def play_ack_in_voice(self, guild_id):
        ORDER.append(("ack", guild_id))
        return True

    async def send(self, chat_id, content, **kwargs):
        self.posted.append(content)
        ORDER.append(("post", chat_id, content))
        return SimpleNamespace(success=True, error=None)


class _MuteAdapter(_EdgeAdapter):
    """A platform edge whose text post always fails."""

    async def send(self, chat_id, content, **kwargs):
        ORDER.append(("post-failed", chat_id, content))
        raise RuntimeError("discord 500")


def _fake_tts(text, output_path=None, **_kwargs):
    ORDER.append(("synth", text))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(b"ID3fake")
    return json.dumps({"success": True, "file_path": output_path})


def _install_edges(monkeypatch, adapter=None, *, delivery=None) -> None:
    """Platform + TTS edges: the brief's synthesis and its two delivery legs never leave the test."""
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", _fake_tts)
    getter = delivery or (lambda self, source: adapter)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_delivery_adapter_for", getter)


@pytest.fixture
def edge(monkeypatch):
    """The platform and TTS edges; the model edge is the ``lanes`` fixture."""
    adapter = _EdgeAdapter()
    _install_edges(monkeypatch, adapter)
    return adapter


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD, chat_id=CHAT_ID, chat_type="group", user_id="u1",
        guild_id=str(GUILD_ID),
    )


def _make_runner(adapter=None) -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._gateway_loop = None
    runner._get_proxy_url = lambda: ""
    return runner


def _wire_turn(monkeypatch) -> None:
    """Everything the turn needs that is not config: the agent, the runtime and the tool roster."""
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: SESSION_MODEL)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: _session_runtime())
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoopAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    from hermes_cli import tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})


def _wire(monkeypatch, tmp_path, config, *, api_mode: str = SESSION_API_MODE) -> dict:
    """Both config reads the turn and the brief make, plus the runtime the turn resolves."""
    (tmp_path / "config.yaml").write_text("agent:\n  model: session-model\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    _wire_turn(monkeypatch)
    if api_mode != SESSION_API_MODE:
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: _session_runtime(api_mode))
    return config


def _run(runner) -> dict:
    async def _go():
        return await runner._run_agent(
            message="check the deploy logs",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="session-1",
            session_key="agent:main:discord:group:123",
            message_type=MessageType.VOICE,
        )

    return asyncio.new_event_loop().run_until_complete(_go())


def _model_lanes() -> list[str]:
    return [step[1] for step in ORDER if step[0] == "model"]


def _spy_runtime(monkeypatch) -> list[tuple]:
    """Record every runtime ``_resolve_session_agent_runtime`` returns, snapshot + live object."""
    seen: list[tuple] = []
    real_resolve = gateway_run.GatewayRunner._resolve_session_agent_runtime

    def _spy(self, **kwargs):
        model, runtime = real_resolve(self, **kwargs)
        seen.append((model, copy.deepcopy(runtime), runtime))
        return model, runtime

    monkeypatch.setattr(gateway_run.GatewayRunner, "_resolve_session_agent_runtime", _spy)
    return seen


def _assert_turn_route_untouched(runner, seen) -> None:
    assert seen, "the turn resolved no runtime at all"
    for resolved_model, snapshot, live in seen:
        assert live == snapshot, "the brief mutated the turn's runtime dict"
        assert resolved_model == SESSION_MODEL and live["provider"] == SESSION_PROVIDER
    assert runner._session_model_overrides == {}


def _model_logs(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == voice_brief.__name__]


# ---------------------------------------------------------------------------
# The helper: the key AND a non-auto resolved provider
# ---------------------------------------------------------------------------

def test_the_helper_requires_the_key_and_a_non_auto_resolved_provider(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: _config(aux={"compression": {"provider": "nous", "model": "m"}}))

    stops = auxiliary_client._task_stops_at_chain

    # The shipped voice_brief block sets the key with provider: auto.
    assert stops("voice_brief", "auto") is False
    assert stops("voice_brief", "") is False
    assert stops("voice_brief", None) is False
    # A resolved pin is what makes the key govern.
    assert stops("voice_brief", "nous") is True
    assert stops(None, "nous") is False


def test_the_helper_is_false_for_a_block_that_does_not_set_the_key(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={
            "compression": {"provider": "nous", "model": "m"},
            "vision": {"provider": "nous", "model": "m", "task_chain_only": False},
        }),
    )

    stops = auxiliary_client._task_stops_at_chain

    assert "task_chain_only" not in auxiliary_client._get_auxiliary_task_config("compression")
    assert stops("compression", "nous") is False
    assert stops("vision", "nous") is False, "an explicit false must not stop at the chain"


# ---------------------------------------------------------------------------
# Door 1: the openrouter/custom auto re-probe in _resolve_call_client
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["openrouter", "custom"])
def test_an_openrouter_or_custom_pin_consults_its_own_chain_before_any_auto_probe(
    lanes, monkeypatch, provider,
):
    lanes.unbuildable = {provider}
    lanes.by_provider = {"lane-a": (_lane("lane-a", "http://127.0.0.1:2/v1", text=BRIEF_OUTPUT), "modelA")}
    lanes.session = _lane("session", SESSION_BASE_URL, text="auto-detected answer")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"voice_brief": _pin(
            provider, "pin/model", chain=[{"provider": "lane-a", "model": "modelA"}],
        )}),
    )

    assert _ask(_bare_route()) == BRIEF_OUTPUT

    # The pin's own chain served it; the auto-detection walk was never reached.
    assert _model_lanes() == ["lane-a"]
    assert "auto" not in lanes.built


@pytest.mark.parametrize("provider", ["openrouter", "custom"])
def test_an_openrouter_or_custom_pin_with_no_chain_bypasses_instead_of_re_probing_auto(
    lanes, monkeypatch, provider,
):
    lanes.unbuildable = {provider}
    lanes.session = _lane("session", SESSION_BASE_URL, text="auto-detected answer")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"voice_brief": _pin(provider, "pin/model")}),
    )

    with pytest.raises(auxiliary_client.AuxiliaryClientUnavailable):
        _ask(_bare_route())

    # Nothing served the brief, and the main pair never saw a request.
    assert _model_lanes() == []
    assert "auto" not in lanes.built


def test_a_pin_without_the_key_keeps_the_auto_re_probe(lanes, monkeypatch):
    """The key is what widens the guard: without it, today's behaviour is unchanged."""
    lanes.unbuildable = {"openrouter"}
    lanes.session = _lane("session", SESSION_BASE_URL, text="auto-detected answer")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"voice_brief": _pin("openrouter", "pin/model", task_chain_only=False)}),
    )

    assert _ask(_bare_route()) == "auto-detected answer"

    assert _model_lanes() == ["session"]
    assert "auto" in lanes.built, "the auto re-probe must still run for a pin without the key"


def test_an_auto_resolution_with_no_client_keeps_the_auto_detection_walk(lanes, monkeypatch, caplog):
    """The shipped block is ``provider: auto`` + ``task_chain_only: true``: inert here."""
    lanes.session = None
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: _config())
    caplog.set_level(logging.INFO, logger=auxiliary_client.__name__)

    with pytest.raises(auxiliary_client.AuxiliaryClientUnavailable) as excinfo:
        _ask(_route())

    assert "No LLM provider configured" in str(excinfo.value)
    assert "auto" in lanes.built
    assert any("trying auto-detection chain" in record.getMessage() for record in caplog.records), (
        [record.getMessage() for record in caplog.records]
    )


# ---------------------------------------------------------------------------
# Door 2: the request-time main-agent net in _ladder_provider_fallback
# ---------------------------------------------------------------------------

def test_a_pinned_brief_exhausting_its_route_never_reaches_the_main_agent_model(lanes, monkeypatch):
    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", error=_exhausted_429("pin-a")), "pinModelA"),
        SESSION_PROVIDER: (_lane("session", SESSION_BASE_URL, text=MAIN_ANSWER), SESSION_MODEL),
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(
            aux={"voice_brief": _pin("pin-a", "pinModelA")},
            main_pair=(SESSION_PROVIDER, SESSION_MODEL),
        ),
    )

    with pytest.raises(Exception) as excinfo:
        _ask(_bare_route())

    assert getattr(excinfo.value, "status_code", None) == 429
    # The pin was tried and the main pair (the net at :7779) never saw the brief.
    assert _model_lanes() == ["pin-a"]


def test_a_task_that_does_not_set_the_key_still_uses_the_main_agent_model(lanes, monkeypatch):
    """Compression/vision keep the net: only a block that sets the key stops at its chain."""
    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", error=_exhausted_429("pin-a")), "pinModelA"),
        SESSION_PROVIDER: (_lane("session", SESSION_BASE_URL, text=MAIN_ANSWER), SESSION_MODEL),
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"compression": _pin("pin-a", "pinModelA")},
                        main_pair=(SESSION_PROVIDER, SESSION_MODEL)),
    )

    response = _call("compression", main_runtime=_bare_route())

    assert _response_text(response) == MAIN_ANSWER
    assert _model_lanes() == ["pin-a", "session"]


def test_an_auth_error_on_a_pinned_task_with_a_chain_never_reaches_the_main_agent_model(
    lanes, monkeypatch,
):
    """``explicit_auth_with_task_chain`` survives the rename: auth stops at the chain too."""
    discovery = _lane("discovery", "http://127.0.0.1:12/v1", text="discovery answer")
    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", error=_auth_401("pin-a")), "pinModelA"),
        "pin-b": (_lane("pin-b", "http://127.0.0.1:10/v1", error=_auth_401("pin-b")), "pinModelB"),
    }
    monkeypatch.setattr(auxiliary_client, "_get_provider_chain",
                        lambda: [("discovery", lambda: (discovery, "discoveryModel"))])
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"compression": _pin(
            "pin-a", "pinModelA", chain=[{"provider": "pin-b", "model": "pinModelB"}],
        )}),
    )

    with pytest.raises(Exception) as excinfo:
        _call("compression", main_runtime=_bare_route())

    assert getattr(excinfo.value, "status_code", None) == 401
    assert _model_lanes() == ["pin-a", "pin-b"]


# ---------------------------------------------------------------------------
# Door 3: the quarantine walk's discovery rung in _next_fallback_after_quarantine
# ---------------------------------------------------------------------------

def test_a_quarantined_pinned_chain_stops_before_the_discovery_rung(lanes, monkeypatch):
    discovery = _lane("discovery", "http://127.0.0.1:12/v1", text="discovery answer")
    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", error=_exhausted_429("pin-a")), "pinModelA"),
        "pin-b": (_lane("pin-b", "http://127.0.0.1:10/v1", error=_exhausted_429("pin-b")), "pinModelB"),
    }
    monkeypatch.setattr(auxiliary_client, "_get_provider_chain",
                        lambda: [("discovery", lambda: (discovery, "discoveryModel"))])
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"voice_brief": _pin(
            "pin-a", "pinModelA", chain=[{"provider": "pin-b", "model": "pinModelB"}],
        )}),
    )

    with pytest.raises(Exception) as excinfo:
        _ask(_bare_route())

    assert getattr(excinfo.value, "status_code", None) == 429
    assert _model_lanes() == ["pin-a", "pin-b"], "the discovery rung served a pinned task"


def test_an_auto_call_after_a_quarantine_keeps_the_main_chain_and_discovery_rungs(lanes, monkeypatch):
    """The shipped block is inert on auto: both rungs of the quarantine walk stay reachable."""
    discovery = _lane("discovery", "http://127.0.0.1:12/v1", text=BRIEF_OUTPUT)
    lanes.session = _lane("session", SESSION_BASE_URL, error=_exhausted_429("session"))
    lanes.by_provider = {
        "lane-a": (_lane("lane-a", "http://127.0.0.1:2/v1", error=_exhausted_429("A")), "modelA"),
    }
    monkeypatch.setattr(auxiliary_client, "_get_provider_chain",
                        lambda: [("discovery", lambda: (discovery, "discoveryModel"))])
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(fallback_providers=[{"provider": "lane-a", "model": "modelA",
                                             "base_url": "http://127.0.0.1:2/v1"}]),
    )

    assert _ask(_bare_route()) == BRIEF_OUTPUT

    assert _model_lanes() == ["session", "lane-a", "discovery"]


# ---------------------------------------------------------------------------
# Door 4: the vision auto re-probe in _resolve_call_client
#
# Not reachable from the gateway seam — the brief's own task is ``voice_brief``, not ``vision`` —
# so these two drive ``auxiliary_client.async_call_llm`` directly, over the same ``lanes`` edges
# (``resolve_provider_client`` returns None for a provider whose credential is missing).
# ---------------------------------------------------------------------------

VISION_ANSWER = "auto vision answer"


def _ask_vision() -> str:
    async def _go():
        return await auxiliary_client.async_call_llm(
            task="vision", main_runtime=_route(), route_info={},
            messages=[{"role": "user", "content": "what is in this image?"}],
        )

    return _response_text(asyncio.new_event_loop().run_until_complete(_go()))


def _auto_vision_lane(lanes) -> None:
    """Make the ``auto`` vision step (main provider first) servable, so the probe is observable."""
    lane = _lane("session", SESSION_BASE_URL, text=VISION_ANSWER)
    lanes.session = lane
    lanes.by_provider = {SESSION_PROVIDER: (lane, SESSION_MODEL)}


def test_a_pinned_vision_task_with_the_key_skips_the_auto_re_probe(lanes, monkeypatch):
    """The pin cannot be built, so the route fails: the vision ``auto`` backends never serve it."""
    lanes.unbuildable = {"pin-a"}
    _auto_vision_lane(lanes)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"vision": _pin("pin-a", "pinModelA", task_chain_only=True)}),
    )

    with pytest.raises(auxiliary_client.AuxiliaryClientUnavailable):
        _ask_vision()

    assert _model_lanes() == [], "an auto vision backend served a pinned vision task"


def test_a_pinned_vision_task_without_the_key_keeps_the_auto_re_probe(lanes, monkeypatch):
    """The key is what gates the vision arm: without it, today's behaviour is unchanged."""
    lanes.unbuildable = {"pin-a"}
    _auto_vision_lane(lanes)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: _config(aux={"vision": _pin("pin-a", "pinModelA", task_chain_only=False)}),
    )

    assert _ask_vision() == VISION_ANSWER

    assert _model_lanes() == ["session"]


# ---------------------------------------------------------------------------
# The seam: a pinned brief on its own route, a bypass, and the untouched turn
# ---------------------------------------------------------------------------

def test_a_pinned_brief_is_served_by_the_pin_and_the_turn_keeps_its_session_pair(
    lanes, edge, monkeypatch, tmp_path,
):
    lanes.session = _lane("session", SESSION_BASE_URL, text="turn route answer")
    lanes.by_provider = {"pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", text=BRIEF_OUTPUT), "pinModelA")}
    _wire(monkeypatch, tmp_path, _config(aux={"voice_brief": _pin("pin-a", "pinModelA")}))
    seen = _spy_runtime(monkeypatch)
    runner = _make_runner(adapter=edge)

    result = _run(runner)

    assert _model_lanes() == ["pin-a"]
    assert edge.posted == [BRIEF_OUTPUT] and edge.played[0][2] is True
    assert result["final_response"] == "Done."
    assert [step[0] for step in ORDER][-1] == "turn"
    # The brief's own route, not the session pair, and no auto route was resolved at all.
    assert lanes.main_route_calls == []
    _assert_turn_route_untouched(runner, seen)


def test_a_429_on_the_pinned_primary_lands_on_the_next_chain_entry(lanes, edge, monkeypatch, tmp_path):
    lanes.session = _lane("session", SESSION_BASE_URL, text="turn route answer")
    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", error=_exhausted_429("pin-a")), "pinModelA"),
        "pin-b": (_lane("pin-b", "http://127.0.0.1:10/v1", text=BRIEF_OUTPUT), "pinModelB"),
    }
    _wire(monkeypatch, tmp_path, _config(aux={"voice_brief": _pin(
        "pin-a", "pinModelA", chain=[{"provider": "pin-b", "model": "pinModelB"}],
    )}))
    runner = _make_runner(adapter=edge)

    result = _run(runner)

    assert _model_lanes() == ["pin-a", "pin-b"]
    assert edge.posted == [BRIEF_OUTPUT]
    assert result["final_response"] == "Done."


def test_a_pin_whose_client_cannot_be_built_bypasses_with_one_message(
    lanes, edge, monkeypatch, tmp_path,
):
    lanes.session = _lane("session", SESSION_BASE_URL, text="turn route answer")
    lanes.unbuildable = {"pin-a"}
    _wire(monkeypatch, tmp_path, _config(aux={"voice_brief": _pin("pin-a", "pinModelA")}))
    seen = _spy_runtime(monkeypatch)
    runner = _make_runner(adapter=edge)

    result = _run(runner)

    # No audio, one short message naming the failed provider and its failure class.
    assert edge.played == []
    assert len(edge.posted) == 1
    message = edge.posted[0]
    assert "pin-a/pinModelA" in message
    assert "unavailable" in message
    assert PIN_KEY not in message and len(message) < 2000
    # The turn ran on its own route, and no provider was called on the brief's behalf.
    assert result["final_response"] == "Done."
    assert [step[0] for step in ORDER][-1] == "turn"
    assert _model_lanes() == []
    _assert_turn_route_untouched(runner, seen)


def test_an_openrouter_pin_whose_client_and_chain_cannot_be_built_bypasses_the_same_way(
    lanes, edge, monkeypatch, tmp_path,
):
    lanes.session = _lane("session", SESSION_BASE_URL, text="turn route answer")
    lanes.unbuildable = {"openrouter", "pin-b"}
    _wire(monkeypatch, tmp_path, _config(aux={"voice_brief": _pin(
        "openrouter", "vendor/model", chain=[{"provider": "pin-b", "model": "pinModelB"}],
    )}))
    runner = _make_runner(adapter=edge)

    result = _run(runner)

    assert edge.played == []
    assert len(edge.posted) == 1 and "openrouter/vendor/model" in edge.posted[0]
    assert result["final_response"] == "Done."
    assert _model_lanes() == []
    assert "auto" not in lanes.built


def test_a_bypass_message_that_cannot_be_posted_leaves_the_turn_running(
    lanes, monkeypatch, tmp_path, caplog,
):
    adapter = _MuteAdapter()
    _install_edges(monkeypatch, adapter)
    lanes.session = _lane("session", SESSION_BASE_URL, text="turn route answer")
    lanes.unbuildable = {"pin-a"}
    _wire(monkeypatch, tmp_path, _config(aux={"voice_brief": _pin("pin-a", "pinModelA")}))
    caplog.set_level(logging.WARNING, logger=gateway_run.logger.name)
    runner = _make_runner(adapter=adapter)

    result = _run(runner)

    assert result["final_response"] == "Done."
    assert [step[0] for step in ORDER][-1] == "turn"
    assert _failed_posts() == 1
    assert any("bypass" in record.getMessage().lower() for record in caplog.records), (
        [record.getMessage() for record in caplog.records]
    )


def _failed_posts() -> int:
    return len([step for step in ORDER if step[0] == "post-failed"])


def test_an_auto_brief_whose_route_and_chain_all_fail_refuses_the_turn(
    lanes, edge, monkeypatch, tmp_path,
):
    lanes.session = _lane("session", SESSION_BASE_URL, error=_exhausted_429("session"))
    lanes.by_provider = {
        "lane-a": (_lane("lane-a", "http://127.0.0.1:2/v1", error=_exhausted_429("A")), "modelA"),
        "lane-b": (_lane("lane-b", "http://127.0.0.1:3/v1", error=_exhausted_429("B")), "modelB"),
    }
    _wire(monkeypatch, tmp_path, _config(fallback_providers=TWO_FALLBACKS))
    seen = _spy_runtime(monkeypatch)
    runner = _make_runner(adapter=edge)

    result = _run(runner)

    # The shipped refusal stands: visible text, no turn, and no bypass message.
    assert voice_brief.FAILED_TEXT in result["final_response"]
    assert "turn" not in [step[0] for step in ORDER]
    assert edge.posted == []
    assert _model_lanes() == ["session", "lane-a", "lane-b"]
    _assert_turn_route_untouched(runner, seen)


# ---------------------------------------------------------------------------
# Two profiles in one process
# ---------------------------------------------------------------------------

def _profile_home(tmp_path, name: str, *, provider: str, model: str, enabled: bool) -> Path:
    home = tmp_path / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "discord:\n"
        "  voice_brief:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
        "auxiliary:\n"
        "  voice_brief:\n"
        f"    provider: \"{provider}\"\n"
        f"    model: \"{model}\"\n"
        f"    api_key: \"{PIN_KEY}\"\n",
        encoding="utf-8",
    )
    return home


def test_two_profiles_in_one_process_each_resolve_their_own_pin(lanes, monkeypatch, tmp_path):
    from hermes_cli.config import load_config_readonly

    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", text="Profile A understood."), "pinModelA"),
        "pin-b": (_lane("pin-b", "http://127.0.0.1:10/v1", text="Profile B understood."), "pinModelB"),
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: load_config_readonly())
    _wire_turn(monkeypatch)
    home_a = _profile_home(tmp_path, "profile-a", provider="pin-a", model="pinModelA", enabled=True)
    home_b = _profile_home(tmp_path, "profile-b", provider="pin-b", model="pinModelB", enabled=True)

    adapter_a, adapter_b = _EdgeAdapter(), _EdgeAdapter()
    current = {"adapter": adapter_a}
    _install_edges(monkeypatch, delivery=lambda self, source: current["adapter"])
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(gateway_run, "_hermes_home", home_a)
    monkeypatch.setattr(gateway_run, "_env_path", home_a / ".env")
    with _profile_runtime_scope(home_a, hydrate_secrets=False):
        result_a = _run(_make_runner(adapter=adapter_a))
        lanes_a = list(_model_lanes())
        ORDER.clear()

    current["adapter"] = adapter_b
    monkeypatch.setattr(gateway_run, "_hermes_home", home_b)
    monkeypatch.setattr(gateway_run, "_env_path", home_b / ".env")
    with _profile_runtime_scope(home_b, hydrate_secrets=False):
        result_b = _run(_make_runner(adapter=adapter_b))
        lanes_b = list(_model_lanes())

    assert lanes_a == ["pin-a"] and adapter_a.posted == ["Profile A understood."]
    assert lanes_b == ["pin-b"] and adapter_b.posted == ["Profile B understood."]
    assert result_a["final_response"] == result_b["final_response"] == "Done."


def test_one_profiles_switch_does_not_enable_the_others_brief(lanes, monkeypatch, tmp_path):
    from hermes_cli.config import load_config_readonly

    lanes.by_provider = {
        "pin-a": (_lane("pin-a", "http://127.0.0.1:9/v1", text="Profile A understood."), "pinModelA"),
        "pin-b": (_lane("pin-b", "http://127.0.0.1:10/v1", text="Profile B understood."), "pinModelB"),
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: load_config_readonly())
    _wire_turn(monkeypatch)
    home_a = _profile_home(tmp_path, "profile-a", provider="pin-a", model="pinModelA", enabled=True)
    home_b = _profile_home(tmp_path, "profile-b", provider="pin-b", model="pinModelB", enabled=False)

    adapter_a, adapter_b = _EdgeAdapter(), _EdgeAdapter()
    current = {"adapter": adapter_a}
    _install_edges(monkeypatch, delivery=lambda self, source: current["adapter"])
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(gateway_run, "_hermes_home", home_a)
    monkeypatch.setattr(gateway_run, "_env_path", home_a / ".env")
    with _profile_runtime_scope(home_a, hydrate_secrets=False):
        _run(_make_runner(adapter=adapter_a))
        lanes_a = list(_model_lanes())
        ORDER.clear()

    current["adapter"] = adapter_b
    monkeypatch.setattr(gateway_run, "_hermes_home", home_b)
    monkeypatch.setattr(gateway_run, "_env_path", home_b / ".env")
    with _profile_runtime_scope(home_b, hydrate_secrets=False):
        result_b = _run(_make_runner(adapter=adapter_b))
        lanes_b = list(_model_lanes())

    assert lanes_a == ["pin-a"] and adapter_a.posted == ["Profile A understood."]
    assert lanes_b == [] and adapter_b.posted == [] and adapter_b.played == []
    assert result_b["final_response"] == "Done."
