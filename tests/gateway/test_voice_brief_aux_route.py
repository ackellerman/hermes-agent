"""The pre-turn brief asks the auxiliary router for its OWN task and its own route.

``_ask_model`` no longer hands ``async_call_llm`` explicit ``provider``/``model``/``base_url``/
``api_key`` arguments: it asks for ``task="voice_brief"`` and passes the turn's live runtime as
``main_runtime``. So the shipped ``provider: auto`` block lands on the session pair (transport
included) and, when that pair is exhausted, walks the profile's ``fallback_providers`` — the
auxiliary client's existing chain, no chain reader in ``gateway/``. One budget governs the call:
``_get_task_timeout("voice_brief", MODEL_TIMEOUT_S)`` bounds the outer ``wait_for`` and the router
applies the same configured value to the request, so no ``timeout=`` kwarg is passed.

Real seams: ``GatewayRunner._run_agent`` → ``_voice_brief_before_turn``, the real
``deliver_preturn_brief`` and ``_ask_model``, and the real ``agent.auxiliary_client`` route
resolution and recovery ladder. Fakes live only at the edges: the provider clients, the TTS
provider and the platform adapter (playback / text post).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import threading
import time
import types
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


# ---------------------------------------------------------------------------
# Config: the real shipped block plus whatever this test pins
# ---------------------------------------------------------------------------

def _config(*, brief=True, timeout=None, fallback_providers=()):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    config = copy.deepcopy(DEFAULT_CONFIG)
    block = dict(config["auxiliary"]["voice_brief"])
    if timeout is not None:
        block["timeout"] = timeout
    config["auxiliary"]["voice_brief"] = block
    if fallback_providers:
        config["fallback_providers"] = [dict(entry) for entry in fallback_providers]
    config["discord"]["voice_brief"]["enabled"] = brief
    return config


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


class _Lanes:
    """The provider clients the real router resolves to, and what it asked for."""

    def __init__(self):
        self.session = None
        self.by_provider: dict[str, tuple] = {}
        self.main_route_calls: list[tuple] = []
        self.fallback_resolutions: list[dict] = []

    def install(self, monkeypatch) -> None:
        def _main_route(provider, model, base_url, api_key, api_mode):
            self.main_route_calls.append((provider, model, base_url, api_key, api_mode))
            if self.session is None:
                return None
            return self.session, model, provider

        def _fallback_entry(entry):
            self.fallback_resolutions.append(dict(entry))
            return self.by_provider[entry["provider"]]

        monkeypatch.setattr(auxiliary_client, "_try_main_provider_route", _main_route)
        monkeypatch.setattr(auxiliary_client, "_resolve_fallback_entry", _fallback_entry)
        monkeypatch.setattr(
            auxiliary_client, "_to_async_client", lambda client, model, **kwargs: (client, model),
        )


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


@pytest.fixture
def edge(monkeypatch):
    """The platform and TTS edges; the model edge is the ``lanes`` fixture."""
    adapter = _EdgeAdapter()

    def fake_tts(text, output_path=None, **_kwargs):
        ORDER.append(("synth", text))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"ID3fake")
        return json.dumps({"success": True, "file_path": output_path})

    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_delivery_adapter_for", lambda self, source: adapter)
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


def _wire(monkeypatch, tmp_path, config, *, api_mode: str = SESSION_API_MODE) -> dict:
    """Both config reads the turn and the brief make, plus the runtime the turn resolves."""
    (tmp_path / "config.yaml").write_text("agent:\n  model: session-model\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: SESSION_MODEL)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: _session_runtime(api_mode))
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoopAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    from hermes_cli import tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
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


def _deliver(adapter, route: dict | None = None) -> str:
    async def _go():
        return await voice_brief.deliver_preturn_brief(
            adapter=adapter, guild_id=GUILD_ID, chat_id=CHAT_ID, message="check the deploy logs",
            route=route if route is not None else _route(),
        )

    return asyncio.new_event_loop().run_until_complete(_go())


def _model_logs(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == voice_brief.__name__]


# ---------------------------------------------------------------------------
# The call contract: the task, the turn's runtime, and no route kwargs of its own
# ---------------------------------------------------------------------------

def test_ask_model_asks_for_the_voice_brief_task_without_its_own_route_kwargs(monkeypatch):
    endpoint = AsyncMock(return_value=_ok_response("I'll check the logs."))
    monkeypatch.setattr(auxiliary_client, "async_call_llm", endpoint)

    asyncio.run(voice_brief._ask_model(_route(), voice_brief.brief_messages("check the logs")))

    kwargs = endpoint.await_args.kwargs
    assert kwargs["task"] == "voice_brief"
    # Explicit route args outrank auxiliary.<task>.*, so passing any of them makes a pin unreachable.
    for key in ("provider", "model", "base_url", "api_key"):
        assert key not in kwargs
    # One budget: the router applies auxiliary.voice_brief.timeout itself.
    assert "timeout" not in kwargs
    assert isinstance(kwargs["route_info"], dict)
    assert kwargs["messages"][-1]["content"].startswith("The user said this by voice:")
    assert kwargs["reasoning_config"] == {"enabled": False}
    assert "tools" not in kwargs and "max_tokens" not in kwargs


def test_ask_model_hands_the_turn_runtime_to_the_router(monkeypatch):
    endpoint = AsyncMock(return_value=_ok_response("I'll check the logs."))
    monkeypatch.setattr(auxiliary_client, "async_call_llm", endpoint)
    route = _route()

    asyncio.run(voice_brief._ask_model(route, voice_brief.brief_messages("check the logs")))

    assert endpoint.await_args.kwargs["main_runtime"] == {
        "provider": SESSION_PROVIDER, "model": SESSION_MODEL, "base_url": SESSION_BASE_URL,
        "api_key": SESSION_API_KEY, "api_mode": SESSION_API_MODE,
    }


# ---------------------------------------------------------------------------
# auto: the session pair, then the profile's own fallback chain
# ---------------------------------------------------------------------------

def test_auto_brief_lands_on_the_session_pair(lanes, monkeypatch, caplog):
    lanes.session = _lane("session", SESSION_BASE_URL, text=BRIEF_OUTPUT)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: _config())
    caplog.set_level(logging.INFO, logger=voice_brief.__name__)

    text = asyncio.run(voice_brief._ask_model(_route(), voice_brief.brief_messages("check the logs")))

    assert text == BRIEF_OUTPUT
    # The request went to the session's own endpoint and model, not to a chain entry.
    assert [step[1:] for step in ORDER] == [("session", SESSION_MODEL)]
    assert lanes.main_route_calls == [(
        SESSION_PROVIDER, SESSION_MODEL, SESSION_BASE_URL, SESSION_API_KEY, SESSION_API_MODE,
    )]
    assert lanes.fallback_resolutions == []
    logs = _model_logs(caplog)
    assert any(
        f"requested {SESSION_PROVIDER}/{SESSION_MODEL}" in line
        and f"served by {SESSION_PROVIDER}/{SESSION_MODEL}" in line
        for line in logs
    ), logs


def test_exhausted_session_route_walks_the_profiles_fallback_providers(lanes, edge, monkeypatch, caplog):
    lanes.session = _lane("session", SESSION_BASE_URL, error=_exhausted_429("session"))
    lanes.by_provider = {
        "lane-a": (_lane("lane-a", "http://127.0.0.1:2/v1", error=_exhausted_429("A")), "modelA"),
        "lane-b": (_lane("lane-b", "http://127.0.0.1:3/v1", text=BRIEF_OUTPUT), "modelB"),
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _config(fallback_providers=TWO_FALLBACKS),
    )
    caplog.set_level(logging.INFO, logger=voice_brief.__name__)

    assert _deliver(edge) == BRIEF_OUTPUT

    # Session pair → fallback_providers[0] → fallback_providers[1]; each lane tried once.
    assert [step[1:] for step in ORDER] == [("session", SESSION_MODEL), ("lane-a", "modelA"),
                                            ("lane-b", "modelB"), (BRIEF_OUTPUT,),
                                            (GUILD_ID, True), (CHAT_ID, BRIEF_OUTPUT)]
    logs = _model_logs(caplog)
    # The whole served-by tail: "lane-b" alone is already the lane NAME and sits in the requested
    # half of this line, so a check that ignores the " via " clause proves nothing.
    assert any("served by lane-b/modelB via lane-b" in line for line in logs), logs


# ---------------------------------------------------------------------------
# Through the real pre-turn seam: the order and the untouched turn route
# ---------------------------------------------------------------------------

def _fallback_wired(monkeypatch, tmp_path, lanes, edge, *, api_mode=SESSION_API_MODE):
    lanes.session = _lane("session", SESSION_BASE_URL, error=_exhausted_429("session"))
    lanes.by_provider = {
        "lane-a": (_lane("lane-a", "http://127.0.0.1:2/v1", error=_exhausted_429("A")), "modelA"),
        "lane-b": (_lane("lane-b", "http://127.0.0.1:3/v1", text=BRIEF_OUTPUT), "modelB"),
    }
    return _wire(
        monkeypatch, tmp_path,
        _config(fallback_providers=TWO_FALLBACKS), api_mode=api_mode,
    )


def test_a_fallback_served_brief_keeps_the_delivery_order(lanes, edge, monkeypatch, tmp_path):
    _fallback_wired(monkeypatch, tmp_path, lanes, edge)

    result = _run(_make_runner(adapter=edge))

    assert [step[0] for step in ORDER] == ["model", "model", "model", "synth", "play", "post", "turn"]
    assert [step[1] for step in ORDER if step[0] == "model"] == ["session", "lane-a", "lane-b"]
    assert edge.posted == [BRIEF_OUTPUT] and edge.played[0][2] is True
    assert result["final_response"] == "Done."


def test_the_brief_does_not_change_the_turn_route(lanes, edge, monkeypatch, tmp_path):
    _fallback_wired(monkeypatch, tmp_path, lanes, edge)
    seen: list[tuple] = []
    real_resolve = gateway_run.GatewayRunner._resolve_session_agent_runtime

    def _spy(self, **kwargs):
        model, runtime = real_resolve(self, **kwargs)
        seen.append((model, copy.deepcopy(runtime), runtime))
        return model, runtime

    monkeypatch.setattr(gateway_run.GatewayRunner, "_resolve_session_agent_runtime", _spy)
    runner = _make_runner(adapter=edge)
    store_attrs = set(vars(runner.session_store))

    _run(runner)

    assert seen, "the turn resolved no runtime at all"
    for resolved_model, snapshot, live in seen:
        assert live == snapshot, "the brief mutated the turn's runtime dict"
        assert resolved_model == SESSION_MODEL and live["provider"] == SESSION_PROVIDER
    assert runner._session_model_overrides == {}
    assert set(vars(runner.session_store)) == store_attrs


def test_the_brief_keeps_the_sessions_api_mode(lanes, edge, monkeypatch, tmp_path):
    _fallback_wired(monkeypatch, tmp_path, lanes, edge, api_mode="anthropic_messages")

    _run(_make_runner(adapter=edge))

    # The auto route is resolved with the session's transport, so an auto brief speaks on it.
    assert lanes.main_route_calls[0][4] == "anthropic_messages"


# ---------------------------------------------------------------------------
# One budget: the outer wait_for and the router read the same configured number
# ---------------------------------------------------------------------------

def test_the_wait_for_bound_follows_the_configured_task_timeout(edge, monkeypatch):
    async def _hang(route, messages):
        await asyncio.sleep(3)

    bounds: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _record(awaitable, *, timeout):
        bounds.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(voice_brief, "_ask_model", _hang)
    monkeypatch.setattr(voice_brief.asyncio, "wait_for", _record)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: _config(timeout=0.05))

    started = time.monotonic()
    with pytest.raises(voice_brief.VoiceBriefError, match="model call failed"):
        _deliver(edge)
    elapsed = time.monotonic() - started
    # The exact bound the outer wait_for received, not just "less than the 15s constant"...
    assert bounds == [0.05], bounds
    # ...and the call really was abandoned at it.
    assert elapsed < 1.5, f"the brief waited {elapsed:.2f}s for a 0.05s configuration"
    assert ORDER == []


def test_the_model_budget_is_the_shipped_fifteen_seconds_and_follows_the_block(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: _config())
    assert voice_brief._model_timeout() == voice_brief.MODEL_TIMEOUT_S == 15.0

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: _config(timeout=42))
    assert voice_brief._model_timeout() == 42.0
