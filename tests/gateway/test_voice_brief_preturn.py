"""Pre-turn spoken brief for Discord voice-input turns (opt-in per profile).

Real seams: ``GatewayRunner._run_agent`` → ``_run_agent_inner`` (the pre-turn position), the real
``gateway.voice_brief`` delivery path, and the real ``DiscordAdapter.play_in_voice_channel``
strict flag. Fakes live only at the edges: the model endpoint, the TTS provider, the platform
adapter (playback / text post).

Order is the contract: model sentence → playback complete → text post → turn starts. Any failure
means no turn. Disabled or typed turns are untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageType
from gateway.session import SessionSource
import gateway.voice_brief as voice_brief

ORDER: list = []
EDGE: list = []
MODEL_OUTPUT = "You want me to check the deployment logs and report the error."
CHAT_ID = "123"
GUILD_ID = 111


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
    """Platform edge: playback + text post + ack, recording observed order."""

    def __init__(self, *, play_ok=True, post_ok=True, hold=None):
        self._voice_text_channels = {GUILD_ID: CHAT_ID}
        self.play_ok, self.post_ok, self.hold = play_ok, post_ok, hold
        self.play_started = asyncio.Event()
        self.posted: list[str] = []
        self.played: list[tuple] = []
        self.acks: list[int] = []

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
        if self.hold is not None:
            self.play_started.set()
            await self.hold.wait()
        return self.play_ok

    async def play_ack_in_voice(self, guild_id):
        self.acks.append(guild_id)
        ORDER.append(("ack", guild_id))
        return True

    async def send(self, chat_id, content, **kwargs):
        self.posted.append(content)
        ORDER.append(("post", chat_id, content))
        return SimpleNamespace(success=self.post_ok, error=None if self.post_ok else "post failed")


@pytest.fixture(autouse=True)
def _reset_order():
    ORDER.clear()
    yield
    ORDER.clear()


@pytest.fixture
def edge(monkeypatch):
    """Fake the three edges and answer with ``MODEL_OUTPUT``."""
    adapter = _EdgeAdapter()

    async def fake_ask(route, messages):
        ORDER.append(("model", route.get("provider"), route.get("model")))
        assert messages and "Restate" in messages[-1]["content"]
        if getattr(monkeypatch, "model_raises", False):
            raise RuntimeError("model exploded")
        return MODEL_OUTPUT

    def fake_tts(text, output_path=None, **_kwargs):
        ORDER.append(("synth", text))
        if getattr(monkeypatch, "tts_fails", False):
            return json.dumps({"success": False, "error": "tts exploded"})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"ID3fake")
        return json.dumps({"success": True, "file_path": output_path})

    import tools.tts_tool as tts_tool

    monkeypatch.setattr(voice_brief, "_ask_model", fake_ask)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_delivery_adapter_for", lambda self, source: adapter)
    EDGE[:] = [adapter]
    return adapter


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD, chat_id=CHAT_ID, chat_type="group", user_id="u1",
        guild_id=str(GUILD_ID),
    )


def _make_runner(brief_enabled: bool = True, proxy: str = "", adapter=None):
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
    runner._get_proxy_url = lambda: proxy
    return runner


def _config(brief_enabled: bool) -> dict:
    return {
        "agent": {"model": "test-model"},
        "discord": {"voice_brief": {"enabled": brief_enabled}},
    }


def _wire(monkeypatch, tmp_path, *, brief_enabled=True, proxy="", api_mode="chat_completions"):
    (tmp_path / "config.yaml").write_text("agent:\n  model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: _config(brief_enabled))
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "test-model")
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "openrouter", "api_mode": api_mode,
                 "base_url": "https://openrouter.ai/api/v1", "api_key": "k"},
    )
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _NoopAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    from hermes_cli import tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})


def _run(runner, message_type):
    if EDGE:
        runner.adapters = {Platform.DISCORD: EDGE[0]}

    async def _go():
        return await runner._run_agent(
            message="check the deploy logs",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="session-1",
            session_key="agent:main:discord:group:123",
            message_type=message_type,
        )

    return asyncio.new_event_loop().run_until_complete(_go())


# ---------------------------------------------------------------------------
# Order / success
# ---------------------------------------------------------------------------

def test_brief_speaks_and_posts_before_the_turn(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    result = _run(_make_runner(), MessageType.VOICE)

    assert [step[0] for step in ORDER] == ["model", "synth", "play", "post", "turn"]
    assert result["final_response"] == "Done."
    assert edge.posted == [MODEL_OUTPUT]
    assert edge.played and edge.played[0][0] == GUILD_ID and edge.played[0][2] is True
    assert edge.played[0][3] is True  # audio existed at playback time


def test_brief_is_the_models_own_sentence_not_a_canned_phrase(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    _run(_make_runner(), MessageType.VOICE)

    spoken = edge.played and ORDER[1][1]
    assert spoken == MODEL_OUTPUT
    assert edge.posted == [MODEL_OUTPUT]
    canned = {"Let me look into that.", "One moment.", "Checking on that now.", "Give it a sec.", "On it."}
    assert edge.posted[0] not in canned


def test_long_model_output_is_capped(edge, monkeypatch, tmp_path):
    global MODEL_OUTPUT
    MODEL_OUTPUT = "You want me to " + ("check every deployment log entry and then " * 20) + "report back."
    try:
        _wire(monkeypatch, tmp_path)
        _run(_make_runner(), MessageType.VOICE)
    finally:
        MODEL_OUTPUT = "You want me to check the deployment logs and report the error."
    posted = edge.posted[0]
    assert len(posted) <= 200
    assert posted.endswith("\u2026")
    assert ORDER[1][1] == posted  # spoken text is the same capped sentence


def test_empty_model_output_refuses_instead_of_inventing_a_phrase(edge, monkeypatch, tmp_path):
    global MODEL_OUTPUT
    MODEL_OUTPUT = "   \n  "
    try:
        _wire(monkeypatch, tmp_path)
        result = _run(_make_runner(), MessageType.VOICE)
    finally:
        MODEL_OUTPUT = "You want me to check the deployment logs and report the error."
    assert ("turn",) not in ORDER
    assert "no sentence returned" in result["final_response"]
    assert edge.played == [] and edge.posted == []


def test_clean_brief_unwraps_quotes_and_one_lines_whitespace():
    assert voice_brief.clean_brief('  "You want the\ndeploy logs."  ') == "You want the deploy logs."
    assert voice_brief.clean_brief("\u201cHi\u201d") == "Hi"
    assert voice_brief.clean_brief("   ") == ""


# ---------------------------------------------------------------------------
# Ordering under suspension: playback is awaited before the turn
# ---------------------------------------------------------------------------

def test_playback_held_means_the_turn_never_starts(edge, monkeypatch, tmp_path):
    held = asyncio.Event()
    edge.hold = held
    _wire(monkeypatch, tmp_path)
    runner = _make_runner()
    runner.adapters = {Platform.DISCORD: edge}

    async def _go():
        task = asyncio.create_task(runner._run_agent(
            message="check the deploy logs", context_prompt="", history=[], source=_source(),
            session_id="session-1", session_key="agent:main:discord:group:123",
            message_type=MessageType.VOICE,
        ))
        await asyncio.wait_for(edge.play_started.wait(), timeout=5)
        await asyncio.sleep(0.1)
        assert ("turn",) not in ORDER          # blocked on playback, turn not started
        assert not edge.posted                 # text post is after playback, not before
        held.set()
        return await task

    result = asyncio.new_event_loop().run_until_complete(_go())
    assert [step[0] for step in ORDER] == ["model", "synth", "play", "post", "turn"]
    assert result["final_response"] == "Done."


# ---------------------------------------------------------------------------
# Failures refuse the turn (visible text, no work)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("failure", ["model", "tts", "playback", "post"])
def test_any_failure_refuses_the_turn_visibly(edge, monkeypatch, tmp_path, failure):
    _wire(monkeypatch, tmp_path)
    if failure == "model":
        monkeypatch.model_raises = True
    elif failure == "tts":
        monkeypatch.tts_fails = True
    elif failure == "playback":
        edge.play_ok = False
    else:
        edge.post_ok = False

    result = _run(_make_runner(), MessageType.VOICE)

    assert ("turn",) not in ORDER
    assert voice_brief.FAILED_TEXT in result["final_response"]
    assert "Done." not in result["final_response"]
    if failure == "playback":
        assert edge.posted == []  # post never happens after a playback failure


def test_playback_failure_reports_not_complete(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    edge.play_ok = False
    result = _run(_make_runner(), MessageType.VOICE)
    assert "playback did not complete" in result["final_response"]


def test_missing_voice_binding_refuses_visibly(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    edge._voice_text_channels = {}
    result = _run(_make_runner(), MessageType.VOICE)
    assert ("turn",) not in ORDER
    assert "no bound voice channel" in result["final_response"]


# ---------------------------------------------------------------------------
# Off / typed turns are untouched
# ---------------------------------------------------------------------------

def test_disabled_voice_turn_is_unchanged(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, brief_enabled=False)
    result = _run(_make_runner(), MessageType.VOICE)
    assert [step[0] for step in ORDER] == ["turn"]
    assert result["final_response"] == "Done."
    assert edge.played == [] and edge.posted == []


def test_typed_turn_is_unchanged_even_when_enabled(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, brief_enabled=True)
    result = _run(_make_runner(), None)
    assert [step[0] for step in ORDER] == ["turn"]
    for key, value in (("final_response", "Done."), ("messages", []), ("api_calls", 1), ("completed", True)):
        assert result[key] == value
    assert edge.played == [] and edge.posted == []


def test_typed_turn_result_is_identical_with_the_setting_on_and_off(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, brief_enabled=True)
    with_setting = _run(_make_runner(), None)
    ORDER.clear()
    _wire(monkeypatch, tmp_path, brief_enabled=False)
    without_setting = _run(_make_runner(), None)
    assert with_setting == without_setting


# ---------------------------------------------------------------------------
# No duplicate canned ack
# ---------------------------------------------------------------------------

def test_brief_replaces_the_canned_ack_for_this_turn(edge, monkeypatch, tmp_path):
    captured = {}

    class _CapturingAgent(_NoopAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["agent"] = self

    _wire(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "run_agent", types.ModuleType("run_agent"))
    sys.modules["run_agent"].AIAgent = _CapturingAgent

    _run(_make_runner(), MessageType.VOICE)

    # The ack callback is only wired when a guild slot is present; the brief cleared it.
    assert captured["agent"].tool_start_callback is None
    assert edge.acks == []


def test_disabled_voice_turn_keeps_the_ack_wiring(edge, monkeypatch, tmp_path):
    captured = {}

    class _CapturingAgent(_NoopAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["agent"] = self

    _wire(monkeypatch, tmp_path, brief_enabled=False)
    monkeypatch.setitem(sys.modules, "run_agent", types.ModuleType("run_agent"))
    sys.modules["run_agent"].AIAgent = _CapturingAgent

    _run(_make_runner(), MessageType.VOICE)
    assert captured["agent"].tool_start_callback is not None


# ---------------------------------------------------------------------------
# Routes that own execution refuse before any work
# ---------------------------------------------------------------------------

def test_proxy_route_refuses_before_any_work(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    runner = _make_runner(proxy="http://127.0.0.1:9")
    runner.adapters = {Platform.DISCORD: edge}

    async def _must_not_run(**kwargs):
        raise AssertionError("proxy path must not run")

    monkeypatch.setattr(gateway_run.GatewayRunner, "_run_agent_via_proxy", _must_not_run)
    result = asyncio.new_event_loop().run_until_complete(runner._run_agent(
        message="check the deploy logs", context_prompt="", history=[], source=_source(),
        session_id="session-1", session_key="agent:main:discord:group:123",
        message_type=MessageType.VOICE,
    ))
    assert result["final_response"] == voice_brief.UNSUPPORTED_ROUTE_TEXT
    assert ORDER == []


def test_codex_app_server_route_refuses_before_any_work(edge, monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, api_mode="codex_app_server")
    result = _run(_make_runner(), MessageType.VOICE)
    assert result["final_response"] == voice_brief.UNSUPPORTED_ROUTE_TEXT
    assert ORDER == []


# ---------------------------------------------------------------------------
# Adapter: the opt-in strict flag cannot change existing callers
# ---------------------------------------------------------------------------

def _bare_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._voice_clients = {GUILD_ID: MagicMock()}
    adapter._voice_clients[GUILD_ID].is_connected.return_value = True
    adapter._voice_receivers = {}
    adapter._voice_mixers = {}
    adapter._voice_fx_cfg = {"lead_silence_ms": 0}
    adapter._cancel_voice_timeout = lambda guild_id: None
    adapter._reset_voice_timeout = lambda guild_id: None
    adapter._lead_silence_bytes = lambda: b""

    async def _timeout(audio_path):
        return -1  # forces the wait below to time out immediately

    adapter._playback_timeout_for_audio = _timeout
    return adapter


def test_playback_timeout_returns_true_by_default_and_false_when_strict(monkeypatch):
    adapter = _bare_adapter()

    async def _go():
        default_result = await adapter.play_in_voice_channel(GUILD_ID, "/tmp/nope.mp3")
        strict_result = await adapter.play_in_voice_channel(GUILD_ID, "/tmp/nope.mp3", strict=True)
        return default_result, strict_result

    default_result, strict_result = asyncio.new_event_loop().run_until_complete(_go())
    assert default_result is True   # historical contract preserved (play_tts / run_voice)
    assert strict_result is False   # the brief can treat a cut-off clip as failure
