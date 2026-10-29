"""After ``/voice leave``, a later auto-join session must still deliver its transcripts.

``/voice leave`` used to null ``adapter._voice_input_callback``. Only adapter connect and
``/voice join`` bind it again; an auto-join does not. So after one ``/voice leave`` every later
auto-join session of that bot silently dropped all speech until the gateway restarted. A failed
``/voice join`` nulled it the same way.

These drive the REAL runner handlers (``_handle_voice_channel_leave``/``_handle_voice_channel_join``)
and the REAL adapter (``_do_auto_join`` -> ``join_voice_channel`` -> ``_process_voice_input`` ->
runner ``_handle_voice_channel_input``). Only the transport edges are fakes: the Discord client and
channel, STT, and ``adapter.handle_message`` (the point a transcript reaches the runner's pipeline).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter

_GUILD, _TEXT, _VOICE, _SPEAKER = 1, 700, 900, 42


class _FakeVoiceClient:
    def __init__(self, channel):
        self.channel = channel
        self._connected = True

    def is_connected(self):
        return self._connected

    def is_playing(self):
        return False

    async def disconnect(self, *, force=False):
        self._connected = False


class _FakeReceiver:
    def __init__(self, vc, allowed_user_ids=None):
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def flush_pending(self):
        return []

    def check_silence(self):
        return []


def _adapter() -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    guild = SimpleNamespace(id=_GUILD, name="Garage", get_member=lambda uid: None)
    text_channel = SimpleNamespace(id=_TEXT, name="drive-notes", topic=None, send=AsyncMock(), guild=guild)
    adapter._client = SimpleNamespace(
        get_guild=lambda gid: guild if gid == _GUILD else None,
        get_channel=lambda cid: text_channel if cid == _TEXT else None,
        get_user=lambda uid: None,
        user=object(),
    )
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter.handle_message = AsyncMock()
    return adapter


def _runner(adapter, tmp_path):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source, **_kw: True
    # Startup wiring (run_adapters binds this when the adapter connects).
    runner._bind_voice_input_callback(adapter)
    return runner


def _command(text: str) -> MessageEvent:
    source = SessionSource(platform=Platform.DISCORD, chat_id=str(_TEXT), user_id="owner",
                           chat_type="group")
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    event.raw_message = SimpleNamespace(guild_id=_GUILD, guild=None)
    return event


async def _auto_join_then_speak(adapter) -> None:
    voice_channel = SimpleNamespace(id=_VOICE, guild=SimpleNamespace(id=_GUILD), name="drive")
    voice_channel.connect = AsyncMock(side_effect=lambda **_kw: _FakeVoiceClient(voice_channel))
    with patch.object(adapter_mod, "DISCORD_AVAILABLE", True), \
         patch.object(adapter_mod, "VoiceReceiver", _FakeReceiver):
        await adapter._do_auto_join(voice_channel, _TEXT, _GUILD)
    assert adapter.is_in_voice_channel(_GUILD), "the auto-join itself must succeed"
    try:
        with patch.object(adapter_mod.VoiceReceiver, "pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "refined the pricing idea"}):
            await adapter._process_voice_input(_GUILD, _SPEAKER, b"\x00" * 9600, _TEXT)
    finally:
        for tasks in (adapter._voice_timeout_tasks, adapter._voice_listen_tasks):
            for task in tasks.values():
                task.cancel()
        await asyncio.gather(*adapter._voice_timeout_tasks.values(),
                             *adapter._voice_listen_tasks.values(), return_exceptions=True)


def _dispatched_text(adapter) -> list[str]:
    return [call.args[0].text for call in adapter.handle_message.await_args_list]


@pytest.mark.asyncio
async def test_voice_leave_then_auto_join_still_delivers_speech(tmp_path):
    adapter = _adapter()
    runner = _runner(adapter, tmp_path)
    adapter._voice_clients[_GUILD] = _FakeVoiceClient(SimpleNamespace(id=_VOICE))
    adapter._voice_text_channels[_GUILD] = _TEXT

    await runner._handle_voice_channel_leave(_command("/voice leave"))
    assert not adapter.is_in_voice_channel(_GUILD)

    await _auto_join_then_speak(adapter)

    assert _dispatched_text(adapter) == ["refined the pricing idea"]
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.VOICE
    assert event.source.chat_id == str(_TEXT) and event.source.user_id == str(_SPEAKER)


@pytest.mark.asyncio
@pytest.mark.parametrize("join_outcome", ["raises", "refused"])
async def test_failed_voice_join_then_auto_join_still_delivers_speech(tmp_path, join_outcome):
    adapter = _adapter()
    runner = _runner(adapter, tmp_path)
    real_join = adapter.join_voice_channel
    adapter.get_user_voice_channel = AsyncMock(return_value=SimpleNamespace(name="drive"))
    adapter.join_voice_channel = AsyncMock(
        side_effect=RuntimeError("No permission") if join_outcome == "raises" else None,
        return_value=False,
    )
    reply = await runner._handle_voice_channel_join(_command("/voice join"))
    assert "fail" in reply.lower() or "permission" in reply.lower(), reply
    adapter.join_voice_channel = real_join

    await _auto_join_then_speak(adapter)

    assert _dispatched_text(adapter) == ["refined the pricing idea"]


@pytest.mark.asyncio
async def test_control_auto_join_with_startup_binding_delivers_speech(tmp_path):
    """Positive control for the harness: with no ``/voice leave`` first, speech reaches the runner
    on any revision. The two tests above differ from this one only by the preceding command."""
    adapter = _adapter()
    _runner(adapter, tmp_path)

    await _auto_join_then_speak(adapter)

    assert _dispatched_text(adapter) == ["refined the pricing idea"]
