"""The spoken-reply mode a ``/voice join`` set ends with the voice session, however it ends.

``/voice join`` sets the bound text chat to ``all`` (every reply spoken) and persists it. Only a
rebind handed that mode back: the no-authorized-users auto-leave reset nothing, and ``/voice leave``
wrote ``off`` into the chat it was typed in rather than the bound one. So after the bot left, every
later text reply in the bound chat still carried a TTS attachment. The join now records the mode
the chat had before it; every leave restores that mode (or removes the key when there was none).
A mode the user set by hand (``/voice on``/``tts``/``off``), before or during the session, is the
user's and is never touched by the leave.

Real ``DiscordAdapter`` (join, leave, the auto-leave closure from the real ``connect()``) and the
real runner mixins (``/voice`` command, join, leave, reply gate); fakes only at the discord.py edge
(``tests/fakes/platforms/discord_voice_session.py``).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from tests.fakes.platforms import discord_voice_session as fx  # first: installs the discord mock
from gateway.config import Platform
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.discord import adapter as adapter_mod

C, D = fx.TEXT, fx.TEXT + 1  # C: the chat the join binds; D: another chat in the same guild
OWNER = 42


@pytest.fixture(autouse=True)
def _fake_receiver():
    fx.FakeReceiver.instances.clear()
    with patch.object(adapter_mod, "VoiceReceiver", fx.FakeReceiver):
        yield


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    adapter = fx.make_adapter()
    handlers = await fx.connect_capturing_handlers(adapter, monkeypatch)
    adapter._auto_join_cfg, adapter._voice_timeout_seconds = {"enabled": True}, 0
    channel = fx.voice_channel(adapter)
    adapter.get_user_voice_channel = AsyncMock(return_value=channel)
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._profile_adapters = {Platform.DISCORD: adapter}, {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db, runner.session_store = None, MagicMock()
    runner._is_user_authorized = lambda source, **_kw: True
    runner._bind_voice_input_callback(adapter)
    yield SimpleNamespace(adapter=adapter, runner=runner, handlers=handlers, channel=channel,
                          mode_path=runner._VOICE_MODE_PATH)
    await fx.stop_voice_tasks(adapter)
    await adapter.disconnect()


def _event(text: str, chat: int = C, message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    source = SessionSource(platform=Platform.DISCORD, chat_id=str(chat), user_id=str(OWNER), chat_type="group")
    event = MessageEvent(text=text, message_type=message_type, source=source)
    event.raw_message = SimpleNamespace(guild_id=fx.GUILD, guild=None)
    return event


async def _voice(rig, args: str, chat: int = C) -> str:
    return await rig.runner._handle_voice_command(_event(f"/voice {args}", chat))


async def _join(rig, chat: int = C) -> None:
    reply = await _voice(rig, "join", chat)
    assert rig.adapter.is_in_voice_channel(fx.GUILD), reply
    assert rig.runner._voice_mode.get(_key(chat)) == "all"  # (1) the join speaks in the bound chat


async def _owner_walks_out(rig) -> None:
    """(2) The last authorized member leaves the VC: the real no-authorized-users auto-leave."""
    driver = fx.member(OWNER, rig.adapter._fake_guild)
    await rig.handlers["on_voice_state_update"](driver, fx.voice_state(rig.channel), fx.voice_state(None))
    assert not rig.adapter.is_in_voice_channel(fx.GUILD), "the auto-leave itself must happen"


def _key(chat: int) -> str:
    return f"discord:{chat}"


def _persisted(rig) -> dict:
    return json.loads(rig.mode_path.read_text(encoding="utf-8")) if rig.mode_path.exists() else {}


def _speaks_text_replies(rig, chat: int = C) -> bool:
    """Would a plain text reply in *chat* carry a TTS attachment? The runner's gate decides for text
    input (it consults the adapter's auto-TTS set when the chat has no mode); the adapter's own
    auto-TTS (``_wants_auto_tts``) fires on voice input only."""
    return rig.runner._should_send_voice_reply(_event("what's next?", chat), "Here is the plan.", [])


@pytest.mark.asyncio
async def test_auto_leave_drops_the_mode_the_join_set(rig):
    await _join(rig)
    await _owner_walks_out(rig)

    # (3) the join's ``all`` is gone, in memory, on disk, and from the adapter's auto-TTS set
    assert rig.runner._voice_mode.get(_key(C)) != "all"
    assert _key(C) not in _persisted(rig)
    assert str(C) not in rig.adapter._auto_tts_enabled_chats
    # (4) so a later text reply in C is text only
    assert not _speaks_text_replies(rig)


@pytest.mark.asyncio
async def test_voice_leave_typed_elsewhere_clears_the_bound_chat(rig):
    await _voice(rig, "on", D)  # D's own, hand-set mode
    await _join(rig)

    await _voice(rig, "leave", D)  # (5)

    assert not rig.adapter.is_in_voice_channel(fx.GUILD)
    assert rig.runner._voice_mode.get(_key(C)) != "all" and _key(C) not in _persisted(rig)
    assert not _speaks_text_replies(rig, C)
    assert rig.runner._voice_mode.get(_key(D)) == "voice_only", "the leave must not rewrite the chat it was typed in"


@pytest.mark.asyncio
async def test_voice_leave_in_the_bound_chat_clears_it(rig):
    await _join(rig)
    await _voice(rig, "leave")
    assert _key(C) not in rig.runner._voice_mode and _key(C) not in _persisted(rig)
    assert not _speaks_text_replies(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["on", "off"])
async def test_auto_leave_restores_the_mode_the_chat_had_before_the_join(rig, prior):
    expected = {"on": "voice_only", "off": "off"}[prior]
    await _voice(rig, prior)
    await _join(rig)
    await _owner_walks_out(rig)  # (6)

    assert rig.runner._voice_mode.get(_key(C)) == expected
    assert _persisted(rig).get(_key(C)) == expected
    assert (str(C) in rig.adapter._auto_tts_enabled_chats) is (expected != "off")
    assert not _speaks_text_replies(rig), "voice_only and off both answer text with text"


@pytest.mark.asyncio
@pytest.mark.parametrize("set_during", ["tts", "on"])
async def test_a_mode_set_by_hand_during_the_session_survives_the_leave(rig, set_during):
    expected = {"tts": "all", "on": "voice_only"}[set_during]
    await _join(rig)
    await _voice(rig, set_during)  # (7) the user now owns C's mode
    await _owner_walks_out(rig)

    assert rig.runner._voice_mode.get(_key(C)) == expected
    assert _persisted(rig).get(_key(C)) == expected
    assert str(C) in rig.adapter._auto_tts_enabled_chats


async def _start_a_fresh_gateway(rig, monkeypatch, tmp_path):
    """A new process: the real ``GatewayRunner.__init__`` over the same mode files."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", rig.mode_path)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    if runner._session_db is not None:
        await runner._session_db.close()
    return runner


def _join_records(rig) -> dict:
    path = rig.runner._voice_join_mode_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [None, "on"])
async def test_a_crash_mid_session_hands_the_mode_back_at_the_next_start(rig, monkeypatch, tmp_path, prior):
    """A SIGKILL/OOM leaves no leave behind. No voice session survives a process start, so the
    next start hands every join record back: the chat is not left speaking every reply."""
    if prior:
        await _voice(rig, prior)
    await _join(rig)  # ...and the process dies here: no leave, no shutdown

    fresh = await _start_a_fresh_gateway(rig, monkeypatch, tmp_path)

    expected = {None: None, "on": "voice_only"}[prior]
    assert fresh._voice_mode.get(_key(C)) == expected
    assert _persisted(rig).get(_key(C)) == expected
    assert _join_records(rig) == {}, "a record outliving its process would hand back a stale prior later"


@pytest.mark.asyncio
async def test_the_start_leaves_an_all_with_no_join_record_alone(rig, monkeypatch, tmp_path):
    """An ``all`` with no record is a hand-set ``/voice tts`` (or a key from before records
    existed): the two cannot be told apart, so the start never clears it."""
    rig.mode_path.write_text(json.dumps({_key(C): "all", _key(D): "all"}), encoding="utf-8")
    rig.runner._voice_join_mode_path().write_text(json.dumps({_key(C): None}), encoding="utf-8")

    fresh = await _start_a_fresh_gateway(rig, monkeypatch, tmp_path)

    assert fresh._voice_mode == {_key(D): "all"}
    assert _persisted(rig) == {_key(D): "all"}


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", [None, "on"])
async def test_a_torn_join_file_cannot_strand_the_joins_all(rig, monkeypatch, tmp_path, prior):
    """The join's ``all`` lives in memory and in the join file only: the mode file keeps the
    mode the chat had before, so losing the join file (torn, truncated) loses only the ``all``."""
    if prior:
        await _voice(rig, prior)
    await _join(rig)
    expected = {None: None, "on": "voice_only"}[prior]
    assert _persisted(rig).get(_key(C)) == expected, "the mode file holds the prior, not the join's all"

    join_file = rig.runner._voice_join_mode_path()
    join_file.write_text(join_file.read_text(encoding="utf-8")[:7], encoding="utf-8")  # torn mid-write
    fresh = await _start_a_fresh_gateway(rig, monkeypatch, tmp_path)

    assert fresh._voice_mode.get(_key(C)) == expected
    assert _persisted(rig).get(_key(C)) == expected


@pytest.mark.asyncio
async def test_both_voice_mode_files_are_written_atomically_join_file_first(rig, monkeypatch):
    import utils
    writes = []

    def recording(path, content, **kw):
        writes.append(path.name)
        return utils.atomic_write_text(path, content, **kw)

    monkeypatch.setattr("gateway.run_voice.atomic_write_text", recording, raising=False)
    await _join(rig)
    assert writes and writes[-2:] == ["gateway_voice_mode.join.json", "gateway_voice_mode.json"], writes


@pytest.mark.asyncio
async def test_control_while_joined_text_replies_are_spoken(rig):
    """Positive control for the harness: the join's ``all`` does speak text replies in C."""
    await _join(rig)
    assert _speaks_text_replies(rig)
