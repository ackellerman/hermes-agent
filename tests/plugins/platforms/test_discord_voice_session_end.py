"""Discord adapter voice-session lifecycle: one record per cold connect, one end per record.

Real ``DiscordAdapter`` (constructor, join, leave, inactivity timer, final flush, auto-leave
closure from the real ``connect()``); fakes only at the discord.py edge
(``tests/fakes/platforms/discord_voice_session.py``). The runner's end handler is the boundary
here and is a recorder; the runner side is covered in ``tests/gateway/test_voice_session_end_hook.py``.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from tests.fakes.platforms import discord_voice_session as fx  # first: installs the discord mock
from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter_voice_session import VoiceSessionEnd
from tests.waiting import async_wait_until


class _Recorder:
    def __init__(self, adapter):
        self.ends: list[VoiceSessionEnd] = []
        adapter._voice_session_end_callback = self

    def __call__(self, end: VoiceSessionEnd):
        self.ends.append(end)
        return None


@pytest.fixture(autouse=True)
def _fake_receiver():
    fx.FakeReceiver.instances.clear()
    with patch.object(adapter_mod, "VoiceReceiver", fx.FakeReceiver):
        yield


@pytest_asyncio.fixture
async def adapter():
    adapter = fx.make_adapter()
    yield adapter
    await fx.stop_voice_tasks(adapter)


async def _join(adapter, cid=fx.VOICE_A, text=fx.TEXT):
    assert await adapter.join_voice_channel(fx.voice_channel(adapter, cid), text_channel_id=text) is True


@pytest.mark.asyncio
async def test_a_leave_ends_the_session_once_and_later_leaves_end_nothing(adapter):
    rec = _Recorder(adapter)
    await _join(adapter)
    session = adapter._voice_sessions[fx.GUILD]

    await adapter.leave_voice_channel(fx.GUILD)
    await adapter.leave_voice_channel(fx.GUILD)
    await adapter.disconnect()

    assert [(e.reason, e.trigger) for e in rec.ends] == [("command", None)]
    end = rec.ends[0]
    assert (end.voice_session_id, end.started_at) == (session.voice_session_id, session.started_at)
    assert (end.guild_id, end.voice_channel_id, end.text_channel_id) == (fx.GUILD, fx.VOICE_A, fx.TEXT)
    assert end.ended_at >= end.started_at


@pytest.mark.asyncio
async def test_concurrent_leaves_end_the_session_once(adapter):
    rec = _Recorder(adapter)
    await _join(adapter)

    await asyncio.gather(
        adapter.leave_voice_channel(fx.GUILD),
        adapter.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="inactivity"),
    )

    assert len(rec.ends) == 1


@pytest.mark.asyncio
async def test_moves_and_rejoins_keep_one_session_and_the_end_reports_the_last_channel(adapter):
    rec = _Recorder(adapter)
    await _join(adapter, fx.VOICE_A)
    first = adapter._voice_sessions[fx.GUILD]
    await _join(adapter, fx.VOICE_B)           # move
    await _join(adapter, fx.VOICE_B)           # same-channel re-join
    assert adapter._voice_sessions[fx.GUILD] is first
    assert rec.ends == []

    await adapter.leave_voice_channel(fx.GUILD)

    assert [(e.voice_session_id, e.voice_channel_id) for e in rec.ends] == [(first.voice_session_id, fx.VOICE_B)]


@pytest.mark.asyncio
async def test_speech_only_in_the_final_flush_is_in_the_record(adapter):
    """The last utterance before an auto-leave is flushed and dispatched INSIDE the leave."""
    rec = _Recorder(adapter)
    await _join(adapter)
    fx.FakeReceiver.instances[-1].pending = [(42, b"\x00" * 9600)]

    async def runner_input(*, guild_id, user_id, transcript):
        # What the runner does once the transcript is dispatched (gateway/run_voice.py).
        adapter.note_voice_session_speech(guild_id, str(user_id), "agent:main:discord:group:700:42")

    adapter._voice_input_callback = runner_input
    with patch("tools.transcription_tools.transcribe_audio",
               return_value={"success": True, "transcript": "one open question"}):
        await adapter.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="no_authorized_users")

    end = rec.ends[0]
    assert end.participants == ("42",)
    assert end.session_key == "agent:main:discord:group:700:42"
    assert end.session_keys == ("agent:main:discord:group:700:42",)
    assert (end.reason, end.trigger) == ("auto_leave", "no_authorized_users")


@pytest.mark.asyncio
async def test_the_session_ends_even_when_the_leave_raises(adapter):
    rec = _Recorder(adapter)
    await _join(adapter)
    adapter._voice_clients[fx.GUILD].disconnect = AsyncMock(side_effect=RuntimeError("socket gone"))

    with pytest.raises(RuntimeError):
        await adapter.leave_voice_channel(fx.GUILD)

    assert [e.reason for e in rec.ends] == ["command"]
    assert fx.GUILD not in adapter._voice_sessions


@pytest.mark.asyncio
async def test_the_inactivity_timer_ends_the_session_as_auto_leave(adapter):
    """Through the real timer task, which cancels itself inside the leave it runs."""
    rec = _Recorder(adapter)
    adapter._voice_timeout_seconds = 1
    await _join(adapter)

    await async_wait_until(lambda: rec.ends, timeout=10, what="inactivity end")

    assert [(e.reason, e.trigger) for e in rec.ends] == [("auto_leave", "inactivity")]
    assert not adapter.is_in_voice_channel(fx.GUILD)


@pytest.mark.asyncio
async def test_the_no_authorized_users_auto_leave_through_the_real_handler(adapter, monkeypatch):
    rec = _Recorder(adapter)
    handlers = await fx.connect_capturing_handlers(adapter, monkeypatch)
    adapter._auto_join_cfg = {"enabled": True}
    adapter._voice_timeout_seconds = 0
    channel = fx.voice_channel(adapter, fx.VOICE_A)
    await adapter.join_voice_channel(channel, text_channel_id=fx.TEXT)
    try:
        driver = fx.member(42, adapter._fake_guild)
        await handlers["on_voice_state_update"](driver, fx.voice_state(channel), fx.voice_state(None))
        assert [(e.reason, e.trigger) for e in rec.ends] == [("auto_leave", "no_authorized_users")]
    finally:
        await adapter.disconnect()
    assert len(rec.ends) == 1


# --- a connection Discord closed for good (VoiceClient.cleanup) ----------------------------------


@pytest.fixture
def hermes_voice_client():
    """Make ``discord.VoiceClient`` a real class so join connects with the Hermes subclass."""
    with patch.object(adapter_mod, "discord", SimpleNamespace(VoiceClient=fx.FakeVoiceClient)), \
         patch.object(adapter_mod, "DISCORD_AVAILABLE", True):
        yield


@pytest.mark.asyncio
async def test_a_final_disconnect_ends_the_session_and_clears_the_stale_state(adapter, hermes_voice_client):
    rec = _Recorder(adapter)
    cleaned = []
    adapter._on_voice_disconnect = cleaned.append
    channel = fx.voice_channel(adapter)
    await adapter.join_voice_channel(channel, text_channel_id=fx.TEXT)
    vc = adapter._voice_clients[fx.GUILD]
    assert channel.connect_classes[-1] is not None and isinstance(vc, fx.FakeVoiceClient)

    vc.closed_by_discord()
    await async_wait_until(lambda: rec.ends, timeout=10, what="disconnect end")

    assert [e.reason for e in rec.ends] == ["disconnect"]
    assert fx.GUILD not in adapter._voice_clients and fx.GUILD not in adapter._voice_receivers
    assert fx.GUILD not in adapter._voice_text_channels
    assert fx.FakeReceiver.instances[-1].stopped
    assert cleaned == [str(fx.TEXT)]


@pytest.mark.asyncio
async def test_our_own_leave_is_not_also_reported_as_a_disconnect(adapter, hermes_voice_client):
    rec = _Recorder(adapter)
    await _join(adapter)
    vc = adapter._voice_clients[fx.GUILD]

    await adapter.leave_voice_channel(fx.GUILD)
    vc.cleanup()  # discord.py calls cleanup() a second time from the voice state's own disconnect
    await asyncio.gather(*adapter.__dict__.get("_voice_close_tasks", ()), return_exceptions=True)

    assert vc.cleanups == 2
    assert [e.reason for e in rec.ends] == ["command"]


@pytest.mark.asyncio
async def test_a_reconnect_ends_nothing_and_the_later_leave_ends_once(adapter, hermes_voice_client):
    """discord.py's reconnect paths disconnect with cleanup=False: cleanup() is never called."""
    rec = _Recorder(adapter)
    await _join(adapter)
    vc = adapter._voice_clients[fx.GUILD]
    vc._connected = False          # the voice websocket dropped ...
    vc._connected = True           # ... and discord.py reconnected it (no cleanup())
    assert rec.ends == []

    await adapter.leave_voice_channel(fx.GUILD)

    assert [e.reason for e in rec.ends] == ["command"]


@pytest.mark.asyncio
async def test_a_cold_rejoin_after_an_unseen_drop_ends_the_old_session_first(adapter):
    rec = _Recorder(adapter)
    await _join(adapter)
    old = adapter._voice_sessions[fx.GUILD]
    adapter._voice_clients[fx.GUILD].closed_by_discord()   # dropped for good; no notifier on a plain client

    assert await adapter.join_voice_channel(fx.voice_channel(adapter)) is True   # names no text channel

    new = adapter._voice_sessions[fx.GUILD]
    assert new.voice_session_id != old.voice_session_id
    assert [(e.voice_session_id, e.reason) for e in rec.ends] == [(old.voice_session_id, "disconnect")]
    assert adapter._voice_text_channels[fx.GUILD] == fx.TEXT    # the binding survives the stale end
    await adapter.leave_voice_channel(fx.GUILD)
    assert [e.voice_session_id for e in rec.ends] == [old.voice_session_id, new.voice_session_id]


class _AutoJoinCaller:
    """The two callers of a cold join: ``/voice join`` (the runner calls ``join_voice_channel``) and
    the adapter's own auto-join (``_do_auto_join``)."""

    @staticmethod
    async def voice_join(adapter, channel):
        return await adapter.join_voice_channel(channel, text_channel_id=fx.TEXT)

    @staticmethod
    async def auto_join(adapter, channel):
        return await adapter._do_auto_join(channel, fx.TEXT, fx.GUILD)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["voice_join", "auto_join"])
async def test_a_join_during_discord_reconnect_window_keeps_the_session(adapter, hermes_voice_client, caller):
    """discord.py is still reconnecting (client registered, not connected, cleanup() not run): a join
    landing then must not end the session, stop the receiver or drop the client. ``connect()``
    refuses as discord.py does (base behaviour); once the reconnect completes the bot is still
    tracked and its real leave ends the session exactly once."""
    rec = _Recorder(adapter)
    channel = fx.voice_channel(adapter)
    await adapter.join_voice_channel(channel, text_channel_id=fx.TEXT)
    vc, receiver = adapter._voice_clients[fx.GUILD], fx.FakeReceiver.instances[-1]
    session = adapter._voice_sessions[fx.GUILD]
    vc.reconnecting()

    outcome = "returned"
    try:
        await getattr(_AutoJoinCaller, caller)(adapter, fx.voice_channel(adapter))
    except RuntimeError:
        outcome = "raised"                       # discord.py's "Already connected" (base behaviour)
    vc._connected = True                         # discord.py finished reconnecting

    state = dict(ends=[e.reason for e in rec.ends], tracked=adapter._voice_clients.get(fx.GUILD) is vc,
                 receiver_alive=not receiver.stopped and adapter._voice_receivers.get(fx.GUILD) is receiver,
                 record=adapter._voice_sessions.get(fx.GUILD) is session,
                 bound=adapter._voice_text_channels.get(fx.GUILD))
    assert state == dict(ends=[], tracked=True, receiver_alive=True, record=True, bound=fx.TEXT), (outcome, state)
    assert adapter.is_in_voice_channel(fx.GUILD)

    await adapter.leave_voice_channel(fx.GUILD)
    assert [(e.voice_session_id, e.reason) for e in rec.ends] == [(session.voice_session_id, "command")]


@pytest.mark.asyncio
async def test_a_stale_close_after_a_rejoin_ends_only_the_old_session(adapter, hermes_voice_client):
    """Drop, then a cold re-join runs before the scheduled close task: the close task finds a NEW
    client for the guild and must leave it alone (the "is this still our client?" checks)."""
    rec = _Recorder(adapter)
    await _join(adapter)
    old = adapter._voice_sessions[fx.GUILD]
    adapter._voice_clients[fx.GUILD].closed_by_discord()        # close task scheduled, not yet run
    await _join(adapter)
    new = adapter._voice_sessions[fx.GUILD]
    new_vc = adapter._voice_clients[fx.GUILD]

    await asyncio.gather(*adapter.__dict__.get("_voice_close_tasks", ()), return_exceptions=True)

    assert [(e.voice_session_id, e.reason) for e in rec.ends] == [(old.voice_session_id, "disconnect")]
    assert adapter._voice_sessions.get(fx.GUILD) is new and adapter._voice_clients.get(fx.GUILD) is new_vc
    assert adapter.is_in_voice_channel(fx.GUILD) and not fx.FakeReceiver.instances[-1].stopped
    await adapter.leave_voice_channel(fx.GUILD)
    assert [e.voice_session_id for e in rec.ends] == [old.voice_session_id, new.voice_session_id]


# --- teardown --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_waits_for_end_handlers_but_only_within_its_budget(adapter):
    quick_done, blocked = asyncio.Event(), asyncio.Event()
    calls = []

    async def quick():
        quick_done.set()

    def handler(end):
        calls.append(end)
        return asyncio.ensure_future(quick() if len(calls) == 1 else blocked.wait())

    adapter._voice_session_end_callback = handler
    await _join(adapter)
    adapter._client.close = AsyncMock()
    await adapter.disconnect()
    assert quick_done.is_set(), "disconnect() returned before a scheduled end handler could run"

    adapter._client = fx.make_adapter()._client
    adapter._client.close = AsyncMock()
    await _join(adapter)
    started = time.monotonic()
    try:
        await adapter.disconnect()
        assert time.monotonic() - started < 4.0
        assert [e.reason for e in calls] == ["shutdown", "shutdown"]
    finally:
        blocked.set()


@pytest.mark.asyncio
async def test_two_bots_in_one_guild_keep_separate_sessions():
    one, two = fx.make_adapter(), fx.make_adapter(owner="work", bot_user_id=998)
    rec_one, rec_two = _Recorder(one), _Recorder(two)
    try:
        await _join(one, fx.VOICE_A)
        await _join(two, fx.VOICE_B)
        kept = one._voice_sessions[fx.GUILD]

        await two.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="no_authorized_users")

        assert len(rec_two.ends) == 1 and rec_one.ends == []
        assert one._voice_sessions[fx.GUILD] is kept and one.is_in_voice_channel(fx.GUILD)
        await one.leave_voice_channel(fx.GUILD)
        assert [e.voice_session_id for e in rec_one.ends] == [kept.voice_session_id]
    finally:
        await fx.stop_voice_tasks(one)
        await fx.stop_voice_tasks(two)


@pytest.mark.asyncio
async def test_a_concurrent_leave_waits_for_the_final_flush_to_be_recorded(adapter):
    """The inactivity timer firing while ``/voice leave`` transcribes the final flush must not end the
    session before that utterance is recorded: the second leave waits on the guild's voice lock."""
    rec = _Recorder(adapter)
    await _join(adapter)
    fx.FakeReceiver.instances[-1].pending = [(42, b"\x00" * 9600)]
    in_stt, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    async def runner_input(*, guild_id, user_id, transcript):
        adapter.note_voice_session_speech(guild_id, str(user_id), "agent:main:discord:group:700:42")

    def transcribe(_path):
        loop.call_soon_threadsafe(in_stt.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result(10)
        return {"success": True, "transcript": "the last sentence"}

    adapter._voice_input_callback = runner_input
    with patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        first = asyncio.ensure_future(adapter.leave_voice_channel(fx.GUILD))
        await in_stt.wait()
        second = asyncio.ensure_future(
            adapter.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="inactivity"))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)

    assert [(e.reason, e.participants) for e in rec.ends] == [("command", ("42",))]
