"""``on_voice_session_end`` end to end: real adapter, real runner mixins, real plugin discovery.

Each test installs an on-disk probe plugin (``tests/fakes/voice_session_end_probe.py``) in a temp
profile home and discovers it the way the gateway does. Voice sessions run on the real
``DiscordAdapter`` (join, final flush, leave paths, auto-leave closure from the real ``connect()``)
with fakes only at the discord.py edge, and the transcripts reach the real runner
``_handle_voice_channel_input``. The follow-up goes through the real ``inject_plugin_message`` and
the runner's real injector into ``adapter.handle_message`` — the transport boundary, captured here.
The published injector is wrapped in a spy so a test can tell "never asked" from "asked and refused".

Acceptance (request 2026-10-29): fires exactly once on both leave paths and never on a reconnect or a
move; shutdown fires without a follow-up; the follow-up is posted as text and not spoken; only the
profile whose bot left gets the event under a multiplexed gateway.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fakes.platforms import discord_voice_session as fx  # first: installs the discord mock
from tests.fakes import voice_session_end_probe as probe
from tests.waiting import async_wait_until

import hermes_cli.plugins as plugins_mod
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.response_filters import VOICE_REPLY_SUPPRESSED_KEY, copy_queued_terminal_metadata, display_kind_for_event
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from gateway.session import SessionStore
from plugins.platforms.discord import adapter as adapter_mod

SPEAKER = 42
WRAP_UP = "Summarise the drive for the notes channel."


@pytest.fixture(autouse=True)
def _fake_receiver():
    fx.FakeReceiver.instances.clear()
    with patch.object(adapter_mod, "VoiceReceiver", fx.FakeReceiver):
        yield


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    work = root / "profiles" / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    plugins_mod._reset_plugin_managers_for_tests()
    yield SimpleNamespace(default=root, work=work, tmp=tmp_path)
    probe.reset()
    plugins_mod._reset_plugin_managers_for_tests()


class Rig:
    """A partial ``GatewayRunner`` (real mixin methods) with a real SessionStore and the real
    injector published behind a spy, plus real Discord adapters bound to it."""

    def __init__(self, tmp_path: Path, *, multiplex: bool = False, primary_name: str = "seat") -> None:
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=multiplex)
        runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=runner.config)
        runner.adapters, runner._profile_adapters = {}, {}
        runner._voice_mode, runner._VOICE_MODE_PATH = {}, tmp_path / "gateway_voice_mode.json"
        runner._session_db = None
        runner._gateway_loop = asyncio.get_running_loop()
        runner._running, runner._draining = True, False
        runner._background_tasks, runner._queued_events = set(), {}
        runner._primary_profile_name = primary_name
        runner._is_user_authorized = lambda source, **_kw: True
        self.runner = runner
        self.injector = MagicMock(side_effect=runner._schedule_plugin_message_injection)
        # The dispatched voice turn opens its session from a background task, as the real turn does
        # (run_turn: get_or_create_session); a test may hold it back to model a slow turn.
        self.turn_may_open = asyncio.Event()
        self.turn_may_open.set()
        plugins_mod.publish_gateway_message_host(runner, self.injector)

    def add_adapter(self, *, owner=None, bot_user_id=999):
        adapter = fx.make_adapter(owner=owner, bot_user_id=bot_user_id)
        if owner:
            self.runner._profile_adapters[owner] = {Platform.DISCORD: adapter}
        else:
            self.runner.adapters[Platform.DISCORD] = adapter
        self.runner._bind_voice_input_callback(adapter)  # what adapter wiring does at connect
        adapter.handle_message = AsyncMock(side_effect=self._turn_opens_its_session)
        return adapter

    async def _turn_opens_its_session(self, event):
        if event.message_type != MessageType.VOICE:
            return

        async def turn():
            await self.turn_may_open.wait()
            await self.runner.async_session_store.get_or_create_session(event.source)

        task = asyncio.get_running_loop().create_task(turn())
        self.runner._background_tasks.add(task)
        task.add_done_callback(self.runner._background_tasks.discard)

    async def settle(self):
        """Wait for every scheduled end dispatch (the hook and the follow-up decision) to finish."""
        while pending := [task for task in self.runner._background_tasks if not task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)


async def _join(adapter, cid=fx.VOICE_A):
    assert await adapter.join_voice_channel(fx.voice_channel(adapter, cid), text_channel_id=fx.TEXT) is True


async def _speak(adapter, user=SPEAKER, text="refined the consulting pricing idea"):
    with patch("tools.transcription_tools.transcribe_audio", return_value={"success": True, "transcript": text}):
        await adapter._process_voice_input(fx.GUILD, user, b"\x00" * 9600, adapter._voice_text_channels.get(fx.GUILD))


def _voice_events(adapter):
    return [c.args[0] for c in adapter.handle_message.await_args_list if c.args[0].message_type == MessageType.VOICE]


def _followups(adapter):
    return [c.args[0] for c in adapter.handle_message.await_args_list
            if (c.args[0].metadata or {}).get("hermes_plugin_injection")]


def _stored_key_of(rig, event) -> tuple[str, str]:
    """What the turn does first with a dispatched event: open its session in the store."""
    entry = rig.runner.session_store.get_or_create_session(event.source)
    return entry.session_key, entry.session_id


def _command(text: str):
    from gateway.platforms.base import SessionSource
    source = SessionSource(platform=Platform.DISCORD, chat_id=str(fx.TEXT), user_id=str(SPEAKER), chat_type="group")
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    event.raw_message = SimpleNamespace(guild_id=fx.GUILD, guild=None)
    return event


# --- T1: fires once on the auto-leave, with the session the transcript really went into -------------


@pytest.mark.asyncio
async def test_t1_auto_leave_fires_once_with_the_dispatched_session(homes, monkeypatch):
    probe.install_probe(homes.default)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    handlers = await fx.connect_capturing_handlers(adapter, monkeypatch)
    adapter._auto_join_cfg, adapter._voice_timeout_seconds = {"enabled": True}, 0
    channel = fx.voice_channel(adapter)
    await adapter.join_voice_channel(channel, text_channel_id=fx.TEXT)
    try:
        await _speak(adapter)
        [voice_event] = _voice_events(adapter)
        stored_key, stored_id = _stored_key_of(rig, voice_event)

        driver = fx.member(SPEAKER, adapter._fake_guild)
        await handlers["on_voice_state_update"](driver, fx.voice_state(channel), fx.voice_state(None))
        await rig.settle()
    finally:
        await adapter.disconnect()
        await rig.settle()

    [record] = probe.records(homes.default)
    assert (record["reason"], record["trigger"]) == ("auto_leave", "no_authorized_users")
    assert record["participants"] == [str(SPEAKER)]
    assert (record["session_key"], record["session_id"]) == (stored_key, stored_id)
    assert record["session_keys"] == [stored_key]
    assert record["profile"] == "seat"  # the bot owner's real profile name, not a literal "default"
    assert (record["guild_id"], record["voice_channel_id"], record["text_channel_id"]) == (
        str(fx.GUILD), str(fx.VOICE_A), str(fx.TEXT))
    assert record["voice_session_id"] and record["ended_at"] >= record["started_at"]


@pytest.mark.asyncio
async def test_t1b_inactivity_timeout_fires_once_as_auto_leave(homes):
    probe.install_probe(homes.default)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    adapter._voice_timeout_seconds = 1
    await _join(adapter)
    await _speak(adapter)

    await async_wait_until(lambda: probe.records(homes.default), timeout=10, what="inactivity end")
    await rig.settle()
    await adapter.leave_voice_channel(fx.GUILD)
    await rig.settle()

    [record] = probe.records(homes.default)
    assert (record["reason"], record["trigger"]) == ("auto_leave", "inactivity")
    assert record["participants"] == [str(SPEAKER)]


# --- T2: fires once on /voice leave ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_t2_voice_leave_fires_once(homes):
    probe.install_probe(homes.default)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    adapter._client.close = AsyncMock()
    await _join(adapter)
    await _speak(adapter)

    await rig.runner._handle_voice_channel_leave(_command("/voice leave"))
    await rig.runner._handle_voice_channel_leave(_command("/voice leave"))
    await adapter.leave_voice_channel(fx.GUILD)              # a programmatic leave after it
    await rig.settle()
    assert [(r["reason"], r["trigger"]) for r in probe.records(homes.default)] == [("command", None)]

    # The next session is a new one: its end is the only further event (no leftover of the first).
    await _join(adapter)
    await rig.runner._handle_voice_channel_leave(_command("/voice leave"))
    await adapter.disconnect()
    await rig.settle()
    records = probe.records(homes.default)
    assert [r["reason"] for r in records] == ["command", "command"]
    assert len({r["voice_session_id"] for r in records}) == 2


# --- T3: never on a reconnect or a move ------------------------------------------------------------


@pytest.mark.asyncio
async def test_t3_moves_and_reconnects_fire_nothing_and_the_leave_fires_once(homes):
    probe.install_probe(homes.default)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    with patch.object(adapter_mod, "discord", SimpleNamespace(VoiceClient=fx.FakeVoiceClient)), \
         patch.object(adapter_mod, "DISCORD_AVAILABLE", True):
        await _join(adapter, fx.VOICE_A)
        started_at = adapter._voice_sessions[fx.GUILD].started_at
        await _join(adapter, fx.VOICE_B)                        # moved
        vc = adapter._voice_clients[fx.GUILD]
        vc._connected = False                                   # websocket dropped,
        vc._connected = True                                    # discord.py reconnected: no cleanup()
        await _join(adapter, fx.VOICE_B)                        # programmatic re-join, same channel
        await rig.settle()
        assert probe.records(homes.default) == []

        await adapter.leave_voice_channel(fx.GUILD)              # positive control in the same test
        await rig.settle()

    [record] = probe.records(homes.default)
    assert (record["reason"], record["voice_channel_id"]) == ("command", str(fx.VOICE_B))
    assert record["started_at"] == started_at


# --- T4: reason=disconnect -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t4_a_connection_discord_closed_fires_disconnect_and_clears_stale_state(homes):
    probe.install_probe(homes.default)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    with patch.object(adapter_mod, "discord", SimpleNamespace(VoiceClient=fx.FakeVoiceClient)), \
         patch.object(adapter_mod, "DISCORD_AVAILABLE", True):
        await _join(adapter)
        adapter._voice_clients[fx.GUILD].closed_by_discord()
        await async_wait_until(lambda: probe.records(homes.default), timeout=10, what="disconnect end")
        await rig.settle()

    [record] = probe.records(homes.default)
    assert record["reason"] == "disconnect"
    assert not adapter.is_in_voice_channel(fx.GUILD)
    assert fx.GUILD not in adapter._voice_receivers and fx.GUILD not in adapter._voice_text_channels


# --- T5: shutdown fires, its follow-up is dropped; a recycle is a disconnect -------------------------


# (gateway _running, _draining) at adapter teardown -> the reason the hook sees.
_TEARDOWN_STATES = {
    "stop": (False, True, "shutdown"),            # GatewayRunner.stop(): _stop_begin_teardown
    "restart_drain": (True, True, "shutdown"),    # request_restart(): still running, draining
    "recycle": (True, False, "disconnect"),       # adapter recycle / profile removal, gateway runs on
}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", list(_TEARDOWN_STATES))
async def test_t5_teardown_fires_and_only_a_real_shutdown_drops_the_followup(homes, caplog, state):
    running, draining, expected_reason = _TEARDOWN_STATES[state]
    probe.install_probe(homes.default, returns={"followup_turn": WRAP_UP}, allow_injection=True)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    adapter._client.close = AsyncMock()
    await _join(adapter)
    await _speak(adapter)
    rig.runner._running, rig.runner._draining = running, draining   # the injector stays published

    with caplog.at_level(logging.INFO), caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
        await adapter.disconnect()
        await rig.settle()

    [record] = probe.records(homes.default)
    dropped = [r for r in caplog.records if "dropping follow-up" in r.getMessage() and "reason=shutdown" in r.getMessage()]
    assert not [r for r in caplog.records if "still running after" in r.getMessage()], \
        "disconnect() had to abandon the end handler"
    assert record["reason"] == expected_reason
    if expected_reason == "shutdown":
        assert dropped and not rig.injector.called
    else:  # an adapter recycle while the gateway keeps running: a disconnect, follow-up allowed
        assert not dropped and rig.injector.call_count == 1


@pytest.mark.asyncio
async def test_t5_a_hung_hook_does_not_hold_shutdown_past_its_budget(homes):
    probe.install_probe(homes.default, block=True)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    adapter._client.close = AsyncMock()
    await _join(adapter)
    rig.runner._running, rig.runner._draining = False, True
    try:
        await adapter.disconnect()      # bounded by its own 2 s wait; the hook call by its 2 s timeout
        assert probe.started(homes.default, "vse-probe").is_set(), "the hook never ran"
        assert not probe.finished(homes.default, "vse-probe").is_set(), "disconnect() waited for the hung hook"
        await rig.settle()              # the dispatch finishes too: the hung call was abandoned
        assert not probe.finished(homes.default, "vse-probe").is_set()
    finally:
        probe.gate(homes.default, "vse-probe").set()
        await rig.settle()


# --- T6/T7/T8: the follow-up: system-originated, text only, behind the injection gate ---------------


def test_t8_a_followup_from_a_callback_no_plugin_owns_is_never_selected():
    """Shell hooks (and anything else no plugin registered) have no injection gate to honour."""
    from gateway.run_voice_session_end import select_voice_followup
    assert select_voice_followup([(None, {"followup_turn": "from a shell hook"})]) is None
    assert select_voice_followup([(None, {"followup_turn": "from a shell hook"}),
                                  ("alpha", {"followup_turn": "from alpha"})]) == ("alpha", "from alpha")


async def _followup_session(homes, *, allow: bool):
    probe.install_probe(homes.default, returns={"followup_turn": WRAP_UP}, allow_injection=allow)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    await _join(adapter)
    await _speak(adapter)
    [voice_event] = _voice_events(adapter)
    stored_key, stored_id = _stored_key_of(rig, voice_event)
    # The chat speaks every reply (mode "all") and auto-TTS is on for it: the strongest TTS setting.
    rig.runner._voice_mode[rig.runner._voice_key_for_source(voice_event.source)] = "all"
    adapter._auto_tts_enabled_chats.add(str(fx.TEXT))
    await adapter.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="no_authorized_users")
    await rig.settle()
    return rig, adapter, voice_event, stored_key, stored_id


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [True, False])
async def test_t8_the_followup_is_behind_the_plugins_injection_gate(homes, caplog, allow):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rig, adapter, _voice, stored_key, _sid = await _followup_session(homes, allow=allow)
    assert len(probe.records(homes.default)) == 1           # the hook fired either way
    if allow:
        await async_wait_until(lambda: _followups(adapter), timeout=10, what="follow-up turn")
        assert rig.injector.call_args.kwargs["session_key"] == stored_key
    else:
        assert not rig.injector.called
        assert any("gateway injection denied" in r.getMessage() for r in caplog.records)
        assert _followups(adapter) == []


@pytest.mark.asyncio
async def test_t6_t7_the_followup_is_a_system_text_turn_that_is_never_spoken(homes):
    from gateway.run import GatewayRunner
    rig, adapter, _voice, stored_key, stored_id = await _followup_session(homes, allow=True)
    [followup] = await async_wait_until(lambda: _followups(adapter), timeout=10, what="follow-up turn")

    # Into the session the transcript went into, pinned.
    assert followup.metadata["gateway_session_key"] == stored_key
    assert followup.metadata["gateway_session_id"] == stored_id
    assert followup.metadata["hermes_plugin_id"] == "vse-probe"
    # T7: system-originated, recorded as an internal notification row.
    assert followup.internal is True and followup.message_type == MessageType.TEXT
    assert followup.text.startswith(f"[system] {WRAP_UP}") and followup.text.endswith(INTERNAL_NOTIFICATION_FOOTER)
    prepared = SimpleNamespace(message_text=followup.text, persist_user_message=None, persist_user_timestamp=None,
                               persist_user_display_kind=display_kind_for_event(followup), persistence_owner=None)
    row = GatewayRunner._hmwa_user_transcript_entry(followup, prepared, 0.0)
    assert row["display_kind"] == "internal_notification" and row["content"] == followup.text
    # T6: text only, although the chat speaks every reply.
    assert followup.metadata[VOICE_REPLY_SUPPRESSED_KEY] is True
    assert rig.runner._should_send_voice_reply(followup, "Today: pricing, Games, one open question.", []) is False
    control = MessageEvent(text=followup.text, message_type=MessageType.TEXT, source=followup.source,
                           internal=True, metadata={k: v for k, v in followup.metadata.items()
                                                    if k != VOICE_REPLY_SUPPRESSED_KEY})
    assert rig.runner._should_send_voice_reply(control, "Today: pricing, Games, one open question.", []) is True
    # Point 3, streaming TTS: never starts for the follow-up (a TEXT turn); control: a VOICE turn does.
    started = []

    class _Consumer:
        active = True

        def __init__(self, **_kw):
            pass

        def start(self):
            started.append(True)

    with patch("gateway.streaming_tts_consumer.StreamingTTSConsumer", _Consumer), \
         patch("tools.tts_tool._load_tts_config", return_value={}):
        holder = [None]
        rig.runner._run_agent_start_streaming_tts(followup.source, followup.message_type, None, holder)
        assert (holder[0], started) == (None, [])
        rig.runner._run_agent_start_streaming_tts(followup.source, MessageType.VOICE, None, holder)
        assert isinstance(holder[0], _Consumer) and started == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("flagged", [True, False])
async def test_t6_a_followup_queued_behind_a_voice_turn_is_not_spoken_either(homes, flagged):
    """The leave flushes the last utterance first, so the follow-up usually queues behind a running
    VOICE turn and is answered in-band; the final send is then judged against that VOICE event."""
    from gateway.run import GatewayRunner
    rig, adapter, voice_event, _key, _sid = await _followup_session(homes, allow=True)
    [followup] = await async_wait_until(lambda: _followups(adapter), timeout=10, what="follow-up turn")
    pending = followup if flagged else MessageEvent(
        text=followup.text, message_type=MessageType.TEXT, source=followup.source, internal=True, metadata={})
    runner = rig.runner
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "Today: pricing.", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=pending.text)
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(source=voice_event.source, session_id="sid", session_key=_key, run_generation=1,
                               _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
                               result_holder=[None])
    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending=pending.text, pending_event=pending,
        response="first reply", result={"interrupted": True, "messages": []}, stream_task=None)
    copy_queued_terminal_metadata(voice_event, merged)       # the outer final send's call site

    interrupt = asyncio.Event()
    speaks = (runner._should_send_voice_reply(voice_event, "Today: pricing.", [], already_sent=True),
              adapter._wants_auto_tts(voice_event, _key, interrupt, "Today: pricing.", []))
    assert speaks == ((False, False) if flagged else (True, True))


# --- T9 / R10: only the profile whose bot left ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("leaver", ["work", "default"])
async def test_t9_only_the_leaving_bots_profile_gets_the_event(homes, monkeypatch, leaver):
    probe.install_probe(homes.default)
    probe.install_probe(homes.work)
    probe.discover(homes.default)
    probe.discover(homes.work)
    monkeypatch.setattr("gateway.run._multiplex_profile_homes",
                        lambda _cfg: [("default", homes.default), ("work", homes.work)])
    rig = Rig(homes.tmp, multiplex=True, primary_name="default")
    primary, work = rig.add_adapter(), rig.add_adapter(owner="work", bot_user_id=998)
    await _join(primary, fx.VOICE_A)
    await _join(work, fx.VOICE_B)
    leaving, staying = (work, primary) if leaver == "work" else (primary, work)
    await _speak(leaving)
    kept = staying._voice_sessions[fx.GUILD]

    await leaving.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="no_authorized_users")
    await rig.settle()

    leaving_home, staying_home = (homes.work, homes.default) if leaver == "work" else (homes.default, homes.work)
    leaving_records, staying_records = probe.records(leaving_home), probe.records(staying_home)
    assert (len(leaving_records), staying_records) == (1, []), (leaving_records, staying_records)
    record = leaving_records[0]
    assert record["profile"] == leaver
    assert record["session_key"] and record["session_key"].startswith(f"agent:{'main' if leaver == 'default' else leaver}:")
    assert staying._voice_sessions[fx.GUILD] is kept and staying.is_in_voice_channel(fx.GUILD)
    await fx.stop_voice_tasks(staying)


@pytest.mark.asyncio
async def test_r10_two_bots_one_guild_the_other_bots_events_leave_this_session_alone(homes, monkeypatch):
    """Both bots' real voice-state handlers see each other's own-state events as ordinary members."""
    probe.install_probe(homes.default)
    probe.install_probe(homes.work)
    probe.discover(homes.default)
    probe.discover(homes.work)
    monkeypatch.setattr("gateway.run._multiplex_profile_homes",
                        lambda _cfg: [("default", homes.default), ("work", homes.work)])
    rig = Rig(homes.tmp, multiplex=True, primary_name="default")
    primary, work = rig.add_adapter(), rig.add_adapter(owner="work", bot_user_id=998)
    handlers = {}
    for name, adapter in (("default", primary), ("work", work)):
        handlers[name] = await fx.connect_capturing_handlers(adapter, monkeypatch)
        adapter._auto_join_cfg, adapter._voice_timeout_seconds = {"enabled": True}, 0
    room_a = fx.voice_channel(primary, fx.VOICE_A)
    room_b = fx.voice_channel(work, fx.VOICE_B)
    try:
        await primary.join_voice_channel(room_a, text_channel_id=fx.TEXT)
        await work.join_voice_channel(room_b, text_channel_id=fx.TEXT)
        kept = primary._voice_sessions[fx.GUILD]

        # The work bot's driver leaves room B; both handlers see it, then the work bot's own leave.
        driver = fx.member(SPEAKER, work._fake_guild)
        work_bot = SimpleNamespace(id=998, guild=work._fake_guild, display_name="bot-998")
        for name in ("default", "work"):
            await handlers[name]["on_voice_state_update"](driver, fx.voice_state(room_b), fx.voice_state(None))
        for name in ("default", "work"):
            await handlers[name]["on_voice_state_update"](work_bot, fx.voice_state(room_b), fx.voice_state(None))
        await rig.settle()

        assert [r["reason"] for r in probe.records(homes.work)] == ["auto_leave"]
        assert probe.records(homes.default) == []
        assert primary._voice_sessions[fx.GUILD] is kept and primary.is_in_voice_channel(fx.GUILD)

        await primary.leave_voice_channel(fx.GUILD)
        await rig.settle()
        assert [r["voice_session_id"] for r in probe.records(homes.default)] == [kept.voice_session_id]
    finally:
        for adapter in (primary, work):
            await adapter.disconnect()
        await rig.settle()
    assert len(probe.records(homes.work)) == 1 and len(probe.records(homes.default)) == 1


@pytest.mark.asyncio
async def test_t9_speech_routed_to_another_profile_records_no_session_and_starts_no_turn(homes, monkeypatch, caplog):
    from gateway.profile_routing import parse_profile_routes
    probe.install_probe(homes.work, returns={"followup_turn": WRAP_UP}, allow_injection=True)
    probe.discover(homes.work)
    other = homes.default / "profiles" / "other"
    other.mkdir()
    monkeypatch.setattr("gateway.run._multiplex_profile_homes",
                        lambda _cfg: [("default", homes.default), ("work", homes.work), ("other", other)])
    rig = Rig(homes.tmp, multiplex=True, primary_name="default")
    rig.runner.config.profile_routes = parse_profile_routes([
        {"name": "re-home", "platform": "discord", "bot_profile": "work", "user_id": str(SPEAKER), "profile": "other"}])
    rig.runner._profile_adapters["other"] = {}
    work = rig.add_adapter(owner="work", bot_user_id=998)
    await _join(work)
    await _speak(work)
    [voice_event] = _voice_events(work)
    assert voice_event.source.profile == "other"           # the route really re-homed the turn

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await work.leave_voice_channel(fx.GUILD)
        await rig.settle()

    [record] = probe.records(homes.work)
    assert (record["participants"], record["session_key"], record["session_id"]) == ([str(SPEAKER)], None, None)
    assert not rig.injector.called
    assert any("no session to continue" in r.getMessage() for r in caplog.records)


# --- T10: a failing or hung hook never blocks the leave ------------------------------------------------


@pytest.mark.asyncio
async def test_t10_a_raising_hook_runs_after_teardown_and_never_fails_the_leave(homes):
    probe.install_probe(homes.default, **{"raise": True})
    probe.install_probe(homes.default, "wrap-up", returns={"followup_turn": WRAP_UP}, allow_injection=True)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    probe.set_check(homes.default, "vse-probe", lambda _kw: (
        fx.GUILD in adapter._voice_clients, fx.GUILD in adapter._voice_text_channels,
        fx.GUILD in adapter._voice_sessions))
    await _join(adapter)
    await _speak(adapter)

    await adapter.leave_voice_channel(fx.GUILD)      # returns: never raises, never waits on the hook
    await rig.settle()

    assert probe.CHECK_RESULTS == [("vse-probe", (False, False, False))]
    # The raise is isolated to its callback: the other plugin was called and its follow-up started.
    assert sorted(r["plugin"] for r in probe.records(homes.default)) == ["vse-probe", "wrap-up"]
    assert rig.injector.call_count == 1 and rig.injector.call_args.kwargs["plugin_id"] == "wrap-up"
    [followup] = await async_wait_until(lambda: _followups(adapter), timeout=10, what="follow-up turn")
    assert followup.metadata["hermes_plugin_id"] == "wrap-up"


@pytest.mark.asyncio
async def test_t10_a_hung_hook_is_abandoned_at_the_timeout_and_the_leave_does_not_wait(homes):
    probe.install_probe(homes.default, returns={"followup_turn": WRAP_UP}, allow_injection=True, block=True)
    cfg = homes.default / "config.yaml"
    import hermes_yaml as yaml
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["plugins"]["hook_callback_timeout"] = 1
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    await _join(adapter)
    await _speak(adapter)
    try:
        await adapter.leave_voice_channel(fx.GUILD)
        assert not adapter.is_in_voice_channel(fx.GUILD)
        assert not probe.finished(homes.default, "vse-probe").is_set()   # returned while the hook hangs
        await rig.settle()                                              # the dispatch gave up on it ...
        assert not probe.finished(homes.default, "vse-probe").is_set()   # ... while it still hangs
        assert not rig.injector.called                              # and its late result is never used
    finally:
        probe.gate(homes.default, "vse-probe").set()
    assert len(probe.records(homes.default)) == 1


# --- F5: an owner profile that no longer resolves ---------------------------------------------------


@pytest.mark.asyncio
async def test_t9_an_unresolvable_owner_profile_skips_the_hook(homes, monkeypatch, caplog):
    """The scope cannot be entered: the event must not fall through to the launch profile's plugins."""
    import hermes_cli.profiles as profiles
    probe.install_probe(homes.default, returns={"followup_turn": WRAP_UP}, allow_injection=True)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter(owner="ghost", bot_user_id=998)
    real_get_profile_dir = profiles.get_profile_dir

    def get_profile_dir(name):
        if name == "ghost":
            raise OSError("profile home vanished mid-run")
        return real_get_profile_dir(name)

    monkeypatch.setattr(profiles, "get_profile_dir", get_profile_dir)
    await _join(adapter)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await adapter.leave_voice_channel(fx.GUILD)
        await rig.settle()

    assert probe.records(homes.default) == []
    assert not rig.injector.called
    assert any(r.levelno == logging.WARNING and "on_voice_session_end" in r.getMessage() and "ghost" in r.getMessage()
               for r in caplog.records)


# --- F6: the last utterance opened the session ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", ["prompt", "slow"])
async def test_t6_a_session_opened_by_the_final_utterance_gets_its_id_and_the_followup(homes, monkeypatch, turn):
    """A single utterance, flushed inside the leave, is the session's first message: its turn opens
    the session from a background task, possibly after the end dispatch first looks it up."""
    probe.install_probe(homes.default, returns={"followup_turn": WRAP_UP}, allow_injection=True)
    probe.discover(homes.default)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    store = rig.runner.session_store
    misses = []
    real_lookup = store.lookup_by_session_key

    def lookup(session_key):
        entry = real_lookup(session_key)
        if entry is None:
            misses.append(session_key)
        return entry

    monkeypatch.setattr(store, "lookup_by_session_key", lookup)
    if turn == "slow":
        rig.turn_may_open.clear()
    await _join(adapter)
    fx.FakeReceiver.instances[-1].pending = [(SPEAKER, b"\x00" * 9600)]

    with patch("tools.transcription_tools.transcribe_audio",
               return_value={"success": True, "transcript": "one open question"}):
        await adapter.leave_voice_channel(fx.GUILD, reason="auto_leave", trigger="no_authorized_users")
    if turn == "slow":
        await async_wait_until(lambda: misses, timeout=10, what="the end dispatch looking the session up")
        rig.turn_may_open.set()
    await rig.settle()

    [voice_event] = _voice_events(adapter)
    key = rig.runner._session_key_for_source(voice_event.source)
    entry = real_lookup(key)
    assert entry is not None, "the voice turn never opened its session"
    [record] = probe.records(homes.default)
    assert (record["session_key"], record["session_id"]) == (key, entry.session_id)
    [followup] = await async_wait_until(lambda: _followups(adapter), timeout=10, what="follow-up turn")
    assert (followup.metadata["gateway_session_key"], followup.metadata["gateway_session_id"]) == (key, entry.session_id)


# --- F7: every hook call gets a finite bound ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("configured, state, expected", [
    (0, "command", 30.0), (10, "command", 10.0), (None, "command", 30.0), (0, "stop", 2.0), (None, "stop", 2.0)])
async def test_t10_every_hook_call_gets_a_finite_timeout(homes, monkeypatch, configured, state, expected):
    """``plugins.hook_callback_timeout: 0`` turns the bounded worker off for every hook; this hook must
    still never pin a default-executor thread for as long as a plugin hangs."""
    import hermes_cli.lifecycle as lifecycle
    import hermes_yaml as yaml
    probe.install_probe(homes.default)
    if configured is not None:
        cfg = homes.default / "config.yaml"
        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        data["plugins"]["hook_callback_timeout"] = configured
        cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    probe.discover(homes.default)
    timeouts = []
    real_invoke = lifecycle.invoke_hook_attributed

    def invoke(hook_name, *, timeout=None, **kwargs):
        timeouts.append(timeout)
        return real_invoke(hook_name, timeout=timeout, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook_attributed", invoke)
    rig = Rig(homes.tmp)
    adapter = rig.add_adapter()
    adapter._client.close = AsyncMock()
    await _join(adapter)
    if state == "stop":
        rig.runner._running, rig.runner._draining = False, True
        await adapter.disconnect()
    else:
        await adapter.leave_voice_channel(fx.GUILD)
    await rig.settle()

    assert timeouts == [expected]
    assert len(probe.records(homes.default)) == 1
