"""Loop-guard-hold, a person releases (spec r15 S7): every Discord human drop path, the runner-side reset before the
early return, per-seat keys, ``/stop``. Drives the REAL DiscordAdapter's drop sites against a real runner
(tests/gateway/lgh_support.py's contract). Adapters are built in the test bodies, not in fixtures, so an RED run fails
by assertion.
"""

from __future__ import annotations

import itertools
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, PERSON, RELAY, THREAD, Cfg, admit, clock, conv, end_turn, event, fill_window, handed, make_runner,
    pump, start_turn,
)

CH = "1571780464853647401"  # a plain server channel
PARENT = "1571780464853647402"  # the parent channel of THREAD
SELF = {"boba": 999, "tars": 998}
OTHER_BOT = SimpleNamespace(id=555, bot=True)
SOMEONE = SimpleNamespace(id=820, bot=False)

_ENV = ("DISCORD_REQUIRE_MENTION", "DISCORD_THREAD_REQUIRE_MENTION", "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_FREE_RESPONSE_AUTO_THREAD", "DISCORD_AUTO_THREAD", "DISCORD_NO_THREAD_CHANNELS",
        "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS", "DISCORD_HISTORY_BACKFILL", "DISCORD_ALLOW_BOTS",
        "DISCORD_IGNORE_NO_MENTION", "DISCORD_BOTS_REQUIRE_INLINE_MENTION")


class _Chan:
    def __init__(self, channel_id, parent=None):
        self.id = int(channel_id)
        self.name = "chan"
        self.guild = SimpleNamespace(name="Hermes Server", id=1)
        self.topic = None
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.send = AsyncMock()

    def history(self, **kw):
        async def _it():
            return
            yield
        return _it()


class _Thread(_Chan):
    pass


class _DM:
    pass


def _discord_env(monkeypatch, **env):
    import plugins.platforms.discord.adapter as mod
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mod.discord, "Thread", _Thread, raising=False)
    monkeypatch.setattr(mod.discord, "DMChannel", _DM, raising=False)


def _seat(runner, seat="boba", *, allowed=True):
    """A real DiscordAdapter serving ``seat`` on ``runner``; what it admits goes to the runner's ingress gate."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=SELF[seat], bot=True))
    adapter._text_batch_delay_seconds = 0
    adapter._owner_profile = seat
    adapter._is_allowed_user = Mock(return_value=allowed)
    adapter._ready_event.set()
    adapter.gateway_runner = runner
    adapter.admitted = []

    async def _to_runner(ev):
        adapter.admitted.append(ev)
        return await runner._hm_admit_event(ev)
    adapter.handle_message = AsyncMock(side_effect=_to_runner)
    return adapter


def _runner(clock, cfg=None, **block):
    runner = make_runner(clock, cfg or Cfg(**block))
    runner._adapter_profile_for_source = lambda source: getattr(source, "profile", None) or ""
    runner._profile_name_for_source = lambda source, adapter_profile=None: adapter_profile
    runner._test_clock = clock
    return runner


_MSG_IDS = itertools.count(1571500000000000000)


def _msg(channel, content, *, author=None, mentions=(), msg_id=None):
    msg_id = msg_id or next(_MSG_IDS)
    return SimpleNamespace(
        id=msg_id, content=content, mentions=list(mentions), attachments=[], reference=None,
        created_at=datetime.now(UTC), channel=channel, guild=channel.guild,
        author=author or SimpleNamespace(id=42, bot=False, display_name="Jezza", name="Jezza"),
        type=__import__("discord").MessageType.default)


def _bot_event(runner, seat, chat, chat_type, text=None, *, user=BOT_A):
    ev = event(user, text, chat=chat)
    ev.source.chat_type = chat_type
    ev.source.profile = seat
    return ev


async def _hold(runner, seat, chat, chat_type="group", n=1):
    """The seat's window for ``chat`` is filled by a bot, then ``n`` bot messages are held. Returns them."""
    for _ in range(10):
        assert await admit(runner, _bot_event(runner, seat, chat, chat_type, user=BOT_B)) is True
    held = [_bot_event(runner, seat, chat, chat_type) for _ in range(n)]
    for ev in held:
        assert await admit(runner, ev) is False
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == held
    return held


async def _released(runner, held):
    """Within one pump wake the head of ``held`` is handed over (a release, not the 15 s timer)."""
    before = len(handed(runner.adapter))
    await pump(runner, runner_clock(runner))
    return handed(runner.adapter)[before:] == [held[0]]


def runner_clock(runner):
    return runner._test_clock


# --- T1: every S7 drop path resets and releases --------------------------------------------------------------


def _site_other_bot(chan):
    return _msg(chan, f"<@{OTHER_BOT.id}> over to you", mentions=[OTHER_BOT]), "other-bot-addressed"


def _site_no_mention_gate(chan):
    return _msg(chan, f"<@{SOMEONE.id}> have a look", mentions=[SOMEONE]), "mentions-others-not-free-channel"


def _site_mention_required(chan):
    return _msg(chan, "A"), "mention-required"


SITES = {
    "other-bot-addressed": _site_other_bot,
    "no-mention-gate": _site_no_mention_gate,
    "mention-required": _site_mention_required,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(SITES))
async def test_T1_each_adapter_drop_site_resets_and_releases(clock, monkeypatch, caplog, site):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    caplog.set_level(logging.INFO)
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH)
    message, reason = SITES[site](_Chan(CH))
    assert await adapter._dispatch_discord_message(message) is False  # the adapter drops it
    assert f"reason={reason}" in caplog.text  # at the site this case means to reach
    adapter.handle_message.assert_not_awaited()
    assert await _released(runner, held) is True  # but the person's message released the hold


@pytest.mark.asyncio
async def test_T1_the_bare_mention_only_ping_releases(clock, monkeypatch, caplog):
    _discord_env(monkeypatch)
    caplog.set_level(logging.INFO)
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH)
    me = adapter._client.user
    assert await adapter._dispatch_discord_message(_msg(_Chan(CH), f"<@{me.id}>", mentions=[me])) is False
    assert "Ignoring mention-only message" in caplog.text
    assert await _released(runner, held) is True


@pytest.mark.asyncio
async def test_T1_the_auto_thread_failure_releases(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_AUTO_THREAD="true")
    runner = _runner(clock)
    adapter = _seat(runner)
    adapter._auto_create_thread = AsyncMock(return_value=None)
    held = await _hold(runner, "boba", CH)
    me = adapter._client.user
    chan = _Chan(CH)
    assert await adapter._dispatch_discord_message(_msg(chan, f"<@{me.id}> hi", mentions=[me])) is False
    adapter._auto_create_thread.assert_awaited()
    assert await _released(runner, held) is True


@pytest.mark.asyncio
async def test_T1_the_recovered_dispatch_pre_admission_drop_releases_and_reports_once(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    adapter = _seat(runner)
    runner._bot_loop_human_seen = Mock(wraps=runner._bot_loop_human_seen)
    held = await _hold(runner, "boba", CH)
    message = _msg(_Chan(CH), "A", msg_id=1571999999999999999)
    assert await adapter._dispatch_recovered_message(message) is False
    assert runner._bot_loop_human_seen.call_count == 1  # one report for one message id (T2)
    assert await _released(runner, held) is True


# --- T2: the key, per seat -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T2_the_report_keys_on_the_thread_id_in_a_thread_and_the_channel_id_otherwise(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true", DISCORD_THREAD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    adapter = _seat(runner)
    held_thread = await _hold(runner, "boba", THREAD, "thread")
    held_channel = await _hold(runner, "boba", CH, "group")
    parent = _Chan(PARENT)
    # a drop in the thread's parent channel releases neither the thread nor the unrelated channel
    await adapter._dispatch_discord_message(_msg(parent, "A", msg_id=1571000000000000001))
    assert await _released(runner, held_thread) is False
    # a drop in the thread releases the thread's hold only
    await adapter._dispatch_discord_message(_msg(_Thread(THREAD, parent), "A", msg_id=1571000000000000002))
    assert await _released(runner, held_thread) is True
    assert runner._bot_loop_guard.held_events(conv(runner, held_channel[0])) == held_channel
    # a drop in the channel releases the channel's hold
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=1571000000000000003))
    assert await _released(runner, held_channel) is True


@pytest.mark.asyncio
async def test_T2_a_drop_in_seat_A_touches_only_seat_A(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    boba, tars = _seat(runner, "boba"), _seat(runner, "tars")
    held_boba = await _hold(runner, "boba", CH)
    held_tars = await _hold(runner, "tars", CH)
    await boba._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=1571000000000000004))
    assert await _released(runner, held_boba) is True
    assert await _released(runner, held_tars) is False  # tars saw nothing: its entry waits for its own timer
    assert tars.handle_message.await_count == 0


# --- T3: what releases nothing -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T3_a_dropped_bot_a_user_not_allowed_human_and_a_non_allowed_channel_reset_nothing(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true", DISCORD_ALLOW_BOTS="none")
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH)
    bot = SimpleNamespace(id=777, bot=True, display_name="peer", name="peer")
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "<@999> hi", author=bot, msg_id=1571000000000000011))
    adapter._is_allowed_user = Mock(return_value=False)  # user-not-allowed
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=1571000000000000012))
    adapter._is_allowed_user = Mock(return_value=True)
    monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "123")  # this channel is not on the seat's list
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=1571000000000000013))
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS")
    monkeypatch.setenv("DISCORD_IGNORED_CHANNELS", CH)  # an ignored channel
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=1571000000000000014))
    assert await _released(runner, held) is False
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == held


@pytest.mark.asyncio
async def test_T3_a_human_relay_dropped_before_its_allowlist_check_resets_nothing_after_it_releases(clock, monkeypatch):
    runner = _runner(clock, human_relays=[RELAY])
    relay = SimpleNamespace(id=int(RELAY), bot=True, display_name="vm-chat", name="vm-chat")
    # before the allowlist check: allow_bots none -> bot-not-allowed
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true", DISCORD_ALLOW_BOTS="none")
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH)
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "<player> hi", author=relay, msg_id=1571000000000000021))
    assert await _released(runner, held) is False
    # after it: allow_bots mentions, no mention -> bot-mention-required
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "<player> hi", author=relay, msg_id=1571000000000000022))
    assert await _released(runner, held) is True


# --- T26 [Q7 A]: /stop discards and lists, a plain message releases ------------------------------------------


@pytest.mark.asyncio
async def test_T26_Q7_a_persons_stop_discards_the_held_messages_and_posts_one_line_listing_them(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=3)
    assert await admit(runner, event(PERSON, "/stop", is_bot=False, chat=THREAD)) is True
    await pump(runner, clock, advance=3600)
    assert handed(runner.adapter) == []  # discarded, not delivered
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == []
    lines = [text for _chat, text in runner.adapter.notices if all(h.message_id in text for h in held)]
    assert len(lines) == 1  # one line listing them (sender and link)


@pytest.mark.asyncio
async def test_T26_Q7_a_plain_message_delivers(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=2)
    assert await admit(runner, event(PERSON, "carry on", is_bot=False, chat=THREAD)) is True
    await pump(runner, clock)
    assert handed(runner.adapter) == [held[0]]


# --- T27: another seat's disallowed person -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T27_a_person_seat_B_does_not_allow_does_not_release_Bs_holds_even_if_seat_A_allows_them(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    boba, tars = _seat(runner, "boba", allowed=True), _seat(runner, "tars", allowed=False)
    held_boba = await _hold(runner, "boba", CH)
    held_tars = await _hold(runner, "tars", CH)
    for adapter, msg_id in ((boba, 1571000000000000031), (tars, 1571000000000000031)):
        await adapter._dispatch_discord_message(_msg(_Chan(CH), "A", msg_id=msg_id))
    assert await _released(runner, held_boba) is True
    assert await _released(runner, held_tars) is False
    assert runner._bot_loop_guard.held_events(conv(runner, held_tars[0])) == held_tars


# --- T28 -----------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T28_a_paused_conversation_with_no_tracked_chain_is_released_by_an_admitted_person(clock):
    runner = _runner(clock, hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15, cooldown_seconds=60)
    held = await _hold(runner, "boba", THREAD, "thread", n=3)
    await pump(runner, clock, advance=15)
    first = handed(runner.adapter)[0]
    runner._bot_loop_hold_turn_claimed(first)
    runner._bot_loop_hold_turn_ended(first)
    guard = runner._bot_loop_guard
    assert guard.is_paused(conv(runner, held[0]))
    clock.advance(200)  # past cooldown_seconds: any call that sweeps forgets the idle chain
    guard.admit(("x", "discord", "elsewhere"))
    assert guard.tracking_chains is False  # so the early return at authz_mixin.py:801-803 would skip the reset
    assert await admit(runner, event(PERSON, "hello?", is_bot=False, chat=THREAD)) is True
    assert guard.is_paused(conv(runner, held[0])) is False
    await pump(runner, clock)
    assert handed(runner.adapter)[-1] is held[1]


# --- T36, T37 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T36_the_native_stop_slash_command_discards_like_the_typed_one(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=2)
    native = event(PERSON, "/stop", is_bot=False, chat=THREAD)
    native.message_type = MessageType.COMMAND  # the slash command reaches the runner as a command event
    assert await admit(runner, native) is True
    await pump(runner, clock, advance=3600)
    assert handed(runner.adapter) == []
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == []


@pytest.mark.asyncio
async def test_T37_a_persons_voice_utterance_releases_like_a_typed_message(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=2)
    utterance = event(PERSON, "what is the status", is_bot=False, chat=THREAD)
    utterance.message_id = None  # speech has no message id (run_voice.py:568)
    assert await admit(runner, utterance) is True
    await pump(runner, clock)
    assert handed(runner.adapter) == [held[0]]


# --- T40, T45: the /stop rule --------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("text,mentions", [
    ("/stop", []),
    (f"<@{OTHER_BOT.id}> /stop", [OTHER_BOT]),  # other-bot-addressed
    (f"<@{SOMEONE.id}> /stop", [SOMEONE]),  # the no-mention gate
])
async def test_T40_a_typed_stop_dropped_by_the_adapter_discards(clock, monkeypatch, text, mentions):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH, n=2)
    assert await adapter._dispatch_discord_message(_msg(_Chan(CH), text, mentions=mentions)) is False
    await pump(runner, clock, advance=3600)
    assert handed(runner.adapter) == []
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == []


@pytest.mark.asyncio
async def test_T40_at_bot_stop_admitted_by_the_mentioned_seat_discards_too(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH, n=2)
    me = adapter._client.user
    assert await adapter._dispatch_discord_message(_msg(_Chan(CH), f"<@{me.id}> /stop", mentions=[me])) is True
    await pump(runner, clock, advance=3600)
    assert handed(runner.adapter) == []
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == []


_BOBA, _TARS = SimpleNamespace(id=999, bot=True), SimpleNamespace(id=998, bot=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,mentions,discards", [
    ("/stop", [], True),
    ("/STOP", [], True),
    ("/stop now", [], True),
    ("/stop@Boba", [], True),  # each dropped at a mention-required site
    ("<@999> /stop", [_BOBA], True),  # Boba admits it, TARS drops it (other-bot-addressed)
    ("<@999> <@998> /stop", [_BOBA, _TARS], True),  # both admit it
    ("<@999> hello /stop", [_BOBA], False),
    ("/new", [], False),
])
async def test_T45_seat_by_seat_stop_discards_at_both_seats_and_anything_else_releases_at_both(
        clock, monkeypatch, text, mentions, discards):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _runner(clock)
    boba, tars = _seat(runner, "boba"), _seat(runner, "tars")
    held = {seat: await _hold(runner, seat, CH, n=2) for seat in ("boba", "tars")}
    for adapter, seat in ((boba, "boba"), (tars, "tars")):
        await adapter._dispatch_discord_message(_msg(_Chan(CH), text, mentions=mentions, msg_id=1571000000000000045))
    await pump(runner, clock)
    if discards:
        await pump(runner, clock, advance=3600)
        assert handed(runner.adapter) == []
        for seat in held:
            assert runner._bot_loop_guard.held_events(conv(runner, held[seat][0])) == []
    else:
        assert {id(e) for e in handed(runner.adapter)} == {id(held[s][0]) for s in held}  # both seats release


# --- T49 -----------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T49_a_dropped_person_and_an_admitted_new_each_release_within_one_pump_wake_without_a_warning(
        clock, monkeypatch, caplog):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    caplog.set_level(logging.WARNING)
    runner = _runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH, n=2)
    await adapter._dispatch_discord_message(_msg(_Chan(CH), f"<@{OTHER_BOT.id}> hi", mentions=[OTHER_BOT]))
    assert await _released(runner, held) is True
    runner._bot_loop_hold_turn_claimed(held[0])
    runner._bot_loop_hold_turn_ended(held[0])
    held2 = await _hold(runner, "boba", THREAD, "thread", n=2)
    assert await admit(runner, event(PERSON, "/new", is_bot=False, chat=THREAD)) is True
    assert await _released(runner, held2) is True
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # no 60 s wait, no warning


@pytest.mark.asyncio
async def test_T49_while_an_admitted_persons_turn_runs_under_the_heads_key_no_released_message_starts(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=2)
    person = event(PERSON, "carry on", is_bot=False, chat=THREAD)
    person.source.profile = "boba"
    key = start_turn(runner, person)  # the person's turn runs under the same session key
    assert await admit(runner, person) is True
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=600)
    assert handed(runner.adapter) == []
    end_turn(runner, key)
    await pump(runner, clock)
    assert handed(runner.adapter) == [held[0]]


@pytest.mark.asyncio
async def test_T49_a_person_the_runner_refuses_releases_nothing_on_the_cold_path_and_the_busy_path(clock):
    runner = _runner(clock)
    held = await _hold(runner, "boba", THREAD, "thread", n=2)
    runner._is_user_authorized_for_source = lambda source, **kw: False
    refused = event(PERSON, "let me in", is_bot=False, chat=THREAD)
    refused.source.profile = "boba"
    assert await admit(runner, refused) is False  # cold path
    key = runner._session_key_for_source(refused.source)
    runner._session_state(key).turn.agent = object()
    assert await runner._handle_active_session_busy_message(refused, key) is True  # busy path, run_busy.py:960
    runner._session_state(key).turn.agent = None
    await pump(runner, clock)
    assert handed(runner.adapter) == []
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == held
    assert runner._bot_loop_guard.is_paused(conv(runner, held[0])) is False
