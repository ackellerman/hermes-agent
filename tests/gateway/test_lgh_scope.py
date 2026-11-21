"""Loop-guard-hold scope (T25, T19): the hold is a Discord rule. Every other platform keeps the base guard
(20 messages / 300 s budget with a 600 s cooldown, the ``max_hops`` drop); on Discord neither applies and a
full window holds instead. ``reply_to_reply`` still drops on Discord, judged when the message arrives.
"""

from __future__ import annotations

import pytest

from gateway.session import Platform

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, CHAT, THREAD, Cfg, FakeClock, admit, clock, conv, event, handed, make_runner, pump,
)

TG_CHAT = "-1001234567890"


def _tg(user=BOT_A, text=None, chat=TG_CHAT):
    return event(user, text, chat=chat, platform=Platform.TELEGRAM)


# --- T25: a non-Discord platform behaves as on base -------------------------------------------------------


def test_telegram_budget_is_still_20_per_300_seconds_then_a_cooldown(clock):
    runner = make_runner(clock, Cfg())
    source = _tg().source
    verdicts = [runner._admit_bot_message_for_source(source) for _ in range(25)]
    assert verdicts == [True] * 20 + [False] * 5
    assert runner._bot_loop_guard.blocked(runner._bot_loop_guard_conversation(source)) is True
    clock.advance(599)
    assert runner._bot_loop_guard.blocked(runner._bot_loop_guard_conversation(source)) is True
    clock.advance(2)  # past the 600 s cooldown: a fresh budget
    assert runner._admit_bot_message_for_source(source) is True


@pytest.mark.asyncio
async def test_telegram_is_still_dropped_at_max_hops(clock):
    runner = make_runner(clock, Cfg(max_events=1000, max_hops=4, bot_replies_expect_reply=True))
    events = [_tg(text=f"ping {n}") for n in range(8)]
    verdicts = [await admit(runner, e) for e in events]
    assert verdicts == [True] * 4 + [False] * 4


@pytest.mark.asyncio
async def test_a_discord_source_is_never_dropped_for_max_hops_while_a_telegram_source_still_is(clock):
    cfg = Cfg(max_events=1000, max_hops=3, bot_replies_expect_reply=True)
    runner = make_runner(clock, cfg)
    discord = [event(BOT_A, f"ping {n}", chat=THREAD) for n in range(8)]
    telegram = [_tg(text=f"ping {n}") for n in range(8)]
    assert [await admit(runner, e) for e in discord] == [True] * 8  # base: dropped from the 4th hop
    assert [await admit(runner, e) for e in telegram] == [True] * 3 + [False] * 5
    assert runner._bot_loop_guard.held_events(conv(runner, discord[0])) == []


@pytest.mark.asyncio
async def test_a_discord_source_has_no_budget_cooldown_while_a_telegram_source_still_has_one(clock):
    runner = make_runner(clock, Cfg(max_events=5, max_hops=1000, bot_replies_expect_reply=True))
    discord = [event(BOT_A, f"ask {n}") for n in range(8)]
    telegram = [_tg(text=f"ask {n}") for n in range(8)]
    assert [await admit(runner, e) for e in discord] == [True] * 8  # base: budget of 5, then refused
    assert [await admit(runner, e) for e in telegram] == [True] * 5 + [False] * 3
    assert runner._bot_loop_guard.blocked(conv(runner, discord[0])) is False


@pytest.mark.asyncio
async def test_the_window_holds_a_discord_flood_and_never_a_telegram_one(clock):
    runner = make_runner(clock, Cfg(max_events=1000, max_hops=1000, bot_replies_expect_reply=True))
    discord = [event(BOT_A, f"ask {n}") for n in range(12)]
    telegram = [_tg(text=f"ask {n}") for n in range(12)]
    assert [await admit(runner, e) for e in discord] == [True] * 10 + [False] * 2
    assert runner._bot_loop_guard.held_events(conv(runner, discord[0])) == discord[10:]
    assert [await admit(runner, e) for e in telegram] == [True] * 12
    assert runner._bot_loop_guard.held_events(conv(runner, telegram[0])) == []


# --- T19 [Q5 A]: reply_to_reply still drops on Discord, judged at arrival only ----------------------------


@pytest.mark.asyncio
async def test_reply_to_reply_still_drops_on_discord_at_arrival(clock):
    runner = make_runner(clock, Cfg(max_events=1000))
    first = event(BOT_A, "what is 2+2?")
    assert await admit(runner, first) is True
    runner._note_bot_reply(first.source)  # our answer to BOT_A went out
    thanks = event(BOT_A, "thanks!")
    assert await admit(runner, thanks) is False  # dropped at arrival (a drop is not a hold: see the next test)
    assert await admit(runner, event(BOT_B, "hello")) is True  # another bot's message is not an answer


@pytest.mark.asyncio
async def test_a_released_message_is_not_dropped_for_reply_to_reply(clock):
    """C is held (window full) before any reply to its sender goes out; by the time it is released our reply has
    gone out. The rule judged C at arrival and never judges it again; a new message from the same sender,
    arriving now, is dropped."""
    runner = make_runner(clock, Cfg(max_events=1000, hold_after_messages=2))
    a, b = event(BOT_A, "first"), event(BOT_B, "second")
    assert await admit(runner, a) is True and await admit(runner, b) is True
    c = event(BOT_A, "third")
    assert await admit(runner, c) is False  # arrival: no reply to BOT_A yet, so held by the window
    assert runner._bot_loop_guard.held_events(conv(runner, c)) == [c]

    runner._note_bot_reply(a.source)  # our reply to BOT_A goes out while C waits
    d = event(BOT_A, "fourth")
    assert await admit(runner, d) is False  # arrival while we have answered BOT_A: reply_to_reply
    assert runner._bot_loop_guard.held_events(conv(runner, d)) == [c]  # D was dropped, not queued

    await pump(runner, clock, advance=130)  # the window has emptied and the hold is due
    assert handed(runner.adapter) == [c]
    assert await runner._hm_admit_event(c) is not None  # the released C passes the door again
