"""Loop-guard-hold, core (spec r15 S1-S5, S10, S11): the window, the backoff, the pause, the config keys and the asks.

Every test here fails on base (no hold exists). Names come from tests/gateway/lgh_support.py's contract.
All owner answers Q2-Q9 are A.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from gateway.bot_loop_guard import settings_from_config

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, PERSON, THREAD, Cfg, FakeClock, admit, clock, conv, event, fill_window, handed, make_runner,
    pump,
)


async def _deliver_one(runner, clock, adapter, wait):
    """Advance to just before ``wait`` s after the last delivery: nothing; then to ``wait``: exactly one more
    event is handed over; the runner then claims and ends its turn (as a real turn would)."""
    before = len(handed(adapter))
    await pump(runner, clock, advance=wait - 0.1)
    assert len(handed(adapter)) == before, f"handed over {wait - 0.1}s into a {wait}s wait"
    await pump(runner, clock, advance=0.1)
    assert len(handed(adapter)) == before + 1, f"not handed over after {wait}s"
    ev = handed(adapter)[-1]
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)
    return ev


# --- T4 -------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T4_the_11th_bot_message_in_120s_is_held_then_handed_over_at_15s_through_the_base_adapter(clock):
    runner = make_runner(clock, Cfg())
    adapter = runner.adapter
    adapter._handle_message = AsyncMock()  # the Discord entry: never called
    await fill_window(runner, clock, 10)
    eleventh = event(BOT_A, "the eleventh")
    assert await admit(runner, eleventh) is False
    assert runner._bot_loop_guard.held_events(conv(runner, eleventh)) == [eleventh]  # held, not dropped (S1, S2)
    await pump(runner, clock, advance=14.9)
    assert handed(adapter) == []
    await pump(runner, clock, advance=0.1)
    assert len(handed(adapter)) == 1 and handed(adapter)[0] is eleventh  # the very event object (S2)
    adapter._handle_message.assert_not_awaited()  # no second dedup or auto-thread (S2)


# --- T5 -------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T5_a_slow_exchange_is_never_held(clock):
    runner = make_runner(clock, Cfg())
    # 11 messages spread over more than 120 s: at most 10 ever inside one window.
    events = []
    for _ in range(11):
        events.append(event(BOT_A))
        assert await admit(runner, events[-1]) is True
        clock.advance(13)
    assert runner._bot_loop_guard.held_events(conv(runner, events[0])) == []
    # 30 bot messages at 60 s gaps with no person: none held, none dropped (no hop cap, no budget on Discord)
    for _ in range(30):
        assert await admit(runner, event(BOT_B)) is True
        clock.advance(60)
    assert runner._bot_loop_guard.held_events(conv(runner, events[0])) == []


# --- T6, T7 ---------------------------------------------------------------------------------------------


async def _runaway(runner, clock, extra=12):
    await fill_window(runner, clock, 10)
    held = [event(BOT_A) for _ in range(extra)]
    for ev in held:
        assert await admit(runner, ev) is False
    return held


@pytest.mark.asyncio
async def test_T6_a_fast_loop_waits_15_30_60_120_240_480_480_480_then_T7_pauses(clock):
    runner = make_runner(clock, Cfg())
    held = await _runaway(runner, clock)
    guard = runner._bot_loop_guard
    for wait in (15, 30, 60, 120, 240, 480, 480, 480):
        await _deliver_one(runner, clock, runner.adapter, wait)
    assert handed(runner.adapter) == held[:8]  # FIFO
    assert guard.is_paused(conv(runner, held[0])) is True  # three capped waits in a row (S5)
    n = len(handed(runner.adapter))
    await pump(runner, clock, advance=3600)
    await pump(runner, clock, advance=3600)
    assert len(handed(runner.adapter)) == n  # nothing handed over while paused
    assert guard.held_events(conv(runner, held[0])) == held[8:]


@pytest.mark.asyncio
async def test_T7_a_person_releases_a_paused_exchange_in_fifo_order_admitted_or_dropped(clock):
    for how in ("admitted", "dropped"):
        runner = make_runner(clock, Cfg(), profile=how)
        held = await _runaway(runner, clock, extra=11)
        for wait in (15, 30, 60, 120, 240, 480, 480, 480):
            await _deliver_one(runner, clock, runner.adapter, wait)
        assert runner._bot_loop_guard.is_paused(conv(runner, held[0]))
        person = event(PERSON, "are you still going?", is_bot=False)
        if how == "admitted":
            assert await admit(runner, person) is True
        else:  # a human message the Discord adapter dropped, reported by _report_human_seen
            runner._bot_loop_human_seen(person.source, person.text)
        assert runner._bot_loop_guard.is_paused(conv(runner, held[0])) is False
        done = len(handed(runner.adapter))
        for _ in range(3):  # released one at a time, FIFO
            await pump(runner, clock)
            ev = handed(runner.adapter)[-1]
            runner._bot_loop_hold_turn_claimed(ev)
            runner._bot_loop_hold_turn_ended(ev)
        assert handed(runner.adapter)[done:] == held[8:11]


# --- T8 [Q2] --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T8_Q2_level_returns_to_1_only_after_a_full_window_with_nothing_held(clock):
    runner = make_runner(clock, Cfg())
    held = await _runaway(runner, clock, extra=3)
    guard = runner._bot_loop_guard
    for wait in (15, 30, 60):
        await _deliver_one(runner, clock, runner.adapter, wait)
    key = conv(runner, held[0])
    assert guard.held_events(key) == [] and guard.hold_level(key) > 1
    clock.advance(119)
    await pump(runner, clock)
    assert guard.hold_level(key) > 1  # the window has room, but not a full 120 s with nothing held
    clock.advance(2)
    await pump(runner, clock)
    assert guard.hold_level(key) == 1


# --- T10 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T10_five_identical_held_messages_queue_once(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10)
    copies = [event(BOT_A, "same words, same sender") for _ in range(5)]
    for ev in copies:
        assert await admit(runner, ev) is False
    assert runner._bot_loop_guard.held_events(conv(runner, copies[0])) == [copies[0]]  # duplicate judged first (S10)


# --- T12 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T12_25_bot_messages_in_60s_are_held_never_dropped_and_all_delivered_in_order(clock):
    runner = make_runner(clock, Cfg(hold_runaway_at_cap=100))  # no pause: the point is nothing is lost
    events = [event(BOT_A) for _ in range(25)]
    verdicts = []
    for ev in events:
        verdicts.append(await admit(runner, ev))
        clock.advance(2.4)
    assert verdicts == [True] * 10 + [False] * 15
    assert runner._bot_loop_guard.held_events(conv(runner, events[0])) == events[10:]
    for _ in range(15):
        before = len(handed(runner.adapter))
        for _ in range(40):  # up to 40 x 15 s: the longest wait is 480 s
            await pump(runner, clock, advance=15)
            if len(handed(runner.adapter)) > before:
                break
        ev = handed(runner.adapter)[-1]
        runner._bot_loop_hold_turn_claimed(ev)
        runner._bot_loop_hold_turn_ended(ev)
    assert handed(runner.adapter) == events[10:]


# --- T16 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T16_each_key_changes_behaviour_and_an_edit_applies_without_a_restart(clock):
    cfg = Cfg(hold_after_messages=3)
    runner = make_runner(clock, cfg)
    await fill_window(runner, clock, 3)
    assert await admit(runner, event(BOT_A)) is False  # hold_after_messages: the 4th
    runner2 = make_runner(clock, Cfg(hold_after_messages=3, hold_window_seconds=10))
    for _ in range(5):  # a 10 s window: at most 2-3 inside it, so nothing is held (a 120 s window would hold the 4th)
        assert await admit(runner2, event(BOT_A)) is True
        clock.advance(4)
    assert runner2._bot_loop_guard.held_events(conv(runner2, event(BOT_A))) == []
    runner3 = make_runner(clock, Cfg(hold_initial_seconds=5, hold_max_seconds=20))
    await fill_window(runner3, clock, 10)
    held = event(BOT_A)
    assert await admit(runner3, held) is False
    await pump(runner3, clock, advance=5)
    assert handed(runner3.adapter) == [held]  # hold_initial_seconds
    runner4 = make_runner(clock, Cfg(hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15))
    held4 = await _runaway(runner4, clock, extra=3)
    await _deliver_one(runner4, clock, runner4.adapter, 15)
    assert runner4._bot_loop_guard.is_paused(conv(runner4, held4[0])) is True  # hold_runaway_at_cap=1, wait == cap


@pytest.mark.asyncio
async def test_T16_an_edit_raising_hold_after_messages_admits_the_next_message_of_a_conversation_full_at_the_old_value(clock):
    cfg = Cfg(hold_after_messages=3)
    runner = make_runner(clock, cfg)
    await fill_window(runner, clock, 3)  # the window is full at 3 (nothing held, so nothing keeps holding)
    probe = event(BOT_A)
    cfg.set(hold_after_messages=1000)  # no restart, the same conversation: the 4th is now inside the limit
    assert await admit(runner, probe) is True
    assert runner._bot_loop_guard.held_events(conv(runner, probe)) == []
    control = make_runner(clock, Cfg(hold_after_messages=3))  # the unedited value holds that very message
    await fill_window(control, clock, 3)
    assert await admit(control, event(BOT_A)) is False


@pytest.mark.asyncio
async def test_T16_an_edit_lowering_hold_after_messages_holds_the_next_message_of_a_conversation_inside_the_old_value(clock):
    cfg = Cfg(hold_after_messages=1000)
    runner = make_runner(clock, cfg)
    await fill_window(runner, clock, 3)
    cfg.set(hold_after_messages=3)  # the same conversation, three already in the window
    held = event(BOT_A)
    assert await admit(runner, held) is False
    assert runner._bot_loop_guard.held_events(conv(runner, held)) == [held]


@pytest.mark.asyncio
async def test_T16_the_pump_reads_each_entrys_own_profile_config(clock):
    """Two seats' entries wait in one guard with different ``hold_initial_seconds`` in their own configs."""
    from gateway.run import GatewayRunner

    by_profile = {"fast": Cfg(hold_initial_seconds=15), "slow": Cfg(hold_initial_seconds=60)}
    active = {"p": "fast"}
    runner = make_runner(clock, lambda: by_profile[active["p"]]())

    def _under(source, check):
        previous, active["p"] = active["p"], getattr(source, "profile", None) or "fast"
        try:
            return check()
        finally:
            active["p"] = previous
    runner._under_authorization_profile = _under
    runner._adapter_profile_for_source = lambda source: getattr(source, "profile", None) or "fast"
    held = {}
    for profile, chat in (("fast", "c-fast"), ("slow", "c-slow")):
        for _ in range(10):  # fill_window's count and gap, but each filler carries the seat's profile
            filler = event(BOT_A, chat=chat)
            filler.source.profile = profile
            assert await admit(runner, filler) is True
            clock.advance(1.0)
        held[profile] = event(BOT_A, chat=chat)
        held[profile].source.profile = profile
        assert await admit(runner, held[profile]) is False
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [held["fast"]]  # the slow seat's entry is not due for 60 s
    assert isinstance(runner, GatewayRunner)


# --- T17 ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key,bad", [
    ("hold_after_messages", 0), ("hold_after_messages", "ten"), ("hold_after_messages", 5000), ("hold_after_messages", 2.5),
    ("hold_window_seconds", 5), ("hold_window_seconds", 99999), ("hold_initial_seconds", 0), ("hold_initial_seconds", 601),
    ("hold_runaway_at_cap", 0), ("hold_runaway_at_cap", 101), ("hold_queue_max", 0), ("hold_queue_max", 10001),
])
def test_T17_a_bad_value_gives_the_default_and_one_warning(key, bad, caplog):
    defaults = {"hold_after_messages": 10, "hold_window_seconds": 120, "hold_initial_seconds": 15,
                "hold_max_seconds": 480, "hold_runaway_at_cap": 3, "hold_queue_max": 50}
    caplog.set_level(logging.WARNING)
    for _ in range(3):  # settings are re-read per message: still one warning for this value
        settings = settings_from_config({"gateway": {"bot_loop_guard": {key: bad}}})
        assert getattr(settings, key) == defaults[key]
    assert len([r for r in caplog.records if key in r.getMessage()]) == 1


def test_T17_hold_max_seconds_below_the_initial_value_falls_back_to_the_larger_of_480_and_the_initial():
    s = settings_from_config({"gateway": {"bot_loop_guard": {"hold_initial_seconds": 20, "hold_max_seconds": 10}}})
    assert (s.hold_initial_seconds, s.hold_max_seconds) == (20, 480)
    s = settings_from_config({"gateway": {"bot_loop_guard": {"hold_initial_seconds": 600, "hold_max_seconds": 100}}})
    assert (s.hold_initial_seconds, s.hold_max_seconds) == (600, 600)  # the initial value is kept


# --- T18 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T18_enabled_false_passes_everything_and_hands_over_a_paused_queue_one_per_wake(clock):
    cfg = Cfg(hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15)
    runner = make_runner(clock, cfg)
    held = await _runaway(runner, clock, extra=4)
    await _deliver_one(runner, clock, runner.adapter, 15)
    assert runner._bot_loop_guard.is_paused(conv(runner, held[0]))
    cfg.set(enabled=False, hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15)
    assert await admit(runner, event(BOT_A)) is True  # the guard is off: everything passes
    n = len(handed(runner.adapter))
    await pump(runner, clock)  # ONE wake: the first of the backlog goes
    assert handed(runner.adapter)[n:] == [held[1]]
    for expected in (held[2], held[3]):  # then one at a time, each after the previous turn
        ev = handed(runner.adapter)[-1]
        runner._bot_loop_hold_turn_claimed(ev)
        runner._bot_loop_hold_turn_ended(ev)
        await pump(runner, clock)
        assert handed(runner.adapter)[-1] is expected


# --- T24 [Q4] -------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T24_Q4_past_hold_queue_max_messages_are_still_queued_and_the_notice_reposts_once(clock, caplog):
    caplog.set_level(logging.WARNING)
    runner = make_runner(clock, Cfg(hold_queue_max=2))
    held = await _runaway(runner, clock, extra=2)
    await pump(runner, clock)
    before = len(runner.adapter.notices)
    more = [event(BOT_A) for _ in range(4)]
    for ev in more:
        assert await admit(runner, ev) is False
    await pump(runner, clock)
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == held + more  # nothing dropped
    assert len(runner.adapter.notices) == before + 1  # re-posted once, not once per message
    assert any(THREAD in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


# --- T30 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T30_timer_releases_count_toward_the_window_person_releases_do_not(clock):
    cfg = Cfg(hold_after_messages=3)
    # timer release: counted
    runner = make_runner(clock, cfg)
    await fill_window(runner, clock, 3, gap=0.1)
    held = event(BOT_A)
    assert await admit(runner, held) is False
    await pump(runner, clock, advance=15)  # handed over by the timer at t=15
    ev = handed(runner.adapter)[0]
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)
    clock.advance(106)  # t=121: the first three have left the window, the released one is still in it
    verdicts = [await admit(runner, event(BOT_B)) for _ in range(3)]
    assert verdicts == [True, True, False]
    # person release: not counted
    runner = make_runner(clock, cfg)
    await fill_window(runner, clock, 3, gap=0.1)
    held = event(BOT_A)
    assert await admit(runner, held) is False
    assert await admit(runner, event(PERSON, "go on", is_bot=False)) is True
    await pump(runner, clock)
    ev = handed(runner.adapter)[0]
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)
    verdicts = [await admit(runner, event(BOT_B)) for _ in range(4)]
    assert verdicts == [True, True, True, False]


# --- T41 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T41_raising_hold_after_messages_during_a_pause_keeps_the_queue_and_enabled_false_delivers_it(clock):
    base = dict(hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15)
    cfg = Cfg(**base)
    runner = make_runner(clock, cfg)
    held = await _runaway(runner, clock, extra=3)
    await _deliver_one(runner, clock, runner.adapter, 15)
    key = conv(runner, held[0])
    assert runner._bot_loop_guard.is_paused(key)
    cfg.set(hold_after_messages=1000, **base)
    later = event(BOT_A)
    assert await admit(runner, later) is False  # a conversation already holding keeps holding (queue waiting)
    assert runner._bot_loop_guard.is_paused(key) and runner._bot_loop_guard.held_events(key) == held[1:] + [later]
    await pump(runner, clock, advance=3600)
    assert len(handed(runner.adapter)) == 1  # still paused
    cfg.set(enabled=False, **base)
    await pump(runner, clock)
    assert handed(runner.adapter)[1:] == [held[1]]  # delivered on the next wake


# --- T44 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T44_a_held_messages_ask_survives_the_idle_sweep_past_cooldown_seconds(clock):
    runner = make_runner(clock, Cfg(cooldown_seconds=60, hold_initial_seconds=600, hold_max_seconds=600))
    await fill_window(runner, clock, 10)
    held = event(BOT_A, "please summarise the thread")
    assert await admit(runner, held) is False
    clock.advance(200)  # past cooldown_seconds and the window: every idle sweep has run
    await admit(runner, event(BOT_B, "unrelated"))
    again = event(BOT_A, "please summarise the thread")
    assert await admit(runner, again) is False
    assert again not in runner._bot_loop_guard.held_events(conv(runner, held))  # dropped as duplicate, not queued
    assert runner._bot_loop_guard.held_events(conv(runner, held))[0] is held


# --- T50 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T50_discord_asks_keep_the_last_50_and_a_held_ask_is_never_evicted(clock):
    cfg = Cfg(hold_after_messages=1000, max_events=100000)
    runner = make_runner(clock, cfg)
    asks = [event(BOT_A, f"distinct ask {n}") for n in range(60)]
    for ev in asks:
        assert await admit(runner, ev) is True
        clock.advance(0.5)
    assert await admit(runner, event(BOT_A, "distinct ask 3")) is True  # evicted: a repeat is admitted
    assert await admit(runner, event(BOT_A, "distinct ask 55")) is False  # kept: still a duplicate
    # a held ask is never evicted
    runner = make_runner(clock, Cfg(hold_after_messages=5, max_events=100000))
    await fill_window(runner, clock, 5, gap=0.01)
    first_held = event(BOT_A, "held ask")
    assert await admit(runner, first_held) is False
    for n in range(60):
        await admit(runner, event(BOT_A, f"queued ask {n}"))
    repeat = event(BOT_A, "held ask")
    assert await admit(runner, repeat) is False
    assert repeat not in runner._bot_loop_guard.held_events(conv(runner, first_held))  # dropped as duplicate
