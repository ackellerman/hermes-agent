"""Loop-guard-hold, pacing and hand-over (spec r15 S2, S4, S8, S13): one at a time, never over a running turn, through
the base adapter, on every path. Names come from tests/gateway/lgh_support.py's contract.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.session_ingress import executing_admission

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, BOT_C, PERSON, THREAD, Cfg, admit, clock, conv, end_turn, event, fill_window, handed,
    make_adapter, make_runner, pump, start_turn,
)


async def _hold_one(runner, clock, text=None, *, user=BOT_A, chat=THREAD):
    await fill_window(runner, clock, 10, gap=0.1, chat=chat)
    ev = event(user, text, chat=chat)
    assert await admit(runner, ev) is False
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == [ev]
    return ev


def _finish(runner, ev):
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)


# --- T13 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T13_a_released_message_on_the_busy_path_is_not_judged_again(clock):
    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [ev]
    # The adapter's handle_message meets a running turn and sends it down the busy path, which judges bot
    # messages (run_busy.py:968). The window is full; the message was released, so no rule judges it again.
    assert runner._admit_bot_message_for_source(ev.source) is True
    assert runner._admit_bot_chain(ev) is True
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == []
    # control: an unreleased message through the same path is held
    other = event(BOT_B)
    assert runner._admit_bot_chain(other) is False
    assert runner._bot_loop_guard.held_events(conv(runner, other)) == [other]


# --- T14 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("attr,on,off", [
    ("_draining", True, False),
    ("_startup_restore_in_progress", True, False),
    ("_hm_estop_paused", lambda: True, lambda: False),
])
async def test_T14_nothing_is_handed_over_during_drain_estop_pause_or_startup_restore(clock, attr, on, off):
    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    setattr(runner, attr, on)
    await pump(runner, clock, advance=60)
    await pump(runner, clock, advance=60)
    assert handed(runner.adapter) == []
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == [ev]  # the entry stays
    setattr(runner, attr, off)
    await pump(runner, clock)
    assert handed(runner.adapter) == [ev]


# --- T20, T42 -------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T20_a_durable_reentry_runs_and_counts_once(clock):
    runner = make_runner(clock, Cfg(hold_after_messages=4))
    first = event(BOT_A, "queued while the window had room")
    assert await admit(runner, first) is True  # counted at arrival: 1
    for _ in range(2):
        assert await admit(runner, event(BOT_B)) is True  # 3 in the window
    token = executing_admission.set(True)
    try:
        assert await runner._hm_admit_event(first) is not None  # executed from the FIFO: not held, not counted again
    finally:
        executing_admission.reset(token)
    assert await admit(runner, event(BOT_B)) is True  # the 4th: had the re-entry counted, this would be the 5th
    full = event(BOT_B)
    assert await admit(runner, full) is False  # and now the window is full: held, not dropped
    assert runner._bot_loop_guard.held_events(conv(runner, full)) == [full]


@pytest.mark.asyncio
async def test_T42_a_bot_message_reentering_under_executing_admission_is_never_held(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10)  # the window is full
    ev = event(BOT_A)
    token = executing_admission.set(True)
    try:
        assert await runner._hm_admit_event(ev) is not None
    finally:
        executing_admission.reset(token)
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == []


# --- T21 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T21_on_an_interrupt_seat_a_due_message_waits_for_the_running_turn_and_does_not_interrupt_it(clock):
    runner = make_runner(clock, Cfg())
    runner._busy_input_mode = "interrupt"
    ev = await _hold_one(runner, clock)
    key = start_turn(runner, ev)
    agent = MagicMock()
    runner._session_state(key).turn.agent = agent
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=120)
    assert handed(runner.adapter) == []
    agent.interrupt.assert_not_called()
    end_turn(runner, key)
    await pump(runner, clock)
    assert handed(runner.adapter) == [ev]


# --- T22 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T22_a_batched_bot_message_is_held_and_released_as_one_event_with_both_chunks(clock):
    from gateway.platforms.base import MessageEvent
    from plugins.platforms.discord.adapter import DiscordAdapter

    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10)
    seen = []

    async def _runner_side(merged):  # what the adapter's flush hands the runner: the merged event
        seen.append(merged)
        await runner._hm_admit_event(merged)
    discord_adapter = object.__new__(DiscordAdapter)
    discord_adapter.platform = SimpleNamespace(value="discord")
    discord_adapter.config = SimpleNamespace(extra={})
    discord_adapter._pending_text_batches = {}
    discord_adapter._pending_text_batch_tasks = {}
    discord_adapter._text_batch_delay_seconds = 0.01
    discord_adapter._text_batch_split_delay_seconds = 0.01
    discord_adapter.handle_message = _runner_side
    first, second = event(BOT_A, "chunk one"), event(BOT_A, "chunk two")
    discord_adapter._enqueue_text_event(first)
    discord_adapter._enqueue_text_event(second)
    for _ in range(200):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert len(seen) == 1 and isinstance(seen[0], MessageEvent)  # one merged event, both chunks in it
    assert "chunk one" in seen[0].text and "chunk two" in seen[0].text
    assert runner._bot_loop_guard.held_events(conv(runner, seen[0])) == [seen[0]]
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [seen[0]]


# --- T23 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T23_an_entry_whose_adapter_is_gone_stays_queued_and_goes_when_it_is_back(clock, caplog):
    caplog.set_level(logging.WARNING)
    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    adapter = runner.adapter
    runner._adapter_for_source = lambda source: None
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=60)
    assert adapter.handle_message.await_count == 0
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == [ev]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1  # once per episode
    runner._adapter_for_source = lambda source: adapter
    await pump(runner, clock)
    assert handed(adapter) == [ev]


# --- T31 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T31_timer_and_person_release_racing_hand_each_entry_over_once(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    held = [event(BOT_A) for _ in range(3)]
    for ev in held:
        assert await admit(runner, ev) is False
    clock.advance(15)  # the head is due AND a person writes in the same instant
    person = event(PERSON, "stop arguing", is_bot=False)
    await asyncio.gather(pump(runner, clock), admit(runner, person), pump(runner, clock), pump(runner, clock))
    for _ in range(4):
        got = handed(runner.adapter)
        if got:
            _finish(runner, got[-1])
        await pump(runner, clock)
    got = handed(runner.adapter)
    assert got == held  # each exactly once, FIFO
    assert len(set(map(id, got))) == 3


# --- T32 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T32_a_released_message_is_not_judged_again_by_duplicate_or_reply_to_reply(clock):
    runner = make_runner(clock, Cfg(bot_replies_expect_reply=False))
    await fill_window(runner, clock, 10, gap=0.1)
    ev = event(BOT_A, "the same words")  # its ask is recorded at arrival
    assert await admit(runner, ev) is False
    runner._note_bot_reply(ev.source)  # our reply to this sender goes out while it waits
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [ev]
    seat = runner._adapter_profile_for_source(ev.source)
    assert runner._bot_loop_guard.is_released_event(seat, ev) is True
    assert await runner._hm_admit_event(ev) is not None  # duplicate of its own ask / answer to our reply: not dropped
    # the pre_gateway_dispatch hook may hand back a rewritten copy: still recognised
    copy = dataclasses.replace(ev, text=ev.text + " (rewritten)")
    assert runner._bot_loop_guard.is_released_event(seat, copy) is True
    assert await runner._hm_admit_event(copy) is not None


# --- T33 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T33_a_due_message_waits_for_a_running_turn_under_its_own_key_and_the_keys_agree(clock):
    from gateway.session import build_session_key

    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    key = runner._session_key_for_source(ev.source)
    assert key == build_session_key(ev.source)  # the runner's key and the adapter's key function agree
    runner._session_state(key).turn.agent = object()  # a queued FIFO turn of that conversation is running
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=30)
    assert handed(runner.adapter) == []
    # the adapter's own guard (a turn it installed but the runner has not claimed yet) also holds it back
    runner._session_state(key).turn.agent = None
    runner.adapter._active_sessions[key] = object()
    await pump(runner, clock, advance=30)
    assert handed(runner.adapter) == []
    runner.adapter._active_sessions.clear()
    await pump(runner, clock)
    assert handed(runner.adapter) == [ev]


@pytest.mark.asyncio
async def test_T33_in_a_plain_channel_two_senders_held_messages_go_out_one_at_a_time(clock):
    runner = make_runner(clock, Cfg())
    chat = "plain-channel-1"
    await fill_window(runner, clock, 10, gap=0.1, chat=chat)
    a = event(BOT_A, chat=chat)
    b = event(BOT_B, chat=chat)
    for ev in (a, b):
        ev.source.chat_type = "group"
        assert await admit(runner, ev) is False
    assert runner._session_key_for_source(a.source) != runner._session_key_for_source(b.source)  # per-sender slots
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=60)
    assert handed(runner.adapter) == [a]  # b waits although its own session key is idle
    _finish(runner, a)
    await pump(runner, clock)
    assert handed(runner.adapter) == [a, b]


# --- T34 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T34_a_hand_over_that_raises_is_retried_three_times_then_named_in_a_lost_line(clock, caplog):
    caplog.set_level(logging.WARNING)
    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    runner.adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
    await pump(runner, clock, advance=15)
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == [ev]  # back at the head
    for _ in range(3):
        await pump(runner, clock, advance=1)  # the next wake retries
        await pump(runner, clock, advance=60)
    assert runner.adapter.handle_message.await_count == 3  # at most 3 hand-overs
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == []
    assert any(ev.message_id in text for _chat, text in runner.adapter.notices)  # named in a lost line (S8 form)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_T34_a_debounced_hand_over_that_returns_normally_is_not_handed_over_again(clock):
    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    runner.adapter.handle_message = AsyncMock(return_value=None)  # a queue seat debounces: returns, no turn yet
    await pump(runner, clock, advance=15)
    for _ in range(5):
        await pump(runner, clock, advance=70)
    assert runner.adapter.handle_message.await_count == 1
    assert runner._bot_loop_guard.held_events(conv(runner, ev)) == []


# --- T38 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T38_a_released_message_carries_no_new_metadata_so_native_admission_accepts_it(clock):
    from gateway.session_envelope import _native_metadata

    runner = make_runner(clock, Cfg())
    ev = await _hold_one(runner, clock)
    before = dict(ev.metadata or {})
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [ev]
    assert dict(ev.metadata or {}) == before == {}  # the mark is in the guard's released set (S2), not on the event
    assert _native_metadata(ev) == {} and not ev.metadata  # no metadata: native admission has nothing to refuse
    seat = runner._adapter_profile_for_source(ev.source)
    assert runner._bot_loop_guard.is_released_event(seat, ev) is True


# --- T39 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T39_the_second_due_message_waits_until_the_first_turn_has_started_and_ended(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    first, second = event(BOT_A), event(BOT_B)
    assert await admit(runner, first) is False and await admit(runner, second) is False
    await pump(runner, clock, advance=15)
    assert handed(runner.adapter) == [first]
    await pump(runner, clock, advance=40)  # the second is due (30 s) but the runner has not claimed the slot
    assert handed(runner.adapter) == [first]
    runner._bot_loop_hold_turn_claimed(first)
    key = start_turn(runner, first)
    await pump(runner, clock)
    assert handed(runner.adapter) == [first]  # claimed and running
    end_turn(runner, key)
    runner._bot_loop_hold_turn_ended(first)
    await pump(runner, clock)
    assert handed(runner.adapter) == [first, second]


@pytest.mark.asyncio
async def test_T39_an_unclaimed_turn_record_closes_after_60s_with_a_warning(clock, caplog):
    caplog.set_level(logging.WARNING)
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    first, second = event(BOT_A), event(BOT_B)
    assert await admit(runner, first) is False and await admit(runner, second) is False
    await pump(runner, clock, advance=15)
    await pump(runner, clock, advance=59)
    assert handed(runner.adapter) == [first]
    await pump(runner, clock, advance=2)
    assert handed(runner.adapter) == [first, second]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# --- T46 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T46_two_seats_holding_the_same_message_keep_separate_released_marks(clock):
    runner = make_runner(clock, Cfg())
    runner._adapter_profile_for_source = lambda source: getattr(source, "profile", None) or ""
    copies = {}
    for seat in ("boba", "tars"):
        for _ in range(10):  # each seat's own window fills (the conversation key carries the seat's profile)
            filler = event(BOT_C)
            filler.source.profile = seat
            assert await admit(runner, filler) is True
        ev = event(BOT_A, message_id="1573000000000000001")  # the same Discord message at both seats
        ev.source.profile = seat
        copies[seat] = ev
        assert await admit(runner, ev) is False
    assert [e.source.profile for e in runner._bot_loop_guard.held_events(conv(runner, copies["boba"]))] == ["boba"]
    await pump(runner, clock, advance=15)
    got = handed(runner.adapter)
    assert set(map(id, got)) == set(map(id, copies.values()))  # both seats' copies go out (one per wake each)
    guard = runner._bot_loop_guard
    assert guard.is_released_event("boba", copies["boba"]) and guard.is_released_event("tars", copies["tars"])
    runner._bot_loop_hold_turn_claimed(copies["boba"])  # seat A's turn starts: only A's mark goes
    runner._bot_loop_hold_turn_ended(copies["boba"])
    assert guard.is_released_event("tars", copies["tars"]) is True
    assert await runner._hm_admit_event(copies["tars"]) is not None  # handed over once, not held again
    assert guard.held_events(conv(runner, copies["tars"])) == []


# --- T48 ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T48_a_bot_message_arriving_while_a_person_released_queue_drains_is_held_behind_it(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    held = [event(BOT_A) for _ in range(3)]
    for ev in held:
        assert await admit(runner, ev) is False
    assert await admit(runner, event(PERSON, "carry on", is_bot=False)) is True  # clears the window, releases all
    await pump(runner, clock)  # the first of the three goes; two still draining
    assert handed(runner.adapter) == [held[0]]
    await fill_window(runner, clock, 10, gap=0.1)  # a bot exchange continues: fills the window, past it
    late = event(BOT_B)
    assert await admit(runner, late) is False
    assert runner._bot_loop_guard.held_events(conv(runner, late))[-1] is late  # behind the draining entries
    assert runner._bot_loop_guard.held_events(conv(runner, late))[:2] == held[1:]
