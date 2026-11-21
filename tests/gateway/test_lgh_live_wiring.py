"""Loop-guard-hold, the live wiring (spec r15 S4, S8, S13): the real GatewayRunner starts and ends turns, stops,
starts and runs its pump task. No test here calls ``_bot_loop_hold_turn_claimed`` / ``_ended`` (the runner's own slot
claim and release must), nor ``_bot_loop_hold_flush_refs`` / ``_bot_loop_hold_report_lost`` (``stop()`` and
``start()`` must) except to leave a refs file for the NEXT start to read. ``_bot_loop_hold_pump_once`` is the one
manual wake, used only where the pump task is not the subject.

The turn itself is a stand-in (``_handle_message_with_agent``); everything around it, the slot claim and the slot
release in ``_handle_message``, is the runner's own. Names come from tests/gateway/lgh_support.py's contract.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.bot_loop_guard import BotLoopGuard
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session_ingress import executing_admission

from tests.gateway.lgh_support import BOT_A, BOT_B, THREAD, Cfg, admit, clock, conv, event, fill_window, handed
from tests.waiting import async_wait_until


class StubDiscord(BasePlatformAdapter):
    """A connected Discord adapter whose hand-overs and hold notices are recorded."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.DISCORD)
        self.notices = []
        self.handle_message = AsyncMock()

    async def connect(self, *, is_reconnect=False):
        self._mark_connected()
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_hold_notice(self, chat_id, text, *args, **kwargs):
        self.notices.append((str(chat_id), text))
        return SendResult(success=True, message_id=str(1572100000000000000 + len(self.notices)))


def _runner(monkeypatch, tmp_path, clock, cfg, *, adapter=None, platforms=False):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_STARTUP_WARMUP_TIMEOUT", "0")
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="t")} if platforms else {},
        sessions_dir=tmp_path / "sessions")
    runner = GatewayRunner(config)
    runner._bot_loop_guard = BotLoopGuard(settings=cfg, clock=clock.now)
    runner._scale_to_zero_note_real_inbound = lambda: None
    return runner


def _wire(runner, adapter, *, authorized=True):
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner._adapter_profile_for_source = lambda source: "boba"
    runner._adapter_for_source = lambda source: adapter


def _stand_in_turns(runner, monkeypatch):
    """The turn waits on a per-event gate; ``started`` lists the events whose turn began."""
    started, gates = [], {}

    async def turn(event, source, quick_key, run_generation):
        started.append(event)
        await gates.setdefault(id(event), asyncio.Event()).wait()
        return "ok"
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    return started, gates


def _hand_over_by_running_the_turn(runner, adapter, tasks):
    """What the base adapter does with a handed-over event: the runner's own message handler runs it."""
    async def _hand(ev):
        tasks.append(asyncio.ensure_future(runner._handle_message(ev)))
    adapter.handle_message = AsyncMock(side_effect=_hand)


async def _held_pair(runner, clock):
    await fill_window(runner, clock, 10, gap=0.1)
    first, second = event(BOT_A), event(BOT_B)
    assert await admit(runner, first) is False and await admit(runner, second) is False
    assert runner._bot_loop_guard.held_events(conv(runner, first)) == [first, second]
    return first, second


# --- S4: the runner's own slot claim and release open and close the turn record -------------------------------


@pytest.mark.asyncio
async def test_S4_the_runners_own_turn_start_and_end_claim_and_release_the_record(monkeypatch, tmp_path, clock):
    runner = _runner(monkeypatch, tmp_path, clock, Cfg())
    adapter = StubDiscord()
    _wire(runner, adapter)
    tasks = []
    _hand_over_by_running_the_turn(runner, adapter, tasks)
    started, gates = _stand_in_turns(runner, monkeypatch)
    first, second = await _held_pair(runner, clock)  # the head is due at 15 s, the second 30 s after it
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()
    await async_wait_until(lambda: started == [first], what="the first hand-over to start its turn")
    clock.advance(31)  # the second is due; the first turn still runs
    await runner._bot_loop_hold_pump_once()
    assert handed(adapter) == [first]
    gates[id(first)].set()
    await tasks[0]  # the runner released the slot and told the guard, 46 s after the hand-over (< the 60 s fallback)
    await runner._bot_loop_hold_pump_once()
    assert handed(adapter) == [first, second]


@pytest.mark.asyncio
async def test_S4_claiming_the_slot_for_a_different_event_of_the_conversation_does_not_close_the_record(
        monkeypatch, tmp_path, clock):
    runner = _runner(monkeypatch, tmp_path, clock, Cfg())
    adapter = StubDiscord()
    _wire(runner, adapter)
    started, gates = _stand_in_turns(runner, monkeypatch)
    first, second = await _held_pair(runner, clock)
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()  # the first is handed over; its turn never starts (a refusal inside the adapter)
    assert handed(adapter) == [first]
    other = event(BOT_B, "an unrelated turn of this conversation")  # a different event, same conversation, same session key
    gates[id(other)] = asyncio.Event()
    gates[id(other)].set()
    token = executing_admission.set(True)  # judged at arrival already: not held again
    try:
        await runner._handle_message(other)  # claims and releases the slot, for ``other``
    finally:
        executing_admission.reset(token)
    assert started == [other]
    clock.advance(31)  # t = 46 s after the hand-over: the second is due, the record bound to ``first`` is still open
    await runner._bot_loop_hold_pump_once()
    assert handed(adapter) == [first]
    clock.advance(20)  # 66 s: the unclaimed record closes (WARNING) and the second goes
    await runner._bot_loop_hold_pump_once()
    assert handed(adapter) == [first, second]


# --- S8, S13: stop, start and the pump task -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_S8_a_graceful_stop_flushes_the_refs_file_and_lists_exactly_what_is_still_held(
        monkeypatch, tmp_path, clock):
    runner = _runner(monkeypatch, tmp_path, clock, Cfg())
    adapter = StubDiscord()
    _wire(runner, adapter)
    held = await _held_pair(runner, clock)
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()  # the first is handed over and accepted
    assert handed(adapter) == [held[0]]
    await asyncio.wait_for(runner.stop(), 60)
    path = tmp_path / "gateway" / "bot_loop_hold_refs.json"
    assert path.exists()
    raw = path.read_text()
    json.loads(raw)
    assert held[1].message_id in raw and held[0].message_id not in raw
    assert held[1].text not in raw


@pytest.mark.asyncio
async def test_S8_the_real_startup_reports_what_the_last_run_lost_after_adapters_connect(monkeypatch, tmp_path, clock):
    first_run = _runner(monkeypatch, tmp_path, clock, Cfg())
    _wire(first_run, StubDiscord())
    lost = await _held_pair(first_run, clock)
    first_run._bot_loop_hold_flush_refs()  # the file a crashed run would have left (written, coalesced, within a second)
    second_run = _runner(monkeypatch, tmp_path, clock, Cfg(), platforms=True)
    stub = StubDiscord()
    monkeypatch.setattr(second_run, "_create_adapter", lambda platform, cfg: stub)
    assert await asyncio.wait_for(second_run.start(), 60) is True
    try:
        await async_wait_until(lambda: stub.notices, what="the startup line naming the lost messages")
        said = " ".join(text for _chat, text in stub.notices)
        assert all(ev.message_id in said for ev in lost)
        assert Platform.DISCORD in second_run.adapters  # said after the adapter connected
    finally:
        await asyncio.wait_for(second_run.stop(), 60)


@pytest.mark.asyncio
async def test_S13_the_pump_task_the_runner_starts_hands_the_head_over_without_a_manual_wake(monkeypatch, tmp_path, clock):
    runner = _runner(monkeypatch, tmp_path, clock, Cfg(), platforms=True)
    stub = StubDiscord()
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: stub)
    assert await asyncio.wait_for(runner.start(), 60) is True
    try:
        _wire(runner, stub)
        first, _second = await _held_pair(runner, clock)
        clock.advance(15)  # due; the pump's own wake (at most its 5 s recheck) finds it
        await async_wait_until(lambda: handed(stub), timeout=12, what="the pump task to hand the head over")
        assert handed(stub) == [first]
    finally:
        await asyncio.wait_for(runner.stop(), 60)
