"""Loop-guard-hold, step-2 pins (reviewers a, b and c of step 2): rules the RED files leave unpinned, plus the bugs the
step-2 review found. A new file: the RED files stay byte-identical.

B1  the restart refs file is per SEAT (spec S8): each seat's own ``<its profile home>/gateway/bot_loop_hold_refs.json``
    under ``multiplex_profiles``; the 500-entry cap and "a missing profile home is never re-created" apply per seat.
N1  the runner's slot claim is the first statement INSIDE the turn's ``try``: an exception in the setup calls that follow
    can no longer leave a claimed record open for 10 minutes.
b1-4  four rules of S1, S5, S7 and S10 that a wrong build could drop without failing any RED test.
c1  the second ``human_relays`` bot-branch drop site (``bot-inline-mention-required``) reports the person and releases.
r2  six more rules of S2, S4, S6 and S7 the re-review found unpinned: a /stop stops the retry of an in-flight hand-over,
    the /stop rule strips ALL leading mentions, the notice recognition is narrow (channel, author, text, id), the notice
    flags clear at the end of an episode, the turn record binds by message id for a hook-rewritten copy, and
    ``_report_human_seen`` reports once per message id.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import logging
import os
import stat
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.bot_loop_guard import BotLoopGuard

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, PERSON, RELAY, THREAD, Cfg, FakeClock, admit, clock, conv, event, fill_window, handed, make_runner,
    pump,
)
from tests.gateway.test_lgh_human_release import (
    CH, _Chan, _discord_env, _hold, _msg, _released, _runner as _release_runner, _seat,
)
from tests.gateway.test_lgh_notice_restart import (
    SELF_ID, _bot_msg, _Chan as _NoticeChan, _real_adapter,
)
from tests.gateway.test_lgh_live_wiring import (
    StubDiscord, _hand_over_by_running_the_turn, _held_pair, _runner as _wiring_runner, _stand_in_turns, _wire,
)

REFS = "bot_loop_hold_refs.json"
_MIDS = itertools.count(1573100000000000000)


def _seat_event(seat, user, text=None, *, chat=THREAD, message_id=None):
    ev = event(user, text, chat=chat, message_id=message_id or str(next(_MIDS)), name=f"Peer{user[-3:]}")
    ev.source.profile = seat
    return ev


async def _fill_seat(runner, seat, chat=THREAD):
    for _ in range(10):
        assert await admit(runner, _seat_event(seat, BOT_B, chat=chat)) is True


async def _hold_seat(runner, seat, n, *, chat=THREAD):
    await _fill_seat(runner, seat, chat)
    held = [_seat_event(seat, BOT_A, chat=chat) for _ in range(n)]
    for ev in held:
        assert await admit(runner, ev) is False
    return held


def _multiseat(clock, tmp_path, monkeypatch, *, create=("boba", "tars"), **block):
    """One gateway process serving two seats, each with its own profile home (``multiplex_profiles``)."""
    main = tmp_path / "main"
    homes = {name: tmp_path / "profiles" / name for name in ("boba", "tars")}
    main.mkdir()
    for name in create:
        homes[name].mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(main))
    runner = make_runner(clock, Cfg(**block))
    runner.config = SimpleNamespace(multiplex_profiles=True, _runtime_profile_homes=list(homes.items()))
    runner._adapter_profile_for_source = lambda source: getattr(source, "profile", None) or ""
    runner._under_authorization_profile = lambda source, check: check()
    return runner, main, homes


def _refs(home):
    return home / "gateway" / REFS


def _all_refs_files(tmp_path):
    return sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob(REFS))


# --- B1: the refs file is per seat -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_B1_each_seat_keeps_its_own_refs_file_under_its_own_profile_home_and_no_shared_file_exists(
        clock, monkeypatch, tmp_path):
    runner, main, homes = _multiseat(clock, tmp_path, monkeypatch)
    boba = await _hold_seat(runner, "boba", 2)
    tars = await _hold_seat(runner, "tars", 1)
    runner._bot_loop_hold_flush_refs()
    assert _all_refs_files(tmp_path) == ["profiles/boba/gateway/" + REFS, "profiles/tars/gateway/" + REFS]
    assert not _refs(main).exists()  # no process-wide file
    raw_boba, raw_tars = _refs(homes["boba"]).read_text(), _refs(homes["tars"]).read_text()
    assert all(ev.message_id in raw_boba for ev in boba) and not any(ev.message_id in raw_boba for ev in tars)
    assert all(ev.message_id in raw_tars for ev in tars) and not any(ev.message_id in raw_tars for ev in boba)
    assert not any(ev.text in raw_boba + raw_tars for ev in boba + tars)  # no message text
    for home in homes.values():
        assert stat.S_IMODE(os.stat(_refs(home)).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(home / "gateway").st_mode) == 0o700


def _write_seat_file(home, entries, not_listed=0):
    """What a seat's previous run left in its own profile home (the fields S8 lists; never message text)."""
    (home / "gateway").mkdir(mode=0o700, exist_ok=True)
    _refs(home).write_text(json.dumps({"version": 1, "not_listed": not_listed, "entries": entries}))


def _lost_entry(seat, message_id, sender="Peer"):
    return {"ref": f"r{message_id}", "profile": seat, "channel": THREAD, "message": message_id, "sender_id": BOT_A,
            "sender_name": sender, "guild": "", "arrived": 1_700_000_000.0}


@pytest.mark.asyncio
async def test_B1_a_restart_reads_each_seats_own_file_names_its_lost_messages_once_and_empties_it(
        clock, monkeypatch, tmp_path):
    _runner, _main, homes = _multiseat(clock, tmp_path, monkeypatch)
    _write_seat_file(homes["boba"], [_lost_entry("boba", "1573200000000000001"), _lost_entry("boba", "1573200000000000002")])
    _write_seat_file(homes["tars"], [_lost_entry("tars", "1573200000000000003")])
    after, _main2, _homes2 = _multiseat_again(clock, tmp_path, monkeypatch)
    await after._bot_loop_hold_report_lost()
    said = [text for _chat, text in after.adapter.notices]
    for mid in ("1573200000000000001", "1573200000000000002", "1573200000000000003"):
        assert sum(mid in text for text in said) == 1  # each lost message is named exactly once
    for home in homes.values():  # every seat's file is emptied once its line was sent
        raw = _refs(home).read_text() if _refs(home).exists() else ""
        assert "15732000000000000" not in raw


def _multiseat_again(clock, tmp_path, monkeypatch):
    """A second process start over the same homes (the first start's dirs and files stay)."""
    homes = {name: tmp_path / "profiles" / name for name in ("boba", "tars")}
    runner = make_runner(clock, Cfg())
    runner.config = SimpleNamespace(multiplex_profiles=True, _runtime_profile_homes=list(homes.items()))
    runner._adapter_profile_for_source = lambda source: getattr(source, "profile", None) or ""
    runner._under_authorization_profile = lambda source, check: check()
    return runner, tmp_path / "main", homes


@pytest.mark.asyncio
async def test_B1_the_500_entry_cap_and_its_count_are_per_seat(clock, monkeypatch, tmp_path):
    runner, _main, homes = _multiseat(clock, tmp_path, monkeypatch, hold_queue_max=10000)
    boba = await _hold_seat(runner, "boba", 501)
    tars = await _hold_seat(runner, "tars", 1)
    runner._bot_loop_hold_flush_refs()
    raw_boba, raw_tars = _refs(homes["boba"]).read_text(), _refs(homes["tars"]).read_text()
    assert sum(ev.message_id in raw_boba for ev in boba) == 500  # the oldest is dropped from the list, per seat
    assert boba[0].message_id not in raw_boba and boba[-1].message_id in raw_boba
    assert tars[0].message_id in raw_tars  # the other seat's one entry is not squeezed out by boba's 501
    after, _main, _homes = _multiseat_again(clock, tmp_path, monkeypatch)
    await after._bot_loop_hold_report_lost()
    said = " ".join(text for _chat, text in after.adapter.notices)
    assert "1 more not listed" in said  # boba's count, not a count shared with tars
    assert tars[0].message_id in said


@pytest.mark.asyncio
async def test_B1_a_missing_seat_home_is_never_re_created_and_does_not_stop_the_other_seats_file(
        clock, monkeypatch, tmp_path):
    runner, main, homes = _multiseat(clock, tmp_path, monkeypatch, create=("boba",))
    boba = await _hold_seat(runner, "boba", 1)
    tars = await _hold_seat(runner, "tars", 1)
    runner._bot_loop_hold_flush_refs()
    assert not homes["tars"].exists()  # never re-created
    assert not (tmp_path / "profiles" / "tars").exists()
    assert boba[0].message_id in _refs(homes["boba"]).read_text()
    assert not _refs(main).exists()
    assert tars  # held, in memory, just not listed anywhere


# --- N1: the claim is inside the turn's try -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_N1_an_exception_in_the_turn_setup_after_the_claim_does_not_leave_a_claimed_record_open(
        monkeypatch, tmp_path, clock):
    import hermes_cli.observability.shared_metrics_gateway as metrics

    runner = _wiring_runner(monkeypatch, tmp_path, clock, Cfg())
    adapter = StubDiscord()
    _wire(runner, adapter)
    tasks = []
    _hand_over_by_running_the_turn(runner, adapter, tasks)
    _stand_in_turns(runner, monkeypatch)
    first, second = await _held_pair(runner, clock)

    def _boom(*args, **kwargs):
        raise RuntimeError("reply clock failed")
    monkeypatch.setattr(metrics, "start_reply_clock", _boom)
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()  # the first is handed over; its turn raises in the setup
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert handed(adapter) == [first] and isinstance(results[0], RuntimeError)
    key = runner._session_key_for_source(first.source)
    runner._session_state(key).turn.agent = None  # the setup's own slot leak is not what this test pins
    clock.advance(121)  # past the second's wait; a CLAIMED record would stay open for 10 minutes
    await runner._bot_loop_hold_pump_once()
    assert handed(adapter) == [first, second]


# --- b1-b4: rules of S1, S5, S7, S10 ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b1_a_third_bots_message_is_held_behind_a_waiting_timer_entry_although_the_window_has_room(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)  # t ~ 1
    held = [event(BOT_A) for _ in range(4)]
    for ev in held:
        assert await admit(runner, ev) is False
    for wait in (16, 30, 60):  # three timer releases, at about 16, 46 and 106 s
        clock.advance(wait)
        await runner._bot_loop_hold_pump_once()
        ev = handed(runner.adapter)[-1]
        runner._bot_loop_hold_turn_claimed(ev)
        runner._bot_loop_hold_turn_ended(ev)
    assert handed(runner.adapter) == held[:3]
    guard = runner._bot_loop_guard
    assert guard.held_events(conv(runner, held[3])) == [held[3]]  # one timer entry still waits
    clock.advance(24)  # t ~ 130: only three messages (the releases) are inside the 120 s window, 7 below the limit
    third = event(BOT_B, "from a third bot")
    assert await admit(runner, third) is False  # not admitted and counted: it queues behind the waiting entry
    assert guard.held_events(conv(runner, third)) == [held[3], third]


@pytest.mark.asyncio
async def test_b2_a_paused_exchange_still_holds_new_bot_messages_after_its_queue_emptied(clock):
    runner = make_runner(clock, Cfg(hold_initial_seconds=15, hold_max_seconds=15, hold_runaway_at_cap=3))
    await fill_window(runner, clock, 10, gap=0.1)
    held = [event(BOT_A) for _ in range(3)]
    for ev in held:
        assert await admit(runner, ev) is False
    for _ in range(3):  # three full-length waits in a row
        clock.advance(15)
        await runner._bot_loop_hold_pump_once()
        ev = handed(runner.adapter)[-1]
        runner._bot_loop_hold_turn_claimed(ev)
        runner._bot_loop_hold_turn_ended(ev)
    guard = runner._bot_loop_guard
    key = conv(runner, held[0])
    assert handed(runner.adapter) == held and guard.is_paused(key) and guard.held_events(key) == []
    clock.advance(400)  # the window is empty again: only the pause can hold the next message
    late = event(BOT_B, "late bot message")
    assert await admit(runner, late) is False
    assert guard.held_events(key) == [late]
    clock.advance(3600)
    await runner._bot_loop_hold_pump_once()
    assert handed(runner.adapter) == held  # still paused: nothing delivered until a person writes


@pytest.mark.asyncio
async def test_b3_a_persons_message_clears_the_asks_so_a_repeat_is_admitted_again(clock):
    runner = make_runner(clock, Cfg())
    first = event(BOT_A, "please summarise the thread")
    assert await admit(runner, first) is True
    assert await admit(runner, event(BOT_A, "please summarise the thread")) is False  # control: a repeat is dropped
    assert await admit(runner, event(PERSON, "go on", is_bot=False)) is True
    assert await admit(runner, event(BOT_A, "please summarise the thread")) is True  # asks cleared by the person


@pytest.mark.asyncio
async def test_b3_a_persons_message_clears_replied_to_so_an_answer_to_our_reply_is_admitted_again(clock):
    runner = make_runner(clock, Cfg(bot_replies_expect_reply=False))
    asked = event(BOT_A, "a question")
    assert await admit(runner, asked) is True
    runner._note_bot_reply(asked.source)  # our reply to BOT_A went out
    assert await admit(runner, event(BOT_A, "thanks for the answer")) is False  # control: answering our reply is dropped
    assert await admit(runner, event(PERSON, "carry on", is_bot=False)) is True
    assert await admit(runner, event(BOT_A, "a new thought")) is True  # replied_to cleared by the person


@pytest.mark.asyncio
async def test_b4_a_persons_message_resets_the_backoff_level_so_the_next_hold_waits_15_seconds_again(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    held = [event(BOT_A) for _ in range(3)]
    for ev in held:
        assert await admit(runner, ev) is False
    for wait in (16, 30):  # two timer releases: the level is now 3 (a 60 s wait)
        clock.advance(wait)
        await runner._bot_loop_hold_pump_once()
        ev = handed(runner.adapter)[-1]
        runner._bot_loop_hold_turn_claimed(ev)
        runner._bot_loop_hold_turn_ended(ev)
    guard = runner._bot_loop_guard
    key = conv(runner, held[2])
    assert guard.hold_level(key) == 3
    assert await admit(runner, event(PERSON, "carry on", is_bot=False)) is True
    assert guard.hold_level(key) == 1  # reset at once, not after a quiet window
    await runner._bot_loop_hold_pump_once()  # the person-released entry goes
    ev = handed(runner.adapter)[-1]
    assert ev is held[2]
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)
    await fill_window(runner, clock, 10, gap=0.1)  # a fresh burst: the person's release cleared the window
    fresh = event(BOT_B, "a fresh burst")
    assert await admit(runner, fresh) is False
    before = len(handed(runner.adapter))
    clock.advance(14.9)
    await runner._bot_loop_hold_pump_once()
    assert len(handed(runner.adapter)) == before  # not yet
    clock.advance(0.1)
    await runner._bot_loop_hold_pump_once()
    assert handed(runner.adapter)[before:] == [fresh]  # the first wait is 15 s again, not 60


# --- c1: the human_relays bot-inline-mention-required drop site ----------------------------------------------------


@pytest.mark.asyncio
async def test_c1_a_human_relay_dropped_for_the_missing_inline_mention_reports_the_person_and_releases(
        clock, monkeypatch, caplog):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true", DISCORD_ALLOW_BOTS="all")
    monkeypatch.setenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "true")
    caplog.set_level(logging.INFO)
    runner = _release_runner(clock, human_relays=[RELAY])
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH)
    plain_bot = SimpleNamespace(id=777, bot=True, display_name="peer", name="peer")
    await adapter._dispatch_discord_message(_msg(_Chan(CH), "no mention", author=plain_bot))
    assert "reason=bot-inline-mention-required" in caplog.text
    assert await _released(runner, held) is False  # control: an ordinary bot dropped here releases nothing
    caplog.clear()
    relay = SimpleNamespace(id=int(RELAY), bot=True, display_name="vm-chat", name="vm-chat")
    assert await adapter._dispatch_discord_message(_msg(_Chan(CH), "<player> hi", author=relay)) is False
    assert "reason=bot-inline-mention-required" in caplog.text  # dropped at the site this case means to reach
    adapter.handle_message.assert_not_awaited()
    assert await _released(runner, held) is True  # but a relay counts as a person: it released the hold


# --- r2-1: a /stop stops the retry of an in-flight hand-over that then raises (S7) ----------------------------------


@pytest.mark.asyncio
async def test_r2_1_a_stop_during_a_hand_over_that_then_raises_is_not_retried_and_leaves_no_lost_line(clock):
    runner = make_runner(clock, Cfg())
    adapter = runner.adapter
    entered, release = asyncio.Event(), asyncio.Event()

    async def _blocked_then_raises(ev):
        entered.set()
        await release.wait()
        raise RuntimeError("the adapter refused it")
    adapter.handle_message.side_effect = _blocked_then_raises
    await fill_window(runner, clock, 10, gap=0.1)
    head, rest = event(BOT_A, "head"), event(BOT_B, "rest")
    assert await admit(runner, head) is False and await admit(runner, rest) is False
    clock.advance(15)
    pump_task = asyncio.ensure_future(runner._bot_loop_hold_pump_once())
    await asyncio.wait_for(entered.wait(), 5)  # the head is in flight inside handle_message
    guard = runner._bot_loop_guard
    key = conv(runner, head)
    assert guard.held_events(key) == [rest]
    assert await admit(runner, event(PERSON, "/stop", is_bot=False)) is True  # the person's /stop arrives first
    release.set()
    await asyncio.wait_for(pump_task, 5)
    assert guard.held_events(key) == []  # neither the head (not put back) nor the rest (discarded)
    for _ in range(3):
        clock.advance(120)
        await runner._bot_loop_hold_pump_once()
    assert adapter.handle_message.await_count == 1  # never retried
    assert not any("could not hand over" in text for _chat, text in adapter.notices)  # and no lost line either


# --- r2-2: the /stop rule strips ALL leading mentions (S7) ---------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("/stop", True),
    ("/STOP", True),
    ("/stop now", True),
    ("/stop@Boba", True),
    ("<@1> /stop", True),
    ("<@1> <@2> /stop", True),
    ("<@!1> <@2> <@!3> /stop now", True),
    ("<@1> hello /stop", False),
    ("<@1> <@2> /new", False),
    ("stop", False),
    ("", False),
])
def test_r2_2_is_stop_text_strips_every_leading_mention_and_nothing_else(text, expected):
    assert BotLoopGuard.is_stop_text(text) is expected


@pytest.mark.asyncio
async def test_r2_2_a_stop_that_mentions_two_other_bots_is_dropped_by_the_adapter_and_still_discards(clock, monkeypatch):
    _discord_env(monkeypatch, DISCORD_REQUIRE_MENTION="true")
    runner = _release_runner(clock)
    adapter = _seat(runner)
    held = await _hold(runner, "boba", CH, n=2)
    first, second = SimpleNamespace(id=555, bot=True), SimpleNamespace(id=556, bot=True)
    message = _msg(_Chan(CH), "<@555> <@556> /stop", mentions=[first, second])
    assert await adapter._dispatch_discord_message(message) is False  # other-bot-addressed: the seat drops it
    await pump(runner, clock, advance=3600)
    assert handed(runner.adapter) == []  # discarded, not released
    assert runner._bot_loop_guard.held_events(conv(runner, held[0])) == []


# --- r2-3: the notice recognition is narrow (S6) ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_r2_3_while_a_send_is_in_flight_only_channel_author_and_exact_text_together_are_recognised(monkeypatch):
    adapter = _real_adapter(monkeypatch)
    results = {}

    class _InSend(_NoticeChan):
        async def send(self, **kwargs):
            text = kwargs["content"]
            other_channel = _NoticeChan(THREAD[:-1] + "9")
            probes = {
                "same channel, author and text": _bot_msg(self, text, author_id=SELF_ID),
                "other author, same text": _bot_msg(self, text, author_id=SELF_ID + 5),
                "same author and text, other channel": _bot_msg(other_channel, text, author_id=SELF_ID),
                "same channel and author, other text": _bot_msg(self, text + " (edited)", author_id=SELF_ID),
            }
            for name, probe in probes.items():
                results[name] = await adapter._dispatch_discord_message(probe)
            return await super().send(**kwargs)
    adapter._channels[THREAD] = _InSend(THREAD)
    await adapter.send_hold_notice(THREAD, "held: delayed, not dropped")
    assert results == {
        "same channel, author and text": False,  # the in-flight notice: dropped before admission
        "other author, same text": True,  # another bot saying the same words during the send is admitted
        "same author and text, other channel": True,
        "same channel and author, other text": True,
    }


@pytest.mark.asyncio
async def test_r2_3_a_registered_message_id_is_recognised_in_its_own_channel_only(monkeypatch):
    adapter = _real_adapter(monkeypatch)
    first = await adapter.send_hold_notice(THREAD, "held: delayed, not dropped")
    second = await adapter.send_hold_notice(THREAD, "held again: delayed, not dropped")
    own = _bot_msg(adapter._channels[THREAD], "some other words", author_id=SELF_ID + 1, msg_id=int(second.message_id))
    assert await adapter._dispatch_discord_message(own) is False  # control: its own channel recognises the id
    elsewhere = _NoticeChan(THREAD[:-1] + "8")
    stranger = _bot_msg(elsewhere, "an ordinary bot message", author_id=SELF_ID + 1, msg_id=int(first.message_id))
    assert await adapter._dispatch_discord_message(stranger) is True  # the same id in ANOTHER channel matches nothing


# --- r2-4: the notice flags clear when an episode ends, not only on a person's message (S6) -------------------


@pytest.mark.asyncio
async def test_r2_4_a_new_episode_after_a_quiet_window_posts_a_new_notice_without_any_person_writing(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=10)  # t = 0..90
    first = event(BOT_A, "first hold")
    assert await admit(runner, first) is False  # t = 100
    await asyncio.sleep(0)
    await pump(runner, clock)
    assert len(runner.adapter.notices) == 1
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()  # t = 115: released by the timer; the queue is empty, the level is 2
    ev = handed(runner.adapter)[-1]
    runner._bot_loop_hold_turn_claimed(ev)
    runner._bot_loop_hold_turn_ended(ev)
    clock.advance(10)
    # An unrelated chat's message runs the idle sweep NOW, while this chat's window is not yet empty: its hold object
    # survives, so only the episode-end rule (not a brand-new hold object) can re-arm the notice flags below.
    assert await admit(runner, event(BOT_A, chat=THREAD[:-1] + "7")) is True  # t = 125
    clock.advance(115)  # t = 240: a full window since the release (t = 115); the next sweep is at t = 245
    await fill_window(runner, clock, 10, gap=0.1)  # a new burst, no person involved
    second = event(BOT_B, "second hold")
    assert await admit(runner, second) is False
    await asyncio.sleep(0)
    await pump(runner, clock)
    assert len(runner.adapter.notices) == 2  # a second episode, a second notice


# --- r2-5: the turn record binds by message id, so a hook-rewritten copy claims and ends it (S2, S4) -----------


@pytest.mark.asyncio
async def test_r2_5_a_hook_rewritten_copy_of_the_handed_over_event_claims_and_ends_the_turn_record(clock):
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10, gap=0.1)
    first, second = event(BOT_A, "first"), event(BOT_B, "second")
    assert await admit(runner, first) is False and await admit(runner, second) is False
    clock.advance(15)
    await runner._bot_loop_hold_pump_once()
    assert handed(runner.adapter) == [first]
    rewritten = dataclasses.replace(first, text=first.text + " (rewritten by pre_gateway_dispatch)")
    assert rewritten is not first and rewritten.message_id == first.message_id
    runner._bot_loop_hold_turn_claimed(rewritten)  # what the runner calls for the event it actually runs
    runner._bot_loop_hold_turn_ended(rewritten)
    clock.advance(31)  # the second is due; an UNCLAIMED record would hold it back for 60 s
    await runner._bot_loop_hold_pump_once()
    assert handed(runner.adapter) == [first, second]


# --- r2-6: _report_human_seen reports once per message id (S7) ---------------------------------------------------


@pytest.mark.asyncio
async def test_r2_6_a_dropped_persons_message_is_reported_once_per_message_id(clock, monkeypatch):
    _discord_env(monkeypatch)
    adapter = _seat(SimpleNamespace())
    seen = Mock()
    adapter.gateway_runner = SimpleNamespace(_bot_loop_human_seen=seen)
    message = _msg(_Chan(CH), "hello there", msg_id=1573300000000000001)
    adapter._report_human_seen(message)
    adapter._report_human_seen(message)  # the recovered path meets the same message again
    assert seen.call_count == 1
    adapter._report_human_seen(_msg(_Chan(CH), "hello there", msg_id=1573300000000000002))
    assert seen.call_count == 2  # a different id is reported
