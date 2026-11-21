"""Loop-guard-hold, notices and restarts (spec r15 S6, S8): the notice text and send path, recognition at every served
seat, the refs file and the startup lines. Names come from tests/gateway/lgh_support.py's contract; the real
DiscordAdapter is used for ``send_hold_notice`` and for recognition at ingress.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import stat
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult

from tests.gateway.lgh_support import (
    BOT_A, BOT_B, PERSON, THREAD, Cfg, admit, clock, conv, event, fill_window, handed, make_adapter, make_runner,
    pump,
)

_IDS = itertools.count(1572000000000000000)
SELF_ID = 999


class _Chan:
    def __init__(self, channel_id):
        self.id = int(channel_id)
        self.name = "chan"
        self.guild = SimpleNamespace(name="Hermes Server", id=1)
        self.topic = None
        self.parent = None
        self.parent_id = None
        self.sent = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(id=next(_IDS), channel=self, **kwargs)

    def history(self, **kw):
        async def _it():
            return
            yield
        return _it()


class _Thread(_Chan):
    pass


def _real_adapter(monkeypatch, *, allow_bots="all", enabled=True, channel=None, user_id=None):
    import plugins.platforms.discord.adapter as mod
    from plugins.platforms.discord.adapter import DiscordAdapter

    for var in ("DISCORD_ALLOW_BOTS", "DISCORD_REQUIRE_MENTION", "DISCORD_AUTO_THREAD", "DISCORD_BOTS_REQUIRE_INLINE_MENTION",
                "DISCORD_HISTORY_BACKFILL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", allow_bots)
    monkeypatch.setenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false")
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")
    monkeypatch.setattr(mod.discord, "Thread", _Thread, raising=False)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    me = SimpleNamespace(id=user_id or SELF_ID, bot=True)
    adapter._client = SimpleNamespace(user=me)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._ready_event.set()
    adapter._channels = {}

    async def _resolve(chat_id):
        return adapter._channels.setdefault(str(chat_id), channel or _Chan(chat_id))
    adapter._resolve_channel = _resolve
    adapter.me = me
    return adapter


def _bot_msg(channel, content, *, author_id=SELF_ID, msg_id=None):
    import discord
    author = SimpleNamespace(id=author_id, bot=True, display_name="Boba", name="Boba")
    return SimpleNamespace(id=msg_id or next(_IDS), content=content, mentions=[], attachments=[], reference=None,
                           created_at=datetime.now(UTC), channel=channel, guild=channel.guild, author=author,
                           type=discord.MessageType.default)


# --- T15: the real adapter's send path and recognition --------------------------------------------------------


@pytest.mark.asyncio
async def test_T15_send_hold_notice_passes_allowed_mentions_none_and_no_reference(monkeypatch):
    import discord

    adapter = _real_adapter(monkeypatch)
    result = await adapter.send_hold_notice(THREAD, "Messages from @Peer are delayed")
    assert result.success is True
    kwargs = adapter._channels[THREAD].sent[0]
    assert kwargs["content"] == "Messages from @Peer are delayed"
    assert isinstance(kwargs["allowed_mentions"], discord.AllowedMentions)
    am = kwargs["allowed_mentions"]
    assert (am.everyone, am.users, am.roles, am.replied_user) == (False, False, False, False)
    assert kwargs.get("reference") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_T15_a_registered_notice_is_dropped_at_an_allow_bots_all_seat_before_admission(monkeypatch, enabled):
    adapter = _real_adapter(monkeypatch)
    adapter.config.extra["bot_loop_guard"] = {"enabled": enabled}
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda ev: seen.append(ev))
    sent = await adapter.send_hold_notice(THREAD, "Held: delayed, not dropped")
    chan = adapter._channels[THREAD]
    echo = _bot_msg(chan, "Held: delayed, not dropped", msg_id=int(sent.message_id))  # the gateway event of our post
    echo.author = SimpleNamespace(id=SELF_ID + 1, bot=True, display_name="Peer", name="Peer")  # a peer seat sees it
    assert await adapter._dispatch_discord_message(echo) is False
    assert seen == []  # dropped before admission: never counted, never held


@pytest.mark.asyncio
async def test_T15_a_notice_in_flight_is_recognised_by_channel_author_and_text_and_another_bots_same_text_is_not(monkeypatch):
    adapter = _real_adapter(monkeypatch)
    gate = {}

    class _Slow(_Chan):
        async def send(self, **kwargs):
            gate["during"] = True  # the send is in flight: the echo can arrive before send() returns
            echo = _bot_msg(self, kwargs["content"], author_id=SELF_ID, msg_id=1572999999999999999)
            gate["dropped"] = await adapter._dispatch_discord_message(echo) is False
            return await super().send(**kwargs)
    adapter._channels[THREAD] = _Slow(THREAD)
    await adapter.send_hold_notice(THREAD, "same words")
    assert gate == {"during": True, "dropped": True}
    other = _bot_msg(adapter._channels[THREAD], "same words", author_id=SELF_ID + 5, msg_id=1572999999999999998)
    other.author.display_name = "Other"
    assert await adapter._dispatch_discord_message(other) is True  # another bot saying the same text is admitted


@pytest.mark.asyncio
async def test_T15_a_failed_send_keeps_the_in_flight_entry_so_a_posted_but_timed_out_notice_is_still_recognised(monkeypatch):
    adapter = _real_adapter(monkeypatch)

    class _Timeout(_Chan):
        async def send(self, **kwargs):
            raise TimeoutError("posted, but the response was lost")
    adapter._channels[THREAD] = _Timeout(THREAD)
    result = await adapter.send_hold_notice(THREAD, "delayed, not dropped")
    assert result.success is False
    echo = _bot_msg(adapter._channels[THREAD], "delayed, not dropped", author_id=SELF_ID)
    assert await adapter._dispatch_discord_message(echo) is False  # recognised for its full 30 s


@pytest.mark.asyncio
async def test_T15_a_notice_posted_in_a_thread_is_recognised_in_that_thread_and_skipped_by_history_backfill(monkeypatch):
    adapter = _real_adapter(monkeypatch)
    thread = _Thread(THREAD)
    adapter._channels[THREAD] = thread
    sent = await adapter.send_hold_notice(THREAD, "Messages are being delayed")
    notice = _bot_msg(thread, "Messages are being delayed", author_id=SELF_ID, msg_id=int(sent.message_id))
    human = SimpleNamespace(id=7, bot=False, display_name="Jezza", name="Jezza")
    talk = _bot_msg(thread, "what do you think?", msg_id=int(sent.message_id) + 1)
    talk.author = human
    trigger = _bot_msg(thread, "ping", msg_id=int(sent.message_id) + 2)
    trigger.author = human

    def _history(**kw):
        async def _it():
            for m in (talk, notice):
                yield m
        return _it()
    thread.history = _history
    context = await adapter._fetch_channel_context(thread, trigger)
    assert "what do you think?" in context
    assert "Messages are being delayed" not in context


async def _post_hold_notice(clock, senders, *, cfg=None, chat=THREAD, held_each=1):
    """A runner whose window is full; ``senders`` (user names) each have messages held; one pump wake posts the notice."""
    runner = make_runner(clock, cfg or Cfg(), profile="boba")
    await fill_window(runner, clock, 10, chat=chat)
    for n, name in enumerate(senders):
        for _ in range(held_each):
            assert await admit(runner, event(f"20000000000000{n:04d}", chat=chat, name=name)) is False
    await asyncio.sleep(0)  # a send the guard scheduled onto the loop runs here (S13)
    await pump(runner, clock)
    return runner


@pytest.mark.asyncio
async def test_T15_the_notice_the_runner_posts_names_the_senders_the_seat_the_delay_and_the_count_and_every_clause(clock):
    runner = await _post_hold_notice(clock, ["Peer Bot", "Other Bot"], cfg=Cfg(bot_replies_expect_reply=False),
                                     held_each=2)
    assert [chat for chat, _text in runner.adapter.notices] == [THREAD]  # one message, to the conversation
    text = runner.adapter.notices[0][1]
    assert "Peer Bot" in text and "Other Bot" in text  # the sender bots
    assert "boba" in text.lower()  # this seat
    assert re.search(r"\b15\s*(s\b|sec)", text)  # the delay
    assert re.search(r"\b4\b", text)  # the number held
    assert "delayed" in text and "not dropped" in text  # held messages are delayed, not dropped
    assert "from the same sender" in text  # the repeat rule that still drops
    assert "reply_to_reply" in text or "answering the reply" in text  # this seat has the rule on
    assert re.search(r"person", text) and re.search(r"releas", text)  # a person's message releases them
    assert "/stop" in text and "discard" in text  # a person's /stop discards them
    assert "answer" in text and "resend" in text  # bots must not answer or resend


@pytest.mark.asyncio
async def test_T15_the_reply_to_reply_clause_is_named_only_where_that_rule_is_on(clock):
    on = await _post_hold_notice(clock, ["Peer"], cfg=Cfg(bot_replies_expect_reply=False))
    off = await _post_hold_notice(clock, ["Peer"], cfg=Cfg(bot_replies_expect_reply=True))
    assert "reply_to_reply" in on.adapter.notices[0][1] or "answering the reply" in on.adapter.notices[0][1]
    assert "reply_to_reply" not in off.adapter.notices[0][1] and "answering the reply" not in off.adapter.notices[0][1]
    assert "from the same sender" in off.adapter.notices[0][1]  # the repeat rule is always named


@pytest.mark.asyncio
async def test_T15_at_most_5_senders_are_named_each_cut_to_32_characters_then_and_n_more(clock):
    names = [f"Bot{n}_" + "x" * 40 for n in range(7)]
    runner = await _post_hold_notice(clock, names)
    text = runner.adapter.notices[0][1]
    assert "Bot0_" + "x" * 27 in text and "Bot0_" + "x" * 28 not in text  # cut to 32
    assert "Bot4_" in text and "Bot5_" not in text and "Bot6_" not in text
    assert "and 2 more" in text


@pytest.mark.asyncio
async def test_T15_names_are_escaped_so_a_sender_cannot_ping_link_or_format_through_the_notice(clock):
    hostile = "Evil @everyone @here #general **b** `c` _i_ ~~s~~ [l](http://x) <@123>"
    runner = await _post_hold_notice(clock, [hostile])
    text = runner.adapter.notices[0][1]
    for raw in ("@everyone", "@here", "#general", "**b**", "`c`", "_i_", "~~s~~", "[l](http://x)", "<@123>"):
        assert raw not in text, f"{raw!r} reaches the channel unescaped"
    assert "everyone" in text and "general" in text  # the name is still readable


@pytest.mark.asyncio
async def test_T15_a_second_notice_is_posted_once_when_the_conversation_pauses_and_none_per_message(clock):
    runner = make_runner(clock, Cfg(hold_runaway_at_cap=1, hold_initial_seconds=15, hold_max_seconds=15),
                         profile="boba")
    await fill_window(runner, clock, 10)
    for _ in range(3):
        assert await admit(runner, event(BOT_A, name="Peer")) is False
    await asyncio.sleep(0)
    await pump(runner, clock)
    assert len(runner.adapter.notices) == 1  # one when it started holding, not one per message
    await pump(runner, clock, advance=15)  # the head is handed over after a full-cap wait
    ev = handed(runner.adapter)[0]
    runner._bot_loop_hold_turn_claimed(ev)  # its turn starts and ends (the live path is pinned in test_lgh_live_wiring)
    runner._bot_loop_hold_turn_ended(ev)
    await asyncio.sleep(0)
    await pump(runner, clock)  # hold_runaway_at_cap=1: one capped wait pauses it
    paused = [text for _chat, text in runner.adapter.notices[1:] if "paus" in text.lower()]
    assert len(paused) == 1 and len(runner.adapter.notices) == 2
    await pump(runner, clock, advance=3600)
    assert len(runner.adapter.notices) == 2  # nothing more while paused


@pytest.mark.asyncio
async def test_T15_a_seat_that_saw_a_peers_notice_but_never_held_posts_its_own_when_it_later_starts_holding(
        clock, monkeypatch):
    peer = _real_adapter(monkeypatch)  # seat A: posts a notice through the real send path
    seat = _real_adapter(monkeypatch, user_id=SELF_ID + 1)  # seat B: sees it as an inbound bot message
    sent = await peer.send_hold_notice(THREAD, "Peer is held: delayed, not dropped. A person's /stop discards them.")
    seen = []
    seat.handle_message = AsyncMock(side_effect=lambda ev: seen.append(ev))
    echo = _bot_msg(seat._channels.setdefault(THREAD, _Chan(THREAD)),
                    "Peer is held: delayed, not dropped. A person's /stop discards them.",
                    author_id=SELF_ID, msg_id=int(sent.message_id))
    assert await seat._dispatch_discord_message(echo) is False  # recognised and dropped: B never counted it
    assert seen == []
    runner = make_runner(clock, Cfg(), adapter=seat, profile="tars")  # B has held nothing so far
    assert seat._channels[THREAD].sent == []
    await fill_window(runner, clock, 10)
    assert await admit(runner, event(BOT_A, name="Peer")) is False  # B starts holding
    await asyncio.sleep(0)
    await runner._bot_loop_hold_pump_once()
    own = [kwargs["content"] for kwargs in seat._channels[THREAD].sent]
    assert len(own) == 1 and "delayed" in own[0]  # B's own notice: the peer's did not stand in for it


# --- T47: lost and discard lines are recognised like the notice -----------------------------------------------


@pytest.mark.asyncio
async def test_T47_lost_and_discard_lines_are_sent_by_send_hold_notice_and_recognised_at_every_seat(clock, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = make_runner(clock, Cfg())
    await fill_window(runner, clock, 10)
    held = [event(BOT_A) for _ in range(2)]
    for ev in held:
        assert await admit(runner, ev) is False
    await asyncio.sleep(0)  # sends scheduled from the guard's sync code run on the loop (S13), not under admit
    await pump(runner, clock)
    assert len(runner.adapter.notices) == 1  # the hold notice
    assert await admit(runner, event(PERSON, "/stop", is_bot=False)) is True  # discard line
    await asyncio.sleep(0)
    await pump(runner, clock)
    assert len(runner.adapter.notices) == 2  # the hold notice and the discard line, both through send_hold_notice
    assert all(ev.message_id in runner.adapter.notices[1][1] for ev in held)
    adapter = _real_adapter(monkeypatch)
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda ev: seen.append(ev))
    for text in ("one line listing what was discarded", "lost on restart: one line"):
        sent = await adapter.send_hold_notice(THREAD, text)
        echo = _bot_msg(adapter._channels[THREAD], text, author_id=SELF_ID + 1, msg_id=int(sent.message_id))
        assert await adapter._dispatch_discord_message(echo) is False  # dropped at every served seat
    assert seen == []


# --- T9, T35, T43: the refs file and the startup lines -------------------------------------------------------


def _refs_path(tmp_path):
    return tmp_path / "gateway" / "bot_loop_hold_refs.json"


def _refs_text(tmp_path):
    """The refs file's text; "" when a build removes the file once it lists nothing (the spec removes entries)."""
    path = _refs_path(tmp_path)
    return path.read_text() if path.exists() else ""


async def _hold_n(runner, clock, n, *, chat=THREAD):
    await fill_window(runner, clock, 10, gap=0.1, chat=chat)
    held = [event(BOT_A, chat=chat, message_id=f"157300000000000{i:04d}", name="Peer") for i in range(n)]
    for ev in held:
        assert await admit(runner, ev) is False
    return held


@pytest.mark.asyncio
async def test_T9_the_refs_file_lists_held_messages_without_text_mode_0600_and_a_restart_names_them_once(clock, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = make_runner(clock, Cfg())
    held = await _hold_n(runner, clock, 3)
    runner._bot_loop_hold_flush_refs()
    path = _refs_path(tmp_path)
    assert path.exists()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    raw = path.read_text()
    assert all(ev.message_id in raw for ev in held)
    assert not any(ev.text in raw for ev in held)  # no message text in the file
    # a restart: fresh runner, same home
    after = make_runner(clock, Cfg())
    await after._bot_loop_hold_report_lost()
    lines = [text for _chat, text in after.adapter.notices if THREAD in text or held[0].message_id in text]
    assert len(lines) == 1  # one line per conversation
    assert all(ev.message_id in lines[0] for ev in held) and "Peer" in lines[0]
    assert not any(ev.message_id in _refs_text(tmp_path) for ev in held)  # the file empties (or goes)


@pytest.mark.asyncio
async def test_T9_a_corrupt_refs_file_is_moved_aside_with_a_warning(clock, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    caplog.set_level(logging.WARNING)
    path = _refs_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ this is not json")
    runner = make_runner(clock, Cfg())
    await runner._bot_loop_hold_report_lost()
    assert not path.exists() or path.read_text() != "{ this is not json"
    assert [p for p in path.parent.iterdir() if p.name != path.name and "refs" in p.name]  # moved aside, not deleted
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert runner.adapter.notices == []


@pytest.mark.asyncio
async def test_T9_a_missing_profile_home_is_never_recreated(clock, monkeypatch, tmp_path):
    home = tmp_path / "gone"
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = make_runner(clock, Cfg())
    await _hold_n(runner, clock, 1)
    runner._bot_loop_hold_flush_refs()
    assert not home.exists()


@pytest.mark.asyncio
async def test_T35_60_lost_messages_post_in_messages_of_at_most_1900_chars_and_a_failed_send_keeps_its_entry(clock, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = make_runner(clock, Cfg(hold_queue_max=1000))
    held = await _hold_n(runner, clock, 60)
    runner._bot_loop_hold_flush_refs()
    after = make_runner(clock, Cfg())
    calls = {"n": 0}
    original = after.adapter.send_hold_notice.side_effect

    async def _flaky(chat_id, text, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # the second message fails
            return SendResult(success=False, error="boom")
        return await original(chat_id, text, *a, **kw)
    after.adapter.send_hold_notice.side_effect = _flaky
    await after._bot_loop_hold_report_lost()
    sent = [text for _chat, text in after.adapter.notices]
    assert sent and all(len(text) <= 1900 for text in sent)
    assert len(sent) <= 5  # at most 5 messages per conversation per start
    left = json.loads(_refs_text(tmp_path))
    listed = json.dumps(left)
    named_sent = {ev.message_id for ev in held if any(ev.message_id in text for text in sent)}
    assert named_sent and named_sent != {ev.message_id for ev in held}
    assert all(ev.message_id in listed for ev in held if ev.message_id not in named_sent)  # kept for the next start
    assert not any(ev.message_id in listed for ev in held if ev.message_id in named_sent)  # removed once named


@pytest.mark.asyncio
async def test_T35_at_most_5_lost_messages_per_conversation_and_the_rest_summed(clock, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = make_runner(clock, Cfg(hold_queue_max=1000))
    await _hold_n(runner, clock, 300)
    runner._bot_loop_hold_flush_refs()
    after = make_runner(clock, Cfg())
    await after._bot_loop_hold_report_lost()
    sent = [text for _chat, text in after.adapter.notices]
    assert 1 < len(sent) <= 5 and all(len(text) <= 1900 for text in sent)
    assert "and " in sent[-1] and "more" in sent[-1]


@pytest.mark.asyncio
async def test_T43_an_entry_is_gone_after_handover_after_a_stop_discard_and_after_the_third_failed_handover(clock, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = make_runner(clock, Cfg())
    held = await _hold_n(runner, clock, 3)
    runner._bot_loop_hold_flush_refs()
    assert all(ev.message_id in _refs_text(tmp_path) for ev in held)
    await pump(runner, clock, advance=15)  # held[0] handed over and accepted
    runner._bot_loop_hold_flush_refs()
    raw = _refs_text(tmp_path)
    assert held[0].message_id not in raw and held[1].message_id in raw
    runner._bot_loop_hold_turn_claimed(held[0])
    runner._bot_loop_hold_turn_ended(held[0])
    assert await admit(runner, event(PERSON, "/stop", is_bot=False)) is True  # discard held[1], held[2]
    runner._bot_loop_hold_flush_refs()
    assert not any(ev.message_id in _refs_text(tmp_path) for ev in held)
    # third failed hand-over
    runner2 = make_runner(clock, Cfg())
    held2 = await _hold_n(runner2, clock, 1, chat="chat-two")
    runner2.adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
    for _ in range(4):
        await pump(runner2, clock, advance=70)
    runner2._bot_loop_hold_flush_refs()
    assert held2[0].message_id not in _refs_text(tmp_path)
    # a graceful stop lists exactly what is still held; a restart posts no line for a delivered one
    runner3 = make_runner(clock, Cfg())
    kept = await _hold_n(runner3, clock, 2, chat="chat-three")
    await pump(runner3, clock, advance=15)  # kept[0] delivered
    runner3._bot_loop_hold_flush_refs()
    raw = _refs_text(tmp_path)
    assert kept[1].message_id in raw and kept[0].message_id not in raw
    after = make_runner(clock, Cfg())
    await after._bot_loop_hold_report_lost()
    said = " ".join(text for _chat, text in after.adapter.notices)
    assert kept[1].message_id in said and kept[0].message_id not in said


# --- T29: a bot voice utterance ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_T29_a_bot_voice_utterance_past_the_window_is_held_released_and_listed_as_a_voice_message_after_a_restart(
        clock, monkeypatch, tmp_path):
    from gateway.platforms.base import MessageEvent

    from tests.gateway.test_discord_voice_bot_speaker import _BOT, _GUILD, _TEXT, _runner as voice_runner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    vrunner, vadapter = voice_runner(monkeypatch, "bound", "all", inline_mention=False)
    runner = make_runner(clock, Cfg())
    for name in ("_voice_mode", "_session_db", "session_store", "adapters"):
        setattr(runner, name, getattr(vrunner, name))
    runner.adapter = vadapter
    runner._adapter_for_source = lambda source: vadapter
    vsent = []
    vadapter.send_hold_notice = AsyncMock(side_effect=lambda c, t: vsent.append(t) or SendResult(success=True, message_id="1"))
    source = runner._voice_input_source(vadapter, _GUILD, _BOT, _TEXT)
    for n in range(10):
        assert await runner._hm_admit_event(MessageEvent(text=f"ask {n}", message_id=f"v{n}", source=source)) is not None
    await runner._handle_voice_channel_input(_GUILD, _BOT, "what broke?", adapter=vadapter)
    spoken = vadapter.handle_message.await_args_list[-1].args[0]
    assert spoken.message_id in (None, "")  # speech has no message id
    assert await runner._hm_admit_event(spoken) is None  # past the window: held
    runner._bot_loop_hold_flush_refs()
    raw = (tmp_path / "gateway" / "bot_loop_hold_refs.json").read_text()
    assert "voice" in raw and "what broke?" not in raw
    after = make_runner(clock, Cfg())
    await after._bot_loop_hold_report_lost()
    said = " ".join(text for _chat, text in after.adapter.notices)
    assert "a voice message from" in said
    # released by the timer in the lived runner
    await pump(runner, clock, advance=15)
    assert vadapter.handle_message.await_args_list[-1].args[0] is spoken
