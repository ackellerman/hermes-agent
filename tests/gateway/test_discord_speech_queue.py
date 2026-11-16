"""One speech queue per guild: no two clips of one guild ever sound at once (plan step 1).

Every case drives the real adapter play paths — ``play_in_voice_channel`` through ``play_tts``, the
pre-turn brief, the ready tone and ``run_voice``, plus ``play_ack_in_voice`` — with a fake mixer at
the edge. The last case judges the log lines with the three regexes of the admin board's
``check-overlap.py`` (:24-26), copied character for character.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from plugins.platforms.discord import adapter as adapter_mod

GUILD = 111
OTHER_GUILD = 222
CHAT_ID = "123"
ADAPTER_LOGGER = "plugins.platforms.discord.adapter"

START_RE = re.compile(r"Voice speech start \(guild=(\d+), clip=([^\s,()]+), kind=(\w+), waited_ms=(\d+)\)")
END_RE = re.compile(r"Voice speech end \(guild=(\d+), clip=([^\s,()]+), kind=(\w+), result=(\w+)\)")


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class _FakeChild:
    """A mixer speech child: silent until the test finishes or stops it."""

    def __init__(self):
        self._finished = False
        self._stopped = False


class _FakeMixer:
    """The mixer edge: one unfinished child per clip, and how many were ever live at once."""

    def __init__(self):
        self.children: list[_FakeChild] = []
        self.max_live = 0
        self.stops = 0

    def _live(self) -> int:
        return sum(1 for child in self.children if child._finished is not True and child._stopped is not True)

    def play_speech(self, pcm, *, gain=1.0):
        child = _FakeChild()
        self.children.append(child)
        self.max_live = max(self.max_live, self._live())
        return child

    def stop_speech(self):
        self.stops += 1
        for child in self.children:
            if child._finished is not True:
                child._stopped = True

    @property
    def speech_active(self) -> bool:
        # The base code polls this; under the queue it is never the signal that ends a clip.
        return self._live() > 0


def _fake_discord():
    """Just enough of discord.py for the legacy one-shot path."""

    class _PCM:
        def __init__(self, *args, **kwargs):
            pass

        def read(self):
            return b""

        def is_opus(self):
            return False

        def cleanup(self):
            pass

    class _AudioSource:
        pass

    return SimpleNamespace(
        AudioSource=_AudioSource,
        FFmpegPCMAudio=_PCM,
        PCMVolumeTransformer=lambda source, volume: source,
    )


def _vc():
    vc = MagicMock()
    vc.is_connected.return_value = True
    return vc


def _bare_adapter(*guilds):
    """A real adapter built without ``__init__``, as the rest of the voice suite builds one."""
    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._voice_clients = {guild: _vc() for guild in guilds}
    adapter._voice_receivers = {}
    adapter._voice_mixers = {}
    adapter._voice_text_channels = {}
    adapter._voice_fx_cfg = {"lead_silence_ms": 0}
    adapter._cancel_voice_timeout = lambda guild_id: None
    adapter._reset_voice_timeout = lambda guild_id: None
    adapter._lead_silence_bytes = lambda: b""

    async def playback_timeout(audio_path):
        return 5.0

    adapter._playback_timeout_for_audio = playback_timeout
    return adapter


def _mixer_adapter(monkeypatch, *guilds):
    """The shared setup: a bare adapter with a fake mixer per guild and a stubbed decode."""
    adapter = _bare_adapter(*guilds)
    mixers = {guild: _FakeMixer() for guild in guilds}
    adapter._voice_mixers = mixers
    monkeypatch.setattr(
        adapter_mod._voice_mixer_module(), "decode_to_pcm", lambda path, **kwargs: str(path).encode()
    )
    return adapter, mixers


def _ack_adapter(monkeypatch):
    """A mixer adapter whose acks are on, with a TTS stub that writes the clip file."""
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    adapter._voice_fx_cfg = {
        "lead_silence_ms": 0, "speech_gain": 1.0, "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    from tools import tts_tool

    def fake_tts(text, output_path=None, **kwargs):
        with open(output_path, "wb") as fh:
            fh.write(b"ack")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)
    return adapter, mixers


# ---------------------------------------------------------------------------
# Waiting helpers
# ---------------------------------------------------------------------------

async def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Poll until the predicate holds; returns False if it never did within the window."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return predicate()
        await asyncio.sleep(0.01)
    return True


async def _until(predicate, timeout: float = 2.0):
    assert await _wait_for(predicate, timeout), "the expected state never arrived"
    return True


async def _no_second_clip(mixer, window: float = 0.5) -> bool:
    """A second clip must not join the mixer while the first child is still unfinished."""
    return not await _wait_for(lambda: len(mixer.children) > 1, timeout=window)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _starts(caplog):
    return [message for message in _messages(caplog) if message.startswith("Voice speech start")]


def _ends(caplog):
    return [message for message in _messages(caplog) if message.startswith("Voice speech end")]


# ---------------------------------------------------------------------------
# One clip at a time, whoever the caller is
# ---------------------------------------------------------------------------

def test_reply_clips_play_one_after_the_other(monkeypatch):
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]

    async def go():
        first = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply-1.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        second = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply-2.mp3"))
        assert await _no_second_clip(mixer), "the second reply started over the first"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await first is True
        assert await second is True

    asyncio.run(go())
    assert mixer.max_live == 1


def test_brief_waits_behind_a_reply(monkeypatch):
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]

    async def go():
        reply = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        brief = asyncio.create_task(
            adapter.play_in_voice_channel(GUILD, "brief.mp3", strict=True, kind="brief")
        )
        assert await _no_second_clip(mixer), "the brief started over the reply"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await reply is True
        assert await brief is True

    asyncio.run(go())
    assert mixer.max_live == 1


def test_ready_tone_waits_behind_a_reply(monkeypatch):
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]

    async def go():
        reply = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        tone = asyncio.create_task(
            adapter.play_in_voice_channel(GUILD, "ready.wav", strict=True, kind="ready")
        )
        assert await _no_second_clip(mixer), "the ready tone started over the reply"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await reply is True
        assert await tone is True

    asyncio.run(go())
    assert mixer.max_live == 1


def test_play_tts_reply_waits_behind_a_brief(monkeypatch):
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]
    adapter._voice_text_channels = {GUILD: CHAT_ID}

    async def go():
        brief = asyncio.create_task(
            adapter.play_in_voice_channel(GUILD, "brief.mp3", strict=True, kind="brief")
        )
        await _until(lambda: len(mixer.children) == 1)
        reply = asyncio.create_task(adapter.play_tts(CHAT_ID, "reply.mp3"))
        assert await _no_second_clip(mixer), "play_tts started over the brief"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await brief is True
        assert (await reply).success is True

    asyncio.run(go())
    assert mixer.max_live == 1


def test_run_voice_reply_waits_behind_a_brief(monkeypatch):
    from gateway.run import GatewayRunner

    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._get_guild_id = lambda event: GUILD
    event = SimpleNamespace(source=SimpleNamespace(chat_id=CHAT_ID))

    async def go():
        brief = asyncio.create_task(
            adapter.play_in_voice_channel(GUILD, "brief.mp3", strict=True, kind="brief")
        )
        await _until(lambda: len(mixer.children) == 1)
        reply = asyncio.create_task(runner._deliver_voice_reply(event, ["reply.mp3"]))
        assert await _no_second_clip(mixer), "the run_voice reply started over the brief"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await brief is True
        assert await reply is None

    asyncio.run(go())
    assert mixer.max_live == 1


def test_ack_waits_behind_a_reply(monkeypatch):
    adapter, mixers = _ack_adapter(monkeypatch)
    mixer = mixers[GUILD]

    async def go():
        reply = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        ack = asyncio.create_task(adapter.play_ack_in_voice(GUILD))
        # Base: the ack adds its child at once and returns True without waiting for it.
        assert not await _wait_for(
            lambda: ack.done() or len(mixer.children) > 1, timeout=0.5
        ), "the ack did not wait for the reply"
        mixer.children[0]._finished = True
        await _until(lambda: len(mixer.children) == 2)
        assert not ack.done(), "the ack returned before its own clip finished"
        mixer.children[1]._finished = True
        assert await ack is True
        assert await reply is True

    asyncio.run(go())
    assert mixer.max_live == 1


def test_two_guilds_do_not_wait_for_each_other(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD, OTHER_GUILD)

    async def go():
        here = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "here.mp3"))
        there = asyncio.create_task(adapter.play_in_voice_channel(OTHER_GUILD, "there.mp3"))
        # Both children are live before either finishes: each guild has its own queue.
        await _until(lambda: bool(mixers[GUILD].children and mixers[OTHER_GUILD].children))
        mixers[GUILD].children[0]._finished = True
        mixers[OTHER_GUILD].children[0]._finished = True
        assert await here is True
        assert await there is True

    asyncio.run(go())
    waits = {}
    for message in _starts(caplog):
        match = START_RE.search(message)
        assert match, f"a start line did not parse: {message}"
        waits[int(match[1])] = int(match[4])
    assert set(waits) == {GUILD, OTHER_GUILD}, "both guilds should have logged a clip start"
    assert waits[GUILD] < 100 and waits[OTHER_GUILD] < 100, f"a guild waited on the other: {waits}"


# ---------------------------------------------------------------------------
# A clip that dies must release the queue
# ---------------------------------------------------------------------------

def test_a_timed_out_clip_releases_the_queue(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]

    async def quick(audio_path):
        return 0.2

    adapter._playback_timeout_for_audio = quick

    async def go():
        stuck = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "stuck.mp3", strict=True))
        await _until(lambda: len(mixer.children) == 1)
        following = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "next.mp3"))
        assert await stuck is False
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await following is True

    asyncio.run(go())
    assert mixer.stops >= 1, "a timed-out clip must drop its child"
    ends = _ends(caplog)
    assert len(ends) == 2, f"expected one end line per clip, got {ends}"
    assert "result=timeout" in ends[0]
    assert "result=done" in ends[1]


def test_a_clip_that_errors_releases_the_queue(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]
    boom = {"armed": True}
    real_play_speech = mixer.play_speech

    def play_speech(pcm, *, gain=1.0):
        if boom["armed"]:
            boom["armed"] = False
            raise RuntimeError("mixer exploded")
        return real_play_speech(pcm, gain=gain)

    mixer.play_speech = play_speech

    async def go():
        broken = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "broken.mp3"))
        await _until(broken.done)
        with pytest.raises(RuntimeError, match="mixer exploded"):
            await broken
        following = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "next.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        mixer.children[0]._finished = True
        assert await following is True

    asyncio.run(go())
    starts, ends = _starts(caplog), _ends(caplog)
    assert len(starts) == 1 and len(ends) == 1, f"the surviving clip logged {starts} / {ends}"
    assert "kind=reply" in starts[0]
    assert "result=done" in ends[0]


def test_a_cancelled_clip_stops_its_child_and_releases(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, mixers = _mixer_adapter(monkeypatch, GUILD)
    mixer = mixers[GUILD]

    async def go():
        first = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "cancelled.mp3"))
        await _until(lambda: len(mixer.children) == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert mixer.children[0]._stopped is True, "the cancelled clip left its child sounding"
        second = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "after.mp3"))
        await _until(lambda: len(mixer.children) == 2)
        mixer.children[1]._finished = True
        assert await second is True

    asyncio.run(go())
    assert mixer.max_live == 1
    ends = _ends(caplog)
    assert len(ends) == 2, f"expected one end line per clip, got {ends}"
    assert "result=stopped" in ends[0] and "result=done" in ends[1], f"unexpected end lines: {ends}"


# ---------------------------------------------------------------------------
# The legacy one-shot path takes the same queue
# ---------------------------------------------------------------------------

def _legacy_adapter(monkeypatch):
    adapter = _bare_adapter(GUILD)
    vc = adapter._voice_clients[GUILD]
    vc.is_playing.return_value = False
    afters: list = []

    def play(source, after=None):
        afters.append(after)

    vc.play.side_effect = play
    monkeypatch.setattr(adapter_mod, "discord", _fake_discord())
    return adapter, vc, afters


def test_legacy_path_plays_one_clip_at_a_time(monkeypatch):
    adapter, _vc_mock, afters = _legacy_adapter(monkeypatch)

    async def go():
        first = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "one.mp3"))
        await _until(lambda: len(afters) == 1)
        second = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "two.mp3"))
        assert not await _wait_for(
            lambda: len(afters) > 1, timeout=0.5
        ), "the second legacy clip played over the first"
        afters[0](None)
        await _until(lambda: len(afters) == 2)
        afters[1](None)
        assert await first is True
        assert await second is True

    asyncio.run(go())


def test_a_cancelled_legacy_clip_stops_and_releases(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, vc, afters = _legacy_adapter(monkeypatch)

    async def go():
        first = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "one.mp3"))
        await _until(lambda: len(afters) == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert vc.stop.called, "the cancelled legacy clip did not stop its source"
        second = asyncio.create_task(adapter.play_in_voice_channel(GUILD, "two.mp3"))
        await _until(lambda: len(afters) == 2)
        afters[1](None)
        assert await second is True

    asyncio.run(go())
    ends = _ends(caplog)
    assert len(ends) == 2, f"expected one end line per clip, got {ends}"
    assert "result=stopped" in ends[0] and "result=done" in ends[1], f"unexpected end lines: {ends}"


# ---------------------------------------------------------------------------
# The log lines, judged with check-overlap.py's own patterns
# ---------------------------------------------------------------------------

def test_log_lines_pair_up_in_the_witness_shape(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    adapter, mixers = _ack_adapter(monkeypatch)
    mixer = mixers[GUILD]

    async def go():
        tasks = [
            asyncio.create_task(adapter.play_in_voice_channel(GUILD, "reply.mp3")),
            asyncio.create_task(
                adapter.play_in_voice_channel(GUILD, "brief.mp3", strict=True, kind="brief")
            ),
            asyncio.create_task(adapter.play_ack_in_voice(GUILD)),
        ]
        for expected in (1, 2, 3):
            await _until(lambda: len(mixer.children) == expected)
            mixer.children[expected - 1]._finished = True
        assert await asyncio.gather(*tasks) == [True, True, True]

    asyncio.run(go())

    events = []
    for index, message in enumerate(_messages(caplog)):
        start, end = START_RE.search(message), END_RE.search(message)
        if start:
            events.append((index, "start", start))
        elif end:
            events.append((index, "end", end))
    assert [event[1] for event in events] == ["start", "end"] * 3, f"start/end did not pair up: {events}"

    ids, kinds, open_clip = [], [], None
    for _index, which, match in events:
        assert int(match[1]) == GUILD
        assert re.fullmatch(r"[^\s,()]+", match[2]), f"clip id has a forbidden character: {match[2]}"
        if which == "start":
            assert open_clip is None, "a clip started while another of its guild was open"
            open_clip = (match[2], match[3])
            ids.append(match[2])
            kinds.append(match[3])
        else:
            assert open_clip == (match[2], match[3]), "a clip's end does not match its start"
            open_clip = None
    assert open_clip is None
    assert len(set(ids)) == 3, f"clip ids repeat: {ids}"
    assert kinds == ["reply", "brief", "ack"], f"unexpected kinds: {kinds}"
