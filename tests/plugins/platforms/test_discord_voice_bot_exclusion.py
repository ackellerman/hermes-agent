"""A seat never transcribes another bot's audio unless its bot policy admits bot speech.

Two seats in one voice channel used to hear each other: each receiver drops only its own SSRC, so a
sibling bot's TTS was decoded and either attributed to the bot (dropped only because the bot was not
on the user allowlist) or, when its SSRC was unmapped, inferred onto the sole allowlisted human (a
feedback loop). Covers the inference rule, the packet-level drop set, and the bot predicate applied in
the listen loop and the leave flush. The predicate is the one #143368 uses for bot speech
(``DISCORD_ALLOW_BOTS=all`` with ``discord.bots_require_inline_mention`` off admits it).
"""
from __future__ import annotations

import struct
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("discord")

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, VoiceReceiver

_GUILD, _ME, _ALPHA, _BETA_BOT, _STRANGER, _GAMMA_BOT = 1, 9001, 101, 202, 303, 404
_SSRC_BETA, _SSRC_ALPHA = 5002, 5001


def _member(uid: int, bot: bool = False) -> SimpleNamespace:
    return SimpleNamespace(id=uid, bot=bot, display_name=f"m{uid}")


def _receiver(members, allowed, *, fleet=(), drop_bots=True) -> VoiceReceiver:
    conn = SimpleNamespace(secret_key=bytes(32), dave_session=None, dave_protocol_version=0,
                           dave_downgraded=False, dave_pending_transitions={}, ssrc=1,
                           hook=None, ws=None, add_socket_listener=lambda cb: None,
                           remove_socket_listener=lambda cb: None)
    vc = SimpleNamespace(_connection=conn, channel=SimpleNamespace(members=list(members)),
                         user=SimpleNamespace(id=_ME))
    return VoiceReceiver(vc, allowed_user_ids=set(allowed), fleet_bot_ids=lambda: set(fleet),
                         drop_bot_audio=drop_bots)


def _infer(receiver: VoiceReceiver, ssrc: int = _SSRC_BETA) -> int:
    with receiver._lock:
        return receiver._infer_user_for_ssrc(ssrc)


# --- inference ---------------------------------------------------------------------------------

@pytest.mark.parametrize("members,allowed,expected", [
    ([_member(_ALPHA), _member(_ME, True)], {str(_ALPHA)}, _ALPHA),                   # sole member: kept
    ([_member(_ALPHA), _member(_BETA_BOT, True), _member(_ME, True)], {str(_ALPHA)}, 0),  # another bot present
    ([_member(_ALPHA), _member(_STRANGER), _member(_ME, True)], {str(_ALPHA)}, 0),    # a second human present
    ([_member(_STRANGER), _member(_ME, True)], {str(_ALPHA)}, 0),                     # sole member not allowlisted
    # Role-only seat (empty user allowlist): the receiver cannot read roles, so it infers only onto
    # a sole non-bot member and the listen loop's role check decides.
    ([_member(_ALPHA), _member(_ME, True)], set(), _ALPHA),
    ([_member(_ALPHA), _member(_BETA_BOT, True), _member(_ME, True)], set(), 0),
    ([_member(_BETA_BOT, True), _member(_ME, True)], set(), 0),
])
def test_inference_needs_exactly_one_non_self_member_that_is_allowlisted(members, allowed, expected):
    assert _infer(_receiver(members, allowed)) == expected


def test_inferring_the_sole_member_as_a_bot_drops_its_ssrc():
    receiver = _receiver([_member(_BETA_BOT, True), _member(_ME, True)], set())
    assert _infer(receiver) == 0
    assert _SSRC_BETA in receiver._dropped_ssrcs


# --- the packet-level drop set -------------------------------------------------------------------

def _rtp(ssrc: int) -> bytes:
    return struct.pack(">BBHII", 0x80, 0x78, 1, 1000, ssrc) + b"e" * 16 + b"nonc"


def _started(receiver: VoiceReceiver) -> VoiceReceiver:
    receiver._running = True
    return receiver


@pytest.mark.parametrize("bot_flag,fleet", [(True, ()), (False, (_BETA_BOT,))])
def test_a_bot_mapped_ssrc_is_dropped_before_decrypt(bot_flag, fleet):
    """Member.bot, or a fleet bot id when the member cache misses it."""
    members = [_member(_ALPHA), _member(_BETA_BOT, bot_flag), _member(_ME, True)]
    receiver = _started(_receiver(members, {str(_ALPHA)}, fleet=fleet))
    receiver.map_ssrc(_SSRC_BETA, _BETA_BOT)
    receiver.map_ssrc(_SSRC_ALPHA, _ALPHA)
    receiver._on_packet(_rtp(_SSRC_BETA))
    assert receiver._bot_audio_dropped == 1
    assert receiver._decode_failed == 0          # never reached the NaCl decrypt
    receiver._on_packet(_rtp(_SSRC_ALPHA))        # control: a person's frame reaches decrypt
    assert receiver._decode_failed == 1


def test_bot_audio_is_kept_when_the_seat_admits_bot_speech():
    receiver = _started(_receiver([_member(_BETA_BOT, True), _member(_ME, True)], set(), drop_bots=False))
    receiver.map_ssrc(_SSRC_BETA, _BETA_BOT)
    receiver._on_packet(_rtp(_SSRC_BETA))
    assert receiver._bot_audio_dropped == 0
    assert receiver._decode_failed == 1


def test_a_reused_ssrc_remapped_to_a_person_is_heard_again():
    receiver = _receiver([_member(_ALPHA), _member(_BETA_BOT, True), _member(_ME, True)], {str(_ALPHA)})
    receiver.map_ssrc(_SSRC_BETA, _BETA_BOT)
    receiver.map_ssrc(_SSRC_BETA, _ALPHA)
    assert _SSRC_BETA not in receiver._dropped_ssrcs


def test_a_continuous_unmapped_sender_cannot_grow_its_buffer_without_bound():
    """A sibling's mixer sends frames continuously, so silence never ends its 'utterance'."""
    receiver = _receiver([_member(_ALPHA), _member(_BETA_BOT, True), _member(_ME, True)], {str(_ALPHA)})
    frame = b"\0" * 3840  # 20 ms of 48 kHz stereo s16
    for _ in range(int(receiver.MAX_UNMAPPED_SECONDS * 50) + 100):
        with receiver._lock:
            receiver._buffers[_SSRC_BETA].extend(frame)
            receiver._last_packet_time[_SSRC_BETA] = time.monotonic()
        assert receiver.check_silence() == []
    cap = receiver.MAX_UNMAPPED_SECONDS * receiver.SAMPLE_RATE * receiver.CHANNELS * 2
    assert len(receiver._buffers.get(_SSRC_BETA, b"")) <= cap


# --- the listen loop and the leave flush ---------------------------------------------------------

class _Batch:
    def __init__(self) -> None:
        self._running = True

    def check_silence(self):
        self._running = False
        return [(_BETA_BOT, b"\0" * 9600), (_ALPHA, b"\0" * 9600)]

    def flush_pending(self):
        return [(_BETA_BOT, b"\0" * 9600), (_ALPHA, b"\0" * 9600)]

    def stop(self):
        pass


def _adapter(monkeypatch, allow_bots=None, inline=True) -> DiscordAdapter:
    for var in ("DISCORD_ALLOW_BOTS", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"):
        monkeypatch.delenv(var, raising=False)
    if allow_bots:
        monkeypatch.setenv("DISCORD_ALLOW_BOTS", allow_bots)
    if not inline:
        monkeypatch.setenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    members = {_ALPHA: _member(_ALPHA), _BETA_BOT: _member(_BETA_BOT, True)}
    guild = SimpleNamespace(id=_GUILD, get_member=members.get)
    adapter._client = SimpleNamespace(get_guild=lambda _id: guild, get_channel=lambda _id: None,
                                      user=SimpleNamespace(id=_ME))
    adapter._voice_text_channels = {_GUILD: 700}
    adapter._voice_receivers[_GUILD] = _Batch()
    # Every speaker passes the user allowlist: only the bot rule can refuse one.
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter._reset_voice_timeout = MagicMock()
    adapter._process_voice_input = AsyncMock()
    return adapter


def _processed(adapter: DiscordAdapter) -> list[int]:
    return [call.args[1] for call in adapter._process_voice_input.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_bots,inline,expected", [
    (None, True, [_ALPHA]),
    ("all", True, [_ALPHA]),               # "all" still needs a typed mention, which speech lacks
    ("all", False, [_BETA_BOT, _ALPHA]),   # the seat admits bot speech: STT runs for it
])
async def test_listen_loop_drops_bot_utterances_before_stt(monkeypatch, allow_bots, inline, expected):
    adapter = _adapter(monkeypatch, allow_bots, inline)
    await adapter._voice_listen_loop(_GUILD)
    assert _processed(adapter) == expected


@pytest.mark.asyncio
async def test_listen_loop_drops_a_fleet_bot_the_member_cache_misses(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter._client.get_guild(_GUILD).get_member = {_ALPHA: _member(_ALPHA)}.get
    adapter.gateway_runner = SimpleNamespace(_fleet_discord_bots=lambda: {_BETA_BOT: "beta"})
    await adapter._voice_listen_loop(_GUILD)
    assert _processed(adapter) == [_ALPHA]


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_bots,inline,expected", [
    (None, True, [_ALPHA]),
    ("all", False, [_BETA_BOT, _ALPHA]),
])
async def test_leave_flush_drops_bot_utterances_before_stt(monkeypatch, allow_bots, inline, expected):
    adapter = _adapter(monkeypatch, allow_bots, inline)
    await adapter.leave_voice_channel(_GUILD)
    assert _processed(adapter) == expected


def test_runner_lists_every_in_process_seat_bot():
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    alpha = SimpleNamespace(_client=SimpleNamespace(user=SimpleNamespace(id=_ME)), _owner_profile=None)
    beta = SimpleNamespace(_client=SimpleNamespace(user=SimpleNamespace(id=_BETA_BOT)), _owner_profile="beta")
    gone = SimpleNamespace(_client=None, _owner_profile="gamma")
    runner.adapters = {Platform.DISCORD: alpha}
    runner._profile_adapters = {"beta": {Platform.DISCORD: beta}, "gamma": {Platform.DISCORD: gone}}
    assert runner._fleet_discord_bots() == {_ME: "default", _BETA_BOT: "beta"}
