"""Discord voice channel flow: join warm-up → utterance → STT handoff.

Re-homed from upstream's ``tests/integration/test_voice_channel_flow.py`` (that tree purged the
integration suite).  Runs the real ``VoiceReceiver`` over the fork's Discord edge fakes
(``tests/fakes/platforms/discord_voice_session.py``) with a scripted transport AEAD and Opus codec,
so the whole packet pipeline — RTP parse, NaCl, padding strip, DAVE, decode, buffer, silence flush —
executes without a Discord connection.

Deliberately NOT re-homed: the PR's inference cases.  Upstream's rule infers a sender while a second
bot is in the channel; this fork's C3 rule needs exactly one other member with bots COUNTED, so those
tests would assert the opposite of the fork's contract.  The fork's rule is pinned in
``tests/plugins/platforms/test_discord_voice_bot_exclusion.py`` and
``tests/gateway/test_voice_energy_gate_ring.py``.
"""
from __future__ import annotations

import struct
import sys
import time
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from plugins.platforms.discord.adapter import VoiceReceiver
from tests.fakes.platforms import discord_voice_session as fx

_ME, _OPERATOR, _SIBLING_BOT = 999, 101, 202
_SSRC_A, _SSRC_B, _SSRC_BOT = 5001, 5002, 5003

FRAME_SECONDS = 0.02
PCM_BYTES_PER_FRAME = int(VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2 * FRAME_SECONDS)


# --- the Discord edge and the two native libraries it needs --------------------------------------

class _FakeAead:
    """Transport AEAD: decrypt returns what the test last staged (one stage per packet)."""

    plaintext = b""
    failing = False

    def __init__(self, _key: bytes) -> None:
        pass

    def decrypt(self, _encrypted, _header, _nonce) -> bytes:
        if _FakeAead.failing:
            raise ValueError("decrypt failed (stale key)")
        return _FakeAead.plaintext


class _Codec:
    """Opus codec: maps a payload to the PCM the test wants it to "decode" to."""

    def __init__(self) -> None:
        self.pcm: dict[bytes, bytes] = {}
        self.calls: list[bytes] = []

    def decode(self, payload: bytes) -> bytes:
        self.calls.append(bytes(payload))
        return self.pcm.get(bytes(payload), b"\x00" * PCM_BYTES_PER_FRAME)


@pytest.fixture(autouse=True)
def fake_native(monkeypatch):
    """PyNaCl and davey ship with the voice extra, not the test env."""
    _FakeAead.plaintext, _FakeAead.failing = b"", False
    nacl_mod = types.ModuleType("nacl")
    secret_mod = types.ModuleType("nacl.secret")
    secret_mod.Aead = _FakeAead  # type: ignore[attr-defined]
    nacl_mod.secret = secret_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nacl", nacl_mod)
    monkeypatch.setitem(sys.modules, "nacl.secret", secret_mod)
    davey_mod = types.ModuleType("davey")
    davey_mod.MediaType = SimpleNamespace(audio="audio")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davey", davey_mod)


@pytest.fixture
def codec(monkeypatch) -> _Codec:
    codec = _Codec()

    class _Decoder:
        def decode(self, payload: bytes) -> bytes:
            return codec.decode(payload)

    monkeypatch.setattr("discord.opus.Decoder", _Decoder)
    return codec


class _RecordingDave:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bytes]] = []

    def decrypt(self, user_id: int, _media_type, data: bytes) -> bytes:
        self.calls.append((user_id, bytes(data)))
        return data


def _receiver(*, members, allowed=(), dave=False, pver=0) -> VoiceReceiver:
    adapter = fx.make_adapter(bot_user_id=_ME)
    channel = fx.voice_channel(adapter, members=list(members))
    vc = fx.FakeVoiceClient(adapter._client, channel)
    vc.user = adapter._client.user
    vc._connection = SimpleNamespace(
        secret_key=bytes(range(32)),
        dave_session=_RecordingDave() if dave else None,
        dave_protocol_version=pver,
        dave_downgraded=False,
        dave_pending_transitions={},
        ssrc=_ME,
        hook=None,
        ws=None,
        add_socket_listener=lambda _cb: None,
        remove_socket_listener=lambda _cb: None,
    )
    receiver = VoiceReceiver(vc, allowed_user_ids=set(allowed))
    receiver.start()
    return receiver


def _tone_pcm(frames: int = 25, amplitude: int = 3000) -> bytes:
    """Stereo s16le square wave; RMS equals the amplitude, so it clears the energy gate."""
    pair = amplitude.to_bytes(2, "little", signed=True) + (-amplitude).to_bytes(
        2, "little", signed=True)
    return (pair * ((PCM_BYTES_PER_FRAME // 2) * frames + 1))[:PCM_BYTES_PER_FRAME * frames]


def _rtp(ssrc: int, seq: int = 1, opus: bytes = b"opus", *, pad: bytes = b"",
         pad_bit: bool = False) -> bytes:
    header = struct.pack(">BBHII", 0xA0 if pad_bit else 0x80, 0x78, seq, 960 * seq, ssrc)
    return header + opus + pad + struct.pack(">I", seq)


def _send(receiver: VoiceReceiver, codec: _Codec, ssrc: int, payloads, *, pcm=None,
          pad: bytes = b"", pad_bit: bool = False) -> None:
    """Deliver payloads as consecutive RTP packets, staging their decoded PCM on the codec."""
    for seq, payload in enumerate(payloads, start=1):
        if pcm is not None:
            codec.pcm[bytes(payload)] = pcm
        _FakeAead.plaintext = payload + pad
        receiver._on_packet(_rtp(ssrc, seq=seq, opus=payload, pad=pad, pad_bit=pad_bit))


def _go_quiet(receiver: VoiceReceiver, ssrc: int, seconds: float = 2.0) -> None:
    receiver._last_packet_time[ssrc] = time.monotonic() - seconds


# --- the flow ------------------------------------------------------------------------------------

class TestJoinFlow:
    def test_join_warmup_silence_never_reaches_stt(self, codec):
        """Discord's Opus comfort frames decode to near-silence; STT must never see them."""
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)])
        receiver.map_ssrc(_SSRC_A, _OPERATOR)

        _send(receiver, codec, _SSRC_A, [b"warm"] * 40)  # 0.8 s, PCM = zeros
        _go_quiet(receiver, _SSRC_A)

        assert codec.calls  # the frames were decoded — the gate is what stops them
        assert receiver.check_silence() == []

    def test_first_utterance_after_join_reaches_stt_exactly_once(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)])
        receiver.map_ssrc(_SSRC_A, _OPERATOR)

        _send(receiver, codec, _SSRC_A, [b"warm"] * 20)
        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1))
        _go_quiet(receiver, _SSRC_A)

        completed = receiver.check_silence()

        assert len(completed) == 1
        user_id, pcm = completed[0]
        assert user_id == _OPERATOR
        assert len(pcm) == 45 * PCM_BYTES_PER_FRAME  # warm-up + tone, both flushed
        assert pcm[-25 * PCM_BYTES_PER_FRAME:] == _tone_pcm(1) * 25
        assert pcm[:20 * PCM_BYTES_PER_FRAME] == b"\x00" * 20 * PCM_BYTES_PER_FRAME

    def test_utterance_without_speaking_event_is_inferred_for_a_sole_human(self, codec):
        """C3: one other member (the bot does not count as a second human) → infer."""
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)],
                             allowed={str(_OPERATOR)})

        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1))
        _go_quiet(receiver, _SSRC_A)

        assert [uid for uid, _pcm in receiver.check_silence()] == [_OPERATOR]

    def test_speaking_event_maps_and_audio_flows(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_SIBLING_BOT, None),
                                      fx.member(_ME, None)])

        receiver.map_ssrc(_SSRC_A, _OPERATOR)
        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1))
        _go_quiet(receiver, _SSRC_A)

        assert [uid for uid, _pcm in receiver.check_silence()] == [_OPERATOR]


class TestDaveFlow:
    def test_unresolved_sender_is_held_then_replayed_in_order(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_SIBLING_BOT, None),
                                      fx.member(_ME, None)],
                             dave=True, pver=1)

        _send(receiver, codec, _SSRC_A, [b"a", b"b"])
        assert codec.calls == []  # never decoded while unattributable

        receiver.map_ssrc(_SSRC_A, _OPERATOR)
        _send(receiver, codec, _SSRC_A, [b"c"])

        assert codec.calls == [b"a", b"b", b"c"]  # replayed in arrival order, then the live frame
        assert len(receiver._buffers[_SSRC_A]) == 3 * PCM_BYTES_PER_FRAME
        assert _SSRC_A not in receiver._dave_pending

    def test_sibling_bot_audio_never_leaves_the_ring(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None),
                                      SimpleNamespace(id=_SIBLING_BOT, bot=True, display_name="sib"),
                                      fx.member(_ME, None)],
                             allowed={str(_OPERATOR)}, dave=True, pver=1)

        _send(receiver, codec, _SSRC_BOT, [b"bot"])
        assert len(receiver._dave_pending[_SSRC_BOT]) == 1

        receiver.map_ssrc(_SSRC_BOT, _SIBLING_BOT)  # SPEAKING identifies the sibling bot

        assert _SSRC_BOT not in receiver._dave_pending
        assert receiver._bot_audio_dropped == 1
        assert codec.calls == []


class TestPacketShaping:
    def test_rtp_padding_is_stripped_before_decode(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)])
        receiver.map_ssrc(_SSRC_A, _OPERATOR)

        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1),
              pad=b"\x00\x00\x00\x04", pad_bit=True)  # RFC 3550: 4 bytes, last counts itself

        assert codec.calls == [b"tone"] * 25  # the 4 padding bytes never reach the codec

    @pytest.mark.parametrize("pad", [b"\x00\x00\x00", b"\x00\x07"], ids=["zero", "overflow"])
    def test_invalid_padding_drops_the_packet(self, codec, pad):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)])
        receiver.map_ssrc(_SSRC_A, _OPERATOR)

        _send(receiver, codec, _SSRC_A, [b"tone"], pad=pad, pad_bit=True)

        assert codec.calls == []
        assert _SSRC_A not in receiver._buffers or len(receiver._buffers[_SSRC_A]) == 0


class TestLifecycle:
    def test_pause_blocks_audio_during_playback(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)])
        receiver.map_ssrc(_SSRC_A, _OPERATOR)

        receiver.pause()
        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1))
        assert codec.calls == []

        receiver.resume()
        _send(receiver, codec, _SSRC_A, [b"tone"] * 25, pcm=_tone_pcm(1))
        _go_quiet(receiver, _SSRC_A)
        assert len(receiver.check_silence()) == 1

    def test_stop_clears_everything(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_SIBLING_BOT, None),
                                      fx.member(_ME, None)],
                             dave=True, pver=1)
        _send(receiver, codec, _SSRC_A, [b"held"])  # unresolved → held
        receiver.map_ssrc(_SSRC_B, _OPERATOR)
        _send(receiver, codec, _SSRC_B, [b"tone"] * 25, pcm=_tone_pcm(1))

        receiver.stop()

        assert receiver._buffers == {}
        assert receiver._decoders == {}
        assert receiver._dave_pending == {}
        assert receiver._ssrc_to_user == {}

    def test_rejoin_automaps_a_new_ssrc(self, codec):
        receiver = _receiver(members=[fx.member(_OPERATOR, None), fx.member(_ME, None)],
                             allowed={str(_OPERATOR)})
        _send(receiver, codec, _SSRC_B, [b"tone"] * 25, pcm=_tone_pcm(1))
        _go_quiet(receiver, _SSRC_B)

        assert [uid for uid, _pcm in receiver.check_silence()] == [_OPERATOR]
        assert receiver._ssrc_to_user[_SSRC_B] == _OPERATOR
