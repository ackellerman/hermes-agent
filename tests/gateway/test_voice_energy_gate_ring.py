"""The Discord receiver's two STT guards: the sustained-energy gate and the DAVE hold ring.

A clip reaches STT only when it holds at least ``MIN_ACTIVE_SPEECH_DURATION`` (0.1 s) of
*contiguous* frames above ``ACTIVE_FRAME_RMS`` (about -50 dBFS).  Below that it is discarded, in the
silence flush and in the leave-time flush alike — otherwise Discord's join-time comfort frames
become invented text.

With DAVE active, a payload from a sender Discord has not attributed yet cannot be decrypted and
must not be Opus-decoded either (ciphertext decodes to shredded PCM that survives the energy gate
and poisons decoder state).  Those payloads are held per SSRC in a bounded ring and replayed in
order once SPEAKING — or the fork's own inference — resolves the sender.

The fork's C3 rule is part of the contract here: inference needs exactly one other member in the
channel, bots COUNTED, so an operator sharing the channel with another bot is not inferred onto.
Its unmapped SSRC stays in the ring until SPEAKING maps it.  And bot audio never leaves the ring:
an SSRC already in the drop set never enters it, and a held SSRC that resolves to a bot has its
frames discarded rather than replayed.
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

_ME, _OPERATOR, _OTHER_BOT, _STRANGER = 9001, 101, 202, 303
_SSRC_OPERATOR, _SSRC_BOT, _SSRC_PERSON = 5001, 5002, 5003

FRAME_SECONDS = 0.02
BYTES_PER_FRAME = int(VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2 * FRAME_SECONDS)


# --- fixtures ------------------------------------------------------------------------------------

class _FakeNacl:
    """Stands in for the transport AEAD: ``decrypt`` returns whatever the test last set."""

    def __init__(self) -> None:
        self.plaintext = b""

    def decrypt(self, encrypted: bytes, header: bytes, nonce: bytes) -> bytes:
        return self.plaintext


@pytest.fixture
def nacl(monkeypatch):
    """Route the adapter's in-function ``import nacl.secret`` to the fake (PyNaCl is an extra)."""
    aead = _FakeNacl()
    nacl_mod = types.ModuleType("nacl")
    secret_mod = types.ModuleType("nacl.secret")
    secret_mod.Aead = lambda _key: aead  # type: ignore[attr-defined]
    nacl_mod.secret = secret_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nacl", nacl_mod)
    monkeypatch.setitem(sys.modules, "nacl.secret", secret_mod)
    return aead


@pytest.fixture
def decoders(monkeypatch):
    """Opus decoders that record the payloads they were handed, so replay order is observable."""
    instances: list = []

    class _Decoder:
        def __init__(self) -> None:
            self.decoded: list[bytes] = []
            instances.append(self)

        def decode(self, payload: bytes) -> bytes:
            self.decoded.append(bytes(payload))
            return bytes(payload) * BYTES_PER_FRAME

    monkeypatch.setattr("discord.opus.Decoder", _Decoder)
    return instances


def _decoded(decoders) -> list[list[bytes]]:
    return [d.decoded for d in decoders]


def _member(uid: int, bot: bool = False) -> SimpleNamespace:
    return SimpleNamespace(id=uid, bot=bot, display_name=f"m{uid}")


def _conn(dave_session=None, dave_pver=0):
    return SimpleNamespace(
        secret_key=bytes(range(32)), dave_session=dave_session, dave_protocol_version=dave_pver,
        dave_downgraded=False, dave_pending_transitions={}, ssrc=_ME, hook=None, ws=None,
        add_socket_listener=lambda cb: None, remove_socket_listener=lambda cb: None,
    )


def _receiver(*, members=(), allowed=(), dave_session=None, dave_pver=0, fleet=()) -> VoiceReceiver:
    vc = SimpleNamespace(_connection=_conn(dave_session, dave_pver),
                         channel=SimpleNamespace(members=list(members)),
                         user=SimpleNamespace(id=_ME))
    receiver = VoiceReceiver(vc, allowed_user_ids=set(allowed), fleet_bot_ids=lambda: set(fleet))
    receiver._running = True
    return receiver


def _square_pcm(duration_s: float, amplitude: int) -> bytes:
    """48 kHz stereo s16le square wave: RMS equals ``amplitude`` exactly."""
    samples = int(VoiceReceiver.SAMPLE_RATE * duration_s) * VoiceReceiver.CHANNELS
    value = int(amplitude).to_bytes(2, "little", signed=True)
    return value * samples


class _RecordingDave:
    """A DAVE session that records the sender it was asked about and returns the payload."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bytes]] = []

    def decrypt(self, user_id: int, media_type, data: bytes) -> bytes:
        self.calls.append((user_id, bytes(data)))
        return data


def _packet(nacl: _FakeNacl, receiver: VoiceReceiver, ssrc: int, seq: int = 1,
            payload: bytes = b"p") -> None:
    nacl.plaintext = payload
    header = struct.pack(">BBHII", 0x80, 0x78, seq, 1000 + seq, ssrc)
    receiver._on_packet(header + payload + b"nonc")


# --- the energy gate -----------------------------------------------------------------------------

def test_pcm_speech_activity_requires_contiguous_frames_above_the_floor():
    assert VoiceReceiver.ACTIVE_FRAME_RMS == 100
    assert VoiceReceiver.MIN_ACTIVE_SPEECH_DURATION == 0.1

    loud = _square_pcm(0.6, 1000)
    assert VoiceReceiver._pcm_speech_activity(loud)[0] is True

    comfort_noise = _square_pcm(0.6, 40)
    assert VoiceReceiver._pcm_speech_activity(comfort_noise)[0] is False

    # One 20 ms pop, however loud, is not sustained speech.
    click = _square_pcm(FRAME_SECONDS, 30000) + _square_pcm(0.58, 0)
    assert VoiceReceiver._pcm_speech_activity(click)[0] is False

    # 5 contiguous active frames = exactly 0.1 s: the floor of the gate.
    five = _square_pcm(5 * FRAME_SECONDS, 100) + _square_pcm(0.5, 0)
    assert VoiceReceiver._pcm_speech_activity(five)[0] is True


def test_near_silent_clip_never_reaches_stt():
    receiver = _receiver(members=[_member(_OPERATOR, True)], allowed={str(_OPERATOR)})
    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    receiver._buffers[_SSRC_PERSON] = bytearray(_square_pcm(1.0, 20))
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic() - 3.0

    assert receiver.check_silence() == []
    assert len(receiver._buffers[_SSRC_PERSON]) == 0


def test_clip_with_a_tenth_of_a_second_of_speech_passes():
    receiver = _receiver(members=[_member(_OPERATOR, True)], allowed={str(_OPERATOR)})
    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    pcm = _square_pcm(0.5, 0) + _square_pcm(0.1, 1000) + _square_pcm(0.4, 0)
    receiver._buffers[_SSRC_PERSON] = bytearray(pcm)
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic() - 3.0

    assert receiver.check_silence() == [(_OPERATOR, pcm)]


def test_leave_flush_applies_the_same_gate():
    receiver = _receiver(members=[_member(_OPERATOR, True)], allowed={str(_OPERATOR)})
    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    receiver._buffers[_SSRC_PERSON] = bytearray(_square_pcm(1.0, 30))
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic()

    assert receiver.flush_pending() == []
    assert _SSRC_PERSON not in receiver._buffers


# --- the DAVE hold ring --------------------------------------------------------------------------

def test_unmapped_dave_frame_is_held_then_replayed_in_order(nacl, decoders):
    dave = _RecordingDave()
    # Two humans: inference cannot place an unknown SSRC, so the ring does the work.
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)

    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    _packet(nacl, receiver, _SSRC_PERSON, seq=2, payload=b"two")
    assert [payload for _at, payload in receiver._dave_pending[_SSRC_PERSON]] == [b"one", b"two"]
    assert _SSRC_PERSON not in receiver._decoders
    assert dave.calls == []

    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    _packet(nacl, receiver, _SSRC_PERSON, seq=3, payload=b"three")

    assert [payload for _uid, payload in dave.calls] == [b"one", b"two", b"three"]
    assert all(uid == _OPERATOR for uid, _payload in dave.calls)
    assert _decoded(decoders) == [[b"one", b"two", b"three"]]
    assert _SSRC_PERSON not in receiver._dave_pending


def test_the_ring_is_bounded_and_counts_what_it_evicts(nacl, decoders):
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    overflow = VoiceReceiver.DAVE_PENDING_MAX_FRAMES + 7

    for seq in range(1, overflow + 1):
        _packet(nacl, receiver, _SSRC_PERSON, seq=seq, payload=b"f")

    assert len(receiver._dave_pending[_SSRC_PERSON]) == VoiceReceiver.DAVE_PENDING_MAX_FRAMES
    assert receiver._dave_unmapped_dropped == 7
    assert dave.calls == []
    assert _decoded(decoders) == []


def test_a_quiet_unresolved_ssrc_releases_its_ring(nacl, decoders):
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    held_at, payload = receiver._dave_pending[_SSRC_PERSON][-1]
    receiver._dave_pending[_SSRC_PERSON][-1] = (
        held_at - VoiceReceiver.DAVE_PENDING_TTL - 1.0, payload)

    receiver.check_silence()

    assert _SSRC_PERSON not in receiver._dave_pending


def test_stop_clears_the_ring(nacl, decoders):
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1)

    receiver.stop()

    assert receiver._dave_pending == {}


def test_passthrough_window_still_decodes_plaintext(nacl, decoders):
    """Inside an open passthrough window an unmapped sender's bytes are Opus, not ciphertext."""
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=_RecordingDave(), dave_pver=1)
    receiver.note_dave_passthrough_window(30.0)

    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"plain")

    assert _SSRC_PERSON not in receiver._dave_pending
    assert _decoded(decoders) == [[b"plain"]]


# --- C3: inference is not loosened by the ring ----------------------------------------------------

def test_operator_plus_another_bot_is_not_inferred_but_held(nacl, decoders):
    """C3: bots count.  With a sibling bot in the channel an unmapped SSRC is not the operator."""
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_OTHER_BOT, True), _member(_ME, True)],
                         allowed={str(_OPERATOR)}, dave_session=dave, dave_pver=1)

    _packet(nacl, receiver, _SSRC_OPERATOR, seq=1, payload=b"one")

    assert _SSRC_OPERATOR not in receiver._ssrc_to_user
    assert dave.calls == []
    assert [payload for _at, payload in receiver._dave_pending[_SSRC_OPERATOR]] == [b"one"]

    # SPEAKING maps it: the held frame is decrypted and replayed for the operator.
    receiver.map_ssrc(_SSRC_OPERATOR, _OPERATOR)
    _packet(nacl, receiver, _SSRC_OPERATOR, seq=2, payload=b"two")
    assert [uid for uid, _payload in dave.calls] == [_OPERATOR, _OPERATOR]


def test_sole_operator_with_the_bot_alone_still_infers(nacl):
    """Control: with no second member, the fork's inference still places the sender."""
    receiver = _receiver(members=[_member(_OPERATOR), _member(_ME, True)],
                         allowed={str(_OPERATOR)}, dave_session=_RecordingDave(), dave_pver=1)
    _packet(nacl, receiver, _SSRC_OPERATOR, seq=1, payload=b"one")
    assert receiver._ssrc_to_user[_SSRC_OPERATOR] == _OPERATOR


# --- bot audio never leaves the ring --------------------------------------------------------------

def test_a_frame_from_a_dropped_ssrc_never_enters_the_ring(nacl, decoders):
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_OTHER_BOT, True), _member(_ME, True)],
                         allowed={str(_OPERATOR)}, dave_session=dave, dave_pver=1)
    receiver.map_ssrc(_SSRC_BOT, _OTHER_BOT)
    assert _SSRC_BOT in receiver._dropped_ssrcs

    _packet(nacl, receiver, _SSRC_BOT, seq=1, payload=b"bot")

    assert _SSRC_BOT not in receiver._dave_pending
    assert dave.calls == []
    assert receiver._bot_audio_dropped == 1


def test_frames_held_for_an_ssrc_that_turns_out_to_be_a_bot_are_discarded(nacl, decoders):
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_OTHER_BOT, True), _member(_ME, True)],
                         allowed={str(_OPERATOR)}, dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_BOT, seq=1, payload=b"bot-1")
    _packet(nacl, receiver, _SSRC_BOT, seq=2, payload=b"bot-2")
    assert len(receiver._dave_pending[_SSRC_BOT]) == 2

    receiver.map_ssrc(_SSRC_BOT, _OTHER_BOT)

    assert _SSRC_BOT not in receiver._dave_pending
    assert receiver._bot_audio_dropped == 2
    assert dave.calls == []
    assert _decoded(decoders) == []
