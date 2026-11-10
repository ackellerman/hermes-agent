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

import asyncio
import logging
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


@pytest.fixture(autouse=True)
def fake_davey(monkeypatch):
    """davey ships with the voice extra, not the test env; only ``MediaType.audio`` is needed."""
    mod = types.ModuleType("davey")
    mod.MediaType = SimpleNamespace(audio="audio")
    monkeypatch.setitem(sys.modules, "davey", mod)
    return mod


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
    receiver.start()  # resolves the credential tuple from the connection, as the real join does
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


class _InPlaceDave:
    """A live session that re-keys itself, the way discord.py 2.7.1 does.

    ``VoiceConnectionState.reinit_dave_session`` calls ``.reinit(...)`` on the *existing* object
    (a new object only when the attribute was None), and an MLS commit (binary op 28/29/30)
    advances the epoch without touching either ``dave_session`` or ``dave_protocol_version``.
    """

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.epoch = 1
        self.calls: list[tuple[int, bytes]] = []

    def reinit(self, epoch: int) -> None:
        self.epoch = epoch

    def decrypt(self, user_id: int, media_type, data: bytes) -> bytes:
        self.calls.append((user_id, bytes(data)))
        return data


def _drive_hook(receiver: VoiceReceiver, op: int, data: dict | None = None) -> None:
    """Deliver one voice-ws op through the receiver's installed hook, as discord.py does.

    ``DiscordVoiceWebSocket.received_message`` dispatches the op (updating ``conn``) and only
    then calls the hook, so the fake connection is already in its post-op state here.
    """
    conn = receiver._vc._connection
    receiver._install_speaking_hook(conn)
    asyncio.run(conn.hook(None, {"op": op, "d": data or {}}))


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


# --- the arrival stamps follow the buffers they describe -----------------------------------------

def test_a_ssrc_handed_from_a_bot_to_a_person_keeps_the_whole_post_tone_segment(nacl, decoders):
    """An SSRC Discord reassigns is never cut with the previous owner's stamps.

    Frames land before SPEAKING claims the SSRC; the mapping then says it was a bot's, so the
    buffer is dropped — its arrival deque must go with it, or the ready cut's byte prefix deletes
    the first frames the person speaks after the tone (integration 2a / e6 1).
    """
    receiver = _receiver(members=[_member(_OPERATOR), _member(_OTHER_BOT, True), _member(_ME, True)],
                         allowed={str(_OPERATOR)})
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"p")
    _packet(nacl, receiver, _SSRC_PERSON, seq=2, payload=b"p")
    assert len(receiver._buffers[_SSRC_PERSON]) == 2 * BYTES_PER_FRAME

    receiver.map_ssrc(_SSRC_PERSON, _OTHER_BOT)          # SPEAKING: it was a bot's all along
    assert _SSRC_PERSON not in receiver._buffers

    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)           # Discord hands the SSRC to a person
    receiver.mark_ready()                                # the tone has played
    for seq in range(3, 33):                             # 0.6 s of speech, after the tone
        _packet(nacl, receiver, _SSRC_PERSON, seq=seq, payload=b"p")
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic() - 3.0

    [segment] = receiver.check_silence()
    assert segment[0] == _OPERATOR
    assert len(segment[1]) == 30 * BYTES_PER_FRAME, "the stale prefix ate the first 40 ms"
    assert receiver._pre_ready_dropped == 0


def test_a_buffer_dropped_at_the_unmapped_cap_takes_its_stamps_with_it(nacl, decoders):
    """The 15 s cap drops the buffer and its arrival deque together.

    Marked ready first on purpose: the cap is the dropper here, not the pre-tone cut.  Left
    behind, a 15 s prefix of stamps would describe bytes that no longer exist (integration 2b).
    """
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)])
    receiver.mark_ready()
    frames = int(VoiceReceiver.MAX_UNMAPPED_SECONDS / FRAME_SECONDS) + 5   # 15.1 s, never pauses
    for seq in range(1, frames + 1):
        _packet(nacl, receiver, _SSRC_PERSON, seq=seq, payload=b"p")
    assert len(receiver._buffers[_SSRC_PERSON]) == frames * BYTES_PER_FRAME

    assert receiver.check_silence() == []                # nobody claims it, so the cap drops it
    assert _SSRC_PERSON not in receiver._buffers
    assert not receiver._arrivals.get(_SSRC_PERSON)

    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)           # the next utterance is attributable
    for seq in range(1, 31):                             # 0.6 s, transcribed whole
        _packet(nacl, receiver, _SSRC_PERSON, seq=seq, payload=b"p")
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic() - 3.0

    [segment] = receiver.check_silence()
    assert segment[0] == _OPERATOR
    assert len(segment[1]) == 30 * BYTES_PER_FRAME


# --- held audio cannot outlive the state that captured it -----------------------------------------

def _two_ring_receiver() -> VoiceReceiver:
    """A DAVE receiver with no inferable sender: any unmapped SSRC is held."""
    return _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                     dave_session=_RecordingDave(), dave_pver=1)


def test_discard_pending_drops_the_hold_ring(nacl, decoders):
    """A text-binding move already drops buffered PCM; held DAVE audio is captured speech too, and
    replaying it after the move transcribes the old channel's utterance into the new one."""
    dave = _RecordingDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    _packet(nacl, receiver, _SSRC_PERSON, seq=2, payload=b"two")
    assert len(receiver._dave_pending[_SSRC_PERSON]) == 2

    receiver.discard_pending()  # what adapter_voice_info._bind_voice_text_channel calls

    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 2  # counted, not silently lost
    assert dave.calls == []
    assert _decoded(decoders) == []


def test_the_drop_helper_reports_the_frames_it_dropped(nacl):
    """e6 calls this helper for its pre-ready cut; the return value is what it reports."""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    _packet(nacl, receiver, _SSRC_OPERATOR, seq=2, payload=b"two")

    assert receiver._drop_held_voice_payloads("pre-ready cut") == 2
    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 2
    assert receiver._drop_held_voice_payloads("pre-ready cut") == 0  # idempotent


def test_a_replaced_dave_session_drops_frames_held_under_the_old_one(nacl, decoders):
    """A session REPLACED by a new object is the one shape the credential tuple can see; the
    comparison is kept as belt-and-braces for it.  (discord.py 2.7.1's ``reinit_dave_session``
    does not swap the object while one exists — it ``reinit()``s it in place, which is what the
    ws hook's ``dave_state_changed`` trigger covers; see the in-place test below.)"""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"old-epoch")
    assert len(receiver._dave_pending[_SSRC_PERSON]) == 1

    receiver._vc._connection.dave_session = _RecordingDave()
    receiver.refresh_credentials("voice ws op 24")

    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 1
    assert _decoded(decoders) == []


def test_a_protocol_transition_on_the_same_session_drops_the_ring(nacl):
    """op 22 downgrading the protocol 1→0 keeps the session object and moves
    ``dave_protocol_version``: the replay would decrypt under plaintext rules, so the tuple's
    version check drops and counts the held frames."""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"old-epoch")

    receiver._vc._connection.dave_protocol_version = 0
    receiver.refresh_credentials("voice ws op 22")

    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 1


def test_an_in_place_reinit_drops_the_ring_through_the_trigger(nacl, decoders):
    """The epoch advance the tuple cannot see: discord.py re-keys DAVE with
    ``DaveSession.reinit(...)`` on the *existing* object, and an MLS commit (binary op 28/29/30)
    moves the epoch without touching ``dave_session`` or ``dave_protocol_version`` at all.  The
    ws hook knows the trigger, so it passes ``dave_state_changed``; the held frames were captured
    under the dead ratchet and must be dropped and counted rather than replayed."""
    dave = _InPlaceDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"old-epoch")
    assert len(receiver._dave_pending[_SSRC_PERSON]) == 1

    dave.reinit(2)
    receiver.refresh_credentials("voice ws op 4", dave_state_changed=True)

    # Nothing the tuple compares moved — the trigger, not a state diff, is what dropped the ring.
    assert receiver._dave_session is dave
    assert receiver._dave_protocol_version == 1
    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 1
    assert dave.calls == []
    assert _decoded(decoders) == []


@pytest.mark.parametrize("op", [21, 22, 24])
def test_the_ws_hook_flags_every_dave_trigger_op(nacl, decoders, op):
    """The flag is wired where the trigger is: op 21/22/24 each arrive with the session object and
    the protocol version unchanged (discord.py reinit()s in place, MLS commits never touch
    ``conn``), so a state-diff predicate would keep the ring forever.  Through the hook, each one
    drops and counts it."""
    dave = _InPlaceDave()
    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=dave, dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"old-epoch")
    assert len(receiver._dave_pending[_SSRC_PERSON]) == 1

    dave.reinit(2)
    _drive_hook(receiver, op, {"protocol_version": 1})

    assert receiver._dave_session is dave
    assert receiver._dave_protocol_version == 1
    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 1
    assert dave.calls == []
    assert _decoded(decoders) == []


def test_a_refresh_that_keeps_the_session_keeps_the_ring(nacl, decoders):
    """op 4 on a live ratchet with no DAVE state change (a transport-key-only rekey, and the
    packet path's decrypt-failure-streak refresh) must not throw held audio away."""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")

    receiver._vc._connection.secret_key = b"\x11" * 32
    receiver.refresh_credentials("voice ws op 4")
    receiver.refresh_credentials("decrypt-failure streak")

    assert [payload for _at, payload in receiver._dave_pending[_SSRC_PERSON]] == [b"one"]
    assert receiver._dave_unmapped_dropped == 0


def test_the_commit_that_finally_settles_dave_keeps_the_ring_for_the_ready_cut(nacl, decoders):
    """A session that never reported ``ready`` established no ratchet, so nothing it held is tied
    to a dead one.  The swap that finally makes DAVE ready is the join's own op-22 commit: it
    passes ``dave_state_changed=True`` like every other DAVE trigger, but the ratchet guard keeps
    the ring — the held frames belong to the deaf window, and the ready cut drops them by ARRIVAL
    time when the ring replays.  Purging them here would lose them uncounted by
    ``pre_ready_dropped``."""
    class _Settling:
        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def decrypt(self, user_id, media_type, data: bytes) -> bytes:
            return data

    receiver = _receiver(members=[_member(_OPERATOR), _member(_STRANGER), _member(_ME, True)],
                         dave_session=_Settling(ready=False), dave_pver=1)
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"pre-tone")
    assert len(receiver._dave_pending[_SSRC_PERSON]) == 1

    receiver._vc._connection.dave_session = _Settling(ready=True)
    receiver.refresh_credentials("voice ws op 22", dave_state_changed=True)

    assert len(receiver._dave_pending[_SSRC_PERSON]) == 1
    assert receiver._dave_unmapped_dropped == 0
    assert _decoded(decoders) == []


def test_the_ttl_sweep_counts_and_logs_what_it_expires(nacl, decoders, caplog):
    """TTL expiry is the ordinary outcome when SPEAKING never arrives: it must be counted and named,
    or the teardown line reads ``dave_unmapped_dropped=0`` while an utterance was lost."""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    _packet(nacl, receiver, _SSRC_PERSON, seq=2, payload=b"two")
    _packet(nacl, receiver, _SSRC_OPERATOR, seq=3, payload=b"three")
    for ssrc in (_SSRC_PERSON, _SSRC_OPERATOR):
        held_at, payload = receiver._dave_pending[ssrc][-1]
        receiver._dave_pending[ssrc][-1] = (
            held_at - VoiceReceiver.DAVE_PENDING_TTL - 1.0, payload)

    with caplog.at_level(logging.INFO):
        assert receiver.check_silence() == []

    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 3
    assert "frames=3" in caplog.text
    assert "ssrcs=2" in caplog.text
    assert _decoded(decoders) == []


def test_the_ttl_sweep_does_not_name_an_empty_ring_as_a_dropped_ssrc(nacl, caplog):
    """An empty ring yields no frame, so the sweep's SSRC count must not claim audio was dropped
    for it — the log named more SSRCs than it dropped."""
    receiver = _two_ring_receiver()
    _packet(nacl, receiver, _SSRC_PERSON, seq=1, payload=b"one")
    _packet(nacl, receiver, _SSRC_OPERATOR, seq=2, payload=b"two")
    receiver._dave_pending[_SSRC_OPERATOR].clear()  # a ring that yielded no frame
    for ssrc in (_SSRC_PERSON, _SSRC_OPERATOR):
        if receiver._dave_pending[ssrc]:
            held_at, payload = receiver._dave_pending[ssrc][-1]
            receiver._dave_pending[ssrc][-1] = (
                held_at - VoiceReceiver.DAVE_PENDING_TTL - 1.0, payload)

    with caplog.at_level(logging.INFO):
        assert receiver.check_silence() == []

    assert receiver._dave_pending == {}
    assert receiver._dave_unmapped_dropped == 1
    assert "frames=1" in caplog.text
    assert "ssrcs=1" in caplog.text


# --- the energy gate does not hold the lock -------------------------------------------------------

def _lock_free_gate(receiver, monkeypatch):
    """Install a gate probe that records whether ``_lock`` was free while it ran."""
    real_gate = VoiceReceiver._pcm_speech_activity
    lock_free: list[bool] = []

    def probe(pcm_data):
        got = receiver._lock.acquire(blocking=False)
        lock_free.append(got)
        if got:
            receiver._lock.release()
        return real_gate(pcm_data)

    monkeypatch.setattr(receiver, "_pcm_speech_activity", probe)
    return lock_free


def test_check_silence_classifies_outside_the_lock(monkeypatch):
    receiver = _receiver(members=[_member(_OPERATOR, True)], allowed={str(_OPERATOR)})
    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    pcm = _square_pcm(0.5, 0) + _square_pcm(0.1, 1000) + _square_pcm(0.4, 0)
    receiver._buffers[_SSRC_PERSON] = bytearray(pcm)
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic() - 3.0
    lock_free = _lock_free_gate(receiver, monkeypatch)

    assert receiver.check_silence() == [(_OPERATOR, pcm)]  # verdict unchanged
    assert lock_free == [True]  # ... and the gate ran with the lock free


def test_flush_pending_classifies_outside_the_lock(monkeypatch, caplog):
    receiver = _receiver(members=[_member(_OPERATOR, True)], allowed={str(_OPERATOR)})
    receiver.map_ssrc(_SSRC_PERSON, _OPERATOR)
    receiver._buffers[_SSRC_PERSON] = bytearray(_square_pcm(1.0, 30))
    receiver._last_packet_time[_SSRC_PERSON] = time.monotonic()
    lock_free = _lock_free_gate(receiver, monkeypatch)

    with caplog.at_level(logging.INFO):
        assert receiver.flush_pending() == []  # verdict unchanged: comfort noise still rejected

    assert lock_free == [True]
    assert "non-speech PCM segment" in caplog.text
    assert _SSRC_PERSON not in receiver._buffers
