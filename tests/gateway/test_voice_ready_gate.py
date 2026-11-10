"""The join ready gate (plan steps 7-11, D-015/D-016/D-017) and its pre-tone cut.

A join is deaf until the model is warm and DAVE has settled; then one tone marks ready, and every
packet that ARRIVED at or before the tone's end is discarded — including a frame the DAVE hold ring
only released afterwards. The gate is channel-level and once per join: a move into another room
re-arms it, a reconnect and a DAVE re-epoch do not.

These tests drive the REAL ``DiscordAdapter.join_voice_channel`` and the REAL ``VoiceReceiver``;
only discord.py objects and the STT lease's worker are fake. DAVE state is driven through the real
websocket hook (``receiver.start()`` wraps ``conn.hook``), so the ready predicate is exercised the
way discord.py drives it, not through a free-standing flag.
"""
from __future__ import annotations

import asyncio
import struct
import sys
import threading
import time
import types
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("discord")

from tests.fakes.platforms import discord_voice_session as fx  # (installs the discord fakes before the adapter)

from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter import VoiceReceiver
from tools import stt_lease

try:  # RED on a tree without element e6: the module the gate lives in does not exist yet.
    from plugins.platforms.discord import adapter_voice_ready as ready_mod
except ImportError:  # pragma: no cover - only the pre-e6 tree takes this branch
    ready_mod = None  # type: ignore[assignment]

GUILD, TEXT, VOICE_A, VOICE_B = fx.GUILD, fx.TEXT, fx.VOICE_A, fx.VOICE_B
ME, OPERATOR, STRANGER = 9000, 4242, 5150
SSRC = 5001
FRAME_SECONDS = VoiceReceiver.PCM_FRAME_DURATION
FRAME_BYTES = int(VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * 2 * FRAME_SECONDS)

_WARM_CPU = {"provider": "local", "warmed": True, "action": "cached", "device": "cpu", "model": "tiny"}
_WARM_CUDA = {**_WARM_CPU, "device": "cuda"}


# --- fixtures --------------------------------------------------------------------------------------


class _FakeNacl:
    """The transport AEAD: ``decrypt`` hands back whatever the test last set."""

    def __init__(self) -> None:
        self.plaintext = b""

    def decrypt(self, encrypted: bytes, header: bytes, nonce: bytes) -> bytes:
        return self.plaintext


@pytest.fixture
def nacl(monkeypatch):
    aead = _FakeNacl()
    nacl_mod, secret_mod = types.ModuleType("nacl"), types.ModuleType("nacl.secret")
    secret_mod.Aead = lambda _key: aead  # type: ignore[attr-defined]
    nacl_mod.secret = secret_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nacl", nacl_mod)
    monkeypatch.setitem(sys.modules, "nacl.secret", secret_mod)
    return aead


@pytest.fixture(autouse=True)
def fake_davey(monkeypatch):
    """davey ships with the voice extra; the receiver only needs ``MediaType.audio``."""
    mod = types.ModuleType("davey")
    mod.MediaType = SimpleNamespace(audio="audio")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davey", mod)
    return mod


@pytest.fixture(autouse=True)
def tone_home(monkeypatch, tmp_path):
    """Every rendered tone lands in the test's own cache dir, never the shared profile home."""
    if ready_mod is not None:
        monkeypatch.setattr(ready_mod, "get_hermes_home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def decoders(monkeypatch):
    """Opus decoders that return one loud frame per payload (the energy gate must pass)."""
    calls = {"n": 0}

    class _Decoder:
        def decode(self, payload: bytes) -> bytes:
            calls["n"] += 1
            return struct.pack("<h", 3000) * (FRAME_BYTES // 2)

    monkeypatch.setattr("discord.opus.Decoder", _Decoder)
    return calls


# --- discord.py objects a real join can drive -------------------------------------------------------


class _Daveless:
    """A DAVE session object with ``ready`` as discord.py's ``can_encrypt`` reads it."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def decrypt(self, user_id: int, media_type, data: bytes) -> bytes:
        return data


class _GateVoiceClient(fx.FakeVoiceClient):
    """``FakeVoiceClient`` plus the connection state and ``play()`` a real tone needs."""

    def __init__(self, client, channel) -> None:
        super().__init__(client, channel)
        self.played: list = []
        self._connection = SimpleNamespace(
            secret_key=bytes(range(32)),
            dave_session=getattr(channel, "dave_session", None),
            dave_protocol_version=getattr(channel, "pver", 0),
            dave_downgraded=False,
            dave_pending_transitions={},
            ssrc=ME,
            hook=None,
            ws=SimpleNamespace(_hook=None),
            add_socket_listener=lambda cb: None,
            remove_socket_listener=lambda cb: None,
        )

    def is_playing(self) -> bool:
        return False

    def play(self, source, after=None) -> None:
        self.played.append(source)
        if source is not None:
            while source.read():  # the encoder drains the clip: a natural EOF, not a stop
                pass
        if after:
            after(None)


class _GateChannel(fx.FakeVoiceChannel):
    def __init__(self, cid, guild, *, members=(), dave_session=None, pver=0) -> None:
        super().__init__(cid, guild, members=list(members))
        self.dave_session, self.pver = dave_session, pver


def _member(uid: int, bot: bool = False) -> SimpleNamespace:
    return SimpleNamespace(id=uid, bot=bot, display_name=f"member-{uid}")


def _channel(adapter, cid=VOICE_A, **kwargs) -> _GateChannel:
    return _GateChannel(cid, adapter._fake_guild, **kwargs)


def _patch_discord(monkeypatch) -> None:
    """A fake ``VoiceClient`` base, and a clip that decodes without spawning ffmpeg."""
    monkeypatch.setattr(adapter_mod, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(adapter_mod.discord, "VoiceClient", _GateVoiceClient, raising=False)

    class _PCM:
        def __init__(self, *args, **kwargs) -> None:
            self._frames = iter([b"\x00\x01" * 480, b""])

        def read(self) -> bytes:
            return next(self._frames, b"")

        def is_opus(self) -> bool:
            return False

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(adapter_mod.discord, "FFmpegPCMAudio", _PCM, raising=False)
    monkeypatch.setattr(adapter_mod.discord, "PCMVolumeTransformer",
                        lambda source, volume: source, raising=False)


class _Env:
    def __init__(self, adapter, calls: list) -> None:
        self.adapter, self.calls = adapter, calls

    @property
    def receiver(self) -> VoiceReceiver:
        return self.adapter._voice_receivers[GUILD]

    @property
    def vc(self):
        return self.adapter._voice_clients[GUILD]

    @property
    def text(self):
        return self.adapter._fake_text

    @property
    def conn(self):
        return self.vc._connection

    def info(self) -> dict:
        return self.adapter._voice_ready_info.get(GUILD) or {}

    @property
    def ready_at(self):
        return getattr(self.receiver, "ready_at", None)


def _env(monkeypatch, *, lease=None, tone=True, timeout=5.0, requested="cpu",
         cuda_extra=False, gpu=False, gain=0.5, mixer=False) -> _Env:
    """A real adapter whose ready gate runs against canned STT and GPU facts."""
    adapter = fx.make_adapter(bot_user_id=ME)
    adapter._voice_fx_cfg = {
        **adapter._voice_fx_cfg,
        "ready_tone_enabled": tone, "ready_timeout_s": timeout, "ready_tone_gain": gain,
        "enabled": mixer,
    }
    adapter._voice_stt_lease_acquire = lease or (lambda name, install=False: dict(_WARM_CPU))
    if ready_mod is not None:
        monkeypatch.setattr(ready_mod, "_cuda_extra_selected", lambda: cuda_extra)
        monkeypatch.setattr(ready_mod, "_cuda_device_visible", lambda: gpu)
    monkeypatch.setattr("tools.transcription_tools._load_stt_config",
                        lambda: {"provider": "local", "local": {"device": requested}})
    _patch_discord(monkeypatch)

    calls: list = []
    real_play = adapter.play_in_voice_channel

    async def spy(guild_id, audio_path, strict=False):
        calls.append({"guild": guild_id, "path": audio_path, "strict": strict})
        return await real_play(guild_id, audio_path, strict=strict)

    monkeypatch.setattr(adapter, "play_in_voice_channel", spy)

    async def _timeout(_path):
        return 5

    monkeypatch.setattr(adapter, "_playback_timeout_for_audio", _timeout)

    async def _no_mixer(guild_id, vc):
        return None

    monkeypatch.setattr(adapter, "_install_voice_mixer", _no_mixer)
    return _Env(adapter, calls)


async def _join(env: _Env, channel=None, *, members=(OPERATOR,)) -> _GateChannel:
    channel = channel or _channel(env.adapter, members=[_member(u) for u in members])
    assert await env.adapter.join_voice_channel(channel, text_channel_id=TEXT) is True
    # The poll loop would consume segments (and reach real STT) while a test asserts on them.
    task = env.adapter._voice_listen_tasks.pop(GUILD, None)
    if task is not None:
        task.cancel()
    return channel


async def _until(predicate, timeout=3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _ready(env: _Env) -> bool:
    return env.ready_at is not None


async def _deliver_commit(env: _Env, session) -> None:
    """The op-22 commit: a ready session lands, and the hook's refresh_credentials re-reads it."""
    env.conn.dave_session = session
    assert env.conn.hook is not None
    await env.conn.hook(None, {"op": 22, "d": {}})


# --- the audio edge --------------------------------------------------------------------------------


def _packet(nacl: _FakeNacl, receiver: VoiceReceiver, ssrc: int, seq: int = 1) -> None:
    nacl.plaintext = b"opus-frame"
    header = struct.pack(">BBHII", 0x80, 0x78, seq, 1000 + seq, ssrc)
    receiver._on_packet(header + nacl.plaintext + b"nonc")


def _speak(env: _Env, nacl: _FakeNacl, ssrc: int, frames: int) -> None:
    for index in range(frames):
        _packet(nacl, env.receiver, ssrc, seq=index + 1)


def _quiet(env: _Env) -> None:
    for ssrc in list(env.receiver._last_packet_time):
        env.receiver._last_packet_time[ssrc] = time.monotonic() - 30


# --- the red test ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_packet_received_before_the_dave_commit_is_cut_and_the_tone_plays(
        monkeypatch, nacl, decoders):
    """RED on 7248a29: a delayed DAVE commit delivered a pre-ready packet to STT, playing no tone."""
    env = _env(monkeypatch)
    channel = _channel(env.adapter, members=[_member(OPERATOR)],
                       dave_session=_Daveless(ready=False), pver=1)
    await _join(env, channel)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)

    _speak(env, nacl, SSRC, 40)                  # 0.8 s of speech heard before the tone
    assert env.calls == [], "the tone played before the join was ready"
    assert getattr(receiver, "ready_at", None) is None

    await _deliver_commit(env, _Daveless(ready=True))
    assert await _until(lambda: len(env.calls) == 1), "the join played no ready tone"

    [call] = env.calls
    assert call["guild"] == GUILD and call["strict"] is True
    assert call["path"].endswith(".wav")
    assert receiver.dave_settled
    assert env.info()["degraded"] is False

    _quiet(env)
    assert receiver.check_silence() == [], "audio received before the tone reached STT"
    assert receiver._pre_ready_dropped == 40


# --- both conditions, and the loop ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tone_waits_for_both_the_dave_commit_and_the_lease(monkeypatch, decoders):
    """Neither condition alone opens the gate; both are needed, in either order."""
    started, release = threading.Event(), threading.Event()
    seen: dict = {}

    def lease(name, install=False):
        seen.update(name=name, install=install, thread=threading.current_thread().name)
        started.set()
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, timeout=10)
    channel = _channel(env.adapter, members=[_member(OPERATOR)],
                       dave_session=_Daveless(ready=False), pver=1)
    loop_thread = threading.current_thread().name
    await _join(env, channel)
    assert await _until(started.is_set)
    assert seen["name"] == f"discord:voice:{GUILD}"
    assert seen["install"] is False, "a join must never trigger an install"
    assert seen["thread"] != loop_thread, "the warm-up ran on the event loop"

    await _deliver_commit(env, _Daveless(ready=True))   # DAVE done, the lease still loading
    for _ in range(5):
        if env.calls:                                    # a tone without the lease: the bug
            break
        await asyncio.sleep(0.02)                        # the loop keeps ticking: not blocked
    assert env.calls == []

    release.set()
    assert await _until(lambda: _ready(env))
    assert len(env.calls) == 1
    # Nothing about per-speaker mapping or the decryptor gates the tone: no packet ever arrived, so
    # no SSRC is mapped and no decoder exists.
    assert env.receiver._ssrc_to_user == {} and env.receiver._decoders == {}


@pytest.mark.asyncio
async def test_the_join_acquires_its_lease_off_the_event_loop(monkeypatch):
    release = threading.Event()
    seen: dict = {}

    def lease(name, install=False):
        seen.update(name=name, thread=threading.current_thread().name)
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, timeout=10)
    loop_thread = threading.current_thread().name
    await _join(env)
    assert await _until(lambda: "name" in seen)
    assert seen["name"] == f"discord:voice:{GUILD}"
    assert seen["thread"] != loop_thread
    ticks = 0
    for _ in range(5):
        # health: allow HX013 -- the tick IS the measurement: five loop turns while the lease's worker is blocked prove the loop was not stalled behind the model load
        await asyncio.sleep(0.01)
        ticks += 1
    assert ticks == 5, "the event loop stalled behind the model load"
    release.set()
    await _until(lambda: _ready(env))


# --- the bounded wait and the degraded line --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_lease_error_opens_the_gate_degraded_at_once(monkeypatch):
    env = _env(monkeypatch, timeout=15.0,
               lease=lambda name, install=False: {"action": "error",
                                                 "error": "faster-whisper not installed"})
    began = time.monotonic()
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert time.monotonic() - began < 3, "a lease error must not wait out ready_timeout_s"

    assert len(env.calls) == 1                       # degraded still plays the tone
    assert env.text.send.await_count == 1
    line = str(env.text.send.await_args.args[0])
    assert line.startswith("voice ready (degraded:") and "faster-whisper" in line
    assert env.info()["degraded"] is True and env.info()["reason"]


@pytest.mark.asyncio
async def test_a_dave_commit_that_never_arrives_opens_the_gate_degraded(monkeypatch):
    env = _env(monkeypatch, timeout=0.2)
    channel = _channel(env.adapter, members=[_member(OPERATOR)],
                       dave_session=_Daveless(ready=False), pver=1)
    began = time.monotonic()
    await _join(env, channel)
    assert await _until(lambda: _ready(env))
    waited = time.monotonic() - began
    assert 0.15 <= waited < 3, f"the bounded wait opened after {waited:.2f}s, not ~ready_timeout_s"

    assert "DAVE" in (env.info()["reason"] or "")
    assert len(env.calls) == 1                       # never deaf: the tone still plays
    assert env.text.send.await_count == 1
    assert "voice ready (degraded: DAVE" in str(env.text.send.await_args.args[0])


# --- degraded is only the GPU the host was set up for ----------------------------------------------


@pytest.mark.asyncio
async def test_a_cuda_pin_that_warms_on_cpu_is_degraded(monkeypatch):
    """The device the warm inference reports is what the gate trusts, not the request."""
    env = _env(monkeypatch, requested="cuda")
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert env.info()["degraded"] is True
    assert "device: cuda" in (env.info()["reason"] or "")
    assert len(env.calls) == 1 and env.text.send.await_count == 1


@pytest.mark.asyncio
async def test_auto_with_the_cuda_extra_and_a_failed_cuda_is_degraded(monkeypatch):
    env = _env(monkeypatch, requested="auto", cuda_extra=True, gpu=True)
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert env.info()["degraded"] is True
    assert "stt-whisper-cuda" in (env.info()["reason"] or "")
    assert len(env.calls) == 1 and env.text.send.await_count == 1


@pytest.mark.asyncio
async def test_auto_with_a_visible_gpu_and_no_cuda_extra_is_plain_ready(monkeypatch):
    env = _env(monkeypatch, requested="auto", cuda_extra=False, gpu=True)
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert env.info()["degraded"] is False
    assert env.text.send.await_count == 0
    assert len(env.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,cuda_extra,gpu,degraded", [
    ("cpu", False, False, False),    # device: cpu — a deliberate pin
    ("auto", False, False, False),   # no CUDA device on the host
    ("auto", False, True, False),    # a GPU is visible but the extra was not selected
    ("auto", True, False, False),    # extra selected on a host with no CUDA device
    ("auto", True, True, True),      # extra selected, GPU visible, CUDA still failed
    ("cuda", False, False, True),    # a pin is expected to hold
])
async def test_the_libcublas_fallback_is_degraded_only_for_the_gpu_the_host_expected(
        requested, cuda_extra, gpu, degraded, monkeypatch):
    """The fake ``load ok / first iteration raises libcublas`` warm reports ``device=cpu`` (e2).

    The gate compares that device with what the host asked for: only a GPU the host was set up for
    counts as degraded, so a CPU-only seat and a seat that declined the extra stay plain.
    """
    env = _env(monkeypatch, requested=requested, cuda_extra=cuda_extra, gpu=gpu)
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert env.info()["degraded"] is degraded
    assert env.text.send.await_count == (1 if degraded else 0)
    assert len(env.calls) == 1                       # the tone plays either way


# --- the lease's lifetime --------------------------------------------------------------------------


@pytest.fixture
def real_lease(monkeypatch):
    """The real lease register, with the STT engine absent (an immediate ``action='error'``)."""
    stt_lease._reset_stt_leases_for_tests()
    monkeypatch.setattr("tools.transcription_tools._HAS_FASTER_WHISPER", False, raising=False)
    yield
    stt_lease._reset_stt_leases_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["leave", "closed_by_discord", "stale"])
async def test_each_session_end_path_releases_the_join_lease(path, monkeypatch, real_lease):
    adapter = fx.make_adapter(bot_user_id=ME)
    adapter._voice_fx_cfg = {**adapter._voice_fx_cfg, "ready_tone_enabled": False}
    adapter._voice_stt_lease_acquire = None   # the real lease register, not a canned one
    _patch_discord(monkeypatch)
    env = _Env(adapter, [])
    await _join(env)
    lease = f"discord:voice:{GUILD}"
    assert await _until(lambda: lease in stt_lease.stt_lease_holders(), timeout=5)

    if path == "leave":
        await adapter.leave_voice_channel(GUILD)
    elif path == "closed_by_discord":
        env.vc.closed_by_discord()
        await adapter._end_voice_session_on_close(GUILD, env.vc)
    else:
        await adapter._end_stale_voice_session_locked(GUILD)

    assert stt_lease.stt_lease_holders() == [], f"the {path} path left the join's lease held"


@pytest.mark.asyncio
async def test_a_move_abandons_the_first_lease_and_releases_it_when_it_finishes(
        monkeypatch, real_lease):
    """A move re-arms the gate, so the abandoned warm must not hold the lease afterwards.

    The first worker is held BEFORE it registers (``acquire_stt_lease`` registers inside the
    worker thread), so the move overwrites a future that never started: the leave releases the new
    lease, and the abandoned worker's late registration has to be released by its own
    done-callback — otherwise ``discord:voice:<guild>`` outlives the join (integration 2c, e6 2).
    """
    adapter = fx.make_adapter(bot_user_id=ME)
    adapter._voice_fx_cfg = {**adapter._voice_fx_cfg, "ready_tone_enabled": False}
    started, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    calls: list[str] = []

    def lease(name, install=False):
        calls.append(name)
        first = len(calls) == 1
        if first:                        # the warm the move abandons
            started.set()
            proceed.wait(10)
        try:
            return stt_lease.acquire_stt_lease(name, install=False)
        finally:
            if first:
                finished.set()

    adapter._voice_stt_lease_acquire = lease
    _patch_discord(monkeypatch)
    env = _Env(adapter, [])
    lease_name = f"discord:voice:{GUILD}"

    await _join(env)
    assert await _until(started.is_set)
    assert lease_name not in stt_lease.stt_lease_holders()

    await _join(env, _channel(adapter, VOICE_B, members=[_member(OPERATOR)]))
    assert env.vc.channel.id == VOICE_B
    assert await _until(lambda: len(calls) == 2), "the move did not take a fresh lease"
    assert calls == [lease_name, lease_name]
    # Let the SECOND warm finish before the leave: its done-callback (if any) releases the shared
    # name, which would mask the abandoned worker's late registration.
    second = adapter._voice_lease_tasks[GUILD]
    assert await _until(second.done, timeout=10)
    assert await _until(lambda: lease_name in stt_lease.stt_lease_holders())

    await adapter.leave_voice_channel(GUILD)
    assert stt_lease.stt_lease_holders() == [], "the leave left the join's lease held"

    proceed.set()                        # the abandoned warm registers and finishes now
    assert await _until(finished.is_set, timeout=5)
    for _ in range(20):                  # the done-callback lands on the loop, not inline
        if stt_lease.stt_lease_holders():
            break
        await asyncio.sleep(0.01)
    assert stt_lease.stt_lease_holders() == [], \
        "the abandoned worker kept the lease after it finished"


# --- the tone --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tone_goes_through_the_strict_play_path(monkeypatch):
    """``play_in_voice_channel(..., strict=True)`` on the legacy FFmpeg path, mixer off."""
    env = _env(monkeypatch)
    await _join(env)
    assert await _until(lambda: _ready(env))
    [call] = env.calls
    assert call["strict"] is True
    [source] = env.vc.played
    assert type(source).__name__ == "_TrackedSource"   # strict wraps the clip that way
    assert env.adapter._voice_mixers.get(GUILD) is None


@pytest.mark.asyncio
async def test_the_tone_goes_through_the_mixer_when_it_is_on(monkeypatch):
    release, seen = threading.Event(), {}

    def lease(name, install=False):
        seen["called"] = True
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, mixer=True, timeout=10)
    monkeypatch.setattr(adapter_mod, "_voice_mixer_module",
                        lambda: SimpleNamespace(decode_to_pcm=lambda path: b"\x00\x01" * 480,
                                               BYTES_PER_MS=192))

    class _Mixer:
        speech_active = False

        def __init__(self) -> None:
            self.played: list = []

        def play_speech(self, pcm, gain=1.0):
            self.played.append((pcm, gain))
            return SimpleNamespace(_finished=True)

    mixer = _Mixer()
    env.adapter._voice_mixers[GUILD] = mixer
    await _join(env)
    assert await _until(lambda: seen.get("called"))
    release.set()
    assert await _until(lambda: _ready(env))
    assert len(env.calls) == 1 and env.calls[0]["strict"] is True
    assert mixer.played and mixer.played[0][1] == 1.0   # speech_gain, not the ready tone's own gain


def test_the_tone_wav_is_rendered_at_the_configured_gain(tone_home):
    soft = Path(ready_mod.ready_tone_cache_path(0.2))
    loud = Path(ready_mod.ready_tone_cache_path(0.8))
    assert soft != loud, "the gain is part of the cache key"
    assert loud.name == "ready_tone_0.800.wav"

    def peak(path: Path) -> int:
        with wave.open(str(path)) as wav:
            frames = wav.readframes(wav.getnframes())
        return max(abs(value) for value in struct.unpack(f"<{len(frames) // 2}h", frames))

    soft_peak, loud_peak = peak(soft), peak(loud)
    assert soft_peak > 0 and loud_peak > soft_peak * 3.5
    # A rendered clip is reused, not re-rendered.
    before = soft.stat().st_mtime_ns
    assert ready_mod.ready_tone_cache_path(0.2) == str(soft)
    assert soft.stat().st_mtime_ns == before


@pytest.mark.asyncio
async def test_the_gate_re_renders_the_tone_after_a_gain_change(monkeypatch):
    env = _env(monkeypatch, gain=0.2)
    await _join(env)
    assert await _until(lambda: _ready(env))
    first = env.calls[0]["path"]
    assert first.endswith("ready_tone_0.200.wav")

    await env.adapter.leave_voice_channel(GUILD)
    env.adapter._voice_fx_cfg = {**env.adapter._voice_fx_cfg, "ready_tone_gain": 0.9}
    await _join(env)
    assert await _until(lambda: len(env.calls) == 2)
    assert env.calls[1]["path"] != first
    assert env.calls[1]["path"].endswith("ready_tone_0.900.wav")


# --- the cut ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ring_frame_held_before_the_tone_is_dropped_when_released_after(monkeypatch, nacl, decoders):
    """Receive time decides: a frame the DAVE ring held before ready is not replayed to STT."""
    env = _env(monkeypatch, timeout=10)
    channel = _channel(env.adapter, members=[_member(OPERATOR), _member(STRANGER)],
                       dave_session=_Daveless(ready=False), pver=1)
    await _join(env, channel)
    receiver = env.receiver

    _packet(nacl, receiver, 777)                 # before the tone: held, because nobody mapped it
    assert len(receiver._dave_pending[777]) == 1
    assert decoders["n"] == 0

    await _deliver_commit(env, _Daveless(ready=True))
    assert await _until(lambda: _ready(env))
    receiver.map_ssrc(777, OPERATOR)             # the SPEAKING event lands after the tone
    _packet(nacl, receiver, 777, seq=2)          # the next frame triggers the replay

    assert 777 not in receiver._dave_pending
    assert decoders["n"] == 1, "the frame held from before the tone was decoded after it"
    assert receiver._pre_ready_dropped >= 1
    assert len(receiver._buffers[777]) == FRAME_BYTES


@pytest.mark.asyncio
async def test_a_packet_arriving_after_the_tone_reaches_stt(monkeypatch, nacl, decoders):
    env = _env(monkeypatch)
    await _join(env)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    _speak(env, nacl, SSRC, 40)                  # before the tone
    assert await _until(lambda: _ready(env))
    _speak(env, nacl, SSRC, 30)                  # after it
    _quiet(env)

    [segment] = receiver.check_silence()
    assert segment[0] == OPERATOR
    assert len(segment[1]) == 30 * FRAME_BYTES, "the segment carried pre-tone audio"


@pytest.mark.asyncio
async def test_a_flush_inside_the_deaf_window_drops_what_the_buffers_hold(
        monkeypatch, nacl, decoders):
    """Before ``t_ready`` is stamped, a flush must drop, not deliver.

    ``_ready_at is None`` covers the whole deaf window — a cold model load, a DAVE negotiation, a
    move's re-arm — and the cut's old ``return 0`` handed STT exactly the audio the plan forbids:
    the speaker only has to stop for ``SILENCE_THRESHOLD``, or the bot only has to leave.  Both
    flush paths are checked, because a leave flushes instead of polling.
    """
    release, seen = threading.Event(), {}

    def lease(name, install=False):
        seen["called"] = True
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, timeout=10)
    await _join(env)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    assert await _until(lambda: seen.get("called"))
    _speak(env, nacl, SSRC, 40)                 # 0.8 s while the model is still loading
    assert env.ready_at is None

    _quiet(env)
    assert receiver.check_silence() == [], "the deaf window's audio was flushed before the tone"
    assert receiver._pre_ready_dropped == 40

    _speak(env, nacl, SSRC, 30)                 # a fresh segment, still before the tone
    assert env.ready_at is None
    assert receiver.flush_pending() == [], "the leave flush delivered pre-tone audio"
    assert receiver._pre_ready_dropped == 70
    release.set()


@pytest.mark.asyncio
async def test_the_arrival_stamp_is_taken_on_receipt_not_at_decode(
        monkeypatch, nacl, decoders):
    """A reader that falls behind still stamps the packet when it reached the socket.

    The transport decrypt is blocked while the tone ends, so the frame was HEARD before
    ``t_ready`` and its decode runs after it.  Stamping at decode entry (what the cut used to
    read) admits exactly the audio the tone is supposed to fence off (e6 finding 4).
    """
    lease_release, seen = threading.Event(), {}

    def lease(name, install=False):
        seen["called"] = True
        lease_release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, timeout=10)
    await _join(env)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    assert await _until(lambda: seen.get("called"))
    assert env.ready_at is None

    in_decrypt, let_decrypt = threading.Event(), threading.Event()

    def slow_decrypt(encrypted, header, nonce):
        in_decrypt.set()
        assert let_decrypt.wait(10)
        return b"opus-frame"

    monkeypatch.setattr(nacl, "decrypt", slow_decrypt)
    header = struct.pack(">BBHII", 0x80, 0x78, 1, 1001, SSRC)
    reader = threading.Thread(target=receiver._on_packet,
                              args=(header + b"opus-frame" + b"nonc",), daemon=True)
    reader.start()
    assert await _until(in_decrypt.is_set)
    receiver.mark_ready()                       # the tone ended while this frame sat undecrypted
    let_decrypt.set()
    reader.join(10)
    assert not reader.is_alive()

    # The stamp travelled through the blocked decrypt: it is receipt time, not decode entry.
    with receiver._lock:
        [(stamp, nbytes)] = list(receiver._arrivals[SSRC])
    assert nbytes == FRAME_BYTES
    assert stamp <= receiver.ready_at, "the frame was stamped at decode, after the tone"

    _quiet(env)
    assert receiver.check_silence() == [], "the frame was admitted by its decode time"
    assert receiver._pre_ready_dropped == 1
    lease_release.set()


@pytest.mark.asyncio
async def test_the_leave_flush_applies_the_same_cut(monkeypatch, nacl, decoders):
    """The leave path flushes instead of polling: the cut must be applied there too."""
    release, seen = threading.Event(), {}

    def lease(name, install=False):
        seen["called"] = True
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, timeout=10)
    await _join(env)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    assert await _until(lambda: seen.get("called"))
    _speak(env, nacl, SSRC, 40)                  # 0.8 s while the model was still loading
    release.set()
    assert await _until(lambda: _ready(env))

    # No post-tone audio at all: everything buffered arrived before t_ready and must not be flushed
    # into the new channel's transcription.
    assert receiver.flush_pending() == []
    assert receiver._pre_ready_dropped == 40


@pytest.mark.asyncio
async def test_a_member_who_speaks_after_the_tone_is_not_cut(monkeypatch, nacl, decoders):
    env = _env(monkeypatch)
    await _join(env)
    assert await _until(lambda: _ready(env))
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    _speak(env, nacl, SSRC, 30)
    _quiet(env)

    [segment] = receiver.check_silence()
    assert len(segment[1]) == 30 * FRAME_BYTES
    assert receiver._pre_ready_dropped == 0


# --- once per join: re-arm, and what does not re-arm ------------------------------------------------


@pytest.mark.asyncio
async def test_a_leave_and_rejoin_rearms_the_gate(monkeypatch):
    env = _env(monkeypatch)
    await _join(env)
    assert await _until(lambda: _ready(env))
    first_receiver, first_ready = env.receiver, env.ready_at

    await env.adapter.leave_voice_channel(GUILD)
    await _join(env)
    assert await _until(lambda: _ready(env))
    assert len(env.calls) == 2, "a new join plays its own tone"
    assert env.receiver is not first_receiver
    assert env.ready_at > first_ready


@pytest.mark.asyncio
async def test_a_move_to_another_channel_rearms_the_gate(monkeypatch):
    env = _env(monkeypatch)
    await _join(env, _channel(env.adapter, VOICE_A, members=[_member(OPERATOR)]))
    assert await _until(lambda: _ready(env))
    receiver, first_ready = env.receiver, env.ready_at

    await _join(env, _channel(env.adapter, VOICE_B, members=[_member(OPERATOR)]))
    assert env.receiver is receiver              # the same session, a new room
    assert env.vc.channel.id == VOICE_B
    assert await _until(lambda: _ready(env))
    assert len(env.calls) == 2
    assert env.ready_at > first_ready


@pytest.mark.asyncio
async def test_a_reconnect_does_not_rearm_the_gate(monkeypatch):
    env = _env(monkeypatch)
    await _join(env)
    assert await _until(lambda: _ready(env))
    receiver, ready_at = env.receiver, env.ready_at

    env.vc.reconnecting()                        # the websocket dropped; cleanup() has not run
    with pytest.raises(RuntimeError):
        await env.adapter.join_voice_channel(_channel(env.adapter, VOICE_A),
                                             text_channel_id=TEXT)
    assert len(env.calls) == 1, "a reconnect window must not play a second tone"
    assert env.receiver is receiver and env.ready_at == ready_at

    env.vc._connected = True                     # discord.py resumed the same client
    await _join(env)                             # a programmatic re-join of the same channel
    assert len(env.calls) == 1
    assert env.ready_at == ready_at


@pytest.mark.asyncio
async def test_a_re_epoch_does_not_rearm_the_gate(monkeypatch):
    env = _env(monkeypatch)
    channel = _channel(env.adapter, members=[_member(OPERATOR)],
                       dave_session=_Daveless(ready=True), pver=1)
    await _join(env, channel)
    assert await _until(lambda: _ready(env))
    ready_at = env.ready_at

    env.conn.dave_protocol_version = 2
    await env.conn.hook(None, {"op": 24, "d": {"epoch": 2}})
    env.conn.dave_protocol_version = 1
    await env.conn.hook(None, {"op": 4, "d": {}})
    for _ in range(5):
        if len(env.calls) > 1:                            # a second tone: the bug under test
            break
        await asyncio.sleep(0.02)

    assert len(env.calls) == 1, "a re-epoch must not play another tone"
    assert env.ready_at == ready_at


# --- the tone disabled (D-016) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_the_tone_disabled_t_ready_is_when_ready_held_and_pre_ready_audio_is_cut(
        monkeypatch, nacl, decoders):
    release, seen = threading.Event(), {}

    def lease(name, install=False):
        seen["called"] = True
        release.wait(10)
        return dict(_WARM_CPU)

    env = _env(monkeypatch, lease=lease, tone=False, timeout=10)
    await _join(env)
    receiver = env.receiver
    receiver.map_ssrc(SSRC, OPERATOR)
    assert await _until(lambda: seen.get("called"))
    _speak(env, nacl, SSRC, 40)                  # heard while the model was still loading
    assert env.ready_at is None

    release.set()
    stamped_from = time.monotonic()
    assert await _until(lambda: _ready(env))
    assert time.monotonic() - stamped_from < 2, "without a tone t_ready is stamped when ready held"

    assert env.calls == [], "the tone is disabled: nothing may be played"
    _quiet(env)
    assert receiver.check_silence() == [], "disabling the tone reopened the pre-ready window"
    assert env.info()["degraded"] is False
