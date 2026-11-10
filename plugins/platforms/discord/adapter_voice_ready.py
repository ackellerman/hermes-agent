"""The per-join ready gate: warm STT, a settled DAVE session, one tone, and the arrival-time cut.

A voice join is deaf until the model is warm and DAVE has finished negotiating: anything a member
says in that window is either undecryptable or a cold-start transcription (plan r1 F3, r2 F5). This
module runs the three conditions as one channel-level cycle per join (D-015):

* the voice connection is up;
* DAVE is settled — plaintext passthrough (``dave_protocol_version == 0``) or a session that reports
  ``ready`` after the op-22/24 commit ``refresh_credentials`` re-reads;
* the join's STT lease reports the model warm, on the device the warm inference actually ran on.

When all three hold, the ready tone plays once and its end stamps ``t_ready`` on the receiver,
whose packet cut then discards everything that ARRIVED at or before it. A bounded wait
(``ready_timeout_s``, D-017) keeps a broken join from staying deaf forever: past it — or at once on
a lease ``action='error'`` — the gate opens degraded, still plays the tone, and posts one
``voice ready (degraded: <reason>)`` line (plan step 9).

Degraded means CPU took over from a GPU the host was set up for: ``device: cuda``, or ``auto`` with
the ``stt-whisper-cuda`` extra selected and CUDA still failing. Every other CPU case is plain ready
with no line — a user who declined the 1.3 GB extra must not be nagged on every join (D-014).
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from tools.stt_lease import acquire_stt_lease, release_stt_lease

logger = logging.getLogger(__name__)

# How often the gate re-checks its three conditions. The conditions are cheap attribute reads, and
# the tone should follow the model by a fraction of a second, not a whole poll interval.
READY_POLL_SECONDS = 0.1
# D-017: a join waits at most 15 s for ready, then opens degraded.
DEFAULT_READY_TIMEOUT_SECONDS = 15.0
DEFAULT_READY_TONE_GAIN = 0.5

# The tone itself: a short soft sine. 660 Hz is well inside Discord's Opus band and reads as a
# chime, not a beep; the 15 ms ramps keep the ends from clicking on the VC's socket.
TONE_SAMPLE_RATE = 48000
TONE_HZ = 660.0
TONE_SECONDS = 0.18
TONE_FADE_SECONDS = 0.015
TONE_PEAK_AT_UNITY_GAIN = 0.22  # -13 dBFS: audible under speech, never startling


def ready_tone_cache_path(gain: float = DEFAULT_READY_TONE_GAIN) -> str:
    """Render (once) and return the ready-tone WAV for *gain*, inside the cache dir.

    ``ready_tone_gain`` is baked into the samples and into the file name, because both playback
    paths apply their own fixed gain on top (the mixer's ``speech_gain``, the legacy path's
    ``PCMVolumeTransformer(volume=1.0)``). A changed gain is therefore a different key, which
    re-renders instead of reusing a stale clip (plan step 10).
    """
    gain = max(0.0, min(1.0, float(gain)))
    directory = Path(get_hermes_home()) / "cache" / "voice_fx"
    path = directory / f"ready_tone_{gain:.3f}.wav"
    if path.is_file() and path.stat().st_size > 44:
        return str(path)
    directory.mkdir(parents=True, exist_ok=True)
    peak = TONE_PEAK_AT_UNITY_GAIN * gain
    frames = int(TONE_SAMPLE_RATE * TONE_SECONDS)
    fade = max(1, int(TONE_SAMPLE_RATE * TONE_FADE_SECONDS))
    pcm = bytearray()
    for index in range(frames):
        envelope = min(1.0, index / fade, (frames - 1 - index) / fade)
        sample = int(peak * envelope * 32767.0
                     * math.sin(2.0 * math.pi * TONE_HZ * index / TONE_SAMPLE_RATE))
        pcm += struct.pack("<h", sample)
    # Write-then-rename: two joins on two seats can race on the same gain key, and a half-written
    # WAV would hand the mixer a broken clip.
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with wave.open(str(staging), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TONE_SAMPLE_RATE)
        wav.writeframes(bytes(pcm))
    os.replace(staging, path)
    logger.debug("[Discord] rendered the ready tone (gain=%.3f) into %s", gain, path)
    return str(path)


def _as_seconds(value: Any, default: float) -> float:
    """A config value in seconds; anything unusable falls back to *default*."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    return seconds if seconds >= 0 else default


def _cuda_extra_selected() -> bool:
    """Is the opt-in ``stt-whisper-cuda`` extra installed in THIS interpreter?

    Its package resolves only when the extra was selected (``pm/extras.py`` ANCHORS:
    ``nvidia.cublas``), which is the same proof the preloader uses. Resolving it imports no wheel.
    """
    try:
        from tools.transcription_local import cuda_library_dir

        return cuda_library_dir() is not None
    except Exception:
        return False


def _cuda_device_visible() -> bool:
    """Driver-level NVIDIA probe — the one ``hermes tools`` uses; it never imports ctranslate2."""
    try:
        from hermes_cli.tools_config_post_setup import _nvidia_gpu_present

        return bool(_nvidia_gpu_present())
    except Exception:
        return False


def _lease_result(task: Any) -> dict[str, Any]:
    """The lease's result as a dict; a cancelled or raising warm-up reads as ``action='error'``.

    ``warm_stt_provider`` never raises, but the ``to_thread`` wrapper can still fail (a bad config
    read, a cancelled future), and the gate must open rather than wait out its bound.
    """
    if task is None:
        return {"action": "error", "error": "no STT lease was started"}
    if task.cancelled():
        return {"action": "error", "error": "the STT warm-up was cancelled"}
    try:
        result = task.result()
    except Exception as exc:  # the thread's exception, re-raised here by future.result()
        return {"action": "error", "error": str(exc) or exc.__class__.__name__}
    return result if isinstance(result, dict) else {"action": "error", "error": "unusable lease result"}


class DiscordVoiceReadyMixin:
    """The ready gate and the STT lease's lifetime, shared by every session-end path."""

    _voice_clients: dict[int, Any]
    _voice_receivers: dict[int, Any]
    _voice_text_channels: dict[int, Any]
    _voice_stt_leases: dict[int, str]
    _voice_lease_tasks: dict[int, asyncio.Future]
    _voice_ready_tasks: dict[int, asyncio.Task]
    _voice_ready_info: dict[int, dict[str, Any]]
    # Swap seam: the fakes replace the lease worker so a fake join never loads a model.
    _voice_stt_lease_acquire: Any = None
    _client: Any
    play_in_voice_channel: Any

    # --- the gate ------------------------------------------------------------------------------

    def _voice_ready_cfg(self) -> dict[str, Any]:
        return getattr(self, "_voice_fx_cfg", None) or {}

    def _ready_map(self, field: str) -> dict:
        """A per-guild map of the cycle, created on demand.

        ``DiscordAdapter.__init__`` seeds them; a fixture that builds a bare adapter with
        ``object.__new__`` (older voice tests do) must still be able to leave a guild.
        """
        return self.__dict__.setdefault(field, {})

    def _start_voice_ready_gate(self, guild_id: int) -> asyncio.Task:
        """Take the join's STT lease (off the event loop) and start the ready cycle."""
        self._cancel_voice_ready_gate(guild_id)
        lease = f"discord:voice:{guild_id}"
        self._ready_map("_voice_stt_leases")[guild_id] = lease
        # The warm blocks for the whole model load (cold download + medium on CPU is minutes), so it
        # must never run on the loop that keeps the gateway answering. ``install=False``: a voice
        # join must never start a dependency rebuild (plan step 7).
        lease_task = asyncio.ensure_future(asyncio.to_thread(self._voice_acquire_stt_lease, lease))
        self._ready_map("_voice_lease_tasks")[guild_id] = lease_task
        task = asyncio.ensure_future(self._voice_ready_gate(guild_id, lease_task=lease_task))
        self._ready_map("_voice_ready_tasks")[guild_id] = task
        return task

    def _voice_acquire_stt_lease(self, lease: str) -> dict[str, Any]:
        """Acquire (and warm) *lease*; the seam test fakes replace, and never an install."""
        acquire = getattr(self, "_voice_stt_lease_acquire", None) or acquire_stt_lease
        return acquire(lease, install=False)

    def _cancel_voice_ready_gate(self, guild_id: int) -> None:
        task = self._ready_map("_voice_ready_tasks").pop(guild_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._ready_map("_voice_ready_info").pop(guild_id, None)

    def _release_voice_stt_lease(self, guild_id: int) -> None:
        """Drop the guild's lease, and again when a still-running warm finishes.

        ``acquire_stt_lease`` registers the holder inside the worker thread, so a leave that lands
        before the thread starts would otherwise leave ``discord:voice:<guild>`` held forever by a
        model load nobody is waiting for.
        """
        lease = self._ready_map("_voice_stt_leases").pop(guild_id, None)
        task = self._ready_map("_voice_lease_tasks").pop(guild_id, None)
        if not lease:
            return
        release_stt_lease(lease)
        if task is not None and not task.done():
            task.add_done_callback(lambda _f, _lease=lease: release_stt_lease(_lease))

    def _rearm_voice_ready_gate(self, guild_id: int) -> None:
        """A move into another channel is a new room: the cut re-arms and the tone plays again."""
        receiver = self._voice_receivers.get(guild_id)
        rearm = getattr(receiver, "rearm_ready_gate", None)
        if callable(rearm):
            rearm()
        if receiver is not None:
            self._start_voice_ready_gate(guild_id)

    def _voice_ready_state(self, guild_id: int, *, lease_task: Any) -> tuple[str, Optional[str]]:
        """``('ready'|'waiting'|'error', detail)`` for the cycle's three channel-level conditions."""
        vc = self._voice_clients.get(guild_id)
        if vc is None or not vc.is_connected():
            return "waiting", "the voice connection is not up"
        receiver = self._voice_receivers.get(guild_id)
        if receiver is None or not getattr(receiver, "dave_settled", False):
            return "waiting", "DAVE is not settled"
        if lease_task is None:
            return "error", "STT unavailable: no lease was started"
        if not lease_task.done():
            return "waiting", "the STT model is not warm"
        result = _lease_result(lease_task)
        if result.get("action") == "error":
            return "error", f"STT unavailable: {result.get('error') or 'unknown error'}"
        if not (result.get("warmed") or result.get("action") == "noop"):
            return "waiting", "the STT model is not warm"
        return "ready", None

    def _voice_ready_degraded_reason(self, result: dict[str, Any]) -> Optional[str]:
        """Only a GPU the host was set up for, taken over by CPU, is degraded (plan step 8)."""
        if str(result.get("device") or "").strip().lower() != "cpu":
            return None
        requested = self._voice_stt_requested_device()
        if requested == "cuda":
            return "device: cuda, but the model warmed on cpu"
        if requested == "auto" and _cuda_extra_selected() and _cuda_device_visible():
            return ("auto with stt-whisper-cuda selected, but CUDA failed; "
                    "the model warmed on cpu")
        return None

    @staticmethod
    def _voice_stt_requested_device() -> str:
        """``stt.local.device`` as configured (``auto`` when unset), the gate's point of comparison."""
        try:
            from tools.transcription_tools import _load_stt_config

            local = (_load_stt_config() or {}).get("local") or {}
            return str(local.get("device") or "auto").strip().lower()
        except Exception:
            return "auto"

    async def _voice_ready_gate(self, guild_id: int, *, lease_task: Any) -> None:
        cfg = self._voice_ready_cfg()
        timeout = _as_seconds(cfg.get("ready_timeout_s"), DEFAULT_READY_TIMEOUT_SECONDS)
        started = time.monotonic()
        deadline = started + timeout
        reason: Optional[str] = None
        while True:
            state, detail = self._voice_ready_state(guild_id, lease_task=lease_task)
            if state == "ready":
                break
            if state == "error":
                reason = detail
                break
            if time.monotonic() >= deadline:
                # The gate opens anyway: a join never stays deaf (plan step 9).
                reason = f"{detail} after {timeout:g}s"
                break
            await asyncio.sleep(READY_POLL_SECONDS)
        result = _lease_result(lease_task)
        if reason is None:
            reason = self._voice_ready_degraded_reason(result)
        if reason:
            logger.warning("[Discord] voice ready (degraded: %s) (guild=%d)", reason, guild_id)
        # When the ready condition held: with the tone silenced (D-016) this IS t_ready, and the
        # cut still applies — only the clip is gone.
        ready_at = time.monotonic()
        if bool(cfg.get("ready_tone_enabled", True)):
            try:
                path = ready_tone_cache_path(cfg.get("ready_tone_gain", DEFAULT_READY_TONE_GAIN))
                await self.play_in_voice_channel(guild_id, path, strict=True)
            except Exception as exc:
                logger.warning("[Discord] ready tone failed to play (guild=%d): %s", guild_id, exc)
            # t_ready is when the play call RETURNED, success or failure: the tone has been heard
            # for exactly as long as it took.
            ready_at = time.monotonic()
        receiver = self._voice_receivers.get(guild_id)
        mark_ready = getattr(receiver, "mark_ready", None)
        if callable(mark_ready):
            mark_ready(ready_at)
        self._ready_map("_voice_ready_info")[guild_id] = {
            "t_ready": ready_at, "reason": reason, "degraded": reason is not None,
            "device": result.get("device"), "model": result.get("model"),
        }
        logger.info(
            "[Discord] voice ready (guild=%d, device=%s, model=%s, degraded=%s, %dms)",
            guild_id, result.get("device") or "?", result.get("model") or "?",
            bool(reason), int((time.monotonic() - started) * 1000),
        )
        if reason:
            await self._post_voice_ready_degraded(guild_id, reason)

    async def _post_voice_ready_degraded(self, guild_id: int, reason: str) -> None:
        """One ``voice ready (degraded: …)`` line to the bound text channel; never fatal."""
        text_channel_id = (getattr(self, "_voice_text_channels", None) or {}).get(guild_id)
        client = getattr(self, "_client", None)
        if text_channel_id is None or client is None:
            return
        try:
            channel = client.get_channel(text_channel_id)
            if channel is None:
                return
            await channel.send(f"voice ready (degraded: {reason})")
        except Exception:
            logger.debug("[Discord] degraded ready line could not be posted (guild=%d)",
                         guild_id, exc_info=True)
