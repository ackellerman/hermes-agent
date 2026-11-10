"""Local-STT lifecycle for the desktop voice-input path: warm-up leases.

Desktop voice transcription (``POST /api/audio/transcribe``) blocks until the
configured STT engine answers inside the renderer's timeout floor (180s for a
short clip). With the ``local`` provider that floor also covers the cold cost:
first-use faster-whisper download + model load, or a reload after
``stt.local.unload_after_idle_seconds`` evicted the cache — on CPU-bound hosts
that alone can exceed the floor, so every attempt times out while the engine
itself is healthy (issue #105955).

Every surface that is about to need speech-to-text holds a *lease* here while
it does (the desktop acquires when the mic opens, releases after the
transcript settles). Acquiring pre-loads the configured local model into the
exact singleton slot transcription reads, so the bill is paid while the user
is still speaking instead of inside the transcription request's timeout.

Mirrors :mod:`tools.tts_tool_lifecycle` with one deliberate divergence:
releasing the last lease does NOT unload the model. The cached local model is
shared with the gateway/CLI/Telegram surfaces in the same backend process —
one surface's "mic closed" must not evict an engine another surface is about
to use. Eviction stays governed by ``stt.local.unload_after_idle_seconds``,
and warming touches the idle timer so a just-warmed model is never
instantly eligible for unload.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
import threading
import time
import wave
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tools.stt_lease")

_stt_lease_lock = threading.Lock()
_stt_leases: set = set()

# A warm-up must prove the DEVICE, not the load: ctranslate2 only dlopens cuBLAS on the first
# inference, so one second of digital silence through ``transcribe`` is what separates a working
# CUDA model from one that fails on the user's first clip (plan r5 F1).
_WARM_AUDIO_SECONDS = 1.0
_WARM_SAMPLE_RATE = 16_000

# One WAV per process: every lease needs the same bytes, and a mkdtemp + 32 KB write on every
# mic-open was pure overhead on the desktop dictation path (review r1 F6).
_warm_audio_lock = threading.Lock()
_warm_audio_paths: dict[tuple[int, float], str] = {}


def _warm_silence_path() -> str:
    """~1 s of 16 kHz mono 16-bit silence as a WAV, rendered once per process.

    A file rather than a numpy array on purpose: numpy is not a core dependency, and the warm must
    work anywhere faster-whisper does (its test environments included). The file is shared by every
    lease and removed once at interpreter exit; readers only pass the path to ``transcribe``, never
    writing to it.
    """
    key = (_WARM_SAMPLE_RATE, _WARM_AUDIO_SECONDS)
    with _warm_audio_lock:
        cached = _warm_audio_paths.get(key)
        if cached is not None and os.path.isfile(cached):
            return cached
        directory = tempfile.mkdtemp(prefix="hermes-stt-warm-")
        path = os.path.join(directory, "silence.wav")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_WARM_SAMPLE_RATE)
            wav.writeframes(b"\x00\x00" * int(_WARM_SAMPLE_RATE * _WARM_AUDIO_SECONDS))
        atexit.register(shutil.rmtree, directory, True)
        _warm_audio_paths[key] = path
        return path


def _effective_device(model: Any) -> str:
    """The device ctranslate2 resolved for *model* (``WhisperModel.model.device``, ``auto`` already
    resolved); ``"unknown"`` for a model object that does not expose one."""
    for holder in (getattr(model, "model", None), model):
        device = getattr(holder, "device", None)
        if isinstance(device, str) and device.strip():
            return device.strip().lower()
    return "unknown"


def _warm_inference(model: Any, model_name: str, stt_config: dict[str, Any]) -> tuple[str, int]:
    """Transcribe ~1 s of silence so the effective device is proven, not assumed.

    Returns ``(effective_device, transcribe_ms)``. A CUDA library error is the same failure
    ``_transcribe_local`` retries on CPU (faster-whisper decodes lazily, so it surfaces while the
    segments are consumed, inside the guard): evict the poisoned model exactly as
    ``_replace_cached_model_on_cpu`` does and warm the CPU one, so the reported device is the one
    the next clip will use.
    """
    from tools import transcription_tools
    from tools.transcription_local import build_local_transcribe_kwargs

    kwargs = build_local_transcribe_kwargs(stt_config)
    # VAD would drop the whole clip and prove nothing about the decode path.
    kwargs["vad_filter"] = False
    kwargs.pop("vad_parameters", None)
    warm_path = _warm_silence_path()
    started = time.monotonic()
    try:
        segments, _info = model.transcribe(warm_path, **kwargs)
        list(segments)
    except Exception as exc:
        if not transcription_tools._looks_like_cuda_lib_error(exc):
            raise
        logger.warning("[STT] warm-up inference on '%s' failed on CUDA (%s) — evicting the "
                       "cached model and warming CPU (int8)", model_name, exc)
        model = transcription_tools._replace_cached_model_on_cpu(model_name)
        segments, _info = model.transcribe(warm_path, **kwargs)
        list(segments)
    return _effective_device(model), int((time.monotonic() - started) * 1000)


def warm_stt_provider(
    stt_config: Optional[dict[str, Any]] = None, provider: Optional[str] = None, *,
    install: bool = True,
) -> dict[str, Any]:
    """Pre-load the configured STT engine so the next transcription starts hot.

    Blocking; never raises. Only the ``local`` (faster-whisper) provider has
    anything resident: it fills the same cache slot ``_transcribe_local``
    reads (including first-use download and lazy install). ``local_command``
    shells out to an external binary and cloud providers are remote, so both
    are ``action: "noop"``. The result carries ``provider`` / ``warmed`` /
    ``action`` / ``error``.

    Every local lease ends with one short inference (``_warm_inference``) so the result's
    ``device`` is the EFFECTIVE one after any CPU fallback, plus ``model`` and
    ``transcribe_ms`` — a cache hit is warmed too, because a cached model has not proved CUDA.

    ``install=False`` skips the lazy install: boot probes and the voice-join lease must never
    start a dependency rebuild (``transcription_local._try_lazy_install_stt``'s docstring). The
    default keeps the desktop lease's behaviour.
    """
    from tools import transcription_tools
    from tools.transcription_local import _normalize_local_model

    if stt_config is None:
        stt_config = transcription_tools._load_stt_config()
    name = (provider or transcription_tools._get_provider(stt_config) or "").lower().strip()
    result: dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}
    if name not in {"local", "local_command"}:
        return result
    if name == "local_command":
        result.update(warmed=True)
        return result
    if not transcription_tools._HAS_FASTER_WHISPER:
        if not install or not transcription_tools._try_lazy_install_stt():
            result.update(action="error", error="faster-whisper not installed")
            return result
    local_cfg = stt_config.get("local") or {}
    # Same name transcription itself resolves (config > default, cloud-only
    # names normalized) so warm-up fills the slot the next request reads.
    model_name = _normalize_local_model(local_cfg.get("model"))
    hot_before = (
        transcription_tools._local_model is not None
        and transcription_tools._local_model_name == model_name
    )
    started = time.monotonic()
    try:
        model = transcription_tools._get_or_load_local_model(model_name, local_cfg)
        if model is None:  # defensive: load failed without raising
            result.update(action="error", error="Local whisper model failed to load")
            return result
        # A just-loaded model counts as activity: the idle-unload watcher must
        # not treat the load as idle time and evict the model. Touch BEFORE the
        # warm inference — with unload_after_idle_seconds > 0 the warm is the
        # longest idle-shaped window in the lease, and touching after it invites
        # an eviction that empties the cache slot mid-warm (review r1 F6).
        transcription_tools._touch_transcription_time()
        # Prove the device, not the load: one short inference, CPU fallback on a CUDA lib error.
        device, transcribe_ms = _warm_inference(model, model_name, stt_config)
        # Register with the idle-lifetime mechanism: a model loaded ONLY by
        # warm-up (mic open → cancel, no transcription) would otherwise sit
        # resident with no watcher, silently ignoring
        # stt.local.unload_after_idle_seconds until some later successful
        # transcription starts one. Also covers the reload-after-prior-eviction
        # case (the previous watcher exited after unloading).
        idle_timeout = transcription_tools._get_idle_unload_seconds(local_cfg)
        if idle_timeout > 0:
            transcription_tools._start_idle_unload_watcher(idle_timeout)
    except Exception as exc:  # engine missing, download failed, bad device…
        logger.warning("[STT] warm-up for local model '%s' failed: %s", model_name, exc)
        result.update(action="error", error=str(exc))
        return result
    elapsed_ms = int((time.monotonic() - started) * 1000)
    result.update(warmed=True, action="cached" if hot_before else "loaded", elapsed_ms=elapsed_ms,
                  model=model_name, device=device, transcribe_ms=transcribe_ms)
    logger.info("[STT] warm-up local model '%s': %s in %dms (model=%s, device=%s, transcribe_ms=%d)",
                model_name, result["action"], elapsed_ms, model_name, device, transcribe_ms)
    return result


def acquire_stt_lease(
    lease: str, stt_config: Optional[dict[str, Any]] = None, *, install: bool = True
) -> dict[str, Any]:
    """Register ``lease`` (e.g. ``desktop:voice-input:abcd1234``) and warm the engine.

    Re-acquiring is idempotent but still re-warms (cheap on a cache hit; heals
    a cache the idle-unload watcher cleared meanwhile). ``install`` is forwarded to
    :func:`warm_stt_provider`: a surface that must never start a dependency rebuild passes
    ``install=False``.
    """
    with _stt_lease_lock:
        _stt_leases.add(lease)
        holders = len(_stt_leases)
    return {**warm_stt_provider(stt_config, install=install), "leases": holders}


def release_stt_lease(lease: str) -> dict[str, Any]:
    """Drop ``lease``. A never-acquired lease is a no-op (still reports the
    holder count) so surfaces can call this unconditionally. Never unloads the
    model — see the module docstring for why this diverges from TTS leases."""
    with _stt_lease_lock:
        _stt_leases.discard(lease)
        holders = len(_stt_leases)
    return {"leases": holders}


def stt_lease_holders() -> list[str]:
    """Snapshot of live lease names (diagnostics / tests)."""
    with _stt_lease_lock:
        return sorted(_stt_leases)


def _reset_stt_leases_for_tests() -> None:
    with _stt_lease_lock:
        _stt_leases.clear()
