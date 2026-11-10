"""The gateway's start-of-process check for local STT on CUDA (plan step 4).

A gateway whose local STT is configured for CUDA must say so at start when the CUDA 12 libraries
ctranslate2 will dlopen are not loadable — the alternative is the operator hearing about it from a
failed first clip, with the fallback hidden in a mid-transcribe warning. The check is cheap and
side-effect free until it decides CUDA can actually work:

* it resolves the STT provider and ``stt.local.device`` and says nothing for anything but
  ``local`` on ``auto``/``cuda``;
* it never starts an install (``transcription_local._try_lazy_install_stt``'s docstring) — an
  absent faster-whisper means "skip", not "install";
* only then does it preload the sonames (``transcription_local.preload_cuda_libraries``) and, when
  nothing is missing and a CUDA device is visible, warm the model through
  ``stt_lease.warm_stt_provider(install=False)`` so the log proves the effective device.

Importing ctranslate2 and loading medium take seconds, so the gateway runs this on a daemon
thread as a fire-and-forget task beside the turn-machinery warm-up, never inside it
(``gateway/run_startup.py::_start_startup_stt_check``): the inbound gate must not wait for a model,
and neither must process exit — a cold download behind ``asyncio.run``'s default executor would be
joined at loop close.
"""

from __future__ import annotations

import importlib.util as _ilu
import logging
import sys
from typing import Any, Optional, Sequence

# The resolver lives with the loader that needs it (same interpreter, same wheels). Imported at
# module scope so a test can replace it here; it pulls in no CUDA/ctranslate2 work by itself.
from tools.transcription_local import preload_cuda_libraries

logger = logging.getLogger(__name__)

INSTALL_HINT = "hermes pm install --extra stt-whisper-cuda"
_CUDA_DEVICES = {"auto", "cuda"}


def _faster_whisper_available() -> bool:
    """Is faster-whisper importable in THIS interpreter right now? (A probe, never an install.)"""
    try:
        return _ilu.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def _cuda_device_count() -> int:
    """Driver-level CUDA device count via ctranslate2 — the library that will run the model."""
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except (ImportError, OSError, RuntimeError, ValueError):
        # No ctranslate2, a library the loader cannot open, or a driver query that fails:
        # all of them mean "no CUDA device this interpreter can use", never a failed check.
        return 0


def _warn_missing_cuda_libraries(missing: Sequence[str]) -> None:
    logger.warning(
        "Local STT is configured for CUDA and a CUDA device is visible, but the CUDA 12 "
        "libraries ctranslate2 dlopens at the first inference cannot be loaded: %s (neither the "
        "nvidia wheels in this interpreter's site-packages nor the dynamic loader can resolve "
        "them; prefix %s). Transcription will run on CPU. "
        "Fix: %s", ", ".join(missing), sys.prefix, INSTALL_HINT)


def _warn_cuda_device_absent(missing: Sequence[str]) -> None:
    logger.warning(
        "Local STT has stt.local.device: cuda but ctranslate2 sees no CUDA device (interpreter "
        "prefix %s, CUDA 12 libraries %s). Transcription will run on CPU (degraded). "
        "Fix: %s", sys.prefix,
        "missing: " + ", ".join(missing) if missing else "present", INSTALL_HINT)


def check_local_stt_at_startup(stt_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Warn once (and warm when CUDA can work) for local STT at gateway start.

    Returns the decision (``action`` / ``reason`` / ``missing`` / ``cuda_devices`` / ``warm``) for
    tests and diagnostics. Blocking; the caller runs it off the event loop's thread — the CUDA
    preload inside it still holds the GIL for the duration of its ``dlopen``s
    (``transcription_local.preload_cuda_libraries``).
    """
    from tools import transcription_tools

    if stt_config is None:
        stt_config = transcription_tools._load_stt_config()  # already a mapping
    # ``stt: enabled`` (a scalar instead of a mapping) must skip, not die on ``.get`` (r1 F3).
    # Shared with the lease path so both entry points degrade identically (review r2 F3).
    stt_config = transcription_tools._coerce_stt_config(stt_config)
    provider = (transcription_tools._get_provider(stt_config) or "").lower().strip()
    result: dict[str, Any] = {"provider": provider, "action": "skipped", "reason": "",
                              "missing": (), "cuda_devices": None}
    if provider != "local":
        result["reason"] = "provider-not-local"
        return result

    local_cfg = stt_config.get("local")
    if not isinstance(local_cfg, dict):
        local_cfg = {}
    requested = str(local_cfg.get("device") or "auto").strip().lower()
    result["device"] = requested
    if requested not in _CUDA_DEVICES:
        # device: cpu — no CUDA, so no warning and nothing held in RAM for the process lifetime.
        result["reason"] = "device-cpu"
        return result
    if not _faster_whisper_available():
        # A boot probe never installs: an install takes the per-install lock for a full rebuild.
        result["reason"] = "faster-whisper-absent"
        logger.debug("[STT] startup CUDA check skipped: faster-whisper is not importable")
        return result

    missing = preload_cuda_libraries()
    visible = _cuda_device_count()
    result.update(missing=missing, cuda_devices=visible)
    if visible <= 0:
        if requested == "cuda":
            # A CPU-only host under auto is the normal case; a pinned cuda with no device is not.
            _warn_cuda_device_absent(missing)
            result.update(action="warned", reason="cuda-device-absent")
            return result
        result["reason"] = "no-cuda-device"
        logger.info("[STT] startup CUDA check: no CUDA device visible — not warming "
                    "(device=%s, model=%s)", requested, local_cfg.get("model"))
        return result
    if missing:
        _warn_missing_cuda_libraries(missing)
        result.update(action="warned", reason="missing-cuda-libraries")
        return result

    from tools.stt_lease import warm_stt_provider
    # install=False: never a dependency rebuild from a boot path, and the plan's step 7 shape.
    warm = warm_stt_provider(stt_config, install=False)
    result.update(action=warm.get("action") or "warm", warm=warm)
    # One field record: model, effective device, transcribe duration (the acceptance log line).
    logger.info("[STT] startup warm: model=%s device=%s transcribe_ms=%s action=%s",
                warm.get("model"), warm.get("device"), warm.get("transcribe_ms"),
                warm.get("action"))
    return result
