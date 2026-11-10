"""The gateway's start-of-process check for local STT on CUDA (plan step 4).

Two things are under test here:

1. ``tools.stt_startup.check_local_stt_at_startup`` — when it warns (a visible GPU with sonames
   the resolver cannot load; ``device: cuda`` with no visible GPU), when it stays silent (a
   CPU-only host, ``device: cpu``), and that it never installs and never warms where CUDA
   cannot work.
2. Its placement in ``gateway/run_startup.py``: a separate fire-and-forget task on an executor
   thread. The turn-machinery warm-up it sits beside is awaited with the inbound gate shut for
   up to 20 s, and importing ctranslate2 / loading medium must never delay gateway start.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import types

import pytest

from tools import stt_startup
from gateway.run_startup import GatewayStartupMixin

LOCAL_CFG = {"local": {"model": "medium", "device": "auto"}}
BOTH_SONAMES = ("libcublasLt.so.12", "libcublas.so.12")
HINT = "hermes pm install --extra stt-whisper-cuda"


class _Runner(GatewayStartupMixin):
    def __init__(self, *, multiplex: bool = False):
        self.config = types.SimpleNamespace(multiplex_profiles=multiplex)


@pytest.fixture
def stt(monkeypatch):
    """A local provider whose faster-whisper import, preload and CUDA probe are all observable."""
    from tools import stt_lease, transcription_tools

    monkeypatch.setattr(transcription_tools, "_get_provider", lambda cfg: "local")
    monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: LOCAL_CFG)
    monkeypatch.setattr(stt_startup, "_faster_whisper_available", lambda: True)
    monkeypatch.setattr(stt_startup, "preload_cuda_libraries", lambda: ())
    monkeypatch.setattr(stt_startup, "_cuda_device_count", lambda: 1)
    warmed: list[dict] = []
    monkeypatch.setattr(
        stt_lease, "warm_stt_provider",
        lambda cfg=None, provider=None, install=True: warmed.append(
            {"cfg": cfg, "provider": provider, "install": install})
        or {"provider": "local", "warmed": True, "action": "loaded", "model": "medium",
            "device": "cuda", "transcribe_ms": 42, "elapsed_ms": 900})
    return types.SimpleNamespace(warmed=warmed)


def _warnings(caplog):
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def test_cloud_provider_is_skipped_without_a_word(stt, caplog, monkeypatch):
    from tools import transcription_tools

    monkeypatch.setattr(transcription_tools, "_get_provider", lambda cfg: "openai")

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup({"provider": "openai"})

    assert result["action"] == "skipped"
    assert result["reason"] == "provider-not-local"
    assert _warnings(caplog) == []
    assert stt.warmed == []


def test_device_cpu_never_warns_and_never_warms(stt, caplog, monkeypatch):
    monkeypatch.setattr(stt_startup, "preload_cuda_libraries", lambda: BOTH_SONAMES)

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(
            {"local": {"model": "medium", "device": "cpu"}})

    assert result["action"] == "skipped"
    assert result["reason"] == "device-cpu"
    assert _warnings(caplog) == []
    assert stt.warmed == []


def test_cpu_only_host_never_warns_and_never_warms(stt, caplog, monkeypatch):
    monkeypatch.setattr(stt_startup, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(stt_startup, "preload_cuda_libraries", lambda: BOTH_SONAMES)

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(LOCAL_CFG)

    assert result["cuda_devices"] == 0
    assert _warnings(caplog) == []
    assert stt.warmed == []


def test_warns_once_when_a_gpu_is_visible_and_sonames_are_missing(stt, caplog, monkeypatch):
    monkeypatch.setattr(stt_startup, "preload_cuda_libraries", lambda: BOTH_SONAMES)

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(LOCAL_CFG)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    for soname in BOTH_SONAMES:
        assert soname in text
    assert sys.prefix in text
    assert HINT in text
    assert result["action"] == "warned"
    assert result["reason"] == "missing-cuda-libraries"
    # Sonames missing under auto: the model is never loaded here (step 2 loads CPU on use).
    assert stt.warmed == []


def test_warns_once_when_device_cuda_with_no_visible_gpu(stt, caplog, monkeypatch):
    monkeypatch.setattr(stt_startup, "_cuda_device_count", lambda: 0)

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(
            {"local": {"model": "medium", "device": "cuda"}})

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "cuda" in text.lower()
    assert sys.prefix in text
    assert HINT in text
    assert result["action"] == "warned"
    assert result["reason"] == "cuda-device-absent"
    assert stt.warmed == []


def test_never_installs_when_faster_whisper_is_absent(stt, caplog, monkeypatch):
    from tools import transcription_tools

    monkeypatch.setattr(stt_startup, "_faster_whisper_available", lambda: False)

    def _no_count():
        raise AssertionError("ctranslate2 must not be imported when faster-whisper is absent")

    monkeypatch.setattr(stt_startup, "_cuda_device_count", _no_count)
    installs: list[int] = []
    monkeypatch.setattr(transcription_tools, "_try_lazy_install_stt",
                        lambda: installs.append(1) or True)

    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(LOCAL_CFG)

    assert result["reason"] == "faster-whisper-absent"
    assert installs == []
    assert _warnings(caplog) == []
    assert stt.warmed == []


def test_warms_through_warm_stt_provider_when_nothing_is_missing(stt, caplog, monkeypatch):
    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(LOCAL_CFG)

    assert len(stt.warmed) == 1
    assert stt.warmed[0]["install"] is False  # a boot check never starts an install
    assert result["action"] == "loaded"
    # One field record: model, effective device, transcribe duration.
    fields = [r.getMessage() for r in caplog.records if "model=" in r.getMessage()]
    assert len(fields) == 1
    assert "model=medium" in fields[0]
    assert "device=cuda" in fields[0]
    assert "transcribe_ms=42" in fields[0]


# ---- placement in the gateway --------------------------------------------------------------


def test_startup_check_runs_off_the_event_loop(monkeypatch):
    """A model load must not block the loop (which is simultaneously serving the gate)."""
    release = threading.Event()
    seen: dict = {}

    def probe():
        seen["thread"] = threading.current_thread().name
        seen["loop_ticked_while_blocked"] = None
        while not release.wait(0.01):
            if "loop_ticked" in seen:
                seen["loop_ticked_while_blocked"] = True
                break
        return {"action": "skipped"}

    monkeypatch.setattr(stt_startup, "check_local_stt_at_startup", probe)

    async def drive():
        task = asyncio.ensure_future(_Runner()._check_startup_stt())
        for _ in range(200):
            if "thread" in seen:
                break
            await asyncio.sleep(0.01)
        seen["loop_ticked"] = True  # the loop is free while the probe is still inside transcribe
        for _ in range(200):
            if seen.get("loop_ticked_while_blocked"):
                break
            await asyncio.sleep(0.01)
        release.set()
        await task

    asyncio.run(drive())

    assert seen["thread"] != threading.main_thread().name
    assert seen["loop_ticked_while_blocked"] is True


def test_startup_check_is_a_fire_and_forget_task(monkeypatch):
    release = threading.Event()

    def probe():
        release.wait(5)
        return {"action": "skipped"}

    monkeypatch.setattr(stt_startup, "check_local_stt_at_startup", probe)

    async def drive():
        runner = _Runner()
        runner._start_startup_stt_check()          # sync entry point: returns without awaiting
        assert runner._startup_stt_task is not None
        assert getattr(runner, "_startup_warmup_task", None) is None  # not the boot-gate warm-up
        await asyncio.sleep(0.05)
        assert not runner._startup_stt_task.done()  # still loading; nothing waits on it
        release.set()
        await runner._startup_stt_task

    asyncio.run(drive())
