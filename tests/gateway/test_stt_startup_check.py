"""The gateway's start-of-process check for local STT on CUDA (plan step 4).

Two things are under test here:

1. ``tools.stt_startup.check_local_stt_at_startup`` — when it warns (a visible GPU with sonames
   the resolver cannot load; ``device: cuda`` with no visible GPU), when it stays silent (a
   CPU-only host, ``device: cpu``), and that it never installs and never warms where CUDA
   cannot work.
2. Its placement in ``gateway/run_startup.py``: a separate fire-and-forget task on a DAEMON
   thread (not the loop's default executor, which ``asyncio.run`` joins at close). The
   turn-machinery warm-up it sits beside is awaited with the inbound gate shut for up to 20 s, and
   importing ctranslate2 / loading medium must never delay gateway start *or* process exit.
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


@pytest.mark.parametrize("bad_config", ["enabled", ["local"], 7])
def test_a_non_mapping_stt_config_degrades_to_defaults(bad_config, stt, caplog):
    """``stt: enabled`` is a str from ``_load_stt_config``; the boot probe must not die on it."""
    with caplog.at_level(logging.INFO):
        result = stt_startup.check_local_stt_at_startup(bad_config)

    # Treated as an empty section: the default device (auto) and no exception on the way.
    assert result["device"] == "auto"
    assert result["action"] == "loaded"
    assert _warnings(caplog) == []


def test_a_non_mapping_local_section_degrades_to_defaults(stt):
    result = stt_startup.check_local_stt_at_startup(
        {"provider": "local", "local": "medium"})

    assert result["device"] == "auto"
    assert result["action"] == "loaded"


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


def test_startup_check_runs_on_a_daemon_thread(monkeypatch):
    """Not the loop's DEFAULT executor: ``asyncio.run`` joins that at close (300 s) (r1 F2)."""
    release = threading.Event()
    seen: dict = {}

    def probe():
        seen["daemon"] = threading.current_thread().daemon
        seen["name"] = threading.current_thread().name
        release.wait(5)
        return {"action": "skipped"}

    monkeypatch.setattr(stt_startup, "check_local_stt_at_startup", probe)

    async def drive():
        runner = _Runner()
        runner._start_startup_stt_check()
        task = runner._startup_stt_task
        for _ in range(200):
            if "daemon" in seen:
                break
            await asyncio.sleep(0.01)
        # Shutdown-shaped cancellation must detach the load, not wait for it.
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        release.set()

    asyncio.run(drive())

    assert seen["daemon"] is True, "an executor thread would be joined by asyncio.run at close"
    assert seen["name"] == "startup-stt-check"


def test_a_failed_check_is_reported_at_warning(monkeypatch, caplog):
    """'No CUDA warning at start' must be distinguishable from 'the check died' (r1 F3)."""
    def probe():
        raise RuntimeError("boom")

    monkeypatch.setattr(stt_startup, "check_local_stt_at_startup", probe)

    async def drive():
        runner = _Runner()
        runner._start_startup_stt_check()
        await asyncio.gather(runner._startup_stt_task, return_exceptions=True)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        asyncio.run(drive())

    warnings = [record for record in caplog.records
                if record.levelno == logging.WARNING
                and "local-STT CUDA check failed" in record.getMessage()]
    assert warnings, [record.getMessage() for record in caplog.records]


# ---- the boot call site (review r1 F4, mutation M5) ------------------------------------------


class _BootRunner(GatewayStartupMixin):
    """``_start_impl`` with the pre-connect steps stubbed and ``_start_prefilter_platforms``
    aborting right after the two boot tasks are started."""

    def __init__(self):
        self.config = types.SimpleNamespace(multiplex_profiles=False)
        self.sequence: list[str] = []

    def _start_install_faulthandler(self):
        self.sequence.append("faulthandler")

    async def _start_log_startup_environment(self):
        self.sequence.append("log-startup-environment")

    async def _abort_startup_if_shutdown_requested(self, adapter=None, platform=None):
        return False

    def _start_check_access_policy(self):
        return False

    async def _start_recover_previous_run(self):
        self.sequence.append("recover-previous-run")

    def _start_startup_warmup(self):
        self.sequence.append("warmup")
        self._startup_warmup_task = object()  # a stand-in: independence is what is asserted

    async def _start_prefilter_platforms(self):
        self.sequence.append("prefilter")
        return True, 0, [], []  # aborted → `_start_impl` returns before any connect work


def test_boot_path_starts_the_stt_check_as_its_own_task(monkeypatch):
    """The call site itself: dropping it leaves the whole startup-warning feature absent."""
    ran: list[str] = []

    async def _check(self):
        ran.append("stt-check")

    monkeypatch.setattr(GatewayStartupMixin, "_check_startup_stt", _check)

    async def drive():
        runner = _BootRunner()
        assert await runner._start_impl() is True
        task = getattr(runner, "_startup_stt_task", None)
        assert task is not None, "the boot path never started the local-STT check"
        assert task is not runner._startup_warmup_task  # its own task, not the gate's warm-up
        await task
        return runner

    runner = asyncio.run(drive())

    assert ran == ["stt-check"]
    assert runner.sequence == ["faulthandler", "log-startup-environment", "recover-previous-run",
                              "warmup", "prefilter"]
