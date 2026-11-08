"""`hermes tools` → Local Whisper selects the opt-in CUDA 12 cuBLAS extra only on an NVIDIA host.

The probe is driver-level (``/proc/driver/nvidia/version`` or ``cuInit``/``cuDeviceGetCount``
on ``libcuda.so.1``) and never imports ctranslate2 or any other wheel: ``hermes tools`` usually
runs where faster-whisper isn't importable yet. D-014: a host without a GPU never gets the wheel.
"""

from __future__ import annotations

import ctypes
import importlib.abc
import sys
from unittest.mock import patch

import pm
import pytest

from hermes_cli import tools_config_post_setup as post_setup
from hermes_cli.tools_config_post_setup import _run_post_setup


class _RefuseWheelImports(importlib.abc.MetaPathFinder):
    """Fail any import of the STT wheels while installed."""

    blocked = ("ctranslate2", "faster_whisper")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.blocked:
            raise AssertionError(f"GPU probe imported {name}")


@pytest.fixture
def no_wheel_imports(monkeypatch):
    for name in [n for n in sys.modules if n.split(".")[0] in _RefuseWheelImports.blocked]:
        monkeypatch.delitem(sys.modules, name)
    finder = _RefuseWheelImports()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    yield
    assert not any(n.split(".")[0] in _RefuseWheelImports.blocked for n in sys.modules)


@pytest.mark.parametrize("gpu,expected", [
    (True, ["stt-whisper", "stt-whisper-cuda"]),
    (False, ["stt-whisper"]),
])
def test_local_whisper_selects_cuda_extra_only_with_a_gpu(monkeypatch, capsys, no_wheel_imports, gpu, expected):
    monkeypatch.setattr(post_setup, "_nvidia_gpu_present", lambda: gpu)
    monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **_: True)
    with patch("pm.sync_venv") as sync:
        _run_post_setup("faster_whisper")
    sync.assert_called_once()
    args, kwargs = sync.call_args
    assert args == (expected,) and kwargs["explicit"] is True
    if gpu:
        # The GPU-derived pick is automatic: a `--without stt-whisper-cuda` stays honored.
        assert kwargs["automatic"] == ["stt-whisper-cuda"]
        assert "CUDA" in capsys.readouterr().out
    else:
        assert "automatic" not in kwargs


def test_gpu_on_an_ungated_platform_skips_the_cuda_extra(monkeypatch):
    """An NVIDIA aarch64 box (Jetson) sees a driver, but the extra is linux/x86_64-only:
    asking for it would fail the whole Local Whisper install."""
    monkeypatch.setattr(post_setup, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **_: extra != "stt-whisper-cuda")
    with patch("pm.sync_venv") as sync:
        _run_post_setup("faster_whisper")
    sync.assert_called_once_with(["stt-whisper"], explicit=True)


def test_other_python_hooks_never_probe_the_gpu(monkeypatch):
    monkeypatch.setattr(post_setup, "_nvidia_gpu_present", lambda: pytest.fail("probe ran for piper"))
    with patch("pm.sync_venv") as sync:
        _run_post_setup("piper")
    sync.assert_called_once_with(["piper"], explicit=True)


# ---- the driver-level probe itself ----


class _FakeCuda:
    def __init__(self, init=0, count=1, status=0):
        self._init, self._count, self._status = init, count, status

    def cuInit(self, flags):
        return self._init

    def cuDeviceGetCount(self, ref):
        ref._obj.value = self._count
        return self._status


def test_probe_trusts_the_driver_proc_file_without_loading_libcuda(tmp_path, monkeypatch, no_wheel_imports):
    proc = tmp_path / "version"
    proc.write_text("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04\n")
    monkeypatch.setattr(post_setup, "_NVIDIA_PROC_VERSION", str(proc))
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("libcuda loaded"))
    assert post_setup._nvidia_gpu_present() is True


@pytest.mark.parametrize("cuda,present", [
    (_FakeCuda(count=2), True),
    (_FakeCuda(count=0), False),
    (_FakeCuda(init=100), False),  # CUDA_ERROR_NO_DEVICE
    (_FakeCuda(count=3, status=1), False),
    (OSError("libcuda.so.1: cannot open shared object file"), False),
])
def test_probe_falls_back_to_the_driver_api(tmp_path, monkeypatch, no_wheel_imports, cuda, present):
    monkeypatch.setattr(post_setup, "_NVIDIA_PROC_VERSION", str(tmp_path / "absent"))
    opened = []

    def cdll(name, *args, **kwargs):
        opened.append(name)
        if isinstance(cuda, Exception):
            raise cuda
        return cuda

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    assert post_setup._nvidia_gpu_present() is present
    assert opened == ["libcuda.so.1"]


def test_probe_on_this_host_imports_no_wheel(no_wheel_imports):
    """Real probe, whatever the host has: it answers a bool and loads no Python wheel."""
    assert post_setup._nvidia_gpu_present() in (True, False)


def test_post_setup_refusal_still_reports_retry(monkeypatch, capsys):
    monkeypatch.setattr(post_setup, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr("pm.extras.extra_supported", lambda extra, **_: True)
    with patch("pm.sync_venv", side_effect=pm.InstallError("venv", "outside frozen feature set")):
        _run_post_setup("faster_whisper")
    out = capsys.readouterr().out
    assert "outside frozen feature set" in out and "Retry with: hermes tools" in out
