"""CUDA 12 preload for local STT: the resolver, the load path and the warm inference.

ctranslate2 only dlopens ``libcublasLt.so.12``/``libcublas.so.12`` at the FIRST inference, so a
model constructed with ``device="auto"`` on a GPU host proves nothing about CUDA (plan
stt-cuda-survives-update, r5 F1). Three contracts are under test:

1. ``preload_cuda_libraries`` finds the wheels the running interpreter imports (``find_spec``,
   never ``nvidia/*/lib``, never a ``.so.13``), loads Lt before cuBLAS with ``RTLD_GLOBAL``,
   is idempotent and reports the sonames it could not load by name.
2. ``_load_local_whisper_model`` preloads for ``auto``/``cuda`` and, under ``auto`` with sonames
   missing, builds the CPU model straight away instead of a CUDA model that dies on the first
   clip. A pinned ``cuda`` stays a CUDA failure (degraded), not a silent device change.
3. ``warm_stt_provider`` runs one inference, so its result carries the EFFECTIVE device after
   any CPU fallback — and a cached model is warmed too, because a cache hit has not proved
   CUDA either.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.transcription_local as tl
import tools.transcription_tools as tt
from tools import stt_lease


# ============================================================================
# Fakes: the CUDA wheel tree, the loader, and a whisper model that fails late
# ============================================================================

CUBLAS12_FILES = ("libcublas.so.12", "libcublas.so.12.9.3.13",
                  "libcublasLt.so.12", "libcublasLt.so.12.9.3.13")


class _FakeCtypes:
    """Stands in for the ``ctypes`` module so the resolver's dlopen calls are observable."""

    RTLD_GLOBAL = ctypes.RTLD_GLOBAL

    def __init__(self):
        self.loaded: list[tuple[str, int]] = []

    def CDLL(self, path, mode=0):
        self.loaded.append((Path(path).name, mode))
        return object()


@pytest.fixture(autouse=True)
def _fresh_preload_state(monkeypatch):
    """Every test starts with an empty process-wide preload cache."""
    monkeypatch.setattr(tl, "_PRELOADED_CUDA_SONAMES", set(), raising=False)


def _install_tree(tmp_path, monkeypatch, files, *, package="nvidia.cublas"):
    """Fake ``find_spec(package)`` pointing at a lib/ dir holding *files*; returns the loader fake."""
    pkg_dir = tmp_path / "site-packages" / "nvidia" / "cublas"
    lib_dir = pkg_dir / "lib"
    lib_dir.mkdir(parents=True)
    for name in files:
        (lib_dir / name).write_bytes(b"")
    spec = SimpleNamespace(submodule_search_locations=[str(pkg_dir)], origin=None)
    monkeypatch.setattr(
        tl, "_ilu",
        SimpleNamespace(find_spec=lambda name: spec if name == package else None))
    fake = _FakeCtypes()
    monkeypatch.setattr(tl, "ctypes", fake)
    return fake


class _FakeWhisper:
    """A model whose LOAD succeeds and whose first iteration fails, like the real cuBLAS dlopen.

    ``model.model.device`` mirrors ``ctranslate2.models.Whisper.device`` (the resolved device,
    ``"auto"`` already resolved by the time faster-whisper hands the model back).
    """

    def __init__(self, device="cuda", error=None):
        self.model = SimpleNamespace(device=device)
        self.error = error
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        error = self.error

        def _segments():
            if error is not None:
                raise error
            return
            yield  # pragma: no cover - makes this a generator, like faster-whisper

        return _segments(), SimpleNamespace(language="en", duration=1.0)


# ============================================================================
# 1. The resolver
# ============================================================================


def test_resolver_finds_cu12_and_loads_lt_before_cublas(tmp_path, monkeypatch):
    fake = _install_tree(tmp_path, monkeypatch, CUBLAS12_FILES)

    assert tl.preload_cuda_libraries() == ()

    # Dependency order: libcublas.so.12 needs the Lt soname resolved first.
    assert [name for name, _mode in fake.loaded] == ["libcublasLt.so.12", "libcublas.so.12"]
    assert all(mode & ctypes.RTLD_GLOBAL for _name, mode in fake.loaded)


def test_resolver_rejects_a_cu13_only_tree(tmp_path, monkeypatch):
    fake = _install_tree(tmp_path, monkeypatch, ("libcublas.so.13", "libcublasLt.so.13"))

    assert tl.preload_cuda_libraries() == ("libcublasLt.so.12", "libcublas.so.12")
    assert fake.loaded == []


def test_resolver_names_a_missing_soname(tmp_path, monkeypatch):
    # A tree with cuBLAS but without its Lt dependency: report exactly the name that is absent.
    fake = _install_tree(tmp_path, monkeypatch, ("libcublas.so.12", "libcublas.so.12.9.3.13"))

    assert tl.preload_cuda_libraries() == ("libcublasLt.so.12",)
    assert [name for name, _mode in fake.loaded] == ["libcublas.so.12"]


def test_resolver_reports_every_soname_when_the_wheel_is_absent(monkeypatch):
    monkeypatch.setattr(tl, "_ilu", SimpleNamespace(find_spec=lambda name: None))
    fake = _FakeCtypes()
    monkeypatch.setattr(tl, "ctypes", fake)

    assert tl.preload_cuda_libraries() == ("libcublasLt.so.12", "libcublas.so.12")
    assert fake.loaded == []


def test_resolver_is_idempotent(tmp_path, monkeypatch):
    fake = _install_tree(tmp_path, monkeypatch, CUBLAS12_FILES)

    assert tl.preload_cuda_libraries() == ()
    assert tl.preload_cuda_libraries() == ()
    assert [name for name, _mode in fake.loaded] == ["libcublasLt.so.12", "libcublas.so.12"]


def test_resolver_reports_a_soname_that_fails_to_load(tmp_path, monkeypatch):
    fake = _install_tree(tmp_path, monkeypatch, CUBLAS12_FILES)

    real_cdll = fake.CDLL
    fake.CDLL = lambda path, mode=0: (_ for _ in ()).throw(OSError("cannot open shared object"))
    assert tl.preload_cuda_libraries() == ("libcublasLt.so.12", "libcublas.so.12")
    fake.CDLL = real_cdll


# ============================================================================
# 2. The load path (_load_local_whisper_model)
# ============================================================================


@pytest.fixture
def created(monkeypatch):
    """Record the (device, compute_type) of every model construction and skip any real load."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tl, "_create_whisper_model",
        lambda name, *, device, compute_type: calls.append((device, compute_type)) or object())
    monkeypatch.setattr(tl, "_should_force_faster_whisper_cpu", lambda: False)
    return calls


def test_auto_with_missing_sonames_loads_straight_on_cpu(monkeypatch, created, caplog):
    monkeypatch.setattr(tl, "preload_cuda_libraries",
                        lambda: ("libcublasLt.so.12", "libcublas.so.12"))

    with caplog.at_level(logging.INFO, logger="tools.transcription_tools"):
        tl._load_local_whisper_model("small", device="auto", compute_type="auto")

    assert created == [("cpu", "int8")]
    # The single INFO line points at the gateway's start warning and names the fix.
    assert "hermes pm install --extra stt-whisper-cuda" in caplog.text
    assert "gateway start" in caplog.text


def test_auto_with_sonames_present_keeps_the_requested_device(monkeypatch, created):
    monkeypatch.setattr(tl, "preload_cuda_libraries", lambda: ())

    tl._load_local_whisper_model("small", device="auto", compute_type="auto")

    assert created == [("auto", "auto")]


def test_pinned_cuda_with_missing_sonames_is_a_cuda_failure(monkeypatch, created, caplog):
    """device: cuda must not silently become a CPU pick — the failure is reported as degraded."""
    monkeypatch.setattr(tl, "preload_cuda_libraries", lambda: ("libcublas.so.12",))

    with caplog.at_level(logging.INFO, logger="tools.transcription_tools"):
        tl._load_local_whisper_model("small", device="cuda", compute_type="float16")

    assert created == [("cuda", "float16")]
    assert "libcublas.so.12" in caplog.text


def test_pinned_device_cpu_never_touches_the_resolver(monkeypatch, created):
    calls = []
    monkeypatch.setattr(tl, "preload_cuda_libraries", lambda: calls.append(1) or ())

    tl._load_local_whisper_model("small", device="cpu", compute_type="int8")

    assert calls == []
    assert created == [("cpu", "int8")]


def test_pinned_cuda_load_failure_still_falls_back_to_cpu(monkeypatch):
    """The existing load-time safety net stays: a CUDA library error at load → CPU + int8."""
    monkeypatch.setattr(tl, "preload_cuda_libraries", lambda: ())
    created = []

    def _create(name, *, device, compute_type):
        created.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        return object()

    monkeypatch.setattr(tl, "_create_whisper_model", _create)
    monkeypatch.setattr(tl, "_should_force_faster_whisper_cpu", lambda: False)

    tl._load_local_whisper_model("small", device="cuda", compute_type="float16")

    assert created == [("cuda", "float16"), ("cpu", "int8")]


# ============================================================================
# 3. The warm inference inside warm_stt_provider
# ============================================================================


@pytest.fixture
def local_provider(monkeypatch):
    monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", True)
    monkeypatch.setattr(
        tt, "_load_stt_config",
        lambda: {"local": {"model": "small", "device": "auto"}})
    monkeypatch.setattr(tt, "_get_provider", lambda cfg: "local")
    monkeypatch.setattr(tt, "_local_model", None)
    monkeypatch.setattr(tt, "_local_model_name", None)
    monkeypatch.setattr(tt, "_touch_transcription_time", lambda: None)
    monkeypatch.setattr(tt, "_get_idle_unload_seconds", lambda cfg: 0)


def _patch_replace(monkeypatch, cpu_model, replaced):
    def _replace(model_name):
        replaced.append(model_name)
        tt._local_model, tt._local_model_name = cpu_model, model_name
        return cpu_model

    monkeypatch.setattr(tt, "_replace_cached_model_on_cpu", _replace)


def test_first_inference_cublas_failure_makes_the_warm_report_cpu(local_provider, monkeypatch):
    """RED: the load succeeds, the first iteration raises — today the warm calls this CUDA."""
    cuda_model = _FakeWhisper(
        "cuda", error=RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"))
    cpu_model = _FakeWhisper("cpu")
    monkeypatch.setattr(tt, "_get_or_load_local_model", lambda name, cfg: cuda_model)
    replaced: list[str] = []
    _patch_replace(monkeypatch, cpu_model, replaced)

    result = stt_lease.warm_stt_provider()

    assert result["warmed"] is True
    assert result["device"] == "cpu"
    assert result["model"] == "small"
    assert replaced == ["small"]
    assert tt._local_model is cpu_model
    assert cpu_model.calls, "the CPU model must be proven too"


def test_warm_reports_the_resolved_device_and_the_transcribe_duration(local_provider, monkeypatch):
    cuda_model = _FakeWhisper("cuda")
    monkeypatch.setattr(tt, "_get_or_load_local_model", lambda name, cfg: cuda_model)

    result = stt_lease.warm_stt_provider()

    assert result["device"] == "cuda"
    assert result["model"] == "small"
    assert isinstance(result["transcribe_ms"], int)
    assert result["action"] == "loaded"
    # One second of silence, VAD off: silence through the VAD would prove nothing.
    assert cuda_model.calls[0]["vad_filter"] is False
    assert "vad_parameters" not in cuda_model.calls[0]


def test_warm_runs_the_inference_on_a_cache_hit(local_provider, monkeypatch):
    """Critic r6 F1: a cached model has not proved CUDA — the warm must re-run every lease."""
    cached = _FakeWhisper("cuda")
    tt._local_model, tt._local_model_name = cached, "small"
    monkeypatch.setattr(tt, "_get_or_load_local_model", lambda name, cfg: cached)

    result = stt_lease.warm_stt_provider()

    assert result["action"] == "cached"
    assert cached.calls, "a cache hit still needs the inference"


def test_warm_inference_failure_that_is_not_cuda_is_reported(local_provider, monkeypatch):
    broken = _FakeWhisper("cuda", error=RuntimeError("CUDA out of memory"))
    monkeypatch.setattr(tt, "_get_or_load_local_model", lambda name, cfg: broken)

    result = stt_lease.warm_stt_provider()

    assert result["warmed"] is False
    assert result["action"] == "error"
    assert "CUDA out of memory" in result["error"]


# ============================================================================
# 4. install=False
# ============================================================================


def test_install_false_skips_the_lazy_install(local_provider, monkeypatch):
    monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
    installs: list[int] = []
    monkeypatch.setattr(tt, "_try_lazy_install_stt", lambda: installs.append(1) or True)

    result = stt_lease.warm_stt_provider(install=False)

    assert installs == []
    assert result["action"] == "error"
    assert result["error"] == "faster-whisper not installed"


def test_default_warm_still_lazy_installs(local_provider, monkeypatch):
    monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
    installs: list[int] = []
    monkeypatch.setattr(tt, "_try_lazy_install_stt", lambda: installs.append(1) or False)

    stt_lease.warm_stt_provider()

    assert installs == [1]


def test_acquire_lease_forwards_install_false(local_provider, monkeypatch):
    monkeypatch.setattr(tt, "_HAS_FASTER_WHISPER", False)
    installs: list[int] = []
    monkeypatch.setattr(tt, "_try_lazy_install_stt", lambda: installs.append(1) or True)
    stt_lease._reset_stt_leases_for_tests()

    try:
        result = stt_lease.acquire_stt_lease("discord:voice:42", install=False)
        assert installs == []
        assert result["action"] == "error"
        assert stt_lease.stt_lease_holders() == ["discord:voice:42"]
    finally:
        stt_lease._reset_stt_leases_for_tests()
