"""The opt-in ``stt-whisper-cuda`` extra (CUDA 12 cuBLAS for local STT, D-014).

Its availability must prove the soname ctranslate2 dlopens (``libcublas.so.12``),
not just an importable ``nvidia`` namespace: otherwise a cu13-only tree reads as
installed, or the default ``stt_whisper_cuda`` anchor never imports and every
``ensure_import`` re-syncs. An explicit sync records it in the venv fact's extras
ledger and later syncs (``hermes update``, repair) rebuild from that ledger, so it
survives a new generation. ``stt-whisper``'s lazy install never selects it.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from pm import client, extras, install

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def site(tmp_path, monkeypatch):
    """An isolated import root: only *tmp_path/site* resolves ``nvidia.*``."""
    root = tmp_path / "site"
    root.mkdir()
    saved = {name: module for name, module in sys.modules.items() if name == "nvidia" or name.startswith("nvidia.")}
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", [str(root)])
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {})
    yield root
    for name in [n for n in sys.modules if n == "nvidia" or n.startswith("nvidia.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def _cublas(root: Path, *sonames: str) -> None:
    lib = root / "nvidia" / "cublas" / "lib"
    lib.mkdir(parents=True)
    for soname in sonames:
        (lib / soname).write_bytes(b"")


def test_anchor_is_false_without_libcublas_so_12(site):
    assert not extras.available("stt-whisper-cuda")  # no nvidia namespace at all
    (site / "nvidia" / "cublas" / "lib").mkdir(parents=True)
    assert not extras.available("stt-whisper-cuda")  # namespace present, soname absent
    assert "libcublas.so.12" in extras.missing("stt-whisper-cuda")


def test_a_cuda13_only_tree_does_not_satisfy_the_extra(site):
    _cublas(site, "libcublas.so.13", "libcublasLt.so.13")
    assert not extras.available("stt-whisper-cuda")


def test_anchor_is_true_with_libcublas_so_12(site):
    _cublas(site, "libcublas.so.12", "libcublasLt.so.12")
    assert extras.available("stt-whisper-cuda")
    assert extras.missing("stt-whisper-cuda") == ()


def test_ensure_import_does_not_resync_once_the_soname_is_present(site, monkeypatch):
    _cublas(site, "libcublas.so.12", "libcublasLt.so.12")
    monkeypatch.setattr(client, "sync_venv", lambda *a, **k: pytest.fail("present extra must not re-sync"))
    extras.ensure_import("stt-whisper-cuda")


def test_install_hint_is_the_fix_command():
    assert extras.install_hint("stt-whisper-cuda") == "hermes pm install --extra stt-whisper-cuda"


def test_legacy_selection_carries_the_extra_only_with_the_cu12_soname(tmp_path, monkeypatch):
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {})
    tree = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    lib = tree / "nvidia" / "cublas" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcublas.so.13").write_bytes(b"")
    assert "stt-whisper-cuda" not in extras.legacy_selection(tmp_path)
    (lib / "libcublas.so.12").write_bytes(b"")
    (lib / "libcublasLt.so.12").write_bytes(b"")
    assert "stt-whisper-cuda" in extras.legacy_selection(tmp_path)


# ---- selection: stt-whisper's lazy install never pulls the CUDA wheel ----


def test_stt_whisper_lazy_install_selects_only_stt_whisper(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(client, "sync_venv", lambda x=None, **k: calls.append(list(x or [])))
    monkeypatch.setattr(extras, "available", lambda e: False)
    extras.ensure_import("stt-whisper")
    assert calls == [["stt-whisper"]]


def test_locked_stt_whisper_closure_has_no_cuda12_wheel():
    """Walk uv.lock from hermes-agent[stt-whisper] (and [voice]): cuBLAS 12 is reachable only via
    [stt-whisper-cuda]."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {pkg["name"]: pkg for pkg in lock["package"]}
    project = packages["hermes-agent"]

    def closure(extra: str) -> set[str]:
        seen: set[str] = set()
        pending = [dep["name"] for dep in project["optional-dependencies"][extra]]
        while pending:
            name = pending.pop()
            if name in seen or name == "hermes-agent":
                continue
            seen.add(name)
            pkg = packages.get(name, {})
            pending.extend(dep["name"] for dep in pkg.get("dependencies", ()))
            for group in pkg.get("optional-dependencies", {}).values():
                pending.extend(dep["name"] for dep in group)
        return seen

    for extra in ("stt-whisper", "voice"):
        assert "nvidia-cublas-cu12" not in closure(extra), extra
    assert "nvidia-cublas-cu12" in closure("stt-whisper-cuda")


# ---- survival: the explicit sync is recorded in the extras ledger ----


class _Venv:
    """The venv package with a build that records each selection it was asked for."""

    def __init__(self):
        self.built: list[list[str]] = []

    @staticmethod
    def project_root():
        from pm.paths import repo_root

        return repo_root()

    def expected_stamp(self, enabled, **_):
        return "+".join(enabled)

    def apply(self, enabled, **_):
        self.built.append(list(enabled))
        return {}


def _recorded():
    from pm.environments import runtime_facts_path
    from pm.lock import Facts
    from pm.paths import repo_root

    return (Facts(runtime_facts_path(repo_root())).get("venv") or {}).get("extras")


def test_explicit_sync_records_the_extra_and_rebuilds_keep_it(monkeypatch, tmp_path):
    from pm import paths

    # A private install root carrying this tree's real pyproject, so the ledger prune
    # (_still_declared) judges the real declaration.
    root = tmp_path / "hermes"
    root.mkdir()
    (root / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(install, "_feature_policy", lambda extras, *, repair: (None, None))
    package = _Venv()
    monkeypatch.setattr(install, "get_package", lambda name: package)
    install.sync_venv(["all"], explicit=True)

    # What `hermes tools` sends for Local Whisper on an NVIDIA host.
    install.sync_venv(["stt-whisper", "stt-whisper-cuda"], explicit=True, automatic=["stt-whisper-cuda"])
    assert "stt-whisper-cuda" in _recorded()
    assert "stt-whisper-cuda" in package.built[-1]

    # `hermes update`'s bare sync and a repair rebuild from the ledger; both keep it.
    for kwargs in ({"explicit": True}, {"repair": True}):
        package.built.clear()
        tag = next(iter(kwargs))  # a new stamp, as a new source tree would give: forces a rebuild
        monkeypatch.setattr(package, "expected_stamp", lambda enabled, _t=tag, **_: f"{_t}:" + "+".join(enabled))
        install.sync_venv(**kwargs)
        assert package.built and "stt-whisper-cuda" in package.built[-1], kwargs
        assert "stt-whisper-cuda" in _recorded()

    # `hermes update` itself (hermes_cli/update_completion.py) syncs with
    # evict_incompatible_plugins=True, i.e. pm.plugin_eviction.sync_evicting. Only the
    # interpreter probe (it runs PM's pinned toolchain) is stubbed: plugin discovery,
    # _target_selection and the Facts ledger are the real code.
    from pm import plugin_eviction

    monkeypatch.setattr(plugin_eviction, "_interpreter_version", lambda: "3.13.0")
    evicting = []
    real_sync_evicting = plugin_eviction.sync_evicting

    def spy(*args, **kwargs):
        evicting.append(kwargs)
        return real_sync_evicting(*args, **kwargs)

    monkeypatch.setattr(plugin_eviction, "sync_evicting", spy)
    package.built.clear()
    monkeypatch.setattr(package, "expected_stamp", lambda enabled, **_: "update:" + "+".join(enabled))
    install.sync_venv(explicit=True, evict_incompatible_plugins=True)
    assert len(evicting) == 1 and evicting[0]["extras"] is None  # the update's bare sync
    assert package.built and "stt-whisper-cuda" in package.built[-1]
    assert "stt-whisper-cuda" in _recorded()


def test_installed_extras_requires_the_proof_sonames(tmp_path, monkeypatch):
    """The bundle inventory (pm.features.installed_extras) uses the same soname proof as
    available()/legacy_selection(): a tree with only CUDA 13 cuBLAS does not list the extra."""
    import subprocess

    from pm import features
    from pm.environments import site_packages

    target_root = tmp_path / "target"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target_root)],
                   capture_output=True, check=True, timeout=60)
    target = target_root / "bin" / "python"
    dependencies = tmp_path / "dependencies"
    tree = site_packages(dependencies)
    lib = tree / "nvidia" / "cublas" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcublas.so.13").write_bytes(b"")
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project.optional-dependencies]\n"stt-whisper-cuda"=[]\n',
                                         encoding="utf-8")
    assert not extras.libraries_in_tree(tree, "stt-whisper-cuda")
    assert features.installed_extras(repo, dependencies, python_exe=target) == []
    (lib / "libcublas.so.12").write_bytes(b"")
    assert features.installed_extras(repo, dependencies, python_exe=target) == []  # Lt still missing
    (lib / "libcublasLt.so.12").write_bytes(b"")
    assert extras.libraries_in_tree(tree, "stt-whisper-cuda")
    assert features.installed_extras(repo, dependencies, python_exe=target) == ["stt-whisper-cuda"]
