"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib

def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def test_wake_dependencies_and_runtime_gate_agree_on_supported_targets():
    from packaging.markers import Marker, default_environment
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    optional = metadata["project"]["optional-dependencies"]
    gates = metadata["tool"]["hermes"]["extras-platforms"]
    targets = [
        ("darwin", "Darwin", "x86_64", False),
        ("darwin", "Darwin", "arm64", True),
        ("linux", "Linux", "x86_64", True),
        ("linux", "Linux", "aarch64", True),
        ("win32", "Windows", "AMD64", True),
        ("win32", "Windows", "ARM64", False),
    ]
    for system, platform_system, machine, supported in targets:
        environment = {**default_environment(), "sys_platform": system,
                       "platform_system": platform_system, "platform_machine": machine}
        for extra in ("wake", "wake-openwakeword"):
            requirement = next(req for spec in optional[extra]
                               if (req := Requirement(spec)).name == "pyopen-wakeword")
            assert requirement.marker.evaluate(environment) is supported, (extra, system, machine)
        assert Marker(gates["wake-openwakeword"]).evaluate(environment) is supported
        selected = {req.name for spec in optional["wake"]
                    if (req := Requirement(spec)).marker is None or req.marker.evaluate(environment)}
        assert {"sherpa-onnx", "pvporcupine", "sentencepiece", "pypinyin"} <= selected
        sherpa_deps = {req.name for spec in optional["wake-sherpa"]
                       if (req := Requirement(spec)).marker is None or req.marker.evaluate(environment)}
        assert {"sherpa-onnx", "sentencepiece", "pypinyin"} <= sherpa_deps
        for extra in ("wake-sherpa", "wake-porcupine"):
            assert all(req.marker is None or req.marker.evaluate(environment)
                       for req in map(Requirement, optional[extra]))
            gate = gates.get(extra)
            assert gate is None or Marker(gate).evaluate(environment)


def test_direct_overrides_preserve_the_declared_exact_version():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8-sig"))
    # uv.lock records normalized names; pyproject keeps the declared spelling (Pillow).
    direct = {canonicalize_name(req.name): req for spec in metadata["project"]["dependencies"]
              if (req := Requirement(spec)).specifier}
    for spec in metadata["tool"]["uv"].get("override-dependencies", []):
        override = Requirement(spec)
        name = canonicalize_name(override.name)
        requirement = direct.get(name)
        if requirement is None or not any(s.operator == "==" for s in requirement.specifier):
            continue
        assert override.specifier == requirement.specifier, (
            f"{override.name}: override {override.specifier} contradicts exact requirement {requirement.specifier}"
        )
        versions = {row["version"] for row in locked["package"] if canonicalize_name(row["name"]) == name}
        assert versions and all(version in requirement.specifier for version in versions)


def test_matrix_extra_not_in_all():
    """The [matrix] extra pulls `mautrix[encryption]` -> `python-olm`,
    which has Linux-only wheels and no native build path on Windows or
    modern macOS (archived libolm, C++ errors with Clang 21+).

    With matrix in [all], `uv sync --locked` on Windows tried to build
    python-olm from sdist and failed on `make`. As of 2026-05-12 the
    [matrix] extra is excluded from [all] entirely and installs on first
    use (pm.ensure_import("matrix")), where the user is expected to have
    a toolchain.
    """
    optional_dependencies = _load_optional_dependencies()

    assert "matrix" in optional_dependencies, "[matrix] extra must still exist for `uv sync --extra matrix`"
    # Must NOT appear in [all] in any form — neither unconditional nor
    # platform-gated. Lazy-install handles it.
    matrix_in_all = [
        dep for dep in optional_dependencies["all"]
        if "matrix" in dep
    ]
    assert not matrix_in_all, (
        "matrix must not appear in [all] — it installs on first use via "
        f"pm.ensure_import('matrix'). Found: {matrix_in_all}"
    )


def test_lazy_installable_extras_excluded_from_all():
    """Policy (2026-05-12): opt-in backends stay out of [all].

    On-demand install exists so one quarantined PyPI release
    (e.g. mistralai 2.4.6) can't break every fresh install. Putting a
    backend in [all] defeats that — fresh installs eager-install it and
    inherit whatever's broken upstream. Opt-in backends are extras that
    install at first use via pm.ensure_import(extra).
    """
    optional_dependencies = _load_optional_dependencies()

    # The on-demand backends as of 2026-05-12. Deliberately a literal
    # list so the test stays a contract — adding a new opt-in backend
    # means updating this list AND verifying [all] doesn't contain it.
    lazy_covered_extras = {
        "anthropic", "bedrock",
        "exa", "firecrawl", "parallel-web",
        "fal",
        "edge-tts", "tts-premium",
        "voice",  # faster-whisper / sounddevice / numpy (composes stt-whisper + audio-io)
        "stt-whisper",
        "stt-whisper-cuda",  # CUDA 12 cuBLAS for local STT; explicit opt-in only (D-014)
        "modal", "daytona", "vercel",
        "messaging", "slack", "matrix", "dingtalk", "feishu",
        "telegram", "discord",
        "wake", "wake-openwakeword", "wake-sherpa", "wake-porcupine",
        "google-chat",
        "keychain",  # keyring — vault key in the OS keychain, installed by `hermes vault key migrate`
        "mistral",  # mistralai — Voxtral STT/TTS, lazy-installed (stt.mistral / tts.mistral)
    }
    all_extra_specs = optional_dependencies["all"]
    undeclared = sorted(lazy_covered_extras - set(optional_dependencies))
    assert not undeclared, f"lazy_covered_extras names undeclared extras: {undeclared}"
    for extra in lazy_covered_extras:
        offending = [
            spec for spec in all_extra_specs
            if f"hermes-agent[{extra}]" in spec
        ]
        assert not offending, (
            f"[{extra}] is in [all] but also in LAZY_DEPS. "
            f"Remove it from [all] in pyproject.toml — it lazy-installs "
            f"at first use. Found in [all]: {offending}"
        )


def test_stt_whisper_cuda_is_a_gated_opt_in_extra_outside_stt_whisper():
    """D-014: the CUDA 12 cuBLAS wheel (~600 MB) is its own opt-in extra.

    ctranslate2 4.8.1 dlopens libcublas.so.12 / libcublasLt.so.12 at the first
    GPU inference (no cuDNN, audited with LD_DEBUG=libs), and a CUDA 13 host
    has neither. The extra exact-pins the wheel, is gated to linux/x86_64 in
    [tool.hermes.extras-platforms], and stays out of [all], [voice] and
    [stt-whisper] so a CPU-only host never downloads it.
    """
    from packaging.markers import Marker
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    optional = metadata["project"]["optional-dependencies"]
    gates = metadata["tool"]["hermes"]["extras-platforms"]
    assert "stt-whisper-cuda" in optional
    reqs = [Requirement(spec) for spec in optional["stt-whisper-cuda"]]
    assert {req.name for req in reqs} == {"nvidia-cublas-cu12"}
    (spec,) = reqs[0].specifier
    assert spec.operator == "==" and spec.version.startswith("12.")
    gate = "sys_platform == 'linux' and platform_machine == 'x86_64'"
    assert Marker(gates["stt-whisper-cuda"]) == Marker(gate)
    assert all(req.marker == Marker(gate) for req in reqs)
    for carrier in ("all", "voice", "stt-whisper"):
        assert not any("cu12" in spec or "stt-whisper-cuda" in spec for spec in optional[carrier]), carrier
    assert "stt-whisper-cuda" not in metadata["tool"]["hermes"].get("opt-in-extras", [])


def _exact_pins(specs):
    pins = {}
    for spec in specs:
        requirement = spec.split(";", 1)[0].strip()
        if "==" not in requirement:
            continue
        package, version = requirement.split("==", 1)
        package = package.split("[", 1)[0].lower().replace("_", "-")
        pins[package] = version
    return pins




def test_extras_pin_each_package_at_one_version():
    """One package, one version, across every extra.

    tools/lazy_deps.py is gone — pyproject.toml is the single authority for
    optional-dependency pins (pm syncs the venv from uv.lock, which resolves
    from here). The drift class that killed us before (#31817: two documents
    pinning the same package differently, update ping-ponging the version)
    is now only possible BETWEEN extras — so pin consistency across extras
    is the whole remaining contract.
    """
    optional_dependencies = _load_optional_dependencies()

    pins: dict[str, dict[str, set[str]]] = {}
    for extra, specs in optional_dependencies.items():
        for package, version in _exact_pins(specs).items():
            pins.setdefault(package, {}).setdefault(version, set()).add(extra)

    drift = {
        package: {v: sorted(extras) for v, extras in versions.items()}
        for package, versions in pins.items()
        if len(versions) > 1
    }
    assert not drift, (
        "extras pin the same package at different versions — uv sync would "
        f"resolve whichever wins and silently downgrade the other: {drift}"
    )






def test_dingtalk_extra_includes_qrcode_for_qr_auth():
    """DingTalk's QR-code device-flow auth (hermes_cli/dingtalk_auth.py)
    needs the qrcode package."""
    optional_dependencies = _load_optional_dependencies()

    dingtalk_extra = optional_dependencies["dingtalk"]
    assert any(dep.startswith("qrcode") for dep in dingtalk_extra)
