"""ONE definition of the environment a Hermes test process may see.

Three places used to answer the same question with three different lists:

* ``scripts/run_tests.sh`` hand-built an ``env -i`` allowlist in bash (two loops plus a
  ``ProgramFiles(x86)`` special case) and dropped everything else;
* ``scripts/run_tests_parallel.py`` inherited that environment verbatim for each per-file
  pytest child;
* ``tests/conftest.py`` blanked a *different* list (credential-shaped names plus a
  hand-written ``_HERMES_BEHAVIORAL_VARS``) and left every other ambient name in place.

They disagreed, and the disagreement is what made a test pass under ``pytest`` and fail
under ``scripts/run_tests.sh``. Measured on one developer host, a bare ``pytest`` run gave
the child 111 environment names where the canonical runner gave it 23 — including
``TERMINAL_*``, ``HERMES_SESSION_*`` and the operator's ``TMP``/``TEMP`` pointing into the
real ``~/.hermes/cache/scratch``. ``HERMES_SCRATCH_DIR`` (the marker that says "Hermes
exported these temp pointers") was dropped by ``env -i`` while ``TEMP``/``TMP`` were
forwarded by the Windows location block, so conftest could no longer tell an exported temp
pointer from an operator one, and a fixture that seeds the ambient environment handed the
real scratch dir to a child process.

This module is the single source of truth:

* :func:`runner_env` — what ``scripts/run_tests.sh`` hands the runner (the bash script asks
  for this; it no longer enumerates anything itself).
* :func:`forwarded_env` — what ``scripts/run_tests_parallel.py`` gives every pytest child.
* :func:`blank_names` — what ``tests/conftest.py`` deletes before tests can see it, so a
  bare ``pytest`` run agrees with the canonical runner by construction.
* :func:`sanitize_temp_vars` — the one place that decides a temp pointer may not point into
  a Hermes home (literal *and* resolved, because ``~/.hermes/cache/scratch`` is frequently a
  symlink into disk-backed storage).

Pure stdlib and import-free of Hermes itself: the bash launcher imports it before any
dependency is guaranteed to be installed.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Mapping, MutableMapping

# ── Deterministic run parameters (set, never inherited) ───────────────────────────────────
DETERMINISTIC: dict[str, str] = {
    "TZ": "UTC",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONHASHSEED": "0",
    "PYTHONUTF8": "1",
}

# ── Always forwarded ─────────────────────────────────────────────────────────────────────
# PATH locates every binary a test spawns; HOME is the one location tests may read (the
# conftest redirects HERMES_HOME, not HOME, so subprocess tests keep a stable home).
BASE_VARS = ("PATH", "HOME")

# ── Temp pointers: forwarded, but sanitised ───────────────────────────────────────────────
# TMPDIR is the runner's per-attempt disk-backed root and must survive (the runner finds
# orphaned children by it). TMP/TEMP are the *other* two spellings of the same thing; on
# POSIX they are only ever operator leftovers, so they are neither allowlisted nor forwarded
# (they used to be, via the Windows block, which is how a developer's real
# ~/.hermes/cache/scratch reached the test suite). On Windows they are required and are in
# WINDOWS_LOCATION_VARS.
TEMP_VARS = ("TMPDIR", "TMP", "TEMP")

# ── Native Windows location variables ─────────────────────────────────────────────────────
# `env -i` forwards HOME, which is enough on POSIX. Native Windows CPython resolves
# Path.home() from USERPROFILE (or HOMEDRIVE+HOMEPATH), stdlib platform paths come from
# LOCALAPPDATA/APPDATA, ssl/sockets need SYSTEMROOT, and tempfile needs TEMP/TMP. Dropping
# them breaks collection on native Windows (issues #67385, #70813). PATHEXT is also
# required: without .EXE, PowerShell opens a native child as a document without waiting for
# its exit. These are location variables, not credentials.
#
# TEMP/TMP are in this list *for Windows only*. They used to be forwarded on POSIX too,
# which is how a developer's Hermes scratch dir reached the test suite.
WINDOWS_LOCATION_VARS = (
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "SYSTEMROOT",
    "TEMP", "TMP",
    "ComSpec", "PATHEXT", "PROGRAMFILES", "ProgramFiles", "PROGRAMDATA", "ProgramData",
    # Parenthesised name: cannot be read with bash's ${!var}, which is why the launcher no
    # longer reads names itself — this module emits the assignment.
    "ProgramFiles(x86)",
)

# Native build toolchain: Windows arm64 has no wheels for every pinned C extension, so
# `uv sync` inside a PM test compiles ruamel-yaml-clib and friends. The MSVC developer
# environment is exported by scripts/build/windows-deps.ps1 into the job env; without
# INCLUDE/LIB/VSINSTALLDIR the build backend reports "Visual C++ 14.0 or greater is
# required". These describe compiler locations, not credentials.
WINDOWS_TOOLCHAIN_VARS = (
    "INCLUDE", "LIB", "LIBPATH", "VSINSTALLDIR", "VCINSTALLDIR", "VCToolsInstallDir",
    "VCToolsVersion", "VCToolsRedistDir", "WindowsSdkDir", "WindowsSDKVersion",
    "WindowsSdkBinPath", "WindowsSdkVerBinPath", "WindowsLibPath", "UCRTVersion",
    "UniversalCRTSdkDir", "VSCMD_ARG_HOST_ARCH", "VSCMD_ARG_TGT_ARCH", "VSCMD_VER",
    "DevEnvDir", "ExtensionSdkDir", "Platform", "CARGO_HOME", "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN", "CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_LINKER",
    "CC_aarch64_pc_windows_msvc", "CC", "CXX", "AR", "VCPKG_ROOT", "OPENSSL_DIR",
    "OPENSSL_STATIC", "OPENSSL_LIB_DIR", "OPENSSL_INCLUDE_DIR",
)

# ── Test-infrastructure knobs ─────────────────────────────────────────────────────────────
# Documented knobs must survive the hermetic environment or they are silent no-ops:
#
#   * HERMES_TEST_WORKERS / PATHS / FILE_TIMEOUT / FILE_RETRIES / SLICE are read by
#     run_tests_parallel.py at argparse-default time — inside the stripped environment.
#   * HERMES_TEST_IMAGE is read by tests/docker/conftest.py to skip its session-scoped
#     `docker build`. CI's docker.yml sets it to the image the build step just loaded;
#     stripping it made every per-file pytest subprocess rebuild the 5GB image from a cold
#     builder cache (~4 min per worker per run), and the rebuilt image lacked the
#     HERMES_GIT_SHA build-arg the workflow bakes in.
#   * HERMES_E2E_REQUIRE_TUI turns a missing Ink TUI build into a failure in
#     tests/e2e/core/terminal instead of a skip (set by the e2e CI job).
#   * CI / GITHUB_ACTIONS tell suites they run on a disposable runner (e.g.
#     tests/e2e/core/upgrade runs the real updater unsandboxed only there).
#   * HERMES_E2E_WINDOWS_INSTALL opts tests/e2e/core/windows_update into running the real
#     install.ps1 (it writes HKCU PATH); HERMES_E2E_MACHINE_ROOT, HERMES_E2E_PROFILES_ROOT
#     and HERMES_E2E_ARTIFACTS place its fake machines, their user profiles and the
#     transcripts CI uploads.
#   * HERMES_E2E_STRICT_ACCEPTANCE makes tests/e2e/core/_pending_fixes.py's known gaps hard
#     failures in an acceptance run.
#   * HERMES_TEST_SSH_* / HERMES_TEST_DOCKER_IMAGE arm the real-backend tests
#     (tests/tools/test_ssh_environment.py, tests/tools/test_docker_backend_real.py);
#     remote-backends.yml points them at a throwaway sshd container and a small public
#     image. The key is a per-run throwaway, never a credential.
#
# SSL_CERT_FILE/DIR are trust-store locations: the pinned interpreter's OpenSSL has no
# compiled-in bundle path on NixOS, so network tests (PM downloads, channel reads) need the
# host's pointer to verify TLS. Keep this an explicit list (no HERMES_TEST_* glob) so the
# "no credential can leak" property stays auditable at a glance.
TEST_KNOB_VARS = (
    "HERMES_TEST_IMAGE", "HERMES_TEST_WORKERS", "HERMES_TEST_PATHS",
    "HERMES_TEST_FILE_TIMEOUT", "HERMES_TEST_FILE_RETRIES", "HERMES_TEST_SLICE",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "HERMES_GATEWAY_LOCK_DIR", "CI", "GITHUB_ACTIONS",
    "HERMES_E2E_WINDOWS_INSTALL", "HERMES_E2E_MACHINE_ROOT", "HERMES_E2E_PROFILES_ROOT",
    "HERMES_E2E_ARTIFACTS", "HERMES_E2E_STRICT_ACCEPTANCE", "HERMES_TEST_SSH_HOST",
    "HERMES_TEST_SSH_USER", "HERMES_TEST_SSH_PORT", "HERMES_TEST_SSH_KEY",
    "HERMES_TEST_DOCKER_IMAGE",
    # Opt-ins, same class as the knobs above.
    "HERMES_RUN_SLOW_PET_TESTS", "HERMES_E2E_BROWSER", "HERMES_RUN_E2E",
    # Launcher/plugin plumbing: PYTHONPATH carries the developer's live-gateway pytest
    # plugin, PYTEST_PLUGINS enables it.
    "PYTHONPATH", "PYTEST_PLUGINS",
)

# ── Operator override ──────────────────────────────────────────────────────────────────────
# A genuinely custom (non-production) HERMES_HOME is honored by tests/conftest.py as the
# session home; production values are sandboxed. It has to be in the allowlist for that
# contract to hold in both paths — otherwise ``env -i`` silently dropped it and the same
# shell produced a different session home under ``pytest`` and under ``scripts/run_tests.sh``.
OPERATOR_OVERRIDE_VARS = ("HERMES_HOME",)

#: POSIX temp pointer that survives the allowlist; it is sanitised, then forwarded.
POSIX_TEMP_VARS = ("TMPDIR",)

#: Names a test process may see, on any platform.
ALLOWED_VARS: tuple[str, ...] = (
    *BASE_VARS, *POSIX_TEMP_VARS, *TEST_KNOB_VARS, *OPERATOR_OVERRIDE_VARS, *DETERMINISTIC,
)


def allowed_names(platform: str | None = None) -> tuple[str, ...]:
    """Every name a test process may see on *platform* (``sys.platform`` vocabulary)."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return (*ALLOWED_VARS, *WINDOWS_LOCATION_VARS, *WINDOWS_TOOLCHAIN_VARS)
    return ALLOWED_VARS


def _present(environ: Mapping[str, str], names: Iterable[str]) -> dict[str, str]:
    return {name: environ[name] for name in names if environ.get(name)}


def forwarded_env(environ: Mapping[str, str], platform: str | None = None) -> dict[str, str]:
    """The environment one pytest child gets: deterministic params plus allowlisted values."""
    env = {name: value for name, value in _present(environ, allowed_names(platform)).items()}
    env.update(DETERMINISTIC)
    return env


def runner_env(environ: Mapping[str, str], platform: str | None = None) -> list[str]:
    """``NAME=VALUE`` assignments for ``scripts/run_tests.sh``'s ``env -i``.

    Values are resolved *here* (in the caller's environment), so the bash script never has
    to read a name with ``${!var}`` — which is what made ``ProgramFiles(x86)`` a special
    case and, more importantly, what let the script's list drift from conftest's.
    """
    pairs = [f"{name}={value}" for name, value in forwarded_env(environ, platform).items()]
    return pairs


def blank_names(environ: Mapping[str, str], platform: str | None = None) -> set[str]:
    """Names present in *environ* that a test process must not see.

    ``tests/conftest.py`` deletes these at import, so a bare ``pytest`` run exposes exactly
    what the canonical runner does. pytest's own ``PYTEST_*`` variables are exempt: they are
    written by pytest itself per test and by the runner's temp-root machinery.
    """
    allowed = set(allowed_names(platform))
    return {
        name for name in environ
        if name not in allowed and not name.startswith("PYTEST_")
    }


def sanitize_temp_vars(
    environ: MutableMapping[str, str],
    home_roots: Iterable[os.PathLike | str],
) -> list[str]:
    """Drop temp pointers that point into a Hermes home; return the dropped names.

    Both the literal and the resolved path are compared. ``$HERMES_HOME/cache/scratch`` is
    routinely a symlink into disk-backed storage, so a resolved-only comparison lets the
    pointer survive and hands the real home to every ``tempfile`` default — the failure that
    put ``hermes-snap-*.sh`` under the operator's ``~/.hermes`` during a test run.

    A pointer into a Hermes home is never a test's own choice: it is either Hermes' boot
    export (``export_scratch_tmp_env``) or an inherited shell. Tests that want a temp dir set
    their own.
    """
    roots = [os.path.abspath(os.fspath(root)) for root in home_roots if root]
    # A Hermes home owns exactly one scratch tree, ``<home>/cache/scratch``, and that tree is
    # commonly a symlink into disk-backed storage. Both spellings of every home are therefore
    # matched, so a pointer that is the literal home path *and* one that is the resolved
    # symlink target are recognised as Hermes' own.
    forbidden: set[str] = set()
    for root in roots:
        resolved = os.path.realpath(root)
        for base in (root, resolved):
            forbidden.add(base)
            forbidden.add(os.path.join(base, "cache", "scratch"))
        forbidden.add(os.path.realpath(os.path.join(root, "cache", "scratch")))
    dropped: list[str] = []
    for name in TEMP_VARS:
        value = (environ.get(name) or "").strip()
        if not value:
            continue
        literal = os.path.abspath(os.path.expanduser(value))
        candidates = {literal}
        try:
            candidates.add(os.path.realpath(literal))
        except OSError:
            pass
        if any(_within(candidate, root) for candidate in candidates for root in forbidden):
            del environ[name]
            dropped.append(name)
    return dropped


def _within(path: str, root: str) -> bool:
    if os.path.normcase(path) == os.path.normcase(root):
        return True
    return os.path.normcase(path).startswith(os.path.normcase(root) + os.sep)


def _main(argv: list[str]) -> int:
    """``--runner-env``: emit the assignments for ``scripts/run_tests.sh``, one per line."""
    if argv[1:2] != ["--runner-env"]:
        print("usage: env_policy.py --runner-env", file=sys.stderr)
        return 2
    for line in runner_env(os.environ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
