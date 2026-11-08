#!/usr/bin/env bash
# Canonical test runner for hermes-agent. Run this instead of calling
# `pytest` directly to guarantee your local run matches CI behavior.
#
# What this script enforces:
#   * Per-file isolation via scripts/run_tests_parallel.py — each test
#     file runs in its own freshly-spawned `python -m pytest <file>`
#     subprocess. No xdist, no shared workers, no module-level leakage
#     between files.
#   * TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0 (deterministic)
#   * Env vars blanked by ONE definition: tests/_fixtures/env_policy.py, which this
#     script, run_tests_parallel.py and tests/conftest.py all read. It used to be two
#     hand-maintained lists (this file's allowlist and conftest's behavioural list) that
#     drifted — see the policy module's docstring for the failure that caused.
#   * The checkout's test environment (via scripts/run-in-hermes-env when needed)
#
# Usage:
#   scripts/run_tests.sh                            # full suite
#   scripts/run_tests.sh -j 4                       # cap parallelism
#   scripts/run_tests.sh tests/agent/               # discover only here
#   scripts/run_tests.sh tests/agent/ tests/acp_adapter/    # multiple roots
#   scripts/run_tests.sh tests/foo.py               # single file
#   scripts/run_tests.sh tests/foo.py -q            # path + bare pytest flag
#   scripts/run_tests.sh tests/foo.py -v --tb=long  # bare flags "just work"
#   scripts/run_tests.sh -k 'pattern'               # value flags pass through too
#   scripts/run_tests.sh tests/foo.py -- --tb=long  # explicit '--' still works
#
# Bare pytest flags (anything starting with '-' that isn't one of this
# runner's own options: -j/--jobs, --paths, --slice, --file-timeout, etc.)
# are forwarded to each per-file pytest invocation automatically — no '--'
# separator required. The explicit '--' form still works and stacks with
# bare flags. Positional path arguments override the default discovery
# root (tests/).

set -euo pipefail

# ── Locate repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Locate python ───────────────────────────────────────────────────────────
# The suite runs under the checkout's isolated test environment (pm.testenv),
# and only there: unless the inherited environment is current for this
# checkout, this script re-executes itself under scripts/run-in-hermes-env,
# which syncs and applies it. That holds however the script is launched
# (`scripts/run_tests.sh`, `bash scripts/run_tests.sh`, from CI or a shell), so
# a branch switch or lock edit never runs the suite against the previous
# dependency set.
#
# Without an activation, an explicit HERMES_PYTHON that has pytest is honored:
# the Nix devShell's editable venv and CI's minimal installer lanes provide
# one on purpose. The import check matters: a wrapped `hermes` binary exports
# HERMES_PYTHON pointing at a release venv without pytest.
_has_pytest() { [ -n "$1" ] && [ -x "$1" ] && "$1" -c 'import pytest' 2>/dev/null; }
# shellcheck source=scripts/_activation.sh
. "$SCRIPT_DIR/_activation.sh"

# A test run under a faked HOME (worktree lanes, probe homes) provisions its
# test environment from the OS account's tool store instead of downloading
# ~1 GB of tools into every temporary home, but only while that store already
# records this checkout's pins (_store_has_pins). The check runs unlocked
# before `pm install` takes the store's install lock, so it keeps runs that
# START on different pins apart; it does not cover the window in between, where
# the account's own `hermes update` (or a run on other pins) can re-point
# facts.json and this run's `pm install` then re-stages its pins over them.
# Only this runner's own provisioning gets HERMES_RUNTIME_DIR: the pytest
# children below start from `env -i` without it, so tests that exercise setup,
# install or update resolve their own stores. Their PATH still holds the
# account store's tool dirs, so a test writing through a PATH tool
# (`pip install`, `npm -g`) would write into the account store. An explicit
# HERMES_RUNTIME_DIR, a custom HERMES_HOME, a home that already holds a store,
# or an account without a usable store keep today's per-home layout.
# Prints the account home, then the pinned Python minor.
_account_tool_store() {
  local account py_minor
  [ -z "${HERMES_RUNTIME_DIR:-}" ] || return 1
  case "${HERMES_HOME:-}" in "" | "$HOME/.hermes" | "$HOME/.hermes/"*) ;; *) return 1 ;; esac
  [ ! -f "$HOME/.hermes/tools/facts.json" ] || return 1
  account="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [ -n "$account" ] || account="$(dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2>/dev/null | sed -n 's/^NFSHomeDirectory: *//p')"
  [ -n "$account" ] && [ -d "$account" ] && [ ! "$account" -ef "$HOME" ] || return 1
  [ -f "$account/.hermes/tools/facts.json" ] && [ -w "$account/.hermes/tools" ] || return 1
  py_minor="$(_store_has_pins "$account/.hermes/tools")" || return 1
  printf '%s\n%s\n' "$account" "$py_minor"
}
# Read-only check that the store already records this checkout's pins
# (scripts/tool_store_pins.py says why and what counts); prints the Python minor.
_store_has_pins() {
  local python
  python="$(HERMES_RUNTIME_DIR="$1" hermes_bootstrap_python "$REPO_ROOT")" || return 1
  PYTHONHOME= "$python" -I -B "$REPO_ROOT/scripts/tool_store_pins.py" "$REPO_ROOT" "$1" 2>/dev/null
}
if _shared="$(_account_tool_store)"; then
  _account="${_shared%%$'\n'*}"
  export HERMES_RUNTIME_DIR="$_account/.hermes/tools"
  # setup-hermes.sh's uv-managed bootstrap interpreter is the same kind of
  # cache, but only reused: a missing pinned minor would download into it.
  if [ -z "${UV_PYTHON_INSTALL_DIR:-}" ] && [ -w "$_account/.local/share/uv/python" ] \
      && compgen -G "$_account/.local/share/uv/python/cpython-${_shared#*$'\n'}[.-]*" >/dev/null; then
    export UV_PYTHON_INSTALL_DIR="$_account/.local/share/uv/python"
  fi
  echo "▶ faked HOME: using the account's tool store $HERMES_RUNTIME_DIR"
fi
unset _account _shared
if [ -z "${__HERMES_ACTIVATED:-}" ] && _has_pytest "${HERMES_PYTHON:-}"; then
  PYTHON="$HERMES_PYTHON"
  echo "▶ not activated — using HERMES_PYTHON: $PYTHON"
else
  if ! hermes_activation_current "$REPO_ROOT"; then
    if [ -n "${__HERMES_TESTS_REEXEC:-}" ]; then
      echo "error: run-in-hermes-env did not produce a current environment for $REPO_ROOT" >&2
      exit 1
    fi
    echo "▶ environment missing or stale for $REPO_ROOT — re-running under run-in-hermes-env" >&2
    export __HERMES_TESTS_REEXEC=1
    exec "$SCRIPT_DIR/run-in-hermes-env" "$BASH" "${BASH_SOURCE[0]}" "$@"
  fi
  PYTHON="${__HERMES_TEST_PYTHON:-}"
  if ! _has_pytest "$PYTHON"; then
    echo "error: activation provided no test interpreter with pytest (__HERMES_TEST_PYTHON=${PYTHON:-unset})" >&2
    exit 1
  fi
fi


# ── Hermetic environment (ONE definition, owned by tests/_fixtures/env_policy.py) ────────
# This script does not enumerate environment names any more. It asks the policy module —
# the same module run_tests_parallel.py uses to build each child's env and conftest.py uses
# to decide what a bare `pytest` run may see — for the assignments to hand `env -i`:
#
#   * PATH/HOME (the two locations every test may read)
#   * TZ/LANG/LC_ALL/PYTHONHASHSEED/PYTHONUTF8 (deterministic)
#   * the temp pointers (TMPDIR is the runner's per-attempt root)
#   * the documented test-runner knobs (HERMES_TEST_*, SSL_CERT_*, CI, E2E opt-ins) — these
#     are read by the runner at argparse-default time and by suites such as
#     tests/docker/conftest.py, tests/e2e/core/windows_update, tests/tools/test_ssh_environment.py
#   * native Windows location + toolchain variables
#
# Why it moved here: the two lists drifted. `env -i` dropped HERMES_SCRATCH_DIR (the marker
# saying "Hermes exported these temp pointers") while the Windows block forwarded TEMP/TMP
# on POSIX, so a developer's real ~/.hermes/cache/scratch reached test processes and conftest
# could no longer recognise it as an exported value. TEMP/TMP are Windows-only now, decided
# in the module rather than by whatever the calling shell happens to export.
#
# PYTHONPATH/PYTEST_PLUGINS carry the developer's live-gateway guard; they are in the
# allowlist, so exporting them here is enough.
if [ -f "$HOME/.hermes/pytest_live_guard.py" ]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HOME/.hermes"
  export PYTEST_PLUGINS="${PYTEST_PLUGINS:+$PYTEST_PLUGINS,}pytest_live_guard"
fi

ENV_POLICY="$REPO_ROOT/tests/_fixtures/env_policy.py"
HERMETIC_ENV=()
while IFS= read -r _assignment; do
  [ -n "$_assignment" ] && HERMETIC_ENV+=("$_assignment")
done < <("$PYTHON" "$ENV_POLICY" --runner-env)

# ── Run in hermetic env ──────────────────────────────────────────────────────
# env -i: start with empty environment, opt-in only what the policy allows.
# No credential var can leak — you'd have to add it to the policy module.
# Never forward HERMES_RUNTIME_DIR or UV_PYTHON_INSTALL_DIR: under a faked HOME
# they point at the account's shared stores (see _account_tool_store).
echo "▶ running per-file parallel test suite via run_tests_parallel.py"
echo "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

cd "$REPO_ROOT"

# ── Pre-compile .pyc bytecode cache ─────────────────────────────────────────
# Each test file runs in its own subprocess via run_tests_parallel.py.
# Pre-building the bytecode cache once here (instead of each subprocess
# compiling on first import) avoids redundant work across ~2000 processes.
# Uses git to list tracked .py files (skips venv, node_modules, etc).
echo "▶ pre-compiling bytecode cache"
"$PYTHON" -m compileall -q -j 0 -- $(git ls-files '*.py') >/dev/null 2>&1 || true

echo "▶ launching test runner"
exec env -i \
  "${HERMETIC_ENV[@]}" \
  "$PYTHON" "$SCRIPT_DIR/run_tests_parallel.py" "$@"
