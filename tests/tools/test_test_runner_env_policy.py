"""One definition of the test environment: the runner, the script and conftest agree.

Guards the class of bug this file exists for. ``scripts/run_tests.sh`` used to keep its own
``env -i`` allowlist while ``tests/conftest.py`` kept a separate hand-written blanking list.
Both drifted, and the drift is invisible until a test passes under ``pytest`` and fails under
the canonical runner (or the reverse). Measured on one developer host before the fix: a bare
``pytest`` run exposed 111 environment names to a test where ``scripts/run_tests.sh`` exposed
23, including the operator's ``TMP``/``TEMP`` pointing into the real ``~/.hermes/cache/scratch``.

These tests fail if a second list comes back, which is the only way the class returns.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._fixtures.env_policy import (
    TEMP_VARS,
    allowed_names,
    blank_names,
    forwarded_env,
    runner_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_tests.sh"
POLICY = REPO_ROOT / "tests" / "_fixtures" / "env_policy.py"


def test_run_tests_sh_has_no_second_allowlist():
    """The launcher must ask the policy module; a bash list is how the drift started."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "env_policy.py" in text and "--runner-env" in text
    for legacy in ("WIN_ENV=(", "TEST_ENV=(", "for _win_var in", "for _test_var in"):
        assert legacy not in text, f"{legacy!r} is a second env list; extend env_policy.py instead"


def test_runner_env_never_forwards_the_account_store():
    """run_tests.sh's _account_tool_store exports these for its own provisioning only."""
    environ = {"PATH": "/usr/bin", "HOME": "/home/u", "HERMES_RUNTIME_DIR": "/home/u/.hermes/tools",
               "UV_PYTHON_INSTALL_DIR": "/home/u/.local/share/uv/python"}
    for platform in ("linux", "win32"):
        names = {pair.split("=", 1)[0] for pair in runner_env(environ, platform)}
        assert "HERMES_RUNTIME_DIR" not in names and "UV_PYTHON_INSTALL_DIR" not in names
        assert {"PATH", "HOME"} <= names, "control: the allowlisted names are forwarded"


def test_launcher_emits_exactly_the_policy_for_the_same_environment():
    """What ``env -i`` is handed equals what the runner gives a child, name for name."""
    probe = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/tester",
        "TMPDIR": "/var/tmp/run",
        "TMP": "/home/tester/.hermes/cache/scratch",  # operator leftovers: not forwarded on POSIX
        "TEMP": "/home/tester/.hermes/cache/scratch",
        "OPENAI_API_KEY": "sk-should-not-leak",
        "HERMES_SESSION_ID": "leaky",
        "TERMINAL_TEMP_DIR": "/tmp/leaky",
        "HERMES_TEST_WORKERS": "4",
        "HERMES_HOME": "/home/tester/custom-hermes",
    }
    emitted = subprocess.run(
        [sys.executable, str(POLICY), "--runner-env"],
        env=probe, capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
    ).stdout.splitlines()
    expected = runner_env(probe)
    assert sorted(emitted) == sorted(expected)
    assert sorted(emitted) == sorted(f"{k}={v}" for k, v in forwarded_env(probe).items())


@pytest.mark.platforms("posix")
def test_posix_never_forwards_the_windows_temp_variables():
    """TEMP/TMP are Windows location variables; forwarding them on POSIX handed every test
    the operator's Hermes scratch dir (the direct cause of the local_env_blocklist failures)."""
    env = forwarded_env({"PATH": "/bin", "TMP": "/home/x/.hermes/cache/scratch",
                         "TEMP": "/home/x/.hermes/cache/scratch"}, platform="linux")
    assert set(env) & {"TMP", "TEMP"} == set()
    assert "TMPDIR" not in env  # the runner sets it per attempt, the policy does not invent one


def test_blank_names_is_the_complement_of_the_allowlist():
    env = {
        "PATH": "/bin", "HOME": "/home/x", "TZ": "UTC",
        "OPENAI_API_KEY": "sk-x", "HERMES_SESSION_ID": "leaky",
        "TERMINAL_PERSISTENT_SHELL": "1", "PYTEST_CURRENT_TEST": "x.py::t (call)",
    }
    blanked = blank_names(env, platform="linux")
    assert blanked == {"OPENAI_API_KEY", "HERMES_SESSION_ID", "TERMINAL_PERSISTENT_SHELL"}
    assert set(env) - blanked - {"PYTEST_CURRENT_TEST"} <= set(allowed_names("linux"))


def test_conftest_left_no_ambient_name_outside_the_allowlist():
    """The real process after conftest import: conftest's own names are the only extras.

    Everything conftest installs (sandboxed HERMES_HOME, the lock dir, the relay policy, the
    isolation markers, the deterministic trio) is set *by* conftest; anything else that is
    still present would be an operator/shell leak the canonical runner never has.
    """
    from tests import conftest

    allowed = set(allowed_names())
    # Names conftest itself owns (it sets them after the sweep).
    conftest_owned = {
        "HERMES_TEST_SANDBOX_HOME", "HERMES_TEST_ISOLATION", "HERMES_GATEWAY_LOCK_DIR",
        "HERMES_NEMO_RELAY_PLUGINS_TOML", "HERMES_DISABLE_LAZY_INSTALLS", "HERMES_HONCHO_HOST",
        "AWS_EC2_METADATA_DISABLED", "AWS_METADATA_SERVICE_TIMEOUT", "AWS_METADATA_SERVICE_NUM_ATTEMPTS",
        "TMPDIR",
    }
    leaked = {
        name for name in os.environ
        if name not in allowed and name not in conftest_owned and not name.startswith("PYTEST_")
    }
    assert leaked == set(), f"ambient names visible to tests outside the allowlist: {sorted(leaked)}"
    assert conftest.SWEPT_ENV_NAMES == tuple(sorted(conftest.SWEPT_ENV_NAMES))


@pytest.mark.platforms("posix")
def test_a_hermes_home_temp_pointer_is_dropped_even_through_a_symlink(tmp_path, monkeypatch):
    """``$HERMES_HOME/cache/scratch`` is routinely a symlink into disk-backed storage.

    A resolved-only containment check let the pointer survive, so a fixture seeding the
    ambient environment handed ``LocalEnvironment`` the real scratch dir and its snapshot
    files landed in ``~/.hermes`` (caught by home_io_guard as "TEST BUG: file I/O against the
    REAL hermes home"). Both spellings must be compared.
    """
    from tests._fixtures.env_policy import sanitize_temp_vars

    home = tmp_path / "hermes"
    (home / "cache").mkdir(parents=True)
    real = tmp_path / "disk-backed"
    real.mkdir()
    (home / "cache" / "scratch").symlink_to(real)
    env = {"TMPDIR": str(home / "cache" / "scratch"), "TMP": str(real), "TEMP": str(tmp_path / "elsewhere")}
    dropped = sanitize_temp_vars(env, [home])
    assert set(dropped) == set(TEMP_VARS[:2])
    assert env == {"TEMP": str(tmp_path / "elsewhere")}
