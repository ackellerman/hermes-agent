"""The gateway a minted POSIX launcher starts is identified as a gateway.

Live installs keep running the launcher script minted by an older ``_launcher_script``: its
``-c`` source starts with ``import os, re, sys`` and has no closure-repair block.
``tests/gateway/test_gateway_inline_bootstrap_identity.py`` only covers the CURRENT minted shape,
so this file pins the older one byte-for-byte (``_V1_SOURCE``, install root neutralised to
``/opt/hermes-agent``). When identity misses that shape, ``live_gateway_pid_for_home`` and every
reader built on it (multiplex migrate confirmation, served-profile list, fleet version check)
see no live gateway although one is running.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gateway import status

PY = "/opt/python/bin/python3"
ROOT = Path("/opt/hermes-agent")

_V1_SOURCE = r'''import os, re, sys
os.environ.pop('PYTHONHOME', None)
os.environ.pop('PYTHONPATH', None)
sys.path.insert(0, '/opt/hermes-agent')
if sys.argv[1:2] == ['--print-runtime-command']: sys.dont_write_bytecode = True
from hermes_constants import get_default_hermes_root
os.environ['HERMES_HOME'] = os.environ.get('HERMES_HOME') or str(get_default_hermes_root())
if sys.argv[1:2] == ['--print-runtime-command']:
    from pathlib import Path
    from hermes_cli._launchers import print_runtime_command
    print_runtime_command(Path('/opt/hermes-agent'), sys.argv[2:])
    sys.exit(0)
import hermes_bootstrap
if sys.argv[1:2] == ['--run-module']:
    import runpy
    if len(sys.argv) < 3: sys.exit('hermes: --run-module needs a module')
    module = sys.argv.pop(2)
    del sys.argv[1]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
    sys.exit(0)
from hermes_cli.main import main
sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
sys.exit(main())
'''

_JOINS = {"space-joined": " ".join, "windows": subprocess.list2cmdline}


def _argv(source: str, *args: str) -> list[str]:
    return [PY, "-I", "-c", source, *args]


def test_fixture_is_the_old_minted_shape() -> None:
    assert _V1_SOURCE.splitlines()[0] == "import os, re, sys"
    assert "_hermes_closure_repair" not in _V1_SOURCE


@pytest.mark.parametrize("join", _JOINS)
@pytest.mark.parametrize("prefix", [(), ("--profile", "example-profile")], ids=["default", "profile"])
def test_v1_launcher_gateway_is_a_gateway(join: str, prefix: tuple[str, ...]) -> None:
    assert status.looks_like_gateway_command_line(_JOINS[join](_argv(_V1_SOURCE, *prefix, "gateway", "run")))


@pytest.mark.parametrize("join", _JOINS)
def test_v1_launcher_chat_is_not_a_gateway(join: str) -> None:
    assert not status.looks_like_gateway_command_line(_JOINS[join](_argv(_V1_SOURCE, "chat")))


def test_live_gateway_pid_for_home_sees_a_v1_launcher_gateway(tmp_path, monkeypatch) -> None:
    from hermes_cli import gateway_multiplex_served
    from hermes_constants import get_default_hermes_root

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    me = os.getpid()
    cmdline = " ".join(_argv(_V1_SOURCE, "gateway", "run"))
    real = status._read_process_cmdline
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: cmdline if pid == me else real(pid))
    status.write_pid_file()
    try:
        assert status.acquire_gateway_runtime_lock()
        # Premises: the pid file and lock exist, so a miss below can only come from identity.
        assert (home / "gateway.pid").is_file()
        assert get_default_hermes_root() == home
        assert status.live_gateway_pid_for_home(home) == me
        assert gateway_multiplex_served.live_default_gateway_pid() == me
    finally:
        status.release_gateway_runtime_lock()
        status.remove_pid_file()


@pytest.mark.parametrize("join", _JOINS)
def test_current_minted_launcher_gateway_is_a_gateway(join: str) -> None:
    from hermes_cli import _launchers

    source = _launchers._launcher_script("hermes", ROOT, None)
    assert status.looks_like_gateway_command_line(_JOINS[join](_argv(source, "gateway", "run")))
