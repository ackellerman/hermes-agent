"""On-disk probe plugin for ``on_voice_session_end`` tests: real discovery, real dispatch.

``install_probe(home, name, ...)`` writes ``<home>/plugins/<name>/{plugin.yaml,__init__.py}`` and
enables it in ``<home>/config.yaml``. The plugin's callback appends every call's kwargs to
``<home>/vse.jsonl`` and then does what ``<plugin dir>/behaviour.json`` says (return a value, raise,
block on a gate, run a check). Nothing in Hermes is mocked: the test reaches the callback only if
Hermes discovers the plugin and dispatches the hook to it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

import hermes_yaml as yaml

_PLUGIN_SOURCE = '''
import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_HOME = _HERE.parent.parent


def _on_voice_session_end(**kwargs):
    behaviour = json.loads((_HERE / "behaviour.json").read_text(encoding="utf-8"))
    with open(_HOME / "vse.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"plugin": _HERE.name, **kwargs}, default=str) + "\\n")
    from tests.fakes import voice_session_end_probe as probe
    probe.run_hooks(str(_HOME), _HERE.name, kwargs, behaviour)
    if behaviour.get("raise"):
        raise RuntimeError("voice-session-end probe failure")
    return behaviour.get("returns")


def register(ctx):
    ctx.register_hook("on_voice_session_end", _on_voice_session_end)
'''

_GATES: dict[tuple[str, str], threading.Event] = {}
_STARTED: dict[tuple[str, str], threading.Event] = {}
_CHECKS: dict[tuple[str, str], Callable[[dict], Any]] = {}
CHECK_RESULTS: list[tuple[str, Any]] = []
_LOCK = threading.Lock()


def _key(home: Path | str, name: str) -> tuple[str, str]:
    return str(Path(home).resolve()), name


def install_probe(home: Path, name: str = "vse-probe", *, returns: Any = None, allow_injection: bool | None = None,
                  enabled: bool = True, **behaviour: Any) -> Path:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0", "description": "voice-session-end probe"}),
        encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN_SOURCE, encoding="utf-8")
    set_behaviour(home, name, returns=returns, **behaviour)
    cfg_path = home / "config.yaml"
    cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}) if cfg_path.exists() else {}
    plugins_cfg = cfg.setdefault("plugins", {})
    if enabled:
        plugins_cfg.setdefault("enabled", []).append(name)
    if allow_injection is not None:
        plugins_cfg.setdefault("entries", {}).setdefault(name, {})["allow_gateway_injection"] = allow_injection
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return plugin_dir


def set_behaviour(home: Path, name: str = "vse-probe", **behaviour: Any) -> None:
    (home / "plugins" / name / "behaviour.json").write_text(json.dumps(behaviour), encoding="utf-8")


def records(home: Path) -> list[dict]:
    path = home / "vse.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover(home: Path):
    """The plugin manager for *home*, discovered under that home's override (as the gateway does)."""
    from hermes_cli import plugins as plugins_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(home))
    try:
        manager = plugins_mod.get_plugin_manager()
        manager.discover_and_load()
        return manager
    finally:
        reset_hermes_home_override(token)


def gate(home: Path | str, name: str) -> threading.Event:
    with _LOCK:
        return _GATES.setdefault(_key(home, name), threading.Event())


def started(home: Path | str, name: str) -> threading.Event:
    with _LOCK:
        return _STARTED.setdefault(_key(home, name), threading.Event())


def set_check(home: Path, name: str, check: Callable[[dict], Any]) -> None:
    with _LOCK:
        _CHECKS[_key(home, name)] = check


def run_hooks(home: str, name: str, kwargs: dict, behaviour: dict) -> None:
    """Called by the on-disk plugin: run the registered check, then block if told to."""
    key = _key(home, name)
    with _LOCK:
        check = _CHECKS.get(key)
    if check is not None:
        CHECK_RESULTS.append((name, check(kwargs)))
    started(home, name).set()
    block = behaviour.get("block")
    if block is True or (isinstance(block, list) and kwargs.get("voice_session_id") in block
                         ) or (block == "first" and len(records(Path(home))) == 1):
        gate(home, name).wait(30)


def reset() -> None:
    """Release every blocked callback and forget all per-test state."""
    with _LOCK:
        for event in _GATES.values():
            event.set()
        _GATES.clear()
        _STARTED.clear()
        _CHECKS.clear()
        CHECK_RESULTS.clear()
