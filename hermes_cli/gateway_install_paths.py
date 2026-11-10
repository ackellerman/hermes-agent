"""The gateway unit's ``LD_LIBRARY_PATH`` line and the install-generation filter behind it.

A systemd unit is regenerated and compared on every start/restart/status, so the line it carries
must survive a dependency sync. This module owns the line's construction and the rule that drops
components living inside an install's ``installs/<key>/environments/`` generation — a path the next
``hermes update`` deletes.

Split out of ``hermes_cli/gateway.py``; the facade keeps ``_hermes_home_for_target_user`` (other
code and tests patch it there), so this module late-imports it inside the function that needs it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Log-record parity with the origin module: the unit tests filter caplog on this name.
logger = logging.getLogger("hermes_cli.gateway")


def _remap_path_for_user(path: str, target_home_dir: str) -> str:
    """Swap the ``Path.home()`` prefix of *path* for *target_home_dir*; other paths return unchanged.
    Intentionally does NOT resolve symlinks."""
    current_home = Path.home()
    p = Path(path).expanduser()
    try:
        relative = p.relative_to(current_home)
        return str(Path(target_home_dir) / relative)
    except ValueError:
        return str(p)


def _systemd_env_line(name: str, value: str) -> str:
    """One ``Environment="NAME=value"`` line: ``\\`` and ``"`` escaped for systemd's quoting, ``%``
    doubled so specifier expansion leaves the value alone."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def _installed_unit_ld_library_path(system: bool) -> str:
    """``LD_LIBRARY_PATH`` baked into the installed unit; ``""`` when absent."""
    from hermes_cli.gateway import _unit_environment_value, get_systemd_unit_path

    return _unit_environment_value(get_systemd_unit_path(system=system), "LD_LIBRARY_PATH") or ""


def _ld_library_path_line(system: bool, target_home_dir: str | None = None) -> str:
    """Carry the installer's LD_LIBRARY_PATH into the unit (glibc reads it only at process start, so
    ~/.hermes/.env is too late for CUDA libs — #14613); system units remap caller-home components.

    The installed unit is the fallback source: the unit is regenerated and compared on every
    start/restart/status, and a shell without the export (ssh, cron, ``sudo`` strips ``LD_*``)
    must not be able to "repair" the line away.

    One exception, applied to both sources: a component inside an install's
    ``installs/<key>/environments/`` generation is dropped, each drop logged at INFO. A dependency
    sync replaces that generation, so a unit carrying it hardcodes a path the next ``hermes update``
    deletes — the failure this filter exists for. Components merely absent from disk are kept: an
    unmounted volume is not stale."""
    raw = os.environ.get("LD_LIBRARY_PATH", "") or _installed_unit_ld_library_path(system)
    components = [p for p in raw.split(":") if p]
    if target_home_dir is not None:
        components = [_remap_path_for_user(p, target_home_dir) for p in components]
    components = _drop_install_environment_components(
        components, _ld_library_path_install_roots(system=system, target_home_dir=target_home_dir)
    )
    return _systemd_env_line("LD_LIBRARY_PATH", ":".join(components)) if components else ""


def _installs_root_for_hermes_home(
    hermes_home: str | Path, target_home_dir: str | Path | None = None
) -> Path:
    """``<root>/installs`` beside a Hermes home, with ``get_default_hermes_root``'s rule but anchored
    on the *target user's* native home (this process's ``Path.home()`` when no target is given).

    A home anywhere inside ``<home>/.hermes`` — a subdirectory, ``profiles/<name>``, or the
    ``profiles`` dir itself — belongs to that native home, which is where pm puts the generations
    (``hermes_constants._hermes_root_of``'s containment rule; the path shape alone gets
    ``~/.hermes/subdir`` wrong). Only a home outside it is its own root, and then a profile lives
    under its data root. The containment test resolves both sides, as ``_hermes_root_of`` does, so
    a ``HERMES_HOME`` that is a symlink into the native home counts as inside it; the root returned
    is still the lexical native home, matching the lexical components the unit carries."""
    home = Path(hermes_home)
    native_home = (Path(target_home_dir) if target_home_dir is not None else Path.home()) / ".hermes"
    if home.resolve(strict=False).is_relative_to(native_home.resolve(strict=False)):
        return native_home / "installs"
    root = home.parent.parent if home.parent.name == "profiles" else home
    return root / "installs"


def _is_install_environment_component(component: str, installs_root: Path) -> bool:
    """True when *component* sits under ``<installs_root>/<install key>/environments/`` — any install
    key, including one from an older or other checkout. Lexical on purpose and it never stats: a
    component that is merely missing from disk (an unmounted volume) is not stale.

    That leniency has a deliberate fail-open set, so this is not a total guard: a relative
    component, a symlinked component (neither is resolved), a component written resolved while
    ``installs_root()`` is unresolved (a symlinked ``HOME``), and a component under a different
    installs root (another data root's ``installs``, e.g. a borrowed checkout) are all kept."""
    try:
        relative = Path(os.path.normpath(component)).relative_to(Path(os.path.normpath(installs_root)))
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= 3 and parts[1] == "environments"


def _ld_library_path_install_roots(system: bool, target_home_dir: str | None) -> list[Path]:
    """The installs roots whose dependency generations the unit must not carry. A system unit is
    written for the target user, so it is filtered against that user's home, as the components were
    remapped; the caller's own root stays in the list for a component the remap left untouched."""
    from hermes_cli.gateway import _hermes_home_for_target_user
    from pm.environments import installs_root

    candidates = [installs_root()]
    if system and target_home_dir:
        candidates.append(
            _installs_root_for_hermes_home(
                _hermes_home_for_target_user(target_home_dir), target_home_dir
            )
        )
    roots: list[Path] = []
    for root in candidates:
        if root not in roots:
            roots.append(root)
    return roots


def _drop_install_environment_components(components: list[str], installs_roots: list[Path]) -> list[str]:
    """Drop the components that point into a dependency generation, keeping every other one in order.
    ``hermes update`` (a pm sync) replaces those generations, so the path is dead by the time CUDA
    dlopens it — the stale unit line that made local STT fall back to CPU."""
    kept: list[str] = []
    for component in components:
        stale_root = next(
            (root for root in installs_roots if _is_install_environment_component(component, root)), None
        )
        if stale_root is None:
            kept.append(component)
            continue
        logger.info(
            "Dropping LD_LIBRARY_PATH component %s from the gateway unit: it lies under %s, an "
            "install generation that the next dependency sync replaces",
            component,
            stale_root,
        )
    return kept
