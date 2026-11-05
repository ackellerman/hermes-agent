"""Startup config-drift warnings: the process holds one snapshot, the disk may say something else.

Owner ruling (2026-11-02): **gates live, plugins restart-only.** ``allow_bots``, ``bot_allow_from``,
``free_response_channels`` and ``free_response_categories`` are re-read per message (see the Discord
adapter), so a change there shows up without a restart. Everything else — plugin enablement, voice, and anything that owns a socket —
is loaded once at startup, so an edit on disk is silently inert until the gateway restarts. That
silence is what this module fixes: once a minute it compares the values the running process actually
loaded against the values on disk now, and says one WARNING naming the keys.

Design notes:

* Callers own the comparison, not the monitor. Each :class:`DriftKey` carries the already-normalized
  ``snapshot`` (what the process loaded) and ``disk`` (what the source says right now), so this
  module never has to know how a given key is read — no new config keys to remember.
* One WARNING per *change*, not per check: the last warned key set is remembered so a persistently
  drifted file does not repeat every minute, and a file that converges resets the latch so a later
  re-drift is reported again.
* The loop is a daemon thread: a config comparison must never keep a gateway alive or block its loop.
* One monitor, many providers: a multiplexed gateway loads a plugin once per profile scope, and every
  copy registers its own row provider. All of them are compared (keyed by defining module, so a reload
  replaces its stale provider), and each adapter prefixes its rows with its profile directory name.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# "once a minute" is the owner's ask; a config edit is not a latency-sensitive signal.
DEFAULT_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class DriftKey:
    """One comparable setting: the process-loaded value vs. what the source says now."""

    name: str
    snapshot: str
    disk: str
    # False => loaded once at startup; only a restart can pick the new value up.
    reloadable: bool = False

    def drifted(self) -> bool:
        return self.snapshot != self.disk


def drifted_keys(rows: Iterable[DriftKey]) -> list[DriftKey]:
    """The subset of ``rows`` whose on-disk value no longer matches the process snapshot."""
    return [row for row in rows if row.drifted()]


def format_warning(drifted: list[DriftKey], *, scope: str = "config") -> Optional[str]:
    """The one-line WARNING body for ``drifted``; ``None`` when nothing drifted.

    Names the keys, and says explicitly which of them need a restart — the whole point of the
    message is that "I edited it and nothing happened" now costs a log line instead of a mystery.
    """
    if not drifted:
        return None
    live = [row.name for row in drifted if row.reloadable]
    restart = [row.name for row in drifted if not row.reloadable]
    parts = []
    if live:
        parts.append(f"{', '.join(live)} changed on disk (applied live)")
    if restart:
        parts.append(
            f"{', '.join(restart)} changed on disk — restart required "
            "(loaded once at startup)"
        )
    return f"[{scope}] {'; '.join(parts)}"


class ConfigDriftMonitor:
    """Compare the loaded snapshot to disk once per ``interval`` and WARN once per change.

    ``key_provider`` is called on every check and must return fresh :class:`DriftKey` rows (its disk
    side is re-read each time). Any exception from it is contained: drift reporting must never be
    able to take a healthy gateway down.
    """

    def __init__(
        self,
        key_provider: Callable[[], Iterable[DriftKey]],
        *,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        scope: str = "config",
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._key_provider = key_provider
        self._interval = max(1.0, float(interval))
        self._scope = scope
        self._log = log or logger
        self._warned: frozenset[str] = frozenset()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── one check (the unit under test) ──────────────────────────────────
    def check_once(self) -> Optional[str]:
        """Run one comparison; log and return the WARNING body when the drifted set changed."""
        try:
            rows = list(self._key_provider() or ())
        except Exception:
            self._log.debug("config drift check failed", exc_info=True)
            return None
        drifted = drifted_keys(rows)
        names = frozenset(row.name for row in drifted)
        if not names:
            # Converged (or nothing to compare): re-arm so a later re-drift is reported again.
            self._warned = frozenset()
            return None
        if names <= self._warned:
            return None  # already said exactly this — one WARNING per change, not per minute
        self._warned = names
        message = format_warning(drifted, scope=self._scope)
        if message:
            self._log.warning(message)
        return message

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> bool:
        """Start the daemon loop; idempotent. Returns False when already running."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"hermes-{self._scope}-config-drift", daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, *, timeout: float = 1.0) -> None:
        """Signal the loop to exit (used by tests and by a shutdown path)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        # Sleep FIRST: at startup the snapshot and the disk agree by construction, and the first
        # useful comparison is the one after an operator has had time to edit something.
        while not self._stop.wait(self._interval):
            self.check_once()


_MONITOR: Optional[ConfigDriftMonitor] = None
_MONITOR_LOCK = threading.Lock()

# Every registered row provider, keyed by the module that defines it. A multiplexed gateway imports a
# plugin once per profile scope (``hermes_plugins.platforms__discord`` and ``…__home_<digest>``), and
# each copy has its own provider; keeping only the first one meant no other profile's edit was ever
# reported. Keyed by module name, not object identity, so a plugin reload (same module name, new
# function object) REPLACES its stale provider instead of running beside it.
_PROVIDERS: dict[str, Callable[[], Iterable[DriftKey]]] = {}


def _provider_key(key_provider: Callable[[], Iterable[DriftKey]]) -> str:
    module = getattr(key_provider, "__module__", None) or ""
    name = getattr(key_provider, "__qualname__", None) or repr(key_provider)
    return f"{module}:{name}"


def registered_provider_count() -> int:
    """How many distinct providers feed the monitor (more than one => rows need an owner prefix)."""
    with _MONITOR_LOCK:
        return len(_PROVIDERS)


def _all_rows() -> list[DriftKey]:
    """Rows from every registered provider; one failing provider does not hide the others."""
    with _MONITOR_LOCK:
        providers = list(_PROVIDERS.values())
    rows: list[DriftKey] = []
    for provider in providers:
        try:
            rows.extend(provider())
        except Exception:
            logger.debug("config drift: a key provider failed", exc_info=True)
    return rows


def start_monitor(
    key_provider: Callable[[], Iterable[DriftKey]],
    *,
    interval: float = DEFAULT_INTERVAL_SECONDS,
    scope: str = "config",
) -> ConfigDriftMonitor:
    """Register ``key_provider`` and start the process-wide monitor once; later calls return it.

    One monitor per process (not per adapter): a multiplexed gateway would otherwise emit the same
    warning once per profile for a change they all share. It compares the rows of EVERY registered
    provider, so each profile's plugin module copy is watched, not only the first to connect.
    """
    global _MONITOR
    with _MONITOR_LOCK:
        _PROVIDERS[_provider_key(key_provider)] = key_provider
        if _MONITOR is None:
            _MONITOR = ConfigDriftMonitor(_all_rows, interval=interval, scope=scope)
        if not _MONITOR.running:
            _MONITOR.start()
        return _MONITOR


def reset_monitor() -> None:
    """Stop and drop the process-wide monitor and its providers (tests; also a clean shutdown hook)."""
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is not None:
            _MONITOR.stop()
        _MONITOR = None
        _PROVIDERS.clear()
