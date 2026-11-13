"""Kept delegated children: the durable record that lets a parent find, hold and later continue a child.

Upstream's stamp (``_subagent_id`` in the child's ``model_config``) and ``stored_child_session`` decide
WHICH stored transcript is a child's and whether a conversation may see it. This module decides the
rest: whether a child is kept, who holds it, its run state and run fence, expiry and orphans.

One row per kept child in ``delegate_children`` of the parent's own ``state.db`` (the file behind the
parent's ``_session_db.db_path``; never a bare ``acquire()`` / ``get_hermes_home()``, so two profiles in
one process write to their own stores). The table is created on first use; there is no
``SCHEMA_VERSION`` bump.

* Key ``child_id``: the child's first session id. ``subagent_id`` is an alias that resolves only through
  ``stored_child_session`` against the caller's conversation, since ``subagent_id``\\s can collide.
* Holder: ``holder_session_id`` + ``conversation_id``. ``claim`` and ``transfer`` rewrite both and record
  the old conversation in ``prior_conversations`` so the lineage check still finds the transcript.
* Status ``running|idle|closed|expired``; ``idle_reason`` ``completed|interrupted|orphaned``.
* Run fence: ``run_token`` + ``runner`` (pid, pid namespace, start ticks) + ``heartbeat_at``. Every
  transition is a compare-and-set; a late write from an older run is ignored by token.
* ``heartbeat_at`` is touched by this module's own 30 s tick while the worker thread lives. It does not
  depend on progress: the delegate heartbeat stops beating after its stale verdict.
* A run ends (``running -> idle``) only from the worker's done-callback, or by the orphan rule. A parent
  wait that ends early while the worker is alive leaves the row ``running`` with ``wait_abandoned_at``.
* Expiry and the orphan rule are applied on every read: the read looks first and takes the write lock only
  when a rule would change a row; a read never creates the table. A store that cannot be written (read-only,
  or locked longer than ``READ_WRITE_BUSY_MS``) shows the derived state instead of raising or stalling.
  ``idle_since`` is reset only by a new run (``take_fence``); reads never touch it.
* A parent whose session store is a managed worker's ``RuntimeSessionStore`` (no local ``state.db`` path) never
  keeps its children: nothing here writes through the worker RPC.

Credentials, provider clients and ACP transport args are never stored; a resume re-derives the route.

The follow-up itself is upstream's ``delegate_task(action='continue')`` (``tools/delegate_tool_continue.py``),
which addresses a child by ``subagent_id`` and rebuilds it from its own record (``resumable_subagents``). This
module only follows it: a continue re-opens the kept row of the child it rebuilt (``note_spawn(rebuilt=True)``
takes the row's fence when the worker starts), so the row says ``running`` while the continue runs and goes
``idle`` when it ends. A continue never creates a kept row: only a spawn decides that a child is kept.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

TABLE = "delegate_children"

# Parent session sources whose model-spawned children are kept (``delegation.keep_sources`` overrides).
# The interactive surfaces and the gateway's chat platforms (gateway/config.py ``Platform``). Left out on
# purpose: api_server, webhook, msgraph_webhook, wecom_callback, relay, local, acp, cron, kanban, tool and
# one-shot runs (``oneshot``): nobody comes back to them to resume a child.
DEFAULT_KEEP_SOURCES = frozenset({
    "cli", "tui", "desktop",
    "telegram", "discord", "slack", "whatsapp", "whatsapp_cloud", "signal", "matrix", "mattermost",
    "email", "sms", "dingtalk", "feishu", "wecom", "weixin", "bluebubbles", "qqbot",
})

# Spawn origins. Only ``model`` (the model's own ``delegate_task`` call) is ever kept.
ORIGIN_MODEL, ORIGIN_REVIEW, ORIGIN_PLUGIN = "model", "review", "plugin"

# A parent agent carrying this attribute truthy never keeps its children (cron jobs set it).
NEVER_KEEP_ATTR = "_delegate_children_never_keep"

DEFAULT_KEEP_IDLE_DAYS = 7
DEFAULT_ORPHAN_AFTER_S = 300
HEARTBEAT_INTERVAL_S = 30.0
# A read path may need a write (persisting a due rule, refreshing ``latest_session_id``): it waits at most this
# long for a writer's lock, then shows the derived state and leaves the write to the next read.
READ_WRITE_BUSY_MS = 250

# The recovered notice's line for a kept child that is now idle-orphaned; it names that child's own id, since
# nothing else in the notice lists a kept child.
ORPHANED_LINE = ("child kept as idle-orphaned; continue it with delegate_task(action='continue', "
                 "subagent_id='{subagent_id}', message=...)")


def orphaned_line(subagent_ids: Iterable[str]) -> str:
    """``ORPHANED_LINE`` once per idle-orphaned child of the unit."""
    return "; ".join(ORPHANED_LINE.format(subagent_id=sid) for sid in subagent_ids)

# ``action='list'`` shows at most this many kept children per list (the plan's 20-child cap). The roster
# (agent/turn_context.py) applies its own smaller cap.
LIST_LIMIT = 20
# A holder session's title is shown as its short label in the profile-wide list.
HOLDER_LABEL_MAX = 60

# The one line that ends a kept child's completion text: model- and user-visible prose, one sentence,
# naming the action and the id that action takes. ``action='continue'`` is addressed by ``subagent_id``
# (``child_id`` is what ``list``/``steer``/``stop`` also accept; ``continue`` does not).
RESUME_HINT = ("This subagent can be continued on its own conversation: "
               "delegate_task(action='continue', subagent_id='{subagent_id}', message=...).")


def resume_hint(subagent_id: str) -> str:
    """The follow-up hint for the child *subagent_id* (see ``RESUME_HINT``)."""
    return RESUME_HINT.format(subagent_id=subagent_id)


def stamp_completion_entry(entry: dict[str, Any], child: Any) -> bool:
    """Add the parent's stable handles to a kept child's completion *entry*: ``child_id`` (its first session
    id — what ``steer``/``stop`` accept) next to ``subagent_id`` (what ``continue`` takes), plus the one-line
    ``resume_hint`` the completion text ends with. A child that is not kept gains nothing (the keys stay absent, never a fake
    id). One definition serves both places a completion entry is built: the finalize pass
    (``delegate_tool_results._stamp_child_identities``) and the durable record of a finished child of a
    still-running unit (``delegate_tool_dispatch._record_finished_child``), so a replay after the owner process
    died carries the same ids the live completion would have."""
    child_id = getattr(child, "_delegate_child_id", None)
    if not (isinstance(child_id, str) and child_id):
        return False
    entry["child_id"] = child_id
    subagent_id = getattr(child, "_subagent_id", None)
    if isinstance(subagent_id, str) and subagent_id:
        entry["subagent_id"] = subagent_id
        entry["resume_hint"] = resume_hint(subagent_id)  # no id to continue by, no hint
    return True


RUNNING, IDLE, CLOSED, EXPIRED = "running", "idle", "closed", "expired"
COMPLETED, INTERRUPTED, ORPHANED = "completed", "interrupted", "orphaned"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    child_id TEXT PRIMARY KEY,
    subagent_id TEXT NOT NULL,
    conversation_id TEXT,
    holder_session_id TEXT,
    prior_conversations TEXT NOT NULL DEFAULT '[]',
    latest_session_id TEXT,
    delegation_id TEXT,
    task_index INTEGER,
    origin TEXT,
    goal TEXT,
    context TEXT,
    title TEXT,
    route TEXT,
    model TEXT,
    provider TEXT,
    toolsets TEXT,
    role TEXT,
    skills TEXT,
    workdir TEXT,
    repo_root TEXT,
    branch TEXT,
    base_commit TEXT,
    profile TEXT,
    depth INTEGER,
    parent_child_id TEXT,
    spawn_generations INTEGER NOT NULL DEFAULT 0,
    max_children INTEGER,
    children_spawned INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    idle_reason TEXT,
    kept INTEGER NOT NULL DEFAULT 1,
    released INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    idle_since REAL,
    orphaned_at REAL,
    wait_abandoned_at REAL,
    run_token TEXT,
    runner TEXT,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_holder ON {TABLE}(holder_session_id);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_subagent ON {TABLE}(subagent_id);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_delegation ON {TABLE}(delegation_id);
"""

_JSON_FIELDS = ("prior_conversations", "route", "toolsets", "skills", "runner")

# Brief / spawn-setting fields a spawn may record (everything else is run state owned here).
_BRIEF_FIELDS = (
    "delegation_id", "task_index", "origin", "goal", "context", "title", "route", "model", "provider",
    "toolsets", "role", "skills", "workdir", "repo_root", "branch", "base_commit", "profile", "depth",
    "parent_child_id", "spawn_generations", "max_children", "children_spawned",
)

# Route keys that may be stored. Never an api_key, client, credential pool or ACP transport (``acp_command`` /
# ``acp_args`` are trusted-config-only and may carry tokens): a resume re-derives the route. Never the
# ``base_url`` either (its userinfo or query can carry a key): ``note_spawn`` stores ``endpoint``, the
# ``endpoint_identity`` digest the resumable record stores for the same child.
_ROUTE_KEYS = ("provider", "model", "api_mode", "reasoning_effort", "service_tier",
               "requested_provider", "profile_home")


# ── config ──────────────────────────────────────────────────────────────────

def _delegation_cfg() -> dict[str, Any]:
    from tools.delegate_tool_config import _load_config  # absorbs every config read error itself
    cfg = _load_config()
    return cfg if isinstance(cfg, dict) else {}


_WARNED: set[tuple[str, str]] = set()


def _warn_invalid(key: str, raw: Any, fallback: str) -> None:
    """One warning per (knob, value), as ``delegate_tool_config._knob`` logs an invalid delegation value."""
    if (key, repr(raw)) not in _WARNED:
        _WARNED.add((key, repr(raw)))
        logger.warning("delegation.%s: invalid value %r; using %s", key, raw, fallback)


def keep_sources(cfg: Optional[dict[str, Any]] = None) -> frozenset[str]:
    """``delegation.keep_sources`` as a set; unset (null) means ``DEFAULT_KEEP_SOURCES``. Any other non-list value
    keeps nothing (fail safe: a scalar must not widen the set) and logs a warning."""
    raw = (_delegation_cfg() if cfg is None else cfg).get("keep_sources")
    if raw is None:
        return DEFAULT_KEEP_SOURCES
    if isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset(str(s).strip().lower() for s in raw if str(s).strip())
    _warn_invalid("keep_sources", raw, "no source (no child is kept)")
    return frozenset()


def _positive_number(key: str, raw: Any, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0:
        _warn_invalid(key, raw, f"the default {default}")
        return default
    return float(raw)


def keep_idle_days(cfg: Optional[dict[str, Any]] = None) -> float:
    raw = (_delegation_cfg() if cfg is None else cfg).get("keep_idle_days")
    return _positive_number("keep_idle_days", raw, DEFAULT_KEEP_IDLE_DAYS)


def orphan_after_s(cfg: Optional[dict[str, Any]] = None) -> float:
    raw = (_delegation_cfg() if cfg is None else cfg).get("orphan_after_s")
    return _positive_number("orphan_after_s", raw, DEFAULT_ORPHAN_AFTER_S)


def protects_transcripts(cfg: Optional[dict[str, Any]] = None) -> bool:
    """``delegation.keep_protects_transcripts`` (default True): session pruning spares kept children."""
    raw = (_delegation_cfg() if cfg is None else cfg).get("keep_protects_transcripts", True)
    return raw is not False and str(raw).strip().lower() not in ("false", "0", "no", "off")


def protects_transcripts_for(db_path: Path | str) -> bool:
    """``protects_transcripts`` as configured in the profile that owns *db_path* (its ``config.yaml`` beside it),
    not the ambient one: gateway housekeeping prunes every served profile without entering its scope."""
    from hermes_cli.config_effective import load_user_config_effective
    cfg = load_user_config_effective(Path(db_path).parent / "config.yaml").get("delegation")
    return protects_transcripts(cfg if isinstance(cfg, dict) else {})


# ── spawn origin ────────────────────────────────────────────────────────────

_SPAWN_ORIGIN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("delegate_spawn_origin", default=None)


@contextlib.contextmanager
def spawn_origin(origin: Optional[str]) -> Iterator[None]:
    """Children built inside this block carry *origin* (``model``, ``review``, ``plugin``)."""
    token = _SPAWN_ORIGIN.set(origin)
    try:
        yield
    finally:
        _SPAWN_ORIGIN.reset(token)


def current_spawn_origin() -> Optional[str]:
    """The origin of the spawn being built; None outside every spawn path (never kept)."""
    return _SPAWN_ORIGIN.get()


# ── db access ───────────────────────────────────────────────────────────────

def _plain_session_db(db: Any) -> Any:
    """The SessionDB behind an ``AsyncSessionDB`` (whose methods are coroutine functions), else *db*."""
    from hermes_state import AsyncSessionDB
    return db._db if isinstance(db, AsyncSessionDB) else db


def parent_db_path(parent_agent: Any) -> Optional[Path]:
    """The parent's ``state.db`` path, or None. Concrete str/Path only (a MagicMock is PathLike)."""
    path = getattr(_plain_session_db(getattr(parent_agent, "_session_db", None)), "db_path", None)
    return Path(path) if isinstance(path, (str, Path)) else None


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create ``delegate_children`` (idempotent); also the session-recovery initializer for the table."""
    conn.executescript(_SCHEMA_SQL)



def _connect(db_path: Path, busy_timeout_ms: int = 10_000) -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db
    # wal=False: SessionDB owns state.db's journal mode.
    return open_db(db_path, db_label=f"state.db ({TABLE})", busy_timeout_ms=busy_timeout_ms, wal=False,
                   row_factory=sqlite3.Row, initialize=ensure_table)


def _table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone() is not None


def _profile_for(db_path: Path, profile_home: Any = None) -> str:
    home = Path(profile_home) if isinstance(profile_home, (str, Path)) else db_path.parent
    return home.name if home.parent.name == "profiles" else "default"


def runner_record() -> dict[str, Any]:
    """This process as a run's runner: pid, pid namespace and kernel start ticks."""
    from hermes_state_common import _proc_start_ticks
    from hermes_state_pidns import pid_namespace_id
    pid = os.getpid()
    return {"pid": pid, "pidns": pid_namespace_id(), "start_ticks": _proc_start_ticks(pid)}


def _runner_checkable(runner: Any) -> bool:
    """Whether *runner*'s liveness can be decided here: a valid pid in our pid namespace with start ticks."""
    from hermes_state_pidns import persistent_record_pidns_checkable
    if not isinstance(runner, dict):
        return False
    try:
        if int(runner["pid"]) <= 0:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return persistent_record_pidns_checkable(runner.get("pidns")) and runner.get("start_ticks") is not None


def _decode(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return None if row is None else _decode_row(row)


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in _JSON_FIELDS:
        raw = out.get(key)
        if isinstance(raw, str):
            try:
                out[key] = json.loads(raw)
            except ValueError:
                out[key] = None
    out["kept"], out["released"] = bool(out.get("kept")), bool(out.get("released"))
    return out


def _runner_alive(runner: Any) -> Optional[bool]:
    """True only with positive proof (the pid exists in our pid namespace and its kernel start ticks equal the
    record); False when provably dead or the pid was reused (other start ticks); None when it cannot be checked
    here (foreign pid namespace, no recorded ticks, or the pid's ticks are unreadable)."""
    from hermes_state_common import _lock_holder_provably_dead, _proc_start_ticks
    if not isinstance(runner, dict):
        return None
    if _lock_holder_provably_dead(runner):
        return False
    if not _runner_checkable(runner):
        return None
    ticks = _proc_start_ticks(int(runner["pid"]))  # /proc/<pid>/stat is readable for other users' pids too
    if ticks is None:
        return None
    return ticks == runner["start_ticks"]


def _rule_changes(conn: sqlite3.Connection, now: float, cfg: dict[str, Any]) -> tuple[bool, list[tuple[Any, Any]]]:
    """What ``_apply_rules`` would change, read only: (any idle child to expire, ``(child_id, run_token)`` of every
    running child to orphan)."""
    expire = conn.execute(f"SELECT 1 FROM {TABLE} WHERE status=? AND idle_since IS NOT NULL AND ? - idle_since > ?"
                          " LIMIT 1", (IDLE, now, keep_idle_days(cfg) * 86400.0)).fetchone() is not None
    horizon = orphan_after_s(cfg)
    orphans = []
    # Tuple unpacking: the async-delegation ledger's connection has no row factory.
    for child_id, run_token, runner_raw, beat in conn.execute(
            f"SELECT child_id, run_token, runner, heartbeat_at FROM {TABLE} WHERE status=? ORDER BY rowid",
            (RUNNING,)).fetchall():
        try:
            runner = json.loads(runner_raw) if runner_raw else None
        except ValueError:
            runner = None
        alive = _runner_alive(runner)
        if alive is False:
            orphan = True
        elif alive is True:  # a provably alive runner is never orphaned on age
            orphan = False
        else:
            orphan = beat is None or now - float(beat) > horizon
        if orphan:
            orphans.append((child_id, run_token))
    return expire, orphans


def _apply_rules(conn: sqlite3.Connection, now: float, cfg: dict[str, Any]) -> None:
    """Orphan a running child whose runner is provably dead, or cannot be checked here and has not beaten for
    ``orphan_after_s``; then expire idle children past ``keep_idle_days``. A provably alive runner is never
    orphaned on age. The orphan pass runs first, so expiry sees the post-orphan state: a run that died longer
    ago than ``keep_idle_days`` is expired once, not idle-orphaned on one read and expired on the next. An
    orphaned run ended at its last heartbeat: ``idle_since`` is that time, not the time somebody looked."""
    _expire, orphans = _rule_changes(conn, now, cfg)
    keep_s = keep_idle_days(cfg) * 86400.0
    for child_id, run_token in orphans:
        conn.execute(
            f"UPDATE {TABLE} SET status=?, idle_reason=?, idle_since=COALESCE(heartbeat_at, ?), orphaned_at=?,"
            " runner=NULL, updated_at=? WHERE child_id=? AND status=? AND run_token IS ?",
            (IDLE, ORPHANED, now, now, now, child_id, RUNNING, run_token))
    conn.execute(f"UPDATE {TABLE} SET status=?, updated_at=? WHERE status=? AND idle_since IS NOT NULL"
                 " AND ? - idle_since > ?", (EXPIRED, now, IDLE, now, keep_s))


def _project(row: dict[str, Any], now: float, orphaned: set[Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """*row* as ``_apply_rules`` would have left it, for a read that could not write (read-only or locked store):
    the same orphan-then-expire state, derived here instead of persisted. The next writable read writes it."""
    if row.get("status") == RUNNING and row.get("child_id") in orphaned:
        row = dict(row, status=IDLE, idle_reason=ORPHANED, runner=None, orphaned_at=now,
                   idle_since=row.get("heartbeat_at") or now)
    if row.get("status") == IDLE and row.get("idle_since") is not None:
        if now - float(row["idle_since"]) > keep_idle_days(cfg) * 86400.0:
            row = dict(row, status=EXPIRED)
    return row


# ── listing (action='list') ─────────────────────────────────────────────────

def _visible_clause(conversations: Iterable[str]) -> tuple[str, tuple[Any, ...]]:
    """``(sql, params)``: the row's holder conversation — ``conversation_id`` or a ``prior_conversations``
    entry of a transfer — is one of *conversations*. Columns are unqualified (one table)."""
    convs = [c for c in conversations if c]
    marks = ", ".join("?" * len(convs))
    clause = (f"(conversation_id IN ({marks}) OR EXISTS (SELECT 1 FROM json_each(CASE WHEN"
               f" json_valid(prior_conversations) THEN prior_conversations ELSE '[]' END) j"
               f" WHERE j.value IN ({marks})))")
    return clause, (*convs, *convs)


def list_entry(row: dict[str, Any], *, conversations: Iterable[str] = (), profile: Optional[str] = None,
               holder_label: Optional[str] = None) -> dict[str, Any]:
    """One ``action='list'`` entry for a kept row: the ids the parent addresses it by, its status and whether
    this session may act on it. No brief, route or transcript: a listing stays small.

    ``read_only`` is True when the session may see the child but not resume, steer or stop it: another
    conversation of the profile holds it, or it ran as another profile (a ``profile_routes`` child, ``profile``
    on the row). A released child stays readable here; ``claim`` is what moves it.
    """
    convs = [c for c in conversations if c]
    held_here = row.get("conversation_id") in convs or bool(
        set(row.get("prior_conversations") or ()) & set(convs))
    other_profile = bool(row.get("profile")) and bool(profile) and row.get("profile") != profile
    read_only = (not held_here) or other_profile
    entry: dict[str, Any] = {
        "child_id": row.get("child_id"),
        "subagent_id": row.get("subagent_id"),
        "goal": row.get("goal"),
        "status": row.get("status"),
        "idle_reason": row.get("idle_reason"),
        "resumable": (not read_only) and row.get("status") == IDLE,
        "holder_session_id": row.get("holder_session_id"),
        "read_only": read_only,
    }
    if row.get("title"):
        entry["title"] = row["title"]
    if other_profile:
        entry["profile"] = row["profile"]
    if holder_label:
        entry["holder_label"] = holder_label[:HOLDER_LABEL_MAX]
    return entry


@dataclass(frozen=True)
class FenceResult:
    """Outcome of ``take_fence``. ``ok`` with a fresh ``token``; otherwise ``reason``: ``running`` (send a
    steer receipt instead of a second run), ``not_holder``, ``expired``, ``closed`` or ``unknown``.
    ``was_orphaned`` marks a run that restarts an orphaned child (the caller warns child and parent)."""
    ok: bool
    token: Optional[str] = None
    reason: Optional[str] = None
    was_orphaned: bool = False
    row: Optional[dict[str, Any]] = None

    @property
    def steer_receipt(self) -> bool:
        return not self.ok and self.reason == RUNNING


class ChildStore:
    """``delegate_children`` of one ``state.db``. Each call opens a short connection, so a store is safe to use
    from any thread (the worker done-callback, the tick, the parent). *session_db*, when given, is the SessionDB
    the lineage reads (``stored_child_session``) go through; otherwise the shared handle for *db_path*."""

    def __init__(self, db_path: Path | str, *, session_db: Any = None, cfg: Optional[dict[str, Any]] = None):
        self.db_path = Path(db_path)
        self._session_db = _plain_session_db(session_db)
        self._cfg = cfg

    @classmethod
    def for_agent(cls, agent: Any) -> Optional[ChildStore]:
        path = parent_db_path(agent)
        return cls(path, session_db=getattr(agent, "_session_db", None)) if path is not None else None

    # -- plumbing --

    def _config(self) -> dict[str, Any]:
        return _delegation_cfg() if self._cfg is None else self._cfg

    @contextlib.contextmanager
    def _write(self, busy_timeout_ms: int = 10_000) -> Iterator[sqlite3.Connection]:
        from hermes_cli.sqlite_util import transaction
        with transaction(_connect(self.db_path, busy_timeout_ms), immediate=True) as conn:
            yield conn

    @contextlib.contextmanager
    def _read_conn(self) -> Iterator[Optional[sqlite3.Connection]]:
        """A plain connection for reads: no DDL, no write transaction (works on a read-only file and while a
        writer holds the lock). None when the database or the table does not exist; nothing is created."""
        if not self.db_path.exists():
            yield None
            return
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            yield conn if _table_exists(conn) else None
        finally:
            conn.close()

    @contextlib.contextmanager
    def _lineage_db(self) -> Iterator[Any]:
        if self._session_db is not None:
            yield self._session_db
            return
        from hermes_state_registry import acquire, release_or_close
        db = acquire(self.db_path)
        try:
            yield db
        finally:
            release_or_close(db)

    # -- maintenance applied on every read --

    def _apply_rules(self, conn: sqlite3.Connection, now: float) -> None:
        _apply_rules(conn, now, self._config())

    def reconcile(self, now: Optional[float] = None) -> None:
        with self._write() as conn:
            self._apply_rules(conn, time.time() if now is None else now)

    def _read(self, sql: str, params: Iterable[Any] = (), *, now: Optional[float] = None) -> list[dict[str, Any]]:
        """Rows for *sql* with expiry and the orphan rule applied. Reads first; takes the write lock only when a
        rule would change a row, and degrades to the state ``_apply_rules`` would have written when the store is
        read-only or locked (the roster reads every turn and must not break on such a store)."""
        now = time.time() if now is None else now
        cfg = self._config()
        with self._read_conn() as conn:
            if conn is None:
                return []
            _expire, orphans = _rule_changes(conn, now, cfg)
            if not orphans and not _expire:
                return [_decode_row(r) for r in conn.execute(sql, tuple(params)).fetchall()]
            before = [_decode_row(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        try:
            with self._write(READ_WRITE_BUSY_MS) as conn:
                self._apply_rules(conn, now)
                return [_decode_row(r) for r in conn.execute(sql, tuple(params)).fetchall()]
        except sqlite3.OperationalError:
            logger.debug("delegate_children: rule write skipped (%s); showing the derived state", self.db_path,
                         exc_info=True)
            orphaned = {child_id for child_id, _ in orphans}
            return [_project(row, now, orphaned, cfg) for row in before]

    # -- record --

    def record_run_start(self, *, child_id: str, subagent_id: str, holder_session_id: Optional[str],
                         conversation_id: Optional[str], brief: dict[str, Any]) -> Optional[str]:
        """Insert the row for a new kept child and open its first run; returns the run token (None when a row
        with this ``child_id`` already exists: a spawn is recorded once)."""
        now, token = time.time(), uuid.uuid4().hex
        fields = {k: brief.get(k) for k in _BRIEF_FIELDS if brief.get(k) is not None}
        for key in ("route", "toolsets", "skills"):
            if key in fields:
                fields[key] = json.dumps(fields[key], default=str)
        fields.update(
            child_id=child_id, subagent_id=subagent_id, holder_session_id=holder_session_id,
            conversation_id=conversation_id, latest_session_id=child_id, status=RUNNING, kept=1, released=0,
            created_at=now, updated_at=now, run_token=token, runner=json.dumps(runner_record()), heartbeat_at=now,
        )
        cols = ", ".join(fields)
        with self._write() as conn:
            cur = conn.execute(f"INSERT OR IGNORE INTO {TABLE} ({cols}) VALUES ({', '.join('?' * len(fields))})",
                               tuple(fields.values()))
            return token if cur.rowcount == 1 else None

    def update_brief(self, child_id: str, run_token: str, **fields: Any) -> None:
        """Fill brief fields known only after the record (the worktree, the workdir). Guarded by the caller's
        ``run_token``: a stale run (its fence was taken over, or the row was orphaned and resumed) must not fill
        fields onto the row a newer run owns. e4: pass the token of the run doing the writing, never a cached one."""
        fields = {k: v for k, v in fields.items() if k in _BRIEF_FIELDS and v is not None}
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._write() as conn:
            conn.execute(f"UPDATE {TABLE} SET {sets}, updated_at=? WHERE child_id=? AND run_token IS ?",
                         (*fields.values(), time.time(), child_id, run_token))

    # -- run fence --

    def beat(self, child_id: str, token: str, now: Optional[float] = None) -> bool:
        """Touch ``heartbeat_at`` of the run *token*; False once that run is over."""
        now = time.time() if now is None else now
        with self._write() as conn:
            return conn.execute(f"UPDATE {TABLE} SET heartbeat_at=? WHERE child_id=? AND run_token=? AND status=?",
                                (now, child_id, token, RUNNING)).rowcount == 1

    def mark_wait_abandoned(self, child_id: str, token: str) -> bool:
        """The parent's wait ended (stale verdict, ``child_timeout``) with the worker alive: stays ``running``."""
        now = time.time()
        with self._write() as conn:
            return conn.execute(f"UPDATE {TABLE} SET wait_abandoned_at=?, updated_at=? WHERE child_id=? AND run_token=?"
                                " AND status=?", (now, now, child_id, token, RUNNING)).rowcount == 1

    def finish_run(self, child_id: str, token: str, reason: str) -> bool:
        """``running -> idle`` with *reason*: one compare-and-set, written only from the worker's done-callback (or
        its tick's retry). A late write from an older run (another token) or onto an already orphaned row changes
        nothing. ``latest_session_id`` is refreshed on read (``show``), not here."""
        now = time.time()
        with self._write() as conn:
            return conn.execute(
                f"UPDATE {TABLE} SET status=?, idle_reason=?, idle_since=?, runner=NULL, updated_at=?"
                " WHERE child_id=? AND run_token=? AND status=?",
                (IDLE, reason, now, now, child_id, token, RUNNING)).rowcount == 1

    def take_fence(self, child_id: str, *, holder_session_id: Optional[str] = None,
                   now: Optional[float] = None) -> FenceResult:
        """Start a new run of an idle child: one compare-and-set ``idle -> running`` with a new ``run_token`` and
        this process as runner. ``idle_since`` resets here and nowhere else. With *holder_session_id*, only the
        holder may take it."""
        now = time.time() if now is None else now
        token = uuid.uuid4().hex
        with self._write() as conn:
            self._apply_rules(conn, now)
            row = _decode(conn.execute(f"SELECT * FROM {TABLE} WHERE child_id=?", (child_id,)).fetchone())
            if row is None:
                return FenceResult(False, reason="unknown")
            if holder_session_id is not None and row["holder_session_id"] != holder_session_id:
                return FenceResult(False, reason="not_holder", row=row)
            if row["status"] != IDLE:
                return FenceResult(False, reason=row["status"], row=row)
            took = conn.execute(
                f"UPDATE {TABLE} SET status=?, run_token=?, runner=?, heartbeat_at=?, idle_since=NULL,"
                " idle_reason=NULL, orphaned_at=NULL, wait_abandoned_at=NULL, updated_at=?"
                " WHERE child_id=? AND status=? AND run_token IS ?",
                (RUNNING, token, json.dumps(runner_record()), now, now, child_id, IDLE, row["run_token"])).rowcount == 1
        if not took:
            return FenceResult(False, reason=RUNNING, row=row)
        return FenceResult(True, token=token, was_orphaned=row["idle_reason"] == ORPHANED, row=row)

    # -- lock --

    def hold(self, child_id: str, holder_session_id: str) -> bool:
        """The holder takes back a child it released (clears ``released``)."""
        with self._write() as conn:
            return conn.execute(f"UPDATE {TABLE} SET released=0, updated_at=? WHERE child_id=? AND holder_session_id=?"
                                " AND released=1 AND status IN (?, ?)",
                                (time.time(), child_id, holder_session_id, RUNNING, IDLE)).rowcount == 1

    def release(self, child_id: str, holder_session_id: str) -> bool:
        """The holder lets another session of the profile ``claim`` the child once it is idle."""
        with self._write() as conn:
            return conn.execute(f"UPDATE {TABLE} SET released=1, updated_at=? WHERE child_id=? AND holder_session_id=?"
                                " AND released=0 AND status IN (?, ?)",
                                (time.time(), child_id, holder_session_id, RUNNING, IDLE)).rowcount == 1

    @staticmethod
    def _move(conn: sqlite3.Connection, row: dict[str, Any], session_id: str, conversation: Optional[str],
              extra_where: str, params: tuple) -> bool:
        prior = list(row.get("prior_conversations") or [])
        old = row.get("conversation_id")
        if old and old != conversation and old not in prior:
            prior.append(old)
        return conn.execute(
            f"UPDATE {TABLE} SET holder_session_id=?, conversation_id=?, prior_conversations=?, released=0,"
            f" updated_at=? WHERE child_id=? AND holder_session_id IS ? {extra_where}",
            (session_id, conversation, json.dumps(prior), time.time(), row["child_id"], row["holder_session_id"],
             *params)).rowcount == 1

    def claim(self, child_id: str, session_id: str, conversation_id: Optional[str]) -> bool:
        """Another session of the profile takes a released idle child. A running child is never claimable: its
        runner owns it until it goes idle."""
        with self._write() as conn:
            self._apply_rules(conn, time.time())
            row = _decode(conn.execute(f"SELECT * FROM {TABLE} WHERE child_id=?", (child_id,)).fetchone())
            if row is None or not row["released"] or row["status"] != IDLE:
                return False
            return self._move(conn, row, session_id, conversation_id, "AND released=1 AND status=?", (IDLE,))

    def transfer(self, from_session_id: str, to_session_id: str, to_conversation_id: Optional[str]) -> dict[str, int]:
        """Move every child *from_session_id* holds (not released, not closed or expired) to the new session."""
        moved = running = 0
        with self._write() as conn:
            self._apply_rules(conn, time.time())
            rows = [_decode_row(r) for r in conn.execute(
                f"SELECT * FROM {TABLE} WHERE holder_session_id=? AND released=0 AND status IN (?, ?)",
                (from_session_id, RUNNING, IDLE)).fetchall()]
            for row in rows:
                if self._move(conn, row, to_session_id, to_conversation_id, "AND released=0", ()):
                    moved += 1
                    running += row["status"] == RUNNING
        return {"moved": moved, "running": running}

    # -- reads (expiry and orphans applied first; ``idle_since`` never touched) --

    def get(self, child_id: str, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
        rows = self._read(f"SELECT * FROM {TABLE} WHERE child_id=?", (child_id,), now=now)
        return rows[0] if rows else None

    def held_by(self, holder_session_id: str, *, now: Optional[float] = None) -> list[dict[str, Any]]:
        return self._read(f"SELECT * FROM {TABLE} WHERE holder_session_id=? AND status IN (?, ?)"
                          " ORDER BY created_at DESC", (holder_session_id, RUNNING, IDLE), now=now)

    @property
    def profile(self) -> str:
        """This db's own profile: the name of its ``profiles/<name>`` home, else ``default``."""
        return _profile_for(self.db_path)

    def kept_children(self, *, conversations: Optional[Iterable[str]] = None,
                      exclude_conversations: Iterable[str] = (), exclude_subagent_ids: Iterable[str] = (),
                      query: Optional[str] = None, now: Optional[float] = None,
                      limit: int = LIST_LIMIT) -> tuple[list[dict[str, Any]], int]:
        """Kept rows for ``action='list'``, newest activity first, with expiry/orphan rules applied on read:
        ``(rows, omitted)``.

        With *conversations*, only the children that session holds (its ``conversation_id`` or a
        ``prior_conversations`` entry — the same visibility ``resolve`` uses); without, every kept row of this
        db except those *exclude_conversations* can see (the profile-wide list). *exclude_subagent_ids* drops
        the rows of children already listed as live in this process. *query* is a case-insensitive substring
        over goal, child_id, subagent_id and title. At most *limit* rows come back; *omitted* counts the rest.
        A read: the table is never created and no write lock is taken unless a rule is due.
        """
        where = ["kept = 1"]
        params: list[Any] = []
        if conversations is not None:
            convs = [c for c in conversations if c]
            if not convs:
                return [], 0
            clause, values = _visible_clause(convs)
            where.append(clause)
            params.extend(values)
        excluded = [c for c in exclude_conversations if c]
        if excluded:
            clause, values = _visible_clause(excluded)
            where.append(f"NOT {clause}")
            params.extend(values)
        live_ids = [s for s in exclude_subagent_ids if s]
        if live_ids:
            where.append(f"subagent_id NOT IN ({', '.join('?' * len(live_ids))})")
            params.extend(live_ids)
        needle = (query or "").strip().lower()
        if needle:
            fields = ("COALESCE(goal, '')", "child_id", "COALESCE(subagent_id, '')", "COALESCE(title, '')")
            where.append("(" + " OR ".join(f"instr(lower({field}), ?) > 0" for field in fields) + ")")
            params.extend([needle] * len(fields))
        activity = "max(COALESCE(updated_at, 0), COALESCE(heartbeat_at, 0), COALESCE(created_at, 0))"
        sql = (f"SELECT * FROM {TABLE} WHERE " + " AND ".join(where)
               + f" ORDER BY {activity} DESC, child_id LIMIT ?")
        rows = self._read(sql, (*params, int(limit) + 1), now=now)
        if len(rows) <= limit:
            return rows, 0
        counted = self._read(f"SELECT COUNT(*) AS total FROM {TABLE} WHERE " + " AND ".join(where),
                             tuple(params), now=now)
        total = int(counted[0]["total"]) if counted else len(rows)
        return rows[:limit], max(0, total - int(limit))

    def holder_labels(self, session_ids: Iterable[Optional[str]]) -> dict[str, str]:
        """``{holder session id: title}`` for the sessions a listing names, one read of the same ``state.db``
        (``sessions`` lives beside ``delegate_children``). ``{}`` when the table or its title column is absent."""
        ids = [s for s in dict.fromkeys(session_ids) if s]
        if not ids:
            return {}
        with self._read_conn() as conn:
            if conn is None:
                return {}
            try:
                rows = conn.execute(f"SELECT id, title FROM sessions WHERE id IN ({', '.join('?' * len(ids))})",
                                    ids).fetchall()
            except sqlite3.Error:
                return {}
        return {str(r["id"]): str(r["title"]).strip() for r in rows
                if r["title"] is not None and str(r["title"]).strip()}

    def show(self, ref: str, conversation_ids: Iterable[str], *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
        """The row for *ref* (a ``child_id`` or a ``subagent_id`` alias) visible to *conversation_ids*, with
        ``latest_session_id`` refreshed. A read: ``idle_since`` is left alone."""
        child_id = self.resolve(ref, conversation_ids, now=now)
        if child_id is None:
            return None
        with contextlib.suppress(Exception):
            self.refresh_latest(child_id)
        return self.get(child_id, now=now)

    def resolve(self, ref: str, conversation_ids: Iterable[str], *, now: Optional[float] = None) -> Optional[str]:
        """``child_id`` for *ref*. A ``child_id`` resolves directly when its holder conversation (or a prior one) is
        among *conversation_ids*; a ``subagent_id`` resolves only through ``stored_child_session``."""
        from tools.delegate_tool_sessions import stored_child_session
        convs = [c for c in conversation_ids if c]
        if not ref or not convs:
            return None

        def visible(row: dict[str, Any]) -> bool:
            return row.get("conversation_id") in convs or bool(set(row.get("prior_conversations") or ()) & set(convs))

        direct = self.get(ref, now=now)
        if direct is not None:
            return ref if visible(direct) else None
        candidates = [r for r in self._read(f"SELECT * FROM {TABLE} WHERE subagent_id=?", (ref,), now=now) if visible(r)]
        if not candidates:
            return None
        # A transferred or claimed child's transcript still descends from the conversation it was spawned in.
        lookup = list(dict.fromkeys([*convs, *(c for r in candidates for c in (r.get("prior_conversations") or ()))]))
        with self._lineage_db() as db:
            tip = stored_child_session(db, ref, lookup)
            if tip is None:
                return None
            ids = {r["child_id"] for r in candidates}
            seen: set[str] = set()
            current: Optional[str] = tip
            while current and current not in seen:  # the tip's own segment chain reaches its first segment
                if current in ids:
                    return current
                seen.add(current)
                row = db.get_session(current)
                current = row.get("parent_session_id") if row else None
        return None

    def refresh_latest(self, child_id: str) -> Optional[str]:
        """Re-read the child's newest segment from ``stored_child_session`` against the holder's conversation and its
        prior conversations; stored on the row when found. A read path: it waits at most ``READ_WRITE_BUSY_MS``
        for a writer's lock and leaves the row for the next read when it cannot write."""
        from tools.delegate_tool_sessions import stored_child_session
        with self._read_conn() as conn:
            row = None if conn is None else _decode(
                conn.execute(f"SELECT * FROM {TABLE} WHERE child_id=?", (child_id,)).fetchone())
        if row is None:
            return None
        convs = [c for c in [row.get("conversation_id"), *(row.get("prior_conversations") or [])] if c]
        with self._lineage_db() as db:
            tip = stored_child_session(db, row["subagent_id"], convs)
        if tip and tip != row.get("latest_session_id"):
            with self._write(READ_WRITE_BUSY_MS) as conn:
                conn.execute(f"UPDATE {TABLE} SET latest_session_id=?, updated_at=? WHERE child_id=?",
                             (tip, time.time(), child_id))
        return tip or row.get("latest_session_id")


# ── keep rule ───────────────────────────────────────────────────────────────

def _parent_source(parent_agent: Any, db_path: Path) -> str:
    """The parent's session source: its stored row's, else what the agent would write."""
    sid = getattr(parent_agent, "session_id", None)
    if isinstance(sid, str) and sid and db_path.exists():
        with contextlib.suppress(Exception):
            conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
            try:
                row = conn.execute("SELECT source FROM sessions WHERE id=?", (sid,)).fetchone()
            finally:
                conn.close()
            if row is not None and row[0]:
                return str(row[0])
    from agent.session_source import session_source_for
    platform = getattr(parent_agent, "platform", None)
    return session_source_for(platform if isinstance(platform, str) else None)


def should_keep(parent_agent: Any, *, origin: Optional[str], child_depth: int, keep: Optional[bool] = None,
                cfg: Optional[dict[str, Any]] = None) -> bool:
    """Whether a child spawned by *parent_agent* is kept: the model spawned it (``origin == "model"``), the parent
    exposes a ``state.db`` path, and either it is a depth-1 child of a parent whose session source is in
    ``delegation.keep_sources`` and is not a one-shot run, or (grandchild) the task said ``keep: true``."""
    if origin != ORIGIN_MODEL or getattr(parent_agent, NEVER_KEEP_ATTR, False) is True:
        return False
    db_path = parent_db_path(parent_agent)
    if db_path is None:
        return False
    if child_depth > 1:
        return keep is True
    if keep is False or getattr(parent_agent, "_oneshot_session", None) is True:
        return False
    with contextlib.suppress(Exception):
        from agent.oneshot_footprint import is_single_query_session
        if is_single_query_session():
            return False
    return _parent_source(parent_agent, db_path).strip().lower() in keep_sources(cfg)


# ── spawn / worker hooks ────────────────────────────────────────────────────

_PENDING_ATTR = "_delegate_children_pending"


def note_spawn(parent_agent: Any, child: Any, *, goal: str, context: Optional[str], toolsets: Any, role: Any,
               model: Optional[str], route: Optional[dict[str, Any]] = None, profile_home: Any = None,
               title: Optional[str] = None, keep: Optional[bool] = None, rebuilt: bool = False) -> bool:
    """Spawn hook (``_build_child_agent``): when the child is kept, stash its record on the child. The row is
    written when its worker starts (``worker_started``), so a child that never runs leaves no row; the holder and
    its conversation are read then too, so a ``/new`` in between lands the row on the new session. *rebuilt*
    (``action='continue'``): the child already ran, so its existing row is found and re-opened instead."""
    try:
        if rebuilt:
            return _note_rebuilt(parent_agent, child)
        depth = int(getattr(child, "_delegate_depth", 1) or 1)
        if not should_keep(parent_agent, origin=current_spawn_origin(), child_depth=depth, keep=keep):
            return False
        child_id = str(getattr(child, "session_id", "") or "")
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        db_path = parent_db_path(parent_agent)
        if not child_id or not subagent_id or db_path is None:
            return False
        base_url = (route or {}).get("base_url")
        route = {k: v for k, v in (route or {}).items() if k in _ROUTE_KEYS and v not in (None, "")}
        if base_url not in (None, ""):
            from tools.delegate_tool_resumable import endpoint_identity
            route["endpoint"] = list(endpoint_identity(route.get("provider"), base_url))
        if "profile_home" in route:
            route["profile_home"] = str(route["profile_home"])
        brief = {
            "origin": ORIGIN_MODEL, "goal": goal, "context": context, "title": title, "route": route or None,
            "model": model, "provider": route.get("provider"), "toolsets": list(toolsets) if toolsets else None,
            "role": role, "skills": list(getattr(child, "_delegate_skills", None) or []) or None,
            "profile": _profile_for(db_path, profile_home), "depth": depth,
            "parent_child_id": getattr(parent_agent, "_delegate_child_id", None),
        }
        child._delegate_child_id = child_id
        setattr(child, _PENDING_ATTR, {
            "db_path": db_path, "session_db": _plain_session_db(getattr(parent_agent, "_session_db", None)),
            "child_id": child_id, "subagent_id": subagent_id, "parent": parent_agent, "brief": brief,
        })
        return True
    except Exception:
        logger.warning("delegate_children: could not note a kept child", exc_info=True)
        return False


def _note_rebuilt(parent_agent: Any, child: Any) -> bool:
    """``action='continue'`` rebuilt a finished child under its own ``subagent_id``: find the kept row its first
    run wrote (lineage-checked, as ``show`` resolves a ``subagent_id``) and stash it for ``worker_started`` to
    re-open. A child with no row was never kept and stays that way."""
    if current_spawn_origin() != ORIGIN_MODEL:
        return False
    from agent.prompt_cache_scope import conversation_id
    store = ChildStore.for_agent(parent_agent)
    subagent_id = str(getattr(child, "_subagent_id", "") or "")
    child_id = store.resolve(subagent_id, [conversation_id(parent_agent) or ""]) if store and subagent_id else None
    if not child_id:
        return False
    child._delegate_child_id = child_id
    setattr(child, _PENDING_ATTR, {"db_path": store.db_path, "child_id": child_id, "subagent_id": subagent_id,
                                   "session_db": _plain_session_db(getattr(parent_agent, "_session_db", None)),
                                   "reopen": True})
    return True


class _Tick:
    """Our own progress-independent heartbeat: touches ``heartbeat_at`` every ``HEARTBEAT_INTERVAL_S`` while the
    worker lives. Once the worker is done (``finish``) it stops beating and, if the done-callback's single write
    did not land, retries that write each interval until it lands; meanwhile the heartbeat goes stale."""

    def __init__(self, store: ChildStore, child_id: str, token: str):
        from agent.memory_provider import spawn_context_thread
        self.store, self.child_id, self.token = store, child_id, token
        self.stopped = threading.Event()
        self.reason: Optional[str] = None
        self.thread = spawn_context_thread(self._loop, name=f"delegate-child-beat-{child_id[:12]}")

    def _loop(self) -> None:
        while not self.stopped.wait(HEARTBEAT_INTERVAL_S):
            try:
                if self.reason is not None:
                    self.store.finish_run(self.child_id, self.token, self.reason)
                    self.stopped.set()
                elif not self.store.beat(self.child_id, self.token):
                    return
            except Exception:  # the tick is the only liveness signal: never let it end early
                logger.debug("delegate_children: heartbeat/final write failed", exc_info=True)

    def start(self) -> _Tick:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stopped.set()

    def finish(self, reason: str) -> bool:
        """The worker is done: stop beating and write ``running -> idle`` on this one attempt, so a write-locked
        store cannot delay the parent's result (the tick, while it runs, retries the write each interval)."""
        self.reason = reason
        try:
            self.store.finish_run(self.child_id, self.token, reason)
        except (sqlite3.Error, OSError):
            logger.debug("delegate_children: final write for %s failed", self.child_id, exc_info=True)
            logger.warning("delegate_children: could not mark child %s idle; %s", self.child_id,
                           "its tick keeps retrying" if self.thread.is_alive()
                           else "it stays running until the process exits")
            return False
        self.stopped.set()
        return True


def _done_reason(future: Any) -> str:
    """``completed`` only for a clean completion (``completed`` is True, not failed or interrupted), which is
    what the parent sees as completed; anything else (failed, max iterations, interrupted, an exception, a
    cancelled future) is ``interrupted``. Both are resumable."""
    if future.cancelled() or future.exception() is not None:
        return INTERRUPTED
    result = future.result()
    if (isinstance(result, dict) and result.get("completed") is True and not result.get("failed")
            and not result.get("interrupted")):
        return COMPLETED
    return INTERRUPTED


def _holder_now(parent: Any) -> tuple[Optional[str], Optional[str]]:
    """The parent's session and conversation as they are now (the row is inserted at worker start)."""
    from agent.prompt_cache_scope import conversation_id
    sid = getattr(parent, "session_id", None)
    if parent is None or not isinstance(sid, str) or not sid:
        return None, None
    return sid, conversation_id(parent) or None


def worker_started(child: Any, future: Any, *, task_index: Optional[int] = None,
                   worktree: Optional[dict[str, Any]] = None, workdir: Optional[str] = None) -> Optional[str]:
    """Worker-start hook (``_ChildRun.await_child``): open the run (insert the row for a new kept child, or use the
    fence a resume took: ``child._delegate_run_token``), set ``idle`` from *future*'s done-callback, and start the
    tick. The callback is registered before the tick starts, so a tick that cannot start still ends the run.
    Returns the run token, or None for a child that is not kept."""
    pending = getattr(child, _PENDING_ATTR, None)
    if not isinstance(pending, dict):
        return None
    try:
        store = ChildStore(pending["db_path"], session_db=pending.get("session_db"))
        child_id = pending["child_id"]
        parent = pending.pop("parent", None)  # no lasting child -> parent reference
        token = getattr(child, "_delegate_run_token", None)
        if pending.get("reopen"):  # a continue: the same row runs again, under a new fence
            fence = store.take_fence(child_id)
            if not fence.ok:
                logger.info("delegate_children: continued child %s keeps its row as is (%s)", child_id, fence.reason)
                return None
            token = fence.token
        elif not token:
            # The async ledger's unit id when the call runs in the background (what a replay is keyed on), else
            # the live-transcript id of the call.
            delegation_id = getattr(child, "_delegation_unit_id", None) or getattr(child, "_delegation_id", None)
            brief = dict(pending["brief"], delegation_id=delegation_id, task_index=task_index, workdir=workdir)
            if worktree:
                brief.update(repo_root=worktree.get("repo_root"), branch=worktree.get("branch"),
                             base_commit=worktree.get("base_commit"), workdir=worktree.get("path") or workdir)
            holder, conversation = _holder_now(parent)
            token = store.record_run_start(child_id=child_id, subagent_id=pending["subagent_id"],
                                           holder_session_id=holder, conversation_id=conversation, brief=brief)
            if token is None:
                return None
    except Exception:
        logger.warning("delegate_children: could not record a kept child's run", exc_info=True)
        return None
    child._delegate_run_token = token
    tick = _Tick(store, child_id, token)
    child._delegate_children_tick = tick
    future.add_done_callback(lambda done_future: tick.finish(_done_reason(done_future)))
    try:
        tick.start()  # exits at once when the callback already ran and wrote the final state
    except RuntimeError:  # no thread to spare: the done-callback still ends the run
        logger.warning("delegate_children: heartbeat for child %s could not start", child_id, exc_info=True)
    return token


def wait_abandoned(child: Any) -> None:
    """The parent stopped waiting while the worker is still alive: the row stays ``running`` (stamped
    ``wait_abandoned_at``) and the tick keeps beating until the worker exits."""
    pending, token = getattr(child, _PENDING_ATTR, None), getattr(child, "_delegate_run_token", None)
    if isinstance(pending, dict) and token:
        with contextlib.suppress(Exception):
            ChildStore(pending["db_path"]).mark_wait_abandoned(pending["child_id"], token)


# ── async-delegation recovery ───────────────────────────────────────────────

def orphaned_unit_note(conn: sqlite3.Connection, delegation_id: str,
                       task_indexes: Optional[Iterable[int]] = None) -> Optional[str]:
    """For ``recover_abandoned_delegations``: apply the orphan rule to the kept children of the abandoned unit
    *delegation_id* (on the ledger's own connection) and return ``ORPHANED_LINE`` for each that is now
    idle-orphaned, naming its ``subagent_id``. Units of a grouped call carry ``<id>-<n>``; their children record the call's id."""
    if not delegation_id or not _table_exists(conn):
        return None
    base = delegation_id.rsplit("-", 1)[0] if delegation_id.rsplit("-", 1)[-1].isdigit() else delegation_id
    _apply_rules(conn, time.time(), _delegation_cfg())
    rows = conn.execute(f"SELECT task_index, status, idle_reason, subagent_id FROM {TABLE} "
                        "WHERE delegation_id IN (?, ?) ORDER BY task_index", (delegation_id, base)).fetchall()
    wanted = set(task_indexes) if task_indexes else None
    hit = [r[3] for r in rows if r[1] == IDLE and r[2] == ORPHANED and (wanted is None or r[0] is None or r[0] in wanted)]
    return orphaned_line(hit) if hit else None


# ── session pruning ─────────────────────────────────────────────────────────

def _kept_row_sql() -> str:
    """The ``delegate_children`` rows whose protection is live: retained (``kept``) and still running/idle."""
    return f"d.kept = 1 AND d.status IN ('{RUNNING}', '{IDLE}')"


def kept_conversation_roots_sql() -> str:
    """Session ids of the holder's conversations for every retained child: ``conversation_id`` plus the
    ``prior_conversations`` of a transfer. These are the rows where each ownership walk stops, spared when the
    walk reaches them (``owns()`` needs the root row to exist to reach it), and it spares nothing above them: a
    root that is not on a walk is not in the guard set, so a root's own parent (the pre-/reset session, say) is
    above the walk and stays prunable. The rows are also the seeds a parent-sparing step must NOT use (a root's
    own parent is not on the chain); ``kept_sessions_sql`` reuses this fragment for its ``root`` CTE."""
    kept_row = _kept_row_sql()
    return (f" SELECT d.conversation_id FROM {TABLE} d WHERE {kept_row} AND d.conversation_id IS NOT NULL"
            " UNION"
            f" SELECT j.value FROM {TABLE} d, json_each(CASE WHEN json_valid(d.prior_conversations)"
            f" THEN d.prior_conversations ELSE '[]' END) j WHERE {kept_row} AND j.type = 'text'")


def kept_sessions_sql(continuation_edge_sql: str) -> str:
    """Session ids that must outlive a prune because a kept child still needs them.

    Two sets, one recursive CTE (the predicate runs inside the prune's own statement, so it must stay
    set-based and fail closed on its own):

    * ``kept_child`` — the child's own rows: its first session, its recorded latest segment, and every
      compression continuation after them (``continuation_edge_sql`` relates child ``c`` to parent ``p``).
    * ``chain`` — the ownership chain above them, walked exactly as ``tools.delegate_tool_sessions``
      ``_lineage_reader`` walks it: a hop is the row's ``_delegate_from`` naming its own parent (a
      delegate hop) or a compression continuation (``p.end_reason = 'compression'`` and the row is not
      an explicit fork/delegate/reset/tool child). The walk stops at the holder's conversation roots
      (``kept_conversation_roots_sql``: ``conversation_id`` plus ``prior_conversations``), which are
      spared too: ``owns()`` needs the root row to exist to reach it. A root's own parent — the
      pre-/reset session, say — is above the walk and is NOT spared.

    Nothing here decides ownership; it only widens what prune must not delete (and must not orphan by
    NULLing ``parent_session_id``). Protection ends with the row's status: an expired or closed child
    contributes neither set, so the chain it shared becomes prunable again.
    """
    from hermes_state_common import _sql_json_extract
    kept_row = _kept_row_sql()
    hop = f"({_sql_json_extract('c.model_config', '$._delegate_from')} = p.id OR ({continuation_edge_sql}))"
    return ("WITH RECURSIVE kept_child(id) AS ("
            f" SELECT d.child_id FROM {TABLE} d WHERE {kept_row}"
            " UNION"
            f" SELECT d.latest_session_id FROM {TABLE} d WHERE {kept_row}"
            " AND d.latest_session_id IS NOT NULL"
            " UNION"
            " SELECT c.id FROM kept_child k JOIN sessions p ON p.id = k.id JOIN sessions c ON c.parent_session_id = p.id"
            f" WHERE {continuation_edge_sql}"
            "), root(id) AS ("
            f"{kept_conversation_roots_sql()}"
            "), chain(id) AS ("
            " SELECT p.id FROM kept_child k JOIN sessions c ON c.id = k.id JOIN sessions p ON p.id = c.parent_session_id"
            f" WHERE {hop} AND NOT EXISTS (SELECT 1 FROM root r WHERE r.id = k.id)"
            " UNION"
            " SELECT p.id FROM chain ch JOIN sessions c ON c.id = ch.id JOIN sessions p ON p.id = c.parent_session_id"
            f" WHERE {hop} AND NOT EXISTS (SELECT 1 FROM root r WHERE r.id = ch.id)"
            ") SELECT id FROM kept_child UNION SELECT id FROM chain")
