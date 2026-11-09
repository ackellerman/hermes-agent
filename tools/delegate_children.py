"""Kept delegated children: the durable record that lets a parent find, hold and later resume a child.

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
* Expiry and the orphan rule are applied on every read. ``idle_since`` is reset only by a new run
  (``take_fence``); reads never touch it.

Credentials and provider clients are never stored; a resume re-derives them.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
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

ORPHANED_LINE = "child kept as idle-orphaned; resumable with action='resume'"

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

# Route keys that may be stored. Never an api_key, client or credential pool.
_ROUTE_KEYS = ("provider", "model", "base_url", "api_mode", "reasoning_effort", "service_tier",
               "requested_provider", "acp_command", "acp_args", "profile_home")


# ── config ──────────────────────────────────────────────────────────────────

def _delegation_cfg() -> dict[str, Any]:
    try:
        from tools.delegate_tool_config import _load_config
        cfg = _load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def keep_sources(cfg: Optional[dict[str, Any]] = None) -> frozenset[str]:
    """``delegation.keep_sources`` as a set; unset or malformed means ``DEFAULT_KEEP_SOURCES``."""
    raw = (_delegation_cfg() if cfg is None else cfg).get("keep_sources")
    if isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset(str(s).strip().lower() for s in raw if str(s).strip())
    return DEFAULT_KEEP_SOURCES


def _positive_number(raw: Any, default: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return default
    return float(raw)


def keep_idle_days(cfg: Optional[dict[str, Any]] = None) -> float:
    return _positive_number((_delegation_cfg() if cfg is None else cfg).get("keep_idle_days"), DEFAULT_KEEP_IDLE_DAYS)


def orphan_after_s(cfg: Optional[dict[str, Any]] = None) -> float:
    return _positive_number((_delegation_cfg() if cfg is None else cfg).get("orphan_after_s"), DEFAULT_ORPHAN_AFTER_S)


def protects_transcripts(cfg: Optional[dict[str, Any]] = None) -> bool:
    """``delegation.keep_protects_transcripts`` (default True): session pruning spares kept children."""
    raw = (_delegation_cfg() if cfg is None else cfg).get("keep_protects_transcripts", True)
    return raw is not False and str(raw).strip().lower() not in ("false", "0", "no", "off")


# ── spawn origin ────────────────────────────────────────────────────────────

_SPAWN_ORIGIN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("delegate_spawn_origin", default=None)


@contextlib.contextmanager
def spawn_origin(origin: str) -> Iterator[None]:
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

def parent_db_path(parent_agent: Any) -> Optional[Path]:
    """The parent's ``state.db`` path, or None. Concrete str/Path only (a MagicMock is PathLike)."""
    path = getattr(getattr(parent_agent, "_session_db", None), "db_path", None)
    return Path(path) if isinstance(path, (str, Path)) else None


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)


def _connect(db_path: Path) -> sqlite3.Connection:
    from hermes_cli.sqlite_util import open_db
    # wal=False: SessionDB owns state.db's journal mode.
    return open_db(db_path, db_label=f"state.db ({TABLE})", busy_timeout_ms=10_000, wal=False,
                   row_factory=sqlite3.Row, initialize=_ensure_table)


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


def _apply_rules(conn: sqlite3.Connection, now: float, cfg: dict[str, Any]) -> None:
    """Expire idle children past ``keep_idle_days``; orphan a running child whose runner is provably dead, or
    cannot be checked here (foreign pid namespace, no start ticks) and has not beaten for ``orphan_after_s``.
    A provably alive runner is never orphaned on age."""
    from hermes_state_common import _lock_holder_provably_dead
    conn.execute(f"UPDATE {TABLE} SET status=?, updated_at=? WHERE status=? AND idle_since IS NOT NULL"
                 " AND ? - idle_since > ?", (EXPIRED, now, IDLE, now, keep_idle_days(cfg) * 86400.0))
    horizon = orphan_after_s(cfg)
    # Tuple unpacking: the async-delegation ledger's connection has no row factory.
    for child_id, run_token, runner_raw, beat in conn.execute(
            f"SELECT child_id, run_token, runner, heartbeat_at FROM {TABLE} WHERE status=?", (RUNNING,)).fetchall():
        try:
            runner = json.loads(runner_raw) if runner_raw else None
        except ValueError:
            runner = None
        if _lock_holder_provably_dead(runner if isinstance(runner, dict) else {}):
            orphan = True
        elif _runner_checkable(runner):
            orphan = False
        else:
            orphan = beat is None or now - float(beat) > horizon
        if orphan:
            conn.execute(
                f"UPDATE {TABLE} SET status=?, idle_reason=?, idle_since=?, orphaned_at=?, runner=NULL, updated_at=?"
                " WHERE child_id=? AND status=? AND run_token IS ?",
                (IDLE, ORPHANED, now, now, now, child_id, RUNNING, run_token))


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
        self._session_db = session_db
        self._cfg = cfg

    @classmethod
    def for_agent(cls, agent: Any) -> Optional["ChildStore"]:
        path = parent_db_path(agent)
        return cls(path, session_db=getattr(agent, "_session_db", None)) if path is not None else None

    # -- plumbing --

    def _config(self) -> dict[str, Any]:
        return _delegation_cfg() if self._cfg is None else self._cfg

    @contextlib.contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        from hermes_cli.sqlite_util import transaction
        with transaction(_connect(self.db_path), immediate=True) as conn:
            yield conn

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
        with self._write() as conn:
            self._apply_rules(conn, time.time() if now is None else now)
            return [_decode_row(r) for r in conn.execute(sql, tuple(params)).fetchall()]

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

    def update_brief(self, child_id: str, **fields: Any) -> None:
        """Fill brief fields known only after the record (the worktree, the workdir)."""
        fields = {k: v for k, v in fields.items() if k in _BRIEF_FIELDS and v is not None}
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._write() as conn:
            conn.execute(f"UPDATE {TABLE} SET {sets}, updated_at=? WHERE child_id=?",
                         (*fields.values(), time.time(), child_id))

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
        """``running -> idle`` with *reason*; only the worker's done-callback calls this. A late write from an older
        run (another token) or onto an already orphaned row changes nothing."""
        now = time.time()
        with self._write() as conn:
            done = conn.execute(
                f"UPDATE {TABLE} SET status=?, idle_reason=?, idle_since=?, runner=NULL, updated_at=?"
                " WHERE child_id=? AND run_token=? AND status=?",
                (IDLE, reason, now, now, child_id, token, RUNNING)).rowcount == 1
        if done:
            with contextlib.suppress(Exception):
                self.refresh_latest(child_id)
        return done

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
        prior conversations; stored on the row when found."""
        from tools.delegate_tool_sessions import stored_child_session
        with self._write() as conn:
            row = _decode(conn.execute(f"SELECT * FROM {TABLE} WHERE child_id=?", (child_id,)).fetchone())
        if row is None:
            return None
        convs = [c for c in [row.get("conversation_id"), *(row.get("prior_conversations") or [])] if c]
        with self._lineage_db() as db:
            tip = stored_child_session(db, row["subagent_id"], convs)
        if tip and tip != row.get("latest_session_id"):
            with self._write() as conn:
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
               title: Optional[str] = None, keep: Optional[bool] = None) -> bool:
    """Spawn hook (``_build_child_agent``): when the child is kept, stash its record on the child. The row is
    written when its worker starts (``worker_started``), so a child that never runs leaves no row."""
    try:
        depth = int(getattr(child, "_delegate_depth", 1) or 1)
        if not should_keep(parent_agent, origin=current_spawn_origin(), child_depth=depth, keep=keep):
            return False
        child_id = str(getattr(child, "session_id", "") or "")
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        db_path = parent_db_path(parent_agent)
        if not child_id or not subagent_id or db_path is None:
            return False
        from agent.prompt_cache_scope import conversation_id
        route = {k: v for k, v in (route or {}).items() if k in _ROUTE_KEYS and v not in (None, "")}
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
            "db_path": db_path, "session_db": getattr(parent_agent, "_session_db", None),
            "child_id": child_id, "subagent_id": subagent_id,
            "holder_session_id": getattr(parent_agent, "session_id", None),
            "conversation_id": conversation_id(parent_agent), "brief": brief,
        })
        return True
    except Exception:
        logger.warning("delegate_children: could not note a kept child", exc_info=True)
        return False


class _Tick:
    """Our own progress-independent heartbeat: touches ``heartbeat_at`` every ``HEARTBEAT_INTERVAL_S`` while the
    worker thread lives; stops when the run is no longer ours or the worker's done-callback stops it."""

    def __init__(self, store: ChildStore, child_id: str, token: str):
        self.store, self.child_id, self.token = store, child_id, token
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, name=f"delegate-child-beat-{child_id[:12]}", daemon=True)

    def _loop(self) -> None:
        while not self.stopped.wait(HEARTBEAT_INTERVAL_S):
            try:
                if not self.store.beat(self.child_id, self.token):
                    return
            except Exception:
                logger.debug("delegate_children: heartbeat write failed", exc_info=True)

    def start(self) -> "_Tick":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.stopped.set()


def _done_reason(future: Any) -> str:
    try:
        if future.cancelled() or future.exception() is not None:
            return INTERRUPTED
        result = future.result()
    except BaseException:
        return INTERRUPTED
    if isinstance(result, dict) and result.get("interrupted"):
        return INTERRUPTED
    return COMPLETED


def worker_started(child: Any, future: Any, *, task_index: Optional[int] = None,
                   worktree: Optional[dict[str, Any]] = None, workdir: Optional[str] = None) -> Optional[str]:
    """Worker-start hook (``_ChildRun.await_child``): open the run (insert the row for a new kept child, or use the
    fence a resume took: ``child._delegate_run_token``), start the tick, and set ``idle`` from *future*'s
    done-callback. Returns the run token, or None for a child that is not kept."""
    pending = getattr(child, _PENDING_ATTR, None)
    if not isinstance(pending, dict):
        return None
    try:
        store = ChildStore(pending["db_path"], session_db=pending.get("session_db"))
        child_id = pending["child_id"]
        token = getattr(child, "_delegate_run_token", None)
        if not token:
            brief = dict(pending["brief"], delegation_id=getattr(child, "_delegation_id", None), task_index=task_index,
                         workdir=workdir)
            if worktree:
                brief.update(repo_root=worktree.get("repo_root"), branch=worktree.get("branch"),
                             base_commit=worktree.get("base_commit"), workdir=worktree.get("path") or workdir)
            token = store.record_run_start(
                child_id=child_id, subagent_id=pending["subagent_id"], holder_session_id=pending["holder_session_id"],
                conversation_id=pending["conversation_id"], brief=brief)
            if token is None:
                return None
        child._delegate_run_token = token
        tick = _Tick(store, child_id, token).start()
        child._delegate_children_tick = tick

        def _on_done(done_future: Any) -> None:
            tick.stop()
            try:
                store.finish_run(child_id, token, _done_reason(done_future))
            except Exception:
                logger.warning("delegate_children: could not mark child %s idle", child_id, exc_info=True)

        future.add_done_callback(_on_done)
        return token
    except Exception:
        logger.warning("delegate_children: could not record a kept child's run", exc_info=True)
        return None


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
    *delegation_id* (on the ledger's own connection) and return ``ORPHANED_LINE`` when any of them is now
    idle-orphaned. Units of a grouped call carry ``<id>-<n>``; their children record the call's id."""
    if not delegation_id or not _table_exists(conn):
        return None
    base = delegation_id.rsplit("-", 1)[0] if delegation_id.rsplit("-", 1)[-1].isdigit() else delegation_id
    _apply_rules(conn, time.time(), _delegation_cfg())
    rows = conn.execute(f"SELECT task_index, status, idle_reason FROM {TABLE} WHERE delegation_id IN (?, ?)",
                        (delegation_id, base)).fetchall()
    wanted = set(task_indexes) if task_indexes else None
    hit = any(r[1] == IDLE and r[2] == ORPHANED and (wanted is None or r[0] is None or r[0] in wanted) for r in rows)
    return ORPHANED_LINE if hit else None


# ── session pruning ─────────────────────────────────────────────────────────

def kept_sessions_sql(continuation_edge_sql: str) -> str:
    """Session ids a kept child owns: its first session, its recorded latest segment, and every compression
    continuation after them (``continuation_edge_sql`` relates child ``c`` to parent ``p``)."""
    return ("WITH RECURSIVE kept_child(id) AS ("
            f" SELECT d.child_id FROM {TABLE} d WHERE d.kept = 1 AND d.status IN ('{RUNNING}', '{IDLE}')"
            " UNION"
            f" SELECT d.latest_session_id FROM {TABLE} d WHERE d.kept = 1 AND d.status IN ('{RUNNING}', '{IDLE}')"
            " AND d.latest_session_id IS NOT NULL"
            " UNION"
            " SELECT c.id FROM kept_child k JOIN sessions p ON p.id = k.id JOIN sessions c ON c.parent_session_id = p.id"
            f" WHERE {continuation_edge_sql}"
            ") SELECT id FROM kept_child")
