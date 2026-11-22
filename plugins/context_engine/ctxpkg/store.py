"""The ``ctxpkg`` episode store: one file per episode, an index, and the warmth ledger.

Layout, under ``<HERMES_HOME>/ctxpkg/<compression-lineage root>/`` (mode 0700):

* ``ep-<NNNN>.md`` — the rendered episode, mode 0600, written atomically (temp file then ``os.replace``), so a
  crash mid-write leaves the previous file intact;
* ``index.jsonl`` — one JSON object per episode: ``ref, created_at, turn_first, turn_last, chars, ids, summary``;
* ``warm.json`` — the turn counter and each ref's last-use turn; ``recall`` and a mention in the assistant text of
  a finished turn renew it.

The lineage root is the same walk ``agent/prompt_cache_scope.py`` uses, so a compaction that rotates the physical
session id keeps writing to one store. It is that walk only: a ``--source tool`` rotation root
(``prompt_cache_scope.conversation_id`` follows it) is not followed here, so a tool-source gateway that rotates
sessions forks the store at each rotation. Nothing writes a store across processes, so one in-process lock
serialises the ``warm.json`` read-modify-write between ``recall``'s tool thread and ``on_turn_complete``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

# A ref is exactly ep-<four ASCII digits>; ``[0-9]`` because ``\d`` also matches non-ASCII digits.
RECALL_REF_RE = re.compile(r"ep-[0-9]{4}")
# A session id becomes one path component under ``<home>/ctxpkg/``: only these characters, and never "." or "..".
_SAFE_SESSION_RE = re.compile(r"[A-Za-z0-9_.-]+")
_UNSAFE_SESSION_FALLBACK = "nosession"
MAX_PART_CHARS = 24_000
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_INDEX_NAME = "index.jsonl"
_WARM_NAME = "warm.json"
# One engine per process writes a store; this lock keeps a recall (tool thread) and a finished turn from losing
# each other's ``warm.json`` update. It is not a cross-process lock.
_WARM_LOCK = threading.Lock()


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* through a temp file in the same directory and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temp_name = ""
    try:
        descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        handle = os.fdopen(descriptor, "wb")
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.chmod(temp_name, _FILE_MODE)
        os.replace(temp_name, path)
        temp_name = ""
    except BaseException:
        if handle is not None:
            with suppress(OSError):
                handle.close()
        if temp_name:
            with suppress(OSError):
                os.unlink(temp_name)
        raise


def split_parts(text: str, limit: int = MAX_PART_CHARS) -> list[str]:
    """Split *text* into <= *limit*-char parts on line boundaries (a single longer line is hard-split)."""
    if limit <= 0:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                parts.append("".join(current))
                current, size = [], 0
            parts.append(line[:limit])
            line = line[limit:]
        if current and size + len(line) > limit:
            parts.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        parts.append("".join(current))
    return parts or [""]


class EpisodeStore:
    """The on-disk episode store for one compression lineage."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def for_session(cls, session_id: str | None, session_db: Any = None) -> EpisodeStore:
        """The store for *session_id*'s compression lineage (its root, or the id itself with no lineage)."""
        from agent.prompt_cache_scope import _lineage_root

        physical = str(session_id or "")
        root_id = _lineage_root(physical, session_db) or physical or ""
        if not _SAFE_SESSION_RE.fullmatch(root_id) or root_id in (".", ".."):
            # An id becomes one path component: anything else (a separator, "..", an empty id) would resolve
            # outside ``<home>/ctxpkg/``, so it lands on one fixed safe name instead.
            root_id = _UNSAFE_SESSION_FALLBACK
        return cls(Path(get_hermes_home()) / "ctxpkg" / root_id)

    # ── layout ──────────────────────────────────────────────────────────────

    def ensure(self) -> None:
        """Create ``<home>/ctxpkg`` and this store's directory, both 0700 (the parent lists the session ids)."""
        base = self.root.parent
        base.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.chmod(base, _DIR_MODE)
        self.root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        with suppress(OSError):
            os.chmod(self.root, _DIR_MODE)

    def _episode_path(self, ref: str) -> Path:
        return self.root / f"{ref}.md"

    def refs(self) -> list[str]:
        """Every stored ref, ascending."""
        try:
            names = [entry.name for entry in self.root.iterdir()]
        except OSError:
            return []
        found = [name[:-3] for name in names
                 if name.endswith(".md") and RECALL_REF_RE.fullmatch(name[:-3])]
        return sorted(found)

    def next_ref(self) -> str:
        """``ep-<highest existing number + 1>``, zero-padded to four digits."""
        highest = 0
        for ref in self.refs():
            with suppress(ValueError):
                highest = max(highest, int(ref.rsplit("-", 1)[1]))
        return f"ep-{highest + 1:04d}"

    def exists(self, ref: str) -> bool:
        """True when *ref* names a regular episode file here (a symlink planted in the dir is not one)."""
        if not RECALL_REF_RE.fullmatch(ref or ""):
            return False
        path = self._episode_path(ref)
        return not path.is_symlink() and path.is_file()

    def nearest(self, ref: str, limit: int = 10) -> list[str]:
        """Up to *limit* stored refs nearest *ref* by number, nearest first."""
        if not RECALL_REF_RE.fullmatch(ref or ""):
            return []
        target = int(ref.rsplit("-", 1)[1])
        known = self.refs()
        known.sort(key=lambda candidate: (abs(int(candidate.rsplit("-", 1)[1]) - target),
                                          int(candidate.rsplit("-", 1)[1])))
        return known[:limit]

    # ── episodes ────────────────────────────────────────────────────────────

    def write_episode(self, text: str, *, turn_first: int, turn_last: int, ids: Iterable[str] = (),
                      summary: str = "", created_at: float | None = None) -> str:
        """Write one episode (atomically) and append its index row; returns its ref."""
        self.ensure()
        ref = self.next_ref()
        atomic_write_text(self._episode_path(ref), text)
        row = {
            "ref": ref,
            "created_at": time.time() if created_at is None else float(created_at),
            "turn_first": int(turn_first),
            "turn_last": int(turn_last),
            "chars": len(text),
            "ids": list(ids),
            "summary": summary,
        }
        self._append_index(row)
        return ref

    def _append_index(self, row: dict[str, Any]) -> None:
        line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(str(self.root / _INDEX_NAME),
                             os.O_CREAT | os.O_WRONLY | os.O_APPEND, _FILE_MODE)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)

    def index(self) -> list[dict[str, Any]]:
        """Every index row, in write order (a malformed line is skipped, not fatal)."""
        rows: list[dict[str, Any]] = []
        try:
            text = (self.root / _INDEX_NAME).read_text(encoding="utf-8")
        except OSError:
            return rows
        for line in text.splitlines():
            with suppress(ValueError):
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def next_turn(self) -> int:
        """The turn number to continue from: the last indexed episode's ``turn_last`` + 1 (1 when empty).

        Survives a restart and a lineage rotation because it comes from the store, not from a per-span count.
        """
        for row in reversed(self.index()):
            turn_last = row.get("turn_last")
            if isinstance(turn_last, int) and not isinstance(turn_last, bool):
                return int(turn_last) + 1
        return 1

    def read(self, ref: str) -> str | None:
        """An episode's text, or None when the ref is unknown (a bad ref never builds a path)."""
        if not RECALL_REF_RE.fullmatch(ref or ""):
            return None
        try:
            # O_NOFOLLOW: a symlink planted in the store dir is refused, never followed out of it.
            descriptor = os.open(str(self._episode_path(ref)), os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            return None
        with os.fdopen(descriptor, "rb") as handle:
            return handle.read().decode("utf-8")

    # ── warmth ──────────────────────────────────────────────────────────────

    def _load_warm(self) -> dict[str, Any]:
        try:
            raw = json.loads((self.root / _WARM_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        state = raw if isinstance(raw, dict) else {}
        last_use = state.get("last_use")
        turn = state.get("turn")
        return {
            "turn": int(turn) if isinstance(turn, int) else 0,
            "last_use": {ref: int(value) for ref, value in (last_use or {}).items()
                         if isinstance(value, int) and RECALL_REF_RE.fullmatch(str(ref))}
            if isinstance(last_use, dict) else {},
        }

    def _save_warm(self, state: dict[str, Any]) -> None:
        self.ensure()
        atomic_write_text(self.root / _WARM_NAME, json.dumps(state, sort_keys=True) + "\n")

    def current_turn(self) -> int:
        return self._load_warm()["turn"]

    def bump_turn(self) -> int:
        """Advance the persisted turn counter (the engine's clock for warmth) and return it."""
        with _WARM_LOCK:
            state = self._load_warm()
            state["turn"] += 1
            self._save_warm(state)
            return state["turn"]

    def last_use(self, ref: str) -> int | None:
        return self._load_warm()["last_use"].get(ref)

    def renew(self, ref: str, turn: int | None = None) -> None:
        """Mark *ref* used now (or at *turn*): its warmth window restarts from there."""
        if not RECALL_REF_RE.fullmatch(ref or ""):
            return
        with _WARM_LOCK:
            state = self._load_warm()
            state["last_use"][ref] = state["turn"] if turn is None else int(turn)
            self._save_warm(state)
