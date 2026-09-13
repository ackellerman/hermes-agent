"""SPEC-0042 dump artifacts: verbatim durable copies of regions before any live drop.

A dump is an append-only JSONL artifact (same per-line serialization as the session
transcript) plus a sidecar meta. The write protocol is: serialize -> fsync -> set
``complete: true`` in meta via atomic rename. A dump without ``complete: true`` is
never referenced by a swap; every consumer must treat an incomplete dump as absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DUMP_SCHEMA_VERSION = 1
DEFAULT_STORAGE_ROOT = Path("/tmp/hermes-compaction")


class DumpIncompleteError(RuntimeError):
    """Raised when a swap/recovery path tries to consume an incomplete dump."""

    def __init__(self, dump_id: str, session_id: str):
        super().__init__(
            f"dump {dump_id!r} (session {session_id!r}) is not complete: refusing to use it"
        )
        self.dump_id = dump_id
        self.session_id = session_id


class DumpNotFoundError(LookupError):
    """Machine-readable unknown-dump error (AC-16): never silent emptiness."""

    def __init__(self, dump_id: str, session_id: Optional[str] = None):
        super().__init__(json.dumps({
            "error": "dump_not_found",
            "dump_id": dump_id,
            "session_id": session_id,
        }))
        self.dump_id = dump_id
        self.session_id = session_id


def serialize_message(msg: Any) -> str:
    """One JSONL line, byte-identical to the transcript's serialization
    (``hermes_state.divert_session_transcript_jsonl``)."""
    record = msg if isinstance(msg, dict) else {"content": str(msg)}
    return json.dumps(record, ensure_ascii=False, default=str)


def region_hash8(messages: List[Any]) -> str:
    """Stable 8-hex-char hash of a region's serialized bytes (idempotent marking, AC-2)."""
    digest = hashlib.sha256()
    for msg in messages:
        digest.update(serialize_message(msg).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:8]


@dataclass(frozen=True)
class DumpRef:
    """Address of a dumped region: every citation resolves to one of these."""

    session_id: str
    dump_id: str
    start_msg: int
    end_msg: int
    storage: str  # "inline" | "transcript-ref"

    @property
    def citation(self) -> str:
        return f"dump:{self.dump_id}#{self.start_msg}-{self.end_msg}"


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DumpStore:
    """Append-only dump store under a config-definable root.

    Layout: ``<root>/<session_id>/<dump_id>.jsonl`` with sidecar
    ``<dump_id>.meta.json``. All writes are durability-ordered: content is fsynced
    BEFORE the meta flips ``complete``, and the meta itself lands via atomic rename.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else DEFAULT_STORAGE_ROOT

    # ── paths ──────────────────────────────────────────────────────────

    def session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    def dump_path(self, session_id: str, dump_id: str) -> Path:
        return self.session_dir(session_id) / f"{dump_id}.jsonl"

    def meta_path(self, session_id: str, dump_id: str) -> Path:
        return self.session_dir(session_id) / f"{dump_id}.meta.json"

    # ── write protocol ─────────────────────────────────────────────────

    def write_dump(
        self,
        session_id: str,
        messages: List[Any],
        *,
        start_msg: int,
        end_msg: int,
        turn: int = 0,
        storage: str = "inline",
    ) -> DumpRef:
        """Serialize ``messages``, fsync, then flip ``complete: true`` atomically.

        Raises ``OSError`` on fsync/write failure — the caller must NOT reference
        the dump when this raises; the meta stays ``complete: false`` if it landed.
        """
        if end_msg < start_msg:
            raise ValueError(f"end_msg ({end_msg}) < start_msg ({start_msg})")
        sid = str(session_id)
        dump_id = f"{turn:04d}-{region_hash8(messages)}"
        sdir = self.session_dir(sid)
        sdir.mkdir(parents=True, exist_ok=True)
        dpath = self.dump_path(sid, dump_id)
        mpath = self.meta_path(sid, dump_id)

        meta: Dict[str, Any] = {
            "session_id": sid,
            "dump_id": dump_id,
            "start_msg": int(start_msg),
            "end_msg": int(end_msg),
            "message_count": len(messages),
            "token_estimate": self.estimate_tokens(messages),
            "created": time.time(),
            "schema_version": DUMP_SCHEMA_VERSION,
            "storage": storage,
            "complete": False,
        }

        # 1. write meta (incomplete) first, so a crash mid-write leaves a visible
        #    tombstone rather than orphaned content with no meta.
        self._atomic_write_json(mpath, meta)
        # 2. serialize content, flush, fsync.
        payload = "".join(serialize_message(m) + "\n" for m in messages)
        with dpath.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # 3. fsync the directory so the content file entry itself is durable.
        _fsync_dir(sdir)
        # 4. only now flip complete:true via atomic rename.
        meta["complete"] = True
        self._atomic_write_json(mpath, meta)
        return DumpRef(
            session_id=sid,
            dump_id=dump_id,
            start_msg=int(start_msg),
            end_msg=int(end_msg),
            storage=storage,
        )

    @staticmethod
    def estimate_tokens(messages: List[Any]) -> int:
        try:
            from agent.model_metadata import estimate_messages_tokens_rough

            return int(estimate_messages_tokens_rough(list(messages)))
        except Exception:  # noqa: BLE001 — estimate is advisory; never block a dump
            return sum(len(serialize_message(m)) for m in messages) // 4

    def _atomic_write_json(self, path: Path, obj: Dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    # ── read protocol ──────────────────────────────────────────────────

    def read_meta(self, session_id: str, dump_id: str) -> Optional[Dict[str, Any]]:
        mpath = self.meta_path(session_id, dump_id)
        if not mpath.is_file():
            return None
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def is_complete(self, session_id: str, dump_id: str) -> bool:
        meta = self.read_meta(session_id, dump_id)
        return bool(meta and meta.get("complete") is True)

    def require_complete(self, session_id: str, dump_id: str) -> None:
        """Refuse (raise) unless the dump is durably complete — the swap gate."""
        meta = self.read_meta(session_id, dump_id)
        if meta is None:
            raise DumpNotFoundError(dump_id, session_id)
        if meta.get("complete") is not True:
            raise DumpIncompleteError(dump_id, session_id)

    def read_messages(
        self, session_id: str, dump_id: str, *, start_msg: Optional[int] = None,
        end_msg: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Verbatim read of a dump range. Incomplete/unknown dumps raise — never
        return silent emptiness."""
        self.require_complete(session_id, dump_id)
        meta = self.read_meta(session_id, dump_id) or {}
        dpath = self.dump_path(session_id, dump_id)
        lo = meta.get("start_msg", 0)
        hi = meta.get("end_msg", lo)
        start = lo if start_msg is None else max(int(start_msg), lo)
        end = hi if end_msg is None else min(int(end_msg), hi)
        out: List[Dict[str, Any]] = []
        with dpath.open("r", encoding="utf-8") as handle:
            idx = lo
            for line in handle:
                if idx > end:
                    break
                if idx >= start:
                    out.append(json.loads(line))
                idx += 1
        return out

    def round_trip_bytes(self, session_id: str, dump_id: str) -> bytes:
        """Raw serialized bytes of the whole dump — the AC-1 byte-identity check."""
        self.require_complete(session_id, dump_id)
        return self.dump_path(session_id, dump_id).read_bytes()