"""SPEC-0042 rehydration: stub registry, read_dump resolution with transcript
fallback, recovery after restart.

The checkpoint row instructs the agent: stubs are references; call read_dump if
and only if the work returns to that region. No automatic injection, no
alternation risk. The immutable transcript is the deep archive; the dump is
the working cache: if a dump is missing but the transcript exists, serve from
the transcript slice and record the fallback (AC-18).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.compaction_dump import DumpNotFoundError, DumpStore

logger = logging.getLogger(__name__)


@dataclass
class FallbackRecord:
    """Telemetry for transcript-fallback serves (AC-18: the log records it)."""
    dump_id: str
    session_id: Optional[str]
    served_from: str = "transcript-fallback"


class StubRegistry:
    """Session-level registry of link-stubs emitted by swaps; persisted to
    ``<storage_root>/<session_id>/stubs.json`` so resolution survives restart
    (AC-28) and requires no live message-list parse.

    The registry is a JSON file under the session dir: an id registered at swap
    time resolves to a (start_msg, end_msg, summary) range after process restart,
    powering the registry -> transcript-fallback serve path (AC-28 / D6d).
    ``StubRegistry(path)`` loads an existing file (or starts empty); ``save()``
    writes it atomically. Use ``for_session(root, session_id)`` for the
    conventional path.
    """

    @classmethod
    def for_session(cls, root: Path, session_id: str) -> "StubRegistry":
        return cls(Path(root) / str(session_id) / "stubs.json")

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self._stubs: Dict[str, Dict[str, Any]] = {}
        if self.path is not None and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._stubs = data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                self._stubs = {}

    def register(self, dump_id: str, start_msg: int, end_msg: int, one_liner: str) -> None:
        self._stubs[str(dump_id)] = {
            "start_msg": int(start_msg), "end_msg": int(end_msg), "summary": one_liner,
        }
        self.save()

    def get(self, dump_id: str) -> Optional[Dict[str, Any]]:
        return self._stubs.get(str(dump_id))

    def resolve(self, dump_id: str) -> Optional[Dict[str, Any]]:
        """Alias for :meth:`get` (registry-resolved fallback naming)."""
        return self._stubs.get(str(dump_id))

    def all(self) -> Dict[str, Any]:
        return dict(self._stubs)

    def save(self) -> None:
        import os
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self._stubs, ensure_ascii=False, sort_keys=True))
            handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 — fsync best-effort on registry writes
            pass
        os.replace(tmp, self.path)


class Rehydrator:
    """Resolves read_dump requests: dump store first, stub-registry-resolved
    transcript fallback second, transcript fallback third, machine-readable
    error last (never silent emptiness, AC-16)."""

    def __init__(self, store: DumpStore,
                 transcript_reader: Optional[Any] = None,
                 stub_registry: Optional[StubRegistry] = None):
        self.store = store
        # transcript_reader(session_id, start, end) -> List[dict]; typically a
        # SessionDB bound method; None disables fallback.
        self.transcript_reader = transcript_reader
        self.stub_registry = stub_registry
        self.fallbacks: List[FallbackRecord] = []

    def read_dump(self, dump_id: str, session_id: Optional[str] = None,
                  start_msg: Optional[int] = None,
                  end_msg: Optional[int] = None) -> Dict[str, Any]:
        """Verbatim message range for a valid id; structured error otherwise.

        Resolution order (AC-16/17/18/28):
        1. dump store (configured root, across every session dir if unknown);
        2. stub-registry-resolved (session, range) -> transcript fallback — an
           id registered at swap time resolves even when the dump is gone;
        3. open transcript fallback for the requested session/range;
        4. structured ``DumpNotFoundError`` — never silent emptiness.
        """
        sid = session_id or ""
        # 1. dump store (across every session dir if sid unknown)
        session_dirs = (
            [self.store.root / sid] if sid
            else [p for p in self.store.root.iterdir() if p.is_dir()]
            if self.store.root.is_dir() else []
        )
        for sdir in session_dirs:
            if (sdir / f"{dump_id}.jsonl").is_file():
                return {
                    "source": "dump",
                    "dump_id": dump_id,
                    "session_id": sdir.name,
                    "messages": self.store.read_messages(
                        sdir.name, dump_id, start_msg=start_msg, end_msg=end_msg),
                }
        # 2. registry-resolved fallback (AC-28): an id registered at swap time
        #    identifies (session, range) even when the dump file is deleted.
        if self.stub_registry is not None:
            entry = self.stub_registry.get(dump_id)
            if entry is not None:
                r_session = sid or str(self.store.root) if self.store.root else str(sid)
                served = self._serve_transcript_fallback(
                    dump_id, r_session or "none",
                    int(entry.get("start_msg", start_msg or 0)),
                    int(entry.get("end_msg", entry.get("start_msg", end_msg or 0))),
                )
                if served is not None:
                    return served
        # 3. transcript fallback (AC-18): dump deleted but transcript present.
        if self.transcript_reader is not None and sid:
            try:
                msgs = self.transcript_reader(sid, start_msg or 0, end_msg)
            except Exception as exc:  # noqa: BLE001 — fall through to the error
                logger.debug("transcript fallback failed for %s: %s", dump_id, exc)
                msgs = None
            if msgs:
                record = FallbackRecord(dump_id=dump_id, session_id=sid)
                self.fallbacks.append(record)
                logger.info("read_dump served from transcript-fallback: %s", record)
                return {
                    "source": "transcript-fallback",
                    "dump_id": dump_id,
                    "session_id": sid,
                    "messages": msgs,
                }
        raise DumpNotFoundError(dump_id, sid)

    def _serve_transcript_fallback(self, dump_id: str, session_id: str,
                                   start: int, end: int) -> Optional[Dict[str, Any]]:
        """Serve ``[start, end]`` from the transcript when the registry pins the
        range; record the fallback and log it. None when not servable."""
        if self.transcript_reader is None:
            return None
        try:
            msgs = self.transcript_reader(session_id, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.debug("registry transcript fallback failed for %s: %s", dump_id, exc)
            return None
        if not msgs:
            return None
        record = FallbackRecord(dump_id=dump_id, session_id=session_id)
        self.fallbacks.append(record)
        logger.info("read_dump served via stub-registry transcript-fallback: %s", record)
        return {
            "source": "transcript-fallback",
            "dump_id": dump_id,
            "session_id": session_id,
            "messages": msgs,
        }