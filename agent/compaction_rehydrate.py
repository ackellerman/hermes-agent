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
    """Session-level registry of link-stubs emitted by swaps; powers stub
    resolution without parsing the live message list."""

    def __init__(self):
        self._stubs: Dict[str, Dict[str, Any]] = {}

    def register(self, dump_id: str, start_msg: int, end_msg: int, one_liner: str) -> None:
        self._stubs[dump_id] = {
            "start_msg": start_msg, "end_msg": end_msg, "summary": one_liner,
        }

    def get(self, dump_id: str) -> Optional[Dict[str, Any]]:
        return self._stubs.get(dump_id)

    def all(self) -> Dict[str, Any]:
        return dict(self._stubs)


class Rehydrator:
    """Resolves read_dump requests: dump store first, transcript fallback
    second, machine-readable error last (never silent emptiness, AC-16)."""

    def __init__(self, store: DumpStore,
                 transcript_reader: Optional[Any] = None):
        self.store = store
        # transcript_reader(session_id, start, end) -> List[dict]; typically a
        # SessionDB bound method; None disables fallback.
        self.transcript_reader = transcript_reader
        self.fallbacks: List[FallbackRecord] = []

    def read_dump(self, dump_id: str, session_id: Optional[str] = None,
                  start_msg: Optional[int] = None,
                  end_msg: Optional[int] = None) -> Dict[str, Any]:
        """Verbatim message range for a valid id; structured error otherwise."""
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
        # 2. transcript fallback (AC-18): dump deleted but transcript present.
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