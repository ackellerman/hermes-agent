"""SPEC-0046 pending-swap record: the staged between-turns swap.

The between-turns pipeline pass runs against the FROZEN context packet (the
message list does not change between turns), computes the swapped replacement
locally, and stages it on disk instead of applying it — there is no live list
to mutate between turns. The next turn start re-reads the record and applies
it as that turn's single prefix mutation, ONLY if the packet identity still
matches (AC-A2/A3); otherwise the record is discarded and the region re-runs
against the fresh packet.

Record contents (schema-versioned):
- ``packet_hash``: sha256 over the frozen packet's ROW IDENTITY — for each
  message in order, its durable DB row id (``_row_id``) when present, else the
  byte-serialization of the message (content hash when ids are absent);
- ``packet_identity``: that per-message identity list (the turn-start seam
  matches it against the live packet's identity prefix);
- ``swapped_messages``: the gate-passed replacement list computed locally;
- ``region_refs``: ``{dump_id, start_msg, end_msg}`` per swapped region;
- ``gate_verdict_digest``: short digest binding the record to the gate
  artifacts it was staged from;
- ``created_ts``.

Storage: one file per session, ``<storage_root>/<session_id>/pending_swap.json``.
Writes are atomic (tmp + fsync + rename); reads are atomic read+clear on
application, and a corrupt file is logged and discarded, never raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PENDING_SWAP_SCHEMA_VERSION = 1
PENDING_SWAP_FILENAME = "pending_swap.json"


def pending_swap_path(storage_root, session_id: str):
    from pathlib import Path

    return Path(storage_root) / (session_id or "none") / PENDING_SWAP_FILENAME


def packet_identity(messages) -> list:
    """Per-message row identity for the frozen packet, in order.

    A message with an int ``_row_id`` (durable DB row) is identified by that
    row id — stable across identical content rewrites of the same row. A
    message without one is identified by its byte-serialization (the same
    serialization dumps use), so in-memory packets hash by content.
    """
    from agent.compaction_dump import serialize_message

    ids: list = []
    for msg in messages or []:
        rid = msg.get("_row_id") if isinstance(msg, dict) else None
        if isinstance(rid, int) and not isinstance(rid, bool):
            ids.append(f"row:{rid}")
        else:
            ids.append("msg:" + serialize_message(msg))
    return ids


def compute_packet_hash(messages) -> str:
    """Stable sha256 over the packet's row identity (order-sensitive)."""
    digest = hashlib.sha256()
    for ident in packet_identity(messages):
        digest.update(ident.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def digest_gate_verdicts(verdicts) -> str:
    """Short digest binding a staged swap to the gate verdicts it passed."""
    digest = hashlib.sha256()
    for verdict in verdicts or []:
        digest.update(json.dumps(verdict, ensure_ascii=False, sort_keys=True,
                                 default=str).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def write_pending_swap(storage_root, session_id: str, record: dict) -> None:
    """Atomically persist the pending-swap record (tmp + fsync + rename)."""
    import os
    from pathlib import Path

    path = pending_swap_path(storage_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 — fsync is best-effort on exotic filesystems
            pass
    os.replace(tmp, path)


def read_pending_swap(storage_root, session_id: str) -> Optional[dict]:
    """Read the pending-swap record; ``None`` when absent or unusable.

    A corrupt file is DELETED and logged, never raised: a crash between the
    stage and the reinject must degrade to "no pending swap", never wedge a
    turn start (SPEC-0046 edge case)."""
    path = pending_swap_path(storage_root, session_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("pending swap record is not an object")
        return record
    except Exception as exc:  # noqa: BLE001 — corrupt JSON: discard, never crash
        logger.warning("discarding corrupt pending swap for session %s: %s",
                       session_id, exc)
        try:
            path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return None


def clear_pending_swap(storage_root, session_id: str) -> None:
    """Remove the pending-swap record (idempotent)."""
    try:
        pending_swap_path(storage_root, session_id).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — a failed clear must not break a turn
        logger.warning("failed to clear pending swap for session %s",
                       session_id, exc_info=True)


def build_pending_swap_record(*, packet_hash: str, packet_identity: list,
                              swapped_messages: list, region_refs: list,
                              gate_verdict_digest: str,
                              created_ts: Optional[float] = None) -> dict:
    """Schema-versioned record constructor — the single shape writer."""
    return {
        "schema_version": PENDING_SWAP_SCHEMA_VERSION,
        "packet_hash": packet_hash,
        "packet_identity": list(packet_identity or []),
        "swapped_messages": list(swapped_messages or []),
        "region_refs": list(region_refs or []),
        "gate_verdict_digest": str(gate_verdict_digest or ""),
        "created_ts": float(created_ts if created_ts is not None else time.time()),
    }