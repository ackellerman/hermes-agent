"""SPEC-0042 rehydration model tool: ``read_dump`` (service-gated, own toolset).

Footprint-ladder rung 3: structured params/returns, appears only when the
compaction pipeline is enabled and a dump store is configured (check_fn).
Reads dump ranges verbatim; the tool result enters as a tool message at the
turn tail — no history mutation, no cache break beyond the normal append.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.registry import registry


def _pipeline_enabled() -> bool:
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("compaction_pipeline", {}).get("enabled", False))
    except Exception:  # noqa: BLE001 — check_fn must never crash discovery
        return False


def check_requirements() -> bool:
    return _pipeline_enabled()


def _storage_root() -> str:
    """Resolve ``storage_root`` from the SAME config path ``enabled`` reads
    (``compaction_pipeline.storage_root``); never a hardcoded path (D6b)."""
    try:
        from hermes_cli.config import load_config
        return str((load_config() or {}).get("compaction_pipeline", {}).get(
            "storage_root", "/tmp/hermes-compaction"))
    except Exception:  # noqa: BLE001 — check_fn must never crash discovery
        return "/tmp/hermes-compaction"


def _session_transcript_reader():
    """Bound reader for the real SessionDB transcript (D6c). Returns a callable
    ``(session_id, start, end) -> List[dict]`` slicing by message index, or None
    when the session DB surface is unavailable at call time. The writer seam is
    ``divert_session_transcript_jsonl``; the primary read is
    ``SessionDB.get_messages`` (hermes_state_messages.py:628).

    The read binds the shared per-path SessionDB via
    ``hermes_state_registry.acquire`` (the same pattern `tools/delegate_tool.py`
    uses to open a child's transcript handle) and releases it in a finally —
    never a bare ``SessionDB(session_id)`` (the first positional is ``db_path``,
    not the session id). ``acquire()`` no-arg resolves the live state.db path at
    call time, honoring a runtime HERMES_HOME redirect (test isolation)."""
    def reader(session_id, start, end):
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            msgs = db.get_messages(session_id, include_inactive=False) or []
        finally:
            release_or_close(db)
        lo = int(start or 0)
        hi = len(msgs) - 1 if end is None else int(end)
        return [m for m in msgs[lo:hi + 1]
                if isinstance(m, dict) and "role" in m]
    return reader


def read_dump(dump_id: str, start_msg: Optional[int] = None,
              end_msg: Optional[int] = None, task_id: Optional[str] = None) -> str:
    from agent.compaction_dump import DumpNotFoundError, DumpStore
    from agent.compaction_rehydrate import Rehydrator, StubRegistry

    if not dump_id or not isinstance(dump_id, str):
        return json.dumps({"error": "invalid_dump_id", "dump_id": dump_id})
    try:
        root = _storage_root()
        store = DumpStore(Path(root))
        rh = Rehydrator(
            store,
            transcript_reader=_session_transcript_reader(),
            stub_registry=StubRegistry.for_session(Path(root), task_id or "none"),
        )
        result = rh.read_dump(dump_id, session_id=task_id,
                              start_msg=start_msg, end_msg=end_msg)
        return json.dumps({"success": True, **result}, ensure_ascii=False, default=str)
    except DumpNotFoundError as exc:
        return json.loads(str(exc))
    except Exception as exc:  # noqa: BLE001 — structured error, never silent
        return json.dumps({"error": "read_dump_failed", "detail": str(exc)})


registry.register(
    name="read_dump",
    toolset="compaction",
    schema={
        "name": "read_dump",
        "description": (
            "Read a verbatim message range from a compaction dump artifact. "
            "Use when a checkpoint link-stub references work the conversation "
            "has returned to. Returns the original messages plus a `source` "
            "field (dump | transcript-fallback)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dump_id": {"type": "string",
                            "description": "The dump id cited by the stub, e.g. the `<turn>-<hash8>` in `[dump: ...]`."},
                "start_msg": {"type": "integer",
                              "description": "Optional first message index to read."},
                "end_msg": {"type": "integer",
                            "description": "Optional last message index to read (inclusive)."},
            },
            "required": ["dump_id"],
        },
    },
    handler=lambda args, **kw: read_dump(
        dump_id=args.get("dump_id", ""),
        start_msg=args.get("start_msg"),
        end_msg=args.get("end_msg"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)