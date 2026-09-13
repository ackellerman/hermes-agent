"""SPEC-0042 rehydration model tool: ``read_dump`` (service-gated, own toolset).

Footprint-ladder rung 3: structured params/returns, appears only when the
compaction pipeline is enabled and a dump store is configured (check_fn).
Reads dump ranges verbatim; the tool result enters as a tool message at the
turn tail — no history mutation, no cache break beyond the normal append.
"""

from __future__ import annotations

import json
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


def read_dump(dump_id: str, start_msg: Optional[int] = None,
              end_msg: Optional[int] = None, task_id: Optional[str] = None) -> str:
    from agent.compaction_dump import DumpNotFoundError, DumpStore
    from agent.compaction_rehydrate import Rehydrator

    if not dump_id or not isinstance(dump_id, str):
        return json.dumps({"error": "invalid_dump_id", "dump_id": dump_id})
    try:
        store = DumpStore()
        rh = Rehydrator(store)
        result = rh.read_dump(dump_id, start_msg=start_msg, end_msg=end_msg)
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