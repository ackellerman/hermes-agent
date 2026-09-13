"""SPEC-0042 swap-in: batched, alternation-safe replacement of dumped regions
by a checkpoint row + per-item link-stubs, plus the mutual-exclusion gates the
backstop and pipeline passes must consult (AC-13/14/15).

Swap mechanics (spec §3.5):
- replace region messages with ONE checkpoint row (assistant role, reusing the
  existing summary-row pattern) tagged ``compaction_checkpoint``;
- per-item link-stubs appended ON the checkpoint row's content, never
  standalone messages — stubs cannot violate alternation or tool-pair integrity;
- ordering: dump complete -> extraction -> gate -> swap; the swap refuses
  unless meta says complete AND gate says eligible;
- backstop degrade path: enabled:false short-circuits before ANY pipeline
  mechanism; when enabled, bounded wait then degrade to the legacy single-call
  summary with a telemetry ``degradation_reason``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.compaction_dump import DumpStore
from agent.compaction_verify import check_alternation_invariant

CHECKPOINT_MARKER = "compaction_checkpoint"


class SwapRefusedError(RuntimeError):
    """The swap refused (incomplete dump / gate not eligible / alternation
    invalid). Never a silent no-op: carries the reason."""


def pipeline_pass_blocked(db, session_id: str) -> bool:
    """A pipeline pass must wait or degrade while the compression lease is live."""
    try:
        holder = db.compression_lock_holder(session_id)
    except AttributeError:
        # Not all SessionDB surfaces expose the probe; fall back to a direct read.
        import sqlite3, time
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id = ? AND expires_at > ?",
                (session_id, time.time()),
            ).fetchone()
        return row is not None
    return bool(holder)


def compression_pass_blocked(db, session_id: str) -> bool:
    """A compression pass (manual /compress included) must wait or degrade while
    the pipeline lock is live (AC-14 direction b)."""
    import sqlite3
    import time
    with sqlite3.connect(db.db_path) as conn:
        row = conn.execute(
            "SELECT holder FROM compaction_pipeline_locks "
            "WHERE session_id = ? AND expires_at > ?",
            (session_id, time.time()),
        ).fetchone()
    return row is not None


def build_checkpoint_row(checkpoint: Dict[str, Any], dump_ref, stubs: List[str]) -> Dict[str, Any]:
    """The one swapped-in row: narrative + section index + dump refs + per-item
    link-stubs, all as DATA on this single assistant message."""
    parts: List[str] = [f"[{CHECKPOINT_MARKER}] {checkpoint.get('narrative', '')}"]
    sections = {k: checkpoint.get(k) for k in (
        "instructions_and_corrections", "decisions", "insights", "commitments",
        "open_threads", "artifacts", "world_effects", "links")}
    parts.append("Sections: " + json.dumps(sections, ensure_ascii=False, default=str))
    parts.append(f"Dump ref: dump:{dump_ref.dump_id}#{dump_ref.start_msg}-{dump_ref.end_msg}")
    parts.append(
        "Stubs are references; call read_dump if and only if the work returns to that region.")
    parts.extend(stubs)
    return {"role": "assistant", "content": "\n\n".join(parts)}


def link_stub(dump_id: str, start: int, end: int, one_liner: str) -> str:
    return f"[dump: {dump_id} {start}-{end}] {one_liner}"


def swap_region(
    messages: List[Dict[str, Any]],
    *,
    start_idx: int,
    end_idx: int,
    checkpoint: Dict[str, Any],
    dump_store: DumpStore,
    session_id: str,
    dump_id: str,
    gate_verdict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Replace [start_idx, end_idx] with the checkpoint row + stubs, refusing
    unless dump complete + gate eligible; verifies alternation on its own output
    before committing (AC-12)."""
    try:
        dump_store.require_complete(session_id, dump_id)
    except Exception as exc:  # incomplete/missing dump -> the swap refuses
        raise SwapRefusedError(str(exc)) from exc
    if gate_verdict is not None and gate_verdict.get("swap_eligible") is not True:
        raise SwapRefusedError(f"gate not eligible: {gate_verdict.get('findings')!r}")

    dump_meta = dump_store.read_meta(session_id, dump_id) or {}
    stubs = []
    for section in ("instructions_and_corrections", "decisions", "commitments",
                    "open_threads", "artifacts", "world_effects", "links", "insights"):
        for item in checkpoint.get(section) or []:
            if isinstance(item, dict) and item.get("cites"):
                cite = item["cites"][0]
                stubs.append(link_stub(
                    dump_id, int(cite[1] if len(cite) == 3 else cite[0]),
                    int(cite[2] if len(cite) == 3 else cite[1]),
                    str(item.get("what", item.get("rationale", "item")))))

    ref = type("Ref", (), {
        "dump_id": dump_id,
        "start_msg": dump_meta.get("start_msg", 0),
        "end_msg": dump_meta.get("end_msg", 0),
    })()
    row = build_checkpoint_row(checkpoint, ref, stubs)
    # Summary-row role pattern (context_compressor.py::_summary_role picker):
    # choose the row's role so template-visible alternation holds around it.
    prev_role = messages[start_idx - 1].get("role") if start_idx > 0 else None
    next_role = messages[end_idx + 1].get("role") if end_idx + 1 < len(messages) else None
    candidates = ("assistant", "user")
    role = next(
        (r for r in candidates if r != prev_role and r != next_role),
        "user" if prev_role == "assistant" else "assistant",
    )
    row = dict(row, role=role)

    out = messages[:start_idx] + [row] + messages[end_idx + 1:]
    is_valid, violations = check_alternation_invariant(out)
    if not is_valid:
        raise SwapRefusedError(f"swap output violates alternation invariant: {violations}")
    return out


def backstop_gate(cfg: Dict[str, Any], db, session_id: str,
                  extraction_state: Optional[Dict[str, Any]] = None,
                  models_reachable: bool = True) -> Dict[str, Any]:
    """What the overflow backstop does before anything else (AC-19/19b/15):

    - ``enabled: false`` (default): return the legacy path unconditionally —
      no lock attempt, no pipeline stage, no dump/map/lock writes.
    - enabled: try to acquire the pipeline lock within ``swap.max_wait_seconds``;
      if the lock is held (lock_timeout), extraction for the region hasn't
      reached gate: passed (extraction_incomplete), or a configured extraction
      model is unreachable (model_unreachable) -> degrade. Control flow does not
      distinguish the three; only the telemetry degradation_reason does.
    """
    if not cfg.get("enabled", False):
        return {"action": "legacy_summary", "degraded": False}
    swap_cfg = cfg.get("swap", {}) or {}
    max_wait = float(swap_cfg.get("max_wait_seconds", 900))
    held = pipeline_pass_blocked(db, session_id)  # compression lease live?
    if held:
        return {"action": "degrade", "degraded": True,
                "degradation_reason": "lock_timeout", "max_wait_seconds": max_wait}
    if extraction_state is not None and extraction_state.get("gate") != "passed":
        return {"action": "degrade", "degraded": True,
                "degradation_reason": "extraction_incomplete", "max_wait_seconds": max_wait}
    if not models_reachable:
        return {"action": "degrade", "degraded": True,
                "degradation_reason": "model_unreachable", "max_wait_seconds": max_wait}
    return {"action": "run_pipeline", "degraded": False}