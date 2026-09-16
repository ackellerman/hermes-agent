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
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.compaction_dump import DumpStore
from agent.compaction_verify import check_alternation_invariant

CHECKPOINT_MARKER = "compaction_checkpoint"

logger = logging.getLogger(__name__)


class SwapRefusedError(RuntimeError):
    """The swap refused (incomplete dump / gate not eligible / alternation
    invalid). Never a silent no-op: carries the reason."""


def pipeline_pass_blocked(db, session_id: str) -> bool:
    """A pipeline pass must wait or degrade while the compression lease is live."""
    if db is None:
        return False  # no session DB -> no compression lease to contend with
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
    if db is None:
        return False  # no session DB -> no pipeline lock to contend with
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
    """The one swapped-in row: narrative + section index + dump ref, all as
    DATA on this single assistant message.

    SPEC-0049 D5: the row carries NO per-item coordinate table — the agent
    enumerates compacted regions via the ``list_regions`` tool (backed by the
    persisted stub registry) and pulls verbatim content via ``read_dump``.
    The stubs argument is accepted for back-compat with existing callers but
    only a bounded count is emitted; coordinates live in the store, not the
    context (a coordinate table in-context forces the agent to re-read its
    whole context to find them — the operator's ruling).
    """
    parts: List[str] = [f"[{CHECKPOINT_MARKER}] {checkpoint.get('narrative', '')}"]
    sections = {k: checkpoint.get(k) for k in (
        "instructions_and_corrections", "decisions", "insights", "commitments",
        "open_threads", "artifacts", "world_effects", "links")}
    parts.append("Sections: " + json.dumps(sections, ensure_ascii=False, default=str))
    # D5 (operator ruling): the row carries NO storage coordinates and NO
    # dump refs — the agent must not even know where regions live, so it
    # cannot go looking; discovery is list_regions, retrieval is read_dump.
    parts.append(
        "Earlier parts of this conversation were compacted. Compacted "
        "regions are catalogued behind the compaction tools: when the work "
        "touches a topic from before this point, call list_regions to "
        "enumerate the regions (one-liners + refs), pick the matching ref, "
        "and call read_dump with it to get the region's verbatim messages.")
    if stubs:
        # D5 back-compat bound: a caller-built stub list is summarized by
        # count, never inlined — the context stays coordinate-free.
        parts.append(f"({len(stubs)} cited items are in the catalogue)")
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


# ── batched swap sweep (D4/D5, AC-25/26) ───────────────────────────────


def swap_sweep(
    messages: List[Dict[str, Any]],
    *,
    ready_regions: List[Dict[str, Any]],
    dump_store: DumpStore,
    session_id: str,
    stub_registry=None,
    current_window: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Swap ALL current-window-matching, complete, gate-passed regions in ONE
    message-list mutation (AC-13 preserved: ≤1 prefix mutation per turn).

    ``ready_regions``: list of dicts from :meth:`CompactionBackstop`-style
    discovery, each with ``dump_id``, ``meta`` (start_msg/end_msg), ``checkpoint``,
    ``gate``. Regions are processed from the HIGHEST ``start_idx`` to the LOWEST
    so index math stays valid as earlier regions are removed. Each swapped-in
    checkpoint registers its link-stubs into ``stub_registry`` (AC-28) and the
    map entries are retired. Alternation is re-verified on the WHOLE output
    before it is committed.

    Region with a dump window that does not match its meta positions is skipped
    (stale-window discipline); a stale-window region must never swap.
    """
    from agent.compaction_map import CompactionMap

    # Verify + stage each region, highest start first so indices stay valid.
    ordered = sorted(ready_regions, key=lambda r: -int(r["meta"].get("start_msg", 0)))
    out = list(messages)
    registry = stub_registry or _NullStubRegistry()
    # ``CompactionMap`` joins the session id itself (``root/<sid>/map.json``), so
    # the root handed to it must be the STORE root, never the session dir — a
    # pre-joined session id here would retire the map at ``<root>/<sid>/<sid>/``
    # (real map never retires) and mkdir a meta-less phantom directory that every
    # consumer's dump discovery then treats as a region (S1).
    map_root = dump_store.root
    for ready in ordered:
        meta = ready["meta"]
        start = int(meta.get("start_msg", 0))
        end = int(meta.get("end_msg", start))
        # Stale-window discipline: a dump whose window does not match the current
        # compression window must NEVER swap (an earlier-pass artifact must not
        # be applied to a different later window).
        if current_window is not None:
            dw = (start, end)
            if dw != (int(current_window[0]), int(current_window[1])):
                logger.warning("swap_sweep: dump %s window %s != current %s; stays live",
                               ready["dump_id"], dw, current_window)
                continue
        if end >= len(out):
            logger.warning("swap_sweep: dump range %d..%d exceeds message count %d; defer",
                           start, end, len(out))
            continue
        if end < start:
            logger.warning("swap_sweep: dump %s has inverted range %d..%d; defer",
                           ready["dump_id"], start, end)
            continue
        gate = ready.get("gate")
        if gate is not None and gate.get("swap_eligible") is not True:
            continue
        try:
            out = swap_region(
                out, start_idx=start, end_idx=end,
                checkpoint=ready["checkpoint"], dump_store=dump_store,
                session_id=session_id, dump_id=ready["dump_id"], gate_verdict=gate,
            )
        except SwapRefusedError as exc:
            logger.warning("swap_sweep: refuse %s (%s); region stays live",
                           ready["dump_id"], exc)
            continue
        except Exception as exc:  # noqa: BLE001 — one bad region must not wedge the sweep
            logger.warning("swap_sweep: %s failed (%s); region stays live",
                           ready["dump_id"], exc)
            continue
        _register_stubs(registry, ready, ready["checkpoint"])
        # Retire the moved window from the moving map (map stays O(live)).
        try:
            CompactionMap(map_root, session_id).retire(start, end)
        except Exception as exc:  # noqa: BLE001
            logger.warning("swap_sweep: map retire deferred (%s)", exc)
    return out


class _NullStubRegistry:
    """No-op registry so callers that don't persist stubs (unit tests) need not
    build one."""

    def register(self, *args, **kwargs):  # noqa: ANN002
        return None


def _register_stubs(registry, ready: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
    """Register each swapped checkpoint's REGION into the registry.

    SPEC-0049 D5: the registry is keyed by dump_id, so a region gets ONE
    entry (the pre-D5 per-item loop overwrote itself — later items clobbered
    earlier ones). The one-liner merges the region's identity: the
    kept_substance refs (the substance catalogue) plus the first cited
    section item, bounded to keep the catalogue row short.
    """
    dump_id = ready["dump_id"]
    meta = ready.get("meta") or {}
    start = int(meta.get("start_msg", 0))
    end = int(meta.get("end_msg", 0))
    liners: List[str] = []
    for entry in checkpoint.get("kept_substance") or []:
        if isinstance(entry, dict) and entry.get("ref"):
            liners.append(f"{entry['ref']}: {str(entry.get('substance', ''))[:120]}")
    for section in ("instructions_and_corrections", "decisions", "commitments",
                    "open_threads", "artifacts", "world_effects", "links", "insights"):
        for item in checkpoint.get(section) or []:
            if isinstance(item, dict) and item.get("cites"):
                what = str(item.get("what", item.get("rationale", "item")))[:120]
                liners.append(f"{section}: {what}")
                break  # one representative per section keeps the row bounded
        if len(liners) >= 8:
            break
    one_liner = " | ".join(liners) if liners else "compacted region"
    registry.register(dump_id, start, end, one_liner)