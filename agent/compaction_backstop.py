"""SPEC-0042 backstop/saem wiring: the production seam that gives
``backstop_gate`` and ``swap_region`` real callers on the overflow path
(review finding F4).

The backstop fires on exactly the condition that fires today's single-call
compaction (``ContextCompressor.should_compress_info`` returning ``(True, None)``
— spec §3.5). This module consults ``backstop_gate`` FIRST (before the legacy
summary runs) and, when the pipeline is enabled AND a gate-passed checkpoint +
complete dump exists for the affected region, swaps the region in via
``swap_region`` instead of the prose summary. Otherwise it degrades to the
legacy single-call path — byte-identical to ``enabled: false`` — and records the
degradation in compression telemetry.

AC-19/19b contract: with ``enabled: false`` this runs ZERO pipeline machinery
(no lock, no dump, no map). The degradation telemetry fields are stamped in
BOTH the enabled and disabled paths so the ON-degrade run introduces no new
telemetry attribute versus the OFF run (AC-19b).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Telemetry keys — emitted on the compression path in BOTH enabled states so
# AC-19b's "no new telemetry attribute" holds under byte-diff.
TELEMETRY_DEGRADED = "compaction_degraded"
TELEMETRY_DEGRADATION_REASON = "compaction_degradation_reason"

_DEFAULT_STORAGE_ROOT = "/tmp/hermes-compaction"


class CompactionBackstop:
    """Decision + (optional) swap for the overflow path. Production caller of
    ``agent.compaction_swap.backstop_gate`` and ``agent.compaction_swap.swap_region``.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    # ── config ────────────────────────────────────────────────────────

    def enabled(self) -> bool:
        return bool(getattr(self.agent, "compaction_pipeline_enabled", False))

    def _session_id(self) -> str:
        return str(getattr(self.agent, "session_id", "") or "none")

    def _storage_root(self) -> Path:
        return Path(getattr(self.agent, "compaction_pipeline_storage_root",
                            _DEFAULT_STORAGE_ROOT))

    def _cfg(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "swap": {"max_wait_seconds": float(
                getattr(self.agent, "compaction_pipeline_max_wait_seconds", 900))},
        }

    # ── gate inputs ───────────────────────────────────────────────────

    def _extraction_state(self) -> Optional[Dict[str, Any]]:
        """Whether a ready (gate: passed) checkpoint exists for this session's
        affected region. Returned dict carries ``gate`` for backstop_gate;
        None means 'unknown/not ready' -> backstop degrades as
        extraction_incomplete rather than blocking the overflow turn."""
        ready = self._find_ready_region()
        if ready is None:
            return None
        return {"gate": "passed", "region": ready}

    def _models_reachable(self) -> bool:
        """Best-effort liveness of the extraction model route. An explicit test
        override (``agent._compaction_models_reachable``) wins; otherwise we
        treat the aux route as reachable when the agent carries a resolvable
        runtime/mode, matching how the compressor resolves its summary model."""
        override = getattr(self.agent, "_compaction_models_reachable", None)
        if override is not None:
            return bool(override)
        return bool(
            getattr(self.agent, "aux_runtime", None)
            or (getattr(self.agent, "provider", None) and getattr(self.agent, "model", None))
        )

    # ── data discovery: ready regions (dump complete + gate passed) ────

    def _find_ready_region(self) -> Optional[Dict[str, Any]]:
        """Scan the pipeline storage for a gate-passed checkpoint tied to a
        complete dump. Returns ``{"dump_id", "meta", "checkpoint", "gate"}`` or
        None. Only swap what the pipeline's own gate already cleared — the swap
        re-checks ``require_complete`` and gate eligibility defensively."""
        session_dir = self._storage_root() / self._session_id()
        if not session_dir.is_dir():
            return None
        for child in sorted(session_dir.iterdir()):
            if not child.is_dir():
                continue
            dump_id = child.name
            stage_c = child / "stage_c.json"
            gate = child / "gate.json"
            if not (stage_c.is_file() and gate.is_file()):
                continue
            try:
                checkpoint = json.loads(stage_c.read_text(encoding="utf-8"))
                gate_verdict = json.loads(gate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if gate_verdict.get("swap_eligible") is not True:
                continue
            from agent.compaction_dump import DumpStore
            store = DumpStore(self._storage_root())
            if not store.is_complete(self._session_id(), dump_id):
                continue
            return {"dump_id": dump_id, "checkpoint": checkpoint, "gate": gate_verdict}
        return None

    # ── swap ──────────────────────────────────────────────────────────

    def _swap_ready_region(self, messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """If a ready dump+checkpoint+gate exists, swap its region into a new
        message list via ``swap_region`` and return it; else None (caller falls
        through to the legacy summary)."""
        ready = self._find_ready_region()
        if ready is None:
            return None
        try:
            from agent.compaction_dump import DumpStore
            from agent.compaction_swap import swap_region
            store = DumpStore(self._storage_root())
            meta = store.read_meta(self._session_id(), ready["dump_id"]) or {}
            start = int(meta.get("start_msg", 0))
            end = int(meta.get("end_msg", start))
            if end >= len(messages):
                logger.info("backstop swap: dump range %d..%d exceeds current message count %d; defer",
                            start, end, len(messages))
                return None
            return swap_region(
                messages, start_idx=start, end_idx=end,
                checkpoint=ready["checkpoint"], dump_store=store,
                session_id=self._session_id(), dump_id=ready["dump_id"],
                gate_verdict=ready["gate"])
        except Exception as exc:  # noqa: BLE001 — never let a swap bug wedge the overflow turn
            logger.warning("backstop swap failed (%s); degrading to legacy summary: %s",
                           self._session_id(), exc)
            return None

    # ── the decision (production entry) ───────────────────────────────

    def decide_and_swap(
        self, messages: List[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """Return ``(action, swapped_or_None, telemetry)``.

        single actions:
        - ``legacy_summary``: run the current single-call path unchanged (no
          pipeline mechanism touched).
        - ``degrade``: run the legacy path AND record the degradation reason —
          byte-identical output to ``enabled: false``.
        - ``swap``: the pipeline's own swap replaced the region in ``swapped``.
        """
        if not self.enabled():
            # AC-19: enabled:false short-circuits before ANY pipeline mechanism.
            return ("legacy_summary", None,
                    {TELEMETRY_DEGRADED: False, TELEMETRY_DEGRADATION_REASON: None})
        from agent.compaction_swap import backstop_gate
        decision = backstop_gate(
            self._cfg(), getattr(self.agent, "db", None), self._session_id(),
            extraction_state=self._extraction_state(),
            models_reachable=self._models_reachable(),
        )
        action = decision["action"]
        if action == "run_pipeline":
            swapped = self._swap_ready_region(messages)
            if swapped is not None:
                return ("swap", swapped,
                        {TELEMETRY_DEGRADED: False, TELEMETRY_DEGRADATION_REASON: None})
            # A ready checkpoint vanished between the gate and the swap -> degrade.
            return ("degrade", None,
                    {TELEMETRY_DEGRADED: True, TELEMETRY_DEGRADATION_REASON: "extraction_incomplete"})
        if action == "degrade":
            return ("degrade", None,
                    {TELEMETRY_DEGRADED: True,
                     TELEMETRY_DEGRADATION_REASON: decision.get("degradation_reason")})
        return ("legacy_summary", None,
                {TELEMETRY_DEGRADED: False, TELEMETRY_DEGRADATION_REASON: None})


def maybe_backstop_swap(
    agent: Any, messages: List[Dict[str, Any]],
) -> Tuple[str, Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Module-level entry wired into ``agent.conversation_compression``
    ``_run_summary_dispatch``. Never raises into the live path: any failure
    degrades to the legacy summary with the failure recorded, so an overflow
    turn can never wedge on a pipeline bug."""
    try:
        return CompactionBackstop(agent).decide_and_swap(messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("compaction backstop decision failed (%s); legacy fallback: %s",
                       getattr(agent, "session_id", "none"), exc)
        return ("degrade", None,
                {TELEMETRY_DEGRADED: True, TELEMETRY_DEGRADATION_REASON: "backstop_error"})