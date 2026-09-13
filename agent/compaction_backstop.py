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
import time
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

    def _current_compression_window(self) -> Optional[Tuple[int, int]]:
        """Return the window calculated for *this* overflow dispatch.

        ``conversation_compression`` stamps this immediately before entering the
        backstop. A previous compression's window is never safe to reuse: dump
        ranges are message indices, so a stale ready dump could otherwise replace
        an unrelated part of a later transcript.
        """
        compressor = getattr(self.agent, "context_compressor", None)
        window = getattr(compressor, "last_compress_window", None)
        if not isinstance(window, tuple) or len(window) != 2:
            return None
        try:
            start, end = int(window[0]), int(window[1])
        except (TypeError, ValueError):
            return None
        return (start, end) if 0 <= start < end else None

    def _extraction_state(self, expected_window: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """Whether a ready, gate-passed checkpoint exists for this dispatch's
        exact compression window. None means unknown/not ready and must degrade
        rather than swapping a stale region."""
        ready = self._find_ready_region(expected_window)
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

    def _find_ready_region(self, expected_window: Optional[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """Find a gate-passed complete dump for ``expected_window`` only.

        Dump ranges are transcript indices, not durable identities. Requiring an
        exact current-window match prevents a ready artifact from an earlier
        compression pass being applied to a different later window.
        """
        if expected_window is None:
            return None
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
            meta = store.read_meta(self._session_id(), dump_id) or {}
            try:
                dump_window = (int(meta["start_msg"]), int(meta["end_msg"]))
            except (KeyError, TypeError, ValueError):
                continue
            if dump_window != expected_window:
                logger.debug("backstop dump %s is for %s, current window is %s; defer",
                             dump_id, dump_window, expected_window)
                continue
            return {"dump_id": dump_id, "meta": meta, "checkpoint": checkpoint, "gate": gate_verdict}
        return None

    # ── degraded-region queue (D3, AC-24) ────────────────────────────

    def _queue_path(self) -> Path:
        return self._storage_root() / self._session_id() / "pipeline_queue.json"

    def _read_queue(self) -> List[Dict[str, Any]]:
        qp = self._queue_path()
        if not qp.is_file():
            return []
        try:
            data = json.loads(qp.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_queue(self, rows: List[Dict[str, Any]]) -> None:
        qp = self._queue_path()
        qp.parent.mkdir(parents=True, exist_ok=True)
        tmp = qp.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(rows, ensure_ascii=False, indent=2))
            handle.flush()
        import os
        try:
            os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 — fsync best-effort on the queue
            pass
        os.replace(tmp, qp)

    def _append_degraded(self, expected_window: Optional[Tuple[int, int]],
                         reason: str, dump_id: Optional[str] = None) -> None:
        """Persist a degraded-region row (D3): survives restart, drained by the
        idle pass. A window mismatch on drain falls back to a fresh dump."""
        if expected_window is None:
            return
        rows = self._read_queue()
        window = [int(expected_window[0]), int(expected_window[1])]
        # Idempotent: don't stack duplicate rows for the same (window, reason, dump).
        if any(r.get("reason") == reason and r.get("window") == window
               and r.get("dump_id") == dump_id for r in rows):
            return
        rows.append({
            "dump_id": dump_id, "window": window, "reason": reason,
            "queued_ts": time.time(),
        })
        self._write_queue(rows)

    def _dump_window(self, messages: List[Dict[str, Any]],
                     expected_window: Optional[Tuple[int, int]]) -> Optional[str]:
        """Write a complete dump of ``messages[expected_window]`` (AC-20/21);
        return the dump id or None on any failure. Idempotent per (window, hash):
        a second call for the same window does not create a new dump file."""
        if expected_window is None:
            return None
        start, end = int(expected_window[0]), int(expected_window[1])
        if end >= len(messages) or start > end:
            return None
        try:
            from agent.compaction_dump import DumpStore
            store = DumpStore(self._storage_root())
            region = messages[start:end + 1]
            ref = store.write_dump(self._session_id(), region,
                                   start_msg=start, end_msg=end, turn=1)
            return ref.dump_id
        except Exception as exc:  # noqa: BLE001 — never let a dump failure wedge the overflow turn
            logger.warning("backstop dump-before-degrade failed (%s): %s",
                           self._session_id(), exc)
            return None

    # ── swap ──────────────────────────────────────────────────────────

    def _swap_ready_region(
        self, messages: List[Dict[str, Any]], expected_window: Optional[Tuple[int, int]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Swap only a ready dump for the current dispatch window; else None."""
        ready = self._find_ready_region(expected_window)
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
        expected_window = self._current_compression_window()
        from agent.compaction_swap import backstop_gate
        decision = backstop_gate(
            self._cfg(), getattr(self.agent, "db", None), self._session_id(),
            extraction_state=self._extraction_state(expected_window),
            models_reachable=self._models_reachable(),
        )
        action = decision["action"]
        if action == "run_pipeline":
            swapped = self._swap_ready_region(messages, expected_window)
            if swapped is not None:
                return ("swap", swapped,
                        {TELEMETRY_DEGRADED: False, TELEMETRY_DEGRADATION_REASON: None})
            # A ready checkpoint vanished between the gate and the swap -> degrade.
            self._dump_window(messages, expected_window)
            self._append_degraded(expected_window, "extraction_incomplete")
            return ("degrade", None,
                    {TELEMETRY_DEGRADED: True, TELEMETRY_DEGRADATION_REASON: "extraction_incomplete"})
        if action == "degrade":
            # AC-21: dump-before-degrade — write the current window durably and
            # enqueue it (AC-24) BEFORE running the legacy summary, so nothing is
            # dropped without a durable copy even on the fallback path (D2). The
            # message-list output stays byte-identical to the OFF case (AC-19b);
            # the dump is a disk-side artifact, never part of the message list.
            reason = decision.get("degradation_reason")
            dump_id = self._dump_window(messages, expected_window)
            self._append_degraded(expected_window, reason or "model_unreachable",
                                  dump_id=dump_id)
            return ("degrade", None,
                    {TELEMETRY_DEGRADED: True,
                     TELEMETRY_DEGRADATION_REASON: reason})
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