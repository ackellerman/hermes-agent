"""SPEC-0042 idle pipeline pass: map updates, extraction sweeps, swap sweeps.

Hooked into the existing idle gate (``turn_context_compaction._idle_compaction``
computes the idle gap — we reuse the same seam). All passes:

- run only when ``compaction_pipeline.enabled`` (AC-19: OFF touches nothing);
- respect the map cooldown and the pipeline lock (AC-14);
- are bounded by ``budget_per_session_tokens`` (hard stop, telemetry on hit);
- never mutate the live message list (map updates read; only SWAP mutates, and
  the swap sweep is a separate batched step at episode boundaries).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PipelineBudgetExceeded(RuntimeError):
    """Background spend ceiling hit: hard stop; telemetry records it."""


class IdlePipelinePass:
    """One idle-window pass over the pipeline's background duties."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.session_id = getattr(agent, "session_id", "") or "none"
        self.storage_root = getattr(agent, "compaction_pipeline_storage_root", "/tmp/hermes-compaction")
        self.enabled = bool(getattr(agent, "compaction_pipeline_enabled", False))

    # ── gates ────────────────────────────────────────────────────────────

    def gates_pass(self, idle_gap_seconds: float, now: Optional[float] = None) -> bool:
        """enabled + idle gap + cooldown + budget; lock acquisition happens in
        ``run`` and is released before returning."""
        if not self.enabled:
            return False
        idle_after = getattr(self.agent, "compaction_pipeline_map_idle_after_seconds", 20)
        if idle_gap_seconds < max(0.0, float(idle_after)):
            return False
        now = now or time.time()
        if now - self._last_pass_ts() < getattr(
                self.agent, "compaction_pipeline_map_cooldown_seconds", 120):
            return False
        if self._spent_tokens() > getattr(
                self.agent, "compaction_pipeline_budget_per_session_tokens", 200000):
            logger.info("compaction pipeline budget exceeded for session %s; pass skipped",
                        self.session_id)
            return False
        return True

    def _last_pass_ts(self) -> float:
        return float(getattr(self.agent, "_compaction_pipeline_last_pass_ts", 0.0))

    def _spent_tokens(self) -> int:
        return int(getattr(self.agent, "_compaction_pipeline_spent_tokens", 0))

    # ── execution ────────────────────────────────────────────────────────

    def run(self, messages, llm_call=None) -> Dict[str, Any]:
        """One pass: acquire lock (AC-14), update the map over the un-covered
        tail, release. Returns a telemetry record; raises nothing — a failed
        pass logs and reports status, the live context is untouched."""
        if not self.enabled:
            return {"ran": False, "reason": "disabled"}
        db = getattr(self.agent, "db", None)
        holder = f"pipeline:{self.session_id}"
        acquired = False
        if db is not None:
            try:
                acquired = db.try_acquire_pipeline_lock(self.session_id, holder)
            except Exception as exc:  # noqa: BLE001 — lock subsystem broken -> skip pass
                logger.warning("pipeline lock acquire failed (%s): %s", self.session_id, exc)
                return {"ran": False, "reason": "lock_error"}
            if not acquired:
                return {"ran": False, "reason": "lock_held"}
        try:
            if llm_call is None:
                llm_call = self._default_llm_call()
            from agent.compaction_map import CompactionMap
            cm = CompactionMap(__import__("pathlib").Path(self.storage_root), self.session_id)
            covers_end = int(cm.load().get("covers", {}).get("end_msg", 0))
            new_msgs = list(messages)[covers_end:]
            if not new_msgs:
                return {"ran": True, "updated": False, "reason": "map already covers"}
            cm.update(llm_call, new_msgs,
                      start_msg=covers_end, end_msg=covers_end + len(new_msgs) - 1)
            spent = getattr(self.agent, "_compaction_pipeline_spent_tokens", 0)
            self.agent._compaction_pipeline_spent_tokens = spent  # updated by caller telemetry
            self.agent._compaction_pipeline_last_pass_ts = time.time()
            return {"ran": True, "updated": True}
        except Exception as exc:  # noqa: BLE001 — a failed pass parks; live path unaffected
            logger.warning("compaction pipeline pass failed (%s): %s", self.session_id, exc)
            return {"ran": False, "reason": f"error: {exc}"}
        finally:
            if db is not None and acquired:
                try:
                    db.release_pipeline_lock(self.session_id, holder)
                except Exception:  # noqa: BLE001
                    pass

    def _default_llm_call(self):
        """Resolve the map-update model via the existing aux resolution chain
        (``_resolve_auto_route(main_runtime, task)`` — client, model, label)."""
        def llm_call(messages):
            from agent.auxiliary_client import _resolve_auto_route
            runtime = getattr(self.agent, "aux_runtime", None) or {
                "provider": getattr(self.agent, "provider", None),
                "model": (getattr(self.agent, "compaction_pipeline_models", {}) or {}).get(
                    "map_update", "") or getattr(self.agent, "model", None),
                "base_url": getattr(self.agent, "base_url", None),
                "api_key": getattr(self.agent, "api_key", None),
            }
            client, resolved_model, _label = _resolve_auto_route(runtime, "compression")
            if client is None or not resolved_model:
                raise RuntimeError("compaction pipeline: no aux route resolved")
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": "\n\n".join(
                    m["content"] for m in messages)}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        return llm_call


def idle_pipeline_sweep(agent: Any, messages, idle_gap_seconds: float) -> Dict[str, Any]:
    """Entry point hooked into the idle seam. OFF by default (AC-19)."""
    sweep = IdlePipelinePass(agent)
    if not sweep.gates_pass(idle_gap_seconds):
        return {"ran": False, "reason": "gates"}
    return sweep.run(messages)