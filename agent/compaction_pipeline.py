"""SPEC-0042/0043 idle pipeline pass: map updates, dump-at-boundary,
scheduled + persisted extraction, gate + loss probe, degraded-queue drain,
and the batched swap sweep.

Hooked into the existing idle gate (``turn_context_compaction._idle_compaction``
computes the idle gap — we reuse the same seam). All passes:

- run only when ``compaction_pipeline.enabled`` (AC-19: OFF touches nothing);
- respect the map cooldown and the pipeline lock (AC-14);
- are bounded by ``budget_per_session_tokens`` (hard stop, telemetry on hit);
- never mutate the live message list except the batched swap sweep at a
  boundary (D4/D5, AC-25/26) — and that sweep is the ONLY producer stage that
  mutates (map/dump/extract/gate read and write disk artifacts).

The swap sweep's returned message list is surfaced via
``record["swapped_messages"]`` so the turn-context seam can (a) adopt the
mutated list and (b) suppress the legacy idle summarizer for the covered
window (D5, AC-26). Budget accounting is a monotonic per-call summed counter
(AC-22 fix: the vestigial read-then-write-back at the old :92 is gone).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SPEC-0047 D4: a region whose extraction fails with the SAME exception
# type+message this many times parks durably until the map's covers advance
# past the dump window end (something changed that might fix the parse).
DETERMINISTIC_FAILURE_THRESHOLD = 2


class PipelineBudgetExceeded(RuntimeError):
    """Background spend ceiling hit: hard stop; telemetry records it."""


class StageTransportError(RuntimeError):
    """SPEC-0048 D-A: the stage lane returned EMPTY content — a transport-level
    failure, retried ONCE in-process. Distinct from a parse error so the pass
    surfaces "empty stage output (transport)" instead of burning a breaker
    rung on a misleading "not JSON" (F-C: content=None on a trivial request,
    16/16 successes on the identical request minutes later)."""


# SPEC-0048 D-A: per-call LLM timeouts. The read timeout bounds a GENUINE
# stall (zero bytes while the tick thread blocks with the lock held — F-A);
# it must NOT kill slow-but-alive calls: the ollama-cloud lane took 688.2s on
# the real 1.09 MB stage_b payload and completed validly, hence the 900s
# default. Configurable via ``compaction_pipeline.models.call_timeout_s``.
DEFAULT_STAGE_CALL_TIMEOUT_S = 900
DEFAULT_STAGE_CONNECT_TIMEOUT_S = 30


class IdlePipelinePass:
    """One idle-window pass over the pipeline's background duties."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.session_id = getattr(agent, "session_id", "") or "none"
        self.storage_root = getattr(agent, "compaction_pipeline_storage_root", "/tmp/hermes-compaction")
        self.enabled = bool(getattr(agent, "compaction_pipeline_enabled", False))
        # Dump ids whose extraction was already attempted in THIS pass object's
        # run (the R5 bypass region, the queue-drain row) — so the scheduled
        # scan never re-attempts them (one attempt per region per pass; the D4
        # breaker ladders once per pass, not twice).
        self._extraction_attempted: set = set()
        # SPEC-0048 D-C/D-D telemetry: the stage currently in flight (named at
        # each call site) and the pass start, for duration_s + slow-pass and
        # lock-stall warnings.
        self._current_stage: Optional[str] = None
        self._pass_started_at: float = 0.0
        self._lock_acquired_at: float = 0.0
        self._slow_warned: bool = False

    # ── gates ────────────────────────────────────────────────────────────

    def gates_pass(self, idle_gap_seconds: float, now: Optional[float] = None,
                   messages: Optional[List[Any]] = None) -> bool:
        """enabled + idle gap + cooldown + budget; lock acquisition happens in
        ``run`` and is released before returning.

        SPEC-0045 R5: when the rough estimate of ``messages`` is at/over the
        LIVE compression trigger (``compressor.threshold_tokens``), the idle-gap
        check is WAIVED — a continuously-prompted session must still reach the
        pipeline at the same boundary the backstop uses. Only the idle gap is
        waived: cooldown, budget, lock, and the enabled check are NOT. A stub
        compressor without ``threshold_tokens`` (tests/evals) gets no bypass.
        """
        if not self.enabled:
            return False
        idle_after = getattr(self.agent, "compaction_pipeline_map_idle_after_seconds", 20)
        if idle_gap_seconds < max(0.0, float(idle_after)) and not self._over_threshold(messages):
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

    def _over_threshold(self, messages: Optional[List[Any]]) -> bool:
        """True when ``estimate_messages_tokens_rough(messages)`` is at/over the
        live compression threshold — the same estimator and the same value the
        preflight cheap gate and the backstop consult (SPEC-0045 R5). Defensive
        ``getattr``: stub compressors (SimpleNamespace) carry no
        ``threshold_tokens``; None means "no bypass", never a raise."""
        if messages is None:
            return False
        compressor = getattr(self.agent, "context_compressor", None)
        threshold = getattr(compressor, "threshold_tokens", None)
        if threshold is None:
            return False
        try:
            from agent.model_metadata import estimate_messages_tokens_rough
            estimate = int(estimate_messages_tokens_rough(list(messages)))
            return estimate >= int(threshold)
        except Exception:  # noqa: BLE001 — the gate must never raise on a bad estimate
            return False

    def _bypass_fires(self, idle_gap_seconds: float, messages: Optional[List[Any]]) -> bool:
        """The R5 waiver actually fired: the idle gate would have refused this
        pass AND the rough estimate is at/over the live compression trigger.
        Pure predicate for telemetry (``record["bypass"]``) and for the W3
        full-cycle branch; cooldown/budget outcomes are not part of it."""
        if idle_gap_seconds >= max(0.0, float(getattr(
                self.agent, "compaction_pipeline_map_idle_after_seconds", 20))):
            return False  # an ordinary idle pass, not a bypass
        return self._over_threshold(messages)

    def gates_pass_between_turns(self, messages: Optional[List[Any]] = None) -> bool:
        """SPEC-0046 between-turns gates: enabled + cooldown + budget ONLY.

        There is NO idle-gap gate on this path (the operator ruling: the pass
        runs automatically between turns because the context packet is frozen —
        a wall-clock gap buys nothing). Lock acquisition happens in ``run``.
        The R5 threshold predicate is irrelevant here: with no idle gate there
        is nothing to waive; cooldown, budget, and enabled still bind so a
        stale session cannot burn spend."""
        if not self.enabled:
            return False
        now = time.time()
        if now - self._last_pass_ts() < getattr(
                self.agent, "compaction_pipeline_map_cooldown_seconds", 120):
            return False
        if self._spent_tokens() > self._budget():
            logger.info("compaction pipeline budget exceeded for session %s; "
                        "between-turns pass skipped", self.session_id)
            return False
        return True

    def _last_pass_ts(self) -> float:
        return float(getattr(self.agent, "_compaction_pipeline_last_pass_ts", 0.0))

    def _spent_tokens(self) -> int:
        return int(getattr(self.agent, "_compaction_pipeline_spent_tokens", 0))

    def _budget(self) -> int:
        return int(getattr(self.agent, "compaction_pipeline_budget_per_session_tokens", 200000))

    def _budget_exhausted(self) -> bool:
        return self._spent_tokens() > self._budget()

    def _spend(self, tokens: int) -> None:
        """Monotonic per-call budget accounting (AC-22): add, never
        read-then-write-back-same."""
        spent = self._spent_tokens()
        self.agent._compaction_pipeline_spent_tokens = spent + max(0, int(tokens))

    def _accounted(self, llm_call):
        """Wrap an LLM callable so each invocation contributes a token estimate
        to the monotonic per-call budget sum. The estimate is a rough
        input+output char count; deterministic injected llms (the falsifiers)
        cost their payloads so AC-22's budget gate is observable in tests."""
        def wrapped(payload):
            raw = llm_call(payload)
            consumed = (
                sum(len(str(p.get("content", ""))) for p in (payload or []) if isinstance(p, dict))
                + len(raw if isinstance(raw, str) else str(raw))
            ) // 4
            self._spend(consumed)
            return raw
        return wrapped

    # ── execution ────────────────────────────────────────────────────────

    def run(self, messages, llm_call=None, bypass: bool = False) -> Dict[str, Any]:
        """One pass: acquire lock (AC-14), then in order per D1:
        queue-drain -> map update -> dump-at-boundary -> extraction cycle ->
        gate + loss probe -> batched swap sweep. Returns a telemetry record;
        raises nothing — a failed stage logs and parks, the live context is
        untouched (except the boundary sweep, whose result is surfaced in
        ``record["swapped_messages"]``).

        SPEC-0045 R5 (``bypass=True`` — the idle gate was waived at the live
        compression threshold): the pass completes the FULL cycle for the
        current window — dump -> extract -> gate -> swap sweep — in this same
        pass. Rationale: the same turn's preflight compression still sees
        over-threshold context; if the bypass pass only dumped, the backstop
        prose-compacted the same turn, the live list was replaced by the
        summary, and the queued window indices no longer mapped to live
        messages (the dump could never swap). Mechanism: extraction for the
        freshly dumped bypass region runs IN THIS PASS, waiving the
        once-per-pass extraction budget of one region ONLY for that region;
        ``extraction.cooldown_seconds`` keeps its semantics — it applies to
        QUEUED drains, not the bypass region (the boundary is now, not idle
        background). The session budget is NOT waived. If extraction parks
        (StageCheckError / models unreachable), the pass returns normally —
        the same turn's backstop then degrades to prose (the existing
        ``decide_and_swap`` path): the bounded fallback; nothing raises into
        the live path. Telemetry: ``bypass_swapped`` / ``bypass_parked``.
        """
        if not self.enabled:
            return {"ran": False, "reason": "disabled"}
        db = getattr(self.agent, "db", None)
        holder = f"pipeline:{self.session_id}"
        acquired = False
        self._pass_started_at = time.time()
        if db is not None:
            try:
                acquired = db.try_acquire_pipeline_lock(self.session_id, holder)
            except Exception as exc:  # noqa: BLE001 — lock subsystem broken -> skip pass
                logger.warning("pipeline lock acquire failed (%s): %s", self.session_id, exc)
                return {"ran": False, "reason": "lock_error"}
            if not acquired:
                return {"ran": False, "reason": "lock_held"}
            # SPEC-0048 D-D: stamp when THIS pass took the lock (the lease row
            # already carries acquired_at; this mirrors it for the watchdog).
            self._lock_acquired_at = time.time()
        try:
            if llm_call is None:
                llm_call = self._default_llm_call()
            record: Dict[str, Any] = {"ran": True}
            if bypass:
                record["bypass"] = True
            now = time.time()
            # Setup: adopt pre-D1 flat dumps into the canonical per-dump layout
            # (SPEC-0045 R1b) — one cheap idempotent probe; a failure logs and
            # leaves the flat pair in place, never wedging the pass.
            self._adopt_flat_dumps(record)
            # 0. drain one queued degraded region (AC-24), if models reachable.
            self._drain_queued(record, messages)
            # 1. map update over the un-covered tail, with real budget accounting.
            self._update_map(messages, llm_call, record)
            # 2. dump-onto-current-window at an over-threshold idle boundary (AC-20).
            self._dump_current_window(messages, record)
            # 2b. R5 bypass full-cycle: the freshly dumped bypass region is
            #     extracted IN THIS PASS (cooldown waived for it; budget and
            #     every other gate unchanged), then the gate targets it and the
            #     swap sweep runs as usual — so the same turn's backstop finds
            #     a ready region instead of prose-compacting the live window.
            bypass_dump_id = record.get("dumped") if bypass else None
            if bypass_dump_id:
                self._extract_bypass_region(bypass_dump_id, record)
            # 3. scheduled + persisted extraction for dumped-unextracted regions (AC-22).
            self._schedule_extraction(record)
            # 4. always-on gate + loss probe, persisted gate artifact (AC-23).
            self._run_gate(record, target_dump_id=bypass_dump_id)
            # 5. batched swap sweep at the idle boundary (D4, AC-25).
            self._run_swap_sweep(messages, record)
            if bypass_dump_id:
                swapped = record.get("swapped_regions") or []
                if bypass_dump_id in swapped:
                    record["bypass_swapped"] = [bypass_dump_id]
                elif "bypass_parked" not in record:
                    # Extracted + gated but not swapped (gate flagged / sweep
                    # refused): the region stays live; the backstop degrades.
                    record["bypass_parked"] = [bypass_dump_id]
            self.agent._compaction_pipeline_last_pass_ts = now
            _finish_pass_telemetry(self, record)
            return record
        except Exception as exc:  # noqa: BLE001 — a failed pass parks; live path unaffected
            logger.warning("compaction pipeline pass failed (%s): %s", self.session_id, exc)
            return {"ran": False, "reason": f"error: {exc}"}
        finally:
            self._current_stage = None
            if db is not None and acquired:
                try:
                    db.release_pipeline_lock(self.session_id, holder)
                except Exception:  # noqa: BLE001
                    pass

    # ── stage: setup — flat-dump adoption (SPEC-0045 R1b) ─────────────────

    def _adopt_flat_dumps(self, record: Dict[str, Any]) -> None:
        """One-time adoption of pre-D1 flat dumps into ``<sid>/<dump_id>/`` at
        the first storage touch of the pass (SPEC-0045 R1b). A failure logs and
        leaves the flat pair on disk; the pass never wedges on it."""
        from agent.compaction_dump import DumpStore
        try:
            adopted = DumpStore(Path(self.storage_root)).ensure_layout(self.session_id)
        except Exception as exc:  # noqa: BLE001 — adoption must never wedge a pass
            logger.warning("flat-dump adoption failed (%s): %s", self.session_id, exc)
            record["adopt_error"] = str(exc)
            return
        if adopted:
            record["adopted_dumps"] = adopted

    # ── stage: queue drain (AC-24) ───────────────────────────────────────

    def _read_queue(self) -> list:
        qp = Path(self.storage_root) / self.session_id / "pipeline_queue.json"
        if not qp.is_file():
            return []
        try:
            import json
            data = json.loads(qp.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:  # noqa: BLE001
            return []

    def _write_queue(self, rows: list) -> None:
        import json
        import os
        qp = Path(self.storage_root) / self.session_id / "pipeline_queue.json"
        qp.parent.mkdir(parents=True, exist_ok=True)
        tmp = qp.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(rows, ensure_ascii=False, indent=2))
            handle.flush()
        os.replace(tmp, qp)

    def _models_reachable(self) -> bool:
        override = getattr(self.agent, "_compaction_models_reachable", None)
        if override is not None:
            return bool(override)
        return bool(
            getattr(self.agent, "aux_runtime", None)
            or (getattr(self.agent, "provider", None) and getattr(self.agent, "model", None))
        )

    def _drain_queued(self, record: Dict[str, Any], messages=None) -> None:
        """Drain AT MOST one queued region per pass (within budget). A drain
        while models are unreachable must NOT shrink the queue (AC-24): the
        row stays so a later pass retries it. A queued row whose own region no
        longer exists falls back to a fresh dump of the live window rather than
        being refused (AC-24), and the row is only dropped once its work is
        actually done."""
        if not self._models_reachable():
            return
        rows = self._read_queue()
        if not rows:
            record["queued"] = 0
            return
        if self._budget_exhausted():
            return
        # Pick the oldest queued row and try to process it via extraction.
        row = rows[0]
        try:
            window = tuple(int(x) for x in (row.get("window") or [0, 0])[:2])
            result = self._run_extraction_for_window(window, row, messages,
                                                     record=record)
            if result is not None:
                # Processed: re-dump fallback happened inside; drop the row.
                remaining = rows[1:]
                if any(r.get("window") == list(window) for r in remaining):
                    remaining = [r for r in remaining
                                 if list(map(int, r.get("window", []))) != list(window)]
                self._write_queue(remaining)
                record.setdefault("queued_drained", 0)
                record["queued_drained"] += 1
            else:
                # Extraction failed (stage park / budget): keep the row.
                record["queue_refused"] = row.get("reason")
        except Exception as exc:  # noqa: BLE001 — never wedge the pass on a bad row
            logger.warning("queue drain failed for row %s: %s", row, exc)
            record["queue_error"] = str(exc)

    # ── stage: map update ────────────────────────────────────────────────

    def _update_map(self, messages, llm_call, record: Dict[str, Any]) -> None:
        from agent.compaction_map import CompactionMap
        cm = CompactionMap(Path(self.storage_root), self.session_id)
        covers_end = int(cm.load().get("covers", {}).get("end_msg", 0))
        new_msgs = list(messages)[covers_end:]
        if not new_msgs:
            record["updated"] = False
            return
        if self._budget_exhausted():
            record["updated"] = False
            record["budget_blocked"] = "map"
            return
        try:
            cm.update(self._accounted(llm_call), new_msgs,
                      start_msg=covers_end, end_msg=covers_end + len(new_msgs) - 1)
            record["updated"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("map update failed (%s): %s", self.session_id, exc)
            record["updated"] = False
            record["map_error"] = str(exc)

    # ── stage: dump-at-boundary (AC-20) ─────────────────────────────────

    def _current_compression_window(self, messages) -> Optional[tuple]:
        compressor = getattr(self.agent, "context_compressor", None)
        if compressor is None:
            return None
        try:
            win = compressor._compress_window(messages)
            if isinstance(win, tuple) and len(win) == 2:
                s, e = int(win[0]), int(win[1])
                if 0 <= s < e:
                    return (s, e)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _dump_current_window(self, messages, record: Dict[str, Any]) -> None:
        """Write a complete dump of the current compression window, idempotent
        per (window, region-hash): a second pass over the same window/content
        does NOT rewrite the existing dump (no fsync write — AC-20). A stale
        window dump is superseded by the fresh window dump."""
        window = self._current_compression_window(messages)
        if window is None:
            return
        s, e = int(window[0]), int(window[1])
        if e >= len(messages):
            return
        from agent.compaction_dump import DumpStore, region_hash8
        store = DumpStore(Path(self.storage_root))
        region = list(messages)[s:e + 1]
        try:
            # Idempotency: an existing complete dump for this window + content
            # hash is left untouched (mtime stable, no fsync write).
            expected = f"{1:04d}-{region_hash8(region)}"
            if store.is_complete(self.session_id, expected):
                record["dumped"] = expected
                record["dump_window"] = [s, e]
                record["dumped_exists"] = True
                return
            ref = store.write_dump(self.session_id, region, start_msg=s, end_msg=e, turn=1)
            record["dumped"] = ref.dump_id
            record["dump_window"] = [s, e]
        except Exception as exc:  # noqa: BLE001
            logger.warning("idle dump failed (%s): %s", self.session_id, exc)
            record["dump_error"] = str(exc)

    def _extract_bypass_region(self, dump_id: str, record: Dict[str, Any]) -> None:
        """R5 full-cycle: run the A->B->C cycle for the freshly dumped bypass
        region IN THIS PASS.

        Waived for THIS region only: the once-per-pass extraction budget of one
        region and ``extraction.cooldown_seconds`` (cooldown applies to QUEUED
        drains — the bypass boundary is now, not idle background). The session
        budget is NOT waived, nor are models-reachable / dump-complete gates.
        A park (StageCheckError / models unreachable / budget / no complete
        dump) is recorded as ``bypass_parked`` telemetry; nothing raises into
        the live path — the same turn's backstop degrades to prose instead."""
        from agent.compaction_dump import DumpStore
        store = DumpStore(Path(self.storage_root))
        if self._budget_exhausted():
            record["bypass_parked"] = [dump_id]
            record["budget_blocked"] = "bypass_extract"
            return
        if not self._models_reachable():
            record["bypass_parked"] = [dump_id]
            record["models_unreachable"] = True
            return
        if not store.is_complete(self.session_id, dump_id):
            record["bypass_parked"] = [dump_id]
            return
        meta = store.read_meta(self.session_id, dump_id) or {}
        window = (int(meta.get("start_msg", 0)), int(meta.get("end_msg", 0)))
        stage_c = self._run_extraction_for_window(window, None,
                                                  bypass_dump_id=dump_id,
                                                  record=record)
        if stage_c is not None:
            record["bypass_extracted"] = dump_id
        else:
            record.setdefault("bypass_parked", [dump_id])

    # ── stage: scheduled + persisted extraction (AC-22) ─────────────────

    def _extraction_cooldown_ok(self) -> bool:
        last = getattr(self.agent, "_compaction_pipeline_last_extract_ts", 0.0)
        cooldown = getattr(self.agent, "compaction_pipeline_extraction_cooldown_seconds", 60)
        return (time.time() - float(last)) >= max(0.0, float(cooldown))

    def _complete_dump_for_window(self, store, window: Optional[tuple]):
        """The complete dump directory whose meta window equals ``window``, or
        None. ``None`` window means "any complete dump" (queue-drain lookup by
        dump_id has no window to match)."""
        want = (int(window[0]), int(window[1])) if window else None
        for d in store.dump_dirs(self.session_id):
            meta = store.read_meta(self.session_id, d.name) or {}
            if not meta.get("complete"):
                continue
            got = (int(meta.get("start_msg", 0)), int(meta.get("end_msg", 0)))
            if want is not None and got != want:
                continue
            return d
        return None

    def _run_extraction_for_window(self, window: Optional[tuple],
                                   row: Optional[dict],
                                   messages=None,
                                   bypass_dump_id: Optional[str] = None,
                                   record: Optional[Dict[str, Any]] = None,
                                   ) -> Optional[Dict[str, Any]]:
        """Run A->B->C for the dumped region covering ``window``; return the
        Stage-C checkpoint on success (so the caller can drain the row), None
        on a park/failure. Honors cooldown + budget — EXCEPT for the R5 bypass
        region (``bypass_dump_id``): the bypass region's extraction runs in the
        same pass as its dump (cooldown applies to QUEUED drains, not the
        bypass region; SPEC-0045 R5), so the once-per-pass and cooldown gates
        are waived for exactly that dump id. Budget still binds.

        ``record`` (optional) is the pass telemetry record; when supplied,
        breaker skips and park events are recorded on it
        (``parked_regions`` / ``extract_park`` / ``extract_failed``).

        ``row`` is the queued row for queue-drain. AC-24 fresh-dump fallback:
        when the row's window has no complete dump (its dump was superseded by a
        later window), the row is neither extracted against the wrong region nor
        silently dropped — the LIVE window is dumped fresh and THAT region is
        extracted, so the queued work actually gets done.
        """
        from agent.compaction_extract import StageCheckError, run_extraction_cycle
        if not self._models_reachable():
            return None  # nobody can run A->B->C without a resolvable route
        is_bypass = bypass_dump_id is not None
        if not is_bypass and not self._extraction_cooldown_ok():
            return None
        if self._budget_exhausted():
            return None
        from agent.compaction_dump import DumpStore
        store = DumpStore(Path(self.storage_root))
        if is_bypass:
            candidate = store.dump_dir(self.session_id, bypass_dump_id)
            if not candidate.is_dir():
                return None
        else:
            candidate = self._complete_dump_for_window(store, window)
            if candidate is None and row is not None and messages is not None:
                fresh = self._dump_fresh_window(messages)
                if fresh is None:
                    return None
                candidate = self._complete_dump_for_window(store, fresh)
        if candidate is None:
            return None
        # SPEC-0047 D4 circuit breaker: a region parked durably after repeated
        # IDENTICAL extraction failures is skipped until the map's covers end
        # advances past the dump window end (new map content might fix the
        # parse). Telemetry records every skip.
        telemetry = record if record is not None else {}
        from agent.compaction_map import CompactionMap
        covers_end_now = int(CompactionMap(Path(self.storage_root), self.session_id)
                             .load().get("covers", {}).get("end_msg", 0) or 0)
        if _park_blocks(candidate, covers_end_now):
            park = _read_park(candidate) or {}
            logger.info(
                "extraction skipped for %s: parked (%d identical failures, "
                "covers end %s vs dump end %s)",
                candidate.name, park.get("failures", 0), covers_end_now,
                park.get("dump_end"))
            telemetry.setdefault("parked_regions", []).append(
                {"dump_id": candidate.name, "reason": park.get("reason"),
                 "failures": park.get("failures", 0)})
            return None
        # A park whose unblock condition is already met clears now: the next
        # attempt ladders from a clean slate instead of inheriting a stale
        # failure count (D4: retry only when something changed).
        _clear_expired_park(candidate, covers_end_now)
        meta = store.read_meta(self.session_id, candidate.name) or {}
        dump_end = int(meta.get("end_msg", 0)) if meta else None
        self._extraction_attempted.add(candidate.name)
        try:
            stage_c = run_extraction_cycle(
                candidate,
                reason_llm=self._accounted(self._stage_llm("reason")),
                extract_llm=self._accounted(self._stage_llm("extract")),
                max_stage_retries=getattr(self.agent, "compaction_pipeline_max_stage_retries", 2),
            )
            self.agent._compaction_pipeline_last_extract_ts = time.time()
            # Success clears any lingering park marker (e.g. a transient
            # failure streak that never hit the breaker threshold).
            _clear_park(candidate)
            return stage_c
        except StageCheckError as exc:
            logger.warning("extraction parked for %s: %s", candidate.name, exc)
            marker = _park_region(candidate, reason=str(exc), exc=exc,
                                  covers_end=covers_end_now, dump_end=dump_end)
            if marker["failures"] < DETERMINISTIC_FAILURE_THRESHOLD:
                telemetry["extract_park"] = {"dump_id": candidate.name,
                                             "failures": marker["failures"]}
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("extraction failed for %s: %s", candidate.name, exc)
            marker = _park_region(candidate, reason=str(exc), exc=exc,
                                  covers_end=covers_end_now, dump_end=dump_end)
            if marker["failures"] < DETERMINISTIC_FAILURE_THRESHOLD:
                telemetry["extract_failed"] = {"dump_id": candidate.name,
                                               "error": str(exc)}
            return None

    def _dump_fresh_window(self, messages) -> Optional[tuple]:
        """Dump the live compression window now, returning its (start, end) — the
        AC-24 fallback target when a queued row's own region is gone."""
        window = self._current_compression_window(messages)
        if window is None:
            return None
        s, e = int(window[0]), int(window[1])
        if e >= len(messages):
            return None
        from agent.compaction_dump import DumpStore, region_hash8
        store = DumpStore(Path(self.storage_root))
        region = list(messages)[s:e + 1]
        try:
            expected = f"{1:04d}-{region_hash8(region)}"
            if not store.is_complete(self.session_id, expected):
                store.write_dump(self.session_id, region, start_msg=s, end_msg=e, turn=1)
        except Exception as exc:  # noqa: BLE001 — fallback failure keeps the row
            logger.warning("fresh-dump fallback failed (%s): %s", self.session_id, exc)
            return None
        return (s, e)

    def _schedule_extraction(self, record: Dict[str, Any]) -> None:
        """Extract dumped-unextracted regions (≤ one region cycle per pass).
        A region whose stage_c exists is already extracted; skip it. A region
        already extraction-attempted THIS pass (the R5 bypass region) is not
        attempted again — one extraction attempt per region per pass, so a
        deterministic failure ladders once per pass, not twice."""
        from agent.compaction_dump import DumpStore
        store = DumpStore(Path(self.storage_root))
        for d in store.dump_dirs(self.session_id):
            if (d / "stage_c.json").is_file():
                continue
            if d.name in self._extraction_attempted:
                continue
            meta = store.read_meta(self.session_id, d.name) or {}
            if not meta.get("complete"):
                continue
            window = (int(meta["start_msg"]), int(meta["end_msg"]))
            stage_c = self._run_extraction_for_window(window, None, record=record)
            if stage_c is not None:
                record["extracted"] = d.name
            else:
                # Parked (D4 breaker skip, StageCheckError, or any failure) —
                # the region stays live and is retried on a later pass.
                record["extract_parked"] = d.name
            return  # ≤ one region cycle per pass
        record["no_extraction"] = True

    # ── stage: gate + loss probe (AC-23) ────────────────────────────────

    def _run_gate(self, record: Dict[str, Any], target_dump_id: Optional[str] = None) -> None:
        """Run the always-on gate over {checkpoint + dump} and persist the gate
        artifact next to the checkpoint. A loss-probe gap flips the region back
        to Stage B (its stage_c is removed so a later pass re-extracts).

        ``target_dump_id`` (SPEC-0045 R5 bypass full-cycle): when set, the gate
        examines that dump FIRST so the freshly extracted bypass region can be
        gated and swapped in the same pass; if it has no stage_c yet (extraction
        parked) the scan continues with the ordinary ordering."""
        from agent.compaction_dump import DumpStore
        from agent.compaction_verify import (
            generate_loss_probe_questions, run_loss_probe, run_review_gate)
        store = DumpStore(Path(self.storage_root))
        sdir = store.session_dir(self.session_id)
        dirs = store.dump_dirs(self.session_id)
        if target_dump_id is not None:
            target_dir = store.dump_dir(self.session_id, target_dump_id)
            if target_dir in dirs:
                dirs.remove(target_dir)
                dirs.insert(0, target_dir)
        for d in dirs:
            if (d / "gate.json").is_file():
                continue
            stage_c = _read_json(d / "stage_c.json")
            if stage_c is None:
                continue
            try:
                msgs = store.read_messages(self.session_id, d.name)
            except Exception:  # noqa: BLE001
                continue
            gate_llm = self._accounted(self._stage_llm("gate"))
            verdict = run_review_gate(gate_llm, stage_c, msgs, seed=0)
            # loss probe (always-on): a gap flips back to Stage B.
            always_on = getattr(self.agent, "compaction_pipeline_gate_always_on", True)
            loss_probe = {"gaps": [], "pass": True, "questions": 0}
            if always_on:
                try:
                    qllm = self._accounted(self._stage_llm("gate"))
                    questions = generate_loss_probe_questions(
                        qllm, msgs, samples=getattr(
                            self.agent, "compaction_pipeline_loss_probe_samples", 8), seed=0)
                    allm = self._accounted(self._stage_llm("gate"))
                    glmm = self._accounted(self._stage_llm("gate"))
                    loss_probe = run_loss_probe(allm, glmm, stage_c, msgs, questions)
                except Exception as exc:  # noqa: BLE001
                    loss_probe = {"gaps": [{"error": str(exc)}], "pass": False, "questions": 0}
                if loss_probe.get("gaps"):
                    # Gap -> back to Stage B: drop stage_c so it re-extracts.
                    try:
                        (Path(sdir) / d.name / "stage_c.json").unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        pass
                    record["gate_flipped_b"] = d.name
                    continue
            verdict["loss_probe"] = loss_probe
            _atomic_write_json(Path(sdir) / d.name / "gate.json", verdict)
            record["gated"] = d.name
            record["swap_eligible"] = verdict.get("swap_eligible")
            return  # one gate per pass is enough

    # ── stage: batched swap sweep (D4, AC-25) ───────────────────────────

    def _run_swap_sweep(self, messages, record: Dict[str, Any]) -> None:
        """At the idle boundary, swap ALL current-window-matching, complete,
        gate-passed regions in one message-list mutation. Surface the swapped
        list via ``record["swapped_messages"]`` so the turn-context seam adopts
        it and suppresses the legacy summarizer (D5/AC-26).

        D3: the sweep computes its OWN window from the live pass messages and
        threads THAT into ``swap_sweep``. The backstop's
        ``_current_compression_window()`` reads ``compressor.last_compress_window``,
        which nothing stamps on an idle pass — it is None there, and a None window
        makes the stale-window guard a no-op (any gate-passed region would swap
        regardless of which window it covers).

        SPEC-0046: the ready regions also land on ``record["pending_ready_regions"]``
        so the between-turns entry can stage them (packet hash + region refs +
        gate digest) without recomputing discovery.
        """
        from agent.compaction_dump import DumpStore
        from agent.compaction_swap import swap_sweep
        store = DumpStore(Path(self.storage_root))
        window = self._current_compression_window(messages)
        record["idle_window"] = list(window) if window is not None else None
        ready = []
        for d in store.dump_dirs(self.session_id):
            stage_c = _read_json(d / "stage_c.json")
            gate = _read_json(d / "gate.json")
            if stage_c is None or gate is None:
                continue
            if gate.get("swap_eligible") is not True:
                continue
            meta = store.read_meta(self.session_id, d.name) or {}
            if not meta.get("complete"):
                continue
            try:
                dw = (int(meta.get("start_msg", 0)), int(meta.get("end_msg", 0)))
            except (TypeError, ValueError):
                continue
            if window is not None and dw != window:
                continue  # stale-window discipline: never swap the wrong window
            ready.append({"dump_id": d.name, "meta": meta,
                          "checkpoint": stage_c, "gate": gate})
        record["pending_ready_regions"] = ready
        if not ready:
            return
        from agent.compaction_rehydrate import StubRegistry
        registry = StubRegistry.for_session(Path(self.storage_root), self.session_id)
        try:
            swapped = swap_sweep(messages, ready_regions=ready, dump_store=store,
                                 session_id=self.session_id, stub_registry=registry,
                                 current_window=window)
            if swapped is not None and swapped != messages:
                record["swapped_messages"] = swapped
                record["swapped_regions"] = [r["dump_id"] for r in ready]
        except Exception as exc:  # noqa: BLE001
            logger.warning("swap sweep failed (%s): %s", self.session_id, exc)
            record["swap_error"] = str(exc)

    # ── stage llm resolution ─────────────────────────────────────────────

    def _stage_llm(self, name: str):
        """Resolve a per-stage LLM callable. Defaults to the map-update aux
        resolution; config ``compaction_pipeline.models.<name>`` overrides the
        model id (resolved through the same aux chain). Deterministic tests
        inject stage llms via ``agent._compaction_stage_llms``.

        SPEC-0048 D-A: every stage call carries ``timeout=(connect, read)`` —
        read default 900s (``compaction_pipeline.models.call_timeout_s``) — and
        treats EMPTY content as a transport-level failure (StageTransportError)
        retried exactly ONCE in-process. SPEC-0048 D-C: the wrapper stamps the
        in-flight stage name for the duration/slow-pass telemetry."""
        overrides = getattr(self.agent, "_compaction_stage_llms", None)
        if isinstance(overrides, dict) and callable(overrides.get(name)):
            base = overrides[name]
        else:
            base = self._aux_stage_llm(name)
        return self._tracked_stage_llm(name, base)

    def _aux_stage_llm(self, name: str):
        models = getattr(self.agent, "compaction_pipeline_models", {}) or {}
        read_timeout = float(models.get("call_timeout_s", DEFAULT_STAGE_CALL_TIMEOUT_S))

        def llm_call(payload):
            from agent.auxiliary_client import _resolve_auto_route
            model_id = models.get(name, "") or getattr(self.agent, "model", None)
            runtime = getattr(self.agent, "aux_runtime", None) or {
                "provider": getattr(self.agent, "provider", None),
                "model": model_id,
                "base_url": getattr(self.agent, "base_url", None),
                "api_key": getattr(self.agent, "api_key", None),
            }
            client, resolved_model, _label = _resolve_auto_route(runtime, "compression")
            if client is None or not resolved_model:
                raise RuntimeError(f"compaction pipeline: no aux route for stage {name}")
            return _call_stage_llm(name, client, resolved_model, payload, read_timeout)
        return llm_call

    def _tracked_stage_llm(self, name: str, base):
        """Wrap a stage callable so the pass knows the stage in flight (D-C)
        and the D-A transport discipline applies to INJECTED lanes too (the
        deterministic-falsifier lane that returns "" is exactly the F-C shape):
        stamped before each call, transport-wrapped, slow-pass checked after."""

        def tracked(payload):
            self._mark_stage(name)
            try:
                return _stage_output_or_transport(name, base, payload)
            finally:
                self._check_slow_pass()
        return tracked

    def _mark_stage(self, name: str) -> None:
        """Stamp the in-flight stage name; cheap slow-pass check at each stage
        boundary (D-C: a pass past slow_pass_warn_s warns ONCE, naming the
        stage, instead of a silent 12-minute gap reading as a hang). Also the
        D-D lock-stall watchdog checkpoint."""
        self._current_stage = name
        self._check_slow_pass()
        _check_lock_stall(self)

    def _check_slow_pass(self) -> None:
        if self._slow_warned or not self._pass_started_at:
            return
        elapsed = time.time() - self._pass_started_at
        warn_s = _slow_pass_warn_s(self.agent)
        if elapsed > warn_s:
            self._slow_warned = True
            logger.warning(
                "slow compaction pass (%s): %.1fs elapsed exceeds "
                "slow_pass_warn_s=%.0fs; stage in flight: %s",
                self.session_id, elapsed, warn_s, self._current_stage)

    def _default_llm_call(self):
        """Resolve the map-update model via the existing aux resolution chain
        (back-compat entry; stages use :meth:`_stage_llm`)."""
        base = self._stage_llm("map_update")
        return base


def _read_json(path) -> Optional[Dict[str, Any]]:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _atomic_write_json(path, obj: Dict[str, Any]) -> None:
    import json
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, default=str))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001
            pass
    os.replace(tmp, path)


# ── SPEC-0048 D-C/D-D: pass-duration telemetry + lock-stall watchdog ──────


def _stage_output_or_transport(name: str, call, payload):
    """Shared stage-output discipline (D-A): invoke ``call(payload)``, treat
    EMPTY content as a transport failure (F-C: transient content=None on the
    lane — 16/16 successes on the identical request minutes later), retry
    exactly ONCE in-process, then surface StageTransportError. The pass's
    park/retry ladder sees its own error string ("empty stage output
    (transport)"), never a misleading "not JSON". Timeout/connection errors
    from the lane map to StageTransportError too, so a stalled call unwinds
    through the normal pass path instead of hanging the tick thread."""
    import openai

    try:
        content = call(payload)
        if not content:
            logger.warning("stage %s: empty content from lane; retrying once "
                           "(transport)", name)
            content = call(payload)
        if not content:
            raise StageTransportError("empty stage output (transport)")
        return content
    except openai.APITimeoutError as exc:
        raise StageTransportError(
            f"stage {name} call timed out (transport): {exc}") from exc
    except openai.APIConnectionError as exc:
        raise StageTransportError(f"stage {name} transport error: {exc}") from exc


def _call_stage_llm(name: str, client, resolved_model: str, payload,
                    read_timeout: float):
    """One aux-lane stage LLM call with a hard client-side timeout (D-A).
    The read timeout bounds a GENUINE stall (F-A: zero bytes while the tick
    thread blocks with the lock held) without killing slow-but-alive calls —
    the ollama-cloud lane legitimately took 688.2s on the real 1.09 MB stage_b
    payload, hence the 900s default. Empty/transport handling is shared with
    the injected-lane path via :func:`_stage_output_or_transport`."""
    import openai

    def once(p):
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": "\n\n".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in p)}],
            temperature=0,
            timeout=(DEFAULT_STAGE_CONNECT_TIMEOUT_S, read_timeout),
        )
        return resp.choices[0].message.content or ""

    try:
        return _stage_output_or_transport(name, once, payload)
    except StageTransportError:
        raise
    except openai.APIStatusError as exc:
        raise StageTransportError(f"stage {name} transport error: {exc}") from exc


def _slow_pass_warn_s(agent: Any) -> float:
    """``compaction_pipeline.slow_pass_warn_s`` (default 300): a pass running
    longer than this logs a WARNING naming the stage in flight (D-C); a pass
    HOLDING THE LOCK longer than 4x this logs a lock-stall WARNING (D-D)."""
    return float(getattr(agent, "compaction_pipeline_slow_pass_warn_s", 300))


def _finish_pass_telemetry(sweep: "IdlePipelinePass",
                           record: Dict[str, Any]) -> None:
    """Stamp ``duration_s`` on the record; warn when the pass exceeded
    ``slow_pass_warn_s`` (D-C) or held the lock beyond ``slow_pass_warn_s * 4``
    (D-D). Diagnostic only: the D-A timeout already bounds a real stall."""
    elapsed = time.time() - sweep._pass_started_at
    record["duration_s"] = round(max(0.0, elapsed), 3)
    warn_s = _slow_pass_warn_s(sweep.agent)
    if elapsed > warn_s:
        record["slow_pass"] = True
    if not sweep._slow_warned and elapsed > warn_s:
        # D-C: warn once per pass, naming the stage in flight (the D-C
        # stage-boundary check may already have warned — never double-log).
        sweep._slow_warned = True
        logger.warning(
            "slow compaction pass (%s): %.1fs exceeds slow_pass_warn_s=%.0fs "
            "(stage in flight at pass end: %s)",
            sweep.session_id, elapsed, warn_s, sweep._current_stage)
    _check_lock_stall(sweep)


def _check_lock_stall(sweep: "IdlePipelinePass") -> None:
    """D-D watchdog, checked cheaply at stage boundaries: a WARNING naming the
    holder when this pass has held the lock beyond ``slow_pass_warn_s * 4``.
    No new thread; no behavior change beyond the log."""
    acquired_at = sweep._lock_acquired_at
    if not acquired_at:
        return
    held = time.time() - acquired_at
    warn_s = _slow_pass_warn_s(sweep.agent)
    if held > warn_s * 4:
        logger.warning(
            "pipeline lock stall (%s): holder pipeline:%s has held the lock "
            "for %.1fs (slow_pass_warn_s*4=%.0fs); stage in flight: %s",
            sweep.session_id, sweep.session_id, held, warn_s * 4,
            sweep._current_stage)


# ── SPEC-0047 D4: deterministic-failure circuit breaker ──────────────────


def _park_path(dump_dir: Path) -> Path:
    """The durable park marker next to the dump: ``<dump_id>.parked.json``."""
    return Path(dump_dir) / "extraction.parked.json"


def _read_park(dump_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json(_park_path(dump_dir))


def _park_region(dump_dir: Path, *, reason: str, exc: Optional[BaseException] = None,
                 covers_end: Optional[int] = None,
                 dump_end: Optional[int] = None) -> Dict[str, Any]:
    """Durably park a region after a failed extraction attempt.

    Increments the failure count when the exception type+message repeats,
    resets it when the failure signature CHANGES (a new failure mode restarts
    the ladder), and stamps the map covers end observed at failure time. The
    marker clears itself once ``covers.end_msg`` advances past the dump
    window end (D4: new map content might fix the parse — retry then).
    """
    marker = {
        "parked": True,
        "reason": reason,
        "failures": 1,
        "last_error_type": type(exc).__name__ if exc is not None else None,
        "last_error": str(exc) if exc is not None else None,
        "covers_end_at_failure": covers_end,
        "dump_end": dump_end,
        "parked_at": time.time(),
        "park_version": 1,
    }
    prior = _read_park(dump_dir) or {}
    if (prior.get("last_error_type") == marker["last_error_type"]
            and prior.get("last_error") == marker["last_error"]):
        marker["failures"] = int(prior.get("failures", 0)) + 1
        marker["parked_at"] = prior.get("parked_at", marker["parked_at"])
    _atomic_write_json(_park_path(dump_dir), marker)
    logger.warning(
        "extraction parked durably for %s (%d identical failures): %s",
        Path(dump_dir).name, marker["failures"], reason)
    return marker


def _clear_park(dump_dir: Path) -> None:
    try:
        _park_path(dump_dir).unlink(missing_ok=True)
    except OSError:
        pass


def _park_blocks(dump_dir: Path, covers_end: Optional[int]) -> bool:
    """True when a parked region must be skipped: the failure ladder has
    reached the breaker threshold AND the map covers have not yet advanced
    past the dump window end. Below the threshold the region keeps retrying
    (transient failures); ``covers_end=None`` (no map / empty map) never
    unblocks an at-threshold park — nothing changed that could fix the parse.
    A corrupt marker does not block (fail-open to a normal retry)."""
    park = _read_park(dump_dir)
    if not park:
        return False
    try:
        failures = int(park.get("failures", 0) or 0)
    except (TypeError, ValueError):
        return False
    if failures < DETERMINISTIC_FAILURE_THRESHOLD:
        return False
    dump_end = park.get("dump_end")
    if not isinstance(dump_end, int):
        return False
    if covers_end is None:
        return True
    try:
        return int(covers_end) < int(dump_end)
    except (TypeError, ValueError):
        return True


def _clear_expired_park(dump_dir: Path, covers_end: Optional[int]) -> None:
    """Drop a park marker whose unblock condition is already satisfied (the
    map advanced past the dump end): the next attempt ladders from a clean
    slate instead of inheriting the stale failure count."""
    park = _read_park(dump_dir)
    if not park:
        return
    dump_end = park.get("dump_end")
    if not isinstance(dump_end, int) or covers_end is None:
        return
    try:
        if int(covers_end) >= int(dump_end):
            _clear_park(dump_dir)
    except (TypeError, ValueError):
        pass


def idle_pipeline_sweep(agent: Any, messages, idle_gap_seconds: float) -> Dict[str, Any]:
    """Entry point hooked into the idle seam. OFF by default (AC-19).

    SPEC-0045 R5: ``messages`` ride into ``gates_pass`` so the idle-gap check
    can be waived at the live compression threshold; ``record["bypass"] = True``
    marks a pass that ran on that waiver (the §4 monitoring signature for
    repeated degrades at the threshold bypass)."""
    sweep = IdlePipelinePass(agent)
    bypass = sweep._bypass_fires(idle_gap_seconds, messages)
    if not sweep.gates_pass(idle_gap_seconds, messages=messages):
        return {"ran": False, "reason": "gates"}
    return sweep.run(messages, bypass=bypass)


# ── SPEC-0046: the between-turns pass (frozen packet, stage-not-apply) ────


def _frozen_packet_messages(agent: Any):
    """Read the frozen between-turns packet without mutating anything.

    Source order: the aliased ``agent._session_messages`` when it is a list
    (it aliases the session history and is not mutated between turns), else
    the session DB's active rows. A fresh gateway session may have neither —
    that is a no-op with a reason, never a raise."""
    aliased = getattr(agent, "_session_messages", None)
    if isinstance(aliased, list) and aliased:
        return list(aliased)
    db = getattr(agent, "db", None) or getattr(agent, "_session_db", None)
    reader = getattr(db, "get_messages_as_conversation", None)
    session_id = getattr(agent, "session_id", "") or ""
    if reader is not None and session_id:
        try:
            rows = reader(session_id, repair_alternation=True, include_row_ids=True)
            if isinstance(rows, list) and rows:
                return list(rows)
        except Exception as exc:  # noqa: BLE001 — a bad read degrades to a no-op
            logger.warning("between-turns packet DB read failed (%s): %s",
                           session_id, exc)
    return None


def _swap_regions_digest(record: Dict[str, Any]) -> str:
    from agent.compaction_pending_swap import digest_gate_verdicts
    return digest_gate_verdicts(
        [r.get("gate") for r in (record.get("pending_ready_regions") or [])])


def between_turns_pass(agent: Any, llm_call=None) -> Dict[str, Any]:
    """SPEC-0046 one-shot entry: run ONE pipeline pass against the frozen
    between-turns packet and STAGE (never apply) the ready swap.

    Between turns the message list is not consumed or mutated by anyone — the
    same bytes the next ``build_turn_context`` will load — so the pass runs the
    full chain (flat-dump adoption, queue drain, map update, dump, extraction,
    gate + loss probe) exactly as :meth:`IdlePipelinePass.run` does, then
    computes the swap sweep's replacement list LOCALLY and persists it as a
    pending-swap record keyed by the packet's row-identity hash. No live list
    exists to mutate and none is touched (AC-A4). The next turn start
    reinjects the staged swap when the packet hash still matches (AC-A3).

    Gates: lock, cooldown, budget — via :meth:`gates_pass` with the frozen
    packet (the R5 threshold waiver rides along unchanged). Telemetry:
    ``trigger="between_turns"``; a staged swap carries ``deferred_swap=True``
    and ``packet_hash``; ``applied`` is always False here (nothing live is
    mutated). One-shot per turn boundary: the tick is armed once at turn end
    and never re-armed while the session stays idle (operator ruling).
    """
    from agent.compaction_pending_swap import (
        build_pending_swap_record,
        clear_pending_swap,
        compute_packet_hash,
        packet_identity,
        write_pending_swap,
    )

    def _noop(reason: str) -> Dict[str, Any]:
        return {"ran": False, "trigger": "between_turns", "reason": reason,
                "applied": False}

    sweep = IdlePipelinePass(agent)
    if not getattr(agent, "compaction_pipeline_between_turns_sweep", True):
        return _noop("between_turns_disabled")
    # The frozen packet: aliased list, else session DB active rows. Absent
    # packet -> no-op with a reason (fresh gateway session before first turn).
    messages = _frozen_packet_messages(agent)
    if not messages:
        return _noop("no_packet")
    # Gates WITHOUT any idle-gap condition (SPEC-0046): enabled + cooldown +
    # budget only. Lock acquisition happens inside ``sweep.run``.
    if not sweep.gates_pass_between_turns(messages=messages):
        return _noop("gates")
    packet_hash = compute_packet_hash(messages)
    record = sweep.run(messages, llm_call=llm_call, bypass=False)
    record["trigger"] = "between_turns"
    record.setdefault("applied", False)
    if not record.get("ran"):
        return record
    swapped = record.get("swapped_messages")
    if swapped is None:
        # Nothing gate-passed to stage; the artifacts (map/dump/gate) still
        # advanced storage — that is the pass's progress for this boundary.
        return record
    # STAGE, never apply: persist the ready swap keyed by the frozen packet's
    # identity, and drop the computed list from the record — nothing live is
    # mutated by this pass (AC-A4).
    ready = record.pop("pending_ready_regions", None) or []
    regions = [
        {"dump_id": r.get("dump_id"),
         "start_msg": int((r.get("meta") or {}).get("start_msg", 0)),
         "end_msg": int((r.get("meta") or {}).get("end_msg", 0))}
        for r in (ready or [])
    ]
    staged = build_pending_swap_record(
        packet_hash=packet_hash,
        packet_identity=packet_identity(messages),
        swapped_messages=swapped,
        region_refs=regions,
        gate_verdict_digest=_swap_regions_digest(record),
    )
    try:
        write_pending_swap(sweep.storage_root, sweep.session_id, staged)
    except Exception as exc:  # noqa: BLE001 — staging failure parks the swap
        logger.warning("pending-swap persist failed (%s): %s", sweep.session_id, exc)
        record["stage_error"] = str(exc)
        record["deferred_swap"] = False
        clear_pending_swap(sweep.storage_root, sweep.session_id)
        return record
    record["deferred_swap"] = True
    record["packet_hash"] = packet_hash
    record["staged_regions"] = [r["dump_id"] for r in regions]
    # Stage-not-apply: the computed list must never ride on the record as an
    # adopted swap (there is no live list; adoption happens at next turn start).
    record.pop("swapped_messages", None)
    record.pop("swapped_regions", None)
    return record