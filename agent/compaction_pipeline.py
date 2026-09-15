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

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            return record
        except Exception as exc:  # noqa: BLE001 — a failed pass parks; live path unaffected
            logger.warning("compaction pipeline pass failed (%s): %s", self.session_id, exc)
            return {"ran": False, "reason": f"error: {exc}"}
        finally:
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
            result = self._run_extraction_for_window(window, row, messages)
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
                                                  bypass_dump_id=dump_id)
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
                                   ) -> Optional[Dict[str, Any]]:
        """Run A->B->C for the dumped region covering ``window``; return the
        Stage-C checkpoint on success (so the caller can drain the row), None
        on a park/failure. Honors cooldown + budget — EXCEPT for the R5 bypass
        region (``bypass_dump_id``): the bypass region's extraction runs in the
        same pass as its dump (cooldown applies to QUEUED drains, not the
        bypass region; SPEC-0045 R5), so the once-per-pass and cooldown gates
        are waived for exactly that dump id. Budget still binds.

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
        try:
            stage_c = run_extraction_cycle(
                candidate,
                reason_llm=self._accounted(self._stage_llm("reason")),
                extract_llm=self._accounted(self._stage_llm("extract")),
                max_stage_retries=getattr(self.agent, "compaction_pipeline_max_stage_retries", 2),
            )
            self.agent._compaction_pipeline_last_extract_ts = time.time()
            return stage_c
        except StageCheckError as exc:
            logger.warning("extraction parked for %s: %s", candidate.name, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("extraction failed for %s: %s", candidate.name, exc)
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
        A region whose stage_c exists is already extracted; skip it."""
        from agent.compaction_dump import DumpStore
        store = DumpStore(Path(self.storage_root))
        for d in store.dump_dirs(self.session_id):
            if (d / "stage_c.json").is_file():
                continue
            meta = store.read_meta(self.session_id, d.name) or {}
            if not meta.get("complete"):
                continue
            window = (int(meta["start_msg"]), int(meta["end_msg"]))
            stage_c = self._run_extraction_for_window(window, None)
            if stage_c is not None:
                record["extracted"] = d.name
            else:
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
        inject stage llms via ``agent._compaction_stage_llms``."""
        overrides = getattr(self.agent, "_compaction_stage_llms", None)
        if isinstance(overrides, dict) and callable(overrides.get(name)):
            return overrides[name]

        def llm_call(payload):
            from agent.auxiliary_client import _resolve_auto_route
            models = getattr(self.agent, "compaction_pipeline_models", {}) or {}
            model_id = (models or {}).get(name, "") or getattr(self.agent, "model", None)
            runtime = getattr(self.agent, "aux_runtime", None) or {
                "provider": getattr(self.agent, "provider", None),
                "model": model_id,
                "base_url": getattr(self.agent, "base_url", None),
                "api_key": getattr(self.agent, "api_key", None),
            }
            client, resolved_model, _label = _resolve_auto_route(runtime, "compression")
            if client is None or not resolved_model:
                raise RuntimeError(f"compaction pipeline: no aux route for stage {name}")
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": "\n\n".join(
                    m.get("content", "") if isinstance(m, dict) else str(m)
                    for m in payload)}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        return llm_call

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