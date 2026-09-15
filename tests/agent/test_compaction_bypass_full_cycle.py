"""SPEC-0045 W3 (R5 rationale) — bypass pass completes the FULL cycle.

A dump-only bypass pass is not sufficient: the same turn's preflight
compression still sees over-threshold context, the backstop prose-compacts,
the live list is replaced by the summary, and the queued window indices no
longer map to live messages — the dump can never swap. On a bypass pass the
freshly dumped region is extracted IN THE SAME PASS (once-per-pass extraction
budget and extraction.cooldown_seconds waived for THAT region only; the session
budget and every other gate unchanged), then gate + swap sweep run as usual.

Deterministic: injected stage llms, no model calls. The parked path asserts the
bounded fallback: the pass returns normally and CompactionBackstop.decide_and_swap
on the same messages degrades with the queue row appended.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.compaction_backstop import (
    TELEMETRY_DEGRADATION_REASON,
    TELEMETRY_DEGRADED,
    CompactionBackstop,
)
from agent.compaction_dump import DumpStore
from agent.compaction_pipeline import IdlePipelinePass


def _messages(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


class _FakeDB:
    def __init__(self):
        self.db_path = "/tmp/fake-never-opened.db"
        self._pipeline_held = False
        self._compression_held = False

    def try_acquire_pipeline_lock(self, session_id, holder, ttl_seconds=300.0):
        if self._pipeline_held:
            return False
        self._pipeline_held = True
        return True

    def release_pipeline_lock(self, session_id, holder):
        self._pipeline_held = False

    def compression_lock_holder(self, session_id):
        return "compressor" if self._compression_held else None

    def pipeline_lock_holder(self, session_id):
        return "pipeline" if self._pipeline_held else None


def _agent(root, window=(0, 5), messages_len=8, models_reachable=True):
    a = SimpleNamespace()
    a.session_id = "sess"
    a.db = None
    a.compaction_pipeline_enabled = True
    a.compaction_pipeline_storage_root = str(root)
    a.compaction_pipeline_map_idle_after_seconds = 0.0
    a.compaction_pipeline_map_cooldown_seconds = 0.0
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 200000
    a.compaction_pipeline_max_wait_seconds = 900.0
    a.compaction_pipeline_gate_always_on = True
    a.compaction_pipeline_loss_probe_samples = 4
    a.compaction_pipeline_models = {}
    a._compaction_models_reachable = models_reachable
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.context_compressor = SimpleNamespace(
        _compress_window=lambda msgs: window if len(msgs) > 4 else None)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    return a


def _stage_llms(dump_id: str, fail_extract: bool = False):
    """Deterministic reason/extract/gate llms (mirrors the producer harness).
    ``fail_extract`` makes Stage C fail its schema check on every attempt so
    the check exhausts retries and raises StageCheckError."""

    def reason(payload):
        return json.dumps({
            "items": [{"map_ref": "ep0", "verdict": "keep", "because": "later work",
                       "cites": [[0, 1]]}],
            "open_questions": [],
            "coverage": {"every_map_item_accounted": True},
        })

    def extract(payload):
        if fail_extract:
            return "not json at all {{{"
        ckpt = {
            "instructions_and_corrections": "null_reason: none in region",
            "decisions": [{"what": "chose jsonl", "cites": [[dump_id, 1, 2]],
                           "rejected_alternatives": []}],
            "insights": "null_reason: none in region",
            "commitments": [{"what": "ship friday", "cites": [[dump_id, 2, 3]]}],
            "open_threads": "null_reason: none in region",
            "artifacts": "null_reason: none in region",
            "world_effects": "null_reason: none in region",
            "links": "null_reason: none in region",
            "narrative": "work.",
            "confidence": 0.9,
            "coverage": {"complete": True},
        }
        return json.dumps(ckpt)

    def gate(payload):
        return json.dumps({"swap_eligible": True, "findings": []})

    return {"reason": reason, "extract": extract, "gate": gate,
            "gate_question": gate, "gate_answer": gate, "gate_grade": gate}


def _plant_map(root, covers=(0, 7)):
    from agent.compaction_map import CompactionMap
    CompactionMap(root, "sess").save({
        "schema_version": 1, "covers": {"start_msg": covers[0], "end_msg": covers[1]},
        "episodes": [{"start_msg": covers[0], "end_msg": covers[1], "name": "ep0"}],
        "entities": [], "edges": []})


# ── AC-R5b: the bypass pass completes dump -> extract -> gate -> swap ────


class TestBypassFullCycle:
    def test_falsifier_bypass_pass_swaps_in_same_pass(self, tmp_path):
        """A bypass pass ends with the region swapped IN THE SAME PASS: stage
        artifacts land, the sweep adopts the swapped list (out.pipeline_swapped,
        out.messages = the swapped list), and bypass_swapped telemetry names
        the dump. The swapped list must be byte-different from the input (the
        checkpoint row replaced the window)."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a.db = _FakeDB()
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=False)
        _plant_map(tmp_path)
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}",
                                      bypass=True)
        assert rec.get("ran") is True
        dump_id = rec.get("dumped")
        assert dump_id, f"a bypass pass must dump the current window: {rec}"
        store = DumpStore(tmp_path)
        ddir = store.dump_dir("sess", dump_id)
        assert (ddir / "stage_c.json").is_file(), \
            f"extraction must run in the SAME pass: {sorted(p.name for p in ddir.iterdir())}"
        assert (ddir / "gate.json").is_file(), "the gate must run in the same pass"
        assert rec.get("bypass_swapped") == [dump_id], rec
        swapped = rec.get("swapped_messages")
        assert swapped is not None and swapped != messages, \
            "the swap must have actually replaced the window"
        # Adopted by the turn seam: pipeline_swapped + the swapped list.
        from agent.turn_context_compaction import CompactionOutcome, _pipeline_idle_sweep
        # _pipeline_idle_sweep needs gap 0 + over-threshold to mark bypass;
        # drive it through the public entry with a tiny threshold.
        a2 = _agent(tmp_path)
        a2.db = _FakeDB()
        a2._compaction_stage_llms = _stage_llms("placeholder", fail_extract=False)
        a2.context_compressor = SimpleNamespace(
            _compress_window=lambda msgs: (0, 5) if len(msgs) > 4 else None,
            threshold_tokens=1)  # everything is over this threshold -> bypass
        out = CompactionOutcome(messages=messages, active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        _pipeline_idle_sweep(a2, out)  # idle_gap computed inside; stub _last_activity_ts
        assert out.pipeline_swapped is True, \
            f"the seam must adopt the swapped list: {out.pipeline_swapped}"
        assert out.messages is not messages and \
            any("[compaction_checkpoint]" in str(m.get("content", ""))
                for m in out.messages), "out.messages must be the swapped list"

    def test_bypass_parked_when_models_unreachable_no_raise(self, tmp_path):
        """Models unreachable -> extraction parks -> the pass returns normally
        (bypass_parked telemetry, no raise, no swapped list)."""
        messages = _messages(8)
        a = _agent(tmp_path, models_reachable=False)
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}",
                                      bypass=True)
        assert rec.get("ran") is True
        assert rec.get("bypass_parked"), rec
        assert "swapped_messages" not in rec

    def test_bypass_extract_waives_cooldown_for_that_region_only(self, tmp_path):
        """extraction.cooldown_seconds semantics preserved: a live cooldown
        blocks the queued drain / scheduled extraction but NOT the bypass
        region's same-pass extraction."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder")
        a.compaction_pipeline_extraction_cooldown_seconds = 600.0
        a._compaction_pipeline_last_extract_ts = time.time()  # cooldown live NOW
        _plant_map(tmp_path)
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}",
                                      bypass=True)
        assert rec.get("bypass_swapped"), \
            f"the bypass region must extract despite the live cooldown: {rec}"

    def test_bypass_extract_does_not_waive_session_budget(self, tmp_path):
        """The session budget is NOT waived: an exhausted budget parks the
        bypass region (budget_blocked telemetry)."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder")
        a._compaction_pipeline_spent_tokens = 500000  # over the 200000 budget
        _plant_map(tmp_path)
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}",
                                      bypass=True)
        assert rec.get("bypass_parked"), rec
        assert rec.get("budget_blocked") == "bypass_extract", rec
        assert (DumpStore(tmp_path).dump_dir("sess", rec["dumped"]) / "stage_c.json") \
            .is_file() is False


# ── AC-R5c: extraction parks -> the backstop degrades (bounded fallback) ──


class TestBypassParkedDegrades:
    def test_falsifier_stage_check_error_parks_then_backstop_degrades(self, tmp_path):
        """Force StageCheckError in extraction: the pass returns bypass_parked
        WITHOUT raising, and CompactionBackstop.decide_and_swap on the same
        messages still degrades to prose with the queue row appended."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=True)
        _plant_map(tmp_path)
        # The pass must not raise even though extraction exhausts its retries.
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}",
                                      bypass=True)
        assert rec.get("ran") is True
        assert rec.get("bypass_parked"), rec
        assert "swapped_messages" not in rec
        dump_id = rec.get("dumped")
        assert dump_id and (DumpStore(tmp_path).dump_dir("sess", dump_id)
                            / "gate.json").is_file() is False

        # The bounded fallback: the same turn's backstop degrades to prose.
        # Stamp the overflow window the live dispatch stamps before the call.
        bs_agent = _agent(tmp_path, window=(0, 5), messages_len=8)
        bs_agent.db = _FakeDB()
        bs_agent.context_compressor = SimpleNamespace(
            last_compress_window=(0, 5), _compress_window=lambda m: (0, 5))
        action, swapped, tel = CompactionBackstop(bs_agent).decide_and_swap(messages)
        assert action == "degrade" and swapped is None
        assert tel[TELEMETRY_DEGRADED] is True
        assert tel[TELEMETRY_DEGRADATION_REASON] == "extraction_incomplete"
        # The queue row appended (dump-before-degrade) for later idle drains.
        qpath = tmp_path / "sess" / "pipeline_queue.json"
        assert qpath.is_file()
        rows = json.loads(qpath.read_text())
        assert rows and rows[-1]["window"] == [0, 5]
        assert rows[-1]["reason"] == "extraction_incomplete"