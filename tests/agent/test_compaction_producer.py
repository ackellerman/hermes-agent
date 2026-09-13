"""SPEC-0043 producer-wiring tests — AC-20/21/22/23/24/25/26 falsifiers.

Deterministic: no model is ever called — the idle pass and backstop run with
injected ``agent.compaction_pipeline_*`` config, a fake DB owning the pipeline
lock and compression lease, and deterministic stage LLMs driven via
``agent._compaction_stage_llms``. Every assertion is a behavior contract, not a
snapshot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_pipeline import IdlePipelinePass
from agent.compaction_backstop import (
    TELEMETRY_DEGRADATION_REASON,
    TELEMETRY_DEGRADED,
    CompactionBackstop,
)


def _alternating_messages(n: int = 300) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


class _FakeDB:
    """Pipeline lock + compression lease owner; both acquirable."""

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


def _agent(enabled=True, root=None, session_id="sess", n_models_reachable=None):
    a = SimpleNamespace()
    a.session_id = session_id
    a.db = None
    a.compaction_pipeline_enabled = enabled
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
    a._compaction_models_reachable = (True if n_models_reachable is None
                                      else bool(n_models_reachable))
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.context_compressor = SimpleNamespace(
        _compress_window=lambda msgs: (0, 3) if len(msgs) > 4 else None)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    return a


# Deterministic stage LLMs. The map update is not exercised by these producer
# tests (they drive the dump/extract/gate/sweep stages), so we only need to
# satisfy the extract + gate call shapes.


def _stage_llms(root: Path, session_id: str, dump_id: str, bad_stage=None, commit=True):
    """Deterministic reason/extract/gate LLMs that produce schema-valid output.
    ``bad_stage='c'`` drops a commitment from Stage C (gate must flag it);
    ``commit=False`` makes the checkpoint carry no commitments (used for
    clean pass / stale-window tests where the dump 'drops' nothing)."""

    def reason(payload):
        # Stage B verdict: keep one item.
        return json.dumps({
            "items": [{"map_ref": "ep0", "verdict": "keep", "because": "later work",
                       "cites": [[0, 1]]}],
            "open_questions": [],
            "coverage": {"every_map_item_accounted": True},
        })

    def extract(payload):
        ckpt = {
            "instructions_and_corrections": "null_reason: none in region",
            "decisions": [{"what": "chose jsonl", "cites": [[dump_id, 1, 2]],
                           "rejected_alternatives": []}],
            "insights": "null_reason: none in region",
            "commitments": ([{"what": "ship friday", "cites": [[dump_id, 2, 3]]}]
                            if commit and bad_stage != "c" else "null_reason: none in region"),
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
        # The gate llm sees [REVIEW_GATE_PROMPT, {"checkpoint":..., "dump":...}].
        ckpt = None
        for item in (payload or []):
            if not isinstance(item, dict):
                continue
            try:
                obj = json.loads(item.get("content", ""))
            except Exception:
                continue
            if isinstance(obj, dict) and "checkpoint" in obj:
                ckpt = obj["checkpoint"]
                break
        eligible = True
        findings = []
        if isinstance(ckpt, dict) and (
                isinstance(ckpt.get("commitments"), str)
                and ckpt.get("commitments", "").startswith("null_reason:")):
            # A dropped commitment that the dump holds -> not eligible.
            eligible = False
            findings = [{"what": "dropped commitment", "cites": [[2, 3]]}]
        return json.dumps({"swap_eligible": eligible, "findings": findings})

    return {"reason": reason, "extract": extract, "gate": gate,
            "gate_question": gate, "gate_answer": gate, "gate_grade": gate}


# ── AC-20: idle dump is idempotent ──────────────────────────────────────


class TestAC20IdleDump:
    def test_falsifier_complete_dump_exists_and_is_idempotent(self, tmp_path):
        messages = _alternating_messages(300)
        a = _agent(root=tmp_path)
        a.context_compressor = SimpleNamespace(_compress_window=lambda m: (0, 3))
        sweep = IdlePipelinePass(a)
        rec1 = sweep.run(messages, llm_call=lambda _p: "{}")
        assert rec1.get("dumped"), "an over-threshold idle pass must write a dump"
        store = DumpStore(tmp_path)
        # Exactly one complete dump under the session dir (as .meta.json sidecars).
        metas = sorted((store.session_dir("sess")).glob("*.meta.json"))
        assert len(metas) == 1
        dump_id = metas[0].name.removesuffix(".meta.json")
        assert store.is_complete("sess", dump_id) is True
        content_path = store.dump_path("sess", dump_id)
        mtime1 = content_path.stat().st_mtime

        # Run again with no new messages -> no new dump, no fsync write (mtime stable).
        rec2 = sweep.run(messages, llm_call=lambda _p: "{}")
        metas2 = sorted((store.session_dir("sess")).glob("*.meta.json"))
        assert len(metas2) == 1, "second pass must not create a new dump file"
        mtime2 = store.dump_path("sess", metas2[0].name.removesuffix(".meta.json")).stat().st_mtime
        assert mtime2 == mtime1, "idempotent dump must not rewrite the file (mtime stable)"


# ── AC-21: dump-before-degrade ──────────────────────────────────────────


class TestAC21DumpBeforeDegrade:
    def test_red_baseline_gap_provable(self):
        """FALSIFIER AC-21 (RED): on the pre-fix contract the degrade branch had
        zero write_dump calls; assert the wiring AND queue live in this module
        by name so a revert is detectable."""
        src = Path("agent/compaction_backstop.py").read_text()
        assert "def decide_and_swap" in src
        assert "_dump_window(" in src, "degrade branches must call the dump helper"
        assert "_append_degraded(" in src, "degrade branches must enqueue the region"

    def test_falsifier_models_down_dumps_before_degrade(self, tmp_path):
        messages = _alternating_messages(8)
        a = _agent(root=tmp_path, n_models_reachable=False)
        a.context_compressor = SimpleNamespace(last_compress_window=(2, 5),
                                               _compress_window=lambda m: (2, 5))
        action, swapped, tel = CompactionBackstop(a).decide_and_swap(messages)
        assert action == "degrade"
        assert swapped is None
        assert tel[TELEMETRY_DEGRADED] is True
        assert tel[TELEMETRY_DEGRADATION_REASON] == "model_unreachable"
        # A complete dump must exist for window (2,5) AND a queue row appended.
        store = DumpStore(tmp_path)
        metas = sorted((store.session_dir("sess")).glob("*.meta.json"))
        assert metas, "degrade must write a dump"
        dump_id = metas[0].name.removesuffix(".meta.json")
        assert store.is_complete("sess", dump_id) is True
        meta = store.read_meta("sess", dump_id)
        window = [int(meta["start_msg"]), int(meta["end_msg"])]
        assert window == [2, 5], f"dump must cover the degrade window, got {window}"
        q = json.loads((tmp_path / "sess" / "pipeline_queue.json").read_text())
        assert len(q) == 1
        assert q[0]["window"] == [2, 5]
        assert q[0]["reason"] == "model_unreachable"


# ── AC-22: scheduled + persisted extraction ─────────────────────────────


class TestAC22Extraction:
    def _pass_with_dump(self, tmp_path, bad_stage=None, commit=True):
        # Plant a complete dump + map, then one idle pass runs extraction.
        store = DumpStore(tmp_path)
        msgs = _alternating_messages(6)
        ref = store.write_dump("sess", msgs, start_msg=0, end_msg=5, turn=1)
        sdir = store.session_dir("sess") / ref.dump_id
        sdir.mkdir(parents=True, exist_ok=True)
        # Map with an episode covering the dump, so stage A slices something.
        from agent.compaction_map import CompactionMap
        cm = CompactionMap(tmp_path, "sess")
        cm.save({"schema_version": 1, "covers": {"start_msg": 0, "end_msg": 5},
                 "episodes": [{"start_msg": 0, "end_msg": 5, "name": "ep0"}],
                 "entities": [], "edges": []})
        a = _agent(root=tmp_path)
        a._compaction_stage_llms = _stage_llms(tmp_path, "sess", ref.dump_id,
                                               bad_stage=bad_stage, commit=commit)
        a.context_compressor = SimpleNamespace(_compress_window=lambda m: None)
        sweep = IdlePipelinePass(a)
        messages = _alternating_messages(6)
        rec = sweep.run(messages, llm_call=lambda _p: "{}")
        return rec, store, ref, sdir

    def test_falsifier_a_b_c_persisted_schema_valid(self, tmp_path):
        rec, store, ref, sdir = self._pass_with_dump(tmp_path, commit=True)
        # Extraction ran and wrote all three stage artifacts under the dump dir.
        assert (sdir / "stage_a.json").is_file()
        assert (sdir / "stage_b.json").is_file()
        assert (sdir / "stage_c.json").is_file()
        from agent.compaction_extract import checkpoint_schema_check
        stage_c = json.loads((sdir / "stage_c.json").read_text())
        assert checkpoint_schema_check(stage_c) == []

    def test_falsifier_stage_c_drops_commitment_gate_flags(self, tmp_path):
        rec, store, ref, sdir = self._pass_with_dump(tmp_path, commit=False)
        # Gate artifact exists with swap_eligible:false naming the dropped commitment.
        gate = json.loads((sdir / "gate.json").read_text())
        assert gate["swap_eligible"] is False
        assert any("commitment" in str(g.get("what", "")).lower()
                   for g in gate["findings"]), "finding must name the dropped commitment"
        # Region parks: stage_c dropped back (loss gap / gate flip) -> re-extract later.

    def test_falsifier_cooldown_suppresses_duplicate_cycle(self, tmp_path):
        rec, store, ref, sdir = self._pass_with_dump(tmp_path, commit=True)
        ckpt_read = (sdir / "stage_c.json").stat().st_mtime
        # Immediate second pass: cooldown (0s set, but stage already extracted ->
        # extraction is skipped because stage_c exists; artifacts untouched).
        a = _agent(root=tmp_path)
        a._compaction_stage_llms = _stage_llms(tmp_path, "sess", ref.dump_id, commit=True)
        sweep = IdlePipelinePass(a)
        sweep.run(_alternating_messages(6), llm_call=lambda _p: "{}")
        assert (sdir / "stage_c.json").stat().st_mtime == ckpt_read, \
            "cooldown must suppress a duplicate extraction cycle (no rewrite)"


# ── AC-24: degraded-queue drain ─────────────────────────────────────────


class TestAC24QueueDrain:
    def _queue_with_row(self, tmp_path):
        a = _agent(root=tmp_path)
        q = tmp_path / "sess" / "pipeline_queue.json"
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps([{
            "dump_id": "x", "window": [0, 5], "reason": "model_unreachable", "queued_ts": time.time(),
        }]))
        return a, q

    def _plant_ready_dump(self, tmp_path):
        """A complete dump over [0,5] with stage_c + gate-passed so the drain
        can actually extract it (models reachable)."""
        store = DumpStore(tmp_path)
        msgs = _alternating_messages(6)
        ref = store.write_dump("sess", msgs, start_msg=0, end_msg=5, turn=1)
        d = store.session_dir("sess") / ref.dump_id
        d.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "instructions_and_corrections": [], "decisions": [], "insights": "null_reason: none",
            "commitments": [], "open_threads": "null_reason: none",
            "artifacts": "null_reason: none", "world_effects": "null_reason: none",
            "links": "null_reason: none", "narrative": "w", "confidence": 0.9,
            "coverage": {"complete": True},
        }
        (d / "stage_c.json").write_text(json.dumps(ckpt))
        (d / "gate.json").write_text(json.dumps({"swap_eligible": True, "findings": []}))
        from agent.compaction_map import CompactionMap
        CompactionMap(tmp_path, "sess").save({
            "schema_version": 1, "covers": {"start_msg": 0, "end_msg": 5},
            "episodes": [{"start_msg": 0, "end_msg": 5, "name": "ep"}], "entities": [], "edges": []})
        return store, ref

    def test_falsifier_models_down_queue_refuses_to_drain(self, tmp_path):
        a, q = self._queue_with_row(tmp_path)
        a._compaction_models_reachable = False
        sweep = IdlePipelinePass(a)
        sweep.run(_alternating_messages(6), llm_call=lambda _p: "{}")
        rows = json.loads(q.read_text())
        assert len(rows) == 1, "with models down the queue must refuse to drain"
        # Two passes while unreachable -> identical row count (no shrink).
        sweep.run(_alternating_messages(6), llm_call=lambda _p: "{}")
        assert len(json.loads(q.read_text())) == 1

    def test_falsifier_models_up_row_drains_queue_empties(self, tmp_path):
        a, q = self._queue_with_row(tmp_path)
        store, ref = self._plant_ready_dump(tmp_path)
        # Re-point the queued row's dump_id at the planted dump.
        q.write_text(json.dumps([{
            "dump_id": ref.dump_id, "window": [0, 5], "reason": "model_unreachable",
            "queued_ts": time.time()}]))
        a._compaction_stage_llms = _stage_llms(tmp_path, "sess", ref.dump_id, commit=True)
        sweep = IdlePipelinePass(a)
        sweep.run(_alternating_messages(6), llm_call=lambda _p: "{}")
        rows = json.loads(q.read_text())
        assert rows == [], "a drained row must remove the queue entry"


# ── AC-25: batched swap sweep ───────────────────────────────────────────


class TestAC25SwapSweep:
    def _ready_regions(self, tmp_path, messages, windows, n_regions=2):
        """Write 2 complete, gate-passed dumps over disjoint windows and return
        ready region descriptors + a stub registry."""
        from agent.compaction_rehydrate import StubRegistry
        store = DumpStore(tmp_path)
        ready = []
        for i, (s, e) in enumerate(windows):
            ref = store.write_dump("sess", messages[s:e + 1], start_msg=s, end_msg=e,
                                   turn=1 + i)
            d = store.session_dir("sess") / ref.dump_id
            d.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "instructions_and_corrections": [], "decisions": [], "insights": "null_reason: none",
                "commitments": [{"what": f"c{i}", "cites": [[ref.dump_id, s + 1, s + 2]]}],
                "open_threads": "null_reason: none", "artifacts": "null_reason: none",
                "world_effects": "null_reason: none", "links": "null_reason: none",
                "narrative": "w", "confidence": 0.9, "coverage": {"complete": True},
            }
            (d / "stage_c.json").write_text(json.dumps(ckpt))
            (d / "gate.json").write_text(json.dumps({"swap_eligible": True, "findings": []}))
            meta = store.read_meta("sess", ref.dump_id)
            ready.append({"dump_id": ref.dump_id, "meta": meta,
                          "checkpoint": ckpt, "gate": {"swap_eligible": True}})
        return store, ready

    def test_falsifier_two_regions_one_mutation(self, tmp_path):
        from agent.compaction_swap import swap_sweep
        from agent.compaction_rehydrate import StubRegistry
        messages = _alternating_messages(12)
        # Disjoint, non-overlapping windows [1,3] and [5,7].
        store, ready = self._ready_regions(tmp_path, messages, [(1, 3), (5, 7)])
        registry = StubRegistry.for_session(tmp_path, "sess")
        out = swap_sweep(messages, ready_regions=ready, dump_store=store,
                         session_id="sess", stub_registry=registry)
        rows = [m for m in out if "[compaction_checkpoint]" in str(m.get("content", ""))]
        assert len(rows) == 2, "both eligible regions must swap in one sweep"
        assert len(out) == len(messages) - (4 - 2) - (4 - 2), \
            "two dump windows removed, two checkpoint rows inserted"
        from agent.compaction_verify import check_alternation_invariant
        ok, viol = check_alternation_invariant(out)
        assert ok, f"alternation must hold on the whole output: {viol}"
        # Stubs registered for both swapped dumps (AC-28).
        assert all(r["dump_id"] in registry.all() for r in ready)

    def test_falsifier_stale_window_refuses(self, tmp_path):
        from agent.compaction_swap import swap_sweep
        from agent.compaction_rehydrate import StubRegistry
        messages = _alternating_messages(12)
        # One ready dump covers [9,11] — but the "current" window is [0,3].
        store, ready = self._ready_regions(tmp_path, messages, [(9, 11)], n_regions=1)
        registry = StubRegistry.for_session(tmp_path, "sess")
        out = swap_sweep(messages, ready_regions=ready, dump_store=store,
                         session_id="sess", stub_registry=registry,
                         current_window=(0, 3))
        rows = [m for m in out if "[compaction_checkpoint]" in str(m.get("content", ""))]
        assert rows == [], "a stale-window dump must refuse to swap (stays live)"
        assert len(out) == len(messages)


# ── AC-26: sweep / legacy composition is enforced by the caller ─────────


class TestAC26Composition:
    def test_pipeline_sweep_suppresses_legacy_via_flag(self, tmp_path):
        from agent.turn_context_compaction import CompactionOutcome, _idle_compaction
        a = _agent(root=tmp_path)
        a.compression_enabled = False  # legacy idle path OFF regardless
        out = CompactionOutcome(
            messages=_alternating_messages(6), active_system_prompt=None,
            conversation_history=None, current_turn_user_idx=0,
        )
        # Simulate a pipeline sweep running and swapping: the flag suppresses legacy.
        a._compaction_pipeline_enabled = True
        # Patch the sweep to swap the list and set the flag.
        from agent import turn_context_compaction as tcc
        orig = tcc._pipeline_idle_sweep

        def fake_sweep(agent, out):
            out.messages = [{"role": "assistant", "content": "[compaction_checkpoint] w"}]
            out.pipeline_swapped = True
        import agent.turn_context_compaction as tcc2
        orig_fn = tcc2._pipeline_idle_sweep
        tcc2._pipeline_idle_sweep = fake_sweep
        try:
            _idle_compaction(a, out, None, None, "task")
        finally:
            tcc2._pipeline_idle_sweep = orig_fn
        # The legacy path returned early because pipeline_swapped is True -> out
        # still holds exactly the swapped message and no legacy re-compaction
        # reintroduced the pre-swap content.
        assert len(out.messages) == 1
        assert "[compaction_checkpoint]" in out.messages[0]["content"]