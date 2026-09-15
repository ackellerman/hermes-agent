"""SPEC-0047 W5 — AC-B3/B4 falsifiers: degenerate-slice refusal + circuit breaker.

AC-B3: extracting a dump window [3, 410] with a 0..0 slice raises
StageCheckError("degenerate slice vs dump window"); the region parks; no
stage_c.json with low confidence is produced.

AC-B4: two identical extraction failures park the region durably
(``extraction.parked.json`` next to the dump); a later pass SKIPS it while the
map covers are unchanged; the park clears when ``covers.end_msg`` advances past
the dump window end.

Deterministic: injected stage llms, no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_extract import (
    RegionExtractor,
    StageCheckError,
    checkpoint_schema_check,
    dump_window_degenerate,
    run_extraction_cycle,
)
from agent.compaction_map import CompactionMap
from agent.compaction_pipeline import (
    DETERMINISTIC_FAILURE_THRESHOLD,
    IdlePipelinePass,
    _park_blocks,
    _park_path,
    _park_region,
)


def _messages(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


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
    CompactionMap(root, "sess").save({
        "schema_version": 1, "covers": {"start_msg": covers[0], "end_msg": covers[1]},
        "episodes": [{"start_msg": covers[0], "end_msg": covers[1], "name": "ep0"}],
        "entities": [], "edges": []})


def _dump(root, messages, start, end):
    store = DumpStore(root)
    ref = store.write_dump("sess", messages[start:end + 1],
                           start_msg=start, end_msg=end, turn=1)
    return store.dump_dir("sess", ref.dump_id), ref.dump_id


# ── AC-B3: degenerate slice refuses to checkpoint ─────────────────────────


class TestDumpWindowDegenerate:
    def test_slice_ending_before_dump_start_is_degenerate(self):
        assert dump_window_degenerate({"start_msg": 0, "end_msg": 2}, (3, 410))

    def test_0_0_slice_on_larger_dump_is_degenerate(self):
        reason = dump_window_degenerate({"start_msg": 0, "end_msg": 0}, (3, 410))
        assert reason and "0..0" in reason

    def test_overlapping_slice_is_not_degenerate(self):
        assert dump_window_degenerate({"start_msg": 0, "end_msg": 700}, (3, 410)) is None
        assert dump_window_degenerate({"start_msg": 5, "end_msg": 9}, (3, 410)) is None

    def test_0_0_slice_on_point_dump_is_fine(self):
        # A 0..0 slice against a 0..0 dump window is the true empty session.
        assert dump_window_degenerate({"start_msg": 0, "end_msg": 0}, (0, 0)) is None

    def test_none_window_is_not_degenerate(self):
        assert dump_window_degenerate({"start_msg": 0, "end_msg": 0}, None) is None

    def test_meta_dict_window_accepted(self):
        assert dump_window_degenerate({"start_msg": 0, "end_msg": 0},
                                      {"start_msg": 3, "end_msg": 410})


class TestACB3DegenerateSliceRefusal:
    def test_falsifier_empty_map_extraction_raises_no_stage_c(self, tmp_path):
        """AC-B3 verbatim: a dump window [3, 410] with a 0..0 (empty) slice
        raises StageCheckError('degenerate slice vs dump window'); no
        stage_c.json is produced (and no low-confidence checkpoint exists)."""
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(411)]
        ddir, dump_id = _dump(tmp_path, msgs, 3, 410)
        # No map at all -> slice() covers 0..0.
        with pytest.raises(StageCheckError) as excinfo:
            run_extraction_cycle(ddir, reason_llm=lambda p: json.dumps(
                {"items": [], "open_questions": [],
                 "coverage": {"every_map_item_accounted": True}}),
                extract_llm=_stage_llms(dump_id)["extract"])
        assert "degenerate slice vs dump window" in str(excinfo.value)
        assert (ddir / "stage_c.json").is_file() is False
        assert (ddir / "stage_b.json").is_file() is False

    def test_slice_ending_before_dump_window_refuses(self, tmp_path):
        """A map that lags the dump (covers.end < dump.start) refuses too."""
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        ddir, dump_id = _dump(tmp_path, msgs, 40, 100)
        _plant_map(tmp_path, covers=(0, 39))  # map ends before the dump starts
        with pytest.raises(StageCheckError) as excinfo:
            run_extraction_cycle(ddir, reason_llm=lambda p: json.dumps(
                {"items": [], "open_questions": [],
                 "coverage": {"every_map_item_accounted": True}}),
                extract_llm=_stage_llms(dump_id)["extract"])
        assert "degenerate slice vs dump window" in str(excinfo.value)
        assert (ddir / "stage_c.json").is_file() is False

    def test_caught_up_map_extracts_normally(self, tmp_path):
        """The catch-up path: once the map covers the dump window, the same
        cycle extracts."""
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        ddir, dump_id = _dump(tmp_path, msgs, 0, 10)
        _plant_map(tmp_path, covers=(0, 15))
        stage_c = run_extraction_cycle(
            ddir, reason_llm=_stage_llms(dump_id)["reason"],
            extract_llm=_stage_llms(dump_id)["extract"])
        assert stage_c["confidence"] == 0.9
        assert (ddir / "stage_c.json").is_file()

    def test_checkpoint_schema_check_rejects_degenerate_bounds(self):
        """checkpoint_schema_check (with slice + dump window) refuses a
        checkpoint whose slice bounds are degenerate vs the dump window."""
        ckpt = {
            "instructions_and_corrections": "null_reason: none in region",
            "decisions": "null_reason: none in region",
            "insights": "null_reason: none in region",
            "commitments": "null_reason: none in region",
            "open_threads": "null_reason: none in region",
            "artifacts": "null_reason: none in region",
            "world_effects": "null_reason: none in region",
            "links": "null_reason: none in region",
            "narrative": "Degenerate empty region.",
            "confidence": 0.2,
            "coverage": {"complete": True},
        }
        errors = checkpoint_schema_check(ckpt,
                                         slice_covers={"start_msg": 0, "end_msg": 0},
                                         dump_window=(3, 410))
        assert any("degenerate slice vs dump window" in e for e in errors)
        # The same checkpoint without bounds context still passes the sections.
        assert checkpoint_schema_check(ckpt) == []

    def test_extractor_stage_c_rejects_degenerate_bounds(self, tmp_path):
        """RegionExtractor.stage_c threads the refusal through its check."""
        store = DumpStore(tmp_path)
        ref = store.write_dump("sess", [{"role": "user", "content": "m"}],
                               start_msg=3, end_msg=410, turn=1)
        ex = RegionExtractor(tmp_path, "sess", ref.dump_id)
        llms = _stage_llms(ref.dump_id)
        with pytest.raises(StageCheckError):
            ex.stage_c(llms["extract"], {"items": []},
                       slice_covers={"start_msg": 0, "end_msg": 0},
                       dump_window=(3, 410))
        assert (ex.dir / "stage_c.json").is_file() is False


# ── AC-B4: deterministic-failure circuit breaker ──────────────────────────


class TestParkMarker:
    def test_park_increments_identical_failures(self, tmp_path):
        d = tmp_path / "0001-abc"
        d.mkdir()
        first = _park_region(d, reason="boom", exc=ValueError("invalid literal for int() with base 10: 'd'"),
                             covers_end=700, dump_end=600)
        second = _park_region(d, reason="boom", exc=ValueError("invalid literal for int() with base 10: 'd'"),
                              covers_end=700, dump_end=600)
        assert first["failures"] == 1 and second["failures"] == 2
        park = json.loads(_park_path(d).read_text(encoding="utf-8"))
        assert park["parked"] is True
        assert park["last_error"] == "invalid literal for int() with base 10: 'd'"
        assert park["dump_end"] == 600

    def test_changed_signature_resets_count(self, tmp_path):
        d = tmp_path / "0001-abc"
        d.mkdir()
        _park_region(d, reason="a", exc=ValueError("x"), dump_end=10)
        second = _park_region(d, reason="b", exc=ValueError("y"), dump_end=10)
        assert second["failures"] == 1

    def test_park_blocks_until_covers_advance(self, tmp_path):
        d = tmp_path / "0001-abc"
        d.mkdir()
        _park_region(d, reason="r", exc=ValueError("x"), covers_end=5, dump_end=600)
        _park_region(d, reason="r", exc=ValueError("x"), covers_end=5, dump_end=600)
        assert _park_blocks(d, 5) is True       # at threshold, covers not advanced
        assert _park_blocks(d, 599) is True     # still short of the dump end
        assert _park_blocks(d, 600) is False    # covers advanced past dump end
        assert _park_blocks(d, 700) is False

    def test_corrupt_or_absent_park_does_not_block(self, tmp_path):
        d = tmp_path / "0001-abc"
        d.mkdir()
        assert _park_blocks(d, 0) is False      # absent
        _park_path(d).write_text("{not json", encoding="utf-8")
        assert _park_blocks(d, 0) is False      # corrupt fails open


class TestACB4CircuitBreaker:
    def test_falsifier_two_identical_failures_park_durably_then_skip(self, tmp_path):
        """AC-B4 verbatim: two identical extraction failures park the region
        durably; a later pass skips it while the map covers are unchanged; the
        park clears when covers.end_msg advances past the dump end."""
        msgs = _messages(8)
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=True)
        # The map must NOT yet cover the dump window end (0..5), or the D4
        # unblock condition is already satisfied and the breaker never arms.
        _plant_map(tmp_path, covers=(0, 4))

        rec1 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}",
                                       bypass=True)
        dump_id = rec1.get("dumped")
        assert dump_id, rec1
        ddir = DumpStore(tmp_path).dump_dir("sess", dump_id)
        # Failure 1: a park marker exists with failures=1.
        park = json.loads(_park_path(ddir).read_text(encoding="utf-8"))
        assert park["failures"] == 1

        rec2 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}",
                                       bypass=True)
        assert rec2.get("dumped") == dump_id  # idempotent re-dump
        park = json.loads(_park_path(ddir).read_text(encoding="utf-8"))
        assert park["failures"] == 2
        assert park["last_error"] == "stage output is not JSON: 'not json at all {{{'"

        # Pass 3: the breaker SKIPS extraction (no third attempt; no new LLM
        # spend — the extract llm would fail identically if it were called).
        calls = {"n": 0}

        def counting_extract(payload):
            calls["n"] += 1
            return "not json {{{"

        a._compaction_stage_llms = dict(_stage_llms(dump_id), extract=counting_extract)
        rec3 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}",
                                       bypass=True)
        assert calls["n"] == 0, "the parked region must be skipped, not retried"
        skipped = rec3.get("parked_regions") or []
        assert any(s.get("dump_id") == dump_id for s in skipped), rec3
        park = json.loads(_park_path(ddir).read_text(encoding="utf-8"))
        assert park["failures"] == 2, "the skip must not grow the failure count"

        # The map catches up past the dump window end -> the park clears and
        # extraction retries.
        _plant_map(tmp_path, covers=(0, 100))  # dump window is 0..5
        rec4 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}",
                                       bypass=True)
        assert calls["n"] >= 1, "the park must clear once covers advance"
        # The retried extraction fails again (fail_extract stub) — but the
        # breaker laddered from a clean slate (failures reset to 1).
        park = json.loads(_park_path(ddir).read_text(encoding="utf-8"))
        assert park["failures"] == 1, park

    def test_park_marker_lives_next_to_dump(self, tmp_path):
        """D4: the marker is durable, NEXT TO THE DUMP (<dump_dir>/), and
        survives process restarts (it is just a file we re-read)."""
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=True)
        _plant_map(tmp_path, covers=(0, 4))  # breaker must be able to arm
        rec = IdlePipelinePass(a).run(_messages(8), llm_call=lambda _p: "{}",
                                      bypass=True)
        ddir = DumpStore(tmp_path).dump_dir("sess", rec["dumped"])
        assert _park_path(ddir).is_file()
        assert _park_path(ddir).parent == ddir

    def test_breaker_does_not_fire_before_threshold(self, tmp_path):
        """One failure only: no durable park yet (transient failures still
        retry on the next pass); extraction is attempted again."""
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=True)
        _plant_map(tmp_path, covers=(0, 4))  # breaker must be able to arm
        rec1 = IdlePipelinePass(a).run(_messages(8), llm_call=lambda _p: "{}",
                                       bypass=True)
        ddir = DumpStore(tmp_path).dump_dir("sess", rec1["dumped"])
        assert json.loads(_park_path(ddir).read_text())["failures"] == 1
        # A pass while the marker sits below the threshold still ATTEMPTS:
        # the marker alone must not block (only an at-threshold park does).
        calls = {"n": 0}

        def counting_extract(payload):
            calls["n"] += 1
            return "not json {{{"

        a._compaction_stage_llms = dict(_stage_llms(rec1["dumped"]),
                                        extract=counting_extract)
        IdlePipelinePass(a).run(_messages(8), llm_call=lambda _p: "{}", bypass=True)
        assert calls["n"] == 1

    def test_success_clears_the_park(self, tmp_path):
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms("placeholder", fail_extract=True)
        _plant_map(tmp_path, covers=(0, 4))  # breaker must be able to arm
        rec1 = IdlePipelinePass(a).run(_messages(8), llm_call=lambda _p: "{}",
                                       bypass=True)
        ddir = DumpStore(tmp_path).dump_dir("sess", rec1["dumped"])
        assert _park_path(ddir).is_file()
        # Now let extraction succeed (and the map cover the dump end).
        a._compaction_stage_llms = _stage_llms(rec1["dumped"])
        _plant_map(tmp_path, covers=(0, 7))
        rec2 = IdlePipelinePass(a).run(_messages(8), llm_call=lambda _p: "{}",
                                       bypass=True)
        assert rec2.get("bypass_swapped") == [rec1["dumped"]], rec2
        assert _park_path(ddir).is_file() is False

    def test_threshold_is_two(self):
        assert DETERMINISTIC_FAILURE_THRESHOLD == 2

    def test_identical_type_and_message_required(self, tmp_path):
        """Only an IDENTICAL type+message pair ladders; a new signature resets."""
        d = tmp_path / "d"
        d.mkdir()
        _park_region(d, reason="r", exc=ValueError("same"), dump_end=5)
        marker = _park_region(d, reason="r", exc=RuntimeError("same"), dump_end=5)
        assert marker["failures"] == 1  # type changed -> reset
        marker = _park_region(d, reason="r", exc=RuntimeError("same"), dump_end=5)
        assert marker["failures"] == 2  # same signature -> ladder