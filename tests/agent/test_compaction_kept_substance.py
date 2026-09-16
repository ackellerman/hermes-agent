"""SPEC-0049 falsifiers: kept_substance schema (AC-D2) + kept-scoped loss
probe question generation (AC-D3), run against the REAL rig artifacts where
possible (AC-D1's real-lane check runs offline via /tmp/gate_probe.py)."""

import json
from pathlib import Path

import pytest

from agent.compaction_extract import (
    CHECKPOINT_SECTIONS,
    MAX_SUBSTANCE_TOKENS_PER_ENTRY,
    checkpoint_schema_check,
)
from agent.compaction_verify import generate_loss_probe_questions, _kept_scope_rows


def _ckpt():
    did = "d-x"
    base = {s: f"null_reason: none" for s in CHECKPOINT_SECTIONS
            if s not in ("confidence", "coverage", "narrative",
                         "instructions_and_corrections")}
    ckpt = {
        "kept_substance": [{"ref": "ep-1", "substance": "chose jsonl", "cites": [[did, 1, 2]]}],
        "instructions_and_corrections": "null_reason: none",
        "narrative": "glue",
        "confidence": 0.8,
        "coverage": {"complete": True},
    }
    ckpt.update(base)
    return ckpt


class TestACD2KeptSubstanceSchema:
    def test_valid_checkpoint_with_kept_substance_passes(self):
        assert checkpoint_schema_check(_ckpt()) == []

    def test_falsifier_missing_kept_substance_rejected(self):
        broken = _ckpt()
        del broken["kept_substance"]
        errors = checkpoint_schema_check(broken)
        assert any("kept_substance" in e for e in errors), errors

    def test_falsifier_entry_without_substance_rejected(self):
        broken = _ckpt()
        broken["kept_substance"] = [{"ref": "ep-1", "cites": [[0, 1, 2]]}]
        errors = checkpoint_schema_check(broken)
        assert any("missing 'substance'" in e for e in errors), errors

    def test_falsifier_entry_without_ref_rejected(self):
        broken = _ckpt()
        broken["kept_substance"] = [{"substance": "facts", "cites": [[0, 1, 2]]}]
        errors = checkpoint_schema_check(broken)
        assert any("missing 'ref'" in e for e in errors), errors

    def test_falsifier_oversize_entry_rejected(self):
        broken = _ckpt()
        broken["kept_substance"] = [
            {"ref": "ep-1", "cites": [[0, 0, 1]],
             "substance": "word " * (MAX_SUBSTANCE_TOKENS_PER_ENTRY + 10)}]
        errors = checkpoint_schema_check(broken)
        assert any(f"> {MAX_SUBSTANCE_TOKENS_PER_ENTRY} tokens" in e for e in errors), errors

    def test_null_reason_kept_substance_passes(self):
        ckpt = _ckpt()
        ckpt["kept_substance"] = "null_reason: no keep verdicts in region"
        assert checkpoint_schema_check(ckpt) == []


class TestACD3KeptScoping:
    VERDICTS = [
        {"map_ref": "ep-keep", "verdict": "keep", "cites": [[0, 0, 1]]},
        {"map_ref": "ep-drop", "verdict": "drop", "cites": [[0, 2, 3]]},
    ]
    DUMP = [
        {"role": "assistant", "content": "KEEPZONE the prefetch knob is 10"},
        {"role": "user", "content": "KEEPZONE acknowledged"},
        {"role": "assistant", "content": "DROPZONE the abandoned essay on orchestration"},
        {"role": "user", "content": "DROPZONE more abandoned content"},
    ]

    def test_falsifier_kept_scope_rows_exclude_drop_ranges(self):
        rows = _kept_scope_rows(self.VERDICTS, self.DUMP)
        assert rows is not None
        text = json.dumps(rows)
        assert "DROPZONE" not in text, "dropped content leaked into kept scope"
        assert "KEEPZONE" in text

    def test_question_generation_prompt_contains_only_kept_content(self):
        captured = {}

        def qllm(payload):
            captured["payload"] = json.dumps(payload)
            return json.dumps({"questions": ["what is the prefetch knob?"]})

        questions = generate_loss_probe_questions(
            qllm, self.DUMP, samples=4, seed=0, stage_b_verdicts=self.VERDICTS)
        assert questions == ["what is the prefetch knob?"]
        assert "DROPZONE" not in captured["payload"], \
            "the question generator must never see dropped content (D3)"
        assert "KEEPZONE" in captured["payload"]

    def test_no_kept_verdicts_falls_back_to_full_dump(self):
        captured = {}

        def qllm(payload):
            captured["payload"] = json.dumps(payload)
            return json.dumps({"questions": ["q"]})

        generate_loss_probe_questions(
            qllm, self.DUMP, samples=1, seed=0,
            stage_b_verdicts=[{"map_ref": "ep", "verdict": "drop", "cites": [[0, 0, 3]]}])
        assert "DROPZONE" in captured["payload"], \
            "no keep cites -> full-dump fallback (never questions about nothing)"

    def test_kept_scope_rows_none_when_no_usable_cites(self):
        assert _kept_scope_rows([], self.DUMP) is None
        assert _kept_scope_rows(
            [{"map_ref": "e", "verdict": "keep", "cites": [["bad"]]}], self.DUMP) is None