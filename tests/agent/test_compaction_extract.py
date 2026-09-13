"""SPEC-0042 extraction stage tests — AC-6/7/8/9 falsifiers."""

import json

import pytest

from agent.compaction_extract import (
    CHECKPOINT_SECTIONS,
    RegionExtractor,
    checkpoint_schema_check,
    item_citation_supported,
    parse_stage_json,
)


def _valid_checkpoint(dump_id="d-abc"):
    item = {"what": "use jsonl format", "cites": [[dump_id, 3, 5]]}
    return {
        "instructions_and_corrections": [dict(item, kind="correction")],
        "decisions": [dict(item, rationale="reason", rejected=["alt"])],
        "insights": [item],
        "commitments": [dict(item, who="user")],
        "open_threads": [item],
        "artifacts": [dict(item, connected_to_work="w", recoverable=True)],
        "world_effects": [item],
        "links": [item],
        "narrative": "Orientation glue.",
        "confidence": 0.9,
        "coverage": {"complete": True},
    }


class TestAC6TemplateConformance:
    def test_valid_checkpoint_passes(self):
        assert checkpoint_schema_check(_valid_checkpoint()) == []

    def test_falsifier_delete_a_section_validator_rejects(self):
        """FALSIFIER AC-6 verbatim: delete a section from a valid checkpoint ->
        validator must reject (non-empty error list)."""
        for section in CHECKPOINT_SECTIONS:
            broken = _valid_checkpoint()
            del broken[section]
            errors = checkpoint_schema_check(broken)
            assert any(section in e for e in errors), f"validator missed deleted {section}"

    def test_empty_section_without_null_reason_rejects(self):
        broken = _valid_checkpoint()
        broken["decisions"] = []
        errors = checkpoint_schema_check(broken)
        assert any("decisions" in e for e in errors)

    def test_empty_section_with_null_reason_passes(self):
        ok = _valid_checkpoint()
        ok["decisions"] = {"null_reason": "no decisions in region"}
        assert checkpoint_schema_check(ok) == []


class TestAC7CitationsResolve:
    def test_citation_resolves_to_real_dump_range(self, tmp_path):
        dump_msgs = [{"content": f"m{i}"} for i in range(10)]
        ckpt = _valid_checkpoint()
        assert all(item_citation_supported(i, lambda d: dump_msgs) for i in ckpt["insights"])

    def test_falsifier_plant_unsupported_item_gate_rejects(self):
        """FALSIFIER AC-7: plant an unsupported item (cite outside the dump)
        -> must reject."""
        dump_msgs = [{"content": f"m{i}"} for i in range(10)]
        bad = {"what": "phantom", "cites": [["d-abc", 50, 80]]}
        assert not item_citation_supported(bad, lambda d: dump_msgs)


class TestAC8CorrectionInvariant:
    """AC-8 is eval-run territory on labeled fixtures (spec §4 item 5); the
    script-enforceable half — every kept correction must carry a follow-up
    citation and corrections must appear in the checkpoint — is tested here."""

    def test_kept_correction_requires_followup_cite(self):
        ckpt = _valid_checkpoint()
        corr = ckpt["instructions_and_corrections"][0]
        assert corr.get("cites"), "kept correction missing follow-up citation"


class TestAC9NoVerbatimRule:
    def test_item_quoting_over_50_tokens_fails(self):
        broken = _valid_checkpoint()
        broken["insights"] = [
            {"what": "quote", "cites": [["d", 0, 1]],
             "text": " ".join(f"w{i}" for i in range(60))}
        ]
        errors = checkpoint_schema_check(broken)
        assert any("50 tokens" in e for e in errors)


class TestStageFlow:
    def test_stage_a_is_mechanical(self, tmp_path):
        ex = RegionExtractor(tmp_path, "sess", "dump1")
        art = ex.stage_a({"complete": True, "episodes": []})
        assert art["stage"] == "a"
        assert ex.load_artifact("a") == art

    def test_stage_b_runs_checks_and_parks_after_retries(self, tmp_path):
        """A stage that never validates parks the region (StageCheckError),
        never silently succeeds."""
        ex = RegionExtractor(tmp_path, "sess", "dump2", max_stage_retries=1)
        bad = {"items": [{"verdict": "maybe"}]}  # schema-invalid, always
        with pytest.raises(Exception):
            ex.stage_b(lambda msgs: json.dumps(bad), {"episodes": []})

    def test_stage_b_valid_output_persists_artifact(self, tmp_path):
        ex = RegionExtractor(tmp_path, "sess", "dump3")
        good = {"items": [{"map_ref": "e1", "verdict": "keep", "because": "dep",
                           "cites": [[0, 5]]}],
                "open_questions": [],
                "coverage": {"every_map_item_accounted": True}}
        out = ex.stage_b(lambda msgs: json.dumps(good), {"episodes": []})
        assert out == good
        assert ex.load_artifact("b") == good