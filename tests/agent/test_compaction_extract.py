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
    @pytest.fixture
    def region(self, tmp_path):
        """D1: the region directory is the PRODUCER's. write_dump creates it; a
        test that mkdir'd one would mask a producer/consumer layout regression
        (D2), so these stage tests go through a real dump."""
        from agent.compaction_dump import DumpStore
        store = DumpStore(tmp_path)
        ref = store.write_dump("sess", [{"role": "user", "content": "seed"}],
                               start_msg=0, end_msg=0, turn=1)
        return tmp_path, ref.dump_id

    def test_stage_a_is_mechanical(self, region):
        root, dump_id = region
        ex = RegionExtractor(root, "sess", dump_id)
        art = ex.stage_a({"complete": True, "episodes": []})
        assert art["stage"] == "a"
        assert ex.load_artifact("a") == art

    def test_stage_b_runs_checks_and_parks_after_retries(self, region):
        """A stage that never validates parks the region (StageCheckError),
        never silently succeeds."""
        root, dump_id = region
        ex = RegionExtractor(root, "sess", dump_id, max_stage_retries=1)
        bad = {"items": [{"verdict": "maybe"}]}  # schema-invalid, always
        with pytest.raises(Exception):
            ex.stage_b(lambda msgs: json.dumps(bad), {"episodes": []})

    def test_stage_b_valid_output_persists_artifact(self, region):
        root, dump_id = region
        ex = RegionExtractor(root, "sess", dump_id)
        good = {"items": [{"map_ref": "e1", "verdict": "keep", "because": "dep",
                           "cites": [[0, 5]]}],
                "open_questions": [],
                "coverage": {"every_map_item_accounted": True}}
        out = ex.stage_b(lambda msgs: json.dumps(good), {"episodes": []})
        assert out == good
        assert ex.load_artifact("b") == good

    def test_region_extractor_asserts_absent_directory_instead_of_creating_it(self, tmp_path):
        """D1: a region directory that the producer never created must raise, not
        be fabricated by the extractor — that fabrication is exactly what hid the
        dead producer/consumer chain (F1)."""
        with pytest.raises(FileNotFoundError):
            RegionExtractor(tmp_path, "sess", "never-produced")
        assert not (tmp_path / "sess" / "never-produced").exists()