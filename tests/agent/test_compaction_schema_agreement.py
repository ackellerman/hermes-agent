"""SPEC-0042 F2 class fix: the checkpoint validator and the eval fixtures MUST
agree on the checkpoint form, or fidelity recall is scored against a form the
pipeline's own gate rejects.

Contract under test (spec §3.3 Stage C + AC-6/AC-7):
- every checkpoint-item citation is a (dump_id, start_msg, end_msg) triple;
  a 2-element [start, end] cite is a form the gate must reject.
- an empty section (no items) must carry an explicit null_reason; a bare []
  empty section must be rejected.
- the eval's reference checkpoint (evals/compaction/fidelity_eval
  .rule_based_checkpoint) must itself pass the real validator — the exact
  disagreement the round-3 review (t_7ba7f718, finding F2) blocked on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from agent.compaction_extract import checkpoint_schema_check

REPO_ROOT = Path(__file__).resolve().parents[2]


def _conformant_checkpoint() -> dict:
    """A checkpoint in the exact Stage C form the validator accepts."""
    did = "fx-abcd1234"
    return {
        "instructions_and_corrections": [
            {"what": "use JSONL not CSV", "kind": "correction",
             "cites": [[did, 3, 3]]},
        ],
        "decisions": [{"what": "Postgres for audit log", "cites": [[did, 9, 10]]}],
        "commitments": [{"what": "ship Friday", "cites": [[did, 7, 8]]}],
        "insights": "null_reason: no insights",
        "open_threads": "null_reason: none open",
        "artifacts": [{"what": "audit-summary.txt", "cites": [[did, 13, 14]],
                       "recoverable": False}],
        "world_effects": [{"what": "staging runs exporter", "cites": [[did, 15, 16]]}],
        "links": "null_reason: no links",
        "narrative": "Export service work.",
        "confidence": 0.8,
        "coverage": {"complete": True},
    }


def test_conformant_checkpoint_passes_gate():
    errors = checkpoint_schema_check(_conformant_checkpoint())
    assert errors == []


def test_two_element_cite_is_rejected():
    """F2 cite-arity class: [start, end] without dump_id must fail. This is the
    exact regressed form the round-3 review found the fs core produced."""
    cp = _conformant_checkpoint()
    cp["commitments"][0]["cites"] = [[7, 8]]  # 2-element, legacy form
    errors = checkpoint_schema_check(cp)
    assert any("not (dump_id, start, end)" in e for e in errors), errors


def test_empty_section_without_null_reason_is_rejected():
    """AC-6: an empty section needs an explicit null_reason."""
    cp = _conformant_checkpoint()
    cp["links"] = []
    errors = checkpoint_schema_check(cp)
    assert any("lacks null_reason" in e for e in errors), errors


def test_eval_reference_checkpoint_passes_the_real_gate():
    """The F2 class guard: the eval's reference checkpoint must be gate-valid.
    This ties the fixtures' scored form to the validator the pipeline runs."""
    sys.path.insert(0, str(REPO_ROOT / "evals" / "compaction"))
    import fidelity_eval

    fixture = fidelity_eval.fx_mod.build_correction_fixture()
    reference = fidelity_eval.rule_based_checkpoint(fixture)
    errors = checkpoint_schema_check(reference)
    assert errors == [], f"eval reference checkpoint rejected by the gate: {errors}"