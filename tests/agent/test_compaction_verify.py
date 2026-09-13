"""SPEC-0042 review gate + loss probe + mechanical alternation invariant.

AC-10/11 (gate) and AC-12 (check_alternation_invariant, four-case falsifier).
"""

import json

import pytest

from agent.compaction_verify import (
    check_alternation_invariant,
    run_review_gate,
    run_loss_probe,
)


def _user(t):
    return {"role": "user", "content": t}


def _asst(t, tool_calls=None):
    msg = {"role": "assistant", "content": t}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(call_id, content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class TestAC12CheckAlternationInvariant:
    def _call(self, name, call_id="c1"):
        return {"id": call_id, "function": {"name": name, "arguments": "{}"}}

    def test_valid_swap_output_passes(self):
        """FALSIFIER AC-12 case (a): hand-built valid swap output -> is_valid
        True, empty violations. Includes an intact tool pair and the checkpoint
        row pattern (assistant summary row followed by the next user turn)."""
        messages = [
            _user("hello"),
            _asst("hi"),
            _user("run it"),
            _asst("", tool_calls=[self._call("terminal")]),
            _tool("c1"),
            _asst("compaction checkpoint: ... [dump: d-1 0-49]"),
            _user("next topic"),
        ]
        is_valid, violations = check_alternation_invariant(messages)
        assert is_valid and violations == []

    def test_falsifier_case_b_adjacent_same_role_fails(self):
        """FALSIFIER AC-12 case (b1): two adjacent same-role messages."""
        messages = [_user("a"), _user("b")]
        is_valid, violations = check_alternation_invariant(messages)
        assert not is_valid
        assert any("adjacent same-role" in v for v in violations)

    def test_falsifier_case_b_synthetic_user_between_tool_pair_fails(self):
        """FALSIFIER AC-12 case (b2): user message between assistant tool-call
        and its tool-result reply."""
        messages = [
            _user("go"),
            _asst("", tool_calls=[self._call("terminal")]),
            _user("synthetic mid-loop injection"),
            _tool("c1"),
        ]
        is_valid, violations = check_alternation_invariant(messages)
        assert not is_valid
        assert any("synthetic user" in v for v in violations)

    def test_falsifier_case_c_missing_tool_result_fails(self):
        """FALSIFIER AC-12 case (b3): tool_use whose matching tool_result is
        missing."""
        messages = [_user("go"), _asst("", tool_calls=[self._call("terminal")])]
        is_valid, violations = check_alternation_invariant(messages)
        assert not is_valid
        assert any("no matching tool_result" in v for v in violations)

    def test_falsifier_case_c_reordered_tool_result_fails(self):
        """FALSIFIER AC-12 case (b3): tool result answering a call not yet made
        (reordered)."""
        messages = [_user("go"), _asst("", tool_calls=[self._call("a")]), _tool("c9")]
        is_valid, violations = check_alternation_invariant(messages)
        assert not is_valid
        assert any("no matching assistant tool_call" in v or "reordered" in v for v in violations)


_DUMP = [
    _user("Actually, use the jsonl format, not csv."),
    _asst("Switched the exporter to jsonl."),
    _user("Commit to shipping the export module Friday."),
    _asst("Noted: Friday ship date."),
    _asst("Generated artifact report-x.txt (ephemeral, unrecoverable)."),
]


class TestAC10GateFlagsMissingCommitment:
    def _ckpt(self, *, with_commitment: bool):
        ckpt = {
            "instructions_and_corrections": [
                {"what": "use jsonl format", "cites": [[0, 1]]}],
            "commitments": ([{"what": "ship export module Friday", "cites": [[2, 3]]}]
                            if with_commitment else []),
            "artifacts": [{"what": "report-x.txt", "recoverable": False,
                           "substance": "the report body", "cites": [[4, 4]]}],
        }
        return ckpt

    def test_falsifier_drop_commitment_gate_flags_it(self):
        """FALSIFIER AC-10: drop the commitment from the checkpoint -> gate
        must flag it -> a gate that passes commits nothing."""
        flagged = run_review_gate(_gate_llm, self._ckpt(with_commitment=False), _DUMP)
        assert flagged["swap_eligible"] is False and flagged["findings"]

        passed = run_review_gate(_gate_llm, self._ckpt(with_commitment=True), _DUMP)
        assert passed["swap_eligible"] is True and not passed["findings"]


def _gate_llm(messages):
    """Deterministic stand-in gate: scans the dump for content words missing
    from the checkpoint text. Temperature-0-equivalent for AC-11."""
    data = json.loads(messages[1]["content"])
    ckpt_text = json.dumps(data["checkpoint"]).lower()
    dump_text = " ".join(str(m.get("content", "")) for m in data["dump"]).lower()
    findings = []
    for marker, label in (("friday", "commitment: ship date"),
                          ("jsonl", "correction: export format"),
                          ("report-x", "artifact: report-x.txt")):
        if marker in dump_text and marker not in ckpt_text:
            findings.append({"what": f"missing {label}", "cites": [[0, 4]]})
    return json.dumps({"swap_eligible": not findings, "findings": findings})


class TestAC11GateVerdictStability:
    def test_falsifier_two_independent_runs_agree(self):
        """FALSIFIER AC-11: verdict flip on identical inputs fails. Two runs,
        same inputs, deterministic seed/temperature contract."""
        ckpt = {
            "instructions_and_corrections": [{"what": "use jsonl", "cites": [[0, 1]]}],
            "commitments": [{"what": "Friday ship", "cites": [[2, 3]]}],
        }
        v1 = run_review_gate(_gate_llm, ckpt, _DUMP, seed=42)
        v2 = run_review_gate(_gate_llm, ckpt, _DUMP, seed=42)
        assert v1["swap_eligible"] == v2["swap_eligible"] and v1["findings"] == v2["findings"]


class TestLossProbe:
    def test_gap_detected_when_checkpoint_lacks_answer(self):
        questions = ["When is the export module shipping?"]
        result = run_loss_probe(
            _answer_llm, _grader_llm,
            checkpoint={"commitments": []},  # checkpoint lacks the answer
            dump_msgs=_DUMP, questions=questions,
        )
        assert result["pass"] is False and result["gaps"]

    def test_no_gap_when_checkpoint_has_answer(self):
        questions = ["When is the export module shipping?"]
        result = run_loss_probe(
            _answer_llm, _grader_llm,
            checkpoint={"commitments": [{"what": "Friday ship", "cites": [[2, 3]]}]},
            dump_msgs=_DUMP, questions=questions,
        )
        assert result["pass"] is True and not result["gaps"]


def _answer_llm(messages):
    data = json.loads(messages[1]["content"])
    ckpt_text = json.dumps(data["checkpoint"]).lower()
    answer = "friday" if "friday" in ckpt_text else "unknown"
    return json.dumps({"answer": answer, "found_in_checkpoint": answer != "unknown"})


def _grader_llm(messages):
    data = json.loads(messages[1]["content"])
    ans = (data["answer"] or {}).get("answer", "")
    dump_text = " ".join(str(m.get("content", "")) for m in data["dump"]).lower()
    correct = "friday" in dump_text
    return json.dumps({"match": ans == "friday" and correct,
                       "why": "checkpoint answer vs dump truth"})