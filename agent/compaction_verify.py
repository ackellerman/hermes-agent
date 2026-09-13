"""SPEC-0042 verification: review gate, loss probe, and the mechanical
``check_alternation_invariant`` the swap path must call on its own output
before committing (AC-12). Also owns checkpoint schema validation re-export
(defined in ``compaction_extract``) so consumers import one module.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.compaction_extract import checkpoint_schema_check  # noqa: F401 — re-export

REVIEW_GATE_PROMPT = """\
You are the review gate of a compaction pipeline. You see a checkpoint (the
compact form a live agent will keep) and the verbatim dump (the full form).
Answer: what would the live agent be devastated to learn it forgot? Report
ONLY findings supported by a dump citation (message range). If the checkpoint
faithfully covers the dump, pass it. Output ONLY JSON:
{"swap_eligible": true|false, "findings": [{"what": "...", "cites": [[start, end]]}]}
"""

LOSS_PROBE_PROMPT = """\
Answer the question using ONLY the checkpoint below. If the checkpoint does not
contain the answer, say so. Output ONLY JSON:
{"answer": "...", "found_in_checkpoint": true|false}
"""


class ToolPairError(ValueError):
    pass


def check_alternation_invariant(messages: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Mechanical, no model call (AC-12): walk once, return (is_valid, violations).

    A violation is any of:
    - two adjacent messages with the same ``role``;
    - a ``user`` message between an assistant tool-call and its tool-result
      reply (synthetic mid-loop injection);
    - a ``tool`` message whose matching assistant tool_call is missing, or a
      tool_use id answered out of order.
    """
    violations: List[str] = []
    prev_role: Optional[str] = None
    pending_tool_ids: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if prev_role is not None and role == prev_role:
            violations.append(
                f"adjacent same-role messages at index {i} (role={role!r})")
        # user injected between a tool_call and its result
        if role == "user" and pending_tool_ids:
            violations.append(
                f"synthetic user message at index {i} between tool-call and tool-result")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = (tc or {}).get("id") or ((tc or {}).get("function") or {}).get("name")
                if tc_id:
                    pending_tool_ids.append(str(tc_id))
        elif role == "tool":
            tool_id = str(msg.get("tool_call_id") or "")
            if tool_id not in pending_tool_ids:
                violations.append(
                    f"tool message at index {i} has no matching assistant tool_call "
                    f"(id={tool_id!r}) or is reordered")
            else:
                pending_tool_ids.remove(tool_id)
        prev_role = role
    for leftover in pending_tool_ids:
        violations.append(f"tool_use {leftover!r} has no matching tool_result")
    return (not violations), violations


def run_review_gate(
    gate_llm: Callable[[List[Dict[str, str]]], str],
    checkpoint: Dict[str, Any],
    dump_msgs: List[Dict[str, Any]],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fresh-context gate over checkpoint + dump. Deterministic contract: the
    caller pins temperature 0 + sampling seed (AC-11); this function stays pure
    given the same llm behavior."""
    verdict_raw = gate_llm([
        {"role": "user", "content": REVIEW_GATE_PROMPT},
        {"role": "user", "content": json.dumps(
            {"checkpoint": checkpoint, "dump": dump_msgs, "sampling_seed": seed},
            ensure_ascii=False, default=str)},
    ])
    verdict = json.loads(verdict_raw.strip().removeprefix("```json").removesuffix("```").strip())
    for key in ("swap_eligible", "findings"):
        if key not in verdict:
            raise ValueError(f"gate verdict missing '{key}': {verdict!r}")
    return verdict


def generate_loss_probe_questions(
    question_llm: Callable[[List[Dict[str, str]]], str],
    dump_msgs: List[Dict[str, Any]],
    *,
    samples: int = 8,
    seed: int = 0,
) -> List[str]:
    """N sampled questions generated FROM dump content."""
    verdict = json.loads(question_llm([
        {"role": "user", "content": (
            f"Generate exactly {samples} distinct questions whose answers are "
            "present in the conversation below. Output ONLY JSON: "
            '{"questions": ["..."]}')},
        {"role": "user", "content": json.dumps(
            {"dump": dump_msgs, "sampling_seed": seed}, ensure_ascii=False, default=str)},
    ]))
    return list(verdict.get("questions", []))


def run_loss_probe(
    answer_llm: Callable[[List[Dict[str, str]]], str],
    grader_llm: Callable[[List[Dict[str, str]]], str],
    checkpoint: Dict[str, Any],
    dump_msgs: List[Dict[str, Any]],
    questions: List[str],
) -> Dict[str, Any]:
    """Answer N dump-derived questions from the checkpoint alone; grade against
    the dump. Any disagreement is a gap -> back to Stage B (caller decides)."""
    gaps: List[Dict[str, Any]] = []
    for q in questions:
        ans = json.loads(answer_llm([
            {"role": "user", "content": LOSS_PROBE_PROMPT},
            {"role": "user", "content": json.dumps(
                {"question": q, "checkpoint": checkpoint},
                ensure_ascii=False, default=str)},
        ]))
        grade = json.loads(grader_llm([
            {"role": "user", "content": (
                "Does the checkpoint answer match the dump's ground truth for "
                'this question? Output ONLY JSON: {"match": true|false, "why": "..."}')},
            {"role": "user", "content": json.dumps(
                {"question": q, "answer": ans, "dump": dump_msgs},
                ensure_ascii=False, default=str)},
        ]))
        if grade.get("match") is not True:
            gaps.append({"question": q, "answer": ans, "why": grade.get("why")})
    return {"questions": len(questions), "gaps": gaps, "pass": not gaps}