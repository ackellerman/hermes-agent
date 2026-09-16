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
    stage_b_verdicts: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """N sampled questions generated FROM dump content.

    SPEC-0049 D3 kept-scoping: when ``stage_b_verdicts`` are supplied, the
    question generator is pointed ONLY at the cite ranges of KEEP verdicts —
    Stage B's drops are out of scope by design (grading a deliberate drop as a
    checkpoint gap inverts the verdict; a 100% flip rate was observed exactly
    that way on the rig's 0001-68bcdd95 region). Superseded items are also
    out of scope: the checkpoint distills the superseding direction.

    D3b (AC-D1 iteration): questions must target LOAD-BEARING facts — named
    mechanisms, definitions, formulas, decisions, explicit lists — the things
    a compact checkpoint MUST retain. Incidental trivia (an env-var value
    echoed in a tool log) is not a compaction obligation; asking it created
    un-satisfiable gaps on the real lane.
    """
    prompt = (
        f"Generate exactly {samples} distinct questions about LOAD-BEARING "
        "facts present in the conversation below: named mechanisms, "
        "definitions, formulas, parameter values, explicit lists, decisions "
        "and their reasons — facts a continuation agent would need. Do NOT "
        "ask about incidental details (session ids, environment variable "
        "values, boilerplate). Each question must have a specific, checkable "
        'answer stated in the conversation. Output ONLY JSON: '
        '{"questions": ["..."]}')
    payload: Dict[str, Any] = {"dump": dump_msgs, "sampling_seed": seed}
    if stage_b_verdicts:
        kept_msgs = _kept_scope_rows(stage_b_verdicts, dump_msgs)
        if kept_msgs is not None:
            payload = {"dump": kept_msgs, "sampling_seed": seed,
                       "kept_scope": True}
    verdict = json.loads(question_llm([
        {"role": "user", "content": prompt},
        {"role": "user", "content": json.dumps(
            payload, ensure_ascii=False, default=str)},
    ]))
    return list(verdict.get("questions", []))


def _kept_scope_rows(stage_b_verdicts: List[Dict[str, Any]],
                     dump_msgs: List[Dict[str, Any]],
                     ) -> Optional[List[Dict[str, Any]]]:
    """The dump rows inside KEEP-verdict cite ranges (each cite is
    [dump_id, start, end] over the region's own dump, indexed 0..len-1), or
    None when no usable keep cites exist (caller falls back to the full dump
    rather than generating questions about nothing)."""
    keep_rows: List[int] = []
    for item in stage_b_verdicts or []:
        if not isinstance(item, dict) or item.get("verdict") != "keep":
            continue
        for cite in item.get("cites") or []:
            if not (isinstance(cite, (list, tuple)) and len(cite) == 3):
                continue
            _dump_id, start, end = cite
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            keep_rows.extend(range(max(0, start), min(end, len(dump_msgs) - 1) + 1))
    if not keep_rows:
        return None
    seen = sorted(set(keep_rows))
    return [dump_msgs[i] for i in seen]


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
                "Does the checkpoint answer convey the ESSENTIAL ground-truth "
                "fact for this question? Match on MEANING, not wording: the "
                "answer may paraphrase or omit incidental specifics, but the "
                "core fact (mechanism name, definition, formula, decision, "
                "list membership) must be present and correct. An answer "
                "that says the checkpoint lacks the answer is NOT a match. "
                'Output ONLY JSON: {"match": true|false, "why": "..."}')},
            {"role": "user", "content": json.dumps(
                {"question": q, "answer": ans, "dump": dump_msgs},
                ensure_ascii=False, default=str)},
        ]))
        if grade.get("match") is not True:
            gaps.append({"question": q, "answer": ans, "why": grade.get("why")})
    return {"questions": len(questions), "gaps": gaps, "pass": not gaps}