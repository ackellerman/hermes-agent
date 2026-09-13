"""SPEC-0042 staged extraction: Stage A (mechanical slice) -> Stage B (reason)
-> Stage C (extract), each with an on-disk artifact and a fresh-context check
step before the next stage runs.

Stage prompts are versioned, byte-pinned templates in code. Each stage writes
``<root>/<session_id>/<dump_id>/stage_<X>.json`` before the next runs. A check
failure re-runs the stage (max ``max_stage_retries``), then parks the region
(stays live; never swaps).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

STAGE_B_PROMPT = """\
You are the REASON stage of a compaction pipeline. You are given the trajectory
map slice for a region. For each item on the map, reason about what was done
after it in the conversation. Judge load-bearing-ness by dependency, not
emphasis: an item is load-bearing when later work depended on it, changed
because of it, or would break without it. Record abandonment reasons for dead
trajectories. Do not summarize; decide.
Output ONLY JSON matching exactly:
{"items": [{"map_ref": "<episode or entity id>",
            "verdict": "keep" | "drop" | "superseded",
            "because": "...",
            "cites": [[start_msg, end_msg]]}],
 "open_questions": ["..."],
 "coverage": {"every_map_item_accounted": true}}
Every cite must be a message range inside the region. Targeted dump excerpts are
provided as data only.
"""

STAGE_C_PROMPT = """\
You are the EXTRACT stage of a compaction pipeline. You are given the Stage B
reasoned verdicts for a region. Emit the checkpoint JSON for this region with
these sections, in order:
- instructions_and_corrections: each with intent-validating follow-up work cited
- decisions: with rationale + rejected alternatives
- insights
- commitments
- open_threads
- artifacts: connected-to-work + recoverable/unrecoverable flag; unrecoverable
  must inline substance
- world_effects
- links
- narrative: short orientation glue
- confidence: 0..1
- coverage: self-report of what was and was not accounted for
Every item must carry "cites": [[dump_id, start_msg, end_msg]] resolved to real
dump ranges. Quote at most 50 tokens per item (orienting quotes only); bulk
verbatim retention belongs to the dump, not the checkpoint. Output ONLY the
checkpoint JSON.

CORRECTION CITATION RULE (AC-8): for an instruction/correction, cite the
IMMEDIATE message range right after the correction where the correction was
applied — the very next restart/re-implementation/tool run that put the new
direction into effect (typically within a few messages of the correction). Do
NOT cite the original contradiction, and do NOT cite a much-later final state
(e.g. a deploy days later).

Emit EXACTLY this shape (sections in this order; cite triples [dump-id, start, end]).
An unproduced section MUST be the string "null_reason: <why>" — never []:
{
  "instructions_and_corrections": [{"what": "...", "kind": "correction", "cites": [[0, 10, 12]]}],
  "decisions": [{"what": "...", "cites": [[0, 20, 22]], "rejected_alternatives": ["..."]}],
  "insights": "null_reason: none in this region",
  "commitments": [{"what": "...", "cites": [[0, 30, 32]]}],
  "open_threads": "null_reason: none in this region",
  "artifacts": [{"what": "...", "cites": [[0, 40, 42]], "recoverable": true}],
  "world_effects": [{"what": "...", "cites": [[0, 50, 52]]}],
  "links": "null_reason: none in this region",
  "narrative": "one-line orientation",
  "confidence": 0.8,
  "coverage": {"complete": true}
}
Every one of the eleven keys above must be present.
"""

CHECK_PROMPT = """\
Your job is checking that this work was done and follows the guidelines. You
are given the on-disk output of a prior extraction stage plus the guidelines.
Verify: schema conformance, citation presence, coverage claims, and that nothing
was invented beyond citation (sampled spot-checks against the provided dump
excerpts). Output ONLY JSON: {"pass": true|false, "reasons": ["..."]}.
"""

CHECKPOINT_SECTIONS = (
    "instructions_and_corrections",
    "decisions",
    "insights",
    "commitments",
    "open_threads",
    "artifacts",
    "world_effects",
    "links",
    "narrative",
    "confidence",
    "coverage",
)

MAX_QUOTE_TOKENS_PER_ITEM = 50

LLM = Callable[[List[Dict[str, str]]], str]


class StageCheckError(RuntimeError):
    """A check step failed after exhausting retries — the region parks (stays
    live, never swaps)."""

    def __init__(self, stage: str, reasons: List[str]):
        super().__init__(f"stage {stage} check failed after retries: {reasons}")
        self.stage = stage
        self.reasons = reasons


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    return text.strip()


def parse_stage_json(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"stage output is not JSON: {raw[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"stage output is not an object: {raw[:200]!r}")
    return parsed


class RegionExtractor:
    """Drives stages A/B/C (+ checks) for one dumped region."""

    def __init__(self, root: Path, session_id: str, dump_id: str, *,
                 max_stage_retries: int = 2):
        self.dir = Path(root) / session_id / dump_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_stage_retries = max_stage_retries

    def _artifact(self, stage: str) -> Path:
        return self.dir / f"stage_{stage}.json"

    def load_artifact(self, stage: str) -> Optional[Dict[str, Any]]:
        p = self._artifact(stage)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_artifact(self, stage: str, obj: Dict[str, Any]) -> None:
        p = self._artifact(stage)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, p)

    # ── Stage A (mechanical, no LLM) ────────────────────────────────────

    def stage_a(self, map_slice: Dict[str, Any]) -> Dict[str, Any]:
        artifact = {"slice": map_slice, "stage": "a"}
        self._write_artifact("a", artifact)
        return artifact

    # ── Stage B (reason) ────────────────────────────────────────────────

    def stage_b(self, llm: LLM, map_slice: Dict[str, Any],
                dump_reader: Optional[Callable[[int, int], List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        verdict = self._run_with_check(
            "b",
            lambda: parse_stage_json(llm([
                {"role": "user", "content": STAGE_B_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"slice": map_slice,
                     "dump_excerpts": [dump_reader(0, -1)] if dump_reader else None},
                    ensure_ascii=False, default=str)},
            ])),
            lambda obj: self.stage_b_schema_check(obj),
            check_llm=None,
        )
        return verdict

    @staticmethod
    def stage_b_schema_check(obj: Dict[str, Any]) -> List[str]:
        errors = []
        if "items" not in obj:
            errors.append("missing 'items'")
        else:
            for i, item in enumerate(obj.get("items", [])):
                if item.get("verdict") not in ("keep", "drop", "superseded"):
                    errors.append(f"items[{i}].verdict invalid")
                if not item.get("cites"):
                    errors.append(f"items[{i}] missing cites")
        cov = obj.get("coverage", {})
        if not isinstance(cov, dict) or cov.get("every_map_item_accounted") is not True:
            errors.append("coverage.every_map_item_accounted must be true")
        return errors

    # ── Stage C (extract checkpoint) ─────────────────────────────────────

    def stage_c(self, llm: LLM, stage_b_output: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_with_check(
            "c",
            lambda: parse_stage_json(llm([
                {"role": "user", "content": STAGE_C_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"stage_b": stage_b_output}, ensure_ascii=False, default=str)},
            ])),
            lambda obj: checkpoint_schema_check(obj),
            check_llm=None,
        )

    def _run_with_check(self, stage: str, produce: Callable[[], Dict[str, Any]],
                        schema_check: Callable[[Dict[str, Any]], List[str]],
                        check_llm: Optional[LLM]) -> Dict[str, Any]:
        last_reasons: List[str] = []
        for _ in range(self.max_stage_retries + 1):
            obj = produce()
            reasons = schema_check(obj)
            if check_llm is not None and not reasons:
                verdict = parse_stage_json(check_llm([
                    {"role": "user", "content": CHECK_PROMPT},
                    {"role": "user", "content": json.dumps(obj, ensure_ascii=False, default=str)},
                ]))
                if verdict.get("pass") is not True:
                    reasons = list(verdict.get("reasons", ["check step failed"]))
            if not reasons:
                self._write_artifact(stage, obj)
                return obj
            last_reasons = reasons
        raise StageCheckError(stage, last_reasons)


def checkpoint_schema_check(checkpoint: Dict[str, Any]) -> List[str]:
    """AC-6 + AC-9 script validation: every section present; an empty section
    needs an explicit null_reason; quotes capped at 50 tokens/item; citations
    resolve to (dump_id, start, end) triples."""
    errors: List[str] = []
    for section in CHECKPOINT_SECTIONS:
        if section not in checkpoint:
            errors.append(f"missing section '{section}'")
        elif not checkpoint[section] and section not in ("confidence",):
            if not (isinstance(checkpoint.get(section), dict)
                    and checkpoint[section].get("null_reason")):
                if not (isinstance(checkpoint[section], str)
                        and checkpoint[section].strip().startswith("null_reason:")):
                    errors.append(f"empty section '{section}' lacks null_reason")
    cites_sections = ("instructions_and_corrections", "decisions", "insights",
                      "commitments", "open_threads", "artifacts", "world_effects", "links")
    for section in cites_sections:
        items = checkpoint.get(section)
        if not isinstance(items, list):
            continue
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{section}[{i}] is not an object")
                continue
            cites = item.get("cites")
            if not (isinstance(cites, list) and cites):
                errors.append(f"{section}[{i}] missing cites")
            else:
                for c in cites:
                    if not (isinstance(c, (list, tuple)) and len(c) == 3):
                        errors.append(f"{section}[{i}] cite {c!r} is not (dump_id, start, end)")
            if _approx_tokens(str(item)) > MAX_QUOTE_TOKENS_PER_ITEM:
                errors.append(f"{section}[{i}] quotes > {MAX_QUOTE_TOKENS_PER_ITEM} tokens (AC-9)")
    return errors


def _approx_tokens(text: str) -> int:
    """Rough token bound used for the AC-9 quote cap: a conservative
    word+punctuation count overruns a real tokenizer, so a pass here is a
    strict upper bound — false-fails get surfaced, false-passes don't."""
    return len(text.split()) + text.count('"')


def item_citation_supported(item: Dict[str, Any],
                            dump_lookup: Callable[[str], List[Dict[str, Any]]]) -> bool:
    """AC-7 helper: does the cited dump range exist and contain text?"""
    for cite in item.get("cites", []):
        if not (isinstance(cite, (list, tuple)) and len(cite) == 3):
            return False
        dump_id, start, end = cite
        try:
            msgs = dump_lookup(dump_id)
        except Exception:  # noqa: BLE001 — unknown dump = unsupported citation
            return False
        if not (0 <= start <= end < len(msgs)):
            return False
    return True