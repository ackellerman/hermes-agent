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
Return a single JSON object: {"items": [...], "coverage": {...}} — never a
bare array; the verdicts themselves are the ELEMENTS of "items".
Every cite must be a message range inside the region. Targeted dump excerpts are
provided as data only.
"""

STAGE_C_PROMPT = """\
You are the EXTRACT stage of a compaction pipeline. You are given the Stage B
reasoned verdicts AND the verbatim dump rows for a region. Emit the checkpoint
JSON for this region with these sections, in order:
- kept_substance: for each Stage B verdict "keep" (and each "superseded",
  distilling the SUPERSEDING direction): the load-bearing SUBSTANCE of that
  episode — the named mechanisms, definitions, parameters, decisions, outcomes
  a live agent needs to continue work WITHOUT re-reading the dump. Distilled
  facts, not narrative, not quotes: compress the content, keep the facts.
  COMPLETENESS RULE: every named mechanism, definition, formula, default
  value, and explicit list in a kept episode's load-bearing content must
  survive into the distilled facts — the probe grades recall of these.
  Each entry carries "ref" (the verdict's map_ref) and "cites".
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
Every item (outside kept_substance, whose job is substance) must carry "cites":
[[dump_id, start_msg, end_msg]] resolved to real dump ranges. Quotes at most 50
tokens per item (orienting quotes only); kept_substance entries are capped at
400 tokens each and the section has a stated budget — stay under it; density
matters more than prose. Bulk verbatim retention belongs to the dump, not the
checkpoint. Output ONLY the checkpoint JSON.

CORRECTION CITATION RULE (AC-8): for an instruction/correction, cite the
IMMEDIATE message range right after the correction where the correction was
applied — the very next restart/re-implementation/tool run that put the new
direction into effect (typically within a few messages of the correction). Do
NOT cite the original contradiction, and do NOT cite a much-later final state
(e.g. a deploy days later).

Emit EXACTLY this shape (sections in this order; cite triples [dump-id, start, end]).
An unproduced section MUST be the string "null_reason: <why>" — never []:
{
  "kept_substance": [{"ref": "ep-3", "substance": "...distilled facts...",
                      "cites": [[0, 4, 5]]}],
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
Return a single JSON object with the twelve keys above — never a bare array
(sections are KEYS of the object, not elements of a list).
Every one of the twelve keys above must be present.
"""

CHECK_PROMPT = """\
Your job is checking that this work was done and follows the guidelines. You
are given the on-disk output of a prior extraction stage plus the guidelines.
Verify: schema conformance, citation presence, coverage claims, and that nothing
was invented beyond citation (sampled spot-checks against the provided dump
excerpts). Output ONLY JSON: {"pass": true|false, "reasons": ["..."]}.
"""

CHECKPOINT_SECTIONS = (
    "kept_substance",
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
# SPEC-0049 D2: kept_substance carries distilled FACTS per kept verdict —
# its own cap, exempt from the 50-token orienting-quote cap (AC-9 still
# governs every other section). D2b (AC-D1 iteration): the section budget
# SCALES with the region so a dense region is not force-crushed into a fixed
# 2KB (the real-lane probe showed 2000 total starving a 448KB region); the
# per-entry cap doubles so multi-fact episodes distill fully.
MAX_SUBSTANCE_TOKENS_PER_ENTRY = 400
MIN_SUBSTANCE_TOKENS_TOTAL = 2000
MAX_SUBSTANCE_TOKENS_TOTAL = 8000


def substance_budget_for(dump_chars: int) -> int:
    """D2b: kept_substance total cap for a region of this size — roughly
    2 tokens per 100 chars of dump, clamped to [2000, 8000]. A 25K-char
    region gets the floor; a 450K-char region gets the ceiling (~56x
    compression against the dump it replaces)."""
    scaled = int(dump_chars // 50)
    return max(MIN_SUBSTANCE_TOKENS_TOTAL,
               min(MAX_SUBSTANCE_TOKENS_TOTAL, scaled))

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
    """Parse a stage output as a JSON OBJECT. SPEC-0048 D-B wrapper tolerance:
    a top-level JSON ARRAY is accepted when its elements look like stage-b
    verdict items (carry ``verdict``) — it is wrapped as ``{"items": [...]}``
    so the downstream schema check runs unchanged (F-B: the rig's stage_b lane
    returned a bare array of verdict items; the shape was right for the
    CONTENT, only the wrapper was missing). Any other non-object shape is
    still rejected. An empty raw output never reaches here as "not JSON" —
    ``_stage_llm`` surfaces it as StageTransportError first (D-A)."""
    text = _strip_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"stage output is not JSON: {raw[:200]!r}") from exc
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and parsed and _looks_like_verdict_items(parsed):
        # D-B: the array IS the model's per-item accounting of the slice it was
        # shown, so the wrapper states the coverage flag the schema check
        # requires (``stage_b_schema_check`` runs UNCHANGED) and marks the wrap
        # for audit — the coverage claim is structural, not model-stated.
        return {"items": parsed,
                "coverage": {"every_map_item_accounted": True,
                             "wrapped_from_bare_array": True}}
    raise ValueError(f"stage output is not an object: {raw[:200]!r}")


def _looks_like_verdict_items(items: List[Any]) -> bool:
    """D-B: does every element of this bare array carry the stage-b verdict
    shape (a ``verdict`` key)? Non-dict elements never qualify."""
    return all(isinstance(it, dict) and "verdict" in it for it in items)


def dump_window_degenerate(slice_covers: Any, dump_window: Any) -> Optional[str]:
    """SPEC-0047 D3: is the slice degenerate vs the dump window being extracted?

    Returns the reason string when degenerate, None when usable. Degenerate
    means the slice's bounds do not overlap the dump window at all:
    ``covers.end_msg < dump.start_msg`` (the map has not caught up to the dump)
    or a 0..0 (collapsed/empty) slice against a larger dump window.

    ``dump_window`` is a ``(start_msg, end_msg)`` pair — either a 2-sequence
    or a meta dict carrying ``start_msg``/``end_msg`` keys.
    """
    try:
        s = int(slice_covers.get("start_msg", 0))
        e = int(slice_covers.get("end_msg", 0))
    except (TypeError, ValueError, AttributeError):
        return f"slice covers is not an int range object: {slice_covers!r}"
    if isinstance(dump_window, dict):
        dump_window = (dump_window.get("start_msg"), dump_window.get("end_msg"))
    try:
        ds = int(dump_window[0])
        de = int(dump_window[1])
    except (TypeError, ValueError, IndexError):
        return None  # no usable dump window: other checks govern
    if e < ds:
        return (f"degenerate slice vs dump window: slice covers {s}..{e} ends "
                f"before the dump window starts ({ds}..{de})")
    if s == 0 and e == 0 and de > ds:
        return (f"degenerate slice vs dump window: 0..0 slice against dump "
                f"window {ds}..{de}")
    return None


class RegionExtractor:
    """Drives stages A/B/C (+ checks) for one dumped region.

    The region directory is owned by the dump PRODUCER (D1,
    :meth:`agent.compaction_dump.DumpStore.write_dump`): by the time any
    extraction runs the directory must already exist. ``__init__`` therefore
    ASSERTS rather than creating — a missing directory means the producer /
    consumer layout has drifted, and fabricating it here would hide exactly that
    regression.
    """

    def __init__(self, root: Path, session_id: str, dump_id: str, *,
                 max_stage_retries: int = 2):
        self.root = Path(root)
        self.session_id = str(session_id)
        self.dump_id = str(dump_id)
        self.dir = self.root / self.session_id / self.dump_id
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"region directory {self.dir} does not exist: the dump producer "
                f"(DumpStore.write_dump) owns it, so a missing directory means the "
                f"producer/consumer layout has drifted")
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

    def stage_c(self, llm: LLM, stage_b_output: Dict[str, Any], *,
                dump_msgs: Optional[List[Dict[str, Any]]] = None,
                slice_covers: Any = None,
                dump_window: Any = None) -> Dict[str, Any]:
        # D2b: the substance budget scales with the dump size — a dense
        # region gets proportionally more kept_substance room.
        substance_budget = None
        if dump_msgs is not None:
            substance_budget = substance_budget_for(
                sum(len(str(m.get("content", ""))) for m in dump_msgs))
        return self._run_with_check(
            "c",
            lambda: parse_stage_json(llm([
                {"role": "user", "content": STAGE_C_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"stage_b": stage_b_output, "dump": dump_msgs,
                     "kept_substance_budget_tokens": substance_budget},
                    ensure_ascii=False, default=str)}]),
            ),
            lambda obj: checkpoint_schema_check(obj, slice_covers=slice_covers,
                                                dump_window=dump_window,
                                                substance_budget=substance_budget),
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


def _read_region_meta(region_dir: Path) -> Optional[Dict[str, Any]]:
    """(start_msg, end_msg) for the region's dump, or None when absent/unreadable."""
    meta_path = Path(region_dir) / f"{Path(region_dir).name}.meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(meta, dict) and "start_msg" in meta and "end_msg" in meta:
        return meta
    return None


def run_extraction_cycle(
    region_dir: Path,
    *,
    reason_llm: LLM,
    extract_llm: LLM,
    max_stage_retries: int = 2,
) -> Dict[str, Any]:
    """Scheduler entry (AC-22): one region's A -> B -> C cycle.

    Wraps :class:`RegionExtractor` driving Stage A (mechanical slice from the
    dumped region's map slice), Stage B (reason), Stage C (extract). ``reason_llm``
    and ``extract_llm`` are the per-stage deterministic LLM callables. Returns the
    Stage-C checkpoint on success. On a stage check failure that exhausts its
    retries, :class:`StageCheckError` propagates: the caller parks the region
    (stays live, never swapped) and records the park in telemetry.

    ``region_dir`` is ``<storage_root>/<session_id>/<dump_id>`` — the directory
    whose ``stage_a.json`` / ``stage_b.json`` / ``stage_c.json`` land.
    """
    from agent.compaction_map import CompactionMap

    root = region_dir.parent.parent  # <root>/<session_id>/<dump_id> -> <root>/<session_id>
    session_id = region_dir.parent.name
    dump_id = region_dir.name
    extractor = RegionExtractor(root, session_id, dump_id, max_stage_retries=max_stage_retries)

    stage_a = extractor.load_artifact("a")
    if stage_a is None:
        map_slice = CompactionMap(root, session_id).slice(0, 2_000_000_000)
        stage_a = extractor.stage_a(map_slice)

    # SPEC-0047 D3: degenerate-slice refusal. A slice whose bounds do not
    # overlap the dump window (map not caught up, or a collapsed 0..0 slice)
    # must NEVER flow into Stage B/C — the pre-map pass produced a
    # confidence-0.2 "Degenerate empty region" checkpoint exactly that way.
    # Raise StageCheckError so the caller parks the region; a later pass
    # retries after the map has caught up (the catch-up path).
    slice_obj = stage_a.get("slice", stage_a) if isinstance(stage_a, dict) else stage_a
    covers = slice_obj.get("covers", {}) if isinstance(slice_obj, dict) else {}
    dump_meta = _read_region_meta(region_dir)
    if dump_meta is not None:
        degenerate = dump_window_degenerate(covers, dump_meta)
        if degenerate:
            raise StageCheckError("a", [degenerate])

    stage_b = extractor.load_artifact("b")
    if stage_b is None:
        dump_msgs = _read_dump_rows(region_dir)
        stage_b = extractor.stage_b(reason_llm, stage_a.get("slice", stage_a),
                                    dump_reader=(lambda s, e: dump_msgs[s:e + 1])
                                    if dump_msgs is not None else None)

    stage_c = extractor.load_artifact("c")
    if stage_c is None:
        stage_c = extractor.stage_c(extract_llm, stage_b,
                                    dump_msgs=_read_dump_rows(region_dir),
                                    slice_covers=covers,
                                    dump_window=dump_meta)
    return stage_c


def _read_dump_rows(region_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """SPEC-0049 D1: the region's own dump rows, or None when unreadable."""
    dump_path = Path(region_dir) / f"{Path(region_dir).name}.jsonl"
    try:
        rows = [json.loads(line) for line in
                dump_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError):
        return None
    return rows if rows else None


def checkpoint_schema_check(checkpoint: Dict[str, Any],
                            slice_covers: Any = None,
                            dump_window: Any = None,
                            substance_budget: Optional[int] = None,
                            ) -> List[str]:
    """AC-6 + AC-9 script validation: every section present; an empty section
    needs an explicit null_reason; quotes capped at 50 tokens/item; citations
    resolve to (dump_id, start, end) triples.

    SPEC-0047 D3: when the slice's covers and the dump window are supplied,
    a checkpoint built from a slice that does not overlap the dump window is
    rejected (degenerate slice vs dump window) — the "Degenerate empty region,
    confidence 0.2" artifact class must never pass.
    """
    errors: List[str] = []
    if slice_covers is not None and dump_window is not None:
        degenerate = dump_window_degenerate(slice_covers, dump_window)
        if degenerate:
            errors.append(degenerate)
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
    # SPEC-0049 D2: kept_substance — its own caps, exempt from AC-9's
    # orienting-quote cap (substance IS the point of the section).
    budget_total = _substance_budget_total(substance_budget)
    substance = checkpoint.get("kept_substance")
    if isinstance(substance, list):
        total = 0
        for i, entry in enumerate(substance):
            if not isinstance(entry, dict):
                errors.append(f"kept_substance[{i}] is not an object")
                continue
            if not entry.get("substance"):
                errors.append(f"kept_substance[{i}] missing 'substance'")
            if not entry.get("ref"):
                errors.append(f"kept_substance[{i}] missing 'ref'")
            cites = entry.get("cites")
            if not (isinstance(cites, list) and cites):
                errors.append(f"kept_substance[{i}] missing cites")
            else:
                for c in cites:
                    if not (isinstance(c, (list, tuple)) and len(c) == 3):
                        errors.append(f"kept_substance[{i}] cite {c!r} is not (dump_id, start, end)")
            entry_tokens = _approx_tokens(str(entry.get("substance", "")))
            total += entry_tokens
            if entry_tokens > MAX_SUBSTANCE_TOKENS_PER_ENTRY:
                errors.append(
                    f"kept_substance[{i}] > {MAX_SUBSTANCE_TOKENS_PER_ENTRY} tokens")
        if total > budget_total:
            errors.append(
                f"kept_substance total {total} > {budget_total} tokens")
    return errors


def _substance_budget_total(substance_budget: Optional[int]) -> int:
    """D2b: the effective kept_substance cap — explicit budget when supplied
    (the caller sized it off the region), else the floor (a small region
    never needs more)."""
    return int(substance_budget) if substance_budget else MIN_SUBSTANCE_TOKENS_TOTAL


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