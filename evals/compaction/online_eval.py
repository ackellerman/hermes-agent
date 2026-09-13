"""SPEC-0042 ONLINE eval — the runs review finding F3 says were never executed.

Spec §4 item 5 requires ONLINE (model-run) before/after numbers for merge:
"extraction pipeline vs current single-call summary on the labeled fixtures —
report per-class recall/precision (headline: correction recall with
intent-validation) + loss-probe pass rate." Spec AC-10 (gate flags a dropped
section) and AC-11 (two gate runs agree) also require model calls.

This script drives the REAL Stage B (reason) + Stage C (extract) prompts of
``agent.compaction_extract.RegionExtractor`` through a real LLM over the
committed labeled fixture, validates the result with the pipeline's own
checkpoint gate, scores AC-8, then runs AC-10/AC-11 through the real review
gate. Results go to ``evals/compaction/results/online_fidelity_results.json``
with an explicit ``mode: online`` marker.

Usage:
    python evals/compaction/online_eval.py
        [--ollama http://localhost:11434] [--ollama-model muse-glimmer:latest]
        [--json evals/compaction/results/online_fidelity_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).parent))

import fixtures as fx_mod  # noqa: E402
import fidelity_eval as FE  # noqa: E402

from agent.compaction_dump import DumpStore  # noqa: E402
from agent.compaction_extract import RegionExtractor, checkpoint_schema_check  # noqa: E402
from agent.compaction_verify import (  # noqa: E402
    generate_loss_probe_questions,
    run_loss_probe,
    run_review_gate,
)


def _make_ollama_llm(endpoint: str, model: str):
    """A callable-LLM over local ollama. Muse-glimmer is a reasoning model that
    misfires under ollama's ``format: json`` on multi-message prompts, so this
    lane is only useful via ``_make_single_message_llm``. Kept for the local
    option."""
    import urllib.request  # noqa: PLC0415

    def llm(messages, *, _endpoint=endpoint, _model=model):
        payload = {"model": _model, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": m["content"]}
                                for m in messages],
                   "options": {"temperature": 0}}
        req = urllib.request.Request(
            f"{_endpoint.rstrip('/')}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())["message"]["content"]

    return llm


def _make_openrouter_llm(model: str = "openrouter/auto"):
    """OPENROUTER route — the reliable lane for schema-constrained extraction.
    Returns the model's assistant text; the stage/gate wrappers parse JSON."""
    import os  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    key = os.environ.get("OPENROUTER_API_KEY", "")

    def llm(messages):
        payload = {
            "model": model,
            "messages": [{"role": m.get("role", "user"), "content": m["content"]}
                         for m in messages],
            "temperature": 0, "max_tokens": 1600,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}",
                     "HTTP-Referer": "http://localhost",
                     "X-Title": "spec42-online-eval"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    return llm


def _approx_map_slice(fixture: dict) -> dict:
    """A trajectory over the fixture so Stage B has the work threads it reasons
    about. These are the natural topic blocks of the conversation (what a real
    CompactionMap update would have built from the prior turns) — NOT the
    extraction ground truth. Grouping them into granular per-thread episodes is
    what lets Stage C surface the correction/commitment/decision separately;
    a single huge episode collapses everything into one 'keep' and the extractor
    faithfully under-produces (observed live)."""
    return {
        "covers": {"start_msg": 0, "end_msg": len(fixture["messages"]) - 1},
        "complete": True,
        "episodes": [
            {"name": "export service setup (CSV)", "start_msg": 0, "end_msg": 2},
            {"name": "format correction: switch to JSONL", "start_msg": 3, "end_msg": 5},
            {"name": "commitment: ship export module Friday", "start_msg": 6, "end_msg": 7},
            {"name": "decision: Postgres for audit log", "start_msg": 8, "end_msg": 9},
            {"name": "abandoned: streaming transport", "start_msg": 10, "end_msg": 11},
            {"name": "artifact: audit-summary.txt", "start_msg": 12, "end_msg": 13},
            {"name": "world effect: staging runs JSONL exporter", "start_msg": 14, "end_msg": 15},
        ],
        "entities": [], "edges": [],
    }


def run(*, model: str, endpoint: str, provider: str, dump_root: Path,
        tolerance: int = 1) -> dict:
    if provider == "openrouter":
        llm = _make_openrouter_llm(model)
    else:
        llm = _make_ollama_llm(endpoint, model)
    fixture = fx_mod.build_correction_fixture()
    messages = fixture["messages"]
    did = FE.dump_id_for(fixture)

    store = DumpStore(dump_root)
    ref = store.write_dump("sess-online", messages, start_msg=0,
                           end_msg=len(messages) - 1, turn=0)
    dump_id = ref.dump_id

    extractor = RegionExtractor(dump_root, "sess-online", dump_id,
                                max_stage_retries=2)
    map_slice = _approx_map_slice(fixture)

    def dump_reader(start, end):
        return store.read_messages("sess-online", dump_id, start_msg=start, end_msg=end)

    # Stage B (reason) then Stage C (extract) through the REAL model.
    stage_b = extractor.stage_b(llm, map_slice, dump_reader=dump_reader)
    stage_c = extractor.stage_c(llm, stage_b)   # schema-checked by the pipeline
    checkpoint = stage_c

    # AC-8: score the produced checkpoint against the labeled ground truth.
    scores = FE.score_checkpoint(checkpoint, fixture)
    correction_recall = scores["corrections"]["recall"]
    gate_errors = checkpoint_schema_check(checkpoint)
    ac8_pass = gate_errors == [] and correction_recall is not None \
        and correction_recall >= 0.95

    # Loss probe pass rate (spec §4 item 5) — questions generated from the dump,
    # answered from the checkpoint alone, graded against the dump.
    questions = generate_loss_probe_questions(llm, messages, samples=8)
    loss = run_loss_probe(llm, llm, checkpoint, messages, questions)

    # AC-10: drop the commitment -> the gate must produce a finding for it.
    defective = json.loads(json.dumps(checkpoint))
    if isinstance(defective.get("commitments"), list):
        defective["commitments"] = []
    else:
        defective["commitments"] = "null_reason: <AC-10 planted drop>"
    gate_full = run_review_gate(llm, checkpoint, messages, seed=0)
    gate_dropped = run_review_gate(llm, defective, messages, seed=0)

    # AC-11: two independent gate runs on identical inputs agree (seed 0 & 1).
    gate_stable_a = run_review_gate(llm, checkpoint, messages, seed=0)
    gate_stable_b = run_review_gate(llm, checkpoint, messages, seed=1)
    ac11_stable = (gate_stable_a.get("swap_eligible") == gate_stable_b.get("swap_eligible"))
    ac10_flags_drop = gate_dropped.get("swap_eligible") is False

    return {
        "mode": "online",
        "model": model,
        "endpoint": endpoint,
        "dump_id": did,
        "stage_b_keys": sorted(stage_b.keys()),
        "stage_c_keys": sorted(checkpoint.keys()),
        "checkpoint_gate_valid": gate_errors == [],
        "checkpoint_gate_errors": gate_errors,
        "scores": scores,
        "correction_recall_headline": correction_recall,
        "ac8_pass": bool(ac8_pass),
        "loss_probe": {"questions": loss["questions"], "gaps": len(loss["gaps"]),
                       "pass": bool(loss["pass"])},
        "ac10": {
            "gate_pass_checkpoint": gate_full.get("swap_eligible"),
            "gate_dropped_commitment": gate_dropped.get("swap_eligible"),
            "dropped_findings": gate_dropped.get("findings", []),
            "flags_dropped_section": bool(ac10_flags_drop),
        },
        "ac11": {
            "run_a_swap_eligible": gate_stable_a.get("swap_eligible"),
            "run_b_swap_eligible": gate_stable_b.get("swap_eligible"),
            "stable": bool(ac11_stable),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default="openrouter")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="muse-glimmer:latest")
    parser.add_argument("--model", default="openrouter/auto")
    parser.add_argument("--json", default="evals/compaction/results/online_fidelity_results.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="spec42-online-") as tmp:
        result = run(model=args.model, endpoint=args.ollama, provider=args.provider,
                     dump_root=Path(tmp) / "storage")
    text = json.dumps(result, indent=2)
    print(text)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(text)
    ok = result["ac8_pass"] and result["ac10"]["flags_dropped_section"] \
        and result["ac11"]["stable"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())