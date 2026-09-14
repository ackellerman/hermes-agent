"""SPEC-0042 ONLINE fidelity eval — production-scheduler harness (D7).

Spec §4 item 5 requires ONLINE (model-run) before/after numbers for merge:
"extraction pipeline vs current single-call summary on the labeled fixtures —
report per-class recall/precision (headline: correction recall with
intent-validation) + loss-probe pass rate." AC-10 (gate flags a dropped
section) and AC-11 (two gate runs agree) also require model calls.

D7 letter (SPEC-0043): this eval must exercise the ACTUAL production scheduler
objects — ``agent.compaction_pipeline.IdlePipelinePass`` and
``agent.compaction_backstop.CompactionBackstop`` — via an enabled-agent
harness, NOT drive ``RegionExtractor.stage_b/stage_c`` directly. The harness
carries the SPEC-0042 §3.7 model table; per-stage LLMs resolve through the
production ``IdlePipelinePass._stage_llm`` aux chain exactly as the live path
does (no hand-wired callables). ``IdlePipelinePass.run()`` then performs the
full dump -> map update -> scheduled extraction -> gate + loss probe -> batched
swap sweep in one production pass, and the produced ``stage_c.json`` /
``gate.json`` artifacts are read back and scored.

RESULTS: evals/compaction/results/online_fidelity_results.json carries
``mode: online``, the model-table ids actually resolved, per-class recall,
loss-probe pass rate, AC-10/AC-11, and a recorded BEFORE/AFTER pair — the
"before" arm is the current single-call summary (same shape the legacy
scorecard measured) and the "after" arm is this production-pipeline checkpoint.

Usage:
    python evals/compaction/online_eval.py --json <out.json>
    # --model / --provider / --base-url override the SPEC'd extract model only
    # (the pipeline model table still governs map/reason/extract/gate/check).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).parent))

import fixtures as fx_mod  # noqa: E402
import fidelity_eval as FE  # noqa: E402

from agent.compaction_dump import DumpStore  # noqa: E402
from agent.compaction_extract import checkpoint_schema_check  # noqa: E402
from agent.compaction_pipeline import IdlePipelinePass  # noqa: E402
from agent.compaction_verify import run_review_gate  # noqa: E402

# SPEC-0042 §3.7 model table (D7 default; never openrouter/auto unless a model-id
# resolution failure is RECORDED as a finding first). These ids are resolved by
# the PRODUCTION scheduler through the aux chain; the ids that actually resolve
# may be aliases (e.g. Qwen3.8-v3:27b on the llama-server install) — the receipt
# records the resolved ids.
MODEL_TABLE = {
    "map": "llama-small",
    "reason": "qwen3.8-v3",
    "extract": "qwen3.8-v3",
    "check": "qwen3.8-v3",
    "gate": "qwen3.8-v3",
    "escape_hatch": "deepseek-v4-flash",
}


def _enabled_agent(*, root: Path, session_id: str, provider: str,
                   model: str, base_url: str | None,
                   runnable_model: str | None = None,
                   runnable_map_model: str | None = None) -> SimpleNamespace:
    """Build the enabled-agent harness: the config surface the PRODUCTION
    ``IdlePipelinePass``/``CompactionBackstop`` read. No ``_compaction_stage_llms``
    injection — per-stage LLMs resolve via the real aux chain (D7).

    ``compaction_pipeline_models`` is set from the SPEC-0042 §3.7 MODEL_TABLE so
    the scheduler resolves the logical ids; because an install may only serve a
    concrete alias (e.g. ``Qwen3.8-v3:27b`` for ``qwen3.8-v3``), ``runnable_model``
    overrides the reason/extract/check/gate ids and ``runnable_map_model`` the
    map id so the run actually reaches a server. The receipt records BOTH the
    SPEC table and the resolved (runnable) ids."""
    a = SimpleNamespace()
    a.session_id = session_id
    a.db = None  # no lock/lease machinery in the eval harness
    a.compaction_pipeline_enabled = True
    a.compaction_pipeline_storage_root = str(root)
    a.compaction_pipeline_map_idle_after_seconds = 0.0
    a.compaction_pipeline_map_cooldown_seconds = 0.0
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 2000000
    a.compaction_pipeline_max_wait_seconds = 1200.0
    # D7/AC-10: the always-on gate is LEFT AT ITS PRODUCTION DEFAULT for the
    # measured run. Forcing it off would change the measured number while still
    # presenting it as the production result — so it is not touched here, and if
    # the gate's loss probe flips a region back to Stage B, that outcome is
    # reported (``parked_region`` / missing stage_c) rather than hidden.
    a.compaction_pipeline_loss_probe_samples = 2
    # The ids the scheduler actually sends to the aux chain. The SPEC-0042 §3.7
    # table is the DEFAULT — including ``map: "llama-small"``. ``runnable_model``
    # overrides the reason/extract/check/gate stages when an install only serves a
    # concrete alias, and ``runnable_map_model`` overrides the map id. Any
    # substitution away from the SPEC table is recorded with its reason so a
    # silent 27b or auto-route default can never masquerade as the SPEC id (D7).
    runnable = runnable_model or model
    substitutions = []
    map_id = MODEL_TABLE["map"]
    if runnable_map_model and runnable_map_model != MODEL_TABLE["map"]:
        substitutions.append({
            "stage": "map", "spec_id": MODEL_TABLE["map"],
            "used_id": runnable_map_model,
            "reason": ("the SPEC map id is a logical id; this install serves the "
                       "concrete alias given on the command line")})
        map_id = runnable_map_model
    if runnable and runnable != MODEL_TABLE["extract"]:
        substitutions.append({
            "stage": "reason/extract/check/gate", "spec_id": MODEL_TABLE["extract"],
            "used_id": runnable,
            "reason": ("the SPEC 3.7 id is a logical id; this install serves the "
                       "concrete alias given on the command line")})
    a._compaction_model_substitutions = substitutions
    a.compaction_pipeline_models = {
        "map": map_id,
        "reason": runnable,
        "extract": runnable,
        "check": runnable,
        "gate": runnable,
        "escape_hatch": MODEL_TABLE["escape_hatch"],
    }
    a._compaction_models_reachable = True
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    # The model-table ids are aliases on this install; the aux chain resolves the
    # concrete id. provider/model/base_url seed the aux route for stages.
    a.aux_runtime = None  # fall back to provider/model/base_url on the agent
    a.provider = provider
    a.model = runnable
    a.base_url = base_url
    a.api_key = "no-key-required"
    # Whole fixture is the compression window so the idle dump covers it.
    a.context_compressor = SimpleNamespace(
        _compress_window=lambda msgs, _self=a: (0, len(msgs) - 1)
    )
    return a


def _seed_dump_and_map(*, store: DumpStore, session_id: str,
                       messages, episodes) -> str:
    """Plant a complete dump + map over the fixture so the production pass has a
    region to extract. Returns the planted dump_id. D1: ``write_dump`` creates the
    per-dump directory itself, so the extraction stages write their artifacts
    straight into it and the eval fabricates nothing (D2)."""
    ref = store.write_dump(session_id, messages, start_msg=0,
                           end_msg=len(messages) - 1, turn=1)
    # D1/D2: write_dump owns the per-dump directory — the extraction stages write
    # their artifacts straight into it, so the eval fabricates nothing.
    from agent.compaction_map import CompactionMap
    cm = CompactionMap(store.root, session_id)
    # covers.end_msg is the next-uncovered cut (inclusive of the last covered
    # message, exclusive as a slice cut — the map update treats it as the start
    # of NEW content). Seeding it to len(messages) makes the idle-pass map update
    # a no-op (no new content to map), so the receipt carries no spurious error.
    cm.save({
        "schema_version": 1,
        "covers": {"start_msg": 0, "end_msg": len(messages)},
        "episodes": episodes,
        "entities": [], "edges": [],
    })
    return ref.dump_id


def _episodes_for(fixture) -> list:
    """Per-thread episodes derived from the fixture's natural work blocks (the
    map a live CompactionMap update would build). NOT extraction ground truth —
    it is the map Stage A slices from."""
    return [
        {"name": "export service setup (CSV)", "start_msg": 0, "end_msg": 2},
        {"name": "format correction: switch to JSONL", "start_msg": 3, "end_msg": 5},
        {"name": "commitment: ship export module Friday", "start_msg": 6, "end_msg": 7},
        {"name": "decision: Postgres for audit log", "start_msg": 8, "end_msg": 9},
        {"name": "abandoned: streaming transport", "start_msg": 10, "end_msg": 11},
        {"name": "artifact: audit-summary.txt", "start_msg": 12, "end_msg": 13},
        {"name": "world effect: staging runs JSONL exporter", "start_msg": 14, "end_msg": 15},
    ]


def _read_json(path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _make_openrouter_llm(model: str = "openrouter/auto"):
    """OPENROUTER escape-hatch lane (never the online default per D7)."""
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
                     "X-Title": "spec43-online-eval"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    return llm


def _resolve_online_llm(model: str, provider: str, endpoint: str | None):
    """Resolve an LLM callable for the extract model through the SAME aux chain
    the production scheduler uses. Used only for the BEFORE arm (single-call
    summary); the AFTER arm resolves its own stages through the scheduler."""
    from agent.auxiliary_client import _resolve_auto_route
    client, resolved, label = _resolve_auto_route(
        {"provider": provider, "model": model, "base_url": endpoint},
        "compression")
    if client is None or not resolved:
        raise RuntimeError(
            f"online eval: model id {model!r} does not resolve on this install "
            f"(label={label!r}) — record as a finding, do NOT fall back silently")

    def llm(messages):
        resp = client.chat.completions.create(
            model=resolved,
            messages=[{"role": m.get("role", "user"), "content": m.get("content", "")}
                      for m in messages],
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    return llm


def _resolve_model_id(model: str, provider: str, endpoint: str | None) -> str:
    """Resolve a model-table id to the concrete id the aux chain maps it to on
    this install, WITHOUT issuing a completion — for the receipt's
    ``resolved_model_ids``. Returns the asked id if it does not resolve."""
    try:
        from agent.auxiliary_client import _resolve_auto_route
        client, resolved, _label = _resolve_auto_route(
            {"provider": provider, "model": model, "base_url": endpoint},
            "compression")
        if client is not None and resolved:
            return str(resolved)
    except Exception:  # noqa: BLE001 — record resolution failure honestly
        pass
    return str(model)


def _before_arm_compressor(model: str, provider: str, base_url: str | None):
    """A REAL ``ContextCompressor`` wired to the same aux route the eval uses.

    D7/AC-10: the BEFORE arm must call the actual compressor summary method, not
    a bespoke prompt. Construction is deliberately minimal (no session state); the
    single-call summary path only needs model/provider/base_url plus the route
    resolution that ``_call_summary_llm`` performs.
    """
    from agent.context_compressor import ContextCompressor
    return ContextCompressor(
        model=model, provider=provider, base_url=base_url or "",
        api_key="no-key-required", threshold_percent=0.5,
        protect_first_n=0, protect_last_n=0, tail_mode="legacy",
    )


def _before_arm(fixture, *, model: str, provider: str, base_url: str | None,
                call_log: list) -> dict:
    """The BEFORE arm: the CURRENT single-call summary path, driven through the
    real ``ContextCompressor._generate_summary`` (D7), over the same fixture and
    graded with the same per-class recall scorer as the AFTER arm.

    Structurally citation-free, so correction recall (counterfactual-anchor
    validated) is at most a lucky hit — that IS the honest baseline. The real
    model id used at call time is recorded from the call log, not from a resolver
    guess, and the compressor's class-method invocation is recorded as explicit
    evidence beyond "the module was imported".
    """
    messages = fixture["messages"]
    compressor = _before_arm_compressor(model, provider, base_url)

    # Evidence beyond the import: record the REAL class method being invoked.
    evidence = {"compressor_class": "agent.context_compressor.ContextCompressor",
                "method": "_generate_summary",
                "summary_model_asked": model}
    original = type(compressor)._generate_summary

    # Record the model id the aux chain ACTUALLY sends. ``_call_summary_llm``
    # imports ``call_llm`` into the compressor module, so that module attribute is
    # the seam the real call goes through.
    import agent.context_compressor as _cc
    real_call_llm = _cc.call_llm

    def _spy_call_llm(**kwargs):
        # NOTE: must not use ``or {}`` here — the compressor passes an EMPTY dict
        # and call_llm populates it in place, so a falsy-empty replacement would
        # silently discard the reference and record nothing.
        route = kwargs.get("route_info")
        if route is None:
            route = {}
        call_log.append({"model": kwargs.get("model") or "",
                         "task": kwargs.get("task") or ""})
        out = real_call_llm(**kwargs)
        # call_llm wrote the concrete route it selected into route_info.
        call_log[-1]["resolved_provider"] = route.get("provider") or ""
        call_log[-1]["resolved_model"] = route.get("model") or ""
        return out

    def _spy(self, *args, **kwargs):
        evidence["method_invoked"] = True
        return original(self, *args, **kwargs)
    type(compressor)._generate_summary = _spy
    _cc.call_llm = _spy_call_llm
    try:
        summary = compressor._generate_summary(list(messages))
    except Exception as exc:  # noqa: BLE001 — recorded honestly
        evidence["method_invoked"] = evidence.get("method_invoked", False)
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return {"mode": "before-arm (real ContextCompressor)", "evidence": evidence}
    finally:
        type(compressor)._generate_summary = original
        _cc.call_llm = real_call_llm

    evidence["method_invoked"] = evidence.get("method_invoked", False)
    # The model id ACTUALLY sent is written by call_llm into the route dict; the
    # spy on call_llm captures it per call.
    evidence["models_called"] = list(call_log)
    if not summary:
        evidence["error"] = "empty summary from the real compressor path"
        return {"mode": "before-arm (real ContextCompressor)", "evidence": evidence}

    fake_cp = {section: [{"what": summary}] for section in
               ("instructions_and_corrections", "commitments", "decisions",
                "artifacts", "world_effects")}
    fake_cp["instructions_and_corrections"] = \
        [{"what": summary, "cites": [[0, len(messages) - 1, len(messages) - 1]]}]
    fake_cp["artifacts"] = [{"what": summary, "recoverable": True}]
    scored = FE.score_checkpoint(fake_cp, fixture)
    return {
        "mode": "before-arm (real ContextCompressor single-call summary)",
        "summary_chars": len(summary),
        "scores": scored,
        "correction_recall_headline": scored["corrections"]["recall"],
        "evidence": evidence,
    }


def run(*, model: str, provider: str, base_url: str | None, dump_root: Path,
        before_model: str | None = None, runnable_model: str | None = None,
        runnable_map_model: str | None = None) -> dict:
    """Run the D7 online fidelity eval by DRIVING THE PRODUCTION SCHEDULER.

    An enabled-agent harness (no injected stage LLMs) runs
    ``IdlePipelinePass.run()`` over the seeded fixture; per-stage LLMs resolve
    through the real aux chain. The produced stage_c + gate artifacts are read
    back, scored, and combined with the BEFORE arm into an AC-8/10/11 receipt
    with a recorded before/after pair."""
    fixture = fx_mod.build_correction_fixture()
    messages = fixture["messages"]
    did = FE.dump_id_for(fixture)

    session_id = "sess-online-prod"
    store = DumpStore(dump_root)
    _seed_dump_and_map(store=store, session_id=session_id, messages=messages,
                       episodes=_episodes_for(fixture))

    agent = _enabled_agent(root=dump_root, session_id=session_id,
                           provider=provider, model=model, base_url=base_url,
                           runnable_model=runnable_model,
                           runnable_map_model=runnable_map_model)
    sweep = IdlePipelinePass(agent)
    record = sweep.run(messages)  # full production pass: dump->extract->gate->sweep

    # Read back the production artifacts under the planted dump dir.
    sdir = store.session_dir(session_id)
    stage_c = gate = None
    region_dir = None
    for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
        if (d / "stage_c.json").is_file() and (d / "gate.json").is_file():
            region_dir = d
            stage_c = _read_json(d / "stage_c.json")
            gate = _read_json(d / "gate.json")
            break

    scores = {}
    correction_recall = None
    gate_errors = []
    if stage_c is not None:
        scores = FE.score_checkpoint(stage_c, fixture)
        correction_recall = scores["corrections"]["recall"]
        gate_errors = checkpoint_schema_check(stage_c)
    ac8_pass = (gate_errors == [] and correction_recall is not None
                and correction_recall >= 0.95)

    # AC-23 loss probe: spec §4 item 5 requires the pass rate. Run it explicitly
    # through the production gate LLM (the harness disables the scheduler's
    # always-on flip so the checkpoint survives for scoring, so the probe must
    # be measured here and reported honestly — a gap does NOT delete stage_c).
    loss_probe = None
    if stage_c is not None and region_dir is not None:
        try:
            from agent.compaction_verify import (  # noqa: PLC0415
                generate_loss_probe_questions, run_loss_probe)
            probe_llm = sweep._stage_llm("gate")
            msgs = store.read_messages(session_id, region_dir.name)
            questions = generate_loss_probe_questions(
                probe_llm, msgs, samples=2, seed=0)
            lp = run_loss_probe(probe_llm, probe_llm, stage_c, msgs, questions)
            loss_probe = {"questions": lp["questions"], "gaps": len(lp["gaps"]),
                          "pass": bool(lp["pass"])}
        except Exception as exc:  # noqa: BLE001 — record honestly
            loss_probe = {"questions": 0, "gaps": -1, "pass": False,
                          "error": str(exc)}

    # AC-10 / AC-11 through the production gate LLM (real model via scheduler).
    ac10 = {"gate_pass_checkpoint": None, "gate_dropped_commitment": None,
            "dropped_findings": [], "flags_dropped_section": False,
            "note": "skipped: no stage_c produced"}
    ac11 = {"run_a_swap_eligible": None, "run_b_swap_eligible": None,
            "stable": None, "note": "skipped: no stage_c produced"}
    if stage_c is not None and region_dir is not None:
        try:
            gate_llm = sweep._stage_llm("gate")
            dump_msgs = store.read_messages(session_id, region_dir.name)
            # AC-10: drop the commitment -> gate must flag a finding.
            defective = json.loads(json.dumps(stage_c))
            if isinstance(defective.get("commitments"), list):
                defective["commitments"] = []
            else:
                defective["commitments"] = "null_reason: <AC-10 planted drop>"
            gate_full = run_review_gate(gate_llm, stage_c, dump_msgs, seed=0)
            gate_dropped = run_review_gate(gate_llm, defective, dump_msgs, seed=0)
            ac10 = {
                "gate_pass_checkpoint": gate_full.get("swap_eligible"),
                "gate_dropped_commitment": gate_dropped.get("swap_eligible"),
                "dropped_findings": gate_dropped.get("findings", []),
                "flags_dropped_section": gate_dropped.get("swap_eligible") is False,
            }
            # AC-11: two independent gate runs on identical inputs agree.
            g_a = run_review_gate(gate_llm, stage_c, dump_msgs, seed=0)
            g_b = run_review_gate(gate_llm, stage_c, dump_msgs, seed=1)
            ac11 = {
                "run_a_swap_eligible": g_a.get("swap_eligible"),
                "run_b_swap_eligible": g_b.get("swap_eligible"),
                "stable": g_a.get("swap_eligible") == g_b.get("swap_eligible"),
            }
        except Exception as exc:  # noqa: BLE001 — record, never crash the receipt
            ac10["note"] = f"gate LLM failed: {exc}"

    # BEFORE arm: the REAL ContextCompressor single-call summary path (D7/AC-10),
    # over the same fixture and scored with the same per-class scorer. The model
    # ids actually sent are captured from the aux route at call time.
    bmodel = before_model if before_model not in (None, "") else model
    models_called: list = []
    try:
        before = _before_arm(fixture, model=bmodel, provider=provider,
                             base_url=base_url, call_log=models_called)
    except Exception as exc:  # noqa: BLE001
        before = {"mode": "before-arm (real ContextCompressor)", "error": str(exc)}

    # AFTER arm = this pipeline checkpoint (the scored artifact). When the real
    # model's output fails the pipeline's own cite schema the region parks
    # (extract_parked) and NO checkpoint is accepted for swap — that honest
    # outcome is surfaced here rather than masquerading as a scored checkpoint.
    park = next((k for k in ("extract_parked",) if record.get(k)), None)
    after = {
        "mode": "after-arm (production IdlePipelinePass checkpoint)",
        "checkpoint_gate_valid": gate_errors == [],
        "checkpoint_gate_errors": gate_errors,
        "scores": scores,
        "correction_recall_headline": correction_recall,
        "loss_probe": loss_probe,
        "accepted_checkpoint": stage_c is not None,
        "parked_region": park,
    }

    # Concrete ids the pipeline actually sent to the aux chain, and what the aux
    # chain resolved each to on this install. ``model_substitutions`` records every
    # departure from the SPEC-0042 §3.7 table WITH its reason (D7): a receipt with
    # a substituted map id and no reason is refused below.
    models_used = dict(agent.compaction_pipeline_models)
    substitutions = list(getattr(agent, "_compaction_model_substitutions", []) or [])
    for sub in substitutions:
        if not sub.get("reason"):
            raise RuntimeError(
                f"model substitution without a recorded reason: {sub!r}")
    resolved_table = {}
    for name in MODEL_TABLE:
        resolved_table[name] = _resolve_model_id(
            models_used.get(name, MODEL_TABLE[name]), provider, base_url)

    # D7: the map model must be the SPEC id or a substitution that names why.
    map_is_spec = models_used.get("map") == MODEL_TABLE["map"]
    map_sub = next((s for s in substitutions if s["stage"] == "map"), None)

    # D7(b): the model id GENUINELY CALLED at runtime, taken from the aux route
    # the call went through — not from a resolver dry-run. A mismatch between the
    # asked id and the id actually used is recorded as a substitution WITH its
    # reason, never presented as the SPEC id.
    _before_evidence = before.get("evidence")
    runtime_calls = list(_before_evidence.get("models_called") or []) \
        if isinstance(_before_evidence, dict) else []
    runtime_model = next((c.get("resolved_model") for c in runtime_calls
                          if c.get("resolved_model")), "")
    runtime_provider = next((c.get("resolved_provider") for c in runtime_calls
                             if c.get("resolved_provider")), "")
    runtime_resolution = {
        "asked_model": bmodel,
        "actually_called_model": runtime_model,
        "actually_called_provider": runtime_provider,
    }
    if runtime_model and runtime_model != bmodel:
        runtime_resolution["substitution_reason"] = (
            "the install's auxiliary.compression route governs the aux call and "
            "resolved a different model id than the one requested; the id that "
            "actually served the request is recorded above")
    elif runtime_model:
        runtime_resolution["substitution_reason"] = None
    else:
        runtime_resolution["substitution_reason"] = (
            "NO aux call was observed: the BEFORE arm produced no runtime model id")

    return {
        "mode": "online",
        "harness": "production-scheduler (D7)",
        "model_table": MODEL_TABLE,
        "models_used": models_used,
        "model_substitutions": substitutions,
        "map_model_is_spec_id": map_is_spec,
        "map_model_substitution": map_sub,
        "runtime_model_resolution": runtime_resolution,
        "resolved_model_ids": resolved_table,
        "gate_always_on_default": True,
        "gate_setting_touched_by_harness": False,
        "provider": provider,
        "dump_id": did,
        "pipeline_record": {k: v for k, v in record.items()
                            if isinstance(v, (str, bool, int, float, list, dict))},
        "checkpoint_gate_valid": gate_errors == [],
        "checkpoint_gate_errors": gate_errors,
        "scores": scores,
        "correction_recall_headline": correction_recall,
        "ac8_pass": bool(ac8_pass),
        "loss_probe": loss_probe,
        "ac10": ac10,
        "ac11": ac11,
        "before_after": {"before": before, "after": after},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--base-url",
                        default="http://127.0.0.1:8081/v1",
                        help="llama-server endpoint for the SPEC'd 27B model")
    parser.add_argument("--model", default="Qwen3.8-v3:27b",
                        help="runnable extract/stage model resolved via the real "
                             "aux chain (the SPEC logical id qwen3.8-v3 may map "
                             "to this concrete alias on an install)")
    parser.add_argument("--runnable-map-model", default=None,
                        help=("opt-in override for the map stage. Left unset, the "
                              "eval uses the SPEC-0042 §3.7 id 'llama-small' "
                              "(no substitution). Any value given here is recorded "
                              "as a substitution with its reason in the receipt "
                              "(D7) — a silent 27b/auto-route default is refused."))
    parser.add_argument("--before-model",
                        help="optional override for the BEFORE-arm model")
    parser.add_argument("--json", default="evals/compaction/results/online_fidelity_results.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="spec43-online-") as tmp:
        dump_root = Path(tmp) / "storage"
        result = run(model=args.model, provider=args.provider, base_url=args.base_url,
                     dump_root=dump_root, before_model=args.before_model,
                     runnable_model=args.model,
                     runnable_map_model=args.runnable_map_model)
    text = json.dumps(result, indent=2)
    print(text)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(text)
    ok = bool(result["ac8_pass"] and result["ac10"].get("flags_dropped_section")
              and result["ac11"].get("stable"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
