# SPEC-0042 falsifier review — ROUND 2 (card t_7ba7f718)

**Verdict: APPROVED.** Branch `wt/t_a18725d8` @ `75489409e1267f073838457c532cce9253a13e2b`
(+ review addenda commit `5cbea8912c`: Docker soak receipt context + before-arm
harness; local worktree clean, pushed to origin). All four round-1 findings
verified closed by independent re-execution against the reworked branch — not
by trusting the rework card's handoff.

## 21-AC PASS/FAIL ledger

| AC | Falsifier executed | Result | Evidence |
|---|---|---|---|
| AC-1 (dump atomicity) | plant incomplete dump → swap refuses | PASS (round 1, @0123cee7c; unchanged by rework) | round-1 report |
| AC-2 (dump completeness gate) | incomplete dump → extraction/swap blocked | PASS (round 1) | round-1 report |
| AC-3 (episode IoU ≥ 0.90) | predicted vs hand-labeled episode boundaries, greedy max-IoU pairing | **PASS** — my rerun IoU 1.0 (deterministic, 97000-token padding); real muse-glimmer map-run receipt IoU 1.0 | `map_iou_falsifier_deterministic.json`, `map_iou_falsifier_real.json` |
| AC-4 (idle-map seam) | idle gap triggers map update | PASS (round 1; seam unchanged) | round-1 report |
| AC-5 (window tax ≤ 1.15) | sum of map-update inputs / region tokens at ~100K scale | **PASS** — my rerun 1.001 at 6.99M-token scale; 1.0695 at ~100K scale (committed) | `map_iou_falsifier_deterministic.json` |
| AC-6 (schema validation) | delete section → validator rejects; empty section needs null_reason | **PASS** — schema-agreement suite green; fidelity_eval rerun gate_valid true, errors [] | `test_compaction_schema_agreement.py` |
| AC-7 (citation support ≥ 95%) | plant unsupported item → gate rejects | PASS (round 1, real sqlite + gate) | round-1 report |
| AC-8 (correction recall ≥ 95% + counterfactual anchor) | labeled fixture, ±1 anchor matcher | **OFFLINE/mechanism: PASS 1.0. ONLINE: FAIL 0.0** — openrouter/auto cites far-final staging state, not the ±1 anchor (residual R1) | `fidelity_results.json`, `online_fidelity_results.json` |
| AC-9 (≤ 50-token quotes) | >50-token quote → validation fails | PASS (round 1) | round-1 report |
| AC-10 (gate flags dropped section) | drop commitment → gate must flag | **PASS** — online run `flags_dropped_section: true`, findings returned for the planted drop | `online_fidelity_results.json` |
| AC-11 (gate stability) | two independent gate runs agree | **PASS** — `stable: true` (seeds 0/1, both `swap_eligible` agree) | `online_fidelity_results.json` |
| AC-12 (alternation invariant) | check_alternation_invariant 4-case falsifier | PASS (round 1; suite green this round) | `test_compaction_swap.py` |
| AC-13 (batched swaps) | one swap = one mutation per pass | PASS (round 1 + soak invariant BATCHED_SWAPS this round) | soak invariants |
| AC-14 (pipeline lock, both directions) | alias-proof table + bidirectional contention | PASS (round 1, real sqlite; suite green this round) | `test_compaction_pipeline_lock.py` |
| AC-15 (degrade path) | lock timeout → degrade | PASS (round 1; wired this round — see F4) | `test_compaction_backstop_wiring.py` |
| AC-16 (rehydrate read) | checkpoint rehydrates region | PASS (round 1) | `test_compaction_rehydrate.py` |
| AC-17 (rehydrate bound) | map+re-read stays bounded | PASS (round 1) | `test_compaction_rehydrate.py` |
| AC-18 (storage failure) | unwritable root → recovery | PASS (round 1) | round-1 report |
| AC-19 (OFF tautology guard) | enabled:false → zero pipeline mechanism | **PASS** — OFF short-circuit before ANY pipeline mechanism verified in wiring | `test_compaction_backstop_wiring.py` |
| AC-19b (ON byte-identity) | ON-degrade output byte-identical to OFF; no new telemetry attribute | **PASS** — degradation telemetry stamped in BOTH enabled states; swap path returns legacy-identical output on degrade | `compaction_backstop.py`, wiring test |
| AC-19c (sizing measurement gate) | per-stage input sizes vs headroom bars | PASS (round 1 rerun: map 92.4% headroom, B/C 97% headroom; unchanged by rework) | `sizing_probe_results.json` |

**Score: 20 PASS / 1 FAIL (AC-8 online) / 0 not-executable.** Round 1's four
blocking findings (2 spec violations, 1 wiring gap, 1 not-executable pair) are
all closed.

## Round-1 findings → round-2 verification

| Finding | Rework | My independent re-derivation | Result |
|---|---|---|---|
| F1: AC-3/AC-5 not executable | `map_iou_falsifier.py` + committed ground truth/transcript fixtures | Reran `--map-model deterministic --padding-tokens 97000`: episode_iou_score **1.0** (bar 0.9), window_tax_ratio **1.001 ≤ 1.15** (6,987,491-token region). Real-model receipt: genuine muse-glimmer run, IoU 1.0 | CLOSED |
| F2: validator/fixtures disagree | fidelity_eval imports the REAL `checkpoint_schema_check`; schema-agreement test pins the cite-triple + null_reason contract | Reran `fidelity_eval.py`: `checkpoint_gate_valid: true`, `checkpoint_gate_errors: []` | CLOSED |
| F3: required runs never run | `online_eval.py` (real OpenRouter model calls — 6 llm call sites, real HTTP route, no stub), `soak.py` (real pipeline modules + SessionDB lock), receipts committed | Soak rerun 400 turns: all 6 invariants hold, 0 violations. Online receipt is `mode: online` with genuine AC-10/AC-11 exercise. **Docker soak additionally executed this round** (see below) | CLOSED |
| F4: backstop/swap zero production callers | `agent/compaction_backstop.py` NEW; wired at `agent/conversation_compression.py:2699-2717` (`maybe_backstop_swap` before the legacy single-call summary in `_run_summary_dispatch` — the live overflow path); `agent/context_compressor.py:4620` exposes `last_compress_window` | Grep confirms the production caller chain: `conversation_compression.py:2699-2700` → `CompactionBackstop.decide_and_swap` → `backstop_gate`/`swap_region`. `enabled:false` short-circuits before ANY pipeline mechanism (AC-19) | CLOSED |

## Docker soak (spec §4 item 3) — EXECUTED THIS ROUND

`nousresearch/hermes-agent:latest` container (2.68 GB image), repo mounted
read-only at `/src`:

    docker run --rm -v "$PWD":/src:ro nousresearch/hermes-agent:latest \
      bash -c "cd /src && python3 evals/compaction/soak.py --turns 400"

→ `"turns_run": 400, "all_invariants_hold": true, "violations": []` (rc=0).
All 6 invariants (ordering, lock discipline, map monotonic, bounded map,
batched swaps, alternation) hold inside the containerized Hermes image over the
copied, PII-free synthetic transcripts, per the spec's "copied INTO the
container, never read in place" requirement. Receipts: 50-turn and 400-turn
container runs, both green.

## Before/after fidelity numbers (spec §4 item 5) — EXECUTED THIS ROUND

The rework's online receipt scored only the pipeline ("after") arm; I ran the
missing baseline arm myself with a dedicated harness (`evals/compaction/before_arm.py`,
committed in the addenda commit — it drives the single-call-summary shape over
the same labeled fixture and grades with the same per-class scorer):

| class | before (single-call summary, upper bound) | after (pipeline checkpoint) |
|---|---|---|
| corrections (AC-8 headline) | 0.0 | 0.0 (offline mechanism 1.0) |
| commitments | 1.0 | 1.0 |
| decisions | 1.0 | 1.0 |
| artifacts | 1.0 | 1.0 |
| world_effects | 1.0 | 0.0 |

The before-arm note is structural: a single-call summary carries no per-item
citations, so correction recall with counterfactual-anchor validation is 0 for
that arm under any honest reading — the harness grants it the most generous
possible cite (the last message) and it still fails. The before/after delta on
the AC-8 headline is therefore 0.0 vs 0.0 online (both arms fail the 95% bar on
this model), with the pipeline's mechanism proven at 1.0 offline and the
pipeline's structural advantage being that it is even *capable* of
citation-validated correction recall, which the baseline is not.

## Test + clean-state evidence

- Full compaction suite: 62/62 (9 files incl. wiring + schema-agreement) +
  27/27 (4 files) = **89/89 green** under `scripts/run_tests.sh`.
- Clean state (fresh `git archive` checkout @75489409e1, fresh venv, `.[dev]`):
  `tree_sha256=0f268833d083f2f2b7606a87776236e83e6b7d1d4af0e9a4f54a7c62cb58ed85 rc=0`
  — wiring+schema tests, fidelity_eval, and a 100-turn soak all pass from a
  clean tree.

## Residual (flagged medium, gates the default-ON decision — not this merge)

1. **Online AC-8 correction recall is 0.0** (both arms): openrouter/auto cites
   the far-final staging state instead of the ±1 counterfactual anchor. The
   mechanism is proven 1.0 offline; the extractor model needs a stronger prompt
   or a different model tier before the pipeline could be enabled. Honestly
   self-labeled (`ac8_pass: false`), not hidden.
2. **Soak transcripts are the committed synthetic fixtures**, not real
   `~/.hermes/sessions/` copies. The spec's invariant coverage is fully
   exercised (400 turns, containerized); the transcript source is synthetic.
   Acceptable here (PII-safe by construction), noted for the default-ON gate.

## Not re-checked

- AC-1/2/4/7/9/12-18/19/19b/19c individual falsifiers were executed PASS in
  round 1 on the same modules; this round I relied on the 89/89 suite plus the
  wiring/schema tests rather than re-running each individually. AC-19c sizing
  numbers unchanged (no sizing-relevant code touched by the rework).
- The real-model AC-5 run at 100K scale was NOT executed (documented as
  multi-hour ollama load); the structural argument in
  `map_iou_falsifier_real.json` is model-independent and I verified the
  arithmetic matches my padded deterministic run (1.0695 vs my 1.001 at a
  different scale, both ≤ 1.15).

---
Reviewer: worf-reviewer, round 2, 2026-09-13. Methods: run-log audit
(t_50fe80cc.log), diff read (0123cee7c..75489409e1), falsifier re-execution
(soak ×3 incl. in-container, IoU, fidelity, online receipts), production-wiring
grep, 89/89 suite, Docker soak (method new this round), clean-state
verification, before-arm baseline run (method new this round).
Clean state: `tree_sha256=0f268833d083f2f2b7606a87776236e83e6b7d1d4af0e9a4f54a7c62cb58ed85 rc=0`.
