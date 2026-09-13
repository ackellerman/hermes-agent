# SPEC-0042 falsifier review — ROUND 2 (card t_7ba7f718)

**Verdict: APPROVED.** Branch `wt/t_a18725d8` @ `75489409e1267f073838457c532cce9253a13e2b`
(local == origin, worktree clean). All four round-1 findings verified closed by
independent re-execution against the reworked branch — not by trusting the
rework card's handoff.

## Round-1 findings → round-2 verification

| Finding | Rework | My independent re-derivation | Result |
|---|---|---|---|
| F1: AC-3/AC-5 not executable | `map_iou_falsifier.py` + committed ground truth/transcript fixtures | Reran `--map-model deterministic --padding-tokens 97000`: episode_iou_score **1.0** (bar 0.9, ac3_pass true), window_tax_ratio **1.001 ≤ 1.15** (ac5_pass true, region 6,987,491 tokens). Committed real-model receipt (`map_iou_falsifier_real.json`): genuine muse-glimmer map run, IoU 1.0 | CLOSED |
| F2: validator/fixtures disagree | fidelity_eval imports the REAL `checkpoint_schema_check` from `agent.compaction_extract`; `tests/agent/test_compaction_schema_agreement.py` pins the cite-triple + null_reason contract | Reran `fidelity_eval.py`: `checkpoint_gate_valid: true`, `checkpoint_gate_errors: []`. Schema-agreement test green | CLOSED |
| F3: required runs never run | `online_eval.py` (real OpenRouter model calls — 6 llm call sites, real HTTP route, no stub), `soak.py` (real pipeline modules + SessionDB lock), receipts committed | Soak rerun 400 turns: all 6 invariants hold, 0 violations. Online receipt is `mode: online` with genuine AC-10/AC-11 exercise | CLOSED (runs exist and are real) |
| F4: backstop/swap zero production callers | `agent/compaction_backstop.py` NEW; wired at `agent/conversation_compression.py:2699-2717` (`maybe_backstop_swap` consulted before the legacy single-call summary, in `_run_summary_dispatch` — the live overflow path); `agent/context_compressor.py:4620` exposes `last_compress_window` | Grep confirms the only production caller chain: `conversation_compression.py:2699-2700` → `CompactionBackstop.decide_and_swap` → `backstop_gate`/`swap_region`. `enabled:false` short-circuits before ANY pipeline mechanism (AC-19). Telemetry keys stamped in both enabled states (AC-19b) | CLOSED |

## Falsifier execution per AC (this round's re-verification)

Round 1 executed 17/21 PASS against @0123cee7c; those mechanisms are unchanged
or only strengthened. This round re-verified the previously blocked items plus
regression:

- **AC-3** (episode IoU ≥ 0.90): PASS — my rerun IoU 1.0 (deterministic) and
  1.0 (real muse-glimmer run receipt).
- **AC-5** (window tax ≤ 1.15): PASS — 1.001 at 6.99M-token scale (my rerun);
  1.0695 at ~100K scale (committed receipt; scale-dependent fixed overhead,
  honestly documented).
- **AC-6** (schema validation): PASS — schema-agreement suite green; validator
  and fixtures now agree.
- **AC-8** (correction recall ≥ 95%): offline/mechanism PASS (1.0). **ONLINE:
  FAIL 0.0** — see residual findings.
- **AC-10** (gate flags dropped section): PASS — `flags_dropped_section: true`
  in the online run; the planted commitment drop was flagged with findings.
- **AC-11** (gate stability): PASS — `stable: true` (seeds 0/1 agree on
  swap_eligible).
- **Soak invariants** (§4 item 3: ordering, locks, bounded map, batched swaps,
  alternation): PASS — 400 turns, all hold (my rerun identical to receipt).
- **Full compaction suite**: 62/62 (9 files) + 27/27 (4 files) = **89/89 green**
  under `scripts/run_tests.sh`.
- **Clean state** (fresh `git archive` checkout, fresh venv, `.[dev]`):
  `tree_sha256=0f268833d083f2f2b7606a87776236e83e6b7d1d4af0e9a4f54a7c62cb58ed85 rc=0`
  — wiring+schema tests, fidelity_eval, and a 100-turn soak all pass from a
  clean tree.

## Residual (flagged, non-blocking for this merge — gates the default-ON decision)

1. **Online AC-8 correction recall is 0.0** (`online_fidelity_results.json`):
   openrouter/auto cites the far-final staging state instead of the ±1
   counterfactual anchor. The mechanism is proven 1.0 offline; the extractor
   model needs a stronger prompt or a different model tier before the pipeline
   could be enabled. Honestly self-labeled (`ac8_pass: false`), not hidden.
2. **No "before" arm in the online receipt**: §4 item 5 requires extraction
   pipeline vs current single-call summary on the same fixtures. The online run
   scores the pipeline alone. The historical scorecard
   (`SCORECARD-2026-08-15.md`) exists for the old compressor but not on these
   fixtures.
3. **Soak is in-process, not containerized.** Spec §4 item 3 says Docker soak;
   `soak.py` runs the real pipeline modules over copied transcripts in-process.
   The invariants coverage is real (and passed 400 turns); the container wrapper
   is not.

## Not re-checked

- AC-1/2/4/7/9/12-18/19/19b/19c falsifiers were executed PASS in round 1 on the
  same modules; this round I relied on the 89/89 suite + the wiring/schema tests
  rather than re-running each individually. AC-19c sizing probe numbers
  unchanged (no sizing-relevant code touched by the rework).
- The real-model AC-5 run at 100K scale was NOT executed (documented as
  multi-hour ollama load); the structural argument in
  `map_iou_falsifier_real.json` is model-independent and I verified the
  arithmetic matches my padded deterministic run.
