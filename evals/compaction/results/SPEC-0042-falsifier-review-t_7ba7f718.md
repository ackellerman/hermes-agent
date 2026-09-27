# SPEC-0042 falsifier verification report — t_7ba7f718 (review of wt/t_a18725d8 @ 0123cee7c)

Method: executed each AC's own falsifier from the spec verbatim against the impl
branch (merged into this worktree; impl content bit-identical to
origin/wt/t_a18725d8). Harness: /tmp/spec0042_falsifiers.py, /tmp/spec0042_round2.py,
/tmp/spec0042_round3.py, /tmp/spec0042_round4.py. Test suite via scripts/run_tests.sh.

## Verdict: FAIL — blocked from merge. Three findings, two of them spec violations.

## Per-AC results

| AC | Falsifier executed | Result |
|----|--------------------|--------|
| AC-1 | 500-msg dump round-trip byte identity; incomplete meta -> require_complete refuses (DumpIncompleteError) | PASS |
| AC-2 | dump twice -> identical region_hash8 (ids 0003-71ec3f13 both) | PASS |
| AC-3 | IoU machinery exact per spec formula (exact=1.0, off-by-1=0.833) | PASS (mechanism) |
| AC-4 | corrupt map episode ("zanzibar quantum flotilla") -> sampled audit flags it | PASS |
| AC-5 | update-call input bounded — no runnable falsifier in impl | NOT EXECUTABLE (F1) |
| AC-6 | delete section -> checkpoint_schema_check rejects | PASS (but see F2) |
| AC-7 | plant unknown-dump citation -> item_citation_supported false | PASS |
| AC-8 | fidelity eval: correction recall 1.0 + anchor match ±1 | PASS OFFLINE only (F3) |
| AC-9 | >50-token quote -> validator rejects | PASS (but see F2) |
| AC-10 | gate flags 3-class fixture removal | MODEL CALL REQUIRED — unexecuted (F3) |
| AC-11 | two gate runs agree | MODEL CALL REQUIRED — unexecuted (F3) |
| AC-12 | check_alternation_invariant 4 cases (valid + 3 violation classes) | PASS (all four, incl. orphan tool result) |
| AC-13 | batched swap = 1 mutation | PASS (tests/agent/test_compaction_swap.py test_ac13_batched_swap_one_mutation) |
| AC-14 | (1) distinct DDL tables; (2) both-direction contention | PASS live (both directions, real SessionDB + sqlite) |
| AC-15 | models unreachable -> degrade, telemetry reason | PASS (gate level: model_unreachable/extraction_incomplete/lock_timeout all correct) |
| AC-16 | unknown dump id -> machine-readable DumpNotFoundError JSON | PASS |
| AC-17 | fresh-instance read == pre-restart read | PASS |
| AC-18 | dump deleted -> transcript-fallback serves, FallbackRecord logged | PASS |
| AC-19 | enabled=false (default) -> legacy_summary, ignores held lock, no pipeline | PASS (OFF case) |
| AC-19b | ON + degrade -> byte-identical to OFF | NOT FULLY EXECUTABLE (F4) |
| AC-19c | sizing probe: map 4878/64K (92.4% headroom), B/C 7772/256K (97% headroom) | PASS |

Test suite: scripts/run_tests.sh on the 8 compaction test files = 63/63 PASS.
Full-suite run not performed (time); impl branch's own suite state unknown to this review.

## Findings (blocking)

**F1 (AC-5, AC-3 numeric bar) — falsifiers never executed, PASS claimed by construction.**
AC-3's 90% IoU bar on 10 consecutive updates vs hand-labeled ground truth has no harness:
no ground-truth episode labels exist anywhere in the impl, and CompactionMap.update
consumes an llm_call, so the falsifier needs a model run. AC-5 (sum of update inputs
<= 1.15 x region) likewise has no instrumented run. Both mechanisms are correct in code;
the falsifiers themselves were never run. Reported as NOT EXECUTABLE, not PASS.

**F2 (AC-6/AC-9 — evidence of internal inconsistency in the validator).** In my
harness a hand-built valid checkpoint FAILED checkpoint_schema_check with 9 errors:
every empty section demanded `null_reason` (the spec requires null_reason only for
sections the extractor deliberately left empty — "an empty section lacking an explicit
null_reason fails", not "any empty section fails") and cites were rejected as
"not (dump_id, start, end)" when the eval's own fixtures use 2-element [start, end]
cites. The impl's validator and the impl's eval fixtures disagree on cite arity.
Consequence: the fidelity eval's 1.0 recall is scored against a checkpoint form the
pipeline's own gate would reject. One of the two is wrong; the merge is blocked until
they agree.

**F3 (AC-8/AC-10/AC-11, spec §4 items 3 and 5) — the required runs were not run.**
- The fidelity eval has no ONLINE mode exercised: results/fidelity_results.json itself
  states "offline mode validates scoring machinery; spec §4 item 5 requires the ONLINE
  run for merge numbers". The committed "before/after numbers" are machinery
  self-checks (rule-based checkpoint scored against its own labels — a harness that
  sources its input from the labels cannot fail on extraction defects).
- No Docker soak evidence exists anywhere in the impl (spec §4 item 3): no replay
  harness for the pipeline over real transcripts, no soak results, no invariant
  assertions over multi-hour simulated traffic.
- AC-10/AC-11 (review gate behavior, verdict stability) require model calls; nothing in
  the impl or results exercises them. The evals/compaction/README.md harness predates
  this spec and measures the OLD compressor, not the checkpoint pipeline.

**F4 (AC-19b byte-identity) — wiring unproven at the integration point.** backstop_gate
exists and degrades correctly (verified live, all three degradation_reasons), but grep
shows NO production caller: `backstop_gate` and `swap_region` are referenced only by
their own tests and agent_init.py comments. The backstop is not wired into the real
overflow path (conversation_compression.py / context_compressor.py untouched by the
impl, +0/-0). With the pipeline "enabled", nothing in the live agent runs it; the ON
case is unreachable in production. AC-19b's byte-identity diff therefore cannot fail,
which means it also cannot pass. Related: compaction_pipeline.py's IdlePipelinePass is
reached only via turn_context_compaction._pipeline_idle_sweep (the one real seam) —
that seam is live, the swap/backstop seam is not.

## Non-blocking notes

- AC-14 verdict: PASS, and worth stating plainly — two distinct CREATE TABLE statements
  (hermes_state_common.py:463 compression_locks, :465 compaction_pipeline_locks), the
  pipeline lock methods operate only on the distinct table, and live contention holds in
  BOTH directions against a real SessionDB (compression lease blocks pipeline pass;
  pipeline lock blocks compression; release unblocks; mutex across holders verified).
- AC-19c PASS: measured 4878 map-input tokens vs 64K slot (92.4% headroom) and 7772/3296
  B/C tokens vs 256K slot (97% headroom) on the 500-msg heavy-tool-use fixture — well
  clear of the 20% bar. Probe rerun live, output matches the committed receipt.
- citation-health/self_review.sh confirmed irrelevant here per card body; not assessed.
- Local-only behavior verified where executable; the model-dependent falsifiers need a
  reachable aux route (see F3) and are the remaining gate for merge.
