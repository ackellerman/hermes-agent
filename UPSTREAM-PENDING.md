# Upstream-pending Hermes patches

This ledger lists the narrow local commits carried by `local/hermes-overlay`.
Every entry needs an upstream PR or a dated operator-approved exception. Remove
the local patch after upstream has merged it and the emitted behavior is verified.

| Upstream PR | Local commit | Purpose | Added | Remove when |
|---|---|---|---|---|
| https://github.com/NousResearch/hermes-agent/pull/97753 (source `d8702aa07577`) | 8470a5a0dc | Treat `kanban_request_review` and `kanban_request_changes` as terminal in the stop guard; preserve a non-terminal-tool regression control. | 2026-09-02 | PR merged and the live upstream source contains both terminal tools. |
| OPERATOR EXCEPTION (2026-09-02, ackellerman) — no upstream PR; hermes-longterm-memory custom sink seam | 0bcce0a942 | Add `on_memory_evict` ABC hook (memory_provider.py) + `_notify_memory_evict` fan-out (memory_manager.py): archive a would-be-dropped built-in memory write into the hermes_longterm_memory fact store on terminal failure (add/replace only, exceptions contained). Wire the on_memory_evict implementation (`LongTermMemoryProvider`) from `hermes-longterm-memory` (design repo, main d983731). | 2026-09-02 | Dated operator exception; revisit if upstream adds a native eviction hook. |
| OPERATOR EXCEPTION (2026-09-04, ackellerman) — no upstream PR by instruction ("no upstream") | 5ac513fa71 | TUI: `session.compress` RPC gets the desktop's 660s budget instead of the generic 120s, and its timeout message says compaction is still running / do not re-run. | 2026-09-04 | Upstream ui-tui gives session.compress a per-method timeout ≥ 660s, or operator lifts the no-upstream instruction and a PR lands. |
| https://github.com/NousResearch/hermes-agent/pull/101372 (maximalang; same fix, param `dependency_ids`; confirmed by comment 2026-09-05) — local K1 per hermes-gates plan §0a | 5b53fe05d6 | `block_task(kind="dependency")` requires `depends_on=[ids]` and writes the `task_links` edge in the same txn; refuses missing/unknown/done/self/cycle with the repair named. Tool + CLI pass it through. Kills the prose-dependency respawn loop (154/167 blocked runs, 268M tokens on 2026-09-04). | 2026-09-04 | Upstream makes dependency blocks edge-backed, or PR from this commit merges. |
| https://github.com/NousResearch/hermes-agent/pull/96733 (anombyte93; same diagnosis, still spends worker turns; re-judge variant suggested by comment 2026-09-05) — local K2 | feeef61ef0 | `run_kanban_goal_loop`: judge `transport_failed` retries the judge (backoff, injectable) and never spends a worker turn; after 5 → block `transient` naming the judge. Was 19 wasted turns/session ×3. | 2026-09-04 | Upstream handles judge transport failure in the kanban loop. |

## Removed from the overlay (2026-09-04, ackellerman: "1. do; 3. yes")

| Was | Why removed | Where it lives now |
|---|---|---|
| f1e0a5b74d board `orchestrator_profile` + `set-orchestrator` | The orchestrator seat is retired; `kanban.auto_decompose` is `false` fleet-wide (root + every profile). Dead code with a live hazard (final fallback = active profile). | nowhere — the kernel planner has no role in the fan-out system |
| K3 decompose refuses worked cards | Same: auto-decompose is off; K3 guarded a path that no longer runs. | nowhere |
| K4 `kanban_show` bounded | Pure `transform_tool_result`; no kernel edit needed. | hermes-gates `plugins/hermes-gates-governance/budget.py` (`bound_show_result`), test `tools/gates/tests/test-governance-budget.py` |
| K5 kanban turn default 8 | Pure `pre_tool_call` modify on `kanban_create`; config key `skills.config.gates.kanban_max_turns`. | same module (`default_turn_budget`) |

Overlay after removal: K1 (dependency block requires `depends_on`, writes the edge), K2 (judge transport failure never spends a worker turn), Model-B `request_changes` (d3d4e0e93b1), kanban_stop terminal guard, memory sink seam, TUI compress RPC. Upstream candidates: K1, K2 (main already tracks `consecutive_transport_failures` for the interactive loop but its kanban loop still spends a turn on a judge outage), Model-B.
