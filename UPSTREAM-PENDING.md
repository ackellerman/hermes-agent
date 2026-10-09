# Upstream-pending Hermes patches

This ledger lists every patch the fork's `development` branch carries on top of upstream `main`.
Rule: every patch `development` carries has a row here with a removal condition. A carried patch
without a row is a defect in this ledger, not an untracked feature. Each row needs an upstream PR
or a dated operator-approved exception. Remove the local patch (and its row) once upstream has merged
an equivalent and the emitted behaviour is verified.

| Upstream PR | Local commit | Purpose | Added | Remove when |
|---|---|---|---|---|
| No upstream PR yet (operator-gated) | 2b1a4698e3 | `venv_sync.relaunch_command` refuses to relaunch a stdin program (`argv[0] == '-'`): upstream re-enters it as `runpy.run_path('<cwd>/-')`, a deterministic `FileNotFoundError` on every resume. Test: `tests/hermes_cli/test_update_launch_completion.py::test_relaunch_refuses_stdin_program`. | 2026-09-26 | Upstream `relaunch_command` refuses or correctly handles `argv[0] == '-'`. |
| https://github.com/NousResearch/hermes-agent/pull/102341 (ported) | 28924dcb98 | `provider_routing` passes unknown keys through to the OpenRouter `provider` object instead of dropping them. Test: `tests/agent/test_provider_routing_passthrough.py`. | 2026-09-26 | #102341 or an equivalent passthrough merges upstream. |
| No upstream PR yet (operator-gated) | d86d496c8c | Worktree GC never deletes the trunk branch the repo is measured against (`development` in `_PROTECTED_BRANCHES` plus the resolved trunk name), at both delete sites. Test: `tests/hermes_cli/test_worktree_gc.py`. | 2026-09-28 | Upstream protects the measured-against trunk branch (not only `main`/`master`) in worktree GC. |
| No upstream PR yet (operator-gated) | a887a1f3f0 | Codex credential-pool selection skips a row whose access token is expiring, so a dead high-priority row no longer shadows a live one; keeps an all-expired fallback so refresh can still repair it. Test: `tests/hermes_cli/test_auth_codex_provider.py`. | 2026-09-28 | Upstream `_pool_codex_credential` selection is expiry-aware. |
| No upstream PR yet; residual of 2d0aa1c7af (+89f24cc4a0) after upstream 9e78058584 / 6590f13a1b | 2d0aa1c7af | The web recommended-default (`hermes_cli/web_routers/models.py`) builds with `for_picker=True, fast_custom_probe=False`, so a provider whose whole pool is in cooldown stays eligible. The rest of the picker patch is upstream now. Test: `tests/hermes_cli/test_authenticated_providers_exhausted_pool.py::test_recommended_default_retains_all_cooldown_provider`. | 2026-09-28 | Upstream's recommended-default passes `for_picker=True` (or keeps all-cooldown providers by another route). |
| No upstream PR yet (closest: #87418, API-server context fields, different surface) | b1b67201a4..0cd0d95050 (4 commits); survived the 2026-10-07 upstream sync merge as a union in `agent/conversation_loop.py` | One-shot callers get the TUI/desktop context gauge's end-of-turn reading: `run_conversation` stamps `context_usage` (`context_usage_fields(compressor)`, None when unknown, never fails the turn) on every run result; `--format stream-json` `result` carries `context {used,max,percent,source,estimated}` (always present, null when unknown); `-z --usage-file` report gains the same `context` block. Additive keys only. | 2026-10-07 | Upstream exports the context reading on stream-json/usage-file (or this lands as a PR) with the same values as the gauge. |
| Fork-local, permanent (not upstreamable) | 791d46ac8b (+ 3ebe9fc30c, 096c9e5c32, 1614378402) | `.githooks/` pre-push gate: refuses a push whose new history adds host paths, local topology or credentials (patterns in `.githooks/patterns/`). | 2026-09-22 | Never while this fork is published from a workstation; it is fork tooling, not a Hermes change. |
| No upstream PR yet (operator-gated) | 80224d0bb3 | Test-only: pins the older minted POSIX launcher shape (first line "import os, re, sys", no closure repair) as a gateway for looks_like_gateway_command_line and live_gateway_pid_for_home; upstream's identity test covers only the current _launcher_script shape. | 2026-10-08 | Upstream's identity tests cover the pre-closure-repair minted launcher (or this test is upstreamed), or no supported install still runs a v1-minted launcher. |

Not carried any more: the kanban patches (K1, K2, Model-B `request_changes`, SPEC-0032 board
resolution), the memory-eviction hook and the compaction work are preserved on the `parked/kanban`,
`parked/memory-evict` and `parked/compaction` branches. The former K4/K5 kanban behaviours (bounded
`kanban_show`, default kanban turn budget) moved to an external governance plugin and need no kernel
patch. The kanban stop-guard terminal tools landed upstream. The TUI `session.compress` timeout
patch is no longer carried; re-landing it is a separate follow-up.
