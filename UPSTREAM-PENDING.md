# Upstream-pending Hermes patches

This ledger lists the narrow local commits carried by `local/hermes-overlay`.
Every entry needs an upstream PR or a dated operator-approved exception. Remove
the local patch after upstream has merged it and the emitted behavior is verified.

| Upstream PR | Local commit | Purpose | Added | Remove when |
|---|---|---|---|---|
| https://github.com/NousResearch/hermes-agent/pull/97753 (source `d8702aa07577`) | 8470a5a0dc | Treat `kanban_request_review` and `kanban_request_changes` as terminal in the stop guard; preserve a non-terminal-tool regression control. | 2026-09-02 | PR merged and the live upstream source contains both terminal tools. |
| https://github.com/NousResearch/hermes-agent/issues/34977 | f1e0a5b74d | Add board.json `orchestrator_profile` plus `kanban boards set-orchestrator`; board override → global fallback → active profile. | 2026-09-02 | Replaced by a merged upstream PR that provides the same board-scoped resolution and native configuration surface. |
