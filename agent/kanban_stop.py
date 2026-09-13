"""Turn-end guard for kanban workers.

Kanban workers must end with a terminal board tool that hands the card to
whoever owns it next. Models (especially GLM / Qwen families) sometimes
narrate the next step ("Let me write the report now") and stop with
``finish_reason=stop`` and no tool calls. Hermes treats that as a clean
exit -> ``rc=0`` -> dispatcher ``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional


# Every tool that hands the card off and ends this worker's responsibility
# for it — not just the two that close it out.  A build worker legitimately
# finishes with ``kanban_request_review`` (goals.py's continuation and
# finalize prompts tell it to, for code changes needing same-card review),
# and a review agent finishes with ``kanban_request_changes`` (the
# force-loaded sdlc-review skill's decision table tells it to).  Both move
# the task out of ``running``, so treating them as non-terminal fired this
# guard at a worker that had already done the right thing — nudging it to
# call ``kanban_complete`` on a card it must not close.
_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def _terminal_tool_result_succeeded(message: dict) -> bool:
    """True unless a structured tool result explicitly reports an error.

    Terminal lifecycle tools are executed before the next loop-head check. A
    pre-tool governance refusal is represented as a normal ``role=tool`` row
    whose JSON payload has ``error``; attempted is not completed, so it must
    return to the model for repair rather than latch the turn at that row.
    """
    content = message.get("content")
    if not isinstance(content, str):
        return True
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return True
    return not (isinstance(payload, dict) and payload.get("error"))


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True only after a terminal kanban tool has completed successfully.

    A terminal call rejected by policy leaves a tool-error row in the
    transcript. Latching on the attempted call strands that row at the tail,
    suppresses the repair model call, and causes TUI resume to replay stale
    commentary. Require the matching successful result instead.
    """
    terminal_call_ids = set()
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        if msg.get("role") == "assistant":
            terminal_call_ids.update(
                str(tc.get("id") or "")
                for tc in msg.get("tool_calls") or []
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS
            )
            continue
        if (
            msg.get("role") == "tool"
            and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS
            and str(msg.get("tool_call_id") or "") in terminal_call_ids
            and _terminal_tool_result_succeeded(msg)
        ):
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
