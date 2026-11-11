"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, five gates may instead append the answer as an
interim row plus a user-role nudge and continue the turn: late language-server verdicts on this
turn's edits (a durable row, see ``_late_lsp_verdicts``), verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, the kanban worker terminal-tool guard, and
the opt-in ``pre_final_stop`` plugin seam (one continuation per turn, edits or not).
Each keeps the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). Nothing here imports ``agent.conversation_loop`` at
module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")

# Python-plugin-only (shell hooks are refused: ``SHELL_UNSUPPORTED_HOOKS``). The plugin owns every
# domain judgement; Hermes only supplies identity and raw call outcomes and enforces the cap.
PRE_FINAL_STOP_HOOK = "pre_final_stop"
_MAX_PRE_FINAL_STOP_CONTINUATIONS = 1


@dataclass(frozen=True)
class PreFinalToolCall:
    """One tool call this turn dispatched, with the outcome its terminal ``post_tool_call``
    event reported (``status`` None: no event arrived). Never carries arguments or results."""

    tool_name: str
    call_id: str
    turn_id: str
    status: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]


def note_dispatched_tool_calls(agent: Any, tool_calls: Any) -> None:
    """Record the turn's own call list, the witness ``pre_final_stop`` checks events against."""
    from agent.message_sanitization import coalesce_tool_call_id

    turn_id = str(getattr(agent, "_current_turn_id", "") or "")
    agent._pre_final_dispatched_calls = tuple(getattr(agent, "_pre_final_dispatched_calls", ())) + tuple(
        (coalesce_tool_call_id(tc), str(getattr(getattr(tc, "function", None), "name", "") or ""), turn_id)
        for tc in tool_calls or ()
    )


def _pre_final_call_outcomes(agent, session_id: str, turn_id: str) -> tuple[tuple[PreFinalToolCall, ...], bool]:
    """Pair this turn's dispatched calls with the post-tool events received for them. Complete only
    when every call has exactly one event and no event names a call this turn did not dispatch."""
    from hermes_cli.plugins_dispatch import post_tool_events

    dispatched = [c for c in getattr(agent, "_pre_final_dispatched_calls", ()) if c[2] == turn_id]
    events, overflowed = post_tool_events(session_id, turn_id)
    by_id: dict[str, list[Any]] = {}
    for event in events:
        by_id.setdefault(event.get("tool_call_id") or "", []).append(event)
    ids = [call_id for call_id, _name, _turn in dispatched]
    complete = not overflowed and len(set(ids)) == len(ids) and "" not in ids and set(by_id) == set(ids) and all(
        len(found) == 1 for found in by_id.values())
    calls = []
    for call_id, name, call_turn in dispatched:
        event = (by_id.get(call_id) or [{}])[0]
        calls.append(PreFinalToolCall(name, call_id, call_turn, event.get("status"), event.get("error_type"),
                                      event.get("error_message")))
    return tuple(calls), complete


def _pre_final_stop_nudge(agent) -> Optional[str]:
    """Ask ``pre_final_stop`` plugins whether this stop gets its one continuation. Only an explicit
    ``{"action": "continue", "message": str}`` continues; ``{"action": "stop"}``, nothing, a raise or
    anything unrecognised finishes the turn. A ``diagnostic`` string is shown either way."""
    if getattr(agent, "_tool_guardrail_halt_decision", None) is not None or getattr(agent, "_interrupt_requested", False):
        return None
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook(PRE_FINAL_STOP_HOOK):
            return None
        session_id = str(getattr(agent, "session_id", "") or "")
        turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        calls, complete = _pre_final_call_outcomes(agent, session_id, turn_id)
        available = getattr(agent, "_pre_final_stop_continuations", 0) < _MAX_PRE_FINAL_STOP_CONTINUATIONS
        results = invoke_hook(
            PRE_FINAL_STOP_HOOK, session_id=session_id, turn_id=turn_id,
            platform=str(getattr(agent, "platform", "") or ""), calls=calls, complete=complete,
            continuation_available=available,
        )
    except Exception:
        logger.warning("pre_final_stop hook check failed; finishing the turn", exc_info=True)
        agent._emit_diagnostic_status("⚠️ pre_final_stop: hook check failed; finishing the turn")
        return None
    message = _read_pre_final_results(agent, results)
    if message and not available:
        agent._emit_diagnostic_status(
            "⚠️ pre_final_stop: continuation exhausted (one per turn already used); finishing the turn")
        return None
    return message


def _read_pre_final_results(agent, results: list[Any]) -> Optional[str]:
    """First explicit continue message wins; every diagnostic and unrecognised result is shown."""
    message = None
    for result in results:
        fields = result if isinstance(result, dict) else {}
        action, text, diagnostic = fields.get("action"), fields.get("message"), fields.get("diagnostic")
        if isinstance(diagnostic, str) and diagnostic.strip():
            agent._emit_diagnostic_status(f"⚠️ {diagnostic.strip()}")
        if action == "continue" and isinstance(text, str) and text.strip():
            message = message or text.strip()
        elif action != "stop":
            agent._emit_diagnostic_status("⚠️ pre_final_stop: unrecognised plugin result ignored; finishing the turn")
    return message


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with a terminal board tool (kanban_complete / kanban_block /
    kanban_request_review / kanban_request_changes); a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


_MAX_LSP_TURN_END_NOTES = 3


def _late_lsp_verdicts(agent) -> Optional[str]:
    """Language-server verdicts on this turn's edits that no tool result carried yet (the edit was the
    last tool call; a one-shot run has no next turn to carry them): waits briefly for checks still
    running (``agent.lsp.deferred.settle_for_agent``) and returns them as a message for the model."""
    issued = getattr(agent, "_lsp_turn_end_notes", 0)
    if not isinstance(issued, int) or issued >= _MAX_LSP_TURN_END_NOTES:
        return None
    try:
        from agent.lsp.deferred import settle_for_agent

        notes = settle_for_agent(agent)
    except Exception:
        logger.debug("late LSP verdict check failed", exc_info=True)
        return None
    if not notes:
        return None
    body = "\n\n".join(text for _path, text in notes)
    return ("[System note: the language server reported problems in files you edited this turn, after "
            "those tool results had returned. Fix them if they are real, then give your final answer.]\n\n"
            + body)


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919). A held turn's answer (typed hidden) is not shown
    and flushes with the turn instead, so a budget fallback can still publish it as the rewrite."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    if final_msg.get("display_kind") == "hidden":
        return
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def apply_stop_gates(
    agent: Any, final_msg: dict[str, Any], *, final_response: Any, messages: list[dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run late LSP → verify-on-stop → pre_verify hook → kanban stop guard → pre_final_stop,
    in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(nudge: str, flag: str) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            # Prefix semantics on purpose: the streamed prefix IS on screen here. The gateway
            # interim verdict uses _interim_content_fully_streamed instead (#88954).
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

    _lsp_note = _late_lsp_verdicts(agent)
    if _lsp_note:
        agent._lsp_turn_end_notes = getattr(agent, "_lsp_turn_end_notes", 0) + 1  # int: checked above
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "late-LSP interim flush failed"
        )
        # A durable row (no scaffolding flag): it is never stripped, so no sent byte changes later.
        append_message(messages, {"role": "user", "content": _lsp_note})
        agent._session_messages = messages
        logger.debug("late LSP verdicts delivered before the final answer (%d)", agent._lsp_turn_end_notes)
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(final_response or ""),
        )

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_diagnostic_status(
            "⚠️ Kanban worker tried to exit without a terminal board call "
            "(kanban_complete/kanban_request_review/kanban_block) — nudging to finish"
        )
        return verdict

    # Last: a plugin's domain check only sees a stop every built-in gate already let through.
    _final_nudge = _pre_final_stop_nudge(agent)
    if _final_nudge:
        agent._pre_final_stop_continuations = getattr(agent, "_pre_final_stop_continuations", 0) + 1
        # The verify-on-stop shape on purpose: the candidate is a provisional verification row the
        # next answer supersedes, and the nudge is scaffolding the finalizer drops.
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_final_stop interim flush failed"
        )
        logger.info("pre_final_stop continuation issued (session=%s)", getattr(agent, "session_id", None))
        return _continue(_final_nudge, "_verification_stop_synthetic")
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )
