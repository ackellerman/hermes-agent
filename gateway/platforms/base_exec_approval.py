"""Shared wording for the exec-approval prompt every messaging surface renders.

The button card (``BasePlatformAdapter._format_exec_approval``) and the plain-text
``/approve`` fallback (``gateway.run._format_exec_approval_fallback``) must tell the user
the same three things: what Hermes wants to run, why it was flagged, and that silence means
the command does NOT run once ``approvals.timeout`` elapses. Keeping the text here means one
edit changes every platform; adapters only wrap these strings in their own markup.

The ``EA_*`` / ``APPROVAL_TIMED_OUT_NOTICE`` module constants are the English source of truth
(catalog keys ``gateway.exec_approval.*``); anything that renders the prompt calls the sibling
accessors (``ea_header_text()`` …) so the active ``display.language`` is honoured at render time.

No imports from ``gateway.platforms.base`` or ``gateway.run`` — both import this module.
"""

from __future__ import annotations

from agent.i18n import t

# Bare strings; adapters add their own bold/HTML around them.
EA_HEADER_TEXT = "Hermes wants to run a command that needs your OK"
EA_REASON_LABEL_TEXT = "Why it was flagged"

# Timeout notice posted when nobody answered the prompt (``{window}`` = "5 minutes").
APPROVAL_TIMED_OUT_NOTICE = (
    "⌛ Approval timed out after {window} — the command was NOT run. "
    "Ask me to try again if you still want it, or raise approvals.timeout in config.yaml.")


def ea_header_text() -> str:
    """Localized ``EA_HEADER_TEXT``."""
    return t("gateway.exec_approval.header")


def ea_reason_label_text() -> str:
    """Localized ``EA_REASON_LABEL_TEXT``."""
    return t("gateway.exec_approval.reason_label")


def ea_smart_deny_line_text() -> str:
    """Localized smart-deny sentence (bare; adapters add their own line breaks/markup)."""
    return t("gateway.exec_approval.smart_deny_line")


def ea_default_reason_text() -> str:
    """Localized default "why it was flagged" reason when the approval carries none."""
    return t("gateway.exec_approval.default_reason")


def ea_action_labels() -> dict[str, str]:
    """Localized ``choice -> button label`` for the four approval tiers."""
    return {
        "once": t("gateway.exec_approval.action_once"),
        "session": t("gateway.exec_approval.action_session"),
        "always": t("gateway.exec_approval.action_always"),
        "deny": t("gateway.exec_approval.action_deny"),
    }


def approval_timed_out_notice(window: str) -> str:
    """Localized ``APPROVAL_TIMED_OUT_NOTICE`` for an already-formatted ``window``."""
    return t("gateway.exec_approval.timed_out_notice", window=window)


def approval_timeout_seconds() -> int:
    """The configured ``approvals.timeout`` (default 300s); module attribute so tests can pin it."""
    from tools.approval_context import _get_approval_timeout
    return _get_approval_timeout()


def format_approval_window(seconds: int) -> str:
    """Human wording for a timeout (300 → "5 minutes"); one formatter shared with the CLI notice and
    the tool result's ``user_summary`` — see ``tools.approval_context.format_approval_window``."""
    from tools.approval_context import format_approval_window as _shared
    return _shared(seconds)


def format_approval_deadline_line(timeout_s: int) -> str:
    """The last line of every approval prompt: doing nothing is a safe no."""
    return t("gateway.exec_approval.deadline", window=format_approval_window(timeout_s))


def format_approval_timed_out_notice(timeout_s: int) -> str:
    return approval_timed_out_notice(format_approval_window(timeout_s))


# Longest command preview a typed ``/approve`` / ``/deny`` confirmation echoes back.
APPROVAL_ANSWER_PREVIEW_CHARS = 120


def approval_answer_detail(resolved: list[dict], remaining: int) -> str:
    """Lines appended to a typed ``/approve`` / ``/deny`` confirmation: the command it resolved (for
    a single request: the redacted text the prompt showed, on one line, cut short, never more than
    the prompt did) and how many requests the session still has pending. Empty when neither applies."""
    lines = []
    if len(resolved) == 1:
        from agent.redact import redact_approval_display
        preview = " ".join(redact_approval_display(resolved[0].get("command"), force=True).split()).replace("`", "'")
        if len(preview) > APPROVAL_ANSWER_PREVIEW_CHARS:
            preview = preview[:APPROVAL_ANSWER_PREVIEW_CHARS].rstrip() + "…"
        if preview:
            lines.append(t("gateway.exec_approval.answered_command", command=preview))
    if remaining > 0:
        lines.append(t("gateway.exec_approval.answered_remaining", count=remaining))
    return "".join(f"\n{line}" for line in lines)


def approval_answer_needs_person(source) -> bool:
    """The one rule on who may answer a command approval by message: never a bot. True when the
    author is a known bot (``SessionSource.is_bot``: the platform's bot flag or the adapter's
    ``bot_users`` opt-in); a typed ``/approve`` / ``/deny`` and the bare-word route both ask here.
    No configuration key: a pending command is granted or refused by a person."""
    return getattr(source, "is_bot", False) is True
