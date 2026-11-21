"""Chat text of the Discord loop-guard hold (loop-guard-hold spec S6-S8): the hold notice, the pause and queue-limit
notices, the ``/stop`` discard line and the restart "lost" lines. English literals, not locale catalog keys.

Everything here is a pure function of its arguments so the guard, the runner and the tests share one wording. Names
that come from other bots are escaped before they are posted (a bot may call itself anything): the notice is sent with
``AllowedMentions.none()``, and the escaping keeps a name from also formatting, linking or pinging the channel.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "MAX_MESSAGE_CHARS",
    "escape_name",
    "format_entry_lines",
    "format_hold_notice",
    "format_listing",
    "format_over_notice",
    "format_pause_notice",
    "message_link",
]

# Discord's limit is 2000; a margin is left for the lead-in.
MAX_MESSAGE_CHARS = 1900
NAME_CHARS = 32
NAMED_SENDERS = 5
_SPECIAL = "\\*~`|>[]()<@#:"


def escape_name(name: object, *, limit: int = NAME_CHARS) -> str:
    """A display name made safe to quote: cut to ``limit`` characters, whitespace folded, and every character that
    could format, link, mention or channel-link escaped with a backslash. An underscore inside a word is left as it
    is (Discord does not italicise it); one at a word edge is escaped."""
    text = " ".join(str(name if name is not None else "").split())[:limit] or "unknown"
    out: list[str] = []
    for index, char in enumerate(text):
        if char == "_":
            before = text[index - 1] if index else " "
            after = text[index + 1] if index + 1 < len(text) else " "
            if before.isalnum() and after.isalnum():
                out.append(char)
                continue
            out.append("\\" + char)
        elif char in _SPECIAL:
            out.append("\\" + char)
        else:
            out.append(char)
    return "".join(out)


def _named(senders: Sequence[object]) -> str:
    """``A, B and C`` for up to five distinct senders, then ``and N more``."""
    distinct: list[str] = []
    for sender in senders:
        label = escape_name(sender)
        if label not in distinct:
            distinct.append(label)
    shown, rest = distinct[:NAMED_SENDERS], len(distinct) - NAMED_SENDERS
    if not shown:
        return "a bot"
    if rest > 0:
        return ", ".join(shown) + f" and {rest} more"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def _seconds(value: float) -> str:
    value = max(0.0, float(value))
    if value >= 120:
        return f"{value / 60:.0f} minutes"
    return f"{value:.0f} seconds"


def format_hold_notice(
    senders: Sequence[object], seat: object, delay_seconds: float, held: int, replies_to_replies: bool,
) -> str:
    """The notice posted once when a conversation starts holding bot messages."""
    rule = "an exact repeat of an earlier bot message from the same sender"
    if replies_to_replies:
        rule += ", and a bot answering the reply it just got (reply_to_reply),"
    text = (
        f"Bot loop guard: {_named(senders)} sent {escape_name(seat)} too many messages too fast, so further messages "
        f"from bots in this conversation are being held and delivered one at a time, the next in {_seconds(delay_seconds)} "
        f"({held} held). Held messages are delayed, not dropped; only {rule} "
        f"{'are' if replies_to_replies else 'is'} still dropped. A person's message releases them and a person's /stop "
        f"discards them. Bots: do not answer this notice and do not resend what was held."
    )
    return text


def format_pause_notice(senders: Sequence[object], seat: object, held: int, replies_to_replies: bool) -> str:
    """The notice posted once more when the exchange pauses (several full-length waits in a row)."""
    return (
        f"Bot loop guard: this bot exchange with {_named(senders)} is paused: {held} held message(s) will not be "
        f"delivered until a person writes here, and a person's /stop discards them. Held messages are delayed, not "
        f"dropped. Bots: do not answer this notice and do not resend what was held."
    )


def format_over_notice(senders: Sequence[object], seat: object, held: int, limit: int) -> str:
    """Re-posted once when the queue grows past ``hold_queue_max``."""
    return (
        f"Bot loop guard: {held} messages are now held in this conversation, past the limit of {limit}; they are "
        f"still queued, not dropped, and wait for a person's message here. A person's /stop discards them. "
        f"Bots: do not answer this notice and do not resend what was held."
    )


def message_link(guild_id: Optional[object], channel_id: object, message_id: object) -> str:
    return f"https://discord.com/channels/{guild_id or '@me'}/{channel_id}/{message_id}"


def _clock_text(arrived: Optional[float]) -> str:
    if not arrived:
        return "an unknown time"
    return time.strftime("%H:%M UTC", time.gmtime(float(arrived)))


def format_entry_lines(entries: Iterable[dict]) -> list[str]:
    """One line per held message: sender and link, or ``a voice message from <name> at <time>``. Each ``entry`` is a
    dict with ``sender_name``, ``channel``, ``message`` (an id, or ``voice``), ``guild`` and ``arrived``."""
    lines: list[str] = []
    for entry in entries:
        name = escape_name(entry.get("sender_name"))
        message = str(entry.get("message") or "")
        if not message or message == "voice":
            lines.append(f"- a voice message from {name} at {_clock_text(entry.get('arrived'))}")
        else:
            lines.append(f"- {name}: {message_link(entry.get('guild'), entry.get('channel'), message)}")
    return lines


def format_listing(
    lead: str, lines: Sequence[str], *, max_messages: int = 5, not_listed: int = 0, limit: int = MAX_MESSAGE_CHARS,
) -> tuple[list[str], list[int]]:
    """Pack ``lines`` under ``lead`` into messages of at most ``limit`` characters, at most ``max_messages`` of them.
    Returns ``(messages, counts)``: ``counts[i]`` is how many of ``lines`` message ``i`` names or sums up. What does
    not fit in ``max_messages`` messages is summed as ``and N more`` at the end of the last one, and ``not_listed`` (a
    count the caller could not list at all) is appended the same way."""
    messages: list[str] = []
    counts: list[int] = []
    current, in_current = lead, 0
    index, total = 0, len(lines)
    while index < total and len(messages) < max_messages:
        line = lines[index]
        # Keep room for the closing "and N more" on the last allowed message.
        reserve = 60 if len(messages) == max_messages - 1 else 0
        if len(current) + 1 + len(line) + reserve > limit and in_current:
            messages.append(current)
            counts.append(in_current)
            current, in_current = lead, 0
            continue
        if len(current) + 1 + len(line) > limit:
            line = line[: max(0, limit - len(current) - 2)]
        current += "\n" + line
        in_current += 1
        index += 1
    if in_current:
        messages.append(current)
        counts.append(in_current)
    rest = total - index
    tail = ""
    if rest > 0:
        tail = f"\nand {rest} more"
    if not_listed > 0:
        tail += f"\nand {not_listed} more not listed"
    if tail:
        if messages and len(messages[-1]) + len(tail) <= limit:
            messages[-1] += tail
            counts[-1] += rest
        elif len(messages) < max_messages or not messages:
            messages.append(lead + tail)
            counts.append(rest)
        else:
            messages[-1] = messages[-1][: limit - len(tail)] + tail
            counts[-1] += rest
    return messages, counts
