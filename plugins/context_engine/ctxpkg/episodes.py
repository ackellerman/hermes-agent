"""Turn segmentation, rendering and id extraction for the ``ctxpkg`` engine.

The engine cuts the compacted span into episodes — one per user turn — writes each rendered episode to the
store, and keeps only its summary and ids in the prompt. Two kinds of message never become episode text:

* a ``[ctxpkg state]`` note and the ``Continuing from the state note.`` filler that follows it (the engine's own
  bookkeeping, not conversation);
* a ``recall`` tool call and its result, which collapse to the one-line stub ``[recalled ep-NNNN]`` so an episode
  is never re-episoded from its own recall.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

STATE_NOTE_PREFIX = "[ctxpkg state]"
STATE_FILLER = "Continuing from the state note."
DEFAULT_MAX_EPISODE_CHARS = 60_000
MAX_IDS = 24

RECALL_REF_RE = re.compile(r"ep-[0-9]{4}")

# Every id kind the reference block carries: shas, task ids, run ids, verdicts and the paths a later step needs.
# ``[0-9]`` rather than ``\d`` everywhere: ``\d`` also matches non-ASCII digits, and a ref is ASCII.
_SA_RE = r"sa-[0-9]+-[0-9a-f]{8}"
_VERDICT_RE = r"VERDICT: (?:ACCEPT|FIXES)"
# A path id is any absolute path of at least two segments, or a ``~/`` one: the episode text may come
# from any host, so no directory prefix is baked in. An absolute path starts only where its slash is
# not already joined to a word character, a scheme's colon or another slash, so ``and/or``, a relative
# ``a/b/c`` and the path inside ``https://x/y`` are not ids.
_PATH_SEGMENT = r"[A-Za-z0-9_.~+-]+"
# The home form first: its own slash is joined to a tilde, which the absolute form does not accept.
_ABS_PATH_RE = rf"(?<![\w:/])/{_PATH_SEGMENT}(?:/{_PATH_SEGMENT})+"
_PATH_RE = rf"(?:~/[A-Za-z0-9_./~+-]*|{_ABS_PATH_RE})"
_ID_RE = re.compile(
    rf"(?P<path>{_PATH_RE})"
    rf"|(?P<verdict>{_VERDICT_RE})"
    rf"|(?P<sa>{_SA_RE})"
    r"|(?P<sha>(?<![0-9A-Za-z])[0-9a-f]{40}(?![0-9A-Za-z]))"
    r"|(?P<ref>R-[0-9]{3})"
    # A digit run is an id only whole: without the boundaries a 21-digit run yields its first 20 digits.
    r"|(?P<num>(?<![0-9])[0-9]{17,20}(?![0-9]))"
    r"|(?P<hex>(?<![0-9A-Za-z])[0-9a-f]{7,12}(?![0-9A-Za-z]))"
)
_HAS_DIGIT = re.compile(r"[0-9]")
_HAS_ALPHA = re.compile(r"[a-f]")


@dataclass
class Episode:
    """One user turn (or one part of a turn too long for a single file)."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_first: int = 1
    turn_last: int = 1


def message_text(message: Any) -> str:
    """Plain text of a message: its string content, or the joined text parts of a multimodal one."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part["text"] for part in content
                 if isinstance(part, dict) and isinstance(part.get("text"), str)]
        return "\n".join(parts)
    return ""


def recall_ref(call: Any) -> str | None:
    """The ``ep-NNNN`` a tool call asks the ``recall`` tool for, or None for any other call."""
    if not isinstance(call, dict):
        return None
    function = call.get("function") or {}
    if function.get("name") != "recall":
        return None
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (TypeError, ValueError):
        return None
    ref = arguments.get("ref") if isinstance(arguments, dict) else None
    return ref if isinstance(ref, str) and RECALL_REF_RE.fullmatch(ref) else None


def recall_stub(ref: str) -> str:
    """The one line a recall pair leaves in its turn."""
    return f"[recalled {ref}]"


def _is_state_note(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and message_text(message).lstrip().startswith(STATE_NOTE_PREFIX)


def _is_state_filler(message: dict[str, Any]) -> bool:
    return (message.get("role") == "assistant" and not message.get("tool_calls")
            and message_text(message).strip() == STATE_FILLER)


def _collapse_recalls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace every recall call and its result with the one-line stub, in place of the pair."""
    recalled = {call.get("id") for message in messages for call in message.get("tool_calls") or []
                if recall_ref(call)}
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") in recalled:
            continue  # the stub on the call's own message stands in for it
        calls = message.get("tool_calls") or []
        refs = [ref for ref in (recall_ref(call) for call in calls) if ref]
        if not refs:
            out.append(message)
            continue
        stubs = [recall_stub(ref) for ref in refs]
        kept_calls = [call for call in calls if not recall_ref(call)]
        replacement = dict(message)
        text = message_text(message)
        replacement["content"] = "\n".join([text] + stubs if text else stubs)
        if kept_calls:
            replacement["tool_calls"] = kept_calls
        else:
            replacement.pop("tool_calls", None)
        out.append(replacement)
    return out


def _prepare(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the engine's own state note and filler, then collapse recall pairs."""
    kept = [message for message in messages
            if not _is_state_note(message) and not _is_state_filler(message)]
    return _collapse_recalls(kept)


def _render_message(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "?")
    bits = []
    text = message_text(message)
    if text:
        bits.append(text)
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        bits.append(f"[tool_call {function.get('name', '')}({function.get('arguments', '')})]")
    return f"{role}: " + "\n".join(bits) + "\n"


def render(episode: Episode) -> str:
    """Role-tagged episode text: tool calls with their full arguments, tool results in full."""
    return "".join(_render_message(message) for message in episode.messages)


def _segments(turn: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a turn at assistant tool-call starts (the only safe cut inside a turn)."""
    starts = [0] + [index for index, message in enumerate(turn)
                    if index and message.get("role") == "assistant" and message.get("tool_calls")]
    return [turn[start:end] for start, end in zip(starts, starts[1:] + [len(turn)])]


def _split_turn(turn: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    if max_chars <= 0 or len("".join(_render_message(m) for m in turn)) <= max_chars:
        return [turn]
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for segment in _segments(turn):
        segment_chars = sum(len(_render_message(message)) for message in segment)
        if current and current_chars + segment_chars > max_chars:
            chunks.append(current)
            current, current_chars = [], 0
        current.extend(segment)
        current_chars += segment_chars
    if current:
        chunks.append(current)
    return chunks


def segment(messages: Iterable[dict[str, Any]], max_chars: int = DEFAULT_MAX_EPISODE_CHARS,
            first_turn: int = 1) -> list[Episode]:
    """One episode per user turn, a turn over *max_chars* split at assistant tool-call boundaries.

    Turn numbers start at *first_turn* so they can be session-wide: the caller passes
    ``store.next_turn()``, and both parts of a split turn keep that turn's number.
    """
    prepared = _prepare(messages)
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for message in prepared:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = [message]
        elif current is not None:
            current.append(message)  # before the first user message there is no turn to join
    if current:
        turns.append(current)
    episodes: list[Episode] = []
    for offset, turn in enumerate(turns):
        number = first_turn + offset
        for part in _split_turn(turn, max_chars):
            episodes.append(Episode(messages=part, turn_first=number, turn_last=number))
    return episodes


def extract_ids(text: str) -> list[str]:
    """Ids, verdicts and paths in first-seen order, deduplicated, capped at ``MAX_IDS``."""
    seen: set[str] = set()
    found: list[str] = []
    for match in _ID_RE.finditer(text or ""):
        value = match.group(0)
        kind = match.lastgroup
        if kind == "path":
            value = value.rstrip(".,;:)")
        elif kind == "hex" and not (_HAS_DIGIT.search(value) and _HAS_ALPHA.search(value)):
            continue  # a hex run needs both a digit and a letter to be an id, not a word
        if not value or value in seen:
            continue
        seen.add(value)
        found.append(value)
        if len(found) >= MAX_IDS:
            break
    return found
