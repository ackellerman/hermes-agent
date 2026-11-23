"""The ``ctxpkg`` structured view: a driver's request served as fixed, written-once material.

``ctxpkg.view: structured`` serves every provider request as

    system | prefill | head | marker blocks | state note | the rest of the request

computed from the pair ``(request, uids)`` the host hands the engine (``agent/turn_context.py`` records
one ``message_uid`` per canonical message) plus the persisted ``view.json``. Persisted history is never
rewritten: the view is request-only. A structured *turn* is skipped whole (its marker line lives in the
block that was written when it moved); a structured *tool result* is served in place as its stub, so the
call that produced it keeps its result and the provider's tool pairing stays valid.

Everything here is a pure function of its arguments plus the files under the episode store root: between
passes a new request only appends to the tail, so every byte up to the tail is identical, and the newest
block keeps its cache anchor. A pass moves only what a judge decided; the step-4 judge is :func:`keep_all`.

Files, beside the episodes (``store.root``, 0700; every file 0600, written atomically or appended):

* ``view.json`` — :class:`ViewState` (blocks, structured uids, stubs, the stub calls, the pass turn of each
  ref, first-seen turns, ``view_turn``, ``last_request_at``, ``last_pass_at``, the state note);
* ``passes.jsonl`` — one row per pass attempt, and one ``view_off`` row per misaligned request;
* ``recalls.jsonl`` — one row per ``recall`` of a ref a pass wrote;
* ``reruns.jsonl`` — one row per tool call in a request that repeats a stubbed result's call.

One in-process lock serialises the read-modify-write of ``view.json`` between the request thread (a pass)
and a finished turn (``view_turn``/``seen``), as ``store._WARM_LOCK`` does for ``warm.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

from agent.context_compressor import _last_template_visible_role

from .compaction import (
    BLOCK_HEADER_RE,
    NOTED_FILLER,
    STATE_FILLER,
    _kilochars,
    block_ack,
    block_text,
    cut_words,
    scrub_ids,
)
from .episodes import extract_ids, message_text, recall_ref
from .store import RECALL_REF_RE, EpisodeStore, atomic_write_text

logger = logging.getLogger(__name__)

VIEW_NAME = "view.json"
PASSES_NAME = "passes.jsonl"
RECALLS_NAME = "recalls.jsonl"
RERUNS_NAME = "reruns.jsonl"
MAX_MARKER_IDS = 6
MAX_STUB_IDS = 3
SUPERSEDED_STUB_SUMMARY = "the same call was made again later"
_FILE_MODE = 0o600
_STUB_KIND_RE = re.compile(r"^\[(ep-[0-9]{4}) · ([a-z_]+) · ")
_MARKER_KIND_RE = re.compile(r"^(ep-[0-9]{4}) · ([a-z_]+) · ")
# One engine per process owns a store; this lock keeps a pass (request thread) and a finished turn from
# losing each other's ``view.json`` update. It is not a cross-process lock.
_VIEW_LOCK = threading.Lock()


def now_epoch() -> float:
    """The clock every view decision reads (tests inject one)."""
    return time.time()


def _utc_stamp(epoch: float) -> str:
    """``YYYY-MM-DD HH:MMZ`` for the marker line."""
    return datetime.fromtimestamp(float(epoch), tz=UTC).strftime("%Y-%m-%d %H:%MZ")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ── persisted state ────────────────────────────────────────────────────────


def _str_map(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(key, str) and isinstance(value, str)}


def _int_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): int(value) for key, value in raw.items()
            if isinstance(key, str) and _is_number(value)}


def _opt_float(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


@dataclass
class ViewState:
    """The session's persisted view state (``view.json`` beside the episodes)."""

    blocks: list[str] = field(default_factory=list)
    structured: dict[str, str] = field(default_factory=dict)
    stubs: dict[str, str] = field(default_factory=dict)
    stub_calls: dict[str, list[str]] = field(default_factory=dict)
    structured_turn: dict[str, int] = field(default_factory=dict)
    seen: dict[str, list[float]] = field(default_factory=dict)
    view_turn: int = 0
    last_request_at: float | None = None
    last_pass_at: float | None = None
    state_note: str = ""

    @classmethod
    def load(cls, store: EpisodeStore) -> ViewState:
        """Read ``view.json``; a missing or malformed field falls back to its empty value."""
        try:
            raw = json.loads((store.root / VIEW_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        raw = raw if isinstance(raw, dict) else {}
        stub_calls: dict[str, list[str]] = {}
        for ref, entry in (raw.get("stub_calls") or {}).items() if isinstance(raw.get("stub_calls"), dict) else ():
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(ref, str) \
                    and RECALL_REF_RE.fullmatch(ref):
                stub_calls[ref] = [str(entry[0]), str(entry[1])]
        seen: dict[str, list[float]] = {}
        for uid, entry in (raw.get("seen") or {}).items() if isinstance(raw.get("seen"), dict) else ():
            if isinstance(uid, str) and isinstance(entry, (list, tuple)) and len(entry) == 2 \
                    and _is_number(entry[0]) and _is_number(entry[1]):
                seen[uid] = [int(entry[0]), float(entry[1])]
        return cls(
            blocks=[text for text in raw.get("blocks") or [] if isinstance(text, str) and text],
            structured=_str_map(raw.get("structured")),
            stubs=_str_map(raw.get("stubs")),
            stub_calls=stub_calls,
            structured_turn=_int_map(raw.get("structured_turn")),
            seen=seen,
            view_turn=int(raw["view_turn"]) if _is_number(raw.get("view_turn")) else 0,
            last_request_at=_opt_float(raw.get("last_request_at")),
            last_pass_at=_opt_float(raw.get("last_pass_at")),
            state_note=raw["state_note"] if isinstance(raw.get("state_note"), str) else "",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocks": list(self.blocks),
            "structured": dict(self.structured),
            "stubs": dict(self.stubs),
            "stub_calls": {ref: list(entry) for ref, entry in self.stub_calls.items()},
            "structured_turn": dict(self.structured_turn),
            "seen": {uid: list(entry) for uid, entry in self.seen.items()},
            "view_turn": int(self.view_turn),
            "last_request_at": self.last_request_at,
            "last_pass_at": self.last_pass_at,
            "state_note": self.state_note,
        }

    def save(self, store: EpisodeStore) -> None:
        """Write the whole state atomically (a crash leaves the previous file intact).

        ``ensure()`` first: this is the FIRST file a session's view writes (the stamp at its first
        allowed request, before any episode or ``warm.json``), and ``atomic_write_text``'s own
        ``mkdir`` would create ``ctxpkg/<session>/`` with the process umask, not the store's 0700.
        """
        store.ensure()
        atomic_write_text(store.root / VIEW_NAME,
                          json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False) + "\n")


def save_state(store: EpisodeStore, state: ViewState) -> None:
    with _VIEW_LOCK:
        state.save(store)


def stamp_request(store: EpisodeStore, state: ViewState, now: float) -> None:
    """Stamp ``last_request_at`` at the end of an allowed request, at most once a second."""
    with _VIEW_LOCK:
        if state.last_request_at is None or float(now) - state.last_request_at >= 1:
            state.last_request_at = float(now)
            state.save(store)


def append_row(store: EpisodeStore, name: str, row: dict[str, Any]) -> None:
    """Append one JSON object to a log beside ``view.json``."""
    store.ensure()
    line = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(str(store.root / name), os.O_CREAT | os.O_WRONLY | os.O_APPEND, _FILE_MODE)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)


def turns_since_structured(state: ViewState, ref: str) -> int:
    """How many served turns have passed since the pass that structured *ref*."""
    turn = state.structured_turn.get(ref)
    if turn is None:
        return 0
    return max(0, int(state.view_turn) - int(turn))


def kind_of_ref(state: ViewState, ref: str) -> str:
    """The kind (``done``/``old``/``superseded``) a pass wrote for *ref*, read back off its own text."""
    for text in state.stubs.values():
        match = _STUB_KIND_RE.match(text or "")
        if match and match.group(1) == ref:
            return match.group(2)
    for block in state.blocks:
        for line in block.splitlines():
            match = _MARKER_KIND_RE.match(line)
            if match and match.group(1) == ref:
                return match.group(2)
    return "unknown"


# ── alignment ──────────────────────────────────────────────────────────────


def alignment(request: Any, uids: Any, prefill_count: Any) -> tuple[int | None, str]:
    """The index where the uid-aligned part of *request* starts, or ``(None, reason)``.

    The contract: ``request[0]`` is the system message, ``request[1:1 + prefill_count]`` the ephemeral
    prefill, and the rest one message per uid. Anything else turns the view off for the request — never a
    view built on a guess (a later host change that inserts or drops a message lands here).
    """
    if not isinstance(uids, list) or not uids:
        return None, "no_uids"
    if not isinstance(prefill_count, int) or isinstance(prefill_count, bool) or prefill_count < 0:
        return None, "prefill_count"
    if not isinstance(request, list) or len(request) != 1 + prefill_count + len(uids):
        return None, "length_mismatch"
    if not isinstance(request[0], dict) or request[0].get("role") != "system":
        return None, "no_system"
    return 1 + prefill_count, ""


def _turns(request: list[dict[str, Any]], offset: int) -> list[list[int]]:
    """A turn is a ``user`` message and everything up to the next one; leading rows join the first turn."""
    turns: list[list[int]] = []
    current: list[int] = []
    for index in range(offset, len(request)):
        if request[index].get("role") == "user" and current:
            turns.append(current)
            current = [index]
        else:
            current.append(index)
    if current:
        turns.append(current)
    return turns


def _block_ack_for(block: str) -> str:
    """The assistant ack that closes a block pair (its span comes off the block's own header)."""
    header = (block or "").splitlines()[0] if block else ""
    match = BLOCK_HEADER_RE.match(header)
    if match:
        return block_ack(match.group(1), match.group(2))
    return "Noted. I recall an episode with recall(ref) when I need its detail."


def build_view(request: list[dict[str, Any]], uids: list[str | None], prefill_count: int,
               state: ViewState) -> list[dict[str, Any]] | None:
    """The served list for one request, or None when the request does not align with the uid list."""
    offset, _reason = alignment(request, uids, prefill_count)
    if offset is None:
        return None
    served = list(request[:offset])
    turns = _turns(request, offset)
    head = turns[0] if turns else []
    for index in head:
        served.append(request[index])
    if state.blocks or state.state_note:
        # A block is a user message, so a head whose last template-visible role is user needs the filler
        # first (compaction.py's own rule): the host merges adjacent users after this hook, which would
        # change the block's bytes on the next request.
        if _last_template_visible_role(served) == "user":
            served.append({"role": "assistant", "content": NOTED_FILLER})
        for block in state.blocks:
            served.append({"role": "user", "content": block})
            served.append({"role": "assistant", "content": _block_ack_for(block)})
        if state.state_note:
            served.append({"role": "user", "content": state.state_note})
            served.append({"role": "assistant", "content": STATE_FILLER})
    rest_start = head[-1] + 1 if head else offset
    for index in range(rest_start, len(request)):
        message = request[index]
        uid = uids[index - offset]
        if uid is not None and uid in state.structured:
            stub = state.stubs.get(uid) if message.get("role") == "tool" else None
            if stub is not None:
                served.append({**message, "content": stub})
            continue  # the rest of a structured turn goes with it
        served.append(message)
    return served


# ── candidates ─────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """One thing a pass may move: a whole turn, or one tool result served as its stub."""

    id: str
    uid: str
    kind: str
    text: str
    chars: int
    ids: list[str] = field(default_factory=list)
    index: int = 0
    is_turn: bool = False
    turn_uids: tuple[str, ...] = ()
    turn_time: float = 0.0


def _call_key(call: Any) -> tuple[str, str]:
    """``(tool name, json.dumps(args, sort_keys=True))`` of one tool call."""
    function = (call or {}).get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        return name, raw if isinstance(raw, str) else ""
    return name, json.dumps(parsed, sort_keys=True)


def _call_for(request: list[dict[str, Any]], call_id: Any) -> dict[str, Any] | None:
    """The assistant tool call *call_id* names, wherever its row sits in the request."""
    if not call_id:
        return None
    for message in request:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") == call_id:
                return call
    return None


def _made_again_later(request: list[dict[str, Any]], index: int, key: tuple[str, str]) -> bool:
    """True when a later assistant message in the request makes exactly this call again."""
    for later in request[index + 1:]:
        for call in later.get("tool_calls") or []:
            if _call_key(call) == key:
                return True
    return False


def _is_warm(store: EpisodeStore | None, ref: str, warm_turns: int) -> bool:
    """Warm means ``store turn - last use < warm_turns``; an unknown or unused ref is cold.

    No store means nothing can be read back here, so the ref is treated as warm (never a candidate): a
    caller that omits the store must not be offered a ``recall`` result it cannot judge the warmth of.
    """
    if store is None:
        return True
    if warm_turns <= 0:
        return False
    last_use = store.last_use(ref)
    if last_use is None:
        return False
    return (store.current_turn() - last_use) < warm_turns


def _conversation_text_by_uid(conversation: Iterable[dict[str, Any]] | None) -> dict[str, str]:
    """Raw message text per uid, from the persisted history (which keeps rows the request dropped)."""
    found: dict[str, str] = {}
    for message in conversation or []:
        uid = message.get("message_uid") if isinstance(message, dict) else None
        if isinstance(uid, str) and uid and uid not in found:
            found[uid] = message_text(message)
    return found


def candidates(request: list[dict[str, Any]], uids: list[str | None], conversation: list[dict[str, Any]] | None,
               state: ViewState, settings: Any, *, store: EpisodeStore | None = None,
               prefill_count: int = 0, now: float | None = None) -> list[Candidate]:
    """The items this pass may move, oldest first — in this step, ``superseded`` tool results only.

    A tool result qualifies when the very same call (tool name and arguments, matched through
    ``tool_call_id``) is made again later in the request, or when it answers a ``recall`` whose ref went
    cold. Never a candidate: the head turn; the current turn (the last ``user`` message onward); the last
    ``keep_turns`` turns; any turn holding a uid-None message; anything already structured; and any turn or
    result whose ids appear in the state note. ``conversation`` supplies the text of rows the request's own
    canonicalization dropped, so a uid's ids are read from the raw history.
    """
    offset, _reason = alignment(request, uids, prefill_count)
    if offset is None:
        return []
    turns = _turns(request, offset)
    if not turns:
        return []
    texts = _conversation_text_by_uid(conversation)
    when = float(now) if now is not None else now_epoch()
    structure = getattr(settings, "structure", None) or settings
    keep_turns = int(getattr(structure, "keep_turns", 0))
    max_candidates = max(1, int(getattr(structure, "max_candidates", 1)))
    min_stub_chars = int(getattr(structure, "min_stub_chars", 0))
    warm_turns = int(getattr(settings, "warm_turns", 0))

    protected: set[int] = {0}
    last_user = None
    for index in range(len(request) - 1, offset - 1, -1):
        if request[index].get("role") == "user":
            last_user = index
            break
    for number, indexes in enumerate(turns):
        if last_user is not None and indexes[0] <= last_user <= indexes[-1]:
            protected.add(number)
        if any(uids[index - offset] is None for index in indexes):
            protected.add(number)
    for number in range(max(0, len(turns) - keep_turns), len(turns)):
        protected.add(number)

    note = state.state_note or ""
    found: list[Candidate] = []
    for number, indexes in enumerate(turns):
        if number in protected:
            continue
        turn_text = "\n".join(texts.get(uids[index - offset] or "", "") for index in indexes)
        turn_messages = [request[index] for index in indexes]
        turn_time = _turn_time(turn_messages, state, when)
        for index in indexes:
            message = request[index]
            uid = uids[index - offset]
            if message.get("role") != "tool" or uid is None or uid in state.structured:
                continue
            text = message_text(message)
            ids = extract_ids(text or turn_text)
            if any(value and value in note for value in ids):
                continue
            call = _call_for(request, message.get("tool_call_id"))
            if call is None:
                continue
            superseded = (len(text) >= min_stub_chars
                          and _made_again_later(request, index, _call_key(call)))
            cold_recall = False
            if not superseded:
                ref = recall_ref(call)
                cold_recall = bool(ref) and not _is_warm(store, ref, warm_turns)
            if not superseded and not cold_recall:
                continue
            found.append(Candidate(
                id=f"c{len(found) + 1}", uid=uid, kind="superseded", text=text, chars=len(text),
                ids=ids, index=index, turn_time=turn_time,
            ))
            if len(found) >= max_candidates:
                return found
    return found


# ── rendering the written-once material ────────────────────────────────────


def marker_line(ref: str, kind: str, when: float, chars: int, summary: str, ids: Iterable[str]) -> str:
    """``ep-NNNN · <kind> · <UTC time> · <k>k chars · <summary> · ids: <ids>`` (no summary when empty)."""
    shown = ", ".join(list(ids)[:MAX_MARKER_IDS]) or "none"
    body = f" · {summary}" if (summary or "").strip() else ""
    return f"{ref} · {kind} · {_utc_stamp(when)} · {_kilochars(chars)}k chars{body} · ids: {shown}"


def stub_line(ref: str, kind: str, chars: int, summary: str, ids: Iterable[str]) -> str:
    """The one-line stand-in a structured tool result is served as (its call keeps its result)."""
    shown = ", ".join(list(ids)[:MAX_STUB_IDS]) or "none"
    text = (summary or "").strip() or SUPERSEDED_STUB_SUMMARY
    return (f'[{ref} · {kind} · {_kilochars(chars)}k chars · {text} · ids: {shown} · '
            f'recall("{ref}") for the full text]')


def _turn_time(messages: list[dict[str, Any]], state: ViewState, now: float) -> float:
    """The turn's first message's ``timestamp``, else its first-seen epoch, else now."""
    if messages:
        stamp = messages[0].get("timestamp")
        if _is_number(stamp):
            return float(stamp)
        for message in messages:
            uid = message.get("message_uid")
            if uid and uid in state.seen:
                return float(state.seen[uid][1])
    return float(now)


# ── the pass ───────────────────────────────────────────────────────────────


@dataclass
class PassResult:
    """One pass attempt's outcome, as ``passes.jsonl`` records it."""

    status: str
    at: float = 0.0
    idle_s: float = 0.0
    ttl_s: float = 0.0
    candidates: int = 0
    kept: int = 0
    structured: int = 0
    marker_only: int = 0
    stubbed: int = 0
    refs: list[str] = field(default_factory=list)
    rough_tokens_before: int = 0
    rough_tokens_after: int = 0
    latency_s: float = 0.0
    flash_in: int = 0
    flash_out: int = 0


def keep_all(candidates_found: list[Candidate], state_note: str) -> dict[str, Any]:
    """The step-4 judge: every candidate kept, so nothing moves until the flash judge lands in step 5."""
    return {}


def rough(messages: Iterable[dict[str, Any]] | None) -> int:
    """The rough token estimate of a message list (``agent.model_metadata``'s own estimator)."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough

        return int(estimate_messages_tokens_rough(list(messages or [])))
    except Exception:
        logger.debug("ctxpkg rough estimate failed", exc_info=True)
        return 0


def should_pass(state: ViewState, cache_ttl_seconds: int | None, now: float) -> bool:
    """The trigger: the tier expired since the last request, and nothing tried since that request."""
    if not cache_ttl_seconds:
        return False
    if state.last_request_at is None:
        return False
    if float(now) - state.last_request_at < float(cache_ttl_seconds):
        return False
    return state.last_pass_at is None or state.last_pass_at < state.last_request_at


def _pass_row(result: PassResult) -> dict[str, Any]:
    return {
        "at": result.at, "idle_s": result.idle_s, "ttl_s": result.ttl_s, "status": result.status,
        "candidates": result.candidates, "kept": result.kept, "structured": result.structured,
        "marker_only": result.marker_only, "stubbed": result.stubbed, "refs": list(result.refs),
        "rough_tokens_before": result.rough_tokens_before, "rough_tokens_after": result.rough_tokens_after,
        "latency_s": result.latency_s, "flash_in": result.flash_in, "flash_out": result.flash_out,
    }


def run_pass(request: list[dict[str, Any]], uids: list[str | None], conversation: list[dict[str, Any]] | None,
             state: ViewState, store: EpisodeStore, judge: Callable[..., Any], now: float, *,
             settings: Any = None, prefill_count: int = 0, ttl_s: int | None = None) -> PassResult:
    """One pass: stamp the attempt, ask the judge, apply only what it decided, record one row.

    ``last_pass_at`` is stamped before anything else, whatever the outcome, so a gap gets one attempt at
    most. A judge failure (None, an exception, a timeout) changes nothing else: no episode, no stub, no
    block, no state note. Only decisions naming a candidate id are applied; everything else is kept.
    """
    settings = settings or _settings()
    result = PassResult(status="ok", at=float(now))
    result.ttl_s = float(ttl_s or 0)
    result.idle_s = max(0.0, float(now) - state.last_request_at) if state.last_request_at is not None else 0.0
    result.rough_tokens_before = rough(conversation)

    state.last_pass_at = float(now)
    save_state(store, state)

    found = candidates(request, uids, conversation, state, settings, store=store,
                       prefill_count=prefill_count, now=now)
    result.candidates = len(found)
    result.kept = len(found)
    if not found:
        result.status = "no_candidates"
        result.rough_tokens_after = result.rough_tokens_before
        append_row(store, PASSES_NAME, _pass_row(result))
        return result

    started = time.monotonic()
    try:
        reply = judge(found, state.state_note)
    except TimeoutError:
        reply = None
        result.status = "timeout"
    except Exception:
        logger.debug("ctxpkg judge failed", exc_info=True)
        reply = None
        result.status = "failed"
    result.latency_s = round(time.monotonic() - started, 3)
    if reply is None:
        if result.status == "ok":
            result.status = "no_route"
        result.rough_tokens_after = result.rough_tokens_before
        append_row(store, PASSES_NAME, _pass_row(result))
        return result

    decisions = reply.get("decisions") if isinstance(reply, dict) else None
    by_id = {candidate.id: candidate for candidate in found}
    applied: list[tuple[Candidate, str, str]] = []
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        candidate = by_id.get(str(decision.get("id") or ""))
        action = decision.get("action")
        if candidate is None or action not in ("structure", "marker_only"):
            continue
        summary = decision.get("summary")
        applied.append((candidate, action, summary if isinstance(summary, str) else ""))

    result.kept = len(found) - len(applied)
    moved_chars = 0
    written_chars = 0
    if applied:
        refs = store.take_refs(len(applied))
        block_lines: list[str] = []
        block_refs: list[str] = []
        for (candidate, action, summary), ref in zip(applied, refs):
            ids = candidate.ids or extract_ids(candidate.text)
            kept_summary = ""
            if action == "structure":
                kept_summary = cut_words(scrub_ids(summary, ids), int(settings.summary_max_chars))
            recorded = state.seen.get(candidate.uid)
            episode_turn = int(recorded[0]) if recorded else max(1, int(state.view_turn))
            store.write_episode(candidate.text, turn_first=episode_turn, turn_last=episode_turn,
                               ids=ids, summary=kept_summary, created_at=float(now), ref=ref)
            state.structured_turn[ref] = int(state.view_turn)
            result.refs.append(ref)
            moved_chars += len(candidate.text)
            if candidate.is_turn:
                line = marker_line(ref, candidate.kind, candidate.turn_time, candidate.chars,
                                   kept_summary, ids)
                block_lines.append(line)
                block_refs.append(ref)
                for uid in candidate.turn_uids:
                    if uid:
                        state.structured[uid] = ref
                written_chars += len(line)
            else:
                stub = stub_line(ref, candidate.kind, candidate.chars, kept_summary, ids)
                state.structured[candidate.uid] = ref
                state.stubs[candidate.uid] = stub
                call = _call_for(request, request[candidate.index].get("tool_call_id"))
                if call is not None:
                    name, arguments = _call_key(call)
                    state.stub_calls[ref] = [name, arguments]
                written_chars += len(stub)
                result.stubbed += 1
            if action == "marker_only":
                result.marker_only += 1
            else:
                result.structured += 1
        if block_lines:
            new_block = block_text(block_refs[0], block_refs[-1], block_lines)
            state.blocks.append(new_block)
            register_anchor(new_block)
        note = reply.get("state_note") if isinstance(reply, dict) else None
        if isinstance(note, str) and note.strip():
            cap = int(getattr(settings.structure, "state_note_chars", len(note) or 1))
            state.state_note = note.strip()[:cap]

    result.rough_tokens_after = max(0, result.rough_tokens_before - max(0, moved_chars - written_chars) // 4)
    result.flash_in = int(getattr(judge, "flash_in", 0) or 0)
    result.flash_out = int(getattr(judge, "flash_out", 0) or 0)
    save_state(store, state)
    append_row(store, PASSES_NAME, _pass_row(result))
    return result


def register_anchor(text: str) -> None:
    try:
        from agent.prompt_cache_boundary import register_cache_anchor

        register_cache_anchor(text)
    except Exception:
        logger.debug("ctxpkg view anchor registration failed", exc_info=True)


def _settings() -> Any:
    from .compaction import load_settings

    return load_settings()


# ── reruns and recalls ─────────────────────────────────────────────────────


def _assistant_calls(request: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in request or []:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def tool_call_ids(request: Iterable[dict[str, Any]]) -> set[str]:
    """Every assistant ``tool_call_id`` in the request (the set one request is compared against the next)."""
    return {str(call.get("id")) for call in _assistant_calls(request) if call.get("id")}


def rerun_rows(request: Iterable[dict[str, Any]], new_call_ids: set[str], state: ViewState,
               now: float) -> list[dict[str, Any]]:
    """One row per call in this request that repeats a stubbed result's call and is new since last time."""
    rows: list[dict[str, Any]] = []
    if not new_call_ids or not state.stub_calls:
        return rows
    for call in _assistant_calls(request):
        call_id = str(call.get("id") or "")
        if call_id not in new_call_ids:
            continue
        name, arguments = _call_key(call)
        for ref, entry in state.stub_calls.items():
            if list(entry) == [name, arguments]:
                rows.append({"at": float(now), "ref": ref,
                             "turns_since_structured": turns_since_structured(state, ref)})
    return rows


def recall_row(state: ViewState, ref: str, status: str, now: float) -> dict[str, Any] | None:
    """The ``recalls.jsonl`` row for a recall of a ref a pass wrote (None for any other ref)."""
    if ref not in state.structured_turn:
        return None
    return {"at": float(now), "ref": ref, "kind": kind_of_ref(state, ref),
            "turns_since_structured": turns_since_structured(state, ref), "status": status}


def view_off_row(state: ViewState, reason: str, now: float) -> dict[str, Any]:
    """The ``view_off`` row a misaligned request records."""
    return {"at": float(now), "status": "view_off", "reason": reason,
            "view_turn": int(state.view_turn)}
