"""``ctxpkg`` compaction: the extend-only reference prefix, the state note and warm protection.

``CtxpkgEngine.compress`` keeps ``ContextCompressor``'s signature and return contract, but the middle of the
transcript is not replaced by a handoff summary. Every compaction cuts the span between the head and the
protected tail into episodes (one per user turn), writes each one whole to the store, and leaves the prompt
with a reference block, a state note and the episodes recalled recently enough to still be warm.

The prefix is extend-only: a compaction appends one block pair after the pairs already in the transcript and
never rewrites an earlier one, so the bytes up to the newest block's ack are the same on the next request and
the cache breakpoint ``agent/prompt_caching`` puts there turns the whole prefix into a read. The engine
registers the new block's ``user`` text as the cache anchor and appends it to ``blocks.jsonl``;
``on_session_start`` re-registers every stored block, because the registry is process-local.

Everything here is a mixin over ``ContextCompressor``: the base class's window planning, tail selection,
boundary alignment, token accounting and compaction bookkeeping (``_begin_compress_attempt``,
``_finalize_compressed``) are called, never re-implemented.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from agent.context_compressor import (
    _COMPACTION_TAIL_MARKER,
    _PRUNE_MIN_CHARS,
    _fresh_compaction_message_copy,
    _last_template_visible_role,
    _pinned_summary_call_kwargs,
)

from .episodes import (
    STATE_FILLER,
    STATE_NOTE_PREFIX,
    Episode,
    extract_ids,
    message_text,
    recall_ref,
    render,
    segment,
)
from .store import EpisodeStore

logger = logging.getLogger(__name__)

BLOCK_PREFIX = "[ctxpkg references "
ACK_PREFIX = "Noted: "
NOTED_FILLER = "Noted."
STATE_MAX_CHARS = 1500
COLD_STUB_RE = re.compile(r"^\[recalled ep-[0-9]{4}, now cold")
BLOCK_HEADER_RE = re.compile(r"^\[ctxpkg references (ep-[0-9]{4})\.\.(ep-[0-9]{4})\]$")
ACK_RE = re.compile(r"^Noted: (ep-[0-9]{4})\.\.(ep-[0-9]{4})\. ")
RECENT_PREFIX = "recent episodes: "
_ID_SUMMARY_RE = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{40}(?![0-9A-Za-z])|sa-[0-9]+-[0-9a-f]{8}")

# Defaults mirror ``hermes_cli/config_defaults.py``'s ``ctxpkg`` block; that module is the source of truth and
# these only stand in when the block is missing or malformed.
DEFAULT_WARM_TURNS = 8
DEFAULT_MAX_EPISODE_CHARS = 60_000
DEFAULT_SUMMARY_MAX_CHARS = 300
# ``ctxpkg.view``: ``off`` keeps today's behaviour byte for byte; ``structured`` serves the request through
# ``view.py``. ``ctxpkg.structure.*`` bounds what a pass may move and what its judge may write.
VIEW_OFF = "off"
VIEW_STRUCTURED = "structured"
DEFAULT_KEEP_TURNS = 8
DEFAULT_OLD_AFTER_MINUTES = 120
DEFAULT_MIN_STUB_CHARS = 2_000
DEFAULT_MAX_CANDIDATES = 40
DEFAULT_CANDIDATE_CHARS = 4_000
DEFAULT_STATE_NOTE_CHARS = 1_200
DEFAULT_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class StructureSettings:
    """The ``ctxpkg.structure`` leaves: the never-candidate window and the candidate/judge bounds."""

    keep_turns: int = DEFAULT_KEEP_TURNS
    old_after_minutes: int = DEFAULT_OLD_AFTER_MINUTES
    min_stub_chars: int = DEFAULT_MIN_STUB_CHARS
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    candidate_chars: int = DEFAULT_CANDIDATE_CHARS
    state_note_chars: int = DEFAULT_STATE_NOTE_CHARS
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CtxpkgSettings:
    """The ``ctxpkg`` config leaves this module reads, clamped to sane values."""

    warm_turns: int = DEFAULT_WARM_TURNS
    max_episode_chars: int = DEFAULT_MAX_EPISODE_CHARS
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
    view: str = VIEW_OFF
    structure: StructureSettings = StructureSettings()


def _coerce_int(value: Any, default: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, value)


def _strict_int(value: Any, minimum: int) -> int | None:
    """*value* when it is an int of at least *minimum*, else None (a bool is not an int here)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _read_view(raw: Any) -> tuple[str, bool]:
    """``(view, invalid)``: only ``off`` and ``structured`` are understood."""
    if raw is None:
        return VIEW_OFF, False
    text = raw.strip().lower() if isinstance(raw, str) else ""
    if text in (VIEW_OFF, VIEW_STRUCTURED):
        return text, False
    return VIEW_OFF, True


def _read_structure(raw: Any) -> tuple[StructureSettings, list[str]]:
    """``(structure, invalid leaves)``: every leaf falls back to its default on its own."""
    if raw is None:
        return StructureSettings(), []
    if not isinstance(raw, dict):
        return StructureSettings(), ["structure"]
    values: dict[str, int] = {}
    invalid: list[str] = []
    for name, default, minimum in (
        ("keep_turns", DEFAULT_KEEP_TURNS, 0),
        ("old_after_minutes", DEFAULT_OLD_AFTER_MINUTES, 1),
        ("min_stub_chars", DEFAULT_MIN_STUB_CHARS, 1),
        ("max_candidates", DEFAULT_MAX_CANDIDATES, 1),
        ("candidate_chars", DEFAULT_CANDIDATE_CHARS, 1),
        ("state_note_chars", DEFAULT_STATE_NOTE_CHARS, 1),
        ("timeout_seconds", DEFAULT_TIMEOUT_SECONDS, 1),
    ):
        coerced = _strict_int(raw.get(name, default), minimum)
        if coerced is None:
            invalid.append(f"structure.{name}")
            coerced = default
        values[name] = coerced
    return StructureSettings(**values), invalid


def load_settings() -> CtxpkgSettings:
    """The live ``ctxpkg`` config block. A missing or malformed block yields the documented defaults."""
    block: Any = {}
    try:
        from hermes_cli.config import load_config_readonly

        block = (load_config_readonly() or {}).get("ctxpkg")
    except Exception:  # a config read must never break a compaction
        logger.debug("ctxpkg settings unreadable; using defaults", exc_info=True)
    block = block if isinstance(block, dict) else {}
    view, view_invalid = _read_view(block.get("view"))
    structure, structure_invalid = _read_structure(block.get("structure"))
    if view_invalid or structure_invalid:
        logger.warning(
            "ctxpkg config: %s not understood; using the default. Fix the ``ctxpkg`` block in config.yaml.",
            ", ".join((["view"] if view_invalid else []) + structure_invalid),
        )
    return CtxpkgSettings(
        warm_turns=_coerce_int(block.get("warm_turns"), DEFAULT_WARM_TURNS, 0),
        max_episode_chars=_coerce_int(block.get("max_episode_chars"), DEFAULT_MAX_EPISODE_CHARS, 1),
        summary_max_chars=_coerce_int(block.get("summary_max_chars"), DEFAULT_SUMMARY_MAX_CHARS, 1),
        view=view,
        structure=structure,
    )


# ── message tests (shape only: the engine's own bookkeeping is recognised by position and text) ────────────

def is_block_user(message: Any) -> bool:
    """True for a reference block's ``user`` message."""
    return (isinstance(message, dict) and message.get("role") == "user"
            and message_text(message).startswith(BLOCK_PREFIX))


def is_block_ack(message: Any) -> bool:
    """True for the ``assistant`` ack that closes a reference block pair."""
    return (isinstance(message, dict) and message.get("role") == "assistant"
            and bool(ACK_RE.match(message_text(message))))


def is_state_note(message: Any) -> bool:
    """True for a ``[ctxpkg state]`` note."""
    return (isinstance(message, dict) and message.get("role") == "user"
            and message_text(message).lstrip().startswith(STATE_NOTE_PREFIX))


def is_state_filler(message: Any) -> bool:
    """True for the ``Continuing from the state note.`` filler."""
    return (isinstance(message, dict) and message.get("role") == "assistant"
            and not message.get("tool_calls") and message_text(message).strip() == STATE_FILLER)


def is_cold_stub(text: str) -> bool:
    """True for a tool result this engine already replaced with its cold stub."""
    return bool(COLD_STUB_RE.match((text or "").lstrip()))


# ── rendering ──────────────────────────────────────────────────────────────

def _kilochars(chars: int) -> int:
    """Nearest whole thousands of chars, never below 1 (an episode is never "0k chars")."""
    return max(1, (int(chars) + 500) // 1000)


def block_line(ref: str, turn_first: int, turn_last: int, chars: int, summary: str,
               ids: Iterable[str]) -> str:
    """One ``ep-<NNNN> · turns a-b · k chars · summary · ids: ...`` line."""
    joined = ", ".join(ids) or "none"
    return (f"{ref} · turns {int(turn_first)}-{int(turn_last)} · {_kilochars(chars)}k chars · "
            f"{summary} · ids: {joined}")


def block_text(first: str, last: str, lines: Iterable[str]) -> str:
    """The reference block's ``user`` text: a header naming the span, then one line per episode."""
    return "\n".join([f"{BLOCK_PREFIX}{first}..{last}]", *lines])


def block_ack(first: str, last: str) -> str:
    """The ``assistant`` ack that closes a block pair."""
    return f"{ACK_PREFIX}{first}..{last}. I recall an episode with recall(ref) when I need its detail."


def state_note(note: str, refs: Iterable[str]) -> str:
    """The state note: the flash note, then the last three episode refs copied by script."""
    recent = list(refs)[-3:]
    lines = [STATE_NOTE_PREFIX, note.strip()] if note.strip() else [STATE_NOTE_PREFIX]
    if recent:
        lines.append(RECENT_PREFIX + ", ".join(recent))
    return "\n".join(lines)


def cold_stub(ref: str, summary: str) -> str:
    """The one-line stand-in a recall's result keeps once the episode goes cold."""
    reason = f"now cold: {summary.strip()}" if (summary or "").strip() else "now cold"
    return f'[recalled {ref}, {reason} — recall("{ref}") for the full text]'


def cut_words(text: str, limit: int) -> str:
    """Cut *text* at the last word boundary at or below *limit*; a single long word is hard-cut."""
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[:limit]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip(" ,;:-—")


def scrub_ids(summary: str, allowed: Iterable[str]) -> str:
    """Drop any sha or ``sa-`` id the episode's own extractor did not find.

    Ids are copied out by script and handed to the summariser; a model that invents one would send the
    driver looking for it, so anything not in the episode's ids never reaches the prompt.
    """
    permitted = {value for value in allowed if value}

    def _keep(match: re.Match) -> str:
        token = match.group(0)
        return token if token in permitted else ""

    return re.sub(r"\s{2,}", " ", _ID_SUMMARY_RE.sub(_keep, summary or "")).strip()


def mechanical_summary(episode: Episode, tool_names: Iterable[str]) -> str:
    """The summary used when the flash model cannot be reached: the turn's opening text plus tool names."""
    user_text = ""
    for message in episode.messages:
        if message.get("role") == "user":
            user_text = message_text(message)
            break
    head = " ".join(user_text.split())[:160].strip()
    tools = ", ".join(dict.fromkeys(name for name in tool_names if name))
    if tools:
        return cut_words(f"{head} (tools: {tools})" if head else f"(tools: {tools})", 400)
    return cut_words(head, 400)


def _tool_names(episode: Episode) -> list[str]:
    names: list[str] = []
    for message in episode.messages:
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


# ── the mixin ──────────────────────────────────────────────────────────────

class CtxpkgCompactionMixin:
    """``ContextCompressor`` compaction for ``ctxpkg``: episodes out, an extend-only prefix back."""

    # ── public hooks ────────────────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Bind session state (``super()`` first), then re-register every stored block as a cache anchor."""
        super().on_session_start(session_id, **kwargs)
        try:
            from agent.prompt_cache_boundary import register_cache_anchor

            for text in self.store.blocks():
                register_cache_anchor(text)
        except Exception:  # an anchor miss costs one cache write, never the session
            logger.debug("ctxpkg anchor re-registration failed", exc_info=True)

    def compress(
        self, messages: list[dict[str, Any]], current_tokens: int | None = None,
        focus_topic: str | None = None, force: bool = False, memory_context: str = "",
        bypass_cooldown: bool = False,
    ) -> list[dict[str, Any]]:
        """Cut the middle of the transcript into episodes and prepend one new reference block.

        The head (system plus the protected early rows) and every earlier block pair stay byte-identical;
        the span becomes episode files plus one line each in the new block; warm recalls move forward
        intact. The base's own bookkeeping runs on the assembled list, so counters, savings and the
        anti-thrash state move exactly as they do for a handoff summary.
        """
        from agent.conversation_compression import _raise_if_stale_attempt

        _raise_if_stale_attempt(self)
        telemetry = self._begin_compress_attempt(current_tokens, force)
        telemetry["has_focus_topic"] = bool(focus_topic)
        # Resolved once: a compaction child row may not exist yet at the moment of a rotation.
        store = self.store
        settings = load_settings()
        canonical_messages = self._drop_blank_echoes(messages)
        spare_pending_images = bool(self._spared_pending_tool_round(messages))
        working, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n, protect_tail_tokens=self.tail_token_budget,
        )
        working = self._drop_blank_echoes(working)
        messages = working
        n_messages = len(messages)

        head_end = self._head_end(messages)
        compress_start = self._align_boundary_forward(messages, head_end)
        compress_end = compress_start
        if compress_start < len(messages):
            compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        if compress_start >= compress_end:
            self._record_compression_regions(
                head_messages=messages[:head_end], middle_messages=[], tail_messages=messages[compress_end:],
            )
            self._structural_no_op_result(
                telemetry, "no_compressible_window",
                f"compress_start ({compress_start}) >= compress_end ({compress_end}) - nothing between the "
                "reference blocks and the protected tail",
                n_messages,
            )
            return canonical_messages

        span = messages[compress_start:compress_end]
        turn = store.current_turn()
        warm_pairs = self._warm_recall_pairs(span, store, turn, settings.warm_turns)
        episodes = self._episodes_for_span(span, settings, store)
        if not episodes:
            # The window holds only the leftover rows of a turn whose head is protected: nothing to cut
            # into episodes, and writing a block with no lines would spend a compaction for no reclaim.
            self._record_compression_regions(
                head_messages=messages[:head_end], middle_messages=span, tail_messages=messages[compress_end:],
            )
            self._structural_no_op_result(
                telemetry, "no_episode_in_span",
                f"the window {compress_start}-{compress_end} holds no user turn to cut into an episode",
                n_messages,
            )
            return canonical_messages
        rendered = [render(episode) for episode in episodes]
        ids = [extract_ids(text) for text in rendered]
        # Refs are named in the summariser's prompt, so they are reserved before the call and handed to
        # write_episode, which refuses a ref that already names a file.
        refs = store.take_refs(len(episodes))
        summaries, summary_failed = self._episode_summaries(refs, episodes, rendered, ids, settings)
        first_ref, last_ref = refs[0], refs[-1]
        lines = [
            block_line(ref, episode.turn_first, episode.turn_last, len(text), summary, episode_ids)
            for ref, episode, text, summary, episode_ids in zip(refs, episodes, rendered, summaries, ids)
        ]
        # The state note is the second flash call, and it is taken here, before the first write: a
        # compaction the host abandoned while that call was in flight writes neither the episodes nor
        # the block (``_fail_if_abandoned`` below).
        note = self._state_letter(messages, lines, summary_failed=summary_failed)
        self._fail_if_abandoned()
        for ref, episode, text, episode_ids, summary in zip(refs, episodes, rendered, ids, summaries):
            store.write_episode(
                text, turn_first=episode.turn_first, turn_last=episode.turn_last,
                ids=episode_ids, summary=summary, ref=ref,
            )

        compressed = self._assemble_head(messages, head_end)
        # The block is a user message, so a head whose last *template-visible* role is user needs the
        # filler first: a strict-alternation template counts neither role:tool nor assistant(tool_calls),
        # so "head ends with a tool result" reads as user → user. Same rule as the base's summary role.
        if compressed and _last_template_visible_role(compressed) == "user":
            compressed.append({"role": "assistant", "content": NOTED_FILLER})
        new_block = block_text(first_ref, last_ref, lines)
        compressed.append({"role": "user", "content": new_block})
        compressed.append({"role": "assistant", "content": block_ack(first_ref, last_ref)})
        compressed.append({"role": "user", "content": state_note(note, refs)})
        for pair in warm_pairs:
            compressed.append({"role": "assistant", "tool_calls": [copy.deepcopy(pair["call"])]})
            compressed.append(copy.deepcopy(pair["result"]))
        tail = self._tail_rows(messages, compress_end)
        if compressed[-1].get("role") == "user" and is_state_note(compressed[-1]) and tail \
                and tail[0].get("role") == "user":
            compressed.append({"role": "assistant", "content": STATE_FILLER})
        compressed.extend(tail)

        self._register_block(store, new_block)
        return self._finalize_compressed(
            compressed, canonical_messages, n_messages, spare_pending_images, pruned_count,
        )

    # ── protection between compactions ──────────────────────────────────────

    def _prune_old_tool_results(
        self, messages: list[dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None, min_prune_chars: int = _PRUNE_MIN_CHARS,
    ) -> tuple[list[dict[str, Any]], int]:
        """Prune, then keep warm recalls verbatim and never touch a block pair or the state note."""
        store = self.store
        settings = load_settings()
        turn = store.current_turn()
        protected = self._protected_indexes(messages, store, turn, settings.warm_turns)
        result, pruned = super()._prune_old_tool_results(
            messages, protect_tail_count, protect_tail_tokens, min_prune_chars,
        )
        restored = 0
        for index in protected:
            if index >= len(result) or index >= len(messages):
                continue
            if result[index] == messages[index]:
                continue
            result[index] = copy.deepcopy(messages[index])
            restored += 1
        stubbed = self._stub_cold_recalls(result, store, turn, settings.warm_turns)
        return result, max(0, pruned - restored) + stubbed

    def _resolve_compact_cursor(
        self, messages: list[dict[str, Any]], head_end: int, tail_start: int,
    ) -> int:
        """Never absorb the extend-only prefix: start after the last block ack, state note or filler."""
        cursor = super()._resolve_compact_cursor(messages, head_end, tail_start)
        return max(cursor, self._prefix_floor(messages, tail_start))

    def _find_one_exchange(
        self, messages: list[dict[str, Any]], start: int, tail_start: int,
    ) -> tuple[int, int] | None:
        """Skip an exchange that holds a warm recall: the pair must move forward intact."""
        exchange = super()._find_one_exchange(messages, start, tail_start)
        if exchange is None:
            return None
        store = self.store
        settings = load_settings()
        turn = store.current_turn()
        begin, end = exchange
        for index in range(begin, end):
            message = messages[index] if index < len(messages) else None
            if not isinstance(message, dict):
                continue
            ref = self._recall_ref_of(message, messages)
            if ref and self._is_warm(ref, store, turn, settings.warm_turns):
                return None
        return exchange

    # ── head, blocks and tail ──────────────────────────────────────────────

    def _block_pairs(self, messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
        """``(user, ack)`` index pairs of every reference block in *messages*, in order."""
        pairs: list[tuple[int, int]] = []
        for index in range(len(messages) - 1):
            if is_block_user(messages[index]) and is_block_ack(messages[index + 1]):
                pairs.append((index, index + 1))
        return pairs

    def _head_end(self, messages: list[dict[str, Any]]) -> int:
        """End of the region that must stay byte-identical: system + protected head + every block pair.

        With blocks present the region is everything up to the newest block's ack. That is *longer* than
        ``_protect_head_size`` once ``protect_first_n`` has decayed after the first compaction, and the
        earlier head rows are exactly what must ride along unchanged. Without blocks it is the base's own
        head rule: a ``Noted.`` filler can only exist after a compaction, and a compaction always leaves
        a block pair behind it, so there is nothing extra to recognise here.
        """
        if not messages:
            return 0
        pairs = self._block_pairs(messages)
        if pairs:
            return min(pairs[-1][1] + 1, len(messages))
        return min(self._protect_head_size(messages), len(messages))

    def _tail_rows(self, messages: list[dict[str, Any]], compress_end: int) -> list[dict[str, Any]]:
        """The protected tail, copied verbatim (content untouched, only persistence stamps dropped)."""
        tail: list[dict[str, Any]] = []
        for message in messages[compress_end:]:
            if not isinstance(message, dict):
                continue
            row = _fresh_compaction_message_copy(message)
            row[_COMPACTION_TAIL_MARKER] = True
            tail.append(row)
        return tail

    def _register_block(self, store: EpisodeStore, text: str) -> None:
        """Make the new block the cache anchor and record it for the next process to re-register."""
        try:
            from agent.prompt_cache_boundary import register_cache_anchor

            register_cache_anchor(text)
        except Exception:
            logger.debug("ctxpkg anchor registration failed", exc_info=True)
        try:
            store.append_block(text)
        except OSError:
            logger.debug("ctxpkg block log write failed", exc_info=True)

    # ── span → episodes ────────────────────────────────────────────────────

    def _episodes_for_span(self, span: list[dict[str, Any]], settings: CtxpkgSettings,
                           store: EpisodeStore) -> list[Episode]:
        """The span's turns as episodes, with the engine's own rows and pre-turn leftovers dropped."""
        messages: list[dict[str, Any]] = []
        started = False
        for message in span:
            if is_state_note(message) or is_state_filler(message):
                continue
            if not started and message.get("role") != "user":
                continue  # nothing before the span's first user message can open an episode
            started = True
            messages.append(message)
        if not messages:
            return []
        return segment(messages, max_chars=settings.max_episode_chars, first_turn=store.next_turn())

    def _warm_recall_pairs(self, span: list[dict[str, Any]], store: EpisodeStore, turn: int,
                           warm_turns: int) -> list[dict[str, Any]]:
        """Warm recall pairs in *span*, oldest first. A recall in the protected tail is left there."""
        calls: dict[str, dict[str, Any]] = {}
        for message in span:
            for call in message.get("tool_calls") or []:
                ref = recall_ref(call)
                call_id = call.get("id") if isinstance(call, dict) else None
                if ref and call_id and call_id not in calls:
                    calls[call_id] = {"call": call, "ref": ref}
        results = {message.get("tool_call_id"): message for message in span
                   if message.get("role") == "tool" and message.get("tool_call_id") in calls}
        pairs: list[dict[str, Any]] = []
        for call_id, entry in calls.items():
            result = results.get(call_id)
            if result is None:
                continue  # an unanswered call is the host sanitizer's problem, not a pair to move
            if not self._is_warm(entry["ref"], store, turn, warm_turns):
                continue
            pairs.append({"call": entry["call"], "result": result, "ref": entry["ref"]})
        return pairs

    def _is_warm(self, ref: str, store: EpisodeStore, turn: int, warm_turns: int) -> bool:
        """Warm means ``current turn - last use < warm_turns``; ``0`` keeps nothing warm."""
        if warm_turns <= 0:
            return False
        last_use = store.last_use(ref)
        return last_use is not None and (turn - last_use) < warm_turns

    # ── cold stubs and protection indexes ──────────────────────────────────

    def _recall_ref_of(self, message: dict[str, Any], messages: list[dict[str, Any]]) -> str | None:
        """The recall ref *message* carries, either as its own call or as a result's answer.

        A result's call is looked up across the whole transcript, not just the exchange it sits in: a
        micro pass can leave a result whose call row is before the cursor, and absorbing that exchange
        would drop the answer and leave the call unanswered.
        """
        for call in message.get("tool_calls") or []:
            ref = recall_ref(call)
            if ref:
                return ref
        if message.get("role") != "tool":
            return None
        return self._recall_ref_for_call_id(message.get("tool_call_id"), messages)

    def _recall_ref_for_call_id(self, call_id: Any, messages: list[dict[str, Any]]) -> str | None:
        """The ref of the ``recall`` call *call_id* answers, wherever that call's row sits."""
        if not call_id:
            return None
        for candidate in messages:
            if not isinstance(candidate, dict):
                continue
            for call in candidate.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id") == call_id:
                    return recall_ref(call)
        return None

    def _stub_cold_recalls(self, messages: list[dict[str, Any]], store: EpisodeStore, turn: int,
                           warm_turns: int) -> int:
        """Replace each cold recall's result with the one-line cold stub; returns how many moved."""
        refs: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                ref = recall_ref(call)
                if ref and isinstance(call, dict) and call.get("id"):
                    refs[call["id"]] = ref
        stubbed = 0
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            ref = refs.get(message.get("tool_call_id"))
            if not ref or self._is_warm(ref, store, turn, warm_turns):
                continue
            text = message_text(message)
            if is_cold_stub(text):
                continue
            messages[index] = {**message, "content": cold_stub(ref, store.summary_of(ref))}
            stubbed += 1
        return stubbed

    def _protected_indexes(self, messages: list[dict[str, Any]], store: EpisodeStore, turn: int,
                           warm_turns: int) -> set[int]:
        """Indexes a prune pass must leave byte-identical: block pairs, state note, warm recall pairs.

        Once a block exists, every index up to the newest ack joins them. That region is the cache prefix
        (system, the first compaction's head rows, every block pair and the state note), and prune rewrites
        rows inside it: its dedupe pass replaces a tool result whose text equals a later one, and its
        demote pass shortens any tool result over ``_PRUNE_MIN_CHARS``. Either one changes bytes before
        the first block, so system + head + every block would miss the cache on the next request.
        """
        protected: set[int] = set()
        pairs = self._block_pairs(messages)
        if pairs:
            protected.update(range(min(pairs[-1][1] + 1, len(messages))))
        for user_index, ack_index in pairs:
            protected.update((user_index, ack_index))
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if is_state_note(message) or is_state_filler(message):
                protected.add(index)
        refs: dict[str, str] = {}
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            for call in message.get("tool_calls") or []:
                ref = recall_ref(call)
                if ref and isinstance(call, dict) and call.get("id"):
                    refs[call["id"]] = ref
                    if self._is_warm(ref, store, turn, warm_turns):
                        protected.add(index)
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            ref = refs.get(message.get("tool_call_id"))
            if ref and self._is_warm(ref, store, turn, warm_turns):
                protected.add(index)
        return protected

    def _prefix_floor(self, messages: list[dict[str, Any]], tail_start: int) -> int:
        """Index after the last block ack, state note or filler inside the compressible window."""
        floor = 0
        limit = min(tail_start, len(messages))
        for index in range(limit):
            message = messages[index]
            if not isinstance(message, dict):
                continue
            if is_block_ack(message) or is_state_note(message) or is_state_filler(message):
                floor = max(floor, index + 1)
        return floor

    # ── summaries ──────────────────────────────────────────────────────────

    def _fail_if_abandoned(self) -> None:
        """Unwind as a cancellation when the host abandoned or cancelled this attempt.

        A compaction sits inside two flash calls while the host's timeout can detach this worker and
        retry the same window in a new one (``_raise_if_stale_attempt`` is what the base's own summary
        path consults). A detached attempt must write no episode, no ``blocks.jsonl`` line and no anchor
        the newer attempt cannot see.
        """
        from agent.auxiliary_client import AuxiliaryExplicitCancellation
        from agent.conversation_compression import _raise_if_stale_attempt

        _raise_if_stale_attempt(self)
        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()

    def _episode_summaries(self, refs: list[str], episodes: list[Episode], rendered: list[str],
                           ids: list[list[str]], settings: CtxpkgSettings) -> tuple[list[str], bool]:
        """One flash call for every episode's summary; any failure falls back to the mechanical line.

        Returns the summaries and whether the call failed; the state note skips its own call after a
        failure, because a second timeout doubles the wait for a note the fallback already covers.
        """
        mechanical = [mechanical_summary(episode, _tool_names(episode)) for episode in episodes]
        if not refs:
            return [], False
        try:
            reply = self._call_summariser(self._summary_prompt(refs, rendered, settings))
        except Exception as exc:
            logger.info("ctxpkg episode summaries failed, using mechanical lines: %s", exc)
            return [cut_words(line, settings.summary_max_chars) for line in mechanical], True
        parsed = _parse_summary_lines(reply)
        summaries: list[str] = []
        for index, ref in enumerate(refs):
            summary = parsed.get(ref)
            if not summary:
                summaries.append(cut_words(mechanical[index], settings.summary_max_chars))
                continue
            summaries.append(cut_words(scrub_ids(summary, ids[index]), settings.summary_max_chars)
                             or cut_words(mechanical[index], settings.summary_max_chars))
        return summaries, False

    def _state_letter(self, messages: list[dict[str, Any]], lines: list[str],
                      summary_failed: bool = False) -> str:
        """One flash call for the state note; a failure keeps the last user text instead."""
        last_user = ""
        for message in reversed(list(messages)):
            if message.get("role") == "user" and not is_state_note(message):
                last_user = message_text(message)
                break
        fallback = " ".join(last_user.split())[:STATE_MAX_CHARS]
        if summary_failed:
            return fallback
        try:
            reply = self._call_summariser(self._state_prompt(messages, lines))
        except Exception as exc:
            logger.info("ctxpkg state note failed, using the last user text: %s", exc)
            return fallback
        note = " ".join((reply or "").split())
        return note[:STATE_MAX_CHARS] if note else fallback

    def _summary_prompt(self, refs: list[str], rendered: list[str],
                        settings: CtxpkgSettings) -> str:
        sections = []
        for ref, text in zip(refs, rendered):
            sections.append(f"### {ref}\n{text}")
        return (
            "You write one-line summaries of compacted conversation episodes for an agent's context.\n"
            f"Reply with exactly one line per episode, in order, as `{refs[0]}: <summary>`.\n"
            f"Each summary is at most {settings.summary_max_chars} characters and says what was done and what it "
            "produced. Copy identifiers (commit shas, task ids, paths) EXACTLY as they appear in the episode; "
            "never invent one. No preamble, no blank lines, no bullet points.\n\n"
            + "\n\n".join(sections)
        )

    def _state_prompt(self, messages: list[dict[str, Any]], lines: list[str]) -> str:
        recent: list[str] = []
        for message in reversed(list(messages)):
            if message.get("role") == "user" and not is_state_note(message):
                recent.append(message_text(message)[:1200])
            if len(recent) >= 3:
                break
        return (
            "You track what a coding agent has in flight, for a short note it reads after its earlier turns "
            "were compacted. In at most "
            f"{STATE_MAX_CHARS} characters, state the goal, the plan path, the step, any open child ids and "
            "the next action. Reply with the note only - no heading, no preamble.\n\n"
            "Recent work:\n" + "\n".join(lines) + "\n\n"
            "Latest requests:\n" + "\n---\n".join(reversed(recent))
        )

    def _call_summariser(self, prompt: str) -> str:
        """One ``auxiliary.compression`` call, routed and interrupt-protected exactly as the base's is."""
        from agent.auxiliary_client import aux_interrupt_protection, call_llm, extract_content_or_reasoning

        call_kwargs: dict[str, Any] = {
            "task": "compression",
            "main_runtime": {
                "model": self.model, "provider": self.provider, "base_url": self.base_url,
                "api_key": self.api_key, "api_mode": self.api_mode,
            },
            "messages": [{"role": "user", "content": prompt}],
            "route_info": {},
        }
        self._apply_summary_route(call_kwargs, _pinned_summary_call_kwargs())
        with aux_interrupt_protection():
            response = call_llm(**call_kwargs)
        return extract_content_or_reasoning(response, max_reasoning_chars=0).strip()


def _parse_summary_lines(reply: str) -> dict[str, str]:
    """``ep-NNNN: summary`` (or ``ep-NNNN · summary``) lines, last one per ref winning."""
    found: dict[str, str] = {}
    for line in (reply or "").splitlines():
        stripped = line.strip().lstrip("-*0123456789. ").strip()
        match = re.match(r"^(ep-[0-9]{4})\s*[:·—-]\s*(.+)$", stripped)
        if match and match.group(2).strip():
            found[match.group(1)] = " ".join(match.group(2).split())
    return found
