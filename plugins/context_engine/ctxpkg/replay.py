"""The ``ctxpkg`` replay witness: what real sessions would have cost under each compressor.

Run as ``python -m plugins.context_engine.ctxpkg.replay --db <state.db> --session <id> [...] --thresholds
80000,120000,160000,300000``. It opens the database read-only, replays each session's message sequence call
by call (a call boundary is every assistant message that carries content or tool calls; an empty placeholder
row is not a call), under two models, and prints one ``RESULT`` row per session, model and threshold, one
``CALIBRATION`` row per session, the ``CHOSEN`` threshold and one ``DECISION=ship|hold`` line. It never prints
message text, and it selects no reasoning column.

The two models
--------------
``default`` is the built-in compressor as falcor runs it: after a compaction the prompt is the head plus a
summary plus the last ``TAIL_MESSAGES`` messages, and the first call after it writes everything after the
system prompt. ``ctxpkg`` is the engine this plan builds: after a compaction the prompt is the head plus
every reference block written so far plus a state note plus the tail, and the first call after it *reads*
the system prompt, the head and every block that existed before this compaction, writing only the new block,
the state note and the tail. That difference — an extend-only prefix instead of a rewritten one — is what the
replay prices.

Neither model includes proactive prune or micro-compaction: they keep real contexts lower and add their own
cache breaks. The calibration gate is the only guard against that error, which is why it checks the
cache-write side separately.

The per-call add-on
-------------------
The rows hold only messages; every real prompt also carries the system prompt, the tool schemas and the
replayed thinking. The plan defines that overhead as the recorded mean prompt per call minus the replay's
mean message size, in one pass and floored at 0. The message size is the mean over calls of the message part
of the default model's prompt at :data:`CALIBRATION_THRESHOLD` — the same replay with an add-on of 0 — and the
resulting add-on is then applied to every call in both models. It is deliberately not solved for the level:
a fixed point would pin the default replay's mean to the recording, and since reads are the prompt minus the
writes, ``read_ratio`` would then agree by construction and the gate's read check would carry no information.
As it is, the ``CALIBRATION`` row's total ratio measures the cost gap left over, and its read/write ratios say
which side the gap sits on.

Column names
------------
Read from ``hermes_state_common.py``'s DDL: ``messages.role``, ``content``, ``tool_calls``, ``active``,
``compacted`` and ``_compressed_summary`` (a 0/1 flag marking a stored compaction summary, never its text),
and ``sessions.api_call_count``, ``input_tokens``, ``cache_read_tokens``, ``cache_write_tokens``,
``output_tokens``, ``estimated_cost_usd``, ``model`` and ``billing_provider``. The reasoning columns
(``reasoning``, ``reasoning_content``, ``reasoning_details``, ``codex_reasoning_items``) exist and are
deliberately never selected: see :data:`SELECTED_COLUMNS`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from agent.model_metadata import estimate_tokens_rough

from .compaction import STATE_MAX_CHARS, block_ack, block_line, block_text, state_note
from .episodes import extract_ids, render, segment

MODEL_DEFAULT = "default"
MODEL_CTXPKG = "ctxpkg"
MODELS = (MODEL_DEFAULT, MODEL_CTXPKG)

#: Falcor's ``compression.protect_first_n`` and ``protect_last_n`` in the sessions this ran on.
PROTECT_FIRST_N = 3
TAIL_MESSAGES = 20
#: The threshold the CALIBRATION row is taken at: falcor's setting in those sessions.
CALIBRATION_THRESHOLD = 300_000
#: The summary size assumed when a session holds no stored summary row of its own.
DEFAULT_SUMMARY_TOKENS = 12_000
#: The block's per-episode summary is fixed, so no model is called while replaying.
FIXED_SUMMARY_CHARS = 200
#: Flash output per episode; one summary call per compaction, plus the state note.
EPISODE_OUTPUT_TOKENS = 40
DEFAULT_THRESHOLDS = (80_000, 120_000, 160_000, 300_000)

#: Rates used for the compression model when the fork's price table has no entry for it.
FLASH_IN_USD_PER_MTOK = Decimal("0.30")
FLASH_OUT_USD_PER_MTOK = Decimal("1.20")
#: The plan's gate bands, as written.
TOTAL_BAND = (Decimal("0.85"), Decimal("1.15"))
COMPONENT_BAND = (Decimal("0.7"), Decimal("1.3"))
SHIP_MARGIN = Decimal("0.9")

_MILLION = Decimal(1_000_000)

#: Every column the replay reads from ``messages``. Nothing here may be a reasoning column.
SELECTED_COLUMNS = ("id", "role", "content", "tool_calls", "_compressed_summary")
ROW_SQL = (
    "SELECT id, role, content, tool_calls, _compressed_summary FROM messages "
    "WHERE session_id = ? AND (active = 1 OR compacted = 1) "
    "AND COALESCE(_compressed_summary, 0) = 0 AND role != 'session_meta' ORDER BY id"
)
SUMMARY_SQL = (
    "SELECT content FROM messages WHERE session_id = ? AND COALESCE(_compressed_summary, 0) = 1 ORDER BY id"
)
SESSION_SQL = (
    "SELECT id, model, billing_provider, api_call_count, input_tokens, cache_read_tokens, "
    "cache_write_tokens, output_tokens, estimated_cost_usd FROM sessions WHERE id = ?"
)


class SessionUnusable(Exception):
    """A session the replay refuses to price; the message is the ``SKIP`` reason it prints."""


# ── prices ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rates:
    """USD per million tokens for the four buckets a call is billed in."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write: Decimal


def session_rates(model: str, provider: str | None) -> Rates:
    """The fork's price for *model* on *provider*, or ``SessionUnusable`` when it has none."""
    from agent.usage_pricing import get_pricing_entry

    entry = get_pricing_entry(model, provider=provider) if model else None
    if entry is None:
        raise SessionUnusable(f"no price for model {model!r}")
    in_rate = entry.input_cost_per_million
    out_rate = entry.output_cost_per_million
    read_rate = entry.cache_read_cost_per_million
    write_rate = entry.cache_write_cost_per_million
    if in_rate is None or out_rate is None or read_rate is None or write_rate is None:
        missing = [name for name, value in (("input", in_rate), ("output", out_rate),
                                            ("cache-read", read_rate), ("cache-write", write_rate))
                   if value is None]
        raise SessionUnusable(f"model {model!r} has no {'/'.join(missing)} price")
    return Rates(input=in_rate, output=out_rate, cache_read=read_rate, cache_write=write_rate)


def compression_model() -> str:
    """The fork's ``auxiliary.compression`` model (empty when unset, which is the fork's default)."""
    try:
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        block = (DEFAULT_CONFIG.get("auxiliary") or {}).get("compression") or {}
        model = block.get("model")
        return model if isinstance(model, str) else ""
    except Exception:  # a config read must never stop a replay
        return ""


def flash_rates() -> tuple[Decimal, Decimal, str]:
    """The compression model's ``(in, out, source)`` rates: the fork's price, else the plan's fallback."""
    from agent.usage_pricing import get_pricing_entry

    model = compression_model()
    entry = get_pricing_entry(model) if model else None
    if entry is not None and entry.input_cost_per_million is not None \
            and entry.output_cost_per_million is not None:
        return entry.input_cost_per_million, entry.output_cost_per_million, f"fork price, {model}"
    if model:
        reason = f"no price for {model}"
    else:
        reason = "auxiliary.compression model unset in the fork's defaults"
    return (FLASH_IN_USD_PER_MTOK, FLASH_OUT_USD_PER_MTOK,
            f"{reason}; using ${FLASH_IN_USD_PER_MTOK}/${FLASH_OUT_USD_PER_MTOK} per Mtok")


# ── rows ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    """One replayed row: what the model may look at, plus its estimated token size."""

    role: str
    content: str = ""
    tool_calls: tuple[Any, ...] = ()
    tokens: int = 0

    def as_api_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = list(self.tool_calls)
        return message

    @property
    def is_call(self) -> bool:
        """Whether this row is a call boundary.

        A call boundary is an assistant message that carries something: content or at least one tool call. An
        assistant row with neither is a placeholder the recorder left (``finish_reason`` NULL, no API call), so
        it joins the prompt like any other row — at zero tokens — but does not open a call.
        """
        return self.role == "assistant" and bool(self.content or self.tool_calls)


@dataclass(frozen=True)
class Recorded:
    """What the session itself recorded."""

    session_id: str
    model: str
    provider: str | None
    calls: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: Decimal

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def mean_prompt(self) -> float:
        return (self.prompt_tokens / self.calls) if self.calls else 0.0


@dataclass(frozen=True)
class Session:
    """A session's replayable rows and its recorded totals."""

    session_id: str
    messages: tuple[Message, ...]
    summary_tokens: int
    recorded: Recorded


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _message_tokens(content: Any, tool_calls: Any) -> int:
    """The engine's own estimator over the two text surfaces a row carries."""
    return estimate_tokens_rough(content or "") + estimate_tokens_rough(tool_calls or "")


def open_read_only(db_path: str | Path) -> sqlite3.Connection:
    """A connection that cannot write: the URI's ``mode=ro``, never a copy of the file."""
    path = Path(db_path)
    if not path.is_file():
        raise SessionUnusable(f"no database at {path.name}")
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _row_dict(cursor: sqlite3.Cursor, row: Any) -> dict[str, Any]:
    """A row as a dict, whether or not the connection set ``sqlite3.Row`` as its factory."""
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return {column[0]: value for column, value in zip(cursor.description or (), row)}


def load_session(conn: sqlite3.Connection, session_id: str) -> Session:
    """Load one session's rows, refusing it when its assistant count misses ``api_call_count``."""
    cursor = conn.execute(SESSION_SQL, (session_id,))
    fetched = cursor.fetchone()
    if fetched is None:
        raise SessionUnusable("no session row")
    row = _row_dict(cursor, fetched)
    cursor = conn.execute(ROW_SQL, (session_id,))
    rows = [_row_dict(cursor, item) for item in cursor.fetchall()]
    assistant_rows = sum(1 for item in rows if item["role"] == "assistant")
    calls = _int(row["api_call_count"])
    if abs(assistant_rows - calls) > 1:
        raise SessionUnusable(f"assistant rows {assistant_rows} vs api_call_count {calls}")
    messages: list[Message] = []
    for item in rows:
        calls_json = item["tool_calls"] or ""
        parsed: tuple[Any, ...] = ()
        if calls_json:
            try:
                value = json.loads(calls_json)
            except ValueError:
                value = []
            parsed = tuple(value) if isinstance(value, list) else ()
        content = item["content"] or ""
        messages.append(Message(
            role=str(item["role"]), content=content, tool_calls=parsed,
            tokens=_message_tokens(content, calls_json),
        ))
    cursor = conn.execute(SUMMARY_SQL, (session_id,))
    summaries = [estimate_tokens_rough(_row_dict(cursor, item)["content"] or "") for item in cursor.fetchall()]
    summary_tokens = (sum(summaries) // len(summaries)) if summaries else DEFAULT_SUMMARY_TOKENS
    if not messages:
        raise SessionUnusable("no replayable rows")
    recorded = Recorded(
        session_id=session_id, model=str(row["model"] or ""), provider=row["billing_provider"] or None,
        calls=calls, input_tokens=_int(row["input_tokens"]),
        cache_read_tokens=_int(row["cache_read_tokens"]), cache_write_tokens=_int(row["cache_write_tokens"]),
        output_tokens=_int(row["output_tokens"]), cost_usd=Decimal(str(row["estimated_cost_usd"] or 0)),
    )
    return Session(session_id=session_id, messages=tuple(messages), summary_tokens=summary_tokens,
                   recorded=recorded)


# ── one replay ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CallRecord:
    """One replayed call: its prompt, and the part of it the cache wrote rather than read."""

    index: int
    prompt_tokens: int
    read_tokens: int
    write_tokens: int
    after_compaction: bool = False
    cached_blocks: int = 0
    wrote_block: bool = False


@dataclass
class ReplayResult:
    """One session, one model, one threshold."""

    session_id: str
    model: str
    threshold: int
    rates: Rates
    calls: list[CallRecord] = field(default_factory=list)
    compactions: int = 0
    blocks: list[int] = field(default_factory=list)
    flash_input_tokens: int = 0
    flash_episodes: int = 0
    flash_in_rate: Decimal = FLASH_IN_USD_PER_MTOK
    flash_out_rate: Decimal = FLASH_OUT_USD_PER_MTOK

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def mean_prompt(self) -> float:
        return (self.prompt_tokens / len(self.calls)) if self.calls else 0.0

    @property
    def read_tokens(self) -> int:
        return sum(call.read_tokens for call in self.calls)

    @property
    def write_tokens(self) -> int:
        return sum(call.write_tokens for call in self.calls)

    @property
    def read_usd(self) -> Decimal:
        return Decimal(self.read_tokens) * self.rates.cache_read / _MILLION

    @property
    def write_usd(self) -> Decimal:
        return Decimal(self.write_tokens) * self.rates.cache_write / _MILLION

    @property
    def flash_usd(self) -> Decimal:
        return self.flash_cost(self.flash_input_tokens, self.flash_episodes)

    def flash_cost(self, input_tokens: int, episodes: int) -> Decimal:
        """One flash summary call per compaction: the span in, forty tokens out per episode."""
        return (Decimal(input_tokens) * self.flash_in_rate
                + Decimal(episodes * EPISODE_OUTPUT_TOKENS) * self.flash_out_rate) / _MILLION

    def output_usd_for(self, output_tokens: int) -> Decimal:
        return Decimal(output_tokens) * self.rates.output / _MILLION

    def total_usd_for(self, output_tokens: int) -> Decimal:
        """Reads, writes, the compaction's flash calls and the session's own recorded output."""
        return self.read_usd + self.write_usd + self.flash_usd + self.output_usd_for(output_tokens)


def _block_tokens(span: Sequence[dict[str, Any]], turn: int, ref_number: int) -> tuple[int, int, int]:
    """The block pair's real byte size for *span*, with a fixed 200-char summary per episode.

    Returns ``(tokens, episodes, next_ref_number)``. The span is cut with the engine's own ``segment`` and
    rendered with its own renderer, so a block's size here is the size the engine would emit; only the
    summary text is a stand-in, which is why no model is called.
    """
    episodes = segment(span, first_turn=turn)
    lines: list[str] = []
    refs: list[str] = []
    for offset, episode in enumerate(episodes):
        text = render(episode)
        ref = f"ep-{ref_number + offset:04d}"
        refs.append(ref)
        lines.append(block_line(ref, episode.turn_first, episode.turn_last, len(text),
                                "x" * FIXED_SUMMARY_CHARS, extract_ids(text)))
    if not lines:
        return 0, 0, ref_number
    text = block_text(refs[0], refs[-1], lines) + "\n" + block_ack(refs[0], refs[-1])
    return estimate_tokens_rough(text), len(episodes), ref_number + len(episodes)


def _state_note_tokens(refs: Sequence[str]) -> int:
    """The state note's size at the engine's own 1500-char cap."""
    return estimate_tokens_rough(state_note("x" * STATE_MAX_CHARS, refs))


def replay(session: Session, add_on: int, threshold: int, model: str,
           rates: Rates | None = None) -> ReplayResult:
    """Replay *session*'s calls under *model*, compacting whenever the prompt reaches *threshold*.

    A compaction happens at most once per call, and only when the span between the head and the protected
    tail holds something: for ``ctxpkg``, something that segments into at least one episode — the engine's
    own no-op rule, since a block with no line would spend a compaction for no reclaim. The compacting call
    is the one that writes the whole prompt it leaves behind; every call after it writes only the rows
    appended since the previous call.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    result = ReplayResult(
        session_id=session.session_id, model=model, threshold=threshold,
        rates=rates or Rates(Decimal(0), Decimal(0), Decimal(0), Decimal(0)),
    )
    summary_tokens = session.summary_tokens
    head_tokens: int | None = None      # set at the first compaction, fixed afterwards
    history: list[tuple[dict[str, Any], int]] = []   # every row seen so far, before any compaction
    body: list[tuple[dict[str, Any], int]] = []      # the prompt's rows after the head, once compacted
    engine_lead = False                 # body[0] is the engine's own row (summary or state note)
    state_tokens = 0
    ref_number = 0
    turn = 1
    pending = 0                         # tokens appended to the prompt since the previous call
    first_call = True

    def message_tokens() -> int:
        if head_tokens is None:
            return sum(tokens for _, tokens in history)
        engine = (summary_tokens if model == MODEL_DEFAULT else state_tokens) if engine_lead else 0
        extra = engine + (sum(result.blocks) if model == MODEL_CTXPKG else 0)
        return head_tokens + extra + sum(tokens for _, tokens in body)

    for message in session.messages:
        row = (message.as_api_message(), message.tokens)
        if message.is_call:
            if head_tokens is None:
                span = history[PROTECT_FIRST_N:len(history) - TAIL_MESSAGES]
            else:
                start = 1 if engine_lead else 0
                span = body[start:len(body) - TAIL_MESSAGES]
            prompt = add_on + message_tokens()
            fire = prompt >= threshold and bool(span)
            size = episodes = 0
            next_ref = ref_number
            if fire and model == MODEL_CTXPKG:
                size, episodes, next_ref = _block_tokens([item for item, _ in span], turn, ref_number + 1)
                fire = episodes > 0
            if fire:
                span_tokens = sum(tokens for _, tokens in span)
                result.compactions += 1
                if head_tokens is None:
                    head_tokens = sum(tokens for _, tokens in history[:PROTECT_FIRST_N])
                    body = list(history[-TAIL_MESSAGES:])
                else:
                    body = list(body[-TAIL_MESSAGES:])
                engine_lead = True
                if model == MODEL_DEFAULT:
                    prompt = add_on + message_tokens()
                    read, write = add_on, prompt - add_on
                    cached_blocks, wrote_block = 0, False
                else:
                    ref_number = next_ref
                    turn += episodes
                    result.blocks.append(size)
                    state_tokens = _state_note_tokens(
                        [f"ep-{number:04d}" for number in range(ref_number - episodes + 1, ref_number + 1)])
                    prompt = add_on + message_tokens()
                    read = add_on + (head_tokens or 0) + sum(result.blocks[:-1])
                    write = prompt - read
                    cached_blocks, wrote_block = len(result.blocks) - 1, True
                    result.flash_input_tokens += span_tokens
                    result.flash_episodes += episodes
                result.calls.append(CallRecord(
                    index=len(result.calls), prompt_tokens=prompt, read_tokens=read, write_tokens=write,
                    after_compaction=True, cached_blocks=cached_blocks, wrote_block=wrote_block,
                ))
            else:
                # No compaction at this call: the compaction's own call (if there was one) already wrote the
                # whole span it left behind, so this one writes only the rows appended since the previous
                # call and reads the rest. A session's very first call is cold and writes it all.
                if first_call:
                    read, write = 0, prompt
                else:
                    write = min(pending, prompt)
                    read = prompt - write
                result.calls.append(CallRecord(
                    index=len(result.calls), prompt_tokens=prompt, read_tokens=read, write_tokens=write,
                ))
            first_call = False
            pending = 0
        history.append(row)
        if head_tokens is not None:
            body.append(row)
        pending += message.tokens
    return result


def calibration_add_on(session: Session, rates: Rates) -> tuple[int, ReplayResult]:
    """The per-call fixed overhead, as the plan states it: one subtraction, floored at 0.

    The add-on is the recorded mean ``input + cache_read + cache_write`` per call minus the replay's mean
    message size. The message size is measured once: the mean over calls of the default model's prompt at
    :data:`CALIBRATION_THRESHOLD` with no add-on, which is exactly the message part, since the add-on is what
    the prompt carries beyond the rows. The subtraction is not iterated to a fixed point — a level pinned to
    the recording would force ``read_ratio`` to agree by construction and empty the gate's read check.
    ``(add_on, replay at that add-on)`` is returned; the second element is the run the ``CALIBRATION`` row
    reads, so its ratios say what is left of the gap and which side it sits on.
    """
    zero = replay(session, 0, CALIBRATION_THRESHOLD, MODEL_DEFAULT, rates)
    if not session.recorded.calls or not zero.calls:
        return 0, zero
    add_on = int(session.recorded.mean_prompt - zero.mean_prompt)
    if add_on <= 0:
        # Floored at 0: a transcript the recording does not cover is a gate finding, not a negative add-on.
        return 0, zero
    return add_on, replay(session, add_on, CALIBRATION_THRESHOLD, MODEL_DEFAULT, rates)


# ── the gate ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Calibration:
    """One session's ``CALIBRATION`` row: the default model at the calibration threshold against the recording."""

    session_id: str
    replay_total_usd: Decimal
    recorded_usd: Decimal
    ratio: Decimal
    read_ratio: Decimal
    write_ratio: Decimal


def calibration_of(session: Session, replay_result: ReplayResult) -> Calibration:
    """Build a session's calibration row from its default-model run at the calibration threshold."""
    recorded = session.recorded
    total = replay_result.total_usd_for(recorded.output_tokens)
    return Calibration(
        session_id=session.session_id,
        replay_total_usd=total,
        recorded_usd=recorded.cost_usd,
        ratio=_ratio(total, recorded.cost_usd),
        read_ratio=_ratio(Decimal(replay_result.read_tokens), Decimal(recorded.cache_read_tokens)),
        write_ratio=_ratio(Decimal(replay_result.write_tokens), Decimal(recorded.cache_write_tokens)),
    )


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    """*numerator* / *denominator*; a zero denominator reads as ``0`` rather than raising."""
    if not denominator:
        return Decimal(0)
    return numerator / denominator


def choose_threshold(totals: Mapping[int, Decimal]) -> int:
    """The threshold with the lowest summed ctxpkg total; ties go to the lower threshold."""
    return min(totals, key=lambda threshold: (totals[threshold], threshold))


def _in_band(value: Decimal, band: tuple[Decimal, Decimal]) -> bool:
    return band[0] <= value <= band[1]


def decide(calibrations: Sequence[Calibration], ctxpkg_totals: Mapping[int, Decimal],
           default_total: Decimal) -> tuple[str, str]:
    """``("ship", "")`` or ``("hold", reason)``, on the plan's bands and no others."""
    if not calibrations or not ctxpkg_totals:
        return "hold", "no session could be replayed"
    reasons: list[str] = []
    low, high = TOTAL_BAND
    for calibration in calibrations:
        if not _in_band(calibration.ratio, TOTAL_BAND):
            reasons.append(f"{calibration.session_id} total ratio {calibration.ratio:.3f} outside "
                           f"[{low},{high}]")
        if not _in_band(calibration.read_ratio, COMPONENT_BAND):
            reasons.append(f"{calibration.session_id} read_ratio {calibration.read_ratio:.3f} outside "
                           f"[{COMPONENT_BAND[0]},{COMPONENT_BAND[1]}]")
        if not _in_band(calibration.write_ratio, COMPONENT_BAND):
            reasons.append(f"{calibration.session_id} write_ratio {calibration.write_ratio:.3f} outside "
                           f"[{COMPONENT_BAND[0]},{COMPONENT_BAND[1]}]")
    chosen = choose_threshold(ctxpkg_totals)
    margin = SHIP_MARGIN * default_total
    if ctxpkg_totals[chosen] > margin:
        share = _ratio(ctxpkg_totals[chosen], default_total)
        reasons.append(f"ctxpkg ${ctxpkg_totals[chosen]:.4f} at threshold {chosen} is {share:.3f}x the "
                       f"default's ${default_total:.4f}, above {SHIP_MARGIN}")
    if reasons:
        return "hold", "; ".join(reasons)
    return "ship", ""


# ── printing and main ──────────────────────────────────────────────────────

def result_line(result: ReplayResult, output_tokens: int) -> str:
    """One ``RESULT <session> <model> <threshold> ...`` row."""
    return (f"RESULT {result.session_id} {result.model} {result.threshold} "
            f"calls={result.call_count} mean_prompt={int(result.mean_prompt)} "
            f"compactions={result.compactions} read$={result.read_usd:.4f} write$={result.write_usd:.4f} "
            f"flash$={result.flash_usd:.4f} total$={result.total_usd_for(output_tokens):.4f}")


def recorded_line(session: Session) -> str:
    """One ``RESULT <session> recorded ...`` row: what the session itself booked."""
    return (f"RESULT {session.session_id} recorded calls={session.recorded.calls} "
            f"total$={session.recorded.cost_usd:.4f}")


def calibration_line(calibration: Calibration) -> str:
    """One ``CALIBRATION <session> ...`` row."""
    return (f"CALIBRATION {calibration.session_id} replay_default_total$={calibration.replay_total_usd:.4f} "
            f"recorded$={calibration.recorded_usd:.4f} ratio={calibration.ratio:.3f} "
            f"read_ratio={calibration.read_ratio:.3f} write_ratio={calibration.write_ratio:.3f}")


def parse_thresholds(text: str) -> tuple[int, ...]:
    """``80000,120000`` → ``(80000, 120000)``; a non-positive or malformed entry is refused."""
    values: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a threshold: {part!r}") from None
        if value <= 0:
            raise argparse.ArgumentTypeError(f"a threshold must be positive: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("no thresholds given")
    return tuple(values)


def run(db_path: str | Path, session_ids: Sequence[str],
        thresholds: Sequence[int]) -> tuple[list[str], int]:
    """Replay every session, print the rows, and return the process's exit code."""
    lines: list[str] = []
    in_rate, out_rate, source = flash_rates()
    lines.append(f"NOTE flash rates in=${in_rate} out=${out_rate} per Mtok ({source})")
    calibrations: list[Calibration] = []
    ctxpkg_totals = {int(threshold): Decimal(0) for threshold in thresholds}
    default_total = Decimal(0)
    try:
        conn = open_read_only(db_path)
    except (SessionUnusable, sqlite3.Error) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return lines, 2
    try:
        for session_id in session_ids:
            try:
                session = load_session(conn, session_id)
                rates = session_rates(session.recorded.model, session.recorded.provider)
            except (SessionUnusable, sqlite3.Error) as exc:
                lines.append(f"SKIP {session_id} reason={exc}")
                continue
            add_on, calibrated = calibration_add_on(session, rates)
            calibration = calibration_of(session, calibrated)
            calibrations.append(calibration)
            default_total += calibrated.total_usd_for(session.recorded.output_tokens)
            lines.append(recorded_line(session))
            for threshold in thresholds:
                for model in MODELS:
                    result = replay(session, add_on, int(threshold), model, rates)
                    result.flash_in_rate = in_rate
                    result.flash_out_rate = out_rate
                    lines.append(result_line(result, session.recorded.output_tokens))
                    if model == MODEL_CTXPKG:
                        ctxpkg_totals[int(threshold)] += result.total_usd_for(session.recorded.output_tokens)
            lines.append(calibration_line(calibration))
    finally:
        conn.close()
    if ctxpkg_totals:
        chosen = choose_threshold(ctxpkg_totals)
        lines.append(f"CHOSEN threshold={chosen}")
    decision, reason = decide(calibrations, ctxpkg_totals, default_total)
    lines.append(f"DECISION={decision}" + (f" reason={reason}" if reason else ""))
    return lines, 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m plugins.context_engine.ctxpkg.replay",
        description="Replay real sessions under the default compressor and ctxpkg, and record a decision.",
    )
    parser.add_argument("--db", required=True, help="path to a Hermes state.db (opened read-only)")
    parser.add_argument("--session", action="append", default=[], required=True, dest="sessions",
                        help="a session id to replay; repeat the flag for each")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
                        help="comma-separated compaction thresholds in tokens")
    args = parser.parse_args(argv)
    try:
        thresholds = parse_thresholds(args.thresholds)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2  # unreachable: parser.error exits
    lines, code = run(args.db, args.sessions, thresholds)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
