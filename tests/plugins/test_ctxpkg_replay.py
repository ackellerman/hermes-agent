"""The ``ctxpkg`` replay witness (plan context-packages, step 5).

The replay prices a real session twice: under the built-in compressor and under ``ctxpkg``. It opens the
state DB read-only, never selects a reasoning column and never prints message text; it refuses a session whose
assistant rows do not match ``api_call_count``; and it records one decision, shipping only when its own
default-model replay reproduces the recording inside the plan's bands and ctxpkg beats it by 10%.

The fixtures here are synthetic sessions with deliberately different sizes per session, so a wrong session's
row or a summed-up-wrong total fails rather than balancing out.
"""

from __future__ import annotations

import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

import hermes_state_common
import plugins.context_engine.ctxpkg.replay as replay_module
from agent.model_metadata import estimate_tokens_rough
from plugins.context_engine.ctxpkg.replay import (
    CALIBRATION_THRESHOLD,
    MODEL_CTXPKG,
    MODEL_DEFAULT,
    PROTECT_FIRST_N,
    ROW_SQL,
    SELECTED_COLUMNS,
    TAIL_MESSAGES,
    Calibration,
    Message,
    Session,
    choose_threshold,
    decide,
    load_session,
    main,
    open_read_only,
    replay,
    session_rates,
)

MODEL = "claude-opus-5-5"
PROVIDER = "anthropic"
MARKER = "zqx-marker-never-print"
REASONING_MARKER = "zqx-reasoning-only"


# ── fixtures ───────────────────────────────────────────────────────────────

def _db(tmp_path: Path) -> tuple[Path, sqlite3.Connection]:
    """A state.db with the real schema, built from ``hermes_state_common.SCHEMA_SQL``."""
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript(hermes_state_common.SCHEMA_SQL)
    return path, conn


def _turn(index: int, chars: int, tools: int = 2) -> list[tuple[str, str | None, list | None]]:
    """One user turn: the request, ``tools`` tool calls with results, then the answer."""
    rows: list[tuple[str, str | None, list | None]] = [("user", f"turn {index} " + "u" * chars, None)]
    for round_index in range(tools):
        rows.append(("assistant", None, [{
            "id": f"c{index}-{round_index}", "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps({"command": f"cmd {index}.{round_index}"})},
        }]))
        rows.append(("tool", f"result {index}.{round_index} " + "r" * (chars * 3), None))
    rows.append(("assistant", f"answer {index}", None))
    return rows


def _transcript(turns: int, chars: int = 400, prefix: str = "") -> list[tuple[str, str | None, list | None]]:
    rows: list[tuple[str, str | None, list | None]] = [("user", f"{prefix}hello driver", None)]
    for index in range(1, turns + 1):
        rows.extend(_turn(index, chars))
    return rows


def _insert(conn: sqlite3.Connection, session_id: str, rows, *, active: int = 1, compacted: int = 0,
            summary: int = 0, stamp: float = 1.0, reasoning: str | None = None) -> int:
    """Insert *rows* as messages; returns how many assistant rows were inserted."""
    assistants = 0
    for role, content, calls in rows:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp, reasoning, active, "
            "compacted, _compressed_summary) VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, role, content, json.dumps(calls) if calls else None, stamp, reasoning, active,
             compacted, summary),
        )
        stamp += 1.0
        assistants += 1 if role == "assistant" else 0
    return assistants


def _session_row(conn: sqlite3.Connection, session_id: str, *, calls: int, read: int = 0, write: int = 0,
                 output: int = 0, cost: float = 0.0, model: str = MODEL) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, model, billing_provider, api_call_count, input_tokens, "
        "cache_read_tokens, cache_write_tokens, output_tokens, estimated_cost_usd) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (session_id, "discord", 1.0, model, PROVIDER, calls, 0, read, write, output, cost),
    )


def _session_of(rows, summary_tokens: int = 900) -> Session:
    """A ``Session`` straight from row tuples, without a database (for the model's own tests)."""
    messages = []
    for role, content, calls in rows:
        content = content or ""
        calls_json = json.dumps(calls) if calls else ""
        messages.append(Message(role=role, content=content, tool_calls=tuple(calls or ()),
                                tokens=estimate_tokens_rough(content) + estimate_tokens_rough(calls_json)))
    return Session(session_id="20260101_120000_abcdef12", messages=tuple(messages),
                   summary_tokens=summary_tokens,
                   recorded=replay_module.Recorded(
                       session_id="20260101_120000_abcdef12", model=MODEL, provider=PROVIDER, calls=1,
                       input_tokens=0, cache_read_tokens=0, cache_write_tokens=0, output_tokens=0,
                       cost_usd=Decimal(0)))


def _rates():
    return session_rates(MODEL, PROVIDER)


def _run_main(capsys, db: Path, sessions, thresholds: str = "1000"):
    argv = ["--db", str(db), "--thresholds", thresholds]
    for session_id in sessions:
        argv.extend(["--session", session_id])
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ── tests ──────────────────────────────────────────────────────────────────

def test_reads_db_read_only(tmp_path, capsys):
    """The DB is opened ``mode=ro``: a 0444 file in a 0555 directory still replays, and a writable file
    still refuses the connection's writes."""
    path, conn = _db(tmp_path)
    session_id = "20260101_120000_aaa00001"
    calls = _insert(conn, session_id, _transcript(turns=3))
    _session_row(conn, session_id, calls=calls, read=1000, write=100, output=50, cost=0.5)
    conn.commit()
    conn.close()

    # the URI, not the filesystem: a writable copy still cannot be written through ``open_read_only``
    other = tmp_path / "writable.db"
    other.write_bytes(path.read_bytes())
    other.chmod(0o644)
    conn = open_read_only(other)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM messages")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('x','t',0)")
    finally:
        conn.close()

    path.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        code, out, err = _run_main(capsys, path, [session_id], thresholds="1000,200000")
        assert code == 0, err
        assert f"RESULT {session_id} recorded" in out
        assert "ERROR" not in out
        # the fixture really is a session the replay priced, not one it skipped
        assert f"SKIP {session_id}" not in out

        conn = open_read_only(path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM messages")
        finally:
            conn.close()
        with pytest.raises(PermissionError):
            path.write_text("clobbered")
    finally:
        tmp_path.chmod(0o755)
        path.chmod(0o644)


def test_no_reasoning_columns_selected(tmp_path, capsys):
    """A marker planted in ``reasoning`` reaches neither the SQL nor the output."""
    path, conn = _db(tmp_path)
    session_id = "20260101_120000_aaa00002"
    rows = _transcript(turns=3)
    calls = _insert(conn, session_id, rows, reasoning=REASONING_MARKER)
    _session_row(conn, session_id, calls=calls, read=1000, write=100, output=50, cost=0.5)
    conn.commit()
    conn.close()

    for name in SELECTED_COLUMNS:
        assert "reasoning" not in name.lower()
    assert "reasoning" not in ROW_SQL.lower()

    executed: list[str] = []
    conn = open_read_only(path)
    conn.set_trace_callback(executed.append)
    try:
        load_session(conn, session_id)
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert executed, "the loader ran no SQL at all"
    assert not any("reasoning" in statement.lower() for statement in executed)

    code, out, err = _run_main(capsys, path, [session_id])
    assert code == 0, err
    assert REASONING_MARKER not in out and REASONING_MARKER not in err


def test_row_filter_and_call_count_gate(tmp_path, capsys):
    """Superseded rows, the summary row and session_meta are out while archived originals stay in; a call
    count two off prints no RESULT, one off is accepted."""
    path, conn = _db(tmp_path)
    good = "20260101_120000_aaa00003"
    near = "20260101_120000_aaa00005"
    off = "20260101_120000_aaa00004"

    calls = _insert(conn, good, _transcript(turns=3, chars=400))
    # Rows the filter must drop: superseded (neither active nor compacted), a stored summary and session_meta.
    _insert(conn, good, [("assistant", "superseded " + "s" * 5000, None)], active=0, compacted=0)
    _insert(conn, good, [("user", "SUPERSEDED-MARKER " + "x" * 400, None)], active=0, compacted=0)
    summary_text = "summary " + "m" * 4000
    _insert(conn, good, [("assistant", summary_text, None)], active=0, compacted=1, summary=1)
    _insert(conn, good, [("session_meta", "META-MARKER " + "z" * 500, None)], active=0, compacted=1)
    # A row the filter must keep: an original the recorder archived (compacted, not superseded).
    _insert(conn, good, [("user", "ARCHIVED-ORIGINAL-MARKER " + "a" * 300, None)], active=0, compacted=1)
    _session_row(conn, good, calls=calls, read=1000, write=100, output=50, cost=0.5)

    # One assistant row more than the recorder counted is inside the plan's tolerance.
    near_calls = _insert(conn, near, _transcript(turns=2, chars=700))
    _session_row(conn, near, calls=near_calls - 1, read=700, write=70, output=30, cost=0.25)

    # Two off is not: the row filter and the recorder disagree, so the session is refused outright.
    off_calls = _insert(conn, off, _transcript(turns=4, chars=900))
    _session_row(conn, off, calls=off_calls + 2, read=10, write=1, output=5, cost=0.05)
    conn.commit()
    conn.close()

    conn = open_read_only(path)
    try:
        session = load_session(conn, good)
    finally:
        conn.close()
    assert all(message.role != "session_meta" for message in session.messages)
    text = "\n".join(message.content for message in session.messages)
    assert "SUPERSEDED-MARKER" not in text
    assert "META-MARKER" not in text
    assert summary_text not in text
    assert "ARCHIVED-ORIGINAL-MARKER" in text
    assert session.summary_tokens == estimate_tokens_rough(summary_text)
    assert sum(1 for message in session.messages if message.role == "assistant") == calls

    code, out, err = _run_main(capsys, path, [good, near, off], thresholds="1000")
    assert code == 0, err
    good_rows = [line for line in out.splitlines() if line.startswith(f"RESULT {good} ")]
    assert good_rows, out
    assert [line for line in out.splitlines() if line.startswith(f"RESULT {near} ")]
    assert not [line for line in out.splitlines() if line.startswith(f"RESULT {off} ")]
    assert f"SKIP {off} reason=assistant rows {off_calls} vs api_call_count {off_calls + 2}" in out
    # the two sessions carry their own numbers: one line per model and threshold for the good one only
    assert len([line for line in good_rows if f" {MODEL_DEFAULT} 1000 " in line]) == 1
    assert len([line for line in good_rows if f" {MODEL_CTXPKG} 1000 " in line]) == 1


def test_default_model_compacts_at_threshold():
    """``>=`` at the boundary: a prompt exactly at the threshold compacts, one token above does not."""
    rows = _transcript(turns=8, chars=600)
    session = _session_of(rows)

    # The first call whose span (head to protected tail) is non-empty is the boundary worth pinning.
    running = 0
    boundary_index = boundary_prompt = None
    for index, message in enumerate(session.messages):
        if message.role == "assistant" and index - PROTECT_FIRST_N - TAIL_MESSAGES > 0:
            boundary_index, boundary_prompt = index, running
            break
        running += message.tokens
    assert boundary_index is not None and boundary_prompt
    # calls are numbered by assistant rows, not by message position
    call_index = sum(1 for message in session.messages[:boundary_index] if message.role == "assistant")

    at_threshold = replay(session, 0, boundary_prompt, MODEL_DEFAULT, _rates())
    above = replay(session, 0, boundary_prompt + 1, MODEL_DEFAULT, _rates())
    assert at_threshold.calls[call_index].after_compaction is True
    assert at_threshold.compactions >= 1
    assert above.calls[call_index].after_compaction is False
    # one token higher moves the first compaction strictly later, it does not suppress every compaction
    assert next(index for index, call in enumerate(above.calls) if call.after_compaction) > call_index
    # and the threshold really is in prompt units: one token less than the prompt keeps firing
    below = replay(session, 0, boundary_prompt - 1, MODEL_DEFAULT, _rates())
    assert below.calls[call_index].after_compaction is True

    # after a compaction the prompt is the head, the summary of the measured size and the last 20 messages
    first_after = at_threshold.calls[call_index]
    history = session.messages[:boundary_index]
    expected = (sum(message.tokens for message in session.messages[:PROTECT_FIRST_N])
                + session.summary_tokens + sum(message.tokens for message in history[-TAIL_MESSAGES:]))
    assert first_after.prompt_tokens == expected
    assert first_after.read_tokens == 0  # add-on is 0 here, so the whole prompt after it is a write
    assert first_after.write_tokens == first_after.prompt_tokens

    # with a real add-on the same call reads it and writes the rest, and every later call writes only what
    # is new: "the first call after a compaction writes everything after the system prompt".
    priced = replay(session, 1000, boundary_prompt, MODEL_DEFAULT, _rates())
    first_priced = priced.calls[call_index]
    assert first_priced.read_tokens == 1000
    assert first_priced.write_tokens == first_priced.prompt_tokens - 1000

    # a run that never compacts: the session's first call is cold (all write), and each later call writes
    # only the messages appended since the previous one and reads the rest.
    quiet = replay(session, 1000, 10**9, MODEL_DEFAULT, _rates())
    assert quiet.compactions == 0
    assert quiet.calls[0].read_tokens == 0 and quiet.calls[0].write_tokens == quiet.calls[0].prompt_tokens
    positions = [index for index, message in enumerate(session.messages) if message.role == "assistant"]
    appended = sum(message.tokens for message in session.messages[positions[0]:positions[1]])
    assert quiet.calls[1].write_tokens == appended
    assert quiet.calls[1].read_tokens == quiet.calls[1].prompt_tokens - appended
    assert quiet.calls[1].read_tokens == (1000 + sum(message.tokens for message in session.messages[:positions[0]]))


def test_ctxpkg_reads_only_prior_blocks():
    """On the first call after the second compaction, block 1 is a read and block 2 is a write."""
    rows = _transcript(turns=14, chars=900)
    session = _session_of(rows, summary_tokens=700)
    rates = _rates()

    ctxpkg = replay(session, 0, 4000, MODEL_CTXPKG, rates)
    default = replay(session, 0, 4000, MODEL_DEFAULT, rates)
    ctx_compactions = [call for call in ctxpkg.calls if call.after_compaction]
    assert len(ctx_compactions) >= 2, f"expected two ctxpkg compactions, got {len(ctx_compactions)}"
    assert len(ctxpkg.blocks) >= 2 and all(size > 0 for size in ctxpkg.blocks[:2])

    second = ctx_compactions[1]
    # block 1 is a read and block 2 a write: the split is exactly add-on (0) + head + block 1 on the read
    # side, and the rest of the prompt — the new block, the state note and the tail — on the write side.
    head_tokens = sum(message.tokens for message in session.messages[:PROTECT_FIRST_N])
    expected_read = head_tokens + ctxpkg.blocks[0]
    assert second.cached_blocks == 1
    assert second.wrote_block is True
    assert second.read_tokens == expected_read
    assert second.write_tokens == second.prompt_tokens - expected_read
    assert second.write_tokens >= ctxpkg.blocks[1]

    default_compactions = [call for call in default.calls if call.after_compaction]
    assert len(default_compactions) >= 2
    assert default_compactions[1].cached_blocks == 0
    assert default_compactions[1].wrote_block is False
    # the default model rewrites the head and the summary instead of reading the prefix
    assert default_compactions[1].write_tokens > second.write_tokens


def test_flash_cost_added_per_compaction():
    """Every ctxpkg compaction pays one flash call: the span in, forty tokens out per episode."""
    rows = _transcript(turns=10, chars=800)
    session = _session_of(rows, summary_tokens=500)

    ctxpkg = replay(session, 0, 3000, MODEL_CTXPKG, _rates())
    assert ctxpkg.compactions >= 1
    assert ctxpkg.flash_input_tokens > 0 and ctxpkg.flash_episodes > 0
    expected = (Decimal(ctxpkg.flash_input_tokens) * replay_module.FLASH_IN_USD_PER_MTOK
                + Decimal(ctxpkg.flash_episodes * replay_module.EPISODE_OUTPUT_TOKENS)
                * replay_module.FLASH_OUT_USD_PER_MTOK) / Decimal(1_000_000)
    assert ctxpkg.flash_usd == expected
    assert ctxpkg.flash_usd > 0
    assert ctxpkg.flash_episodes >= ctxpkg.compactions  # at least one episode per compaction

    total = ctxpkg.total_usd_for(1000)
    parts = ctxpkg.read_usd + ctxpkg.write_usd + ctxpkg.output_usd_for(1000)
    assert total - parts == ctxpkg.flash_usd

    # no compaction, no flash cost; and the default model never calls flash at all
    quiet = replay(session, 0, 10_000_000, MODEL_CTXPKG, _rates())
    assert quiet.compactions == 0 and quiet.flash_usd == 0
    default = replay(session, 0, 3000, MODEL_DEFAULT, _rates())
    assert default.compactions >= 1
    assert default.flash_usd == 0 and default.flash_input_tokens == 0


def test_calibration_ratio_line(tmp_path, capsys):
    """A session whose recorded columns came from the replay calibrates at ratio 1 on every field."""
    path, conn = _db(tmp_path)
    ids = ["20260101_120000_bbb00001", "20260101_120000_bbb00002"]
    sizes = [(4, 400), (6, 1200)]  # distinct sizes, so a mixed-up session row cannot pass
    for session_id, (turns, chars) in zip(ids, sizes):
        calls = _insert(conn, session_id, _transcript(turns=turns, chars=chars))
        _session_row(conn, session_id, calls=calls, output=250, cost=0.0)
    conn.commit()

    # Fill each session's recorded token columns from its own default-model replay, so the calibration is
    # self-consistent and any wrong pairing or wrong bucket moves a ratio off 1.
    for session_id in ids:
        session = load_session(conn, session_id)
        result = replay(session, 0, CALIBRATION_THRESHOLD, MODEL_DEFAULT, _rates())
        conn.execute(
            "UPDATE sessions SET input_tokens = 0, cache_read_tokens = ?, cache_write_tokens = ?, "
            "output_tokens = 250, estimated_cost_usd = ? WHERE id = ?",
            (result.read_tokens, result.write_tokens, float(result.total_usd_for(250)), session_id),
        )
    conn.commit()
    conn.close()

    code, out, err = _run_main(capsys, path, ids, thresholds="1000,300000")
    assert code == 0, err
    lines = [line for line in out.splitlines() if line.startswith("CALIBRATION ")]
    assert len(lines) == 2
    pattern = re.compile(
        r"^CALIBRATION (?P<sid>\S+) replay_default_total\$=(?P<total>[0-9.]+) recorded\$=(?P<recorded>[0-9.]+) "
        r"ratio=(?P<ratio>[0-9.]+) read_ratio=(?P<read>[0-9.]+) write_ratio=(?P<write>[0-9.]+)$")
    seen = {}
    for line in lines:
        match = pattern.match(line)
        assert match, line
        assert match.group("sid") in ids
        assert Decimal(match.group("ratio")) == Decimal("1.000")
        assert Decimal(match.group("read")) == Decimal("1.000")
        assert Decimal(match.group("write")) == Decimal("1.000")
        seen[match.group("sid")] = Decimal(match.group("total"))
    assert seen[ids[0]] != seen[ids[1]], "the two sessions recorded the same total: the fixture is degenerate"

    # The add-on is *solved*, not assumed: a recording whose mean prompt sits above what the rows account
    # for gets a positive add-on, and the default replay lands back on the recorded mean.
    conn = sqlite3.connect(path)
    conn.execute("UPDATE sessions SET cache_read_tokens = cache_read_tokens + 40000 WHERE id = ?", (ids[0],))
    conn.commit()
    conn.close()
    conn = open_read_only(path)
    try:
        session = load_session(conn, ids[0])
    finally:
        conn.close()
    add_on, calibrated = replay_module.calibration_add_on(session, _rates())
    assert add_on > 0
    assert abs(calibrated.mean_prompt - session.recorded.mean_prompt) < 2


def test_decision_ship_and_hold():
    """The gate's four bands, each bound tested from both sides."""
    def calibration(ratio, read_ratio="1.0", write_ratio="1.0"):
        return Calibration(
            session_id="20260101_120000_ccc00001", replay_total_usd=Decimal(10),
            recorded_usd=Decimal(10), ratio=Decimal(ratio), read_ratio=Decimal(read_ratio),
            write_ratio=Decimal(write_ratio),
        )

    default_total = Decimal(100)
    at_margin = {1000: Decimal(90)}
    inside = {1000: Decimal(50)}

    # ship: every ratio on its boundary, and the margin exactly at 0.9x
    decision, reason = decide([calibration("0.85", "0.7", "1.3")], at_margin, default_total)
    assert decision == "ship" and reason == ""
    decision, _ = decide([calibration("1.15", "1.3", "0.7")], at_margin, default_total)
    assert decision == "ship"

    # hold by the total ratio alone, just outside each end
    decision, reason = decide([calibration("0.8499")], inside, default_total)
    assert decision == "hold" and "total ratio" in reason
    decision, reason = decide([calibration("1.1501")], inside, default_total)
    assert decision == "hold" and "total ratio" in reason

    # hold by a component ratio alone, with the total inside the band
    decision, reason = decide([calibration("1.0", "0.699", "1.0")], inside, default_total)
    assert decision == "hold" and "read_ratio" in reason and "total ratio" not in reason
    decision, reason = decide([calibration("1.0", "1.0", "1.3001")], inside, default_total)
    assert decision == "hold" and "write_ratio" in reason and "total ratio" not in reason
    decision, reason = decide([calibration("1.0", "1.301", "1.0")], inside, default_total)
    assert decision == "hold" and "read_ratio" in reason

    # hold by the margin alone, with every ratio inside its band
    decision, reason = decide([calibration("1.0")], {1000: Decimal("90.01")}, default_total)
    assert decision == "hold" and "above 0.9" in reason
    decision, _ = decide([calibration("1.0")], {1000: Decimal(90)}, default_total)
    assert decision == "ship"

    # one failing session is enough, and the reason names it
    other = Calibration(session_id="20260101_120000_ccc00002", replay_total_usd=Decimal(10),
                        recorded_usd=Decimal(10), ratio=Decimal("1.0"), read_ratio=Decimal("1.0"),
                        write_ratio=Decimal("2.0"))
    decision, reason = decide([calibration("1.0"), other], inside, default_total)
    assert decision == "hold" and "ccc00002" in reason

    # the chosen threshold is the cheapest one, and ties go to the lower threshold
    assert choose_threshold({300000: Decimal(20), 120000: Decimal(10), 160000: Decimal(30)}) == 120000
    assert choose_threshold({300000: Decimal(10), 120000: Decimal(10)}) == 120000
    decision, reason = decide([calibration("1.0")], {}, default_total)
    assert decision == "hold" and "no session" in reason


def test_output_has_no_message_text(tmp_path, capsys):
    """A marker in every content and every tool call never reaches stdout or stderr."""
    path, conn = _db(tmp_path)
    session_id = "20260101_120000_ddd00001"
    rows = [(role, (text + " " + MARKER) if text else text, calls) for role, text, calls in _transcript(turns=6)]
    marked_calls = []
    for role, text, calls in rows:
        if calls:
            calls = [{"id": call["id"], "type": "function",
                      "function": {"name": call["function"]["name"],
                                   "arguments": call["function"]["arguments"] + f" {MARKER}"}} for call in calls]
        marked_calls.append((role, text, calls))
    calls = _insert(conn, session_id, marked_calls)
    _session_row(conn, session_id, calls=calls, read=2000, write=200, output=80, cost=0.2)
    conn.commit()
    conn.close()

    code, out, err = _run_main(capsys, path, [session_id], thresholds="1000,200000")
    assert code == 0, err
    assert "RESULT" in out
    assert MARKER not in out
    assert MARKER not in err
    assert MARKER.lower() not in out.lower()
    # the rows carry sizes and costs only: no field may hold a slice of the transcript
    assert "cmd " not in out and "result " not in out
