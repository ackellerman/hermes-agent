"""Edge pins for the ctxpkg replay (reviews of step 5, rounds 1 and 2).

The pinned file ``tests/plugins/test_ctxpkg_replay.py`` keeps the plan's nine names. Every rule the reviews
found unguarded is pinned here instead: each case holds values that differ from the case beside it, so a
mutant that drops a term, prices a bucket at the wrong rate, reads the margin at the wrong threshold or sums
one session instead of all of them moves a number this file asserts.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import plugins.context_engine.ctxpkg.replay as replay_module
from agent.model_metadata import estimate_tokens_rough
from plugins.context_engine.ctxpkg.compaction import block_ack, block_line, block_text
from plugins.context_engine.ctxpkg.episodes import extract_ids, render, segment
from plugins.context_engine.ctxpkg.replay import (
    CALIBRATION_THRESHOLD,
    MODEL_CTXPKG,
    MODEL_DEFAULT,
    PROTECT_FIRST_N,
    TAIL_MESSAGES,
    Calibration,
    Recorded,
    calibration_add_on,
    decide,
    replay,
)
from tests.plugins.test_ctxpkg_replay import (
    MODEL,
    PROVIDER,
    _db,
    _insert,
    _rates,
    _run_main,
    _session_of,
    _session_row,
    _transcript,
)


def _one_compaction(model: str):
    session = _session_of(_transcript(40, chars=200), summary_tokens=10)
    unbounded = replay(session, 0, 10**9, model, _rates())
    threshold = unbounded.calls[int(len(unbounded.calls) * 0.8)].prompt_tokens
    result = replay(session, 0, threshold, model, _rates())
    return session, result


def test_only_the_compacting_call_rewrites_the_prefix_default():
    """Plan: the first call after a compaction writes everything; each later call writes only its new messages."""
    _, result = _one_compaction(MODEL_DEFAULT)
    assert result.compactions == 1
    index = next(i for i, call in enumerate(result.calls) if call.after_compaction)
    follow = result.calls[index + 1]
    assert follow.write_tokens < follow.prompt_tokens // 2, (follow, result.calls[index])


def test_only_the_compacting_call_rewrites_the_prefix_ctxpkg():
    _, result = _one_compaction(MODEL_CTXPKG)
    assert result.compactions == 1
    index = next(i for i, call in enumerate(result.calls) if call.after_compaction)
    follow = result.calls[index + 1]
    assert follow.write_tokens < follow.prompt_tokens // 2, (follow, result.calls[index])


def test_every_call_after_a_compaction_writes_only_its_new_rows():
    """The compaction's own call is the only full write; the calls after it are appends, in both models."""
    for model in (MODEL_DEFAULT, MODEL_CTXPKG):
        session, result = _one_compaction(model)
        index = next(i for i, call in enumerate(result.calls) if call.after_compaction)
        positions = [k for k, message in enumerate(session.messages) if message.is_call]
        for offset in (1, 2, 3):
            call = result.calls[index + offset]
            appended = sum(message.tokens
                           for message in session.messages[positions[index + offset - 1]:positions[index + offset]])
            assert call.write_tokens == appended, (model, offset, call, appended)
            assert call.read_tokens == call.prompt_tokens - appended


def test_flash_output_is_forty_tokens_per_episode():
    result = replay_module.ReplayResult(
        session_id="s", model=MODEL_CTXPKG, threshold=1, rates=_rates(),
        flash_in_rate=Decimal(0), flash_out_rate=Decimal(1_000_000))
    assert result.flash_cost(0, 3) == Decimal(120)


def test_writes_are_priced_at_the_write_rate():
    rates = _rates()
    result = replay_module.ReplayResult(session_id="s", model=MODEL_DEFAULT, threshold=1, rates=rates)
    result.calls.append(replay_module.CallRecord(index=0, prompt_tokens=1_000_000, read_tokens=0,
                                                 write_tokens=1_000_000))
    assert result.write_usd == rates.cache_write
    assert result.read_usd == 0
    assert rates.cache_write != rates.cache_read


def test_default_summary_size_is_twelve_thousand_without_summary_rows(tmp_path):
    path, conn = _db(tmp_path)
    assistants = _insert(conn, "s1", _transcript(2))
    _session_row(conn, "s1", calls=assistants)
    conn.commit()
    conn.close()
    ro = replay_module.open_read_only(path)
    assert replay_module.load_session(ro, "s1").summary_tokens == 12_000


def test_state_note_counts_in_the_post_compaction_write():
    """The ctxpkg post-compaction prompt carries the state note, at the engine's own 1500-char cap."""
    session, result = _one_compaction(MODEL_CTXPKG)
    assert result.compactions == 1
    index = next(i for i, call in enumerate(result.calls) if call.after_compaction)
    call = result.calls[index]
    positions = [k for k, message in enumerate(session.messages) if message.is_call]
    head = sum(message.tokens for message in session.messages[:PROTECT_FIRST_N])
    tail = sum(message.tokens
               for message in session.messages[positions[index] - TAIL_MESSAGES:positions[index]])
    refs = [f"ep-{number:04d}" for number in range(1, result.flash_episodes + 1)]
    state = replay_module._state_note_tokens(refs)
    assert state > 0
    assert call.read_tokens == head          # add-on 0 here: the head is the only cached prefix
    assert call.prompt_tokens == head + result.blocks[0] + state + tail


def test_block_bytes_include_the_ack():
    """A block's size in the replay is the engine's own bytes: the lines, the header and the ack."""
    span = [{"role": "user", "content": "u" * 400}, {"role": "assistant", "content": "answer one"}]
    episodes = segment(span, first_turn=1)
    assert len(episodes) == 1
    text = render(episodes[0])
    line = block_line("ep-0001", 1, 1, len(text), "x" * replay_module.FIXED_SUMMARY_CHARS, extract_ids(text))
    body = block_text("ep-0001", "ep-0001", [line])
    assert len(block_ack("ep-0001", "ep-0001")) > 40
    size, count, next_ref = replay_module._block_tokens(span, 1, 1)
    assert (count, next_ref) == (1, 2)
    assert size == estimate_tokens_rough(body + "\n" + block_ack("ep-0001", "ep-0001"))
    assert size > estimate_tokens_rough(body)


def test_margin_is_read_at_the_chosen_threshold():
    """The 0.9x margin is applied to the CHOSEN threshold's total, not to the first or the dearest one."""
    def calibration():
        return Calibration(
            session_id="20260101_120000_eee00001", replay_total_usd=Decimal(10), recorded_usd=Decimal(10),
            ratio=Decimal("1.0"), read_ratio=Decimal("1.0"), write_ratio=Decimal("1.0"),
        )

    default_total = Decimal(100)
    # 80000 is above the margin, 120000 is cheaper and therefore chosen: the cheap one decides.
    decision, reason = decide([calibration()], {80000: Decimal(95), 120000: Decimal(89)}, default_total)
    assert decision == "ship" and reason == ""
    # and the reason names the chosen one, never the other threshold
    decision, reason = decide([calibration()], {80000: Decimal(95), 120000: Decimal(91)}, default_total)
    assert decision == "hold"
    assert "at threshold 120000" in reason and "threshold 80000" not in reason


def test_default_total_is_summed_over_all_sessions(tmp_path, capsys):
    """The margin reads the summed default total, not one session's (the two fixtures differ in size)."""
    path, conn = _db(tmp_path)
    ids = ["20260101_120000_ggg00001", "20260101_120000_ggg00002"]
    sizes = [(4, 500), (7, 1500)]  # distinct sizes, so a kept-only-one total cannot match the sum
    for session_id, (turns, chars) in zip(ids, sizes):
        calls = _insert(conn, session_id, _transcript(turns=turns, chars=chars))
        _session_row(conn, session_id, calls=calls, output=250, cost=0.0)
    conn.commit()
    for session_id in ids:
        session = replay_module.load_session(conn, session_id)
        result = replay(session, 0, CALIBRATION_THRESHOLD, MODEL_DEFAULT, _rates())
        conn.execute(
            "UPDATE sessions SET cache_read_tokens = ?, cache_write_tokens = ?, output_tokens = 250, "
            "estimated_cost_usd = ? WHERE id = ?",
            (result.read_tokens, result.write_tokens, float(result.total_usd_for(250)), session_id),
        )
    conn.commit()
    conn.close()

    # One threshold, above the fixture's whole prompt: neither model compacts, so both cost the same and
    # only the margin can fail. Its reason is the one place the summed default total is printed.
    code, out, err = _run_main(capsys, path, ids, thresholds=str(CALIBRATION_THRESHOLD))
    assert code == 0, err
    per_session = [Decimal(line.split("replay_default_total$=")[1].split(" ")[0])
                   for line in out.splitlines() if line.startswith("CALIBRATION ")]
    assert len(per_session) == 2 and per_session[0] != per_session[1]
    summed = sum(per_session, Decimal(0))
    decision = [line for line in out.splitlines() if line.startswith("DECISION=")]
    assert len(decision) == 1 and decision[0].startswith("DECISION=hold")
    assert f"the default's ${summed:.4f}" in decision[0]
    assert f"the default's ${per_session[0]:.4f}" not in decision[0]
    assert f"the default's ${per_session[1]:.4f}" not in decision[0]


def test_unpriced_model_is_skipped_not_priced_at_zero(tmp_path, capsys):
    """A session whose model has no price is refused with a SKIP line, never priced at $0."""
    from agent.usage_pricing import get_pricing_entry

    unpriced = "no-such-model-s5r2"
    assert get_pricing_entry(unpriced, provider=PROVIDER) is None

    path, conn = _db(tmp_path)
    session_id = "20260101_120000_fff00001"
    calls = _insert(conn, session_id, _transcript(turns=3, chars=600))
    _session_row(conn, session_id, calls=calls, read=1000, write=100, output=50, cost=0.5, model=unpriced)
    conn.commit()
    conn.close()

    code, out, err = _run_main(capsys, path, [session_id])
    assert code == 0, err
    assert f"SKIP {session_id} reason=no price for model '{unpriced}'" in out
    assert not [line for line in out.splitlines() if line.startswith(f"RESULT {session_id} ")]
    assert "DECISION=hold reason=no session could be replayed" in out


def test_empty_assistant_row_is_not_a_call():
    """An assistant row with no content and no tool calls joins the prompt but opens no call."""
    rows = _transcript(turns=4, chars=300)
    with_placeholder = rows[:5] + [("assistant", "", None)] + rows[5:]
    plain = replay(_session_of(rows), 0, 10**9, MODEL_DEFAULT, _rates())
    placeheld = replay(_session_of(with_placeholder), 0, 10**9, MODEL_DEFAULT, _rates())
    real_assistants = sum(1 for role, _, _ in rows if role == "assistant")
    assert plain.call_count == real_assistants
    assert sum(1 for role, _, _ in with_placeholder if role == "assistant") == real_assistants + 1
    assert placeheld.call_count == real_assistants
    assert [call.prompt_tokens for call in placeheld.calls] == [call.prompt_tokens for call in plain.calls]


def test_add_on_is_the_subtraction_not_a_fixed_point():
    """The add-on is the recorded mean minus the message mean, floored at 0 — never solved to match it."""
    session = _session_of(_transcript(turns=60, chars=2000), summary_tokens=800)
    zero = replay(session, 0, CALIBRATION_THRESHOLD, MODEL_DEFAULT, _rates())
    recorded_mean = 320_000
    session = replace(session, recorded=Recorded(
        session_id=session.session_id, model=MODEL, provider=PROVIDER, calls=zero.call_count,
        input_tokens=0, cache_read_tokens=int(recorded_mean * zero.call_count), cache_write_tokens=0,
        output_tokens=0, cost_usd=Decimal(0)))

    add_on, calibrated = calibration_add_on(session, _rates())
    assert add_on == int(session.recorded.mean_prompt - zero.mean_prompt)
    assert add_on > 0
    # the add-on is large enough to push the replay over the calibration threshold, where it compacts...
    assert calibrated.compactions > 0
    # ...so its mean is NOT pinned to the recording, which is the difference from a solved fixed point
    assert calibrated.mean_prompt < session.recorded.mean_prompt - 1000


def test_add_on_floors_at_zero():
    """A transcript the recording does not cover gets an add-on of 0, not a negative one."""
    session = _session_of(_transcript(turns=6, chars=800))
    zero = replay(session, 0, CALIBRATION_THRESHOLD, MODEL_DEFAULT, _rates())
    carried = replace(session.recorded, calls=zero.call_count, input_tokens=1, cache_read_tokens=0,
                      cache_write_tokens=0, output_tokens=0)
    add_on, calibrated = calibration_add_on(replace(session, recorded=carried), _rates())
    assert add_on == 0
    assert calibrated.mean_prompt == zero.mean_prompt
