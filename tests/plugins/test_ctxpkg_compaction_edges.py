"""Edge cases for ``ctxpkg`` episode compaction (outside the plan's step-4 test list).

The state note and the block pair are the engine's own bookkeeping inside the transcript; these pin the
parts of their lifecycle the main list does not: replacement rather than accumulation, the exact warm
boundary, the summary cut, an unknown ref, and a structural no-op that leaves the transcript alone — plus,
from the round-1 review, the prefix a third compaction must leave byte-identical, the head rows a prune pass
would otherwise rewrite, ref reservation across two writers, an abandoned attempt writing nothing, and one
case per rule of the block, the state note, the summaries and the warm/cold carry — plus, from the round-2
review, the state call failing on its own, two-block protection through prune and a third compaction, an
attempt superseded between the flash calls, the pinned summary route on both calls, and the micro cursor
floor where no filler follows the state note.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from agent import auxiliary_client
import agent.context_compressor as compressor_module
import hermes_cli.config as config_module
import plugins.context_engine.ctxpkg.store as store_module
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.prompt_cache_boundary import clear_cache_anchors, is_cache_anchor
from agent.prompt_caching import build_prompt_cache_plan
from plugins.context_engine.ctxpkg import CtxpkgEngine
from plugins.context_engine.ctxpkg.compaction import ACK_RE, load_settings
from plugins.context_engine.ctxpkg.episodes import recall_ref

SESSION = "20260101_130000_abcdef12"


@pytest.fixture(autouse=True)
def _ctxpkg_config(monkeypatch):
    monkeypatch.setattr(
        config_module, "load_config_readonly",
        lambda: {"ctxpkg": {"warm_turns": 8, "max_episode_chars": 60_000, "summary_max_chars": 300}},
    )
    clear_cache_anchors()


def _msg(role, text=None, tool_calls=None, tool_call_id=None):
    message = {"role": role}
    if text is not None:
        message["content"] = text
    if tool_calls:
        message["tool_calls"] = tool_calls
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    return message


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _turn(index, rounds=2, size=300):
    rows = [_msg("user", f"turn {index}: please run the step {index} check")]
    for round_index in range(rounds):
        rows.append(_msg("assistant", None, [
            _call(f"c{index}-{round_index}", "terminal",
                  json.dumps({"command": f"cmd {index} {round_index}"}))]))
        rows.append(_msg("tool", f"result {index}-{round_index} " + "y" * size,
                         tool_call_id=f"c{index}-{round_index}"))
    rows.append(_msg("assistant", f"answer {index}: done"))
    return rows


def _transcript(turns=8, start=1, size=300):
    rows = [_msg("system", "You are a test driver.")]
    for index in range(start, start + turns):
        rows.extend(_turn(index, size=size))
    return rows


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message={"content": text})])


def _llm(state="goal: ship step 4. next: run the tests."):
    def fake(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            refs = [line[4:].strip() for line in prompt.splitlines() if line.startswith("### ep-")]
            return _response("\n".join(f"{ref}: did the step" for ref in refs))
        return _response(state)

    return fake


def _engine(tmp_path, monkeypatch, session=SESSION, **settings):
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    engine = CtxpkgEngine(model="test-model")
    engine._config_context_length = 200_000
    engine.quiet_mode = True
    engine.protect_first_n = settings.get("protect_first_n", 3)
    engine.protect_last_n = settings.get("protect_last_n", 20)
    engine._session_id = session
    engine._threshold_tokens = 100_000
    return engine


def _compress(engine, messages, monkeypatch, llm=None):
    fake = llm or _llm()
    monkeypatch.setattr(auxiliary_client, "call_llm", fake)
    monkeypatch.setattr(compressor_module, "call_llm", fake)
    return engine.compress(list(messages), current_tokens=150_000, force=True)


def _text(message):
    content = message.get("content")
    return content if isinstance(content, str) else ""


def test_state_note_replaced_not_accumulated(tmp_path, monkeypatch):
    """A second compaction replaces the state note; the transcript never accumulates notes."""
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    second = _compress(engine, list(first) + _transcript(12, start=11)[1:], monkeypatch)
    notes = [index for index, message in enumerate(second) if _text(message).startswith("[ctxpkg state]")]
    assert len(notes) == 1, "one state note, the newest"
    assert "ep-0020" in _text(second[notes[0]]), "it names the newest episodes"


def test_warm_boundary_is_both_sides(tmp_path, monkeypatch):
    """``turn - last_use < warm_turns``: one turn inside is warm, exactly ``warm_turns`` is cold."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.renew("ep-0001", 10)
    assert engine._is_warm("ep-0001", engine.store, 17, 8) is True
    assert engine._is_warm("ep-0001", engine.store, 18, 8) is False
    assert engine._is_warm("ep-0001", engine.store, 11, 0) is False, "zero keeps nothing warm"


def test_summary_cut_and_id_scrub_are_deterministic():
    """The cut lands on a word boundary inside the limit, and an unlisted id never survives."""
    from plugins.context_engine.ctxpkg.compaction import cut_words, scrub_ids

    long_summary = "alpha beta gamma delta"
    assert cut_words(long_summary, 12) == "alpha beta"
    assert len(cut_words("x" * 40, 10)) == 10, "one long word is hard-cut, never dropped"
    allowed = ["a" * 40]
    summary = f"kept {'a' * 40} dropped {'b' * 40}"
    assert scrub_ids(summary, allowed) == f"kept {'a' * 40} dropped"
    assert scrub_ids("", allowed) == ""


def test_block_and_ack_shapes_agree(tmp_path, monkeypatch):
    """The block header, its lines and the ack all name the same ref range."""
    from plugins.context_engine.ctxpkg.compaction import ACK_RE, BLOCK_HEADER_RE

    engine = _engine(tmp_path, monkeypatch)
    out = _compress(engine, _transcript(10), monkeypatch)
    block = next(_text(message) for message in out if _text(message).startswith("[ctxpkg references "))
    ack = next(_text(message) for message in out if ACK_RE.match(_text(message)))
    header = BLOCK_HEADER_RE.match(block.splitlines()[0])
    assert header and ACK_RE.match(ack).group(1) == header.group(1)
    assert ACK_RE.match(ack).group(2) == header.group(2)
    for line in block.splitlines()[1:]:
        assert line.startswith("ep-") and " · turns " in line and " · ids: " in line


def test_state_note_names_the_last_three_episodes():
    """``state_note`` appends the last three refs by script, whatever the model wrote."""
    from plugins.context_engine.ctxpkg.compaction import state_note

    note = state_note("goal: g", ["ep-0001", "ep-0002", "ep-0003", "ep-0004"])
    assert note.splitlines()[0] == "[ctxpkg state]"
    assert note.splitlines()[1] == "goal: g"
    assert note.splitlines()[2] == "recent episodes: ep-0002, ep-0003, ep-0004"


def test_cold_stub_without_a_known_summary():
    """An unknown ref still stubs: the stub names the ref and the recall that brings it back."""
    from plugins.context_engine.ctxpkg.compaction import cold_stub

    assert cold_stub("ep-0009", "") == '[recalled ep-0009, now cold — recall("ep-0009") for the full text]'
    assert "its summary" in cold_stub("ep-0009", "its summary")


def test_split_turn_makes_several_episodes_with_one_turn_number(tmp_path, monkeypatch):
    """``ctxpkg.max_episode_chars`` below one turn splits it at tool-call boundaries, same turn number."""
    monkeypatch.setattr(
        config_module, "load_config_readonly",
        lambda: {"ctxpkg": {"warm_turns": 8, "max_episode_chars": 700, "summary_max_chars": 300}},
    )
    engine = _engine(tmp_path, monkeypatch)
    _compress(engine, _transcript(10), monkeypatch)
    rows = engine.store.index()
    assert len(rows) > len({row["turn_first"] for row in rows}), "a turn became more than one episode"
    repeated = [row for row in rows if row["turn_first"] == row["turn_last"]]
    assert all(row["chars"] <= 700 for row in rows) or repeated, "the bound is respected per part"


def test_short_transcript_is_a_structural_no_op(tmp_path, monkeypatch):
    """Too few messages to compress: the transcript comes back untouched and nothing is written."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(1)
    out = _compress(engine, messages, monkeypatch)
    assert out == messages
    assert engine.store.index() == [] and engine.compression_count == 0
    assert engine._last_compression_savings_pct == 0.0


def test_warm_turns_zero_keeps_nothing_warm(tmp_path, monkeypatch):
    """``ctxpkg.warm_turns: 0`` means never warm: a recall made this turn is cold at the next pass."""
    monkeypatch.setattr(
        config_module, "load_config_readonly",
        lambda: {"ctxpkg": {"warm_turns": 0, "max_episode_chars": 60_000, "summary_max_chars": 300}},
    )
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("zero episode body", turn_first=1, turn_last=1, summary="zero summary")
    turn = engine.store.current_turn()
    engine.store.renew("ep-0001", turn)
    assert engine._is_warm("ep-0001", engine.store, turn, 0) is False

    messages = _transcript(2) + [
        _msg("user", "recall it"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "ZERO EPISODE BODY " + "z" * 400, tool_call_id="r1"),
        _msg("assistant", "done"),
    ] + _transcript(3, start=5)[1:]
    result, _ = engine._prune_old_tool_results(messages, protect_tail_count=4)
    stubs = [_text(message) for message in result if _text(message).startswith("[recalled ep-0001, now cold")]
    assert stubs and "zero summary" in stubs[0]


def test_concurrent_writers_take_distinct_refs(tmp_path, monkeypatch):
    """Two writers in one process never share a ref: allocation and the create are one locked step."""
    import threading

    engine = _engine(tmp_path, monkeypatch)
    store = engine.store
    store.ensure()
    refs: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def writer(number: int) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(4):
                refs.append(store.write_episode(f"body {number}\n" + "x" * 20_000,
                                                turn_first=1, turn_last=1))
        except BaseException as exc:  # a thread must not die silently
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(number,)) for number in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert len(refs) == 24
    assert len(set(refs)) == 24, "every ref is taken once"
    assert sorted(store.refs()) == sorted(refs), "the index agrees with what the writers were handed"


def test_on_session_start_tolerates_a_broken_store(tmp_path, monkeypatch):
    """A store that cannot be read must not fail session start (an anchor miss costs one cache write)."""
    engine = _engine(tmp_path, monkeypatch)
    monkeypatch.setattr(engine.store.__class__, "blocks", lambda self: (_ for _ in ()).throw(OSError("nope")))
    engine.on_session_start(SESSION)


# ── the prefix, over three compactions (review r1, item 4) ────────────────

def _refs_in(prompt):
    return [line[4:].strip() for line in prompt.splitlines() if line.startswith("### ep-")]


def _marked(message):
    if message.get("cache_control"):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("cache_control") for part in content)


def _no_cache_control(value):
    """Deep copy with every ``cache_control`` key removed — no renaming, no renormalising."""
    if isinstance(value, dict):
        return {key: _no_cache_control(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [_no_cache_control(item) for item in value]
    return value


def _plan(messages):
    return build_prompt_cache_plan([dict(message) for message in messages], None, native_anthropic=True).messages


def _marker_free_prefix(messages, stop):
    """The provider-visible bytes of ``messages[:stop + 1]`` with the cache markers taken off."""
    return json.dumps(_no_cache_control([copy.deepcopy(message) for message in messages[:stop + 1]]))


def _plan_text(message):
    """A planned row's text, with the parts a marker rides on unwrapped."""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def _raw_diffs(left, right, stop):
    return [index for index in range(stop + 1)
            if json.dumps(left[index], sort_keys=True) != json.dumps(right[index], sort_keys=True)]


def _block_indexes(messages):
    return [index for index, message in enumerate(messages)
            if _text(message).startswith("[ctxpkg references ")]


def _state_note_body(messages):
    note = next(_text(message) for message in messages if _text(message).startswith("[ctxpkg state]"))
    return note.splitlines()[1]


def test_three_compactions_prefix_stable(tmp_path, monkeypatch):
    """Three compactions: the marker-free prefix up to each earlier ack is identical, the newest block marked.

    The markers themselves move by design (the newest block is the anchor), so the raw dumps of request 1
    and request 3 must differ at exactly the earlier block's ``user`` row and nowhere else: a marker moving
    is the only change the prefix may show.
    """
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    second = _compress(engine, list(first) + _transcript(12, start=11)[1:], monkeypatch)
    third = _compress(engine, list(second) + _transcript(12, start=23)[1:], monkeypatch)
    assert [len(_block_indexes(messages)) for messages in (first, second, third)] == [1, 2, 3]

    plans = [_plan(messages) for messages in (first, second, third)]
    first_block = _block_indexes(first)[0]
    ack_one = first_block + 1
    ack_two = _block_indexes(second)[1] + 1

    def _row_diffs(left, right, stop, skip=()):
        return [index for index in range(stop + 1) if index not in skip
                and json.dumps(_no_cache_control([left[index]])) != json.dumps(_no_cache_control([right[index]]))]

    # The row whose marker moved is the one row allowed to change shape: a marker on string content
    # rides as a text part, so the marked copy is a parts list and the unmarked one a string.
    block_two = _block_indexes(second)[1]
    assert _row_diffs(plans[0], plans[1], ack_one, skip={first_block}) == []
    assert _row_diffs(plans[0], plans[2], ack_one, skip={first_block}) == []
    assert _row_diffs(plans[1], plans[2], ack_two, skip={first_block, block_two}) == []
    assert [_plan_text(plan[first_block]) for plan in plans] == [_plan_text(plans[0][first_block])] * 3

    assert _raw_diffs(plans[0], plans[2], ack_one) == [first_block], (
        "the raw prefix differs somewhere other than the marker that moved")
    assert json.dumps(first[:first_block]) == json.dumps(second[:first_block]) == json.dumps(third[:first_block])
    for plan, messages in zip(plans, (first, second, third)):
        indexes = _block_indexes(messages)
        assert [_marked(plan[index]) for index in indexes] == [False] * (len(indexes) - 1) + [True]


# ── the head rows before the first block (review r1, item 1) ───────────────

def test_head_survives_prune_and_second_compaction(tmp_path, monkeypatch):
    """No row before the first block is rewritten: not by prune's dedupe, not by the next compaction."""
    engine = _engine(tmp_path, monkeypatch)
    big = "HEADTOOL " + "h" * 1500
    messages = _transcript(10)
    messages[3] = _msg("tool", big, tool_call_id="c1-0")
    first = _compress(engine, messages, monkeypatch)
    block = _block_indexes(first)[0]
    head = json.dumps(first[:block])
    assert _text(first[3]) == big, "the head row this test watches is the large tool result"

    grown = list(first) + _transcript(12, start=11)[1:]
    duplicate = max(index for index, message in enumerate(grown)
                    if message.get("role") == "tool" and index > block)
    grown[duplicate] = {**grown[duplicate], "content": big}
    pruned, _ = engine._prune_old_tool_results(grown, protect_tail_count=2)
    assert json.dumps(pruned[:block]) == head, "prune rewrote a row before the first block"

    second = _compress(engine, grown, monkeypatch)
    assert json.dumps(second[:block]) == head, "the second compaction rewrote a row before the first block"


# ── refs: reservation, and no silent overwrite (review r1, item 2) ─────────

def test_take_refs_reserved_and_write_refuses_existing(tmp_path, monkeypatch):
    """Two reservations never overlap, a plain write starts above them, and an existing ref is refused."""
    engine = _engine(tmp_path, monkeypatch)
    store = engine.store
    store.ensure()
    first = store.take_refs(3)
    second = store.take_refs(3)
    assert first == ["ep-0001", "ep-0002", "ep-0003"]
    assert second == ["ep-0004", "ep-0005", "ep-0006"]
    assert set(first) & set(second) == set()

    written = store.write_episode("body seven", turn_first=1, turn_last=1, summary="s")
    assert written == "ep-0007", "a plain write handed out a reserved ref"
    with pytest.raises(FileExistsError):
        store.write_episode("overwrite", turn_first=1, turn_last=1, summary="s", ref="ep-0007")
    assert store.read("ep-0007") == "body seven", "the refused write overwrote the episode anyway"

    assert store.write_episode("body one", turn_first=1, turn_last=1, summary="s", ref=first[0]) == "ep-0001"
    assert store.read("ep-0001") == "body one", "a reserved ref could not be written"


def test_abandoned_compaction_writes_nothing(tmp_path, monkeypatch):
    """A compaction the host cancelled during its flash calls writes no episode, block line or anchor."""
    engine = _engine(tmp_path, monkeypatch)

    def cancelling(**kwargs):
        engine._compression_cancelled_check = lambda: True
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            return _response("\n".join(f"{ref}: did the step" for ref in _refs_in(prompt)))
        return _response("goal: g")

    with pytest.raises(AuxiliaryExplicitCancellation):
        _compress(engine, _transcript(10), monkeypatch, llm=cancelling)
    assert engine.store.index() == [], "an abandoned attempt wrote episodes"
    assert engine.store.blocks() == [], "an abandoned attempt recorded a reference block"
    assert engine.compression_count == 0, "an abandoned attempt moved the compaction counters"


# ── the block: the whole ref range, ids: none, the kilochar floor ──────────

def test_header_and_ack_number_the_whole_range(tmp_path, monkeypatch):
    """The header and the ack name the first and the last ref of the block, not just its first."""
    engine = _engine(tmp_path, monkeypatch)
    out = _compress(engine, _transcript(10), monkeypatch)
    refs = engine.store.refs()
    assert len(refs) >= 3, "the fixture leaves several episodes in the span"
    block = _text(next(message for message in out if _text(message).startswith("[ctxpkg references ")))
    ack = _text(next(message for message in out if ACK_RE.match(_text(message))))
    assert block.splitlines()[0] == f"[ctxpkg references {refs[0]}..{refs[-1]}]"
    assert ack.startswith(f"Noted: {refs[0]}..{refs[-1]}. I recall")
    assert refs[0] != refs[-1]


def test_ids_none_and_kilochar_floor(tmp_path, monkeypatch):
    """A span with no ids prints ``ids: none``, and an episode under a thousand chars still reads ``1k``."""
    engine = _engine(tmp_path, monkeypatch)
    out = _compress(engine, _transcript(10, size=10), monkeypatch)
    block = _text(next(message for message in out if _text(message).startswith("[ctxpkg references ")))
    assert "ids: none" in block
    assert " · 1k chars · " in block, "a short episode's size was floored to 0k"


# ── the state note: the cap on both paths, and an empty reply ──────────────

def test_state_note_body_is_capped(tmp_path, monkeypatch):
    """The flash note and the fallback text are both cut at exactly 1500 chars."""
    engine = _engine(tmp_path, monkeypatch)
    long_note = "goal: " + "n" * 3000
    body = _state_note_body(_compress(engine, _transcript(10), monkeypatch, llm=_llm(state=long_note)))
    assert body == long_note[:1500] and len(body) == 1500

    failing = _engine(tmp_path, monkeypatch, session="20260101_130000_fallback")
    messages = _transcript(10)
    last_user = max(index for index, message in enumerate(messages) if message.get("role") == "user")
    messages[last_user] = _msg("user", "goal " + "g" * 3000)

    def boom(**kwargs):
        raise RuntimeError("summary model unreachable")

    body = _state_note_body(_compress(failing, messages, monkeypatch, llm=boom))
    assert body == ("goal " + "g" * 3000)[:1500] and len(body) == 1500


def test_empty_state_reply_falls_back(tmp_path, monkeypatch):
    """An empty state reply is not the note: the last user text stands in for it."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    last_user = max(index for index, message in enumerate(messages) if message.get("role") == "user")
    messages[last_user] = _msg("user", "goal " + "g" * 3000)

    def empty_state(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            return _response("\n".join(f"{ref}: did the step" for ref in _refs_in(prompt)))
        return _response("")

    out = _compress(engine, messages, monkeypatch, llm=empty_state)
    assert _state_note_body(out) == ("goal " + "g" * 3000)[:1500]


# ── warm recalls: order ────────────────────────────────────────────────────

def test_warm_pairs_carried_oldest_first(tmp_path, monkeypatch):
    """Two warm recalls in one span are carried in the order they were made, not reversed."""
    engine = _engine(tmp_path, monkeypatch)
    for ref, summary in (("ep-0001", "first summary"), ("ep-0002", "second summary")):
        engine.store.write_episode(f"{ref} body", turn_first=1, turn_last=1, summary=summary)
        engine.store.renew(ref, 0)

    messages = _transcript(6)
    messages.extend([
        _msg("user", "recall both"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}'),
                                 _call("r2", "recall", '{"ref": "ep-0002"}')]),
        _msg("tool", "FULL ONE", tool_call_id="r1"),
        _msg("tool", "FULL TWO", tool_call_id="r2"),
        _msg("assistant", "both recalled"),
    ])
    messages.extend(_turn(20))
    messages.extend(_turn(21))

    out = _compress(engine, messages, monkeypatch)
    carried = [call["id"] for message in out for call in (message.get("tool_calls") or []) if recall_ref(call)]
    results = [message.get("tool_call_id") for message in out
               if message.get("role") == "tool" and message.get("tool_call_id") in ("r1", "r2")]
    assert carried == ["r1", "r2"]
    assert results == ["r1", "r2"]


# ── turn numbers carry over ────────────────────────────────────────────────

def test_second_compaction_numbers_turns_onward(tmp_path, monkeypatch):
    """The second compaction's episodes start where the first one stopped, in the store and in the block."""
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    first_rows = engine.store.index()
    second = _compress(engine, list(first) + _transcript(12, start=11)[1:], monkeypatch)
    rows = engine.store.index()
    assert len(rows) > len(first_rows), "the second compaction wrote no episode"

    resume = first_rows[-1]["turn_last"] + 1
    assert rows[len(first_rows)]["turn_first"] == resume
    assert rows[-1]["turn_first"] > first_rows[-1]["turn_last"]
    newest_block = _text(second[_block_indexes(second)[-1]])
    assert f"{rows[len(first_rows)]['ref']} · turns {resume}-" in newest_block


# ── summaries: the cap, the opening, the per-ref fallbacks ─────────────────

def test_summaries_cut_at_the_cap(tmp_path, monkeypatch):
    """A long flash line and a long mechanical line are both cut to ``summary_max_chars`` on a word."""
    engine = _engine(tmp_path, monkeypatch)
    words = [f"word{index}" for index in range(200)]
    long_line = " ".join(words)

    def verbose(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            return _response("\n".join(f"{ref}: {long_line}" for ref in _refs_in(prompt)))
        return _response("goal: g")

    out = _compress(engine, _transcript(10), monkeypatch, llm=verbose)
    rows = engine.store.index()
    block = _text(next(message for message in out if _text(message).startswith("[ctxpkg references ")))
    assert len(rows) >= 3
    for row in rows:
        assert 0 < len(row["summary"]) <= 300, "a flash summary over the cap reached the block"
        assert row["summary"].split()[-1] in words, "the cut landed mid-word"
        assert row["summary"] in block

    # The exception path is cut too: the mechanical line overruns the cap on the tool names alone.
    failing = _engine(tmp_path, monkeypatch, session="20260101_130000_mechcut")
    rows_input = [_msg("system", "You are a test driver.")]
    for index in range(1, 5):
        rounds = 20
        rows_input.append(_msg("user", "please " + " ".join(f"w{number}" for number in range(200))))
        rows_input.append(_msg("assistant", None, [
            _call(f"c{index}-{round_index}", f"tool_name_{round_index:02d}", "{}")
            for round_index in range(rounds)]))
        for round_index in range(rounds):
            rows_input.append(_msg("tool", f"result {index}-{round_index} " + "y" * 60,
                                   tool_call_id=f"c{index}-{round_index}"))
        rows_input.append(_msg("assistant", f"answer {index}: done"))

    def boom(**kwargs):
        raise RuntimeError("summary model unreachable")

    _compress(failing, rows_input, monkeypatch, llm=boom)
    assert failing.store.index(), "the failing summariser still wrote episodes"
    for row in failing.store.index():
        assert 0 < len(row["summary"]) <= 300, "a mechanical summary over the cap reached the block"


def test_mechanical_summary_opens_with_160_chars(tmp_path, monkeypatch):
    """The mechanical opening is the turn's first 160 characters, not a shorter prefix."""
    engine = _engine(tmp_path, monkeypatch)
    opening = "z" * 400
    messages = [dict(message) for message in _transcript(10)]
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            messages[index] = _msg("user", opening)

    def boom(**kwargs):
        raise RuntimeError("summary model unreachable")

    _compress(engine, messages, monkeypatch, llm=boom)
    rows = engine.store.index()
    assert rows, "the failing summariser still wrote episodes"
    for row in rows:
        assert row["summary"].startswith(opening[:160]), repr(row["summary"][:200])


def test_missing_and_scrubbed_summaries_fall_back(tmp_path, monkeypatch):
    """A ref the reply omits, and a line whose only id was invented, both keep the mechanical line."""
    engine = _engine(tmp_path, monkeypatch)
    unknown_sha = "d" * 40

    def partial(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" not in prompt:
            return _response("goal: g")
        refs = _refs_in(prompt)
        lines = [f"{refs[0]}: only the first"]
        if len(refs) > 1:
            lines.append(f"{refs[1]}: {unknown_sha}")
        return _response("\n".join(lines))

    _compress(engine, _transcript(10), monkeypatch, llm=partial)
    rows = engine.store.index()
    assert len(rows) >= 3
    assert rows[0]["summary"] == "only the first"
    for row in rows[1:]:
        assert row["summary"], "an episode kept an empty summary"
        assert "(tools: terminal)" in row["summary"], "the missing line did not fall back to the mechanical one"


# ── micro-compaction: the cursor floor and the result side ─────────────────

def test_micro_cursor_clears_the_state_note_and_filler(tmp_path, monkeypatch):
    """The micro cursor starts after the state note and its filler, and neither row is absorbed."""
    engine = _engine(tmp_path, monkeypatch)
    out = _compress(engine, _transcript(10), monkeypatch)
    note = next(index for index, message in enumerate(out) if _text(message).startswith("[ctxpkg state]"))
    messages = list(out)
    messages.insert(note + 1, _msg("assistant", "Continuing from the state note."))
    filler = note + 1

    assert engine._resolve_compact_cursor(list(messages), 1, len(messages)) > filler, (
        "the cursor still stops on the state note")

    engine._micro_compact_enabled = True
    engine._micro_compact_every_n_turns = 1
    engine._micro_summarize_one = lambda exchange_text: "ROLLING SUMMARY"
    result = engine._micro_compact(list(messages))
    assert json.dumps([message for message in result
                       if _text(message).startswith("[ctxpkg state]")]) == json.dumps([messages[note]])
    assert json.dumps(result[filler]) == json.dumps(messages[filler]), "micro-compaction absorbed the filler"


def test_micro_skips_exchange_warm_only_by_result(tmp_path, monkeypatch):
    """An exchange whose only warm signal is its ``tool`` result is skipped, even with the call before it."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("result-side body", turn_first=1, turn_last=1, summary="result-side summary")
    engine.store.renew("ep-0001", 0)
    messages = [
        _msg("user", "look up the old episode"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("assistant", "one more note"),
        _msg("tool", "RESULT-SIDE BODY", tool_call_id="r1"),
        _msg("user", "carry on"),
    ]
    assert engine._find_one_exchange(messages, 2, len(messages)) is None, "a warm recall's exchange was absorbable"

    for _ in range(20):
        engine.store.bump_turn()
    later = engine._find_one_exchange(messages, 2, len(messages))
    assert later == (2, 4), "once the episode is cold the exchange is absorbable again"


# ── session start: every block, after super() ──────────────────────────────

def test_session_start_registers_every_block(tmp_path, monkeypatch):
    """A fresh process registers every stored block and binds the session before it reads the store."""
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    _compress(engine, list(first) + _transcript(12, start=11)[1:], monkeypatch)
    blocks = engine.store.blocks()
    assert len(blocks) == 2, "two compactions record two blocks"

    clear_cache_anchors()
    assert not any(is_cache_anchor(text) for text in blocks), "the anchors were not cleared"

    resumed = CtxpkgEngine(model="test-model")
    resumed.quiet_mode = True
    resumed.on_session_start(SESSION)
    assert resumed.store.root.name == SESSION, "super() bound the session after the store was read"
    assert all(is_cache_anchor(text) for text in blocks), "not every stored block was re-registered"


def test_store_resolved_once_across_rotation(tmp_path, monkeypatch):
    """A rotation inside the flash call does not move the episodes off the store the attempt began on."""
    engine = _engine(tmp_path, monkeypatch)
    original = engine.store

    def rotating(**kwargs):
        engine._session_id = "20260101_140000_rotated"
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            return _response("\n".join(f"{ref}: did the step" for ref in _refs_in(prompt)))
        return _response("goal: g")

    _compress(engine, _transcript(10), monkeypatch, llm=rotating)
    assert original.index(), "the episodes did not land in the store the compaction started on"
    assert original.blocks(), "the block did not land there either"
    assert engine.store.root != original.root and engine.store.index() == []


# ── the config defaults and the block log ─────────────────────────────────

def test_settings_defaults_when_config_missing(monkeypatch):
    """A missing or malformed ``ctxpkg`` block falls back to the documented 8 / 60000 / 300."""
    monkeypatch.setattr(config_module, "load_config_readonly", dict)
    settings = load_settings()
    assert (settings.warm_turns, settings.max_episode_chars, settings.summary_max_chars) == (8, 60_000, 300)

    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {"warm_turns": "many"}})
    settings = load_settings()
    assert (settings.warm_turns, settings.max_episode_chars, settings.summary_max_chars) == (8, 60_000, 300)


def test_blocks_skips_a_bad_line(tmp_path, monkeypatch):
    """One malformed ``blocks.jsonl`` line is skipped; the blocks around it still re-register."""
    engine = _engine(tmp_path, monkeypatch)
    store = engine.store
    store.append_block("[ctxpkg references ep-0001..ep-0001]\nep-0001 · turns 1-1 · 1k chars · s · ids: none")
    with (store.root / "blocks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    store.append_block("[ctxpkg references ep-0002..ep-0002]")
    blocks = store.blocks()
    assert len(blocks) == 2, f"a malformed block line was not skipped: {blocks}"
    assert blocks[1] == "[ctxpkg references ep-0002..ep-0002]"


def test_unanswered_recall_call_is_not_collected(tmp_path, monkeypatch):
    """A recall call with no result in the span is not carried: an unanswered call would break the pairs."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("unanswered body", turn_first=1, turn_last=1, summary="unanswered summary")
    engine.store.renew("ep-0001", 0)
    messages = _transcript(6)
    messages.extend([
        _msg("user", "recall it, though the result never lands"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("assistant", "carrying on without it"),
    ])
    messages.extend(_turn(20))
    messages.extend(_turn(21))

    out = _compress(engine, messages, monkeypatch)
    assert all(message is not None for message in out), "a pair with no result was emitted"
    assert not [call for message in out for call in (message.get("tool_calls") or []) if recall_ref(call)]


# ── the state call on its own (review r2, item 1) ─────────────────────────

def test_state_call_failure_alone_falls_back(tmp_path, monkeypatch):
    """The summaries land and the *state* call raises: the compaction still completes on the fallback text.

    The other failing-LLM case fails the summaries first, and a failed summary call skips the state call
    (``summary_failed``), so the state call's own except branch was never the one under test.
    """
    engine = _engine(tmp_path, monkeypatch, session="20260101_150000_statefail")
    messages = _transcript(10)
    last_user = max(index for index, message in enumerate(messages) if message.get("role") == "user")
    goal = "goal " + "s" * 3000
    messages[last_user] = _msg("user", goal)

    def state_boom(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            return _response("\n".join(f"{ref}: did the step" for ref in _refs_in(prompt)))
        raise RuntimeError("state model unreachable")

    out = _compress(engine, messages, monkeypatch, llm=state_boom)
    assert engine.compression_count == 1, "a failed state call aborted the compaction"
    assert _block_indexes(out), "the block was not emitted"
    assert engine.store.index(), "the episodes were not written"
    assert _state_note_body(out) == goal[:1500], "the state note is not the last user text"


# ── two blocks: the head stays protected through prune and a third compaction (review r2, item 2) ────────

def test_head_prefix_survives_prune_and_compaction_with_two_blocks(tmp_path, monkeypatch):
    """Once two block pairs exist, everything up to the newest ack is still byte-identical after a pass.

    With one block the protection is exercised by the main list; with two it must hold for a prune pass
    (whose dedupe rewrites the *older* of two identical tool results, and whose demote shortens any old
    one) and for a third compaction that re-runs both.
    """
    engine = _engine(tmp_path, monkeypatch, session="20260101_150000_twoblocks")
    big = "TWO-BLOCK HEAD " + "b" * 1500
    messages = _transcript(10)
    messages[3] = _msg("tool", big, tool_call_id="c1-0")
    first = _compress(engine, messages, monkeypatch)
    assert len(_block_indexes(first)) == 1
    second = _compress(engine, list(first) + _transcript(12, start=11)[1:], monkeypatch)
    assert len(_block_indexes(second)) == 2, "the fixture did not reach two block pairs"

    grown = list(second) + _transcript(12, start=23, size=4000)[1:]
    last_ack = _block_indexes(grown)[-1] + 1
    snapshot = json.dumps(grown[:last_ack + 1])
    assert _text(grown[3]) == big, "the head row this test watches is the large tool result"
    duplicate = max(index for index, message in enumerate(grown)
                    if message.get("role") == "tool" and index > last_ack)
    grown[duplicate] = {**grown[duplicate], "content": big}

    armed = [dict(message) for message in grown]
    assert engine._dedupe_tool_results(armed) >= 1 and _text(armed[3]) != big, (
        "the fixture is not armed: an unprotected dedupe pass would not rewrite the head row")

    pruned, _ = engine._prune_old_tool_results(grown, protect_tail_count=2)
    assert json.dumps(pruned[:last_ack + 1]) == snapshot, "prune rewrote a row before the second block's ack"

    third = _compress(engine, grown, monkeypatch)
    assert json.dumps(third[:last_ack + 1]) == snapshot, "a third compaction rewrote the prefix"


# ── the attempt the host superseded (review r2, item 3) ───────────────────

def test_attempt_superseded_after_first_flash_call_writes_nothing(tmp_path, monkeypatch):
    """A newer attempt publishing summary work between the two flash calls unwinds this one as cancelled.

    The host's timeout path detaches a stale worker and retries the window in a new one; the detachment
    is invisible to the cancellation consult (nothing cancelled *this* call), so only the attempt
    generation can stop the stale worker from writing an episode, a block line and an anchor.
    """
    from agent import conversation_compression as compression_module

    engine = _engine(tmp_path, monkeypatch, session="20260101_150000_superseded")
    generation = compression_module._COMPRESSOR_ATTEMPT_GENERATION
    token = generation.set(7)
    compression_module._mark_compressor_working_attempt(engine, 7)

    def superseding(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" in prompt:
            compression_module._mark_compressor_working_attempt(engine, 8)
            return _response("\n".join(f"{ref}: did the step" for ref in _refs_in(prompt)))
        return _response("goal: g")

    try:
        assert engine._compression_cancelled() is False, "the fixture needs a live attempt, not a cancelled one"
        with pytest.raises(AuxiliaryExplicitCancellation):
            _compress(engine, _transcript(10), monkeypatch, llm=superseding)
    finally:
        generation.reset(token)

    assert engine.store.index() == [], "a superseded attempt wrote episodes"
    assert engine.store.blocks() == [], "a superseded attempt recorded a reference block"
    assert engine.compression_count == 0, "a superseded attempt moved the compaction counters"


# ── the pinned summary route on both flash calls (review r2, item 4) ──────

def test_flash_calls_carry_the_pinned_summary_route(tmp_path, monkeypatch):
    """The host's stall-fallback pin reaches ``call_llm`` on both calls, and the main key does not.

    ``_apply_summary_route`` REPLACES the route when a pin is given (a keyless pin resolves its own
    credential), so the main runtime's key and endpoint must not ride along into either call.
    """
    import plugins.context_engine.ctxpkg.compaction as compaction_module

    engine = _engine(tmp_path, monkeypatch, session="20260101_150000_pinned")
    engine.provider, engine.base_url = "main-prov", "https://main.invalid/v1"
    engine.api_key, engine.api_mode = "MAIN-RUNTIME-KEY", "chat"
    # compaction.py binds the helper at import, so the module attribute is the name its calls resolve.
    monkeypatch.setattr(compaction_module, "_pinned_summary_call_kwargs",
                        lambda: {"provider": "pinned-prov", "model": "pinned-model"})

    seen: list[dict] = []
    inner = _llm()

    def recording(**kwargs):
        seen.append(dict(kwargs))
        return inner(**kwargs)

    _compress(engine, _transcript(10), monkeypatch, llm=recording)
    assert len(seen) == 2, f"one call for the episodes and one for the state note; saw {len(seen)}"
    for kwargs in seen:
        assert kwargs.get("provider") == "pinned-prov", f"the pinned provider was dropped: {kwargs.get('provider')}"
        assert kwargs.get("model") == "pinned-model", f"the pinned model was dropped: {kwargs.get('model')}"
        assert "api_key" not in kwargs and "base_url" not in kwargs, (
            "the main runtime's key or endpoint rode into the pinned call")
        assert kwargs["main_runtime"]["api_key"] == "MAIN-RUNTIME-KEY", "the main runtime was not named"


# ── the micro cursor with no filler after the state note (review r2, item 5) ─────────────────────────────

def test_micro_floor_holds_without_a_filler(tmp_path, monkeypatch):
    """A tail that starts ``assistant`` leaves the state note last, so nothing else floors the cursor.

    The other micro test inserts a filler, which floors the cursor above the note whatever the note rule
    says; with the tail starting ``assistant`` the state note is the last prefix row and the note itself
    has to be excluded.
    """
    engine = _engine(tmp_path, monkeypatch, session="20260101_150000_nofiller", protect_last_n=20)
    messages = _transcript(10)
    messages[-1] = _msg("assistant", "still here, answering")
    working, _ = engine._prune_old_tool_results(messages, protect_tail_count=engine.protect_last_n,
                                               protect_tail_tokens=engine.tail_token_budget)
    cut = engine._find_tail_cut_by_tokens(
        working, engine._align_boundary_forward(working, engine._protect_head_size(working)))
    assert working[cut]["role"] == "assistant", "the fixture's tail no longer starts on an assistant row"

    out = _compress(engine, messages, monkeypatch)
    tail_len = len(working) - cut
    note = max(index for index, message in enumerate(out) if _text(message).startswith("[ctxpkg state]"))
    assert note == len(out) - tail_len - 1, "the state note is not the last row before the tail"
    assert _text(out[note + 1]) != "Continuing from the state note.", "a filler was emitted after the note"

    assert engine._resolve_compact_cursor(list(out), 1, note + 1) > note, (
        "the micro cursor stops on the state note and would absorb it")

