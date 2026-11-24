"""Episode compaction for the ``ctxpkg`` context engine (plan context-packages, step 4).

The engine replaces the middle of the transcript with episode files plus one reference block, and the
prefix it leaves behind is extend-only: a later compaction appends a second block and never rewrites the
first one, so the bytes up to the newest block's ack stay a prompt-cache read. These tests pin that, the
state note, warm/cold recall handling through prune and micro-compaction, and that the base class's own
compaction bookkeeping still runs.
"""

from __future__ import annotations

import json
from itertools import pairwise
from types import SimpleNamespace

import pytest

from agent import auxiliary_client
import agent.context_compressor as compressor_module
import hermes_cli.config as config_module
import plugins.context_engine.ctxpkg.store as store_module
from agent.context_compressor import _COMPACTION_TAIL_MARKER
from agent.prompt_cache_boundary import clear_cache_anchors, is_cache_anchor
from agent.prompt_caching import build_prompt_cache_plan, strip_anthropic_cache_control
from plugins.context_engine.ctxpkg import CtxpkgEngine
from plugins.context_engine.ctxpkg.episodes import recall_ref

SESSION = "20260101_120000_abcdef12"
WARM_TURNS = 8
SUMMARY_MAX_CHARS = 300


# ── fixtures and helpers ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ctxpkg_config(monkeypatch):
    """Pin the ``ctxpkg`` config block; the engine reads it through ``hermes_cli.config`` per compaction."""
    monkeypatch.setattr(
        config_module, "load_config_readonly",
        lambda: {"ctxpkg": {"warm_turns": WARM_TURNS, "max_episode_chars": 60_000,
                            "summary_max_chars": SUMMARY_MAX_CHARS}},
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
    """One user turn: the request, ``rounds`` tool calls with results, then the answer."""
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


def _llm(summary="did the step and passed", state="goal: ship step 4. next: run the tests."):
    """A stand-in for the flash summariser: one line per ref, then the state note."""
    prompts: list[str] = []

    def fake(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        if "one-line summaries" in prompt:
            refs = [line[4:].strip() for line in prompt.splitlines() if line.startswith("### ep-")]
            return _response("\n".join(f"{ref}: {summary}" for ref in refs))
        return _response(state)

    fake.prompts = prompts
    return fake


def _engine(tmp_path, monkeypatch, session=SESSION, **settings):
    """An engine over a throwaway home, wired only through public constructor attributes."""
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
    """Run one compaction with both call sites of ``call_llm`` standing in for the summariser."""
    fake = llm or _llm()
    monkeypatch.setattr(auxiliary_client, "call_llm", fake)
    monkeypatch.setattr(compressor_module, "call_llm", fake)
    return engine.compress(list(messages), current_tokens=150_000, force=True)


def _text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _block_indexes(messages):
    return [index for index, message in enumerate(messages) if _text(message).startswith("[ctxpkg references ")]


def _marked(message):
    if message.get("cache_control"):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(part, dict) and part.get("cache_control") for part in content)


def _role_problems(messages):
    """Strict-alternation violations and tool calls without a result, as a template would see them."""
    visible = [message for message in messages
               if message.get("role") != "tool"
               and not (message.get("role") == "assistant" and message.get("tool_calls"))]
    repeats = [(index, left.get("role"), right.get("role"))
               for index, (left, right) in enumerate(pairwise(visible))
               if left.get("role") == right.get("role")]
    answered = {message.get("tool_call_id") for message in messages if message.get("role") == "tool"}
    unanswered = [call.get("id") for message in messages for call in (message.get("tool_calls") or [])
                  if call.get("id") not in answered]
    return repeats, unanswered


def _recall_calls(messages):
    return [call for message in messages for call in (message.get("tool_calls") or []) if recall_ref(call)]


# ── the test list ──────────────────────────────────────────────────────────

def test_prefix_bytes_stable_across_two_compactions(tmp_path, monkeypatch):
    """Two compactions: the bytes up to the first block's ack are identical, and the newest block is the anchor."""
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    first_blocks = _block_indexes(first)
    assert len(first_blocks) == 1, "one compaction emits exactly one reference block"

    grown = list(first) + _transcript(12, start=11)[1:]
    second = _compress(engine, grown, monkeypatch)
    second_blocks = _block_indexes(second)
    assert len(second_blocks) == 2, "a later compaction appends a block instead of rewriting the first"

    cut_first = first_blocks[0] + 1
    cut_second = second_blocks[0] + 1
    plan_first = build_prompt_cache_plan([dict(message) for message in first], None, native_anthropic=True)
    plan_second = build_prompt_cache_plan([dict(message) for message in second], None, native_anthropic=True)
    assert _block_indexes(plan_first.messages) == first_blocks, "planning must not move a message"
    assert [_marked(plan_first.messages[index]) for index in first_blocks] == [True]
    assert [_marked(plan_second.messages[index]) for index in second_blocks] == [False, True], (
        "only the newest block carries the breakpoint")
    # Cache markers are request decoration and move to the newest block by design; the bytes they decorate
    # must not. Stripping them (in place) is what leaves the provider-visible prefix comparable.
    prefix_first = strip_anthropic_cache_control(plan_first.messages)[:cut_first + 1]
    prefix_second = strip_anthropic_cache_control(plan_second.messages)[:cut_second + 1]
    assert json.dumps(prefix_first) == json.dumps(prefix_second)
    assert engine.store.blocks() == [_text(first[first_blocks[0]]), _text(second[second_blocks[1]])]


def test_new_block_registered_as_anchor(tmp_path, monkeypatch):
    """The block a compaction emits is registered as the cache anchor and recorded for the next process."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    out = _compress(engine, messages, monkeypatch)

    block_indexes = _block_indexes(out)
    assert block_indexes, "a compaction emits a reference block"
    block = _text(out[block_indexes[0]])
    assert block.startswith("[ctxpkg references ep-0001..")
    assert is_cache_anchor(block)
    assert engine.store.blocks() == [block]
    assert not is_cache_anchor(_text(messages[0])), "nothing else is registered"


def test_anchor_reregistered_on_session_start(tmp_path, monkeypatch):
    """A fresh process re-registers every stored block at session start: the registry is process-local."""
    engine = _engine(tmp_path, monkeypatch)
    out = _compress(engine, _transcript(10), monkeypatch)
    block_indexes = _block_indexes(out)
    assert block_indexes, "a compaction emits a reference block"
    block = _text(out[block_indexes[0]])
    assert is_cache_anchor(block)

    clear_cache_anchors()
    assert not is_cache_anchor(block)

    resumed = _engine(tmp_path, monkeypatch)
    resumed.on_session_start(SESSION)
    assert is_cache_anchor(block)


def test_summary_unknown_sha_dropped(tmp_path, monkeypatch):
    """A summary inventing a sha or an ``sa-`` id loses it; the ids the episode holds survive."""
    engine = _engine(tmp_path, monkeypatch)
    real_sha = "b" * 40
    unknown_sha = "c" * 40
    real_task = "sa-41-abcdef01"
    unknown_task = "sa-99-deadbeef"
    messages = _transcript(10)
    messages[5] = _msg("user", f"turn 2: check {real_sha} on {real_task}")

    def fake(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "one-line summaries" not in prompt:
            return _response("goal: g.")
        refs = [line[4:].strip() for line in prompt.splitlines() if line.startswith("### ep-")]
        return _response("\n".join(
            f"{ref}: fixed {real_sha} and {real_task}, also {unknown_sha} and {unknown_task}" for ref in refs))

    out = _compress(engine, messages, monkeypatch, llm=fake)
    block_indexes = _block_indexes(out)
    assert block_indexes, "a compaction emits a reference block"
    block = _text(out[block_indexes[0]])
    assert real_sha in block and real_task in block
    assert unknown_sha not in block and unknown_task not in block


def test_llm_failure_mechanical_summary(tmp_path, monkeypatch):
    """An unreachable summariser still compacts: mechanical per-episode lines and the last user text."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    calls: list[str] = []

    def boom(**kwargs):
        calls.append(kwargs["messages"][0]["content"])
        raise RuntimeError("summary model unreachable")

    out = _compress(engine, messages, monkeypatch, llm=boom)
    assert engine.compression_count == 1, "a dead summariser must not abort the compaction"
    assert len(calls) == 1, "the state note's second call is skipped once the summary call failed"

    rows = engine.store.index()
    assert rows, "the span still became episodes"
    for row in rows:
        assert row["summary"], "every episode keeps a line"
        assert "(tools: terminal)" in row["summary"], "the mechanical line names the turn's tools"
        assert len(row["summary"]) <= SUMMARY_MAX_CHARS

    state_index = next(index for index, message in enumerate(out) if _text(message).startswith("[ctxpkg state]"))
    note = _text(out[state_index])
    last_user = [message["content"] for message in messages if message["role"] == "user"][-1]
    assert note.splitlines()[1] == last_user[:1500]


def test_warm_recall_survives_compaction(tmp_path, monkeypatch):
    """A warm recall moves forward intact, exactly once, and the next compaction adopts it again."""
    # (a) a recall with a sibling call in the span: the pair is re-emitted alone
    engine = _engine(tmp_path, monkeypatch, session="20260101_120000_aaaaaaa1")
    engine.store.write_episode("episode one body", turn_first=1, turn_last=1, summary="first summary")
    engine.store.renew("ep-0001", 0)
    messages = _transcript(6)
    messages.append(_msg("user", "now look up the old detail"))
    messages.append(_msg("assistant", "checking", [
        _call("r1", "recall", '{"ref": "ep-0001"}'), _call("t1", "terminal", '{"command": "ls"}')]))
    messages.append(_msg("tool", "FULL EPISODE ONE BODY", tool_call_id="r1"))
    messages.append(_msg("tool", "SIBLING OUTPUT", tool_call_id="t1"))
    messages.append(_msg("assistant", "answer: recalled"))
    messages.extend(_turn(20))
    messages.extend(_turn(21))
    out = _compress(engine, messages, monkeypatch)

    calls = _recall_calls(out)
    assert len(calls) == 1, "the warm recall is carried once"
    assert calls[0]["id"] == "r1"
    carrier = next(index for index, message in enumerate(out)
                   if any(recall_ref(call) for call in (message.get("tool_calls") or [])))
    assert out[carrier]["tool_calls"] == [calls[0]], "no sibling call rides along"
    assert not out[carrier].get("content"), "the carrier holds the call and no text"
    assert out[carrier + 1]["role"] == "tool" and out[carrier + 1]["tool_call_id"] == "r1"
    assert out[carrier + 1]["content"] == "FULL EPISODE ONE BODY"
    assert not any(call["id"] == "t1" for message in out for call in (message.get("tool_calls") or []))

    # (b) a warm recall already in the protected tail stays there, once
    taily = _engine(tmp_path, monkeypatch, session="20260101_120000_bbbbbbb2")
    taily.store.write_episode("tail episode body", turn_first=1, turn_last=1, summary="tail summary")
    taily.store.renew("ep-0001", 0)
    taily_messages = _transcript(8)
    taily_messages.extend([
        _msg("user", "recall the tail episode"),
        _msg("assistant", None, [_call("r9", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "TAIL EPISODE BODY", tool_call_id="r9"),
        _msg("assistant", "done"),
    ])
    taily_out = _compress(taily, taily_messages, monkeypatch)
    assert sum(1 for message in taily_out if _text(message) == "TAIL EPISODE BODY") == 1
    assert len(_recall_calls(taily_out)) == 1

    # (c) a second compaction whose span starts with the first one's warm pair adopts it again
    again = _compress(engine, list(out) + _transcript(12, start=30)[1:], monkeypatch)
    assert len(_block_indexes(again)) == 2
    assert len(_recall_calls(again)) == 1, "the pair is collected from the span and carried once"
    state_index = next(index for index, message in enumerate(again) if _text(message).startswith("[ctxpkg state]"))
    carried = next(index for index, message in enumerate(again)
                   if any(recall_ref(call) for call in (message.get("tool_calls") or [])))
    assert carried == state_index + 1, "carried pairs follow the state note"
    newest = [engine.store.read(ref) for ref in engine.store.refs()[-8:]]
    assert all(text.startswith("user: ") for text in newest), "no episode starts without a user message"


def test_cold_recall_becomes_stub(tmp_path, monkeypatch):
    """Past ``warm_turns`` the recalled body is replaced by the one-line stub naming its summary."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("cold episode body", turn_first=1, turn_last=1, summary="cold summary")
    engine.store.renew("ep-0001", 0)
    for _ in range(20):
        engine.store.bump_turn()

    messages = _transcript(2) + [
        _msg("user", "recall it"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "COLD EPISODE BODY " + "z" * 400, tool_call_id="r1"),
        _msg("assistant", "done"),
    ] + _transcript(3, start=5)[1:]
    result, _ = engine._prune_old_tool_results(messages, protect_tail_count=4)
    stubs = [_text(message) for message in result if message.get("role") == "tool"
             and _text(message).startswith("[recalled ep-0001, now cold")]
    assert stubs, "the cold recall's result was stubbed"
    assert "cold summary" in stubs[0] and 'recall("ep-0001")' in stubs[0]
    assert not any("COLD EPISODE BODY" in _text(message) for message in result)
    assert _recall_calls(result), "the call itself stays, so the pair is still valid"


def test_warm_recall_survives_prune(tmp_path, monkeypatch):
    """Prune keeps a warm recall's result verbatim, even before its boundary would demote it."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("warm episode body", turn_first=1, turn_last=1, summary="warm summary")
    engine.store.renew("ep-0001", 0)

    messages = _transcript(2) + [
        _msg("user", "recall it"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "WARM EPISODE BODY " + "q" * 2000, tool_call_id="r1"),
        _msg("assistant", "done"),
    ] + _transcript(4, start=5)[1:]
    result, pruned = engine._prune_old_tool_results(messages, protect_tail_count=4)

    kept = [_text(message) for message in result if message.get("role") == "tool"
            and message.get("tool_call_id") == "r1"]
    assert kept == ["WARM EPISODE BODY " + "q" * 2000], "a warm recall is never demoted"
    assert pruned >= 1, "other tool results in the same window still prune, so this asserted something"


def test_warm_recall_survives_micro_compact(tmp_path, monkeypatch):
    """Micro-compaction skips an exchange that holds a warm recall; the same exchange is absorbable cold."""
    engine = _engine(tmp_path, monkeypatch)
    engine.store.write_episode("micro episode body", turn_first=1, turn_last=1, summary="micro summary")
    engine.store.renew("ep-0001", 0)
    messages = _transcript(4) + [
        _msg("user", "recall it"),
        _msg("assistant", None, [_call("r1", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "MICRO EPISODE BODY " + "m" * 300, tool_call_id="r1"),
        _msg("assistant", "carried"),
    ] + _transcript(8, start=10)[1:]

    tail_start = engine._find_tail_cut_by_tokens(messages, 1)
    start = next(index for index, message in enumerate(messages)
                 if message.get("role") == "assistant" and message.get("tool_calls")
                 and any(recall_ref(call) for call in message["tool_calls"]))
    assert engine._find_one_exchange(messages, start, tail_start) is None

    for _ in range(20):
        engine.store.bump_turn()
    later = engine._find_one_exchange(messages, start, tail_start)
    assert later is not None and later[0] == start, "once cold, the exchange is absorbable again"

    engine._micro_compact_enabled = True
    engine._micro_compact_every_n_turns = 1
    engine._micro_summarize_one = lambda text: "ROLLING SUMMARY"
    engine.store.renew("ep-0001", engine.store.current_turn() - 1)
    result = engine._micro_compact(list(messages))
    kept = [_text(message) for message in result if message.get("role") == "tool"
            and message.get("tool_call_id") == "r1"]
    assert kept and "MICRO EPISODE BODY" in kept[0]


def test_blocks_survive_micro_compact(tmp_path, monkeypatch):
    """Micro-compaction starts after the prefix: the head and every block pair are byte-identical after it."""
    engine = _engine(tmp_path, monkeypatch)
    first = _compress(engine, _transcript(10), monkeypatch)
    grown = list(first) + _transcript(12, start=11)[1:]
    head = grown[: _block_indexes(grown)[0] + 2]

    engine._micro_compact_enabled = True
    engine._micro_compact_every_n_turns = 1
    engine._micro_summarize_one = lambda text: "ROLLING SUMMARY"
    result = engine._micro_compact(list(grown))

    assert json.dumps(result[: len(head)]) == json.dumps(head)
    assert _block_indexes(result) == _block_indexes(head)
    assert len(result) < len(grown), "the pass still absorbed something after the prefix"


def test_blocks_survive_prune(tmp_path, monkeypatch):
    """Prune never touches the prefix: from the system prompt to the state note, bytes hold.

    The head row at index 3 is a tool result over ``_PRUNE_MIN_CHARS``, so a prune pass that reaches it
    rewrites it (its dedupe pass on a later identical result, its demote pass on the length), and the
    slice is taken from index 0: a cut that started at the block would miss the head the prefix needs.
    """
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    # Index 3 is the first turn's first tool result, inside the head the first compaction protects.
    messages[3] = _msg("tool", "HEAD RESULT " + "h" * 1500, tool_call_id="c1-0")
    first = _compress(engine, messages, monkeypatch)
    grown = list(first) + _transcript(12, start=11, size=4000)[1:]
    prefix_end = next(index for index, message in enumerate(grown)
                      if _text(message).startswith("[ctxpkg state]")) + 1
    head = grown[:prefix_end]
    assert len(_text(grown[3])) > 1500, "the fixture really does hold a large head tool result"

    result, pruned = engine._prune_old_tool_results(grown, protect_tail_count=2)
    assert pruned > 0, "the boundary reached the old tool results"
    assert json.dumps(result[:prefix_end]) == json.dumps(head)
    assert _block_indexes(result) == _block_indexes(grown)


def test_compression_counters_advance(tmp_path, monkeypatch):
    """Every counter the base maintains moves as it does for a handoff summary on the same transcript."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    out = _compress(engine, messages, monkeypatch)
    first = dict(engine._last_compression_telemetry)
    assert engine.compression_count == 1
    assert engine._last_compression_made_progress is True
    assert engine._last_compression_savings_pct > 0
    assert first["messages_after"] == len(out) and first["messages_before"] == len(messages)
    assert first["tokens_reclaimed"] > 0

    _compress(engine, list(out) + _transcript(12, start=11)[1:], monkeypatch)
    assert engine.compression_count == 2
    assert engine._last_compression_telemetry["messages_after"] > 0

    base = compressor_module.ContextCompressor(model="test-model")
    base._config_context_length = 200_000
    base.quiet_mode = True
    base.protect_first_n = 3
    base.protect_last_n = 20
    base._threshold_tokens = 100_000
    base_out = _compress(base, messages, monkeypatch)
    assert base.compression_count == 1
    assert engine._last_compression_telemetry["method"] == base._last_compression_telemetry["method"]
    assert base._last_compression_made_progress is True
    assert _block_indexes(out) and not _block_indexes(base_out), (
        "the two compactions are different rewrites, not the same code path")


def test_roles_alternate_and_tool_pairs_valid(tmp_path, monkeypatch):
    """Four heads and tails, four clean role runs; no tool call is left unanswered."""
    shapes = {}

    shapes["tail_starts_user"] = (_transcript(10), 1)
    tail_assistant = _transcript(10)
    tail_assistant[-1] = _msg("assistant", "still here, answering")
    shapes["tail_starts_assistant"] = (tail_assistant, 20)
    inside_run = _transcript(10)
    inside_run.insert(4, _msg("assistant", None, [_call("extra", "terminal", '{"command": "x"}')]))
    inside_run.insert(5, _msg("tool", "extra result " + "e" * 100, tool_call_id="extra"))
    shapes["cut_inside_tool_run"] = (inside_run, 20)
    head_ends_user = [_msg("system", "You are a test driver."), _msg("assistant", "ready")] + _transcript(10)[1:]
    shapes["head_ends_user"] = (head_ends_user, 2)

    for name, (messages, protect_last) in shapes.items():
        engine = _engine(tmp_path, monkeypatch, session=f"20260101_120000_{abs(hash(name)) % 10**8:08d}",
                         protect_first_n=2 if name == "head_ends_user" else 3,
                         protect_last_n=protect_last)
        if name.startswith("tail_starts_"):
            # Before compressing: afterwards ``protect_first_n`` has decayed and the head is not the same.
            working, _ = engine._prune_old_tool_results(messages, protect_tail_count=protect_last,
                                                        protect_tail_tokens=engine.tail_token_budget)
            cut = engine._find_tail_cut_by_tokens(
                working, engine._align_boundary_forward(working, engine._protect_head_size(working)))
            expected = name.rsplit("_", 1)[-1]
            assert working[cut]["role"] == expected, f"{name}: the fixture really does start its tail on a {expected}"
        out = _compress(engine, messages, monkeypatch)
        repeats, unanswered = _role_problems(out)
        assert repeats == [], f"{name}: consecutive same-role messages {repeats}"
        assert unanswered == [], f"{name}: unanswered tool calls {unanswered}"
        assert _block_indexes(out), f"{name}: a compaction emits a block"
        assert all(recall_ref(call) is None for call in _recall_calls(out))


def test_tail_verbatim(tmp_path, monkeypatch):
    """The protected tail reaches the output unchanged, in order, with its calls and results intact."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    # The same tail the compressor will select: prune the copy the way compress does and cut at the same
    # boundary (before compressing, so ``protect_first_n`` has not decayed yet).
    working, _ = engine._prune_old_tool_results(messages, protect_tail_count=engine.protect_last_n,
                                                protect_tail_tokens=engine.tail_token_budget)
    cut = engine._find_tail_cut_by_tokens(
        working, engine._align_boundary_forward(working, engine._protect_head_size(working)))
    tail = working[cut:]
    assert len(tail) >= 4, "the fixture leaves a real tail to compare"

    out = _compress(engine, messages, monkeypatch)
    kept = out[-len(tail):]
    assert len(kept) == len(tail)
    for original, emitted in zip(tail, kept):
        assert emitted["role"] == original["role"]
        assert _text(emitted) == _text(original)
        assert emitted.get("tool_calls") == original.get("tool_calls")
        assert emitted.get("tool_call_id") == original.get("tool_call_id")
    assert all(emitted.get(_COMPACTION_TAIL_MARKER) for emitted in kept), (
        "every tail row carries the marker the host's archive dedupe reads")


def test_episode_files_written_for_span(tmp_path, monkeypatch):
    """One file per turn in the span, numbered session-wide, holding the tool calls and their results."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _transcript(10)
    # Every user turn the window holds becomes exactly one episode: a dropped or duplicated turn is a loss.
    # Computed before compressing, while ``protect_first_n`` still sets the head.
    working, _ = engine._prune_old_tool_results(messages, protect_tail_count=engine.protect_last_n,
                                                protect_tail_tokens=engine.tail_token_budget)
    start = engine._align_boundary_forward(working, engine._protect_head_size(working))
    cut = engine._find_tail_cut_by_tokens(working, start)
    span_users = [message for message in working[start:cut] if message.get("role") == "user"]

    out = _compress(engine, messages, monkeypatch)
    refs = engine.store.refs()
    assert refs and refs[0] == "ep-0001"
    assert refs == [f"ep-{number:04d}" for number in range(1, len(refs) + 1)]
    assert len(refs) == len(span_users), "one episode per user turn in the span"
    rows = engine.store.index()
    assert [row["turn_first"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["turn_first"] == row["turn_last"] for row in rows)
    for ref in refs:
        text = engine.store.read(ref)
        assert text.startswith("user: "), "an episode opens on a user turn"
        assert "[tool_call terminal(" in text and "\ntool: result " in text
        assert text.count("\nuser: ") == 0, "one episode is one turn"
    block = _text(out[_block_indexes(out)[0]]) if _block_indexes(out) else ""
    assert block, "a compaction emits a reference block"
    assert all(text in block for text in (f"{refs[0]} · turns 1-1", "chars · ", "ids: "))
    assert len(refs) < 10, "the protected tail's turns are not episodes"
