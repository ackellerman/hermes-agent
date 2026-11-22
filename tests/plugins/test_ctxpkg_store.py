"""The ``ctxpkg`` context engine's episode store, segmenter and ``recall`` tool (plan context-packages step 3).

The engine keeps every compacted turn as an episode file, indexes it, and hands the driver a ``recall(ref)``
tool that reads an episode back in 24000-char parts. These tests pin the store's file layout and modes, the
atomic write, the per-turn segmentation (state notes and recall stubs never become episodes), the id
extractor's kinds and 24-item cap, and the tool's ref validation (a ref is refused before any path is built).
"""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.agent_init import _build_context_engine, _parse_compression_config
from agent.context_engine import ContextEngine
from plugins.context_engine.ctxpkg import CtxpkgEngine
from plugins.context_engine.ctxpkg import store as store_module
from plugins.context_engine.ctxpkg.episodes import Episode, extract_ids, render, segment
from plugins.context_engine.ctxpkg.store import EpisodeStore, split_parts

SESSION = "20260101_000000_abcdef12"


# ── fixtures and helpers ─────────────────────────────────────────────────────

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


def _tmp_store(tmp_path, monkeypatch, session=SESSION):
    """An EpisodeStore whose home is *tmp_path* (never the real Hermes home)."""
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    return EpisodeStore.for_session(session, None)


def _engine(tmp_path, monkeypatch, session=SESSION):
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    engine = CtxpkgEngine(model="")
    engine._session_id = session
    return engine


def _two_turn_transcript():
    """Two user turns with different texts and a tool run each, plus a state note and a recall pair."""
    return [
        _msg("system", "You are a test driver."),
        _msg("user", "first turn text"),
        _msg("assistant", None, [_call("call-1", "terminal", '{"command": "ls -la"}')]),
        _msg("tool", "file list: a.py b.py", tool_call_id="call-1"),
        _msg("assistant", "first answer"),
        # a state note and its filler sit mid-list: only dropping them by CONTENT keeps the turn count at 2
        _msg("user", "[ctxpkg state] goal: ship the engine"),
        _msg("assistant", "Continuing from the state note."),
        _msg("user", "second turn text differs"),
        _msg("assistant", None, [_call("call-2", "recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "RECALLED-BODY-MARKER the full episode body", tool_call_id="call-2"),
        _msg("assistant", "second answer"),
    ]


# ── loading and host wiring ──────────────────────────────────────────────────

def test_engine_discovered_by_name(tmp_path, monkeypatch):
    """``context.engine: ctxpkg`` finds the bundled engine, and its toolset is the ``recall`` tool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.context_engine import discover_context_engines, load_context_engine

    engine = load_context_engine("ctxpkg")

    assert engine is not None, "the bundled ctxpkg engine did not load"
    assert engine.name == "ctxpkg"
    assert isinstance(engine, ContextCompressor)
    assert isinstance(engine, ContextEngine)
    assert "recall" in {schema.get("name") for schema in engine.get_tool_schemas()}
    assert "ctxpkg" in {name for name, _description, _available in discover_context_engines()}


def _compress_cfg(**overrides):
    cfg = {
        "threshold": 0.50,
        "protect_first_n": 3,
        "protect_last_n": 20,
        "proactive_prune_tokens": 0,
        "target_ratio": 0.20,
    }
    cfg.update(overrides)
    return cfg


def _built_engine(context_length, compression, engine_cfg=None, max_tokens=None):
    """Drive the host's real ``_build_context_engine`` for ``context.engine: ctxpkg``."""
    agent = SimpleNamespace(
        model="ctxpkg-test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key-1234567890",
        provider="anthropic",
        api_mode="chat_completions",
        quiet_mode=True,
        max_tokens=max_tokens,
        session_id=SESSION,
        context_compressor=None,
        _compression_threshold_autoraised=None,
    )
    config = {"context": engine_cfg or {"engine": "ctxpkg"}, "compression": compression}
    cs = _parse_compression_config(agent, config)
    with patch("agent.model_metadata.get_model_context_length", return_value=context_length):
        _build_context_engine(agent, config, cs, [], context_length, None)
    return agent.context_compressor


def test_engine_applies_compression_config():
    """Every ``compression.*`` key the built-in branch passes reaches ctxpkg, and the percent comes from
    ``cs.threshold`` (not ``cs.protect_*``/defaults) because ``update_model`` re-derives the trigger."""
    engine = _built_engine(
        400_000,
        _compress_cfg(threshold=0.50, threshold_tokens=123456, protect_first_n=2, protect_last_n=7,
                      proactive_prune_tokens=99000, target_ratio=0.15),
    )
    assert engine.name == "ctxpkg"
    assert engine.threshold_tokens_cap == 123456
    assert engine.protect_first_n == 2
    assert engine.protect_last_n == 7
    assert engine.proactive_prune_tokens == 99000
    assert engine.summary_target_ratio == 0.15
    # cap 123456 is below 0.50 x 400000, so the cap is the effective trigger
    assert engine.threshold_tokens == 123456

    # cap above ratio x window: the ratio wins, and it is the CONFIGURED percent (0.55), not the 0.50 default
    wide = _built_engine(1_000_000, _compress_cfg(threshold=0.55, threshold_tokens=600000))
    assert wide.threshold_tokens_cap == 600000
    assert wide.threshold_percent == pytest.approx(0.55)
    assert wide.threshold_tokens == int((1_000_000 - (wide.max_tokens or 0)) * 0.55)
    assert wide.threshold_tokens != int(1_000_000 * 0.50)
    assert wide._configured_threshold_percent == pytest.approx(0.55), "the raw percent did not reach the engine"

    # The remaining knobs of the built-in branch's constructor call, each at a value no default matches: a dropped
    # assignment leaves the constructor default behind (max_tokens None, abort off, 1 tail user message, lean tail,
    # prune mins 8000/4096).
    full = _built_engine(
        1_000_000,
        _compress_cfg(threshold=0.55, abort_on_summary_failure=True, min_tail_user_messages=3,
                      tail_mode="legacy", proactive_prune_min_result_chars=5000,
                      proactive_prune_min_reclaim_tokens=0),
        max_tokens=8192,
    )
    assert full.abort_on_summary_failure is True
    assert full.min_tail_user_messages == 3
    assert full.tail_mode == "legacy"
    assert full.max_tokens == 8192
    assert full.proactive_prune_min_result_chars == 5000
    # a configured 0 is the built-in's 8000 default (``int(x or 8000)``), never the 200 floor
    assert full.proactive_prune_min_reclaim_tokens == 0
    zero = _built_engine(1_000_000, _compress_cfg(proactive_prune_min_result_chars=0))
    assert zero.proactive_prune_min_result_chars == 8000
    assert zero.proactive_prune_min_reclaim_tokens == 4096
    floor = _built_engine(1_000_000, _compress_cfg(proactive_prune_min_result_chars=100))
    assert floor.proactive_prune_min_result_chars == 200, "the prune minimum lost its floor"


def test_prepared_compaction_skipped():
    """``prepare_ahead`` must not attach a PreparedCompaction to an engine that says it cannot use one, while
    the built-in compressor (no such attribute) still gets one."""
    engine = _built_engine(400_000, _compress_cfg(prepare_ahead=True, protect_first_n=2))
    assert engine.name == "ctxpkg"
    assert engine.supports_prepared_compaction is False
    assert getattr(engine, "prepared_compaction", None) is None

    builtin = _built_engine(400_000, _compress_cfg(prepare_ahead=True), engine_cfg={"engine": "compressor"})
    assert builtin.name == "compressor"
    assert builtin.prepared_compaction is not None, "prepare_ahead stopped working for the built-in engine"


# ── segmenter and renderer ───────────────────────────────────────────────────

def test_segment_one_episode_per_turn():
    """One episode per user turn; the state note, its filler and the recall pair never become episodes."""
    episodes = segment(_two_turn_transcript())

    assert len(episodes) == 2, "the state note split an extra turn"
    first, second = episodes
    assert first.messages[0] == _msg("user", "first turn text")
    assert second.messages[0] == _msg("user", "second turn text differs")
    assert "second turn text differs" not in render(first)
    assert "first turn text" not in render(second)
    assert (first.turn_first, first.turn_last) == (1, 1)
    assert (second.turn_first, second.turn_last) == (2, 2)

    assert "[ctxpkg state]" not in render(first) + render(second)
    assert "Continuing from the state note." not in render(first) + render(second)
    # the recall pair collapses to its one-line stub, and the recalled body never becomes episode text
    assert "[recalled ep-0001]" in render(second)
    assert "RECALLED-BODY-MARKER" not in render(first) + render(second)
    assert not any(m.get("tool_calls") for m in second.messages), "the recall tool call survived"

    # The state note is dropped only at the START of a user message, the filler only as a whole message without a
    # tool call, and only a ``recall`` call is stubbed: a sibling call, its result and a decoy name stay.
    # a message that merely MENTIONS the note is a turn start; a leading space still marks one
    mentions = [
        _msg("user", "see [ctxpkg state] above"),
        _msg("assistant", "ok"),
        _msg("user", "  [ctxpkg state] indented note"),
        _msg("assistant", "Continuing from the state note.\n"),
        _msg("assistant", "done"),
    ]
    turns = segment(mentions)

    assert [ep.messages[0]["content"] for ep in turns] == ["see [ctxpkg state] above"]
    assert "indented note" not in render(turns[0]), "an indented state note was kept"
    assert "Continuing from the state note." not in render(turns[0]), "a filler with a trailing newline was kept"

    # the filler is dropped only as the whole message: with a tool call it is real assistant output
    filler_with_call = [
        _msg("user", "turn one"),
        _msg("assistant", "Continuing from the state note.",
             [_call("call-a", "terminal", '{"command": "ls"}')]),
        _msg("tool", "FILLER-RESULT", tool_call_id="call-a"),
    ]
    kept = segment(filler_with_call)
    assert len(kept) == 1
    assert "Continuing from the state note." in render(kept[0])
    assert "FILLER-RESULT" in render(kept[0])
    assert kept[0].messages[1].get("tool_calls"), "the filler message's tool call was dropped with it"

    # a recall beside a sibling call: only the recall is stubbed, the sibling and its result survive
    siblings = [
        _msg("user", "turn one"),
        _msg("assistant", "checking", [_call("c-r", "recall", '{"ref": "ep-0002"}'),
                                       _call("c-t", "terminal", '{"command": "ls"}')]),
        _msg("tool", "RECALL-BODY", tool_call_id="c-r"),
        _msg("tool", "SIBLING-RESULT", tool_call_id="c-t"),
        _msg("assistant", "done"),
    ]
    one = segment(siblings)
    assert len(one) == 1
    text = render(one[0])
    assert "[recalled ep-0002]" in text
    assert "RECALL-BODY" not in text
    assert "SIBLING-RESULT" in text
    assert one[0].messages[1]["tool_calls"] == [_call("c-t", "terminal", '{"command": "ls"}')], \
        "the sibling call was dropped with the recall"

    # only a call NAMED recall is stubbed
    decoy = [
        _msg("user", "turn one"),
        _msg("assistant", None, [_call("c-n", "not_recall", '{"ref": "ep-0001"}')]),
        _msg("tool", "DECOY-RESULT", tool_call_id="c-n"),
    ]
    decoy_episode = segment(decoy)
    assert "DECOY-RESULT" in render(decoy_episode[0]), "a call named not_recall was stubbed as a recall"
    assert decoy_episode[0].messages[1].get("tool_calls")

    # a multimodal user turn is a turn start and its text parts reach the episode
    multimodal = [_msg("user", [{"type": "text", "text": "look at MULTIMODAL-MARKER"},
                                {"type": "image_url", "image_url": {"url": "data:,"}}]),
                  _msg("assistant", "seen")]
    multi = segment(multimodal)
    assert len(multi) == 1
    assert "MULTIMODAL-MARKER" in render(multi[0]), "a multimodal user turn's text was ignored"


def _long_turn(pad):
    """One turn: user text padded to *pad* extra chars, then tool runs and a final answer."""
    rounds = []
    for index in range(40):
        rounds.append(_msg("assistant", None, [_call(f"call-{index}", "terminal",
                                                     '{"command": "printf ' + str(index) + '"}')]))
        rounds.append(_msg("tool", "r" * 400, tool_call_id=f"call-{index}"))
    return [_msg("user", "long turn" + "x" * pad)] + rounds + [_msg("assistant", "done")]


def test_long_turn_split_at_tool_boundaries():
    """A turn over max_episode_chars splits only before an assistant tool call; exactly at the bound, no split."""
    turn = _long_turn(0)
    total = len(render(Episode(messages=turn, turn_first=1, turn_last=1)))
    at_edge = _long_turn(60_000 - total)
    assert len(render(Episode(messages=at_edge, turn_first=1, turn_last=1))) == 60_000

    assert len(segment(at_edge)) == 1, "a turn exactly at max_episode_chars must not split"
    over = segment(_long_turn(60_000 - total + 1))
    assert len(over) == 2, "a turn one char over max_episode_chars must split once"
    for part in over:
        assert over[0].turn_first == over[1].turn_first == 1
        assert part.messages[0]["role"] in ("user", "assistant")
        if part.messages[0]["role"] == "assistant":
            assert part.messages[0].get("tool_calls"), "a split landed inside a tool run"
        # every tool result in the part is preceded by its own tool call
        answered = set()
        for message in part.messages:
            if message["role"] == "assistant":
                answered |= {c["id"] for c in message.get("tool_calls") or []}
            elif message["role"] == "tool":
                assert message["tool_call_id"] in answered
        assert len(render(part)) <= 60_000

    # Chunk boundary: when two tool-call segments fill the budget EXACTLY, they share the chunk and the next
    # segment starts another (`>=` would break a segment early and put the second call in the wrong part).
    calls = [_msg("assistant", None, [_call(f"call-b{index}", "terminal", json.dumps({"command": f"cmd{index}"}))])
             for index in range(3)]
    results = [_msg("tool", f"RESULT-{index}", tool_call_id=f"call-b{index}") for index in range(3)]
    head = [_msg("user", "chunk turn"), calls[0], results[0], calls[1], results[1]]
    at_bound = len(render(Episode(messages=head, turn_first=1, turn_last=1)))
    chunked_turn = head + [calls[2], results[2]]
    assert len(render(Episode(messages=chunked_turn, turn_first=1, turn_last=1))) > at_bound

    chunks = segment(chunked_turn, max_chars=at_bound)
    assert len(chunks) == 2, "the turn did not split at the exact chunk boundary"
    assert "RESULT-1" in render(chunks[0]) and "RESULT-2" not in render(chunks[0])
    assert "RESULT-2" in render(chunks[1])
    assert (chunks[0].turn_first, chunks[1].turn_first) == (1, 1)


def test_render_keeps_full_tool_results():
    """Render is role-tagged, keeps every tool call's full arguments and every tool result in full."""
    body = ("line one\n" * 200) + "TOOL-RESULT-TAIL-MARKER"
    arguments = json.dumps({"command": "cat " + "a" * 3000, "timeout": 120})
    one = Episode(
        messages=[
            _msg("user", "run it"),
            _msg("assistant", None, [_call("call-9", "terminal", arguments)]),
            _msg("tool", body, tool_call_id="call-9"),
        ],
        turn_first=1,
        turn_last=1,
    )

    text = render(one)

    assert body in text, "the tool result was truncated"
    assert "TOOL-RESULT-TAIL-MARKER" in text
    assert arguments in text, "the tool call's arguments were truncated or dropped"
    assert text.startswith("user: run it\n")
    assert "\nassistant: " in text and "\ntool: " in text
    assert "len=" not in text


def test_extract_ids_kinds_and_cap():
    """Each id kind is extracted once, in first-seen order, and the list is capped at 24 items."""
    sha = "4f3a9c2b1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f90"
    text = (
        f"commit {sha} then sa-17-9f8e7d6c and R-042 and 12345678901234567 "
        "at /srv/proj/x/plan.md and ~/notes/a.md VERDICT: FIXES "
        "deadbeef1 abcabcabc abc123 0123456789abcdef0123456789abcdef01234567 "
        f"repeat {sha} again"
    )

    ids = extract_ids(text)

    assert ids == [
        sha,
        "sa-17-9f8e7d6c",
        "R-042",
        "12345678901234567",
        "/srv/proj/x/plan.md",
        "~/notes/a.md",
        "VERDICT: FIXES",
        "deadbeef1",
        "0123456789abcdef0123456789abcdef01234567",
    ], "kinds, order or dedupe are wrong"
    assert "abcabcabc" not in ids, "a hex run with no digit was kept"
    assert "abc123" not in ids, "a 6-char token was kept"

    many = " ".join(f"sa-{index}-{index:08x}" for index in range(25))
    capped = extract_ids(many)
    assert len(capped) == 24, "the 24-item cap did not hold"
    assert capped[0] == "sa-0-00000000"
    assert "sa-24-00000018" not in capped, "the 25th id was not dropped"
    assert capped[:24] == [f"sa-{index}-{index:08x}" for index in range(24)]

    # Each bound, both sides. A digit run needs no letter (it is a number, not a word) and must be WHOLE.
    assert extract_ids("1234567890123456") == [], "a 16-digit run was kept"
    assert extract_ids("12345678901234567") == ["12345678901234567"]
    assert extract_ids("12345678901234567890") == ["12345678901234567890"]
    assert extract_ids("123456789012345678901") == [], "a 21-digit run was trimmed to its first 20 digits"
    # a hex run needs both a digit and a letter, between 7 and 12 chars
    assert extract_ids("1234567") == [], "a 7-digit run with no letter was kept as a hex id"
    assert extract_ids("12345678901") == [], "an 11-digit run with no letter was kept as a hex id"
    assert extract_ids("abcdef012345") == ["abcdef012345"]
    assert extract_ids("abcdef0123456") == [], "a 13-hex token was kept"
    # a 40-hex sha and a 7-hex token are ids only free-standing, never inside a longer word
    assert extract_ids("x4f3a9c2b1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f90") == [], "a sha inside a word was kept"
    assert extract_ids("4f3a9c2b1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f90z") == [], "a sha followed by a letter was kept"
    assert extract_ids("xdeadbeef1") == [], "a 7-hex token inside a word was kept"
    assert extract_ids("deadbeef1z") == [], "a 7-hex token followed by a letter was kept"
    # a path loses the punctuation that ended it, and an sa- id takes ASCII digits only
    assert extract_ids("/srv/a/b.md.") == ["/srv/a/b.md"]
    assert extract_ids("(see ~/x)") == ["~/x"]
    assert extract_ids("sa-\u0661\u0662\u0663-9f8e7d6c") == ["9f8e7d6c"], \
        "a sa- id with non-ASCII digits was extracted"


# ── store ────────────────────────────────────────────────────────────────────

def test_store_modes_and_atomic_write(tmp_path, monkeypatch):
    """The store dir is 0700, its files 0600, refs number from the highest, and a crashed write keeps the old file."""
    store = _tmp_store(tmp_path, monkeypatch)

    ref = store.write_episode("episode one body", turn_first=1, turn_last=1,
                              ids=["R-001"], summary="first", created_at=1_700_000_000.0)
    assert ref == "ep-0001"
    assert store.root == tmp_path / "ctxpkg" / SESSION
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    # the parent lists every session id, so it is private too (mkdir's mode only reaches the leaf)
    assert stat.S_IMODE((tmp_path / "ctxpkg").stat().st_mode) == 0o700
    episode_path = store.root / "ep-0001.md"
    assert episode_path.read_text() == "episode one body"
    assert stat.S_IMODE(episode_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((store.root / "index.jsonl").stat().st_mode) == 0o600

    row = json.loads((store.root / "index.jsonl").read_text().splitlines()[-1])
    assert row == {
        "ref": "ep-0001", "created_at": 1_700_000_000.0, "turn_first": 1, "turn_last": 1,
        "chars": len("episode one body"), "ids": ["R-001"], "summary": "first",
    }

    def _crash(source, destination):
        raise OSError("simulated crash before os.replace")

    with monkeypatch.context() as crash:
        crash.setattr(store_module.os, "replace", _crash)
        with pytest.raises(OSError):
            store.write_episode("episode one REWRITTEN", turn_first=1, turn_last=1, ids=[], summary="second")

    assert episode_path.read_text() == "episode one body", "a crashed write replaced the old episode"
    assert len((store.root / "index.jsonl").read_text().splitlines()) == 1
    for leftover in store.root.glob("*.tmp"):
        leftover.unlink()
    assert store.next_ref() == "ep-0002"
    assert store.refs() == ["ep-0001"]

    # The index is opened O_APPEND: two writers at once each land one COMPLETE line (O_TRUNC loses one).
    def _append(number):
        store._append_index({"ref": f"ep-{number:04d}", "created_at": 0.0, "turn_first": number,
                             "turn_last": number, "chars": 0, "ids": [], "summary": "s"})

    threads = [threading.Thread(target=_append, args=(number,)) for number in (7, 8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (store.root / "index.jsonl").read_text().splitlines()
    assert len(lines) == 3, f"an append overwrote the index: {lines}"
    assert {json.loads(line)["ref"] for line in lines[1:]} == {"ep-0007", "ep-0008"}


# ── the recall tool ─────────────────────────────────────────────────────────

def test_recall_returns_parts(tmp_path, monkeypatch):
    """``recall`` returns one 24000-char part at a time, split on line boundaries, and the parts join back."""
    store = _tmp_store(tmp_path, monkeypatch)
    engine = _engine(tmp_path, monkeypatch)

    lines = [("a" * 69) + "\n" for _ in range(850)]  # 59500 chars, and 24000 is NOT a line boundary
    text = "".join(lines)
    store.write_episode(text, turn_first=1, turn_last=3, ids=[], summary="x")

    payload = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001"}))
    assert payload["ref"] == "ep-0001"
    assert payload["part"] == 1
    assert payload["parts"] == 3
    assert len(payload["text"]) <= 24_000
    parts = [payload["text"]]
    for index in (2, 3):
        parts.append(json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001", "part": index}))["text"])
    assert "".join(parts) == text, "the parts do not reconstruct the episode"
    assert all(part.endswith("\n") for part in parts[:-1])

    # exact edges: 24000 chars is one part even when it fills line by line, 24001 is two
    edge_body = "".join(("c" * 99) + "\n" for _ in range(240))  # exactly 24000, on line boundaries
    store.write_episode(edge_body, turn_first=4, turn_last=4, ids=[], summary="edge")
    edge = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0002"}))
    assert (edge["parts"], len(edge["text"])) == (1, 24_000)
    assert edge["text"] == edge_body, "a full part ending exactly on the limit was split"
    store.write_episode(edge_body + "d", turn_first=5, turn_last=5, ids=[], summary="edge+1")
    over = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0003"}))
    assert over["parts"] == 2
    assert len(over["text"]) == 24_000
    second = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0003", "part": 2}))
    assert second["text"] == "d"
    assert split_parts(edge_body + "d") == [edge_body, "d"]
    assert split_parts("d" * 24_001) == ["d" * 24_000, "d"]


def test_recall_unknown_ref_lists_nearest(tmp_path, monkeypatch):
    """An unknown ref reports up to 10 nearest refs by number, nearest first, ties to the lower number."""
    store = _tmp_store(tmp_path, monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    for number in range(1, 13):  # ep-0001..ep-0012
        store.write_episode(f"body {number}", turn_first=number, turn_last=number, ids=[], summary="s")
    (store.root / "ep-0009.md").unlink()  # a hole: 9 is now unknown while 1-8 and 10-12 remain

    payload = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0009"}))

    assert "error" in payload
    assert "ep-0009" in payload["error"]
    assert len(payload["known"]) == 10, "the known list is not capped at 10"
    assert payload["known"][:4] == ["ep-0008", "ep-0010", "ep-0007", "ep-0011"]
    assert payload["known"][-2:] == ["ep-0003", "ep-0002"]
    assert "ep-0001" not in payload["known"], "the farthest ref was not dropped"
    assert all(ref.startswith("ep-") for ref in payload["known"])


def test_recall_refuses_bad_ref_path(tmp_path, monkeypatch):
    """A ref not matching ``ep-[0-9]{4}`` is refused before a path is built or a file outside the store is read."""
    store = _tmp_store(tmp_path, monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    store.write_episode("inside the store", turn_first=1, turn_last=1, ids=[], summary="s")
    outside = tmp_path / "outside.md"
    outside.write_text("SENTINEL-OUTSIDE-THE-STORE")

    opened = []
    real_open = Path.open

    def _recording_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _recording_open)

    for bad in ("../../x", "ep-12", "ep-0001/../x", "ep-0001.md", "ep-00001", "ep-\u0660\u0660\u0660\u0661"):
        payload = json.loads(engine.handle_tool_call("recall", {"ref": bad}))
        assert "error" in payload, f"{bad!r} was not refused"
        assert "known" not in payload
        assert "SENTINEL-OUTSIDE-THE-STORE" not in json.dumps(payload)

    store_root = str(store.root)
    strays = [path for path in opened if not path.startswith(store_root)]
    assert strays == [], f"a refused ref opened a path outside the store: {strays}"


def test_recall_and_mention_renew_warmth(tmp_path, monkeypatch):
    """A recall and a mention of a known ref in assistant text both renew its last-use turn, persisted."""
    store = _tmp_store(tmp_path, monkeypatch)
    store.write_episode("body one", turn_first=1, turn_last=1, ids=[], summary="s1")
    store.write_episode("body two", turn_first=2, turn_last=2, ids=[], summary="s2")
    engine = _engine(tmp_path, monkeypatch)

    engine.on_turn_complete([_msg("assistant", "I checked ep-0001 for the header")])
    assert store.current_turn() == 1
    assert store.last_use("ep-0001") == 1

    for _ in range(3):
        engine.on_turn_complete([_msg("assistant", "nothing to renew here")])
    assert store.current_turn() == 4
    assert store.last_use("ep-0001") == 1, "an unrelated turn renewed the ref"

    payload = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0002"}))
    assert payload["text"] == "body two"
    assert store.last_use("ep-0002") == 4, "the recall did not renew the ref"

    fresh = _engine(tmp_path, monkeypatch)
    assert fresh.store.current_turn() == 4, "the turn counter was not persisted"
    assert fresh.store.last_use("ep-0002") == 4
    assert json.loads((store.root / "warm.json").read_text())["last_use"]["ep-0002"] == 4

    # a mention of an UNKNOWN ref renews nothing
    engine.on_turn_complete([_msg("assistant", "ep-0999 is not stored")])
    assert store.current_turn() == 5
    assert "ep-0999" not in json.loads((store.root / "warm.json").read_text())["last_use"]

    # one assistant message naming two known refs renews both
    engine.on_turn_complete([_msg("assistant", "ep-0001 and ep-0002 both matter")])
    assert (store.last_use("ep-0001"), store.last_use("ep-0002")) == (6, 6)

    # only ASSISTANT text renews: a user message and a tool result naming a ref renew nothing
    engine.on_turn_complete([_msg("user", "remind me about ep-0001"),
                             _msg("tool", "ep-0001 body text", tool_call_id="call-x"),
                             _msg("assistant", "no refs here")])
    assert store.current_turn() == 7
    assert (store.last_use("ep-0001"), store.last_use("ep-0002")) == (6, 6)

    # a malformed warm.json value is dropped, not fatal
    (store.root / "warm.json").write_text(json.dumps({"turn": 3,
                                                      "last_use": {"ep-0001": "x", "ep-0002": 9}}))
    assert store.current_turn() == 3
    assert store.last_use("ep-0001") is None, "a non-integer last-use value was kept"
    assert store.last_use("ep-0002") == 9

    # an unknown TOOL name is refused even when the ref it carries exists
    assert "error" in json.loads(engine.handle_tool_call("other", {"ref": "ep-0001"}))

    assert "error" in json.loads(engine.handle_tool_call("not_a_tool", {}))

    class _BrokenStore:
        def bump_turn(self):
            raise OSError("disk gone")

    monkeypatch.setattr(CtxpkgEngine, "store", property(lambda self: _BrokenStore()))
    engine.on_turn_complete([_msg("assistant", "ep-0001")])  # must not raise: a broken store is not a turn failure
