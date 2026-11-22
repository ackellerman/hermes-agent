"""Edge cases for the ``ctxpkg`` engine that have no home in the plan's 12 named tests.

The pinned file ``tests/plugins/test_ctxpkg_store.py`` carries exactly the twelve names the plan lists; everything
here is a case that is real but belongs to no single one of them: the host hook measured against the built-in
constructor, a failing hook, ref validation inside the store, ref listing, part bounds, lineage, symlinks, session
ids, the warmth lock, session-wide turn numbering and the config defaults. One test may probe several sibling paths
of the same rule; each mutant it kills is named in its docstring.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import agent_init
from agent.agent_init import _build_context_engine, _parse_compression_config
from plugins.context_engine.ctxpkg import CtxpkgEngine
from plugins.context_engine.ctxpkg import store as store_module
from plugins.context_engine.ctxpkg.episodes import (
    DEFAULT_MAX_EPISODE_CHARS,
    Episode,
    extract_ids,
    render,
    segment,
)
from plugins.context_engine.ctxpkg.store import EpisodeStore

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
    engine._session_db = None
    return engine


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
    """Drive the host's real ``_build_context_engine``; ``engine_cfg`` picks ctxpkg or the built-in compressor."""
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


class _Lineage:
    """A session DB stub answering only the lineage walk (and optionally raising)."""

    def __init__(self, mapping, explode=False):
        self.mapping = mapping
        self.explode = explode

    def get_compression_lineage(self, session_id):
        if self.explode:
            raise RuntimeError("session db gone")
        return self.mapping.get(session_id)


# ── the host hook, measured against the built-in constructor ────────────────

HOOK_ATTRIBUTES = (
    "summary_target_ratio",
    "abort_on_summary_failure",
    "max_tokens",
    "proactive_prune_min_result_chars",
    "proactive_prune_min_reclaim_tokens",
    "min_tail_user_messages",
    "tail_mode",
    "_configured_threshold_percent",
    "protect_first_n",
    "protect_last_n",
    "threshold_tokens_cap",
    "proactive_prune_tokens",
    "threshold_tokens",
)

# (label, context_length, compression overrides, agent max_tokens, expected on the ctxpkg engine)
HOOK_CONFIGS = (
    ("defaults", 1_000_000, {}, None,
     {"summary_target_ratio": 0.20, "abort_on_summary_failure": False, "max_tokens": None,
      "proactive_prune_min_result_chars": 8000, "proactive_prune_min_reclaim_tokens": 4096,
      "min_tail_user_messages": 1, "tail_mode": "lean", "_configured_threshold_percent": 0.50}),
    ("zeros-and-legacy", 300_000,
     {"proactive_prune_min_result_chars": 0, "proactive_prune_min_reclaim_tokens": 0,
      "target_ratio": 0.05, "abort_on_summary_failure": True, "min_tail_user_messages": 3,
      "tail_mode": "legacy"},
     None,
     {"summary_target_ratio": 0.10, "abort_on_summary_failure": True, "max_tokens": None,
      "proactive_prune_min_result_chars": 8000, "proactive_prune_min_reclaim_tokens": 0,
      "min_tail_user_messages": 3, "tail_mode": "legacy", "_configured_threshold_percent": 0.50}),
    ("floors-and-bogus", 1_000_000,
     {"proactive_prune_min_result_chars": 100, "target_ratio": 0.95, "tail_mode": "bogus",
      "threshold": 0.62},
     8192,
     {"summary_target_ratio": 0.80, "abort_on_summary_failure": False, "max_tokens": 8192,
      "proactive_prune_min_result_chars": 200, "proactive_prune_min_reclaim_tokens": 4096,
      "min_tail_user_messages": 1, "tail_mode": "lean", "_configured_threshold_percent": 0.62}),
)


@pytest.mark.parametrize("label,context_length,overrides,max_tokens,expected", HOOK_CONFIGS,
                         ids=[config[0] for config in HOOK_CONFIGS])
def test_hook_matches_builtin_constructor(label, context_length, overrides, max_tokens, expected):
    """Every knob the hook sets equals the built-in compressor built from the same ``cs``, over three configs.

    Kills a2 (no ``_configured_threshold_percent``), a6b/a6c (ratio clamps 0.10/0.80), a7 (no abort flag), a8 (no
    max_tokens), a10 (no prune min chars), a10b (floor dropped), a11 (no prune min reclaim), a12 (no min tail),
    a13/a13b (tail mode dropped or unvalidated), h3 (max_tokens=None passed to the hook), and every assignment the
    hook drops: each value here differs from the ``ContextCompressor`` constructor default.
    """
    compression = _compress_cfg(**overrides)
    engine = _built_engine(context_length, compression, max_tokens=max_tokens)
    builtin = _built_engine(context_length, compression, engine_cfg={"engine": "compressor"},
                            max_tokens=max_tokens)

    assert engine.name == "ctxpkg"
    assert builtin.name == "compressor"
    for attribute in HOOK_ATTRIBUTES:
        assert getattr(engine, attribute) == getattr(builtin, attribute), \
            f"{label}: {attribute} diverges from the built-in constructor"
    for attribute, value in expected.items():
        assert getattr(engine, attribute) == value, f"{label}: {attribute}"


def test_hook_failure_warns_and_construction_completes(tmp_path, monkeypatch, caplog):
    """A raising ``apply_compression_settings`` logs one warning (with the engine's name) and the agent still builds.

    Kills the bare ``suppress(Exception)``: silent is the wrong failure, because the thresholds fall back to the
    50% default and nothing says why.
    """

    class _Exploding(CtxpkgEngine):
        def apply_compression_settings(self, cs, max_tokens=None):
            raise RuntimeError("simulated hook failure")

    engine = _Exploding(model="")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(agent_init, "_select_context_engine", lambda _cfg: engine)

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        built = _built_engine(400_000, _compress_cfg(threshold=0.55))

    assert built is engine, "the failing hook aborted agent construction"
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("ctxpkg" in record.getMessage()
               and "apply_compression_settings failed" in record.getMessage()
               and "compression.* settings not applied" in record.getMessage()
               for record in warnings), [record.getMessage() for record in warnings]
    assert any(record.exc_info for record in warnings), "the warning carries no traceback"
    # the engine still got update_model (the hook's failure is not the agent's)
    assert engine.context_length == 400_000
    assert engine.threshold_tokens > 0


# ── store: refs, refs validation, next_turn ──────────────────────────────────

def test_next_ref_from_highest(tmp_path, monkeypatch):
    """``next_ref`` numbers from the highest existing ref even after lower files are gone (kills s4: count-based)."""
    store = _tmp_store(tmp_path, monkeypatch)
    for number in range(1, 4):
        store.write_episode(f"body {number}", turn_first=number, turn_last=number, ids=[], summary="s")
    (store.root / "ep-0001.md").unlink()
    (store.root / "ep-0002.md").unlink()

    assert store.refs() == ["ep-0003"]
    assert store.next_ref() == "ep-0004", "the next ref came from the file count, not the highest number"

    before = (store.root / "ep-0003.md").read_text()
    assert store.write_episode("body four", turn_first=4, turn_last=4, ids=[], summary="s") == "ep-0004"
    assert (store.root / "ep-0003.md").read_text() == before, "the write touched an existing episode"


def test_store_read_and_renew_validate_refs(tmp_path, monkeypatch):
    """``store.read`` never builds a path from a bad ref, and ``renew`` writes nothing for one.

    Kills s7 (``read`` without the fullmatch: it reads a file outside the store) and s10 (``renew`` without it).
    """
    store = _tmp_store(tmp_path, monkeypatch)
    store.write_episode("inside the store", turn_first=1, turn_last=1, ids=[], summary="s")
    # a file one level up, exactly where "../x" resolves: only a missing fullmatch reaches it
    (store.root.parent / "x.md").write_text("OUTSIDE-THE-STORE")
    warm = store.root / "warm.json"

    assert store.read("../x") is None, "read followed a ref with a separator"
    assert store.read("../x") != "OUTSIDE-THE-STORE"
    assert store.read("ep-0001/../x") is None
    assert store.read("ep-00001") is None
    assert store.read("") is None

    store.renew("junk")
    store.renew("../x")
    assert not warm.exists(), "renew wrote warm.json for a ref it refused"
    assert store.last_use("ep-0001") is None


def test_refs_ignores_non_episode_files(tmp_path, monkeypatch):
    """``refs()`` lists only ``ep-<four digits>.md`` (kills s19: any ``.md`` accepted)."""
    store = _tmp_store(tmp_path, monkeypatch)
    store.ensure()
    (store.root / "notes.md").write_text("notes")
    (store.root / "ep-12.md").write_text("short number")
    (store.root / "ep-0005.txt").write_text("wrong suffix")

    ref = store.write_episode("real episode", turn_first=1, turn_last=1, ids=[], summary="s")

    assert ref == "ep-0001", "a non-episode file consumed the next ref"
    assert store.refs() == ["ep-0001"]
    assert store.next_ref() == "ep-0002"


def test_next_turn_from_index(tmp_path, monkeypatch):
    """``next_turn`` is the last index row's ``turn_last + 1``, 1 when empty, and it survives a bad index line."""
    store = _tmp_store(tmp_path, monkeypatch)
    assert store.next_turn() == 1

    store.write_episode("one", turn_first=1, turn_last=1, ids=[], summary="s")
    store.write_episode("two", turn_first=2, turn_last=2, ids=[], summary="s")
    assert store.next_turn() == 3

    # a split turn spans several turns: the next one starts after its turn_last, not after its count
    store.write_episode("three", turn_first=3, turn_last=5, ids=[], summary="s")
    assert store.next_turn() == 6

    with (store.root / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert store.next_turn() == 6, "a malformed index line reset the turn numbering"


def test_store_follows_compression_lineage(tmp_path, monkeypatch):
    """The store dir is the compression-lineage root, not the physical id (kills s5), with a fallback when the
    lineage walk raises or there is no db; an episode the child writes is recalled from the root."""
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    db = _Lineage({"child-id": ["root-id", "child-id"], "root-id": ["root-id"]})
    base = tmp_path / "ctxpkg"

    assert EpisodeStore.for_session("child-id", db).root == base / "root-id"
    assert EpisodeStore.for_session("root-id", db).root == base / "root-id"
    assert EpisodeStore.for_session("solo", db).root == base / "solo"
    assert EpisodeStore.for_session("solo", _Lineage({}, explode=True)).root == base / "solo"
    assert EpisodeStore.for_session("solo", None).root == base / "solo"

    child = CtxpkgEngine(model="")
    child._session_id, child._session_db = "child-id", db
    child.store.write_episode("child body", turn_first=1, turn_last=1, ids=["R-001"], summary="s")

    root = CtxpkgEngine(model="")
    root._session_id, root._session_db = "root-id", db
    payload = json.loads(root.handle_tool_call("recall", {"ref": "ep-0001"}))
    assert payload["text"] == "child body", "the root-bound engine does not see the child's episode"


# ── store: path safety and warmth under concurrency ──────────────────────────

def test_recall_refuses_symlinked_episode(tmp_path, monkeypatch):
    """A symlinked ``ep-NNNN.md`` is an unknown episode: never followed, never counted as existing."""
    store = _tmp_store(tmp_path, monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    store.ensure()
    sentinel = tmp_path / "sentinel.md"
    sentinel.write_text("SENTINEL-OUTSIDE-THE-STORE")
    (store.root / "ep-0002.md").symlink_to(sentinel)

    assert store.exists("ep-0002") is False, "a symlink counted as a stored episode"
    assert store.read("ep-0002") is None, "read followed the symlink out of the store"
    payload = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0002"}))
    assert "error" in payload
    assert "SENTINEL-OUTSIDE-THE-STORE" not in json.dumps(payload)
    # and a mention of it renews nothing
    engine.on_turn_complete([_msg("assistant", "ep-0002 is here")])
    assert store.last_use("ep-0002") is None


def test_store_refuses_unsafe_session_id(tmp_path, monkeypatch):
    """A session id becomes one path component under ``ctxpkg/``; anything else lands on one fixed safe name."""
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    base = tmp_path / "ctxpkg"

    for unsafe in ("../x", "a/b", "..", ".", "", "../../etc/passwd", "a\\b"):
        root = EpisodeStore.for_session(unsafe, None).root
        assert root.parent == base, f"{unsafe!r} escaped ctxpkg/: {root}"
        assert root.name not in ("", ".", ".."), f"{unsafe!r} became the directory name"

    assert EpisodeStore.for_session(None, None).root == base / "nosession"
    assert EpisodeStore.for_session("", None).root == base / "nosession"
    # a lineage root is validated too, not only the physical id
    escaped = EpisodeStore.for_session("child", _Lineage({"child": ["../escape"]})).root
    assert escaped.parent == base, f"a lineage root escaped ctxpkg/: {escaped}"


def test_warm_state_survives_concurrent_renews(tmp_path, monkeypatch):
    """Twenty threads renewing distinct refs all survive: the ``warm.json`` read-modify-write is serialised."""
    store = _tmp_store(tmp_path, monkeypatch)
    store.ensure()
    refs = [f"ep-{number:04d}" for number in range(1, 21)]
    barrier = threading.Barrier(len(refs))

    def worker(ref):
        barrier.wait()
        store.renew(ref, 5)

    threads = [threading.Thread(target=worker, args=(ref,)) for ref in refs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state = json.loads((store.root / "warm.json").read_text())
    assert sorted(state["last_use"]) == refs, "a concurrent renew lost another's update"
    assert set(state["last_use"].values()) == {5}

    # the persisted turn counter is the same read-modify-write: twenty concurrent bumps must all land
    assert store.current_turn() == 0
    bumpers = [threading.Thread(target=store.bump_turn) for _ in range(20)]
    for thread in bumpers:
        thread.start()
    for thread in bumpers:
        thread.join()
    assert store.current_turn() == 20, "a concurrent bump lost another's increment"


# ── recall part bounds ───────────────────────────────────────────────────────

def test_recall_part_bounds(tmp_path, monkeypatch):
    """``part`` out of range is refused before the ref is renewed, and only ``part`` 99 reports ``parts``.

    Kills r2 (part < 1 accepted), r3 (part > parts accepted) and r14 (renew before the part check).
    """
    store = _tmp_store(tmp_path, monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    store.write_episode("body one", turn_first=1, turn_last=1, ids=[], summary="s")
    engine.on_turn_complete([])  # turn 1
    payload = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001"}))
    assert payload["text"] == "body one"
    assert store.last_use("ep-0001") == 1
    # move the clock on, so a refused part that still renewed would be visible as turn 2
    engine.on_turn_complete([])
    assert store.current_turn() == 2
    assert store.last_use("ep-0001") == 1

    for bad in (0, -1, "x", 99):
        refused = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001", "part": bad}))
        assert "error" in refused, f"part={bad!r} was accepted"
        assert store.last_use("ep-0001") == 1, f"part={bad!r} renewed the ref before refusing it"
    over = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001", "part": 99}))
    assert over["parts"] == 1
    assert "text" not in over

    # the default part is 1, and the boundary is both-sided
    assert json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001"}))["part"] == 1
    assert json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001", "part": 1}))["part"] == 1


# ── session-wide turn numbers ────────────────────────────────────────────────

def test_segment_first_turn_offset(tmp_path, monkeypatch):
    """``segment(..., first_turn=n)`` numbers turns from *n*, every part of a split turn keeps that number, and a
    store-fed numbering continues across calls (kills e11 numbering from 0, e12 a wrong ``turn_last``)."""
    transcript = [
        _msg("user", "first turn text"),
        _msg("assistant", "first answer"),
        _msg("user", "second turn text differs"),
        _msg("assistant", "second answer"),
    ]

    episodes = segment(transcript, first_turn=7)

    assert [(episode.turn_first, episode.turn_last) for episode in episodes] == [(7, 7), (8, 8)]
    assert segment(transcript)[0].turn_first == 1

    # a split turn keeps one number across its parts
    long_turn = [_msg("user", "long turn")]
    for index in range(40):
        long_turn.append(_msg("assistant", None, [_call(f"call-{index}", "terminal", '{"command": "ls"}')]))
        long_turn.append(_msg("tool", "r" * 400, tool_call_id=f"call-{index}"))
    split = segment(long_turn, max_chars=2_000, first_turn=4)
    assert len(split) > 1
    assert {episode.turn_first for episode in split} == {4}
    assert {episode.turn_last for episode in split} == {4}

    # the store is what makes the numbering session-wide: next_turn() picks up after the last write
    store = _tmp_store(tmp_path, monkeypatch)
    for episode in segment(transcript):
        store.write_episode(render(episode), turn_first=episode.turn_first,
                            turn_last=episode.turn_last, ids=[], summary="s")
    later = segment(transcript, first_turn=store.next_turn())
    assert [episode.turn_first for episode in later] == [3, 4]


# ── config defaults ─────────────────────────────────────────────────────────

def test_ctxpkg_config_defaults():
    """The three ``ctxpkg`` leaves and the segmenter's default agree (kills c1-c4: a wrong or missing block)."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["ctxpkg"] == {"warm_turns": 8, "max_episode_chars": 60000,
                                        "summary_max_chars": 300}
    assert DEFAULT_MAX_EPISODE_CHARS == DEFAULT_CONFIG["ctxpkg"]["max_episode_chars"]

    # the leaves are documented, and the docs name the engine's knobs
    docs = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "configuration.md"
    text = docs.read_text(encoding="utf-8")
    for leaf in ("ctxpkg.warm_turns", "ctxpkg.max_episode_chars", "ctxpkg.summary_max_chars"):
        assert leaf in text, f"{leaf} is undocumented"


def test_extract_ids_cap_boundary_at_store_default():
    """The id cap is exact at 24 (the 24th kept, the 25th dropped) for ids of every kind mixed together."""
    kinds = " ".join(f"sa-{index}-{index:08x}" for index in range(12)) + " " + \
        " ".join(f"R-{index:03d}" for index in range(12)) + " R-999"
    ids = extract_ids(kinds)
    assert len(ids) == 24
    assert ids[-1] == "R-011"
    assert "R-999" not in ids
