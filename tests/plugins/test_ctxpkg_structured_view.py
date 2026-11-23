"""The ``ctxpkg`` structured view (plan structured-context, step 4).

``ctxpkg.view: structured`` serves a driver's request as
``system | prefill | head | marker blocks | state note | the rest of the request``, computed on every
provider request from the aligned pair ``(request, uids)`` the host hands the engine. The view is a pure
function of that pair plus ``view.json``: between passes a new request only appends to the tail, so every
byte up to the tail stays the same. Material moves only when a pass runs, and a pass runs only on the first
request after the session's own cache tier expired since the last request. Everything not allowed reads and
writes no view file, log or stamp: children and forks keep today's behaviour exactly.

The tests drive the engine through ``select_context`` with synthetic message lists, fake uids, an injected
clock and judge; the store root lives under ``tmp_path`` (``HERMES_HOME`` monkeypatched).
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.config as config_module
import plugins.context_engine.ctxpkg.store as store_module
from agent.conversation_loop import _apply_context_engine_selection, _engine_overrides_hook
from plugins.context_engine.ctxpkg import CtxpkgEngine
from plugins.context_engine.ctxpkg.compaction import (
    NOTED_FILLER,
    block_text,
    load_settings,
)
from plugins.context_engine.ctxpkg.episodes import STATE_FILLER
from plugins.context_engine.ctxpkg.store import EpisodeStore

SESSION = "20260101_120000_abcdef12"
TTL = 300
KEEP_TURNS = 3
MIN_STUB_CHARS = 200
VIEW_PREFIX = "[ctxpkg references "
STUB_MARK = "recall(\"ep-"
SHARED_ARGS = json.dumps({"command": "pytest -q"})
TOOL_UID = object()


# ── fixtures and helpers ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ctxpkg_config(monkeypatch):
    """Pin the ``ctxpkg`` config block in structured mode (and a short ``keep_turns`` for the tests)."""
    monkeypatch.setattr(
        config_module,
        "load_config_readonly",
        lambda: {"ctxpkg": {
            "warm_turns": 8,
            "max_episode_chars": 60_000,
            "summary_max_chars": 300,
            "view": "structured",
            "structure": {
                "keep_turns": KEEP_TURNS,
                "old_after_minutes": 120,
                "min_stub_chars": MIN_STUB_CHARS,
                "max_candidates": 40,
                "candidate_chars": 4000,
                "state_note_chars": 1200,
                "timeout_seconds": 45,
            },
        }},
    )
    # ``session_source_for`` reads the session env first; a stray value would decide ``view_allowed``.
    for name in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(name, raising=False)


def _view():
    """The module under test, imported lazily so the red run fails per test, not at collection."""
    from plugins.context_engine.ctxpkg import view

    return view


class _Clock:
    """An injected clock: ``view.now_epoch`` is monkeypatched to this callable."""

    def __init__(self, start: float = 1_000_000.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += float(seconds)
        return self.t


def _clock(monkeypatch, start: float = 1_000_000.0) -> _Clock:
    clock = _Clock(start)
    monkeypatch.setattr(_view(), "now_epoch", clock)
    return clock


def _msg(role, text=None, *, uid=None, tool_calls=None, tool_call_id=None, timestamp=None):
    message = {"role": role}
    if text is not None:
        message["content"] = text
    if tool_calls:
        message["tool_calls"] = tool_calls
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    if uid is not None:
        message["message_uid"] = uid
    if timestamp is not None:
        message["timestamp"] = timestamp
    return message


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class _umask:
    """Run a block under a pinned process umask and restore it (the store's modes are the claim)."""

    def __init__(self, value: int):
        self.value = value
        self.old = 0o022

    def __enter__(self):
        self.old = os.umask(self.value)
        return self

    def __exit__(self, *exc_info):
        os.umask(self.old)
        return False


def _tool_turn(index, *, args=None, uid_result=TOOL_UID, text=None, size=300, name="terminal"):
    """One turn: the request, one call with its result, then the answer. ``args`` repeats a call."""
    arguments = args if args is not None else json.dumps({"command": f"run step {index}"})
    result_uid = f"t{index}" if uid_result is TOOL_UID else uid_result
    return [
        _msg("user", f"turn {index}: please run it", uid=f"u{index}"),
        _msg("assistant", None, uid=f"a{index}", tool_calls=[_call(f"c{index}", name, arguments)]),
        _msg("tool", text if text is not None else "y" * size, uid=result_uid, tool_call_id=f"c{index}"),
        _msg("assistant", f"done {index}", uid=f"z{index}"),
    ]


def _calls_turn(index, calls):
    """One turn whose assistant message makes several calls, each answered by its own result row."""
    rows = [_msg("user", f"turn {index}: please run them", uid=f"u{index}"),
            _msg("assistant", None, uid=f"a{index}",
                 tool_calls=[_call(call_id, name, arguments) for call_id, name, arguments in calls])]
    for call_id, _name, _arguments in calls:
        rows.append(_msg("tool", "y" * 300, uid=f"t{index}-{call_id}", tool_call_id=call_id))
    rows.append(_msg("assistant", f"done {index}", uid=f"z{index}"))
    return rows


def _recall_turn(index, ref):
    """One turn that recalls *ref*: the call and its result, then the answer."""
    return [
        _msg("user", f"turn {index}: recall {ref}", uid=f"u{index}"),
        _msg("assistant", None, uid=f"a{index}",
             tool_calls=[_call(f"cr{index}", "recall", json.dumps({"ref": ref}))]),
        _msg("tool", "y" * 300, uid=f"tr{index}", tool_call_id=f"cr{index}"),
        _msg("assistant", f"done {index}", uid=f"z{index}"),
    ]


def _rounds_turn(index, rounds=2, size=300):
    """The ctxpkg suite's own turn shape: two tool calls with results, then the answer."""
    rows = [_msg("user", f"turn {index}: please run the step {index} check", uid=f"u{index}")]
    for round_index in range(rounds):
        call_id = f"c{index}-{round_index}"
        rows.append(_msg("assistant", None, uid=f"a{index}-{round_index}",
                         tool_calls=[_call(call_id, "terminal",
                                           json.dumps({"command": f"cmd {index} {round_index}"}))]))
        rows.append(_msg("tool", f"result {index}-{round_index} " + "y" * size,
                         uid=f"t{index}-{round_index}", tool_call_id=call_id))
    rows.append(_msg("assistant", f"answer {index}: done", uid=f"z{index}"))
    return rows


def _ctxpkg_transcript(turns=10):
    """A transcript the mixin's own suite proves compactable (``test_ctxpkg_compaction``'s shape)."""
    rows = [_system()]
    for index in range(1, turns + 1):
        rows += _rounds_turn(index)
    return rows


def _superseded_history(turns=6, size=300):
    """Turn 2's tool result is superseded: its call is made again, verbatim, in turn 3."""
    rows = _tool_turn(1)
    rows += _tool_turn(2, args=SHARED_ARGS, size=size)
    rows += _tool_turn(3, args=SHARED_ARGS, size=size)
    for index in range(4, turns + 1):
        rows += _tool_turn(index, size=size)
    rows.append(_msg("user", "now the current question", uid="ucur"))
    return rows


def _uids(conversation):
    return [message.get("message_uid") for message in conversation]


def _system(text="You are a driver."):
    return {"role": "system", "content": text}


def _engine(tmp_path, monkeypatch, *, allowed=True, session=SESSION, ttl=TTL, threshold=100_000,
            tail_budget=None):
    """An engine over a throwaway home, wired only through public constructor attributes."""
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    engine = CtxpkgEngine(model="test-model")
    engine.quiet_mode = True
    engine.protect_first_n = 3
    engine.protect_last_n = 20
    engine._config_context_length = 200_000
    engine.context_length = 200_000
    engine.threshold_tokens = threshold
    if tail_budget is not None:
        engine.tail_token_budget = tail_budget
    engine._session_id = session
    engine.view_allowed = allowed
    engine.cache_ttl_seconds = ttl
    return engine


def _store_path(tmp_path, session=SESSION):
    return Path(tmp_path) / "ctxpkg" / session


def _store(tmp_path, session=SESSION):
    return EpisodeStore(_store_path(tmp_path, session))


def _seed(tmp_path, session=SESSION, **fields):
    """Write ``view.json`` for the session, as a previous process left it."""
    state = _view().ViewState.load(_store(tmp_path, session))
    for name, value in fields.items():
        setattr(state, name, value)
    state.save(_store(tmp_path, session))
    return state


def _ask(engine, conversation, *, prefill=0, request=None):
    """One allowed provider request over ``conversation``; returns ``(request, served)``."""
    engine.request_uids = _uids(conversation)
    engine.prefill_count = prefill
    if request is None:
        request = [_system(), *conversation]
    served = engine.select_context(request, conversation_messages=[dict(m) for m in conversation])
    return request, served


def _rows(tmp_path, name, session=SESSION):
    path = _store_path(tmp_path, session) / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _state(tmp_path, session=SESSION):
    return _view().ViewState.load(_store(tmp_path, session))


def _text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _index_of(served, prefix):
    for index, message in enumerate(served):
        if _text(message).startswith(prefix):
            return index
    return -1


def _role_problems(messages):
    """Strict-alternation repeats and unanswered tool calls, as a template would see them."""
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


def _pairing_problems(messages):
    """Tool results whose ``tool_call_id`` is not in the preceding assistant message's ``tool_calls``."""
    problems = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        wanted = message.get("tool_call_id")
        preceding = messages[index - 1] if index else None
        calls = {call.get("id") for call in (preceding or {}).get("tool_calls") or []}
        if not preceding or preceding.get("role") != "assistant" or wanted not in calls:
            problems.append(wanted)
    return problems


def _judge_structure_all(*, action="structure", summary="the same call was made again later", state_note=None):
    """A judge that structures every candidate the pass offers it, optionally moving the state note."""

    def judge(candidates, state_note_arg):
        reply = {"decisions": [{"id": candidate.id, "action": action, "summary": summary}
                               for candidate in candidates]}
        if state_note is not None:
            reply["state_note"] = state_note
        return reply

    return judge


def _fake_agent(engine, **overrides):
    agent = SimpleNamespace(
        context_compressor=engine,
        session_id=SESSION,
        model="claude-opus-5-5",
        provider="anthropic",
        _cache_ttl="1h",
        _use_prompt_caching=True,
        _request_message_uids=["m1", "m2"],
        prefill_messages=[],
        _context_engine_tool_names={"recall"},
        platform="discord",
        _memory_write_context="foreground",
        _persist_disabled=False,
        context_length=200_000,
    )
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent


def _wire_agent(**overrides):
    """A host-shaped agent for the REAL ``build_api_messages`` path: no live model, no side effect."""
    agent = SimpleNamespace(
        model="claude-opus-5-5",
        provider="anthropic",
        base_url="",
        api_mode=None,
        _current_turn_timestamp=1_000_000.0,
        ephemeral_system_prompt="",
        _copy_reasoning_content_for_api=lambda msg, api_msg: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    for name, value in overrides.items():
        setattr(agent, name, value)
    return agent


def _build_request(agent, messages, current_turn_user_idx):
    """The request ``build_api_messages`` assembles, exactly as the host hands it to the hook."""
    from agent.turn_context import build_api_messages

    request, _system_text = build_api_messages(
        agent, messages, current_turn_user_idx=current_turn_user_idx, ext_prefetch_cache=None,
        plugin_user_context=None, moa_config=None, active_system_prompt="You are a driver.",
    )
    return request


# ── the test list ──────────────────────────────────────────────────────────


def test_view_off_returns_none(tmp_path, monkeypatch, caplog):
    """``view: off`` declines the hook on the instance, so the host leaves the request untouched."""
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {"view": "off"}})
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    engine = CtxpkgEngine(model="test-model")
    assert engine.select_context is None, "off must set the instance attribute to None, not just return None"

    agent = _fake_agent(engine)
    conversation = _superseded_history()
    request = [_system(), *conversation]
    assert _apply_context_engine_selection(
        agent, request, conversation, None, logger=logging.getLogger("test"),
    ) is request
    assert not (_store_path(tmp_path) / "view.json").exists(), "a view-off engine never writes view state"

    # Settings: a missing or invalid ``view`` parses as ``off`` (never ``structured``), and every
    # ``structure`` leaf falls back to its own default with exactly one warning. ``StructureSettings`` is
    # imported here, not at module top: it arrives with the structured view, so a module-top import would
    # make this whole file uncollectable on a tree that does not have the view yet.
    from plugins.context_engine.ctxpkg.compaction import StructureSettings

    caplog.clear()
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {}})
    settings = load_settings()
    assert settings.view == "off", "a missing view is off, not structured"
    assert (settings.structure.keep_turns, settings.structure.min_stub_chars,
            settings.structure.timeout_seconds) == (8, 2000, 45), (
        "the structure defaults are the plan's own (keep_turns 8, min_stub_chars 2000, timeout 45)")
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING], (
        "a missing block is the default, not a warning")

    caplog.clear()
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {"view": "Structured!"}})
    assert load_settings().view == "off", "an unrecognised view is off"
    assert len([record for record in caplog.records if "ctxpkg config" in record.getMessage()]) == 1

    caplog.clear()
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {"structure": {
        "keep_turns": True, "min_stub_chars": 0, "timeout_seconds": "45"}}})
    settings = load_settings()
    assert settings.structure == StructureSettings(), (
        "a bool, a value under the minimum and a non-int each take their own default")
    warnings = [record.getMessage() for record in caplog.records if "ctxpkg config" in record.getMessage()]
    assert len(warnings) == 1, "several bad leaves make one warning"
    assert "structure.keep_turns" in warnings[0] and "structure.min_stub_chars" in warnings[0]

    caplog.clear()
    monkeypatch.setattr(config_module, "load_config_readonly",
                        lambda: {"ctxpkg": {"view": "off", "structure": "nope"}})
    settings = load_settings()
    assert settings.structure == StructureSettings(), "a structure block that is not a mapping is the defaults"
    assert len([record for record in caplog.records if "ctxpkg config" in record.getMessage()]) == 1, (
        "a non-mapping structure warns")


def test_view_off_hook_not_overridden(tmp_path, monkeypatch):
    """The host's override check is False with ``view: off`` and True with ``view: structured``."""
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: {"ctxpkg": {"view": "off"}})
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    assert _engine_overrides_hook(CtxpkgEngine(model="test-model"), "select_context") is False

    monkeypatch.setattr(config_module, "load_config_readonly",
                        lambda: {"ctxpkg": {"view": "structured"}})
    assert _engine_overrides_hook(CtxpkgEngine(model="test-model"), "select_context") is True


def test_view_bytes_stable_between_passes(tmp_path, monkeypatch):
    """Two requests with one appended tail message and no pass between them: the view is a prefix."""
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    view = _view()
    with _umask(0o022):
        blocks = [
            block_text("ep-0001", "ep-0001",
                       ["ep-0001 · superseded · 2026-11-23 09:00Z · 1k chars · an old result · ids: none"]),
            block_text("ep-0002", "ep-0002",
                       ["ep-0002 · superseded · 2026-11-23 09:10Z · 1k chars · a later result · ids: none"]),
        ]
        _seed(tmp_path, blocks=blocks, state_note="[ctxpkg state]\ngoal: ship the view")
        anchored = []
        monkeypatch.setattr(view, "register_anchor", anchored.append)

        conversation = _superseded_history()
        _, first = _ask(engine, conversation)
        assert first is not None

        # Every block is served, oldest first, each with its ack; the state note pair comes after them.
        first_block = _index_of(first, "[ctxpkg references ep-0001")
        second_block = _index_of(first, "[ctxpkg references ep-0002")
        note_index = _index_of(first, "[ctxpkg state]")
        assert 0 < first_block < second_block < note_index, "blocks are served in the order they were written"
        assert first[second_block + 1] == {"role": "assistant", "content": "Noted: ep-0002..ep-0002. "
                                           "I recall an episode with recall(ref) when I need its detail."}
        assert first[note_index + 1] == {"role": "assistant", "content": STATE_FILLER}, (
            "the state note is a user message with its assistant filler")
        assert anchored == [blocks[-1]], "the newest stored block is re-registered on the first allowed request"

        grown = conversation + [_msg("user", "one more message", uid="uextra")]
        _, second = _ask(engine, grown)
        assert second is not None

        assert len(second) == len(first) + 1, "a request that added one message adds exactly one to the view"
        assert second[:len(first)] == first, "every byte up to the appended tail is identical"
        assert json.dumps(first, sort_keys=True) == json.dumps(second[:len(first)], sort_keys=True)
        assert _rows(tmp_path, "passes.jsonl") == [], "no pass ran while the cache was warm"
        assert anchored == [blocks[-1]], "the anchor is registered once per engine instance, not per request"

        # The first file a session writes creates the store the way every other writer does: 0700
        # directories and a 0600 file, whatever the process umask is (F4).
        root = _store_path(tmp_path)
        assert stat.S_IMODE((tmp_path / "ctxpkg").stat().st_mode) == 0o700
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / "view.json").stat().st_mode) == 0o600


def test_no_pass_while_cache_warm(tmp_path, monkeypatch):
    """A request inside the tier never runs a pass."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()

    _ask(engine, conversation)
    started = clock.t
    clock.advance(TTL - 1)
    _ask(engine, conversation)

    assert _rows(tmp_path, "passes.jsonl") == []
    assert _state(tmp_path).last_request_at == pytest.approx(started + TTL - 1)
    assert _state(tmp_path).last_pass_at is None


def test_pass_after_ttl_since_last_request(tmp_path, monkeypatch):
    """The next request at or past the tier runs exactly one pass."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()

    _ask(engine, conversation)
    clock.advance(TTL)
    _request, served = _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1, "one pass for the expired gap"
    assert rows[0]["status"] == "ok"
    assert rows[0]["candidates"] == 1, "only turn 2's superseded result is a candidate"
    assert rows[0]["structured"] == 0, "the step's judge keeps everything"
    assert rows[0]["ttl_s"] == TTL
    assert rows[0]["idle_s"] == pytest.approx(TTL)
    assert served is not None

    # A request with one prefill row still aligns, and the pass is handed the same prefill count: the
    # candidate is offered (a pass that lost the count would find the request one message too long).
    prefill = {"role": "user", "content": "Prefill: answer with one line."}
    clock.advance(TTL)
    _ask(engine, conversation)                       # re-arms: the pass's own stamp bounds the gap
    clock.advance(TTL)
    _ask(engine, conversation, prefill=1, request=[_system(), prefill, *conversation])
    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert rows[1]["candidates"] == 1, "run_pass aligns with the prefill rows between the system and the uids"

    # A cold ``recall`` result is a candidate through a real pass: run_pass hands candidates the store.
    recalled = _tool_turn(1) + _recall_turn(2, "ep-0005") + _tool_turn(3) + _tool_turn(4) \
        + [_msg("user", "current", uid="ucur-recall")]
    clock.advance(TTL)
    _ask(engine, recalled)
    clock.advance(TTL)
    _ask(engine, recalled)
    rows = _rows(tmp_path, "passes.jsonl")
    assert rows[-1]["status"] == "ok" and rows[-1]["candidates"] == 1, (
        "a cold recall result is offered to the pass; a pass that dropped the store would see none")


def test_gap_measured_from_request_not_response(tmp_path, monkeypatch):
    """The tier clock starts at the request: a response 60 s later does not shorten the gap."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()

    started = clock.t
    _ask(engine, conversation)
    # A second request inside the same second does not move the stamp: it is written at most once a
    # second, and the cache clock the provider holds started at the FIRST one.
    clock.advance(0.4)
    _ask(engine, conversation)
    assert _state(tmp_path).last_request_at == pytest.approx(started, rel=0, abs=1e-6), (
        "the stamp is at most once a second; a second request in the same second keeps the first")
    # Two seconds later the stamp MOVES (a once-a-second rule, not a once-a-minute one).
    clock.advance(1.6)
    _ask(engine, conversation)
    assert _state(tmp_path).last_request_at == pytest.approx(started + 2, rel=0, abs=1e-6), (
        "a request past the one-second limit restamps last_request_at")

    clock.advance(58)
    engine.update_from_response({"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11})
    clock.advance(TTL - 30)
    _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1, "the gap is measured from the last request, not the response"
    assert rows[0]["idle_s"] == pytest.approx(TTL + 28)


def test_no_pass_without_last_request(tmp_path, monkeypatch):
    """The first request after the keys go live has no gap to measure and gets no pass (nor does an unset tier)."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()

    _ask(engine, conversation)
    assert _rows(tmp_path, "passes.jsonl") == []
    assert _state(tmp_path).last_request_at == pytest.approx(clock.t)

    clock.advance(1)
    _ask(engine, conversation)
    assert _rows(tmp_path, "passes.jsonl") == []

    # ``cache_ttl_seconds`` unset (no prompt cache, or the config says none): no tier, so no gap and no
    # pass — and no float(None) inside the request path.
    clock.advance(TTL)
    engine.cache_ttl_seconds = None
    _, served = _ask(engine, conversation)
    assert served is not None
    assert _rows(tmp_path, "passes.jsonl") == [], "a session with no cache tier never runs a pass"
    assert _state(tmp_path).last_request_at == pytest.approx(clock.t)


def test_one_pass_per_idle_gap(tmp_path, monkeypatch):
    """One attempt per expired window: ``last_pass_at`` stamps even a request that runs no pass.

    The plan's trigger: a pass runs when ``now - last_request_at >= cache_ttl_seconds`` AND
    ``last_pass_at < last_request_at``. Both stamps are written by the pass request, so the next request
    after the tier does not retry; the request after that re-arms it.
    """
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()

    _ask(engine, conversation)
    clock.advance(TTL)
    _ask(engine, conversation)
    assert len(_rows(tmp_path, "passes.jsonl")) == 1

    clock.advance(TTL)
    _ask(engine, conversation)
    assert len(_rows(tmp_path, "passes.jsonl")) == 1, "the pass's own stamp bounds the next window to no attempt"

    clock.advance(TTL)
    _ask(engine, conversation)
    assert len(_rows(tmp_path, "passes.jsonl")) == 2, "a later request re-arms the trigger"

    # A pass that finds nothing still records its row: ``no_candidates`` (G4 counts it, not a pass).
    quiet = _tool_turn(1) + _tool_turn(2) + [_msg("user", "current", uid="ucur-quiet")]
    clock.advance(TTL)
    _ask(engine, quiet)                              # the pass's own stamp: no attempt in this gap
    clock.advance(TTL)
    _ask(engine, quiet)
    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 3
    assert rows[-1]["status"] == "no_candidates" and rows[-1]["candidates"] == 0, (
        "a pass with nothing to offer still writes its row")


def test_failed_pass_not_retried_in_same_gap(tmp_path, monkeypatch):
    """A judge failure stamps ``last_pass_at`` too: the gap gets one attempt at most."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)

    def failing(candidates, state_note):
        raise RuntimeError("judge exploded")

    engine._judge = lambda: failing
    conversation = _superseded_history()

    _ask(engine, conversation)
    clock.advance(TTL)
    _ask(engine, conversation)
    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    stamped = _state(tmp_path).last_pass_at
    assert stamped == pytest.approx(clock.t)

    clock.advance(TTL)
    _ask(engine, conversation)
    assert len(_rows(tmp_path, "passes.jsonl")) == 1, (
        "the failed attempt's own stamp bounds the next expired window: one attempt per gap")
    assert _state(tmp_path).last_pass_at == pytest.approx(stamped)

    clock.advance(TTL)
    _ask(engine, conversation)
    assert len(_rows(tmp_path, "passes.jsonl")) == 2, "a later request re-arms the trigger, as for a real pass"
    assert "failed" in [row["status"] for row in _rows(tmp_path, "passes.jsonl")]


def test_last_request_at_survives_new_engine(tmp_path, monkeypatch):
    """``last_request_at`` is persisted, so a freshly built engine measures the gap from it."""
    clock = _clock(monkeypatch)
    first = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()
    _ask(first, conversation)
    stamped = _state(tmp_path).last_request_at

    clock.advance(TTL)
    second = _engine(tmp_path, monkeypatch)
    _ask(second, conversation)
    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1, "the new instance read last_request_at from view.json"
    assert rows[0]["idle_s"] == pytest.approx(TTL)
    assert _state(tmp_path).last_request_at > stamped

    other = tmp_path / "other-home"
    fresh = _engine(other, monkeypatch)
    _ask(fresh, conversation)
    assert _rows(other, "passes.jsonl") == [], "a store with no last_request_at gets no pass"


def test_alignment_after_stripped_tail(tmp_path, monkeypatch):
    """The real request build records one uid per canonical message, and the view aligns on that list."""
    view = _view()
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    agent = _wire_agent()

    # A turn that died on a read-only call: the replay canonicalization drops the dangling assistant row.
    prefix = _tool_turn(1) + [
        _msg("user", "look at the file", uid="u2"),
        _msg("assistant", None, uid="a2",
             tool_calls=[_call("cx", "read_file", json.dumps({"path": "~/notes/a.md"}))]),
    ]
    messages = prefix + [_msg("user", "now continue", uid="unow")]
    request = _build_request(agent, messages, len(messages) - 1)
    uids = agent._request_message_uids
    assert len(uids) == len(request) - 1, "the system message is the only request row without a uid"
    assert uids == ["u1", "a1", "t1", "z1", "u2", "unow"]
    assert "a2" not in uids, "the stripped dangling call is not in the request, so it has no uid"

    engine.request_uids = uids
    engine.prefill_count = 0
    served = engine.select_context(request, conversation_messages=messages)
    assert served is not None, "the view is built on the uid list the real request build recorded"
    # The stripped call's own user row survives and now meets the current user row: the HOST merges
    # adjacent user rows after the hook, so the view's contract here is the tool pairing, not alternation.
    assert _pairing_problems(served) == []
    assert not [row for row in _rows(tmp_path, "passes.jsonl") if row["status"] == "view_off"]

    # The uid list of the pre-canonicalization history is one longer than the request: misaligned.
    assert view.build_view(request, _uids(messages), 0, _state(tmp_path)) is None

    # One prefill row sits between the system message and the uids: the aligned part starts at offset 2.
    prefill = {"role": "user", "content": "Prefill: answer in JSON."}
    prefixed = [_system(), prefill, *request[1:]]
    assert view.alignment(prefixed, uids, 1) == (2, "")
    engine.prefill_count = 1
    served_prefill = engine.select_context(prefixed, conversation_messages=messages)
    assert served_prefill is not None and served_prefill[:2] == [_system(), prefill]

    # A dangling SIDE-EFFECTING call is kept and answered by a fabricated recovery result with no uid:
    # the recorded list carries that None, so the row is served and the view still aligns.
    terminal_prefix = _tool_turn(1) + [
        _msg("user", "restart it", uid="u9"),
        _msg("assistant", None, uid="a9",
             tool_calls=[_call("c9", "terminal", json.dumps({"command": "docker restart x"}))]),
    ]
    messages2 = terminal_prefix + [_msg("user", "now continue", uid="unow2")]
    request2 = _build_request(agent, messages2, len(messages2) - 1)
    uids2 = agent._request_message_uids
    assert len(uids2) == len(request2) - 1
    assert uids2[-2:] == [None, "unow2"], "the fabricated recovery result has no uid"
    assert [message.get("role") for message in request2[-3:]] == ["assistant", "tool", "user"]
    assert request2[-2].get("tool_call_id") == "c9" and "message_uid" not in request2[-2]

    engine.request_uids = uids2
    engine.prefill_count = 0
    served2 = engine.select_context(request2, conversation_messages=messages2)
    assert served2 is not None, "a uid-None row in the middle does not turn the view off"
    assert _pairing_problems(served2) == []


def test_alignment_mismatch_view_off_logged(tmp_path, monkeypatch, caplog):
    """A length mismatch, a request with no system message and no uid list: None, one INFO line, one row each."""
    caplog.set_level(logging.INFO)
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()
    with _umask(0o022):
        engine.request_uids = _uids(conversation)

        longer = [_system(), *conversation, {"role": "user", "content": "extra"}]
        assert engine.select_context(longer, conversation_messages=conversation) is None
        assert _state(tmp_path).last_request_at == pytest.approx(clock.t), (
            "a misaligned request is still an allowed select_context: it stamps last_request_at")

        clock.advance(5)
        no_system = [*conversation, {"role": "user", "content": "extra"}]
        assert engine.select_context(no_system, conversation_messages=conversation) is None

        clock.advance(5)
        engine.request_uids = None
        assert engine.select_context([_system(), *conversation], conversation_messages=conversation) is None

        off = [row for row in _rows(tmp_path, "passes.jsonl") if row["status"] == "view_off"]
        assert len(off) == 3, "every misaligned request records one view_off row"
        assert len({row["reason"] for row in off}) == 3, "each row names its own reason"
        assert "no_uids" in {row["reason"] for row in off}, "a request with no uid list is refused, not guessed"
        # A log file is 0600 at umask 022, like every other file the store writes.
        assert stat.S_IMODE((_store_path(tmp_path) / "passes.jsonl").stat().st_mode) == 0o600
    lines = [record for record in caplog.records if "structured view off" in record.getMessage()]
    assert len(lines) == 1, "the INFO line is logged once per session"
    assert all(record.levelno == logging.INFO for record in lines)


def test_uidless_messages_served_never_candidates(tmp_path, monkeypatch):
    """A uid-None recovery result is served as is, and its turn is no candidate."""
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    view = _view()

    conversation = _tool_turn(1)
    conversation += _tool_turn(2, args=SHARED_ARGS, uid_result=None)
    conversation += _tool_turn(3, args=SHARED_ARGS)
    for index in range(4, 7):
        conversation += _tool_turn(index)
    conversation.append(_msg("user", "now the current question", uid="ucur"))

    request = [_system(), *conversation]
    settings = load_settings()
    found = view.candidates(request, _uids(conversation), conversation, _state(tmp_path), settings,
                            store=_store(tmp_path))
    assert [candidate.uid for candidate in found] == [], "a turn holding a uid-None row is never a candidate"
    assert all(candidate.uid is not None for candidate in found)

    _, served = _ask(engine, conversation)
    assert served is not None
    assert any(_text(message) == "y" * 300 for message in served), "the uid-None result is served, not skipped"


def test_stub_keeps_tool_pairing(tmp_path, monkeypatch):
    """A structured tool result is served as its stub in place: every kept call keeps its result."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    result_text = "sa-77-deadbeef " + "y" * 300
    invented_sha = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    summary = f"moved {invented_sha} aside sa-77-deadbeef " + "and " * 120
    note = "goal: ship the view. " + "next: " * 300
    engine._judge = lambda: _judge_structure_all(summary=summary, state_note=note)

    conversation = _superseded_history()
    for message in conversation:
        if message.get("message_uid") == "t2":
            message["content"] = result_text
    _ask(engine, conversation)
    engine.on_turn_complete([dict(message) for message in conversation])   # a served turn: view_turn 1
    clock.advance(TTL)
    _, served = _ask(engine, conversation)

    assert served is not None
    assert _pairing_problems(served) == []
    assert _role_problems(served) == ([], [])
    stub_index = _index_of(served, "[ep-0001 · superseded · ")
    assert stub_index > 0, "the superseded result is served as its stub"
    stub = _text(served[stub_index])
    assert STUB_MARK in stub
    assert served[stub_index]["role"] == "tool"
    assert served[stub_index]["tool_call_id"] == "c2", "the stub keeps its own call id"
    assert served[stub_index - 1].get("role") == "assistant"
    assert any(call.get("id") == "c2" for call in served[stub_index - 1].get("tool_calls") or []), (
        "the call that produced the stubbed result is still served right before it")
    assert (_store_path(tmp_path) / "ep-0001.md").read_text(encoding="utf-8") == result_text, (
        "the episode keeps the full tool text, not a summary")
    # The judge's summary is scrubbed of ids the episode does not hold, and cut at the cap.
    shown = stub.split(" · ")[3]
    assert invented_sha not in stub, "an id the episode does not hold never reaches the prompt"
    assert "sa-77-deadbeef" in stub, "an id the episode does hold survives the scrub"
    assert len(shown) <= 300, "a summary longer than summary_max_chars is cut at a word boundary"

    # The pass recorded what a rerun is detected by, and when (which served turn) the ref moved.
    state = _state(tmp_path)
    assert state.structured == {"t2": "ep-0001"}
    assert state.structured_turn == {"ep-0001": 1}, "the pass records the view_turn it ran at, not 0"
    assert state.stub_calls == {"ep-0001": ["terminal", json.dumps(json.loads(SHARED_ARGS), sort_keys=True)]}
    # The judge's state note is applied, capped at ctxpkg.structure.state_note_chars, and served.
    assert state.state_note == note.strip()[:1200]
    assert len(state.state_note) == 1200, "a state note past the cap is cut"
    note_index = _index_of(served, "goal: ship the view")
    assert note_index > 0 and served[note_index]["role"] == "user"
    assert served[note_index + 1] == {"role": "assistant", "content": STATE_FILLER}

    # A recall of the ref the pass wrote is today's recall (unchanged text) plus one recalls.jsonl row.
    answered = json.loads(engine.handle_tool_call("recall", {"ref": "ep-0001"}))
    assert answered["text"] == result_text and answered["part"] == 1
    rows = _rows(tmp_path, "recalls.jsonl")
    assert len(rows) == 1
    assert rows[0]["ref"] == "ep-0001" and rows[0]["kind"] == "superseded"
    assert rows[0]["status"] == "ok" and rows[0]["turns_since_structured"] == 0

    # A failed recall of a pass-written ref still logs, with status "error".
    engine.handle_tool_call("recall", {"ref": "ep-0001", "part": 99})
    rows = _rows(tmp_path, "recalls.jsonl")
    assert len(rows) == 2 and rows[-1]["status"] == "error"
    # A ref no pass wrote gets no row at all.
    engine.handle_tool_call("recall", {"ref": "ep-0404"})
    assert len(_rows(tmp_path, "recalls.jsonl")) == 2
    # And while the view is not allowed the recall still answers, but writes no row.
    engine.view_allowed = False
    assert "y" in engine.handle_tool_call("recall", {"ref": "ep-0001"})
    assert len(_rows(tmp_path, "recalls.jsonl")) == 2, "a recall of a child or fork logs nothing"


def test_structured_turn_removed_whole(tmp_path, monkeypatch):
    """A structured turn leaves no message behind: the rest of its turn goes with it."""
    _clock(monkeypatch)
    _engine(tmp_path, monkeypatch)
    view = _view()

    first = _tool_turn(1)
    second = _tool_turn(2)
    third = _tool_turn(3)
    conversation = first + second + third + _tool_turn(4) + [_msg("user", "current", uid="ucur")]

    turn_uids = [message["message_uid"] for message in second]
    marker = "ep-0007 · superseded · 2026-11-23 09:00Z · 1k chars · turn two · ids: c2"
    state = _seed(
        tmp_path,
        blocks=[block_text("ep-0007", "ep-0007", [marker])],
        structured={uid: "ep-0007" for uid in turn_uids},
        structured_turn={"ep-0007": 1},
    )

    served = view.build_view([_system(), *conversation], _uids(conversation), 0, state)
    assert served is not None
    texts = [_text(message) for message in served]
    assert "turn 2: please run it" not in texts, "the turn's user message is gone"
    assert not [text for text in texts if text.startswith("done 2")]
    assert "c2" not in {message.get("tool_call_id") for message in served}
    assert _pairing_problems(served) == []
    assert _role_problems(served) == ([], [])
    assert "turn 3: please run it" in texts, "only the structured turn is skipped"
    assert _index_of(served, VIEW_PREFIX) > 0


def test_filler_between_user_and_block(tmp_path, monkeypatch):
    """A head whose last template-visible role is user gets ``NOTED_FILLER`` before the first block."""
    _clock(monkeypatch)
    _engine(tmp_path, monkeypatch)
    view = _view()

    head = [
        _msg("user", "turn 1: read it", uid="u1"),
        _msg("assistant", None, uid="a1", tool_calls=[_call("c1", "read_file", json.dumps({"path": "~/notes/a.md"}))]),
        _msg("tool", "z" * 300, uid="t1", tool_call_id="c1"),
    ]
    conversation = head + _tool_turn(2) + _tool_turn(3) + [_msg("user", "current", uid="ucur")]
    block = block_text("ep-0001", "ep-0001", ["ep-0001 · superseded · 2026-11-23 09:00Z · 1k chars · x · ids: none"])
    state = _seed(tmp_path, blocks=[block])

    served = view.build_view([_system(), *conversation], _uids(conversation), 0, state)
    assert served is not None
    block_index = _index_of(served, VIEW_PREFIX)
    assert block_index > 0
    assert served[block_index - 1] == {"role": "assistant", "content": NOTED_FILLER}
    assert served[block_index]["role"] == "user"
    assert _role_problems(served) == ([], [])


def test_never_candidates(tmp_path, monkeypatch):
    """Every never-candidate rule keeps its own row out, and the one free result stays in.

    Turn 7 (inside ``keep_turns``) re-makes every earlier call, so each earlier result would be a
    ``superseded`` candidate but for the rule that owns it.
    """
    _clock(monkeypatch)
    _engine(tmp_path, monkeypatch)
    view = _view()

    rows = _tool_turn(1)                                                      # head
    rows += _tool_turn(2, args=SHARED_ARGS)                                   # the only candidate
    turn3 = _tool_turn(3, args=SHARED_ARGS)                                   # a uid-None row in the turn
    del turn3[0]["message_uid"]                          # (the fabricated recovery row: served, no uid)
    rows += turn3
    rows += _tool_turn(4, args=json.dumps({"command": "run step 4"}))
    rows += _tool_turn(5, args=json.dumps({"command": "run step 5"}),
                       text="y" * 300 + " sa-77-deadbeef")
    rows += _tool_turn(6, args=json.dumps({"command": "keep me"}))            # inside keep_turns
    rows += _calls_turn(7, [
        ("c7a", "terminal", json.dumps({"command": "run step 1"})),
        ("c7b", "terminal", SHARED_ARGS),
        ("c7c", "terminal", json.dumps({"command": "run step 4"})),
        ("c7d", "terminal", json.dumps({"command": "run step 5"})),
        ("c7e", "terminal", json.dumps({"command": "keep me"})),
    ])
    current = [
        _msg("user", "now the current question", uid="ucur"),
        _msg("assistant", None, uid="acur",
             tool_calls=[_call("ccur", "recall", json.dumps({"ref": "ep-0099"}))]),
        _msg("tool", "y" * 300, uid="tcur", tool_call_id="ccur"),
    ]
    rows += current

    state = _seed(tmp_path, structured={"t4": "ep-0003"}, state_note="[ctxpkg state]\ngoal: sa-77-deadbeef")
    request = [_system(), *rows]
    settings = load_settings()
    found = view.candidates(request, _uids(rows), rows, state, settings, store=_store(tmp_path))
    assert [candidate.uid for candidate in found] == ["t2"], (
        "head, current turn, keep_turns, uid-None turn, already structured and state-note ids all stay out")

    # Each rule on its own: with the state note cleared, turn 5's result becomes a candidate.
    free = _seed(tmp_path, structured={"t4": "ep-0003"}, state_note="")
    found = view.candidates(request, _uids(rows), rows, free, settings, store=_store(tmp_path))
    assert "t5" in [candidate.uid for candidate in found]
    # And with turn 4 no longer structured, its result becomes one too.
    unstuck = _seed(tmp_path, structured={}, state_note="")
    found = view.candidates(request, _uids(rows), rows, unstuck, settings, store=_store(tmp_path))
    assert "t4" in [candidate.uid for candidate in found]

    # A superseded result under ``structure.min_stub_chars`` is never a candidate.
    small = _tool_turn(1) + _tool_turn(2, args=SHARED_ARGS, size=20) + _tool_turn(3, args=SHARED_ARGS) \
        + _tool_turn(4) + [_msg("user", "current", uid="ucur-small")]
    found = view.candidates([_system(), *small], _uids(small), small, _state(tmp_path), settings,
                            store=_store(tmp_path))
    assert found == [], "a result smaller than min_stub_chars never becomes a stub"

    # A recall exchange whose ref is cold is a candidate; once warm, it is not.
    recalled = _tool_turn(1) + _recall_turn(2, "ep-0005") + _tool_turn(3) + _tool_turn(4) \
        + [_msg("user", "current", uid="ucur-recall")]
    found = view.candidates([_system(), *recalled], _uids(recalled), recalled, _state(tmp_path),
                            settings, store=_store(tmp_path))
    assert [candidate.uid for candidate in found] == ["tr2"], "a cold recall result is a candidate"
    _store(tmp_path).renew("ep-0005")
    found = view.candidates([_system(), *recalled], _uids(recalled), recalled, _state(tmp_path),
                            settings, store=_store(tmp_path))
    assert found == [], "a warm recall result is not a candidate"

    # With ``keep_turns: 0`` the turn in flight is still protected — only the current-turn rule can do it.
    no_keep = replace(settings, structure=replace(settings.structure, keep_turns=0))
    in_flight = _tool_turn(1) + _recall_turn(2, "ep-0006") + [
        _msg("user", "the turn in flight", uid="u-now"),
        _msg("assistant", None, uid="a-now",
             tool_calls=[_call("cr-now", "recall", json.dumps({"ref": "ep-0006"}))]),
        _msg("tool", "y" * 300, uid="t-now", tool_call_id="cr-now"),
    ]
    found = view.candidates([_system(), *in_flight], _uids(in_flight), in_flight, _state(tmp_path),
                            no_keep, store=_store(tmp_path))
    assert [candidate.uid for candidate in found] == ["tr2"], (
        "the turn in flight is never a candidate, keep_turns or not")

    # One pass offers at most ``max_candidates``, oldest first.
    again = [json.dumps({"command": f"again {index}"}) for index in range(1, 4)]
    many = _tool_turn(1)
    for index, arguments in enumerate(again, start=2):
        many += _tool_turn(index, args=arguments)
    many += _calls_turn(5, [(f"c5-{index}", "terminal", arguments)
                            for index, arguments in enumerate(again)])
    many += _tool_turn(6)
    many += [_msg("user", "now", uid="ucur-cap")]
    capped = replace(settings, structure=replace(settings.structure, max_candidates=2, keep_turns=1))
    found = view.candidates([_system(), *many], _uids(many), many, _state(tmp_path), capped,
                            store=_store(tmp_path))
    assert [candidate.uid for candidate in found] == ["t2", "t3"], (
        "three superseded results, one pass, the two oldest offered")

    # The warm boundary is strict: exactly ``warm_turns`` turns since the last use is COLD.
    boundary = _store(tmp_path)
    boundary.renew("ep-0007")
    for _ in range(int(settings.warm_turns)):
        boundary.bump_turn()
    boundary_history = _tool_turn(1) + _recall_turn(2, "ep-0007") + _tool_turn(3) + _tool_turn(4) \
        + [_msg("user", "current", uid="ucur-boundary")]
    found = view.candidates([_system(), *boundary_history], _uids(boundary_history), boundary_history,
                            _state(tmp_path), settings, store=boundary)
    assert [candidate.uid for candidate in found] == ["tr2"], (
        "exactly warm_turns since the last use is cold; only fewer than warm_turns is warm")

    # A ``candidates()`` call with no store treats a recall as warm: nothing is offered, never cold.
    cold = _tool_turn(1) + _recall_turn(2, "ep-0009") + _tool_turn(3) + _tool_turn(4) \
        + [_msg("user", "current", uid="ucur-cold")]
    assert [candidate.uid for candidate in view.candidates(
        [_system(), *cold], _uids(cold), cold, _state(tmp_path), settings, store=_store(tmp_path))] == ["tr2"], (
        "a recall ref the store never saw is cold")
    assert view.candidates([_system(), *cold], _uids(cold), cold, _state(tmp_path), settings) == [], (
        "with no store to read, a recall is warm: never a candidate")


def test_judge_failure_changes_nothing(tmp_path, monkeypatch):
    """A judge failure writes its row and nothing else."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()
    _ask(engine, conversation)
    before = _state(tmp_path).state_note
    clock.advance(TTL)

    def failing(candidates, state_note):
        raise TimeoutError("judge timed out")

    engine._judge = lambda: failing
    request, served = _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["timeout"]
    after = _state(tmp_path)
    assert after.structured == {} and after.stubs == {} and after.blocks == []
    assert after.structured_turn == {}
    assert after.state_note == before
    assert not list(_store_path(tmp_path).glob("ep-*.md")), "nothing moved, so no episode was written"
    assert served == request, "a failed pass leaves the request as it was"

    # The same rule on the generic failure branch: a judge raising anything else changes nothing either.
    def exploding(candidates, state_note):
        raise RuntimeError("judge exploded")

    engine._judge = lambda: exploding
    clock.advance(TTL)
    _ask(engine, conversation)          # no pass: the timeout's own stamp bounds this window
    clock.advance(TTL)
    _, served = _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["timeout", "failed"]
    after = _state(tmp_path)
    assert after.structured == {} and after.stubs == {} and after.blocks == []
    assert after.structured_turn == {}
    assert after.state_note == before
    assert len(list(_store_path(tmp_path).glob("ep-*.md"))) == 0
    assert not [row for row in rows if row["stubbed"] or row["structured"]], "no decision survived the failure"

    # A judge that answers nothing is ``no_route`` (no call was made), not ``ok`` and not ``failed``.
    engine._judge = lambda: (lambda candidates, state_note: None)
    clock.advance(TTL)
    _ask(engine, conversation)          # the failure's own stamp bounds this window
    clock.advance(TTL)
    _, served = _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["timeout", "failed", "no_route"]
    assert rows[-1]["candidates"] == 1 and rows[-1]["kept"] == 1, "nothing moved, the candidate is kept"
    assert served == request
    after = _state(tmp_path)
    assert after.structured == {} and after.stubs == {} and after.blocks == [] and after.structured_turn == {}


def test_unknown_decision_ids_ignored(tmp_path, monkeypatch):
    """Only decisions naming a candidate id are applied, and only ``structure``/``marker_only``."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _superseded_history()
    _ask(engine, conversation)

    def judge(candidates, state_note):
        return {"decisions": [{"id": "c99", "action": "structure", "summary": "invented"},
                              {"id": candidates[0].id, "action": "keep", "summary": "keep it"}]}

    engine._judge = lambda: judge
    clock.advance(TTL)
    _, served = _ask(engine, conversation)

    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert rows[0]["kept"] == 1 and rows[0]["structured"] == 0 and rows[0]["stubbed"] == 0
    assert _state(tmp_path).structured == {}
    assert not list(_store_path(tmp_path).glob("ep-*.md"))
    assert "invented" not in json.dumps(served)


def test_raw_size_drives_should_compress_info(tmp_path, monkeypatch):
    """The backstop judges the larger of the provider figure and the raw history's rough size."""
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch, ttl=None, threshold=20_000)
    conversation = _superseded_history(turns=8, size=20_000)

    assert engine.should_compress_info(1_000)[0] is False, "before any allowed request there is no estimate"
    _ask(engine, conversation)
    assert engine._rough_tokens > engine.threshold_tokens, "the raw history is over the threshold"
    assert engine.should_compress_info(1_000)[0] is True, "the raw size, not the served figure, drives it"


def test_structured_compress_is_builtin_and_resets_estimate(tmp_path, monkeypatch):
    """In structured mode ``compress`` is the built-in handoff, and the estimate resets to 0."""
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch, ttl=None, threshold=20_000, tail_budget=600)
    big = _superseded_history(turns=8, size=20_000)
    _ask(engine, big)
    assert engine.should_compress_info(1_000)[0] is True

    transcript = _ctxpkg_transcript()
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(engine, "_generate_summary", lambda *a, **kw: "SUMMARY BODY")
        out = engine.compress(transcript, current_tokens=150_000, force=True)

    assert out is not transcript
    assert len(out) < len(transcript), "the middle is gone"
    assert "SUMMARY BODY" in json.dumps(out), "the built-in summariser is the one who ran"
    assert not [message for message in out if _text(message).startswith(VIEW_PREFIX)]
    assert _store(tmp_path).blocks() == [], "no ctxpkg compaction block in structured mode"
    assert not list(_store_path(tmp_path).glob("ep-*.md")), "no episode was cut"
    assert engine._rough_tokens == 0
    assert engine.should_compress_info(1_000)[0] is False, "the estimate reset; the provider figure decides"

    # A new session and a session reset clear the estimate too: the next turn starts from the provider figure.
    engine._rough_tokens = 999_999
    engine.on_session_start(SESSION)
    assert engine._rough_tokens == 0
    assert engine.should_compress_info(1_000)[0] is False, "on_session_start resets the estimate"
    engine._rough_tokens = 999_999
    engine.on_session_reset()
    assert engine._rough_tokens == 0
    assert engine.should_compress_info(1_000)[0] is False, "on_session_reset resets the estimate"


def test_not_allowed_writes_nothing(tmp_path, monkeypatch):
    """A child, a fork and an empty session id write no view file, log or stamp."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch, allowed=False)
    conversation = _superseded_history()

    for _ in range(3):
        assert _ask(engine, conversation)[1] is None
        clock.advance(TTL)
    engine.on_turn_complete([dict(message) for message in conversation])
    assert "ep-0001" in engine.handle_tool_call("recall", {"ref": "ep-0001"})

    root = _store_path(tmp_path)
    for name in ("view.json", "passes.jsonl", "recalls.jsonl", "reruns.jsonl"):
        assert not (root / name).exists(), name
    assert (root / "warm.json").exists(), "today's warmth update still runs for every engine"
    assert engine._rough_tokens == 0


def test_not_allowed_keeps_mixin_compaction(tmp_path, monkeypatch):
    """A never-allowed engine keeps the mixin's compaction and its own trigger."""
    from agent import auxiliary_client
    import agent.context_compressor as compressor_module

    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch, allowed=False, tail_budget=600)
    big = _superseded_history(turns=8, size=20_000)
    _ask(engine, big)
    assert engine.should_compress_info(1_000)[0] is False, "the raw estimate is never recorded off the view"

    def fake(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message={"content": "ep-0001: an old turn"})])

    monkeypatch.setattr(auxiliary_client, "call_llm", fake)
    monkeypatch.setattr(compressor_module, "call_llm", fake)
    transcript = _ctxpkg_transcript()
    out = engine.compress(transcript, current_tokens=150_000, force=True)

    assert len(out) < len(transcript), "the mixin compacted"
    assert [message for message in out if _text(message).startswith(VIEW_PREFIX)], (
        "the mixin's reference block is still what compaction emits")
    assert _store(tmp_path).blocks(), "the mixin records the block for the next process"


def test_reruns_skip_first_request(tmp_path, monkeypatch):
    """The first allowed request only records the call set; a later repeat of a stub's call is a rerun."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    _seed(tmp_path,
          stubs={"t2": "[ep-0001 · superseded · 1k chars · x · ids: none · recall(\"ep-0001\") for the full text]"},
          structured={"t2": "ep-0001"},
          stub_calls={"ep-0001": ["terminal", json.dumps(json.loads(SHARED_ARGS), sort_keys=True)]},
          structured_turn={"ep-0001": 1},
          view_turn=2)

    conversation = _superseded_history()
    _ask(engine, conversation)
    assert _rows(tmp_path, "reruns.jsonl") == [], "the stubbed result's own earlier call is not a rerun"

    # The same tool name with different arguments is not a rerun: the match is the whole call.
    clock.advance(1)
    other_args = _msg("assistant", None, uid="a98",
                      tool_calls=[_call("c98", "terminal", json.dumps({"command": "ls -la"}))])
    _ask(engine, conversation[:-1] + [other_args] + conversation[-1:])
    assert _rows(tmp_path, "reruns.jsonl") == [], "name alone is not a match; the arguments must be equal too"

    clock.advance(1)
    rerun = _msg("assistant", None, uid="a99",
                 tool_calls=[_call("c99", "terminal", json.dumps(json.loads(SHARED_ARGS)))])
    grown = conversation[:-1] + [rerun] + conversation[-1:]
    _ask(engine, grown)

    rows = _rows(tmp_path, "reruns.jsonl")
    assert len(rows) == 1
    assert rows[0]["ref"] == "ep-0001"
    assert rows[0]["turns_since_structured"] == 1


def test_fork_turns_not_counted(tmp_path, monkeypatch):
    """``view_turn`` counts only turns served while the view is allowed; the store's counter counts all."""
    _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    messages = _superseded_history()

    _ask(engine, messages)
    engine.on_turn_complete([dict(message) for message in messages])
    engine.on_turn_complete([dict(message) for message in messages])
    engine.view_allowed = False
    engine.on_turn_complete([dict(message) for message in messages])

    state = _state(tmp_path)
    assert state.view_turn == 2, "the fork's turn did not count"
    assert _store(tmp_path).current_turn() == 3
    assert state.seen["u1"][0] == 1, "the first seen turn is the first allowed turn"
    assert all(entry[1] > 0 for entry in state.seen.values()), "seen carries an epoch too"


def test_host_sets_engine_attributes(tmp_path, monkeypatch):
    """The host hands the engine its tier, uids, prefill count and ``view_allowed`` before the hook."""
    class _RecordingEngine:
        name = "recording"

        def __init__(self):
            self.seen = None

        def select_context(self, request, **kwargs):
            self.seen = {"ttl": self.cache_ttl_seconds, "uids": self.request_uids,
                         "prefill": self.prefill_count, "allowed": self.view_allowed}

    def ask(**overrides):
        engine = _RecordingEngine()
        agent = _fake_agent(engine, **overrides)
        request = [_system(), {"role": "user", "content": "hi"}]
        _apply_context_engine_selection(agent, request, request[1:], None, logger=logging.getLogger("t"))
        return engine.seen

    assert ask(_cache_ttl="1h")["ttl"] == 3600
    assert ask(_cache_ttl="5m")["ttl"] == 300
    assert ask(_cache_ttl=None)["ttl"] is None
    assert ask(_use_prompt_caching=False)["ttl"] is None
    # The route clamps 1h to 5m (a Qwen model): the engine gets the tier the cache really holds.
    assert ask(_cache_ttl="1h", model="qwen-plus")["ttl"] == 300
    # ...and the provider alone clamps too: an Alibaba-family route drops 1h even for a model name that
    # is not Qwen, so the engine gets 300, never the requested hour (it would wait 55 minutes too long).
    assert ask(_cache_ttl="1h", provider="alibaba", model="claude-opus-5-5")["ttl"] == 300

    seen = ask(_request_message_uids=["a", "b"], prefill_messages=[{"role": "user", "content": "p"}])
    assert seen["uids"] == ["a", "b"]
    assert seen["prefill"] == 1
    # ``engine.request_uids`` is a copy: editing the agent's own list afterwards leaves the engine's list
    # alone, so the engine can never write into the request it was handed.
    copied_agent = _fake_agent(_RecordingEngine(), _request_message_uids=["a", "b"])
    request = [_system(), {"role": "user", "content": "hi"}]
    _apply_context_engine_selection(copied_agent, request, request[1:], None,
                                    logger=logging.getLogger("t"))
    seen_uids = copied_agent.context_compressor.seen["uids"]
    copied_agent._request_message_uids.append("later")
    assert seen_uids == ["a", "b"], "the engine's uid list is a copy of the agent's, not the same list"
    assert seen_uids is not copied_agent._request_message_uids
    agent = _fake_agent(_RecordingEngine())
    del agent._request_message_uids
    request = [_system(), {"role": "user", "content": "hi"}]
    engine = agent.context_compressor
    _apply_context_engine_selection(agent, request, request[1:], None, logger=logging.getLogger("t"))
    assert engine.seen["uids"] is None

    assert ask()["allowed"] is True
    assert ask(platform="subagent")["allowed"] is False
    assert ask(_context_engine_tool_names=set())["allowed"] is False
    assert ask(_memory_write_context="background_review")["allowed"] is False
    assert ask(_persist_disabled=True)["allowed"] is False
    assert ask(session_id="")["allowed"] is False

    # A third-party engine that refuses the four names (no __dict__: __slots__) must be served exactly as
    # at base: no exception out of request assembly, the hook still runs, the request stays as assembled.
    class _SlotsEngine:
        __slots__ = ("ran",)
        name = "slots"

        def __init__(self):
            self.ran = False

        def select_context(self, request, **kwargs):
            self.ran = True

    agent = _fake_agent(_SlotsEngine())
    request = [_system(), {"role": "user", "content": "hi"}]
    assert _apply_context_engine_selection(
        agent, request, request[1:], None, logger=logging.getLogger("t"),
    ) is request, "an engine that refuses the attributes is fail-open, not an AttributeError"
    assert agent.context_compressor.ran, "the hook still ran"

    # A failed derivation hands over the safe values: no tier, no uids, no prefill, NOT allowed.
    class _RaisingAgent:
        def __init__(self, engine):
            self.context_compressor = engine
            self.session_id = SESSION
            self.model = "claude-opus-5-5"
            self.provider = "anthropic"
            self._use_prompt_caching = True
            self._request_message_uids = ["a", "b"]
            self.prefill_messages = [{"role": "user", "content": "p"}]
            self._context_engine_tool_names = {"recall"}
            self.platform = "discord"
            self._memory_write_context = "foreground"
            self._persist_disabled = False

        @property
        def _cache_ttl(self):
            raise RuntimeError("cache tier unavailable")

    recording = _RecordingEngine()
    agent = _RaisingAgent(recording)
    request = [_system(), {"role": "user", "content": "hi"}]
    assert _apply_context_engine_selection(
        agent, request, request[1:], None, logger=logging.getLogger("t"),
    ) is request
    assert recording.seen == {"ttl": None, "uids": None, "prefill": 0, "allowed": False}, (
        "an engine never gets a view with a half-derived set of attributes")

    # An engine that refuses only the FIRST name still gets the safe value for every other one: the
    # fallback writes each name in its own guard, so one refused attribute does not cost the rest.
    class _PartialRefusalEngine:
        name = "partial"
        prefill_count = 7
        view_allowed = True

        def __init__(self):
            self.seen = None

        @property
        def cache_ttl_seconds(self):
            return 3600  # readable, but the setter is refused

        def select_context(self, request, **kwargs):
            self.seen = {"prefill": self.prefill_count, "allowed": self.view_allowed}

    partial = _PartialRefusalEngine()
    partial_agent = _fake_agent(partial, prefill_messages=[{"role": "user", "content": "p"}])
    request = [_system(), {"role": "user", "content": "hi"}]
    assert _apply_context_engine_selection(
        partial_agent, request, request[1:], None, logger=logging.getLogger("t"),
    ) is request
    assert partial.seen == {"prefill": 0, "allowed": False}, (
        "a refused first name must not leave the names after it at the engine's own values")

    # A validating engine raises AttributeError for one name but ValueError or RuntimeError for others:
    # the plan's fallback is "safe value on ANY error", so each refusal, whatever its type and wherever
    # it sits, is contained by its own guard and every name after it still takes its safe value.
    class _ValidatingEngine:
        name = "validating"
        cache_ttl_seconds = 3600
        request_uids = ["stale"]
        prefill_count = 7
        view_allowed = True

        def __init__(self, refusals=()):
            self.__dict__["refusals"] = dict(refusals)
            self.__dict__["seen"] = None

        def __setattr__(self, name, value):
            error = self.__dict__["refusals"].get(name)
            if error is not None:
                raise error
            self.__dict__[name] = value

        def select_context(self, request, **kwargs):
            self.seen = {"ttl": self.cache_ttl_seconds, "uids": self.request_uids,
                         "prefill": self.prefill_count, "allowed": self.view_allowed}

    # Every name refused, the first with AttributeError (a read-only property's own error) and the two
    # after it with ValueError and RuntimeError: the hook still runs, assembly still returns the request,
    # and the last name, after all three refusals, still reads its safe value.
    strict = _ValidatingEngine({
        "cache_ttl_seconds": AttributeError("read-only"),
        "request_uids": ValueError("this engine validates its uid list"),
        "prefill_count": RuntimeError("this engine validates its prefill count"),
    })
    request = [_system(), {"role": "user", "content": "hi"}]
    assert _apply_context_engine_selection(
        _fake_agent(strict), request, request[1:], None, logger=logging.getLogger("t"),
    ) is request, "a ValueError or RuntimeError out of a setter is fail-open too, not only AttributeError"
    assert strict.seen == {"ttl": 3600, "uids": ["stale"], "prefill": 7, "allowed": False}, (
        "the hook ran, and the one name after every refusal took its safe value")

    # Only a later name refused, with RuntimeError: the tier and the uids are rewritten to their safe
    # values as well, so a validating setter costs nothing but the name it refuses.
    later = _ValidatingEngine({"prefill_count": RuntimeError("this engine validates its prefill count")})
    request = [_system(), {"role": "user", "content": "hi"}]
    assert _apply_context_engine_selection(
        _fake_agent(later), request, request[1:], None, logger=logging.getLogger("t"),
    ) is request
    assert later.seen == {"ttl": None, "uids": None, "prefill": 7, "allowed": False}, (
        "a non-AttributeError on a later name leaves the names before it safe, not half-derived")
