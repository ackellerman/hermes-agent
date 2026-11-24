"""The ctxpkg flash judge and the ``done``/``old`` candidates (plan structured-context, step 5).

Step 4 served the view but moved nothing: its judge, ``keep_all``, kept every candidate. This step adds the
two whole-turn candidate kinds (``done``, ``old``) and the judge that reasons over them, one
``auxiliary.context_structure`` call per pass. The call is monkeypatched throughout: these tests never touch
a network. The store root lives under ``tmp_path`` (``HERMES_HOME`` monkeypatched), the clock is injected.

Two carried step-4 rules live here: a pass that structures a *turn* writes its marker line, appends one block
to ``view.json`` and registers that block with the cache-anchor registry, with every uid of the turn recorded
in ``structured``; and the pass row carries the judge's real ``flash_in``/``flash_out``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

# ``call_llm`` is patched here, before any module that imports it lazily can capture the real one.
from agent import auxiliary_client
import hermes_cli.config as config_module
import plugins.context_engine.ctxpkg.store as store_module
from plugins.context_engine.ctxpkg import CtxpkgEngine

SESSION = "20260101_120000_abcdef12"
TTL = 300
KEEP_TURNS = 3
OLD_MINUTES = 120
OLD_SECONDS = OLD_MINUTES * 60
SUMMARY_MAX = 300
STATE_NOTE_CHARS = 1200
CANDIDATE_CHARS = 4000
MIN_STUB_CHARS = 200
CLOCK = 1_000_000.0
FLASH_IN = 1234
FLASH_OUT = 56
ROUTE = {
    "provider": "ollama-cloud",
    "model": "deepseek-v4.1-flash:cloud",
    "reasoning_effort": "medium",
    "timeout": 45,
}


# ── fixtures and helpers ───────────────────────────────────────────────────


def _config(*, route=ROUTE):
    """The pinned config: ``ctxpkg`` in structured mode, plus a ``context_structure`` route unless dropped."""
    config = {
        "ctxpkg": {
            "warm_turns": 8,
            "max_episode_chars": 60_000,
            "summary_max_chars": SUMMARY_MAX,
            "view": "structured",
            "structure": {
                "keep_turns": KEEP_TURNS,
                "old_after_minutes": OLD_MINUTES,
                "min_stub_chars": MIN_STUB_CHARS,
                "max_candidates": 40,
                "candidate_chars": CANDIDATE_CHARS,
                "state_note_chars": STATE_NOTE_CHARS,
                "timeout_seconds": 45,
            },
        },
    }
    if route:
        config["auxiliary"] = {"context_structure": dict(route)}
    return config


@pytest.fixture(autouse=True)
def _ctxpkg_config(monkeypatch):
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: _config())
    # ``session_source_for`` reads the session env first; a stray value would decide ``view_allowed``.
    for name in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(name, raising=False)


class _Calls:
    """The monkeypatched ``call_llm``: one record per call, a queue of canned replies, a canned usage."""

    def __init__(self):
        self.records: list[dict] = []
        self.replies: list[str] = []
        self.usage = (FLASH_IN, FLASH_OUT)
        self.error: BaseException | None = None

    def __call__(self, *args, **kwargs):
        self.records.append({"args": args, **kwargs})
        if self.error is not None:
            raise self.error
        text = self.replies.pop(0) if self.replies else json.dumps({"decisions": []})
        prompt, completion = self.usage
        return SimpleNamespace(choices=[SimpleNamespace(message={"content": text})],
                                usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))

    @property
    def calls(self) -> int:
        return len(self.records)


@pytest.fixture(autouse=True)
def calls(monkeypatch):
    """No test reaches a provider: every ``call_llm`` goes through the recorder."""
    recorder = _Calls()
    monkeypatch.setattr(auxiliary_client, "call_llm", recorder)
    return recorder


class _Clock:
    """An injected clock: ``view.now_epoch`` is monkeypatched to this callable."""

    def __init__(self, start: float = CLOCK):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += float(seconds)
        return self.t


def _view():
    """The module under test, imported lazily so the red run fails per test, not at collection."""
    from plugins.context_engine.ctxpkg import view

    return view


def _clock(monkeypatch, start: float = CLOCK) -> _Clock:
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


def _turn(index, *, command=None, result=None, name="terminal", result_uid=None, size=60):
    """One turn: the request, one tool call with its result, then the answer."""
    arguments = json.dumps({"command": command if command is not None else f"run the step {index} check"})
    uid = f"t{index}" if result_uid is None else result_uid
    return [
        _msg("user", f"turn {index}: please run it", uid=f"u{index}"),
        _msg("assistant", None, uid=f"a{index}", tool_calls=[_call(f"c{index}", name, arguments)]),
        _msg("tool", result if result is not None else "y" * size, uid=uid, tool_call_id=f"c{index}"),
        _msg("assistant", f"done {index}", uid=f"z{index}"),
    ]


def _done_turn(index):
    """A turn whose terminal command marks a step done: a ``done`` signal, no age needed."""
    return _turn(index, command=f"work step {index} --done --sha 27a26263f52d")


def _history(*turns, pad_from: int = 90, pads=KEEP_TURNS + 1, current=None, head=None):
    """The head turn, *turns*, quiet padding turns, then the current turn (the last three are protected)."""
    rows = _turn(1) if head is None else head
    for turn in turns:
        rows += turn
    for index in range(pad_from, pad_from + pads):
        rows += _turn(index)
    return rows + (current if current is not None
                   else [_msg("user", "now the current question", uid="ucur")])


def _two_done_history():
    return _history(_done_turn(2), _done_turn(3))


def _uids(conversation):
    return [message.get("message_uid") for message in conversation]


def _system(text="You are a driver."):
    return {"role": "system", "content": text}


def _engine(tmp_path, monkeypatch, *, allowed=True, session=SESSION, ttl=TTL, threshold=100_000):
    """An engine over a throwaway home, wired only through public constructor attributes."""
    monkeypatch.setattr(store_module, "get_hermes_home", lambda: tmp_path)
    engine = CtxpkgEngine(model="test-model")
    engine.quiet_mode = True
    engine.protect_first_n = 3
    engine.protect_last_n = 20
    engine._config_context_length = 200_000
    engine.context_length = 200_000
    engine.threshold_tokens = threshold
    engine._session_id = session
    engine.view_allowed = allowed
    engine.cache_ttl_seconds = ttl
    return engine


def _store_path(tmp_path, session=SESSION):
    return Path(tmp_path) / "ctxpkg" / session


def _store(tmp_path, session=SESSION):
    return store_module.EpisodeStore(_store_path(tmp_path, session))


def _seed(tmp_path, session=SESSION, **fields):
    """Write ``view.json`` for the session, as a previous process left it."""
    state = _view().ViewState.load(_store(tmp_path, session))
    for name, value in fields.items():
        setattr(state, name, value)
    state.save(_store(tmp_path, session))
    return state


def _state(tmp_path, session=SESSION):
    return _view().ViewState.load(_store(tmp_path, session))


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


def _text(message):
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def _candidates(view, count=3, *, kind="done", ids=None):
    return [view.Candidate(id=f"c{index}", uid=f"t{index}", kind=kind,
                           text=f"turn {index} text", chars=len(f"turn {index} text"),
                           ids=list(ids or []), reason=kind)
            for index in range(1, count + 1)]


def _next_pass(clock, engine, conversation):
    """Walk the trigger: the first request runs the pass, the second one follows it."""
    clock.advance(TTL)
    _ask(engine, conversation)
    clock.advance(TTL)
    return _ask(engine, conversation)


# ── the test list ──────────────────────────────────────────────────────────


def test_done_candidate_signals(tmp_path, monkeypatch):
    """Each of the four ``done`` signals makes a whole turn a candidate, and ``done`` outranks ``old``."""
    _clock(monkeypatch)
    view = _view()
    settings = view._settings()
    store = _store(tmp_path)

    step_done = _done_turn(2)
    commit = _turn(3, command="git commit -q -m 'structured-context step 5'",
                   result="[plan/structured-context 27a26263f5] structured-context step 5: the flash judge")
    row = _turn(4, command="restart-register add --kind public-push --by falcor",
                result="R-101 public-push  pending")
    notice = [
        _msg("user", "[ASYNC DELEGATION BATCH COMPLETE] 3 children finished", uid="u5"),
        _msg("assistant", "all three are home", uid="z5"),
    ]
    finished = _turn(6, command="work finish --note 'step 4 landed'")
    quiet = _turn(7, command="rg -n 'def test_' tests/")
    rows = _history(step_done, commit, row, notice, finished, quiet)

    found = view.candidates([_system(), *rows], _uids(rows), rows, view.ViewState.load(store), settings,
                            store=store, now=CLOCK)
    assert [(candidate.turn_uids[0], candidate.kind) for candidate in found] == [
        ("u2", "done"), ("u3", "done"), ("u4", "done"), ("u5", "done"), ("u6", "done")], (
        "the step-done command, the commit line, the registry row, the batch notice and `work finish` are the"
        " five signals; a quiet turn and every protected turn stay out")
    assert [candidate.id for candidate in found] == ["c1", "c2", "c3", "c4", "c5"], (
        "candidates are numbered oldest first")
    assert all(candidate.is_turn and candidate.chars == len(candidate.text) for candidate in found)
    assert found[0].turn_uids == ("u2", "a2", "t2", "z2"), "a turn candidate carries every uid of its turn"
    assert "work step 2 --done" in found[0].text, "the episode text keeps the command that proves the job done"
    assert "27a26263f5" in found[1].ids, "the commit sha is copied out by script, never by the model"
    assert "R-101" in found[2].ids

    # A turn that is both done and past old_after_minutes is done: the priority the plan names.
    aged = {uid: [1, CLOCK - (OLD_SECONDS + 600)] for uid in found[0].turn_uids}
    state = _seed(tmp_path, seen=aged)
    found = view.candidates([_system(), *rows], _uids(rows), rows, state, settings, store=store, now=CLOCK)
    assert {candidate.turn_uids[0]: candidate.kind for candidate in found}["u2"] == "done", (
        "done outranks old when a turn reads as both")

    # The batch notice is a signal only when a later turn follows it: nothing to notice otherwise.
    assert view._done_signal("[ASYNC DELEGATION BATCH COMPLETE] 3 children finished", False) == ""
    assert view._done_signal("[ASYNC DELEGATION BATCH COMPLETE] 3 children finished", True) != ""
    # A registry row needs the command AND the id it named: an id alone (a plan's prose) is not a done turn.
    assert view._done_signal("the plan lists R-101 and R-102", True) == ""
    assert view._done_signal("restart-register add --kind deploy", True) == ""
    assert view._done_signal("restart-register add --kind deploy\nR-101 deploy", True) != ""


def test_old_candidate_needs_age_and_no_recent_ids(tmp_path, monkeypatch):
    """``old`` needs ``old_after_minutes`` since first seen AND no id of the turn in the last ``keep_turns`` turns."""
    _clock(monkeypatch)
    view = _view()
    settings = view._settings()
    store = _store(tmp_path)

    old_turn = _turn(2, result="a plain result sa-77-deadbeef " + "y" * 40)
    quiet_rows = _turn(1) + old_turn + _turn(3) + _turn(4) + _turn(5) \
        + [_msg("user", "now the current question", uid="ucur")]
    # The same history with the id named again inside keep_turns (t5 is protected, so the turn is in use).
    naming_rows = _turn(1) + old_turn + _turn(3) + _turn(4) \
        + _turn(5, result="a later turn says sa-77-deadbeef again") \
        + [_msg("user", "now the current question", uid="ucur")]

    def found(state, rows):
        return view.candidates([_system(), *rows], _uids(rows), rows, state, settings, store=store, now=CLOCK)

    # Exactly at the bound is old; one second short is not (the band has both edges in).
    assert [(c.turn_uids[0], c.kind) for c in found(
        _seed(tmp_path, seen={"u2": [1, CLOCK - OLD_SECONDS]}), quiet_rows)] == [("u2", "old")]
    assert found(_seed(tmp_path, seen={"u2": [1, CLOCK - OLD_SECONDS + 1]}), quiet_rows) == [], (
        "one second under old_after_minutes is not old")
    # No first-seen epoch: no age to judge, so never old (a turn the view never saw served).
    assert found(_seed(tmp_path, seen={}), quiet_rows) == [], "a turn with no first-seen epoch is never old"
    # An id of the turn still named by the last keep_turns turns keeps it: it is in use.
    assert found(_seed(tmp_path, seen={"u2": [1, CLOCK - (OLD_SECONDS + 600)]}), naming_rows) == [], (
        "an id named by a turn inside keep_turns is still in use, so the turn is never old")
    # ...and the same turn is old once the recent turns stop naming it.
    state = _seed(tmp_path, seen={"u2": [1, CLOCK - (OLD_SECONDS + 600)]})
    found_quiet = found(state, quiet_rows)
    assert [candidate.kind for candidate in found_quiet] == ["old"]
    assert "minutes since the turn was first seen" in found_quiet[0].reason


def test_never_candidates_hold_for_turns(tmp_path, monkeypatch):
    """Every step-4 never-candidate rule keeps a ``done`` turn out too; lift one and that turn comes back."""
    _clock(monkeypatch)
    view = _view()
    settings = view._settings()
    store = _store(tmp_path)

    head = _done_turn(1)                          # the head turn carries a signal and must still stay out
    free = _done_turn(2)                         # the only one that stays free
    uidless = _done_turn(3)
    del uidless[2]["message_uid"]                # a fabricated recovery result inside the turn
    already = _done_turn(4)
    named = _turn(5, command="work step 5 --done", result="result sa-99-cafebabe")
    rows = _history(free, uidless, already, named, current=_done_turn(6), head=head)

    state = _seed(tmp_path, structured={"z4": "ep-0009"},
                  state_note="[ctxpkg state]\nwaiting on sa-99-cafebabe")
    request = [_system(), *rows]
    found = view.candidates(request, _uids(rows), rows, state, settings, store=store, now=CLOCK)
    assert [candidate.turn_uids[0] for candidate in found] == ["u2"], (
        "head, current turn, keep_turns, a uid-None row, an already structured turn and a state-note id"
        " all stay out")

    # Each rule on its own: clearing the state note frees the turn whose id it named...
    free_note = _seed(tmp_path, structured={"z4": "ep-0009"}, state_note="")
    found = view.candidates(request, _uids(rows), rows, free_note, settings, store=store, now=CLOCK)
    assert "u5" in [candidate.turn_uids[0] for candidate in found]
    # ...dropping the structured uid frees that turn...
    unstuck = _seed(tmp_path, structured={}, state_note="")
    found = view.candidates(request, _uids(rows), rows, unstuck, settings, store=store, now=CLOCK)
    assert "u4" in [candidate.turn_uids[0] for candidate in found]
    # ...and restoring the result's uid frees the uid-None turn.
    healed = [({**message, "message_uid": "t3"} if message.get("tool_call_id") == "c3" else dict(message))
              for message in rows]
    found = view.candidates([_system(), *healed], _uids(healed), healed, unstuck, settings, store=store, now=CLOCK)
    assert "u3" in [candidate.turn_uids[0] for candidate in found]
    # With ``keep_turns: 0`` the turn in flight is still protected: only the current-turn rule can do it.
    settings_no_keep = replace(settings, structure=replace(settings.structure, keep_turns=0))
    found = view.candidates(request, _uids(rows), rows, unstuck, settings_no_keep, store=store, now=CLOCK)
    assert "u6" not in [candidate.turn_uids[0] for candidate in found], (
        "the turn in flight is never a candidate, keep_turns or not")


def test_no_route_makes_no_call(tmp_path, monkeypatch, calls):
    """No ``auxiliary.context_structure.provider``: no route, no call, no reply."""
    view = _view()
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: _config(route={}))
    found = _candidates(view)
    assert view.flash_judge(found, "", view._settings()) is None
    assert calls.calls == 0, "a pass with no route makes no call"
    # A blank provider is no route either, and the judge reads the config itself when no settings are handed in.
    monkeypatch.setattr(config_module, "load_config_readonly",
                        lambda: _config(route={"provider": "   ", "model": "x"}))
    assert view.flash_judge(found, "", None) is None
    assert calls.calls == 0

    # Through a pass: a judge that never called is `no_route` with no usage; `failed` is for a call that was made.
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _two_done_history()
    _ask(engine, conversation)
    clock.advance(TTL)
    _ask(engine, conversation)
    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["no_route"], "no route: the one row says no call was made"
    assert rows[0]["candidates"] == 2 and rows[0]["kept"] == 2
    assert (rows[0]["flash_in"], rows[0]["flash_out"]) == (0, 0), "no call means no usage on the row"
    assert calls.calls == 0, "no route through a pass: the counter stays at zero"


def test_auto_provider_makes_no_call(tmp_path, monkeypatch, calls):
    """``provider: auto`` routes to the main model, which the engine refuses: no call, one ``no_route`` row."""
    clock = _clock(monkeypatch)
    monkeypatch.setattr(config_module, "load_config_readonly", lambda: _config(route={"provider": "auto"}))
    engine = _engine(tmp_path, monkeypatch)
    conversation = _two_done_history()

    _ask(engine, conversation)
    clock.advance(TTL)
    _request, served = _ask(engine, conversation)

    assert calls.calls == 0, "an ``auto`` route would reach Opus: the judge must not call"
    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows] == ["no_route"]
    assert rows[0]["candidates"] == 2 and rows[0]["kept"] == 2, "the candidates are kept, nothing moved"
    state = _state(tmp_path)
    assert state.structured == {} and state.stubs == {} and state.blocks == []
    assert served is not None and _text(served[1]).startswith("turn 1"), "the view is still served, nothing moves"


def test_one_call_per_pass(tmp_path, monkeypatch, calls):
    """One pass, one call, one row with the real usage; the turn branch writes its block and anchor."""
    clock = _clock(monkeypatch)
    view = _view()
    engine = _engine(tmp_path, monkeypatch)
    anchored = []
    monkeypatch.setattr(view, "register_anchor", anchored.append)
    calls.replies = [json.dumps({
        "decisions": [{"id": "c1", "action": "structure", "summary": "step four landed"},
                      {"id": "c2", "action": "structure", "summary": "the row was filed"}],
        "state_note": "goal: step five; next: the tests",
    })]
    conversation = _two_done_history()

    _ask(engine, conversation)
    clock.advance(TTL)
    _request, served = _ask(engine, conversation)

    assert calls.calls == 1, "one call per pass, whatever the candidate count"
    call = calls.records[0]
    assert call["args"] == (), "the call is keyword-only"
    assert call["task"] == "context_structure"
    assert call["timeout"] == 45, "the call is bounded by ctxpkg.structure.timeout_seconds"
    assert call["extra_body"] == {"response_format": {"type": "json_object"}}
    assert [message["role"] for message in call["messages"]] == ["system", "user"]
    system = call["messages"][0]["content"].lower()
    assert "keep" in system and "structure" in system and "marker_only" in system
    assert "never invent" in system, "the judge is told never to invent an id"
    payload = json.loads(call["messages"][1]["content"])
    assert [candidate["id"] for candidate in payload["candidates"]] == ["c1", "c2"]
    assert {candidate["kind"] for candidate in payload["candidates"]} == {"done"}
    assert payload["state_note"] == ""
    assert "turn 92" in payload["recent"] and "turn 90" not in payload["recent"], (
        "``recent`` is the last keep_turns turns")
    assert payload["candidates"][0]["text"].startswith("user: turn 2"), (
        "a candidate carries its turn text, role-tagged, tool calls included")
    for candidate in payload["candidates"]:
        assert set(candidate) == {"id", "kind", "reason", "age_minutes", "chars", "text"}
        assert candidate["chars"] == len(candidate["text"])

    row = _rows(tmp_path, "passes.jsonl")[0]
    assert row["status"] == "ok"
    assert (row["candidates"], row["kept"], row["structured"]) == (2, 0, 2)
    assert row["refs"] == ["ep-0001", "ep-0002"]
    assert (row["flash_in"], row["flash_out"]) == (FLASH_IN, FLASH_OUT), (
        "the pass row carries the judge's own usage, not zeros")
    assert row["latency_s"] >= 0 and row["ttl_s"] == TTL and row["idle_s"] == TTL

    state = _state(tmp_path)
    assert len(state.blocks) == 1, "a pass that structured turns appends one block"
    assert state.blocks[-1] == anchored[-1], "the newest block is registered with the cache anchor"
    assert "step four landed" in state.blocks[-1] and "ids:" in state.blocks[-1]
    assert all(uid in state.structured for uid in ("u2", "a2", "t2", "z2", "u3", "a3", "t3", "z3")), (
        "every uid of a structured turn is recorded, so the turn is skipped whole")
    assert state.state_note == "goal: step five; next: the tests"
    assert (_store_path(tmp_path) / "ep-0001.md").read_text(encoding="utf-8").startswith("user: turn 2")
    texts = [_text(message) for message in served]
    assert "turn 2: please run it" not in texts and "turn 3: please run it" not in texts
    assert any(text.startswith("[ctxpkg references ep-0001..ep-0002]") for text in texts)

    # Nothing left to offer: the next gap's pass makes no second call and records ``no_candidates``.
    _next_pass(clock, engine, conversation)
    assert calls.calls == 1, "a pass with no candidates calls no judge"
    assert [row["status"] for row in _rows(tmp_path, "passes.jsonl")] == ["ok", "no_candidates"]

    # The two payload bounds step 5 names, on their own history: a candidate's text arrives cut to
    # ``candidate_chars`` with BOTH ends kept, and ``recent`` arrives cut to 12000 chars with the newest
    # turn's tail kept. Each is read off the payload the judge was handed, never off a helper.
    other = "20260101_120000_abcdef34"
    long_engine = _engine(tmp_path, monkeypatch, session=other)
    long_turn = _turn(2, command="work step 2 --done --sha 27a26263f52d",
                      result="FIRST-LINE-OF-THE-RESULT " + "y" * (CANDIDATE_CHARS * 2)
                      + " LAST-LINE-OF-THE-RESULT")
    long_rows = _turn(1) + long_turn
    for index in range(90, 93):
        long_rows += _turn(index, result="z" * 6000)
    long_rows += [_msg("user", "now the current question", uid="ucur")]

    _ask(long_engine, long_rows)
    clock.advance(TTL)
    _ask(long_engine, long_rows)
    assert calls.calls == 2, "the second session's expired gap runs its own pass"
    long_payload = json.loads(calls.records[-1]["messages"][1]["content"])
    cut = {candidate["id"]: candidate for candidate in long_payload["candidates"]}["c1"]["text"]
    assert len(cut) == CANDIDATE_CHARS, (
        "a candidate's text is cut to ctxpkg.structure.candidate_chars, not sent whole")
    assert view.CUT_MARK in cut, "the cut is marked where it happened"
    assert cut.startswith("user: turn 2") and cut.rstrip("\n").endswith("assistant: done 2"), (
        "the cut keeps the item's first line AND its last line: never the head only")
    assert len(long_payload["recent"]) == 12_000, (
        "``recent`` is cut to 12000 chars, however long the last keep_turns turns are")
    assert long_payload["recent"].rstrip("\n").endswith("now the current question"), (
        "the cut keeps the newest turn's tail")


def test_reply_with_fences_parses(tmp_path, monkeypatch, calls):
    """A fenced JSON reply is the reply: the fences come off before the parse."""
    view = _view()
    settings = view._settings()
    found = _candidates(view)
    body = json.dumps({"decisions": [{"id": "c1", "action": "structure", "summary": "a summary"}],
                       "state_note": "a note"})

    calls.replies = [f"```json\n{body}\n```"]
    reply = view.flash_judge(found, "", settings)
    assert reply == {"decisions": [{"id": "c1", "action": "structure", "summary": "a summary"}],
                     "state_note": "a note"}

    calls.replies = [f"\n```\n{body}\n```\n"]
    assert view.flash_judge(found, "", settings) == reply, "a bare fence, or one with a language, both parse"
    # A decision on an id the pass did not offer is dropped: it can never move anything else.
    calls.replies = [json.dumps({"decisions": [{"id": "c99", "action": "structure", "summary": "invented"}]})]
    assert view.flash_judge(found, "", settings) == {"decisions": []}


def test_malformed_reply_changes_nothing(tmp_path, monkeypatch, calls):
    """Any parse or shape failure is no reply at all: nothing moves in the store or the pass."""
    clock = _clock(monkeypatch)
    view = _view()
    settings = view._settings()
    found = _candidates(view)
    malformed = [
        "",
        "not json at all",
        "Sure! Here is the JSON:",
        "[]",
        json.dumps({"decisions": "keep"}),
        json.dumps({"decisions": 5}),
        json.dumps({"decisions": True}),
        json.dumps({"decisions": {"c1": {"id": "c1", "action": "keep"}}}),
        json.dumps({"decisions": ["c1"]}),
        json.dumps({"decisions": [{"id": "c1"}]}),
        json.dumps({"decisions": [{"id": "c1", "action": "archive"}]}),
        json.dumps({"decisions": [{"id": 1, "action": "keep"}]}),
        json.dumps({"decisions": [{"id": "c1", "action": "keep", "summary": 5}]}),
        json.dumps({"decisions": [], "state_note": 5}),
        json.dumps({"decisions": [{"id": "c1", "action": "structure", "summary": "x"}]}) + " trailing",
    ]
    for bad in malformed:
        calls.replies = [bad]
        assert view.flash_judge(found, "", settings) is None, bad

    # Through a pass: the malformed reply writes its row and touches nothing else.
    engine = _engine(tmp_path, monkeypatch)
    conversation = _two_done_history()
    calls.replies = ["{\"decisions\": [{\"id\": \"c1\", \"action\": \"structure\""]
    _ask(engine, conversation)
    clock.advance(TTL)
    _ask(engine, conversation)

    state = _state(tmp_path)
    assert state.structured == {} and state.stubs == {} and state.blocks == [] and state.state_note == ""
    assert not list(_store_path(tmp_path).glob("ep-*.md")), "no episode was written on a parse failure"
    rows = _rows(tmp_path, "passes.jsonl")
    assert len(rows) == 1 and rows[0]["structured"] == 0 and rows[0]["kept"] == 2
    assert rows[0]["status"] == "failed", (
        "the judge made the call and could not use the reply: the pass is `failed`, never `no_route`")
    assert (rows[0]["flash_in"], rows[0]["flash_out"]) == (FLASH_IN, FLASH_OUT), (
        "the call's own usage is recorded even though the reply moved nothing")


def test_summary_ids_scrubbed(tmp_path, monkeypatch, calls):
    """A summary keeps only ids its own item holds, cut at the cap; the note is scrubbed against its input."""
    view = _view()
    settings = view._settings()
    real_id = "sa-77-deadbeef"
    invented_sha = "cafebabecafebabecafebabecafebabecafebabe"
    candidate = view.Candidate(id="c1", uid="t2", kind="superseded",
                                text=f"result {real_id} " + "y" * 300, chars=313, ids=[real_id],
                                reason="superseded")
    summary = f"moved {invented_sha} and {real_id} aside " + "padding " * 120
    note = f"goal: {invented_sha} {real_id} " + "next: " * 400
    calls.replies = [json.dumps({"decisions": [{"id": "c1", "action": "structure", "summary": summary}],
                                 "state_note": note})]

    reply = view.flash_judge([candidate], "[ctxpkg state]\nold", settings)
    kept = reply["decisions"][0]["summary"]
    assert invented_sha not in kept, "an id the candidate does not hold never reaches the prompt"
    assert real_id in kept, "an id the candidate does hold survives the scrub"
    assert len(kept) <= SUMMARY_MAX, "a summary longer than ctxpkg.summary_max_chars is cut"
    assert kept.endswith("padding"), "the cut lands on a word boundary"

    stored = reply["state_note"]
    assert invented_sha not in stored, "the note is scrubbed against every id in the judge's own input"
    assert real_id in stored, "an id the candidates carry survives the note's scrub"
    assert len(stored) <= STATE_NOTE_CHARS


def test_marker_only_turn_has_no_summary(tmp_path, monkeypatch, calls):
    """``marker_only`` on a turn writes its marker with no summary, and the turn still moves whole."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    calls.replies = [json.dumps({"decisions": [{"id": "c1", "action": "marker_only",
                                               "summary": "a summary the marker must not carry"}]})]
    conversation = _history(_done_turn(2))

    _ask(engine, conversation)
    clock.advance(TTL)
    _request, served = _ask(engine, conversation)

    row = _rows(tmp_path, "passes.jsonl")[0]
    assert (row["marker_only"], row["structured"], row["stubbed"]) == (1, 0, 0)
    state = _state(tmp_path)
    assert len(state.blocks) == 1
    line = state.blocks[0].splitlines()[1]
    assert line.startswith("ep-0001 · done · "), "the marker names the ref and the kind"
    assert "a summary the marker must not carry" not in state.blocks[0]
    assert len(line.split(" · ")) == 5, (
        "ref · kind · time · k chars · ids: no summary segment between the char count and the ids")
    assert "k chars · ids: " in line, "the char count runs straight into the ids"
    assert all(uid in state.structured for uid in ("u2", "a2", "t2", "z2"))
    assert "turn 2: please run it" not in [_text(message) for message in served]

    # The judge hands the pass no summary for a marker_only item: the marker line is built from "".
    calls.replies = [json.dumps({"decisions": [{"id": "c1", "action": "marker_only",
                                               "summary": "a summary the marker must not carry"}]})]
    judged = _view().flash_judge(_candidates(_view(), 1), "", _view()._settings())
    assert judged["decisions"][0] == {"id": "c1", "action": "marker_only", "summary": ""}


def test_stub_keeps_ids(tmp_path, monkeypatch, calls):
    """``marker_only`` on a tool result keeps its ids and loses its summary; the stub is served in place."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    command = "pytest -q tests/plugins"
    conversation = _turn(1) \
        + _turn(2, command=command, result="a result sa-77-deadbeef " + "y" * 300) \
        + _turn(3, command=command) \
        + _turn(4) + _turn(5) + _turn(6) \
        + [_msg("user", "now the current question", uid="ucur")]
    calls.replies = [json.dumps({"decisions": [{"id": "c1", "action": "marker_only",
                                               "summary": "must not be used"}]})]

    _ask(engine, conversation)
    clock.advance(TTL)
    _request, served = _ask(engine, conversation)

    state = _state(tmp_path)
    assert state.structured == {"t2": "ep-0001"} and state.blocks == [], "a tool result writes no block"
    stub = state.stubs["t2"]
    assert stub.startswith("[ep-0001 · superseded · 1k chars · the same call was made again later · ids: ")
    assert "sa-77-deadbeef" in stub, "``marker_only`` keeps the script-copied ids"
    assert "must not be used" not in stub
    assert stub.endswith(' · recall("ep-0001") for the full text]')
    served_stub = [message for message in served if message.get("tool_call_id") == "c2"]
    assert len(served_stub) == 1 and _text(served_stub[0]) == stub, "the stub is served in place"
    assert (_store_path(tmp_path) / "ep-0001.md").read_text(encoding="utf-8").startswith("a result sa-77-deadbeef")
    row = _rows(tmp_path, "passes.jsonl")[0]
    assert (row["stubbed"], row["marker_only"], row["structured"]) == (1, 1, 0)


def test_state_note_kept_only_after_success(tmp_path, monkeypatch, calls):
    """A failed pass keeps the old state note; a later successful pass stores the new one, decisions or not."""
    clock = _clock(monkeypatch)
    engine = _engine(tmp_path, monkeypatch)
    conversation = _history(_done_turn(2), _done_turn(3), _done_turn(4))

    calls.replies = [json.dumps({
        "decisions": [{"id": "c1", "action": "structure", "summary": "one"}],
        "state_note": "first note",
    })]
    _next_pass(clock, engine, conversation)
    assert _state(tmp_path).state_note == "first note"

    calls.error = RuntimeError("the judge exploded")
    _next_pass(clock, engine, conversation)
    rows = _rows(tmp_path, "passes.jsonl")
    assert [row["status"] for row in rows][-1] == "failed"
    assert _state(tmp_path).state_note == "first note", "a failed pass stores nothing, the note included"

    calls.error = None
    calls.replies = [json.dumps({"decisions": [{"id": "c2", "action": "keep"},
                                               {"id": "c3", "action": "keep"}],
                                 "state_note": "second note"})]
    _next_pass(clock, engine, conversation)
    state = _state(tmp_path)
    assert state.state_note == "second note", "a successful pass stores its note even when every item is kept"
    assert set(state.structured.values()) == {"ep-0001"}, "only the first turn moved"

    # A reply with no state note leaves the stored one alone.
    calls.replies = [json.dumps({"decisions": [{"id": "c2", "action": "keep"}]})]
    _next_pass(clock, engine, conversation)
    assert _state(tmp_path).state_note == "second note"
