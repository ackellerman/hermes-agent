"""An INTERRUPTED child is not refused by the resume window (``delegation.resume_window_seconds``).

``delegate_task(action='continue')`` refuses a child that ENDED on its own terms and then sat idle past its
profile's window: its prompt cache is cold and the resume re-reads the whole prompt. A run that never ended on its
own terms is a different case, and the durable record already says so: its ``outcome`` is ``interrupted`` (a gateway
restart, a parent ``stop`` or a kill mid-run) or ``unknown`` (boot recovery and the dead-owner reclaim, which finish
a run whose owner process died). Those children are resumable whatever their idle age, including when no idle age can
be read at all — today's fail-closed branch — and only a finished run (``completed``, ``failed``, ``timeout``,
``error``) is still refused.

Every sibling the rule covers is pinned separately: the two boot-recovery outcomes (``unknown`` with a readable age,
``interrupted`` with none), the dead-owner reclaim, a clean gateway-restart stop, a parent's stop, the finished
control on both ``completed`` and ``failed``, and the predicate itself. The assertion for "it resumed" is a
``_start`` spy: it also shows that the record which reached the start is the interrupted one, so a predicate reading
some other field would fail here.

Only the model is scripted; registry, child build, SessionDB, the kept store and the durable store are the real ones,
and the clock is injected far from every stored timestamp (as ``test_delegate_resume_window`` does). ``_window_exempt``
is read off the module rather than imported: at the base commit it does not exist, and the five resume tests must fail
on the refusal they get back, not on a collection error for the whole file.
"""

import json
import sqlite3
from unittest.mock import patch

from tests.tools.test_delegate_continue_kept_children import _continue, _durable
from tests.tools.test_delegate_resume_window import (
    _Clock, _DEAD_PID, _set_record, _set_row, _set_session, _spawned, _use_clock,
)
from tools import delegate_tool_continue as dtc
from tools import delegate_tool_resumable as resumable

WINDOW = 270            # the built-in window every test here runs under (``{default: 270}``)
PAST = WINDOW + 3600    # an idle age an hour past it


def _stop_cause(parent, sid):
    """The row's ``stop_cause`` column (``_durable`` carries the record JSON, not the columns beside it): what boot
    recovery and the dead-owner reclaim wrote, and what a live stop left."""
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    try:
        row = conn.execute("SELECT stop_cause FROM resumable_subagents WHERE subagent_id=?", (sid,)).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _spy_start(monkeypatch):
    """Stand in for ``_start``: record what reached it, and the outcome of the record it was handed, instead of
    running a child. ``_start`` is looked up on the module, so patching it there is what the refusal guards."""
    calls = []

    def start(sid, message, spec, parent_agent, background, record=None, built=None):
        calls.append({"subagent_id": sid, "message": message, "outcome": (record or {}).get("outcome")})
        return json.dumps({"status": "dispatched", "delegation_id": "dlg-1", "subagent_ids": [sid]}), None

    monkeypatch.setattr(dtc, "_start", start)
    return calls


def _resumed(monkeypatch, parent, sid, outcome, message="go on"):
    """The continue of *sid* must reach ``_start`` — no refusal text — and its record must say *outcome*."""
    calls = _spy_start(monkeypatch)
    cont = _continue(parent, sid, message)
    assert "error" not in cont, cont
    assert "resume window" not in cont.get("error", "") and "Spawn a fresh task" not in cont.get("error", ""), cont
    assert cont.get("status") == "dispatched", cont
    assert calls == [{"subagent_id": sid, "message": message, "outcome": outcome}], calls
    return cont


# ── boot recovery: both of its outcomes ─────────────────────────────────────

def test_boot_recovered_unknown_resumes_past_window(monkeypatch, tmp_path):
    """Boot recovery finished a run whose owner died with no gateway shutdown marked: ``outcome`` ``unknown``, no
    ``stopped_at`` (a boot must not make a long-dead child read as just stopped), and the session ended two hours
    before the continue — an idle age 7200 s, well past the 270 s window that refuses the same record under
    ``completed``."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _set_session(parent, model.turns[0]["session_id"], ended_at=clock.now - 7200)
    _set_row(parent, sid, state="running", owner_pid=_DEAD_PID, owner_started_at=None, stop_cause=None)
    resumable._write(parent._session_db.db_path, lambda conn: resumable.recover_abandoned(
        conn, {sid: (_DEAD_PID, None, [])}))
    recovered = _durable(parent, sid)
    assert recovered["state"] == "resumable" and recovered["outcome"] == "unknown", recovered
    assert "stopped_at" not in recovered, recovered

    clock.now = clock.now + 10
    _use_clock(monkeypatch, dtc, clock)
    idle = dtc._idle_seconds({"session_id": recovered["session_id"]}, recovered, parent, clock.now)
    assert idle is not None and idle > WINDOW, idle  # a readable age, past the window

    _resumed(monkeypatch, parent, sid, "unknown")


def test_boot_recovered_interrupted_resumes_with_no_idle_age(monkeypatch, tmp_path):
    """A gateway shutdown was marked before the process died, so boot recovery finishes the run as ``interrupted``
    with no ``stopped_at``; the child's session carries no ``ended_at`` either, so no idle age can be read at all —
    today's fail-closed branch — and the child still resumes."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _set_session(parent, model.turns[0]["session_id"], ended_at=None)
    _set_row(parent, sid, state="running", owner_pid=_DEAD_PID, owner_started_at=None,
             stop_cause=resumable.GATEWAY_SHUTDOWN)
    resumable._write(parent._session_db.db_path, lambda conn: resumable.recover_abandoned(
        conn, {sid: (_DEAD_PID, None, [])}))
    recovered = _durable(parent, sid)
    assert recovered["state"] == "resumable" and recovered["outcome"] == "interrupted", recovered
    assert "stopped_at" not in recovered and recovered["last_run_seconds"] is None, recovered

    _use_clock(monkeypatch, dtc, clock)
    assert dtc._idle_seconds({"session_id": recovered["session_id"]}, recovered, parent, clock.now) is None

    _resumed(monkeypatch, parent, sid, "interrupted")


# ── the reclaim, and the two stops that record themselves ───────────────────

def test_reclaimed_dead_owner_resumes_past_window(monkeypatch, tmp_path):
    """The dead-owner reclaim rather than boot recovery: the row is still ``running`` under a dead pid when the
    continue claims it, so ``claim`` finishes it as ``unknown`` — and the child resumes an hour past the window."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _set_session(parent, model.turns[0]["session_id"], ended_at=clock.now - PAST)
    _set_row(parent, sid, state="running", owner_pid=_DEAD_PID, owner_started_at=None, stop_cause=None)
    assert _durable(parent, sid)["state"] == "running"  # the reclaim, not recovery, is what finishes this one
    _use_clock(monkeypatch, dtc, clock)

    _resumed(monkeypatch, parent, sid, "unknown")
    assert _durable(parent, sid)["outcome"] == "unknown"  # the claim wrote the interrupted outcome itself


def test_restart_stopped_child_resumes_past_window(monkeypatch, tmp_path):
    """A gateway restart stopped the child cleanly: ``record_stop`` stamped ``stopped_at`` with ``outcome``
    ``interrupted`` and ``stop_cause`` ``gateway_shutdown``. The idle age is readable and an hour past the window,
    which is exactly what refuses it today."""
    clock = _Clock()
    _model, parent, sid = _spawned(tmp_path)
    _set_record(parent, sid, outcome="interrupted", stopped_at=clock.now - PAST)
    _set_row(parent, sid, state="resumable", stop_cause=resumable.GATEWAY_SHUTDOWN)
    record = _durable(parent, sid)
    assert (record["outcome"], _stop_cause(parent, sid), record["state"]) == (
        "interrupted", "gateway_shutdown", "resumable"), record
    _use_clock(monkeypatch, dtc, clock)
    idle = dtc._idle_seconds({"session_id": record["session_id"]}, record, parent, clock.now)
    assert idle is not None and idle > WINDOW, idle

    _resumed(monkeypatch, parent, sid, "interrupted")


def test_parent_stopped_child_resumes_past_window(monkeypatch, tmp_path):
    """A parent ``stop`` (or a kill) mid-run: ``status`` ``interrupted``, a fresh ``stopped_at`` and NO stop cause, so
    ``classify`` calls the row ``stopped`` (a person, no pin). It is still a record a continue can claim, and still
    exempt — the exemption reads the outcome, never the cause."""
    clock = _Clock()
    _model, parent, sid = _spawned(tmp_path)
    _set_record(parent, sid, outcome="interrupted", stopped_at=clock.now - PAST)
    _set_row(parent, sid, state="stopped", stop_cause=None)
    record = _durable(parent, sid)
    assert (record["outcome"], _stop_cause(parent, sid), record["state"]) == ("interrupted", None, "stopped"), record
    _use_clock(monkeypatch, dtc, clock)

    _resumed(monkeypatch, parent, sid, "interrupted")


# ── the control: a finished run is still refused ────────────────────────────

def test_finished_child_past_window_still_refused(monkeypatch, tmp_path):
    """A run that ended on its own terms — ``completed``, and separately ``failed`` — one second past the window is
    refused before anything is built, with today's exact text, and its claim is handed back."""
    from tools.delegate_tool_registry import _recent_subagents

    clock = _Clock()
    for outcome in ("completed", "failed"):
        model, parent, sid = _spawned(tmp_path / outcome)
        _use_clock(monkeypatch, dtc, clock)
        _set_record(parent, sid, outcome=outcome, stopped_at=clock.now - (WINDOW + 1))
        calls = _spy_start(monkeypatch)
        cont = _continue(parent, sid, "go on")
        assert cont.get("error") == (
            f"Subagent '{sid}' (profile default) has been idle 4 min, past its 4 min resume window: its prompt "
            "cache is cold. Spawn a fresh task instead, with a package of what it needs."), (outcome, cont)
        assert calls == [], (outcome, calls)  # never reached the start
        assert len(model.builds) == 1 and len(model.turns) == 1, outcome  # nothing built, nothing started
        assert _durable(parent, sid)["state"] == "completed", outcome  # not left `claimed`
        assert _recent_subagents[sid].get("resume"), outcome  # the in-process spec was handed back too


# ── the predicate itself ────────────────────────────────────────────────────

def test_window_exempt_predicate():
    """Exactly two outcomes exempt, and only on a dict record: ``interrupted`` and ``unknown``. Every finished
    outcome (``completed``, ``failed``, ``timeout``, ``error``) and every value that is not a record (``None``,
    ``{}``, a string, a list) is not exempt. The return is a bool, never a truthy value."""
    exempt = dtc._window_exempt
    assert exempt({"outcome": "interrupted"}) is True
    assert exempt({"outcome": "unknown"}) is True
    for outcome in ("completed", "failed", "timeout", "error"):
        assert exempt({"outcome": outcome}) is False, outcome
    assert exempt({}) is False
    assert exempt(None) is False
    assert exempt("interrupted") is False  # a string carrying the word is not a record
    assert exempt(["outcome"]) is False
