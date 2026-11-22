"""The resume window for kept children (``delegation.resume_window_seconds``).

``delegate_task(action='continue')`` may resume a kept child only while that child's prompt cache can still be
warm: an idle child past its profile's window is refused, its claim handed back, and the caller told to spawn a
fresh child with a package. ``action='list'`` shows each kept child's ``idle_seconds``, its last run's
``last_run_seconds`` and whether it is ``warm``.

The clock is injected in the module under test and set FAR from every stored timestamp, so a guard that reads the
wrong time (the kept row's ``idle_since``, the record's ``created_at``, or the wall clock instead of the injected
one) answers on the other side of the bound. Only the model is scripted; registry, child build, SessionDB, the
kept store and the durable store are the real ones.
"""

import json
import logging
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion
from tests.tools.test_delegate_continue_kept_children import (
    _KeptModel, _continue, _durable, _idle, _kept_parent, _spawn_kept, _write_profile,
)
from tools import delegate_children as dc
from tools import delegate_tool_continue as dtc
from tools import delegate_tool_resumable as resumable
from tools.delegate_tool_config import _resume_window_seconds
from tools.delegate_tool_continue import _window_refusal

FAR = 2_000_000_000.0  # 2033: far from every real timestamp these tests write
_DEAD_PID = 2 ** 22 + 7


class _Clock:
    """A clock the module under test reads instead of the wall clock."""

    def __init__(self, now=FAR):
        self.now = now


class _TimedModel(_KeptModel):
    """A kept child whose run takes *seconds* of *clock*: ``record_start`` reads the clock, the run happens, and
    ``_finish`` reads it again, so the run's recorded length is exactly *seconds*."""

    def __init__(self, clock, seconds):
        super().__init__()
        self.clock = clock
        self.seconds = seconds

    def __call__(self, **kw):
        child = super().__call__(**kw)
        inner = child.run_conversation.side_effect

        def run_conversation(*args, **kwargs):
            result = inner(*args, **kwargs)
            self.clock.now += self.seconds
            return result
        child.run_conversation.side_effect = run_conversation
        return child


def _use_clock(monkeypatch, module, clock):
    """*module* sees ``time.time()`` as *clock*; everything else in it keeps the real ``time``."""
    monkeypatch.setattr(module, "time", SimpleNamespace(
        time=lambda: clock.now, monotonic=time.monotonic, strftime=time.strftime))


def _configure(delegation):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": delegation}), encoding="utf-8")


def _spawned(tmp_path, delegation=None, model=None):
    """A kept child that finished normally: ``(model, parent, subagent_id)``."""
    _configure({} if delegation is None else delegation)
    model = _KeptModel() if model is None else model
    parent = _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "write the parser")
    sid = out["subagent_ids"][0]
    _idle(parent)
    return model, parent, sid


def _set_record(parent, sid, *, expires_at=None, drop=(), **fields):
    """Rewrite the child's durable record: the fields given, minus *drop*, unexpired at the injected clock."""
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        raw = conn.execute("SELECT record_json FROM resumable_subagents WHERE subagent_id=?", (sid,)).fetchone()[0]
        record = {**json.loads(raw), **fields}
        for key in drop:
            record.pop(key, None)
        conn.execute("UPDATE resumable_subagents SET record_json=?, expires_at=? WHERE subagent_id=?",
                     (json.dumps(record), FAR + 86_400 if expires_at is None else expires_at, sid))
    conn.close()


def _set_row(parent, sid, **cols):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute(f"UPDATE resumable_subagents SET {', '.join(f'{k}=?' for k in cols)} WHERE subagent_id=?",
                     (*cols.values(), sid))
    conn.close()


def _drop_record(parent, sid):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute("DELETE FROM resumable_subagents WHERE subagent_id=?", (sid,))
    conn.close()


def _set_session(parent, session_id, **cols):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute(f"UPDATE sessions SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
                     (*cols.values(), session_id))
    conn.close()


def _drop_session(parent, session_id):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
    with conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.close()


# ── the window, both edges ──────────────────────────────────────────────────

def test_continue_within_window_runs(monkeypatch, tmp_path):
    """269 s idle, one second inside the built-in 270 s window, runs: the durable record's ``stopped_at`` decides,
    not the kept row's ``idle_since`` (hours off at this clock) and not the session's ``ended_at``."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    session = model.turns[0]["session_id"]
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 269)
    _set_session(parent, session, ended_at=clock.now - 100_000)  # a wrong source would refuse
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "now add tests")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(model.builds) == 2 and len(model.turns) == 2


def test_continue_past_default_window_refused(monkeypatch, tmp_path):
    """271 s idle, one second past the built-in 270 s window, is refused before anything is built; the session's
    recent ``ended_at`` must not rescue it."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    session = model.turns[0]["session_id"]
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 271)
    _set_session(parent, session, ended_at=clock.now - 10)  # a fallback source would say warm
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "now add tests")
    assert "resume window" in cont.get("error", ""), cont
    assert len(model.builds) == 1 and len(model.turns) == 1  # nothing built, nothing started


def test_refusal_releases_claim_for_fresh_claim(monkeypatch, tmp_path):
    """A window refusal hands the durable claim and the in-process spec back: the row says what it said before the
    continue, and a fresh claim of the same id succeeds."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 271)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "now add tests")
    assert "resume window" in cont.get("error", ""), cont
    assert _durable(parent, sid)["state"] == "completed"  # not left `claimed`
    from tools.delegate_tool_registry import _recent_subagents
    assert _recent_subagents[sid].get("resume")  # the in-process spec was handed back too
    record, err = resumable.claim(parent, sid)
    assert err is None and record is not None and record["state"] == "completed", (record, err)
    resumable.release(parent, sid, record)  # leave the row as we found it


# ── per-profile windows ─────────────────────────────────────────────────────

class _Roles:
    """A ``profile_routes`` backend that routes a task to the profile it names."""

    name = "profile_routes"

    def __init__(self, roles):
        self.roles = tuple(roles)

    def task_parameters(self, config):
        return {"profile": {"type": "string", "enum": list(self.roles), "description": "Run as this profile."}}

    def description(self, config):
        return ""

    def route_task(self, task, config):
        return {"profile": task["profile"], "label": task["profile"]} if task.get("profile") else {}


@pytest.fixture
def roles(tmp_path, monkeypatch):
    """A launch profile with ``{default: 270, reviewer: 3540}`` and two profiles a child can run as."""
    launch = tmp_path / "os-home" / ".hermes"
    for role in ("reviewer", "implementer"):
        _write_profile(launch / "profiles" / role, host=f"{role}.invalid", model=f"{role}-model", key=f"{role}-key")
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key",
                   delegation={"resume_window_seconds": {"default": 270, "reviewer": 3540}})
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROFILE_KEY", "launch-key")
    with patch("tools.delegate_tool_routes._active_provider", return_value=_Roles(("reviewer", "implementer"))):
        yield launch


def test_reviewer_profile_window_from_config(monkeypatch, tmp_path, roles):
    """1000 s idle runs for a child that ran as ``reviewer`` (3540 s) and is refused for one that ran as
    ``implementer``, which has no entry and so takes ``default`` (270 s)."""
    clock = _Clock()
    review = _KeptModel()
    parent = _kept_parent(tmp_path / "review-db")
    with patch("run_agent.AIAgent", side_effect=review):
        out, _done = _spawn_kept(parent, "review the change", profile="reviewer")
    sid = out["subagent_ids"][0]
    _idle(parent)
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 1000)
    with patch("run_agent.AIAgent", side_effect=review):
        cont = _continue(parent, sid, "and the delta")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(review.builds) == 2 and len(review.turns) == 2

    impl = _KeptModel()
    other = _kept_parent(tmp_path / "impl-db")
    with patch("run_agent.AIAgent", side_effect=impl):
        out2, _done2 = _spawn_kept(other, "build it", profile="implementer")
    sid2 = out2["subagent_ids"][0]
    _idle(other)
    _set_record(other, sid2, stopped_at=clock.now - 1000)
    with patch("run_agent.AIAgent", side_effect=impl):
        refused = _continue(other, sid2, "again")
    assert "has been idle 16 min, past its 4 min resume window" in refused.get("error", ""), refused
    assert len(impl.builds) == 1 and len(impl.turns) == 1


def test_zero_window_disables_guard(monkeypatch, tmp_path):
    """``0`` means no window for that profile: an ancient stop runs, and so does a child with no idle signal at
    all (a disabled guard cannot fail closed)."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path, delegation={"resume_window_seconds": {"default": 0}})
    session = model.turns[0]["session_id"]
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 100_000)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "much later")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(model.turns) == 2

    _set_record(parent, sid, drop=("stopped_at",))
    _set_session(parent, session, ended_at=None)
    with patch("run_agent.AIAgent", side_effect=model):
        again = _continue(parent, sid, "no signal either")
        assert again.get("status") == "dispatched", again
        _completion(again["delegation_id"])
    assert len(model.turns) == 3


# ── the idle signal, and its absence ────────────────────────────────────────

def test_missing_idle_signal_refused(monkeypatch, tmp_path):
    """Neither source readable — the child's session has no ``ended_at``, or has no row at all — is refused (fail
    closed), nothing built, and the claim handed back."""
    clock = _Clock()
    for variant in ("ended_at-null", "no-session-row"):
        model, parent, sid = _spawned(tmp_path / variant)
        _use_clock(monkeypatch, dtc, clock)
        _set_record(parent, sid, drop=("stopped_at",))
        session = model.turns[0]["session_id"]
        if variant == "ended_at-null":
            _set_session(parent, session, ended_at=None)
        else:
            _drop_session(parent, session)
        with patch("run_agent.AIAgent", side_effect=model):
            cont = _continue(parent, sid, "again")
        assert "window cannot be checked" in cont.get("error", ""), (variant, cont)
        assert "Spawn a fresh task instead, with a package of what it needs." in cont["error"]
        assert len(model.builds) == 1 and len(model.turns) == 1, variant
        assert _durable(parent, sid)["state"] == "completed", variant


def test_ended_at_fallback_used(monkeypatch, tmp_path):
    """A record whose owner died after boot recovery carries no ``stopped_at`` (recovery must not stamp one, or a
    long-dead child reads as just stopped): the idle age comes from the session's ``ended_at``, which is inside
    the window here, so the continue runs."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _set_session(parent, model.turns[0]["session_id"], ended_at=clock.now - 100)
    _set_row(parent, sid, state="running", owner_pid=_DEAD_PID, owner_started_at=None)
    record, err = resumable.claim(parent, sid)  # the claim reclaims the dead owner's row
    assert err is None and record["state"] == "resumable", (record, err)
    assert "stopped_at" not in record and record["last_run_seconds"] is None
    resumable.release(parent, sid, record)

    _use_clock(monkeypatch, dtc, clock)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "go on")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(model.turns) == 2


def test_in_process_spec_uses_spec_session(monkeypatch, tmp_path):
    """``_claim`` returns no durable record: the idle age comes from the ``ended_at`` of the session the spec
    names. Inside the window it runs; far past it, it is refused rather than skipped."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    session = model.turns[0]["session_id"]
    _drop_record(parent, sid)  # no durable record: (spec, None, None)
    _set_session(parent, session, ended_at=clock.now - 100)
    _use_clock(monkeypatch, dtc, clock)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "go on")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    assert len(model.turns) == 2

    other = _KeptModel()
    parent2 = _kept_parent(tmp_path / "parent-db-2")
    with patch("run_agent.AIAgent", side_effect=other):
        out2, _done2 = _spawn_kept(parent2, "write the parser")
    sid2 = out2["subagent_ids"][0]
    _idle(parent2)
    _drop_record(parent2, sid2)
    _set_session(parent2, other.turns[0]["session_id"], ended_at=clock.now - 100_000)
    with patch("run_agent.AIAgent", side_effect=other):
        refused = _continue(parent2, sid2, "much later")
    assert "resume window" in refused.get("error", ""), refused
    assert len(other.builds) == 1 and len(other.turns) == 1


def test_recovered_child_not_warm(monkeypatch, tmp_path):
    """A child boot recovery finished, whose session ended two hours before the recovery, is refused at
    ``recovery + 10``: ``recover_abandoned`` must not stamp ``stopped_at`` on it."""
    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _set_session(parent, model.turns[0]["session_id"], ended_at=clock.now - 7200)
    _set_row(parent, sid, state="running", owner_pid=_DEAD_PID, owner_started_at=None)
    resumable._write(parent._session_db.db_path, lambda conn: resumable.recover_abandoned(
        conn, {sid: (_DEAD_PID, None, [])}))
    recovered = _durable(parent, sid)
    assert recovered["state"] == "resumable", recovered
    assert "stopped_at" not in recovered and recovered["last_run_seconds"] is None

    clock.now = clock.now + 10
    _use_clock(monkeypatch, dtc, clock)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "go on")
    assert "has been idle 120 min, past its 4 min resume window" in cont.get("error", ""), cont
    assert len(model.builds) == 1 and len(model.turns) == 1
    assert _durable(parent, sid)["state"] == "resumable"


# ── config ──────────────────────────────────────────────────────────────────

def test_invalid_config_warns_and_uses_default(caplog, monkeypatch):
    """Every invalid shape logs one warning and gives the built-in 270 s; valid mappings are read as written."""
    from tools import delegate_tool_config
    cases = [
        ({}, None, 270, False),                                  # no key: the built-in default, quietly
        ({}, "reviewer", 270, False),                             # an empty mapping has no `default` either
        ({"default": 600}, "implementer", 600, False),            # a profile with no entry takes `default`
        ({"default": 600, "reviewer": 3540}, "reviewer", 3540, False),
        ({"default": 600, "reviewer": 0}, "reviewer", 0, False),  # 0 = no window for that profile only
        ({"default": 600, "reviewer": 0}, "implementer", 600, False),
        ({"default": [270]}, None, 270, True),                    # a list is not a number of seconds
        ({"default": "270"}, None, 270, True),                    # nor is a string
        ({"default": -5}, None, 270, True),                       # a negative window is meaningless
        ({"default": True}, None, 270, True),                      # bool is an int in Python; still not seconds
        ({"default": 3540.5}, None, 270, True),                   # a float is not a whole number of seconds
        ({"reviewer": 3540.5}, "reviewer", 270, True),
        ({"reviewer": -1}, "reviewer", 270, True),
        ("270", None, 270, True),                                 # the value must be a mapping
        ([270], None, 270, True),
        (270, None, 270, True),
    ]
    for value, profile, want, warns in cases:
        _configure({"resume_window_seconds": value})
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
            got = _resume_window_seconds(profile)
        assert got == want, (value, profile)
        warned = [r for r in caplog.records if "resume_window_seconds" in r.getMessage()]
        assert bool(warned) is warns, (value, profile, [r.getMessage() for r in caplog.records])
        assert len(warned) <= 1, (value, profile, [r.getMessage() for r in caplog.records])  # one warning

    # A mapping with no `default` at all: the loader deep-merges `{default: 270}` into every real config, so this
    # shape is only reachable when the config is set in-process. An unlisted profile still gets the built-in
    # 270 s — never the mapping's first value (3540 here) and never 0.
    monkeypatch.setattr(delegate_tool_config, "_cfg", lambda: {"resume_window_seconds": {"reviewer": 3540}})
    assert _resume_window_seconds("implementer") == 270
    assert _resume_window_seconds("reviewer") == 3540


def test_refusal_text_exact(monkeypatch, tmp_path):
    """The refusal is the plan's sentence, integers rounded down, ``unknown`` for an unresolved profile; the wired
    refusal carries exactly it."""
    assert _window_refusal("sa-x1", "implementer", 1000, 270) == (
        "Subagent 'sa-x1' (profile implementer) has been idle 16 min, past its 4 min resume window: its prompt "
        "cache is cold. Spawn a fresh task instead, with a package of what it needs.")
    assert _window_refusal("sa-x1", None, 4000, 3540) == (
        "Subagent 'sa-x1' (profile unknown) has been idle 66 min, past its 59 min resume window: its prompt "
        "cache is cold. Spawn a fresh task instead, with a package of what it needs.")
    assert _window_refusal("sa-x1", "reviewer", 3540, 3540) is None   # the window itself still runs
    assert _window_refusal("sa-x1", "reviewer", 3541, 3540)          # one second past it does not

    clock = _Clock()
    model, parent, sid = _spawned(tmp_path)
    _use_clock(monkeypatch, dtc, clock)
    _set_record(parent, sid, stopped_at=clock.now - 1000)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "again")
    assert cont["error"] == _window_refusal(sid, "default", 1000, 270), cont


# ── the run's own length ────────────────────────────────────────────────────

def test_last_run_seconds_recorded(monkeypatch, tmp_path):
    """``record_start`` stamps the run's start and ``_finish`` records its length as ``stopped_at - run_started_at``
    (a run that takes 37 s of the injected clock records 37, not -37); a continue restarts the clock, so the
    second run's length is its own and never the idle gap before it."""
    clock = _Clock()
    _use_clock(monkeypatch, resumable, clock)
    model = _TimedModel(clock, 37)
    model, parent, sid = _spawned(tmp_path, delegation={"resume_window_seconds": {"default": 100_000}},
                                  model=model)
    first = _durable(parent, sid)
    assert first["run_started_at"] == FAR and first["stopped_at"] == FAR + 37
    assert first["last_run_seconds"] == 37, first  # the run's own length, and its sign

    clock.now = clock.now + 600  # ten idle minutes between the runs
    _use_clock(monkeypatch, dtc, clock)
    with patch("run_agent.AIAgent", side_effect=model):
        cont = _continue(parent, sid, "go on")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])
    second = _durable(parent, sid)
    assert second["run_started_at"] == FAR + 637  # the continue restarted the run's clock
    assert second["last_run_seconds"] == 37, second  # not 637: the idle gap is not the run's length


# ── action='list' ───────────────────────────────────────────────────────────

def test_list_entry_shows_idle_and_warm(monkeypatch, tmp_path):
    """``list_entry`` reports the idle age, the last run's length, the profile's window and warmth; a record with
    no ``stopped_at`` falls back to the kept row's ``idle_since``; the window comes from config when not passed."""
    now = FAR
    row = {"child_id": "sa-1", "subagent_id": "sa-1", "status": "idle", "idle_since": now - 900,
           "conversation_id": "parent-sess"}
    kwargs = {"conversations": ["parent-sess"], "profile": "default", "now": now}
    warm = dc.list_entry(row, resume={"stopped_at": now - 100, "last_run_seconds": 42.7},
                         resume_window_seconds=270, **kwargs)
    assert (warm["idle_seconds"], warm["last_run_seconds"], warm["resume_window_seconds"], warm["warm"]) == (
        100, 42, 270, True)
    cold = dc.list_entry(row, resume={"stopped_at": now - 275, "last_run_seconds": None},
                         resume_window_seconds=270, **kwargs)
    assert (cold["idle_seconds"], cold["last_run_seconds"], cold["warm"]) == (275, None, False)
    off = dc.list_entry(row, resume={"stopped_at": now - 100_000, "last_run_seconds": 5},
                        resume_window_seconds=0, **kwargs)
    assert off["warm"] is True and off["resume_window_seconds"] == 0
    recovered = dc.list_entry(row, resume={"last_run_seconds": None}, resume_window_seconds=270, **kwargs)
    assert (recovered["idle_seconds"], recovered["warm"]) == (900, False)  # no stopped_at: the row's idle_since

    # The boundary, as the plan states it: warm = idle_seconds < window. Exactly at the window the child is not
    # warm, while `continue` still runs it (it refuses only past the window) — the listing errs one second cold.
    edge = dc.list_entry(row, resume={"stopped_at": now - 270, "last_run_seconds": None},
                         resume_window_seconds=270, **kwargs)
    assert (edge["idle_seconds"], edge["warm"]) == (270, False)
    inside = dc.list_entry(row, resume={"stopped_at": now - 269}, resume_window_seconds=270, **kwargs)
    assert (inside["idle_seconds"], inside["warm"]) == (269, True)
    assert _window_refusal("sa-1", "default", 270, 270) is None  # continue runs at exactly the window

    _configure({"resume_window_seconds": {"default": 270, "reviewer": 3540}})
    reviewer = dc.list_entry({**row, "profile": "reviewer"}, conversations=["parent-sess"], profile="default",
                             now=now)
    assert reviewer["resume_window_seconds"] == 3540  # the row's own profile, not the viewer's

    _model, parent, sid = _spawned(tmp_path)
    from tools.delegate_tool import delegate_task
    entry = json.loads(delegate_task(action="list", parent_agent=parent))["children"][0]
    assert entry["subagent_id"] == sid
    assert isinstance(entry["idle_seconds"], int) and entry["idle_seconds"] >= 0
    assert isinstance(entry["last_run_seconds"], int) and entry["last_run_seconds"] >= 0
    assert entry["resume_window_seconds"] == 270 and entry["warm"] is True, entry
