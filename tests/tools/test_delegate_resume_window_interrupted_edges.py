"""Zero-window edges of the resume window: no window, no idle age worked out.

``delegation.resume_window_seconds`` at ``0`` is no window for that profile, so ``delegate_task(action='continue')``
must not work out an idle age at all: ``continue_subagent`` computes it only when the window is non-zero (and the
child is not exempt). The returned value hides a missing guard — ``_window_refusal`` answers None for every
``window <= 0`` whatever the idle age — so what is asserted here is the call itself: ``_idle_seconds`` is replaced by
a recorder, and a continue that consults it fails even though its result looks right.

``_idle_seconds`` is not free: for a record without ``stopped_at`` it opens the transcript store and does
``db.get_session`` on the child's session. Both shapes a continue can meet are covered, because they leave the
branch by different doors: a ``completed`` record (not exempt) and an ``interrupted`` one (exempt, so at a non-zero
window it logs and starts without an age). Each continue must dispatch and leave the recorder empty.
"""

from unittest.mock import patch

from tests.tools.test_delegate_continue import _completion
from tests.tools.test_delegate_continue_kept_children import _continue, _durable
from tests.tools.test_delegate_resume_window import _set_record, _set_session, _spawned
from tools import delegate_tool_continue as dtc

ZERO = {"resume_window_seconds": {"default": 0}}


def _idle_recorder(monkeypatch):
    """``_idle_seconds`` replaced by a recorder: every call is remembered, and answers "no age readable"."""
    calls = []

    def idle_seconds(spec, record, parent_agent, now):
        calls.append({"session_id": spec.get("session_id"), "outcome": (record or {}).get("outcome")})
        # no return: no age readable, the answer the caller's fail-closed branch uses

    monkeypatch.setattr(dtc, "_idle_seconds", idle_seconds)
    return calls


def test_zero_window_reads_no_idle_age(monkeypatch, tmp_path):
    """Under ``{default: 0}`` a ``completed`` child and an ``interrupted`` one — both with no ``stopped_at`` and no
    ``ended_at`` on their session, the shape that makes ``_idle_seconds`` open the transcript store — are
    continued: each dispatch runs, and ``_idle_seconds`` is never called."""
    for outcome in ("completed", "interrupted"):
        model, parent, sid = _spawned(tmp_path / outcome, delegation=ZERO)
        _set_session(parent, model.turns[0]["session_id"], ended_at=None)  # no age from the session
        if outcome == "completed":
            _set_record(parent, sid, drop=("stopped_at",))  # the spawn's own finish, minus its stamp
        else:
            _set_record(parent, sid, outcome=outcome, drop=("stopped_at",))
        record = _durable(parent, sid)
        assert record is not None and record["outcome"] == outcome, (outcome, record)
        assert "stopped_at" not in record, (outcome, record)  # no age from the record either
        assert dtc._window_exempt(record) is (outcome == "interrupted"), (outcome, record)  # the exempt shape

        calls = _idle_recorder(monkeypatch)
        with patch("run_agent.AIAgent", side_effect=model):
            cont = _continue(parent, sid, "go on")
            assert "error" not in cont and cont.get("status") == "dispatched", (outcome, cont)
            _completion(cont["delegation_id"])
        assert calls == [], (outcome, calls)  # no window, so no idle age is worked out (and no store opened)
        assert len(model.turns) == 2, outcome  # the continue really ran
