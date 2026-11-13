"""A continue that fails after its child was rebuilt leaves nothing of that child behind.

The rebuilt child is attached to the parent for interrupt fan-out and its session is reopened before the
dispatch. When the dispatch then fails, the child is detached and closed (which ends its session again), the
same teardown the spawn path gives a child that never started, and a retry dispatches.
"""

import json
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel, _spawn


class _ClosingModel(_ScriptedModel):
    """Children whose ``close()`` ends their session row, as ``AIAgent.close()`` does."""

    def __call__(self, **kw):
        child = super().__call__(**kw)
        db = kw["session_db"]
        child.close.side_effect = lambda: db.end_session(child.session_id, "agent_close")
        self.children = [*getattr(self, "children", []), child]
        return child


@pytest.mark.parametrize("failing", ["tools.delegate_tool_dispatch._run_batch",
                                     "tools.delegate_tool_resumable.last_workspace"],
                         ids=["raised-in-dispatch", "raised-before-the-batch"])
def test_a_continue_that_fails_after_the_build_detaches_and_closes_the_child(failing):
    from tools.delegate_tool import delegate_task

    model, parent = _ClosingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        session_id = model.turns[0]["session_id"]
        assert parent._active_children == []
        assert parent._session_db.get_session(session_id)["ended_at"] is not None

        with patch(failing, side_effect=RuntimeError("boom")):
            out = json.loads(delegate_task(action="continue", subagent_id=sid, message="now add tests",
                                           background=True, parent_agent=parent))
        assert "Could not continue subagent" in out["error"], out
        rebuilt = model.children[1]
        assert len(model.builds) == 2 and len(model.turns) == 1  # built, never ran

        assert parent._active_children == []  # not left for the parent's interrupt fan-out
        assert rebuilt.close.call_count == 1
        assert parent._session_db.get_session(session_id)["ended_at"] is not None  # reopened, then ended again

        out = json.loads(delegate_task(action="continue", subagent_id=sid, message="now add tests",
                                       background=True, parent_agent=parent))
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])
    assert model.turns[1]["message"] == "now add tests"
    assert parent._active_children == []


def test_a_continue_refused_before_the_build_closes_nothing():
    from tools.delegate_tool import delegate_task

    model, parent = _ClosingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        with patch("tools.delegate_tool._oneshot_spawn_budget", return_value="no budget"):
            out = json.loads(delegate_task(action="continue", subagent_id=sid, message="more", background=True,
                                           parent_agent=parent))
        assert out["error"] == "no budget", out
    assert len(model.children) == 1 and model.children[0].close.call_count == 1  # only the first run's own close
