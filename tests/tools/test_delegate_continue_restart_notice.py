"""The restart notice names the subagents it says can be continued.

Boot recovery puts ``resumable_subagent_ids`` on the event and tells the model to call
``delegate_task(action='continue', subagent_id=..., message=...)``; the text the model reads has to carry the
ids too, or a child that is not listed anywhere else cannot be continued after a gateway restart.

Real dispatch, real boot recovery and the real notification formatter against a temp HERMES_HOME; only the model
is scripted.
"""

import json
import sqlite3
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _parent, _ScriptedModel
from tests.tools.test_delegate_resumable_pins import _DEAD_PID, _completions


@pytest.mark.parametrize("child", ["crashed", "stopped-by-shutdown"])
@pytest.mark.parametrize("goals", [["write the parser"], ["write the first parser", "write the second parser"]],
                         ids=["one-child", "two-children"])
def test_the_recovered_notice_text_names_every_continuable_subagent_id(goals, child):
    from tools import async_delegation as ad
    from tools.delegate_tool import delegate_task
    from tools.process_registry_notifications import format_process_notification
    parent, model = _parent(), _ScriptedModel()
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(delegate_task(tasks=[{"goal": g} for g in goals], background=True, parent_agent=parent))
        units = [u["delegation_id"] for u in out["units"]] if out.get("units") else [out["delegation_id"]]
        _completions(units)
    ids = out.get("subagent_ids") or [out["subagent_id"]]
    assert len(ids) == len(goals), out
    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        # The owner died before any ledger row turned terminal.
        conn.execute("UPDATE async_delegations SET state='running', owner_pid=?, owner_started_at=NULL, "
                     "result_json=NULL", (_DEAD_PID,))
        if child == "crashed":
            conn.execute("UPDATE resumable_subagents SET state='running', owner_pid=?, owner_started_at=NULL",
                         (_DEAD_PID,))
        else:
            conn.execute("UPDATE resumable_subagents SET state='resumable', stop_cause='gateway_shutdown'")

    assert ad.recover_abandoned_delegations() == len(units)

    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        events = [json.loads(raw) for (raw,) in conn.execute("SELECT event_json FROM async_delegations")]
    texts = [format_process_notification(evt) for evt in events]
    for evt, text in zip(events, texts):
        assert evt.get("resumable_subagent_ids"), evt
        assert "action='continue'" in text
        for sid in evt["resumable_subagent_ids"]:  # the id the instruction needs is in the text it is given in
            assert sid in text, text
    assert sorted(sid for evt in events for sid in evt["resumable_subagent_ids"]) == sorted(ids)


def test_a_notice_without_continuable_children_gains_no_line():
    from tools.process_registry_notifications import _recovery_lines
    evt = {"last_known_status": "running", "task_transcripts": {}}
    assert not any("ontinuable" in line for line in _recovery_lines(evt))
    assert not any("ontinuable" in line for line in _recovery_lines({**evt, "resumable_subagent_ids": []}))
