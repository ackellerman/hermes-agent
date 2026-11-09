"""A cron job's agent never keeps its delegated children (``tools.delegate_children.NEVER_KEEP_ATTR``):
nobody comes back to a finished job to resume them (plan delegate-resume, e1)."""

from __future__ import annotations

import cron.scheduler as cron_scheduler
from tools import delegate_children as dc

_SEEN: dict = {}


class _FakeCronAgent:
    def __init__(self, *args, **kwargs):
        self.session_id = kwargs["session_id"]
        self.task_id = self.session_id

    def run_conversation(self, prompt, *, task_id=None):
        _SEEN["never_keep"] = getattr(self, dc.NEVER_KEEP_ATTR, None)
        _SEEN["keeps"] = dc.should_keep(self, origin=dc.ORIGIN_MODEL, child_depth=1)
        return {"completed": True, "failed": False, "final_response": "done", "turn_exit_reason": ""}

    def close(self):
        pass


def test_cron_job_agent_is_marked_never_keep(monkeypatch, tmp_path, cron_owner):
    _SEEN.clear()
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setattr("run_agent.AIAgent", _FakeCronAgent)
    monkeypatch.setattr("hermes_constants.resolve_reasoning_config", lambda *_a, **_k: None)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **_k: {
        "api_key": "test-key", "base_url": None, "provider": "test-provider", "api_mode": None, "command": None,
        "args": None})
    monkeypatch.setattr("tools.mcp_tool_discovery.discover_mcp_tools", list)
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(cron_scheduler, "_guard_job_credential_exfil", lambda _job: None)
    success, _output, final_response, error = cron_scheduler.run_job(
        {"id": "never-keep", "name": "Never keep", "prompt": "Run", "schedule_display": "manual"})
    assert (success, final_response, error) == (True, "done", None)
    assert _SEEN == {"never_keep": True, "keeps": False}
