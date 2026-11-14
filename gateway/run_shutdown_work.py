"""Active-work accounting split out of ``gateway/run_shutdown.py``: the counts the restart wait and
the ``stop()`` drain read, and the one dict per unit that names them to an observer.

Bound onto ``GatewayRunner`` through ``GatewayShutdownMixin``'s bases. Background ``delegate_task``
units live in a process-global registry (one gateway process serves every profile), so their source
here is module-level and fail-soft.
"""

from __future__ import annotations

import os
import time
from contextlib import suppress

from gateway.session_runtime_workers import uncounted_runtime_work_units


def active_delegation_count() -> int:
    """Live background ``delegate_task`` units in this process, across every served profile.

    The registry (``tools.async_delegation``) is module-global and one gateway process serves every
    profile, so this counts all of their children. Module-level, not a method: the shutdown-path
    doubles that drive these counters are bare ``SimpleNamespace`` objects with no registry of their
    own. Fail-soft like ``_active_api_run_count``: an unreadable registry counts as 0.
    """
    try:
        from tools.async_delegation import active_count
        return max(0, int(active_count()))
    except Exception:  # health: allow BLE001 -- a drain must never be the thing that raises: an unreadable registry is 0 live units, exactly as _active_api_run_count reads a dead adapter
        return 0


def _delegation_units(now: float) -> list[dict]:
    """One unit per live background delegation: its id, its goal (80 chars) and its age."""
    from tools.async_delegation import _LIVE_STATES, list_async_delegations

    units = []
    for record in list_async_delegations():
        if record.get("status") not in _LIVE_STATES:
            continue
        unit: dict = {"kind": "delegation", "delegation_id": record.get("delegation_id"),
                      "goal": str(record.get("goal") or "")[:80], "pid": os.getpid()}
        if record.get("dispatched_at"):
            unit["elapsed_s"] = round(now - float(record["dispatched_at"]), 1)
        units.append(unit)
    return units


class GatewayShutdownWorkMixin:
    """Active-work reporting for the shutdown/restart path."""

    def _describe_active_work(self) -> list:
        """One dict per in-flight work unit the restart wait is holding for, so an observer
        (``hermes update``, ``hermes gateway status``) can name it instead of printing a bare count.

        ``kind`` ∈ ``chat`` (session turn), ``cron`` (job id + external worker pid when the run was handed to a
        restart-safe scope), ``managed`` (session + worker pid when a worker holds it), ``api`` / ``deferred``
        (count only), ``delegation`` (background ``delegate_task`` unit: delegation id + goal). Best-effort: a
        source that can't be read is omitted, never raises."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        now = time.time()
        units: list = []
        for key, state in list(self._sessions_map().items()):
            agent = state.turn.agent
            if agent is None:
                continue
            unit: dict = {"kind": "chat", "session": key, "pid": os.getpid()}
            if state.turn.started_ts:
                unit["elapsed_s"] = round(now - state.turn.started_ts, 1)
            if agent is not _AGENT_PENDING_SENTINEL:
                unit["model"] = getattr(agent, "model", None)
                summary_fn = getattr(agent, "get_activity_summary", None)
                if callable(summary_fn):
                    with suppress(Exception):
                        summary = summary_fn()
                        unit["current_tool"] = summary.get("current_tool")
                        unit["idle_s"] = summary.get("seconds_since_activity")
            units.append(unit)
        with suppress(Exception):
            from cron.scheduler import get_running_job_details, get_wedged_job_ids
            wedged = get_wedged_job_ids()
            for job in get_running_job_details():
                units.append({"kind": "cron", "job_id": job["job_id"], "elapsed_s": job["elapsed_s"],
                              "pid": job["worker_pid"] or os.getpid(), "external": bool(job["worker_pid"]),
                              "wedged": job["job_id"] in wedged,
                              "restart_safe": bool(job.get("restart_safe"))})
        units.extend(uncounted_runtime_work_units(self, os.getpid()))
        for kind, count in (("api", self._active_api_run_count()), ("deferred", self._active_deferred_agent_worker_count())):
            units.extend({"kind": kind, "pid": os.getpid()} for _ in range(count))
        with suppress(Exception):
            units.extend(_delegation_units(now))
        return units
