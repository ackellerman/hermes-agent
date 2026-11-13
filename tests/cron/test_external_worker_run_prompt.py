"""A manual run's transient prompt must cross the restart-safe worker handoff.

``cronjob(action='run', prompt=...)`` calls ``run_one_job(job, extra_prompt=...)``. In a managed
gateway that call hands the job to an external worker process and returns; the worker re-enters
``run_one_job`` from a serialized payload. The per-run prompt is not part of the stored job, so it
only reaches the worker if the payload carries it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

RUN_PROMPT = "This run only reads and reports; it changes nothing."


def _hand_off_through_fake_worker_spawn(scheduler, monkeypatch, tmp_path, job, extra_prompt):
    """Drive the real ``run_one_job`` down the managed-gateway path; return the spawned payload."""
    from tools.process_registry import GatewayChildDispatch

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if timeout is None:  # the reaper thread: the fake worker has nothing left to do
                self.returncode = 0
                return 0
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

    payload_files = []

    def popen(command, **_kwargs):
        source = Path(command[command.index("--external-worker-file") + 1])
        copy = tmp_path / "captured-payload.json"
        copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        payload_files.append(copy)
        Path(command[command.index("--ack-file") + 1]).write_text(
            json.dumps({"pid": 4321, "execution_id": job["execution_id"]}), encoding="utf-8")
        return FakeProcess()

    statuses = iter([
        {"id": job["execution_id"], "status": "running"},
        {"id": job["execution_id"], "status": "completed"},
    ])
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "note_cron_execution", lambda _job: None)
    monkeypatch.setattr(
        scheduler, "mark_execution_handoff_pending",
        Mock(return_value={"id": job["execution_id"], "handoff_pending": 1}))
    monkeypatch.setattr(scheduler, "get_execution", lambda _execution_id: next(statuses))
    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv",
        lambda command, **_kwargs: GatewayChildDispatch("scoped", ["scope", "--", *command]))
    monkeypatch.setattr(
        "cron.scheduler_worker_failure.record_unknown_worker_outcome", lambda *_a, **_k: True)
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)

    assert scheduler.run_one_job(job, extra_prompt=extra_prompt) is True
    assert len(payload_files) == 1, "the job was not handed to an external worker"
    return payload_files[0]


def _prompt_the_worker_builds(scheduler, monkeypatch, tmp_path, payload_file):
    """Run the worker entry point on *payload_file*; return the prompt its fire would assemble."""
    from cron.scheduler_prompt import _build_job_prompt

    received = {}

    def worker_run_one_job(job, **kwargs):
        received["prompt"] = _build_job_prompt(job, extra_prompt=kwargs.get("extra_prompt"))
        return True

    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"})
    monkeypatch.setattr(scheduler, "run_one_job", worker_run_one_job)
    assert scheduler._run_external_worker_payload(payload_file, tmp_path / "worker.ready") is True
    return received["prompt"]


def test_manual_run_prompt_reaches_the_external_worker(tmp_path, monkeypatch):
    from cron import scheduler

    job = {"id": "job-1", "execution_id": "exec-1", "name": "nightly", "prompt": "Do the work."}
    payload_file = _hand_off_through_fake_worker_spawn(
        scheduler, monkeypatch, tmp_path, job, RUN_PROMPT)

    prompt = _prompt_the_worker_builds(scheduler, monkeypatch, tmp_path, payload_file)

    assert f"## Run Context\n{RUN_PROMPT}" in prompt


def test_scheduled_fire_hands_the_worker_no_run_context(tmp_path, monkeypatch):
    from cron import scheduler

    job = {"id": "job-1", "execution_id": "exec-1", "name": "nightly", "prompt": "Do the work."}
    payload_file = _hand_off_through_fake_worker_spawn(
        scheduler, monkeypatch, tmp_path, job, None)

    prompt = _prompt_the_worker_builds(scheduler, monkeypatch, tmp_path, payload_file)

    assert "Do the work." in prompt
    assert "## Run Context" not in prompt
