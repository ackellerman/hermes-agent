"""A gateway restart and its stop() drain wait for live background ``delegate_task`` units.

Observed live on 2026-11-14: ``SIGUSR1 received ... performing a graceful gateway restart`` at
02:14:49,290 and ``Stopping gateway for restart...`` 52 ms later, then ``interrupted 1 background
delegation(s)`` — the after-turn wait counted chat turns, cron runs, managed turns, api runs and
deferred workers but never a background delegation, so it saw nothing to hold for and ``stop()``
amputated the child. The same blind spot kept them out of the stop() drain, of
``gateway_state.json``'s ``active_agents``, of the draining status report and of the ``active-work``
verb.

Each test drives the real ``tools.async_delegation`` registry: a real unit whose child blocks on a
``threading.Event``, so every count asserted here is a live unit and not a stub.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_active_work_verb import active_work_verb
from tests.gateway.restart_test_helpers import make_restart_runner
from tests.waiting import async_wait_until, wait_until
from tools import async_delegation as delegation

SESSION_KEY = "agent:main:discord:thread:1"


@pytest.fixture(autouse=True)
def _clean_delegations():
    """The delegation registry is process-global: start and end each test with nothing in it."""
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A restart runner whose counts are the real ones (bare ``object.__new__`` instance)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner, _adapter = make_restart_runner()
    runner._scale_to_zero_status = MagicMock()
    return runner


def _dispatch_blocked_child(release: threading.Event, goal: str = "gated child") -> str:
    """Dispatch a real background unit whose child blocks on *release*; return its delegation id."""
    started = threading.Event()

    def _child() -> dict:
        started.set()
        release.wait()
        return {"status": "completed"}

    handle = delegation.dispatch_async_delegation(
        goal=goal, context=None, toolsets=None, role="leaf", model=None, session_key=SESSION_KEY,
        runner=_child, interrupt_fn=release.set,
    )
    assert handle["status"] == "dispatched", handle
    wait_until(started.is_set, timeout=5.0, what="the background child to start")
    return handle["delegation_id"]


def _stop_recorder(stop_event: asyncio.Event) -> AsyncMock:
    """``runner.stop`` stand-in that records its call and sets *stop_event* (so a test can wait on it)."""

    async def _stop(**_kwargs):
        stop_event.set()

    return AsyncMock(side_effect=_stop)


def test_live_delegation_is_active_and_awaitable_work(runner):
    """Both totals the restart consults count a live child, and neither lingers after it finishes."""
    release = threading.Event()
    try:
        _dispatch_blocked_child(release)
        assert runner._active_work_count() == 1
        assert runner._awaitable_work_count() == 1
    finally:
        release.set()
    wait_until(lambda: delegation.active_count() == 0, timeout=5.0,
               what="the delegation to finish")
    assert runner._active_work_count() == 0
    assert runner._awaitable_work_count() == 0


@pytest.mark.asyncio
async def test_restart_wait_holds_stop_until_the_delegation_finishes(runner):
    """The after-turn wait defers stop() while a background child is the only live work."""
    runner._restart_after_turn_timeout = 30.0
    stop_called = asyncio.Event()
    runner.stop = _stop_recorder(stop_called)
    release = threading.Event()
    try:
        _dispatch_blocked_child(release)
        assert runner.request_restart(detached=False, via_service=True) is True
        # Half a second in the child is still running, so stop() must not have run.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stop_called.wait(), timeout=0.5)
        runner.stop.assert_not_awaited()
    finally:
        release.set()
    await asyncio.wait_for(runner._restart_task, timeout=5.0)
    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )


@pytest.mark.asyncio
async def test_restart_wait_stops_at_the_cap_with_the_delegation_still_running(runner):
    """The cap still ends the wait — a child cannot pin the gateway past restart_after_turn_timeout."""
    runner._restart_after_turn_timeout = 0.5
    stop_called = asyncio.Event()
    runner.stop = _stop_recorder(stop_called)
    release = threading.Event()
    try:
        _dispatch_blocked_child(release)
        started = time.monotonic()
        assert runner.request_restart(detached=False, via_service=True) is True
        await asyncio.wait_for(runner._restart_task, timeout=5.0)
        elapsed = time.monotonic() - started
        assert 0.45 <= elapsed < 3.0, elapsed
        runner.stop.assert_awaited_once_with(
            restart=True, detached_restart=False, service_restart=True
        )
        # Never released: stop() interrupted it after the cap, exactly as for a timed-out turn.
        assert delegation.active_count() == 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_stop_drain_waits_for_the_delegation_on_the_drain_deadline(runner):
    """stop()'s drain counts the child as a chat turn: it waits, then gives up on its own deadline."""
    assert runner._drain_work_counts() == (0, 0, 0, 0)
    release = threading.Event()
    try:
        _dispatch_blocked_child(release)
        assert runner._drain_work_counts()[0] == 1
        started = time.monotonic()
        _snapshot, timed_out = await runner._drain_active_agents(0.5)
        elapsed = time.monotonic() - started
        assert timed_out is True
        assert elapsed >= 0.45, elapsed
    finally:
        release.set()
    await async_wait_until(lambda: delegation.active_count() == 0, timeout=5.0,
                           what="the first delegation to finish")

    # Control: a child that finishes inside the budget ends the drain, it does not run it out.
    second = threading.Event()
    try:
        _dispatch_blocked_child(second, goal="quick child")
        asyncio.get_running_loop().call_later(0.2, second.set)
        started = time.monotonic()
        _snapshot, timed_out = await runner._drain_active_agents(5.0)
        elapsed = time.monotonic() - started
    finally:
        second.set()
    assert timed_out is False
    assert elapsed < 3.0, elapsed


def test_draining_status_and_active_work_verb_name_the_delegation_once(runner):
    """The draining status (gateway_state.json, the wait's log line) and the verb each name it once."""
    release = threading.Event()
    try:
        delegation_id = _dispatch_blocked_child(release)
        delegation_units = [u for u in runner._describe_active_work() if u["kind"] == "delegation"]
        assert len(delegation_units) == 1, delegation_units
        assert delegation_units[0]["delegation_id"] == delegation_id
        answer = active_work_verb(runner, SimpleNamespace(call_soon_threadsafe=lambda fn: fn()))({})
        assert answer["active_agents"] == 1
        assert [u["kind"] for u in answer["active_work"]] == ["delegation"]
        assert answer["active_work"][0]["delegation_id"] == delegation_id
    finally:
        release.set()
