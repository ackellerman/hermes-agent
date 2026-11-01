"""The generic typing refresh honours a cancel that CPython 3.11 ``asyncio.wait_for`` swallowed,
and the turn's final typing stop does not disturb a refresh task that is legitimately unwinding.

On 3.11, ``wait_for(send_typing(...))`` returns the inner result when the refresh task's cancel
races ``send_typing``'s completion; the cancel is lost but stays counted (``cancelling()``). A
refresh that ignores that keeps re-creating platform typing after its turn stopped it. The
3.11 semantics are emulated (``tests/gateway/wait_for_py311.py``) because the hermetic test
interpreter is newer.
"""

import asyncio
import time

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter
from tests.gateway.wait_for_py311 import install as install_py311_wait_for

CHAT = "-1001"


@pytest.mark.asyncio
async def test_refresh_exits_when_its_cancel_is_swallowed_by_311_wait_for(monkeypatch):
    """Deterministic: the refresh task is cancelled while its ``send_typing`` is completing (the
    cancel is issued from inside that call), so the emulated 3.11 ``wait_for`` swallows it. The
    refresh must still end, and must not call ``send_typing`` again."""
    emulated = install_py311_wait_for(monkeypatch, "_keep_typing")
    adapter = ProgressCaptureAdapter()
    refresh: list[asyncio.Task] = []
    calls = []

    async def send_typing(chat_id, metadata=None):
        calls.append(time.monotonic())
        if len(calls) == 2:
            refresh[0].cancel()  # lands while this inner call completes in the same step

    adapter.send_typing = send_typing
    refresh.append(asyncio.create_task(adapter._keep_typing(CHAT, interval=0.05)))
    try:
        for _ in range(100):
            if refresh[0].done() or len(calls) > 4:
                break
            await asyncio.sleep(0.02)
        assert emulated.swallowed == 1, "the emulated 3.11 wait_for must have lost the cancel"
        assert refresh[0].done(), f"refresh survived its swallowed cancel ({len(calls)} send_typing calls)"
        assert len(calls) == 2
        assert CHAT not in adapter._typing_paused
    finally:
        refresh[0].cancel()
        await asyncio.gather(refresh[0], return_exceptions=True)


@pytest.mark.asyncio
async def test_refresh_keeps_running_without_a_cancel(monkeypatch):
    """Control: under the same emulation, an uncancelled refresh keeps refreshing."""
    install_py311_wait_for(monkeypatch, "_keep_typing")
    adapter = ProgressCaptureAdapter()
    task = asyncio.create_task(adapter._keep_typing(CHAT, interval=0.05))
    try:
        await asyncio.sleep(0.3)
        assert not task.done()
        assert sum(1 for t in adapter.typing if not (t["metadata"] or {}).get("stopped")) >= 3
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class SlowRefreshStop(ProgressCaptureAdapter):
    """Non-Discord adapter whose refresh-task ``stop_typing`` (run from ``_keep_typing``'s
    finally) is slower than the turn's 0.5 s shielded wait."""

    SLOW = 1.0

    def __init__(self):
        super().__init__()
        self.log = []

    async def stop_typing(self, chat_id):
        task = asyncio.current_task()
        name = getattr(getattr(task.get_coro(), "cr_code", None), "co_name", None)
        if name == "_keep_typing":
            self.log.append("refresh-stop-start")
            try:
                await asyncio.sleep(self.SLOW)
            except asyncio.CancelledError:
                self.log.append("refresh-stop-CANCELLED")
                raise
            self.log.append("refresh-stop-done")
        else:
            self.log.append("turn-stop")
        await super().stop_typing(chat_id)


@pytest.mark.asyncio
async def test_final_stop_does_not_recancel_an_unwinding_refresh():
    """The turn's final bounded stop must not cancel a refresh task that is still in its own
    cleanup: that would abort the platform stop and leave ``_status_text`` behind."""
    adapter = SlowRefreshStop()

    async def handler(_e):
        adapter.set_status_text(CHAT, "running tool X")
        await asyncio.sleep(0.3)
        return "done"

    adapter.set_message_handler(handler)
    ev = MessageEvent(text="hi", message_type=MessageType.TEXT,
                      source=SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm"),
                      message_id="m1")
    await asyncio.wait_for(adapter._process_message_background(ev, "agent:main:telegram:dm:-1001"), 5)
    await asyncio.sleep(SlowRefreshStop.SLOW + 0.3)
    assert "refresh-stop-CANCELLED" not in adapter.log, adapter.log
    assert "refresh-stop-done" in adapter.log, adapter.log
    assert dict(getattr(adapter, "_status_text", {})) == {}
    assert CHAT not in adapter._typing_paused
