"""The generic typing refresh survives a cancel a platform adapter swallowed inside ``send_typing``,
and the turn's final typing stop does not disturb a refresh task that is legitimately unwinding.

When ``send_typing`` returns from inside ``except CancelledError`` (the adapter swallowed the
refresh task's cancel), the tick ends normally while the cancel request stays counted on the task
(``Task.cancelling()``). A refresh that ignores that keeps re-creating platform typing after its
turn stopped it. The ``cancelling()`` guard at the end of the tick is the defence; these tests
witness it directly (self-cancelling ``send_typing``), which also stays live now that the tick's
per-tick bound is ``asyncio.timeout`` (the upstream carry c6b6a8a8d9) instead of 3.11's
``wait_for``.
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter

CHAT = "-1001"


@pytest.mark.asyncio
async def test_refresh_exits_when_send_typing_swallows_its_cancel():
    """Deterministic: the refresh task's ``send_typing`` cancels its own task and swallows the
    CancelledError. The tick ends normally while the cancel stays counted; the guard must honour
    it: the refresh ends here and never calls ``send_typing`` again."""
    adapter = ProgressCaptureAdapter()
    calls = []

    async def send_typing(chat_id, metadata=None):
        calls.append(chat_id)
        if len(calls) == 2:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()  # cancel the refresh task from inside its own tick
            try:
                await asyncio.sleep(0)  # deliver the cancel to this frame
            except asyncio.CancelledError:
                return  # platform adapter swallows the cancel

    adapter.send_typing = send_typing
    refresh = asyncio.create_task(adapter._keep_typing(CHAT, interval=0.05))
    try:
        done, _ = await asyncio.wait({refresh}, timeout=1.0)
        assert refresh in done, f"refresh survived a swallowed cancel ({len(calls)} send_typing calls)"
        assert len(calls) == 2, calls
        assert CHAT not in adapter._typing_paused
    finally:
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)


@pytest.mark.asyncio
async def test_refresh_keeps_running_without_a_cancel():
    """Control: with no cancel issued, the refresh keeps refreshing."""
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
