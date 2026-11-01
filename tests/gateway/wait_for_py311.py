"""CPython 3.11 ``asyncio.wait_for`` semantics, for tests that must reproduce its lost cancel on
any interpreter.

On 3.11, when the task awaiting ``wait_for`` is cancelled after the inner awaitable has already
completed but before the task resumed, ``wait_for`` returns the inner result (or raises the
inner's exception) and the CancelledError is swallowed; the cancel request stays counted on the
task (``Task.cancelling()``). 3.12 rewrote ``wait_for`` on ``asyncio.timeout`` and re-raises.
The hermetic test interpreter is newer than 3.11, so a test that needs the 3.11 behaviour
installs :func:`install` for the coroutine it targets.
"""

import asyncio
import contextlib

_real_wait_for = asyncio.wait_for


class Py311WaitFor:
    """Replacement for ``asyncio.wait_for`` that applies 3.11 semantics only inside tasks whose
    coroutine is named ``target`` (everything else gets the interpreter's own ``wait_for``).
    ``swallowed`` counts the cancels it lost, so a test can prove the path was exercised."""

    def __init__(self, target: str = "_keep_typing"):
        self.target = target
        self.swallowed = 0

    async def _wait_for_311(self, fut, timeout):
        # Mirrors Lib/asyncio/tasks.py wait_for in CPython 3.11 (timeout > 0 branch).
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()

        def _release(_f=None):
            if not waiter.done():
                waiter.set_result(None)

        handle = loop.call_later(timeout, _release)
        fut = asyncio.ensure_future(fut)
        fut.add_done_callback(_release)
        try:
            try:
                await waiter
            except asyncio.CancelledError:
                if fut.done():
                    self.swallowed += 1
                    return fut.result()  # 3.11: the cancel is swallowed here
                fut.remove_done_callback(_release)
                fut.cancel()
                with contextlib.suppress(BaseException):
                    await fut
                raise
            if fut.done():
                return fut.result()
            fut.remove_done_callback(_release)
            fut.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await fut
            raise TimeoutError
        finally:
            handle.cancel()

    def __call__(self, fut, timeout=None):
        task = asyncio.current_task()
        coro = task.get_coro() if task is not None else None
        name = getattr(getattr(coro, "cr_code", None), "co_name", None)
        if name == self.target and timeout is not None and timeout > 0:
            return self._wait_for_311(fut, timeout)
        return _real_wait_for(fut, timeout)


def install(monkeypatch, target: str = "_keep_typing") -> Py311WaitFor:
    patched = Py311WaitFor(target)
    monkeypatch.setattr(asyncio, "wait_for", patched)
    return patched
