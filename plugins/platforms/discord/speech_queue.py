"""One speech queue per guild: a clip sounds only while no other clip of its guild does, in arrival
order, so the bot never talks over itself.

Stdlib only. The adapter holds one :class:`GuildSpeechQueue` per guild for the life of the process;
guilds never wait for each other, because each has its own lock.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator


def next_clip_id() -> str:
    """A clip token unique for the life of the process: 12 hex chars, so it holds no space, comma
    or parenthesis and needs no shared counter (a second import of this module cannot repeat one)."""
    return uuid.uuid4().hex[:12]


class GuildSpeechQueue:
    """FIFO turn-taking for one guild's speech.

    ``asyncio.Lock`` wakes its waiters in arrival order and survives a cancelled waiter, which is
    exactly the contract here. ``_pending`` counts the callers holding or waiting for a turn, so
    :meth:`busy` stays True through the gap between one clip's release and the next waiter waking.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending = 0

    def busy(self) -> bool:
        """True while a clip holds the turn or any clip waits for it."""
        return self._pending > 0 or self._lock.locked()

    @asynccontextmanager
    async def turn(self) -> AsyncIterator[int]:
        """Wait for this guild's turn; yields the whole milliseconds spent waiting.

        The turn is released on every exit path — normal return, error, timeout or cancellation of
        the caller — so a clip that dies can never wedge the guild's queue.
        """
        self._pending += 1
        try:
            started = time.monotonic()
            async with self._lock:
                yield int((time.monotonic() - started) * 1000)
        finally:
            self._pending -= 1
