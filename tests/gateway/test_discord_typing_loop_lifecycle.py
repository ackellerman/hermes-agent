"""A Discord typing loop never outlives the turn (or the adapter) that owns it.

``DiscordAdapter.send_typing`` starts one persistent ``_typing_loop`` task per chat (a typing POST
every 12 s) and returns early while that chat is registered in ``_typing_tasks``; only
``stop_typing`` (or teardown) ends it. These tests drive the production turn path
(``BasePlatformAdapter._process_message_background``) and the production adapter typing code
against a recording fake of Discord's HTTP surface; only the ``discord`` import is faked, as in
the other gateway Discord tests. The refresh interval is shrunk so a leak shows inside a
test-sized window.
"""

import asyncio
import contextlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kw: (lambda fn: fn),
        choices=lambda **kw: (lambda fn: fn),
        Choice=lambda **kw: SimpleNamespace(**kw),
    )
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", MagicMock())
    sys.modules.setdefault("discord.ext.commands", MagicMock())


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_adapter_mod  # noqa: E402
from gateway.platforms.base import SendResult  # noqa: E402
from gateway.platforms.event import MessageEvent, MessageType  # noqa: E402
from gateway.session import SessionSource, build_session_key  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from tests.gateway.wait_for_py311 import install as install_py311_wait_for  # noqa: E402

CHAT = "555000111"
INTERVAL = 0.2  # production refresh is 2.0 s


class Recorder:
    """Fake Discord HTTP. Every typing POST is timestamped; ``inflight`` models the round-trip
    and ``unwind`` the time a cancelled in-flight request takes to unwind (aiohttp)."""

    def __init__(self, inflight: float = 0.0, unwind: float = 0.0):
        self.posts: list[float] = []
        self.inflight = inflight
        self.unwind = unwind

    async def request(self, route):
        if isinstance(route, dict) and str(route.get("path", "")).endswith("/typing"):
            self.posts.append(time.monotonic())
        if self.inflight:
            try:
                await asyncio.sleep(self.inflight)
            except asyncio.CancelledError:
                if self.unwind:
                    await asyncio.sleep(self.unwind)
                raise
        return None


class FastDiscordAdapter(DiscordAdapter):
    """Production DiscordAdapter with only the generic refresh interval shrunk."""

    async def _keep_typing(self, chat_id, interval=None, metadata=None, stop_event=None):
        return await super()._keep_typing(
            chat_id, INTERVAL if interval is None else interval, metadata, stop_event)


@pytest.fixture(autouse=True)
def _route_as_dict(monkeypatch):
    monkeypatch.setattr(
        discord_adapter_mod.discord, "http",
        SimpleNamespace(Route=lambda method, path, **kw: {"method": method, "path": path, **kw}),
        raising=False,
    )


def _build(rec: Recorder | None = None):
    adapter = FastDiscordAdapter(PlatformConfig(enabled=True, token="***"))
    rec = rec or Recorder()
    adapter._client = SimpleNamespace(
        http=rec,
        tree=SimpleNamespace(command=lambda **kw: (lambda fn: fn)),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
        close=AsyncMock(),
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    return adapter, rec


def _event(text="hello", mid="1"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id=CHAT, chat_type="dm",
                             user_id="42", user_name="tester"),
        raw_message=SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock()),
        message_id=mid,
    )


def _live_typing_tasks():
    """Every not-done task running the typing machinery (registered or not)."""
    out = []
    for task in asyncio.all_tasks():
        if task.done():
            continue
        name = getattr(getattr(task.get_coro(), "cr_code", None), "co_name", None)
        if name in ("_typing_loop", "_keep_typing"):
            out.append(name)
    return sorted(out)


async def _posts_after(rec: Recorder, seconds: float) -> int:
    before = len(rec.posts)
    await asyncio.sleep(seconds)
    return len(rec.posts) - before


@contextlib.asynccontextmanager
async def _reap():
    """Cancel anything a failing test leaves behind so it cannot bleed into the next test."""
    try:
        yield
    finally:
        stray = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                 and getattr(getattr(t.get_coro(), "cr_code", None), "co_name", None)
                 in ("_typing_loop", "_keep_typing", "_process_message_background")]
        for t in stray:
            t.cancel()
        await asyncio.gather(*stray, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_ordinary_turn_leaves_no_typing_loop():
    """Clean control: an ordinary turn ends with no registration, no live loop, no POSTs."""
    async with _reap():
        adapter, rec = _build()

        async def handler(_e):
            await asyncio.sleep(INTERVAL * 3)
            return "the answer"

        adapter.set_message_handler(handler)
        ev = _event()
        await asyncio.wait_for(adapter._process_message_background(ev, build_session_key(ev.source)), 10)
        await asyncio.sleep(0)
        assert rec.posts, "the control must actually have typed during the turn"
        posts_after = await _posts_after(rec, 1.0)
        assert CHAT not in adapter._typing_tasks
        assert _live_typing_tasks() == []
        assert posts_after == 0


async def _double_cancelled_turn_leaks() -> list[str]:
    """One Mode B turn: cancelled twice, the second landing in its unwind. Returns the leak
    evidence (empty when nothing is left behind)."""
    async with _reap():
        adapter, rec = _build()
        in_handler = asyncio.Event()

        async def handler(_e):
            in_handler.set()
            await asyncio.sleep(30)
            return "never"

        adapter.set_message_handler(handler)
        ev = _event()
        task = asyncio.create_task(adapter._process_message_background(ev, build_session_key(ev.source)))
        await asyncio.wait_for(in_handler.wait(), 5)
        await asyncio.sleep(INTERVAL * 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        await asyncio.sleep(0)
        posts_after = await _posts_after(rec, INTERVAL * 5)
        leaks = []
        if not task.done():
            leaks.append("turn task still running")
        if CHAT in adapter._typing_tasks:
            leaks.append("chat still registered")
        if _live_typing_tasks():
            leaks.append(f"live typing tasks {_live_typing_tasks()}")
        if posts_after:
            leaks.append(f"{posts_after} typing POST(s) after the turn")
        return leaks


MODE_B_ROUNDS = 8


@pytest.mark.asyncio
async def test_mode_b_double_cancelled_turn_leaves_no_typing_loop(monkeypatch):
    """Mode B: the turn task is cancelled twice (the second lands in its unwind). Under CPython
    3.11 ``asyncio.wait_for`` semantics (emulated here, because the hermetic interpreter is
    newer) the refresh task's own cancel can be lost when it races ``send_typing``'s
    completion; the survivor then re-registers a Discord typing loop nobody owns once the chat is
    unpaused. Repeated so the race is actually hit: the emulation must have swallowed at least
    one refresh cancel, and no round may leave a key, a live loop, or a POST behind."""
    emulated = install_py311_wait_for(monkeypatch, "_keep_typing")
    leaks = []
    for round_no in range(MODE_B_ROUNDS):
        found = await _double_cancelled_turn_leaks()
        if found:
            leaks.append((round_no, found))
    assert emulated.swallowed >= 1, "the 3.11 lost-cancel path was never exercised (vacuous run)"
    assert leaks == [], f"{len(leaks)}/{MODE_B_ROUNDS} rounds leaked (swallowed={emulated.swallowed}): {leaks}"


@pytest.mark.asyncio
async def test_mode_b_native_wait_for_leaves_no_typing_loop():
    """The same double-cancelled turn on the interpreter's own ``wait_for`` (3.12+ re-raises
    the cancel): a guard that the fix does not depend on the emulation."""
    assert await _double_cancelled_turn_leaks() == []


@pytest.mark.asyncio
async def test_mode_a_stop_racing_send_leaves_no_orphan_loop():
    """Mode A: ``stop_typing`` is awaiting an old loop that unwinds slowly (cancel delivered
    mid-POST) while a concurrent ``send_typing`` arrives. The registry may end empty, but no
    typing loop may be left running, and nothing may POST after the stop returned."""
    async with _reap():
        inflight = 0.03
        adapter, rec = _build(Recorder(inflight=inflight, unwind=inflight))
        await adapter.send_typing(CHAT)
        for _ in range(3):
            await asyncio.sleep(0)  # the loop is inside its in-flight POST
        assert len(rec.posts) == 1

        async def concurrent_producer():
            await asyncio.sleep(inflight / 2)  # while stop_typing awaits the unwinding loop
            await adapter.send_typing(CHAT)

        await asyncio.gather(adapter.stop_typing(CHAT), concurrent_producer())
        await asyncio.sleep(0)
        # Whatever is still registered must be stoppable; nothing unregistered may be alive.
        registered = [t for t in adapter._typing_tasks.values() if not t.done()]
        live = _live_typing_tasks()
        assert len(live) == len(registered), (
            f"orphan typing loop: live={live} registered={len(registered)}")
        if registered:
            await adapter.stop_typing(CHAT)
        posts_after = await _posts_after(rec, 0.5)
        assert _live_typing_tasks() == []
        assert CHAT not in adapter._typing_tasks
        assert posts_after == 0


@pytest.mark.asyncio
async def test_stop_typing_cancelled_mid_await_still_drops_its_entry():
    """A second CancelledError aborting ``stop_typing``'s await must not leave the entry behind
    nor the loop running (guard for the cancel-then-await-then-pop ordering)."""
    async with _reap():
        adapter, rec = _build(Recorder(inflight=0.05, unwind=0.05))
        await adapter.send_typing(CHAT)
        for _ in range(3):
            await asyncio.sleep(0)
        loop_task = adapter._typing_tasks[CHAT]
        stopper = asyncio.create_task(adapter.stop_typing(CHAT))
        await asyncio.sleep(0.01)  # stop_typing is awaiting the unwinding loop
        stopper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopper
        await asyncio.sleep(0.2)
        assert loop_task.done()
        assert CHAT not in adapter._typing_tasks
        assert await _posts_after(rec, 0.3) == 0


@pytest.mark.asyncio
async def test_disconnect_stops_unowned_typing_loops():
    """Teardown: loops started with no owning turn (proxy-stream, queued follow-up and
    progress-restore call sites) are cancelled by ``disconnect()``, before ``client.close()``."""
    async with _reap():
        adapter, rec = _build()
        order = []
        adapter._client.close = AsyncMock(
            side_effect=lambda: order.append(("close", _live_typing_tasks())))
        await adapter.send_typing(CHAT)
        await adapter.send_typing("555000222")
        await asyncio.sleep(0)
        assert _live_typing_tasks() == ["_typing_loop", "_typing_loop"]
        await adapter.disconnect()
        assert order == [("close", [])], "typing loops must be gone before client.close()"
        assert adapter._typing_tasks == {}
        assert _live_typing_tasks() == []
        assert await _posts_after(rec, 0.3) == 0


@pytest.mark.asyncio
async def test_cancel_background_tasks_clears_typing_registry():
    """The gateway's bounded teardown runs ``cancel_background_tasks()`` first; it must also
    leave no typing loop behind (a reconcile builds a NEW adapter; nobody holds the old one)."""
    async with _reap():
        adapter, rec = _build()
        await adapter.send_typing(CHAT)
        await asyncio.sleep(0)
        await adapter.cancel_background_tasks()
        assert adapter._typing_tasks == {}
        assert _live_typing_tasks() == []
        assert await _posts_after(rec, 0.3) == 0


@pytest.mark.asyncio
async def test_send_typing_refused_while_disconnecting():
    """Between ``disconnect()``'s typing clear and ``client.close()`` a straggler turn (or an
    owner-less call site) must not register a fresh loop on the adapter being torn down."""
    async with _reap():
        adapter, rec = _build()
        adapter._disconnecting = True
        await adapter.send_typing(CHAT)
        await asyncio.sleep(0)
        assert adapter._typing_tasks == {}
        assert _live_typing_tasks() == []
        assert rec.posts == []
        # A reconnect clears the flag (connect() resets it), and typing works again.
        adapter._disconnecting = False
        await adapter.send_typing(CHAT)
        await asyncio.sleep(0)
        assert CHAT in adapter._typing_tasks
        await adapter.stop_typing(CHAT)
        assert _live_typing_tasks() == []


@pytest.mark.asyncio
async def test_disconnect_refuses_typing_registered_during_teardown():
    """A turn that asks for typing while ``disconnect()`` is already past its typing clear (here:
    while the bot task is being cancelled) leaves no loop behind."""
    async with _reap():
        adapter, rec = _build()
        straggler = []

        async def slow_cancel_bot_task():
            await adapter.send_typing(CHAT)  # the straggler, mid-teardown
            straggler.append(dict(adapter._typing_tasks))

        adapter._cancel_bot_task = slow_cancel_bot_task
        await adapter.disconnect()
        assert straggler == [{}]
        assert adapter._typing_tasks == {}
        assert _live_typing_tasks() == []
        assert await _posts_after(rec, 0.3) == 0
