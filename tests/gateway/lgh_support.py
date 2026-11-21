"""Shared harness for the loop-guard-hold tests (plans/loop-guard-hold/spec.md r15). Not a conftest.

THE CONTRACT. Spec r15 fixes behaviour; the tests need handles on it. These names are what the build provides
(RED fails on base because they do not exist or behave as base). Nothing else is assumed beyond names the
spec itself gives (``send_hold_notice``, ``_report_human_seen``, ``bot_loop_hold_refs.json``, the config keys,
``handle_message`` of the serving adapter, ``_is_session_running``, ``_active_sessions``).

Settings (``gateway.bot_loop_guard``; ``BotLoopGuardSettings`` fields and ``settings_from_config`` keys):
  hold_after_messages=10, hold_window_seconds=120.0, hold_initial_seconds=15.0, hold_max_seconds=480.0,
  hold_runaway_at_cap=3, hold_queue_max=50.

Guard (``BotLoopGuard(settings=..., clock=...)``; the clock is the only time source the hold reads):
  ``guard.held_events(conversation) -> list[MessageEvent]``   FIFO, the very objects held (in memory)
  ``guard.hold_level(conversation) -> int``                  S3 ``level`` (1 when nothing is backing off)
  ``guard.is_paused(conversation) -> bool``                  S5 runaway pause
  ``guard.is_released_event(seat, event) -> bool``           S2 released set, keyed (seat, channel, message id)
                                                             or the voice ``source`` object by identity
Runner (``GatewayRunner``; ``conversation`` is ``runner._bot_loop_guard_conversation(source)``):
  ``await runner._hm_admit_event(event)`` returns None for a held message (as for any dropped one) and a
      ``(event, source, False)`` tuple for an admitted one; busy path ``_handle_busy_session_message`` likewise.
  ``await runner._bot_loop_hold_pump_once()``   ONE pump wake at the guard's clock: hands over what is due, one
      at a time, through ``adapter.handle_message(event)`` of ``runner._adapter_for_source(event.source)``
      (entries whose adapter is None stay queued), never over a running turn (S4), nothing while
      ``runner._draining``, ``runner._startup_restore_in_progress`` or estop pause (``runner._hm_estop_paused()``
      truthy) (S8), and sends notices through ``adapter.send_hold_notice(chat_id, text)``.
  ``runner._bot_loop_hold_turn_claimed(event)`` / ``runner._bot_loop_hold_turn_ended(event)``  the two guard
      callbacks the runner calls at slot claim and slot release (S4); tests call them where the runner would.
  ``runner._bot_loop_human_seen(source, text, *, voice=False)``  the runner-side sink of a person's message:
      the reset and release (S7); the Discord adapter's ``_report_human_seen(message)`` calls it with the
      source it builds for the dropped message. ``runner._admit_bot_chain(event)`` for an admitted person
      calls the same decision, so ``/stop`` handling is one rule (``guard.is_stop_text(text)``).
  ``await runner._bot_loop_hold_report_lost()``  S8 startup lines, after adapters connect.
  ``runner._bot_loop_hold_flush_refs()``          S8 graceful-stop flush. The refs file is
      ``<profile home>/gateway/bot_loop_hold_refs.json`` where the profile home is ``HERMES_HOME`` (tests set it).
Adapter fake (see ``make_adapter``): ``handle_message`` (AsyncMock), ``send_hold_notice`` (AsyncMock returning a
  ``SendResult``), ``_active_sessions`` (dict), ``name``. A real ``DiscordAdapter`` provides
  ``send_hold_notice(chat_id, text)`` and ``_report_human_seen(message)``.
Notice text: ``gateway.bot_loop_guard.format_hold_notice(senders, seat, delay_seconds, held, replies_to_replies) -> str``
  (S6 wording: names escaped and cut to 32, at most 5 named then "and N more").
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.bot_loop_guard import BotLoopGuard, settings_from_config
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import Platform, SessionSource

CHAT = "1571780464853647400"
THREAD = "1440000000000000001"
BOT_A = "1476257859233644656"   # peer bots
BOT_B = "1476257859233644657"
BOT_C = "1476257859233644658"
PERSON = "100200300"
PERSON2 = "100200301"
RELAY = "1571240603155435684"

_IDS = itertools.count(1_000_000)


class FakeClock:
    def __init__(self, start: float = 10_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    __call__ = now


def cfg_settings(**block):
    """Settings the way a seat parses them from its config.yaml block (``bot_loop_guard.*``)."""
    return settings_from_config({"gateway": {"bot_loop_guard": block}})


class Cfg:
    """Mutable holder, so a test can edit the config live (the guard re-reads per call)."""

    def __init__(self, **block) -> None:
        self.block = block

    def set(self, **block) -> None:
        self.block = block

    def __call__(self):
        return cfg_settings(**self.block)


def make_adapter(name: str = "discord"):
    adapter = SimpleNamespace(name=name, handle_message=AsyncMock(), _active_sessions={})
    sent = []

    async def _send_hold_notice(chat_id, text, *args, **kwargs):
        sent.append((str(chat_id), text))
        return SendResult(success=True, message_id=str(next(_IDS)))

    adapter.send_hold_notice = AsyncMock(side_effect=_send_hold_notice)
    adapter.notices = sent
    return adapter


def make_runner(clock: FakeClock, cfg, adapter=None, *, profile: str = "boba"):
    """A bare GatewayRunner whose ingress gate and hold run against fakes. Admission is not under test."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._adapter_profile_for_source = lambda source: profile
    runner.adapter = adapter if adapter is not None else make_adapter()
    runner._adapter_for_source = lambda source: runner.adapter
    runner._hm_estop_paused = lambda: False
    runner._bot_loop_guard = BotLoopGuard(settings=cfg, clock=clock.now)
    return runner


def source(user: str, *, is_bot: bool = True, chat: str = THREAD, chat_type: str = "thread",
           platform=Platform.DISCORD, name: str | None = None):
    return SessionSource(platform=platform, chat_id=chat, chat_type=chat_type, user_id=user,
                         user_name=name or f"u{user}", is_bot=is_bot)


def event(user: str, text: str | None = None, *, is_bot: bool = True, chat: str = THREAD,
          platform=Platform.DISCORD, message_id: str | None = None, name: str | None = None):
    n = next(_IDS)
    return MessageEvent(text=text if text is not None else f"ask {n}", message_id=message_id or f"m{n}",
                        source=source(user, is_bot=is_bot, chat=chat, platform=platform, name=name))


def conv(runner, ev):
    return runner._bot_loop_guard_conversation(ev.source)


async def admit(runner, ev) -> bool:
    """The cold-path ingress gate."""
    return await runner._hm_admit_event(ev) is not None


async def fill_window(runner, clock, n: int = 10, *, user: str = BOT_A, gap: float = 1.0, chat: str = THREAD):
    """n admitted bot messages (distinct texts) spaced ``gap`` seconds; asserts all were admitted."""
    for _ in range(n):
        assert await admit(runner, event(user, chat=chat)) is True
        clock.advance(gap)


async def pump(runner, clock, advance: float = 0.0) -> None:
    if advance:
        clock.advance(advance)
    await runner._bot_loop_hold_pump_once()


def handed(adapter) -> list:
    return [c.args[0] for c in adapter.handle_message.await_args_list]


def start_turn(runner, ev) -> str:
    """Mark a running turn under the event's session key (what ``_claim_active_session_slot`` guards)."""
    key = runner._session_key_for_source(ev.source)
    runner._session_state(key).turn.agent = object()
    return key


def end_turn(runner, key: str) -> None:
    runner._session_state(key).turn.agent = None


@pytest.fixture
def clock():
    return FakeClock()
