"""A background child's approval after its parent turn ended, on the real Discord adapter.

Upstream's proof for the background approval transport (``test_background_child_approval_after_turn``)
uses a platform-neutral adapter double on a Telegram DM. This file repeats it with the real
``DiscordAdapter`` over a fake discord.py client, wired the way a gateway turn is
(``_run_agent_bind_turn_wiring``), for the three places a Discord session lives:

* a channel message (the card goes to the channel);
* a message inside a thread or a forum post (the card goes to the thread, never its parent);
* a session another bot's message started.

The answer goes through the production ``/approve`` and ``/deny`` handlers and through the card's own
button. Where Discord cannot show the prompt the child's command is refused, never left pending.
``approvals.mode: off`` and a smart-approval APPROVE verdict send no prompt at all, as before.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Module level on purpose: gateway.run's import side effects stat the checkout's git dir, which the
# fixture-scoped home-I/O guard would refuse inside a test.
import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.discord import adapter as discord_adapter
from tools import approval, async_delegation as ad

CHANNEL, THREAD, FORUM, FORUM_POST = "555", "777", "600", "888"
COMMAND = "rm -rf /tmp/background-child-victim"


class _Channel:
    """One Discord channel or thread: records what was posted into it."""

    def __init__(self, channel_id: str) -> None:
        self.id, self.sent, self.messages = int(channel_id), [], []

    async def send(self, content=None, **kwargs: Any):
        if content is not None:
            kwargs["content"] = content
        self.sent.append(kwargs)
        embed = kwargs.get("embed")
        message = SimpleNamespace(id=9000 + len(self.sent), edit=AsyncMock(), embeds=[embed] if embed else [])
        self.messages.append(message)
        return message

    def get_partial_message(self, _message_id):
        return self.messages[-1]


class _Discord:
    """The gateway side of one Discord bot: real adapter, fake client, bare runner, one event loop."""

    def __init__(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "config.yaml").write_text("display: {}")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.channels = {int(c): _Channel(c) for c in (CHANNEL, THREAD, FORUM, FORUM_POST)}
        self.adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        self.adapter._client = SimpleNamespace(
            get_channel=lambda cid: self.channels.get(int(cid)), fetch_channel=self._fetch_missing)
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.DISCORD: self.adapter}
        runner.config = runner.hooks = runner.session_authority = None
        runner._pending_approvals = {}
        runner._consume_pending_native_image_paths = lambda key: []
        self.runner = runner

    async def _fetch_missing(self, channel_id):
        raise RuntimeError(f"Unknown Channel {channel_id}")

    def call(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(10)

    def source(self, where: str, *, user_id: str = "42", is_bot: bool = False) -> SessionSource:
        """The source ``DiscordAdapter._handle_message`` builds: a thread or forum post is its own
        chat (``chat_id`` and ``thread_id`` are both the thread's id) under ``parent_chat_id``."""
        if where == "channel":
            return self.adapter.build_source(
                chat_id=CHANNEL, chat_type="group", user_id=user_id, is_bot=is_bot, guild_id="1", message_id="5001")
        chat, parent = (THREAD, CHANNEL) if where == "thread" else (FORUM_POST, FORUM)
        return self.adapter.build_source(
            chat_id=chat, chat_type="thread", thread_id=chat, parent_chat_id=parent, user_id=user_id,
            is_bot=is_bot, guild_id="1", message_id="5001")

    def turn(self, source: SessionSource) -> TurnRunner:
        """A turn wired like ``_run_agent_inner`` wires it: status adapter, chat and thread metadata."""
        key = self.runner._session_key_for_source(source)
        ctx = TurnContext(source=source, session_key=key, session_id="s1", user_config={},
                          message="go", _run_still_current=lambda: True)
        turn = TurnRunner(self.runner, ctx)

        async def wire():
            self.runner._run_agent_bind_turn_wiring(ctx, turn, source, source.message_id, False)

        self.call(wire())
        return turn

    def slash(self, verb: str, source: SessionSource):
        """A typed ``/approve`` or ``/deny`` through the production handler."""
        event = MessageEvent(text=f"/{verb}", message_type=MessageType.COMMAND, source=source, message_id="5002")
        handler = self.runner._handle_approve_command if verb == "approve" else self.runner._handle_deny_command
        return self.call(handler(event))

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


@pytest.fixture
def discord(tmp_path, monkeypatch):
    """Manual approvals on a messaging surface; every test starts and ends with no approval state."""
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 10)
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, True, False))
    monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 10)
    ad._reset_for_tests()
    bot = _Discord(tmp_path, monkeypatch)
    yield bot
    ad.interrupt_all()
    ad._reset_for_tests()
    with approval._lock:
        approval._gateway_notify_cbs.clear()
        approval._gateway_queues.clear()
        approval._pending.clear()
    bot.close()


class _Child:
    """A background child that asks for one dangerous command when told to, after its parent ended."""

    def __init__(self, guard=None) -> None:
        self.ask, self.done, self.decisions, self.ran = threading.Event(), threading.Event(), [], []
        self._guard = guard or (lambda: approval.check_all_command_guards(COMMAND, "local"))

    def __call__(self):
        assert self.ask.wait(5)
        decision = self._guard()
        self.decisions.append(decision)
        if decision.get("approved"):
            self.ran.append(COMMAND)  # the terminal tool runs the command only on an approved gate
        self.done.set()
        return {"status": "completed"}


def _parent_dispatches_child_then_ends(turn: TurnRunner, child) -> str:
    """The parent's run dispatches a background child and returns: the turn's notifier is gone."""
    key, dispatched = turn._ctx.session_key, {}

    def run_conversation(message, **_kw):
        dispatched.update(ad.dispatch_async_delegation(
            goal="clean up", context=None, toolsets=None, role="leaf", model=None,
            session_key=key, runner=child))
        return {"final_response": "started a helper"}

    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run_conversation), [], None, None, None)
    assert dispatched["status"] == "dispatched"
    with approval._lock:
        assert key not in approval._gateway_notify_cbs, "the parent turn ended: its notifier is gone"
    return key


def _wait(predicate, what):
    pause = threading.Event()
    for _ in range(80):
        if predicate():
            return
        pause.wait(0.1)
    raise AssertionError(what)


def _card_reached(bot: _Discord, key: str, where: str) -> dict:
    """The child's card is in the session's own channel or thread, with buttons, and nowhere else."""
    target = {"channel": CHANNEL, "thread": THREAD, "forum_post": FORUM_POST}[where]
    _wait(lambda: approval.has_blocking_approval(key) and bot.channels[int(target)].sent,
          f"the child's approval card must reach {where} {target}")
    posted = {str(cid): len(ch.sent) for cid, ch in bot.channels.items() if ch.sent}
    assert posted == {target: 1}, f"exactly one card, in the session's own {where}, never its parent: {posted}"
    card = bot.channels[int(target)].sent[0]
    assert "background-child-victim" in card["content"], "the card shows the command it asks about"
    assert card["view"].session_key == key
    pending = approval.list_gateway_approvals(key)
    assert [p["request_id"] for p in pending] == [card["view"].request_id], "/approve must find the request"
    return card


@pytest.mark.parametrize("where", ["channel", "thread", "forum_post"])
@pytest.mark.parametrize("verb", ["approve", "deny"])
def test_child_card_reaches_the_discord_chat_and_the_typed_answer_decides(discord, where, verb):
    child = _Child()
    asker = discord.source(where)
    key = _parent_dispatches_child_then_ends(discord.turn(asker), child)
    child.ask.set()
    _card_reached(discord, key, where)
    # A thread is one shared session: another member answers it. A channel session is its opener's.
    answerer = asker if where == "channel" else discord.source(where, user_id="43")
    reply = discord.slash(verb, answerer)
    assert reply, "the handler confirms the answer"
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is (verb == "approve")
    assert child.ran == ([COMMAND] if verb == "approve" else []), "approve runs the command, deny blocks it"
    assert not approval.has_blocking_approval(key)
    if verb == "deny":
        assert child.decisions[0]["outcome"] == "denied"
        assert child.decisions[0].get("status") != "pending_approval"


@pytest.mark.parametrize("where", ["thread", "channel"])
def test_session_started_by_another_bot_gets_the_card_and_a_member_answers(discord, where):
    """Another bot's message opened the session. The card still goes to that chat. In a thread any
    member's ``/approve`` answers it; in a channel the session is the opening bot's own, so a
    member's typed ``/approve`` finds nothing and the card's button is the answer surface."""
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source(where, user_id="9001", is_bot=True)), child)
    child.ask.set()
    card = _card_reached(discord, key, where)
    member = discord.source(where, user_id="43")
    if where == "thread":
        assert discord.runner._session_key_for_source(member) == key
        assert discord.slash("approve", member)
    else:
        assert discord.runner._session_key_for_source(member) != key
        discord.slash("approve", member)
        assert approval.has_blocking_approval(key), "a member's typed /approve is for the member's own session"
        view = card["view"]

        async def press():
            view._gate = AsyncMock(return_value=True)
            interaction = SimpleNamespace(
                message=discord.channels[int(CHANNEL)].messages[0], user=SimpleNamespace(display_name="member", id=43),
                response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()))
            await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")

        discord.call(press())
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is True
    assert child.ran == [COMMAND]


def test_child_in_a_thread_discord_can_no_longer_reach_is_refused_not_pending(discord):
    """The thread is gone when the child asks: neither the card nor the text prompt can be posted."""
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source("thread")), child)
    del discord.channels[int(THREAD)]
    child.ask.set()
    assert child.done.wait(10)
    decision = child.decisions[0]
    assert decision["approved"] is False
    assert decision.get("status") != "pending_approval"
    assert decision["outcome"] == "notify_failed"
    assert child.ran == []
    assert not approval.has_blocking_approval(key) and key not in approval._pending
    assert not any(ch.sent for ch in discord.channels.values()), "nothing is posted into another chat instead"


def test_child_of_a_chatless_parent_gets_the_no_approval_channel_refusal(discord):
    """A parent no chat platform carries (API server) leaves its child nowhere to ask."""
    child = _Child()
    source = SessionSource(platform=Platform.API_SERVER, chat_id="run-1", chat_type="dm", user_id="api")
    discord.runner.adapters[Platform.API_SERVER] = discord.adapter
    key = _parent_dispatches_child_then_ends(discord.turn(source), child)
    child.ask.set()
    assert child.done.wait(5)
    decision = child.decisions[0]
    assert decision["approved"] is False
    assert decision.get("status") != "pending_approval"
    assert decision["outcome"] == "notify_failed"
    assert "no approval channel" in decision["message"]
    assert child.ran == []
    assert not approval.has_blocking_approval(key) and key not in approval._pending
    assert not any(ch.sent for ch in discord.channels.values())


def test_approvals_off_child_runs_without_a_prompt(discord, monkeypatch):
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "off")
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source("thread")), child)
    child.ask.set()
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is True
    assert child.ran == [COMMAND]
    assert not any(ch.sent for ch in discord.channels.values()), "approvals off never posts a card"
    assert not approval.has_blocking_approval(key) and key not in approval._pending


def test_smart_approve_child_runs_without_a_prompt(discord, monkeypatch):
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval, "_smart_verdict", lambda *a, **k: "approve")
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source("thread")), child)
    child.ask.set()
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is True
    assert child.decisions[0].get("smart_approved") is True
    assert child.ran == [COMMAND]
    assert not any(ch.sent for ch in discord.channels.values()), "a smart APPROVE never posts a card"
    assert not approval.has_blocking_approval(key) and key not in approval._pending


def _text_prompt_reached(bot: _Discord, key: str) -> dict:
    """The plain-text ``/approve`` prompt is in the session's thread, and nowhere else."""
    _wait(lambda: approval.has_blocking_approval(key) and bot.channels[int(THREAD)].sent,
          "the child's text prompt must reach the thread")
    posted = {str(cid): len(ch.sent) for cid, ch in bot.channels.items() if ch.sent}
    assert posted == {THREAD: 1}, posted
    prompt = bot.channels[int(THREAD)].sent[0]
    assert "view" not in prompt and "/approve" in prompt["content"] and "/deny" in prompt["content"]
    return prompt


@pytest.mark.parametrize("verb", ["approve", "deny"])
def test_child_text_prompt_reaches_the_thread_when_the_card_cannot_be_sent(discord, monkeypatch, verb):
    """Discord refuses the button card: the ``/approve`` text prompt goes to the same thread."""
    from gateway.platforms.base import SendResult

    async def no_card(prompt):
        return SendResult(success=False, error="Missing Permissions")

    monkeypatch.setattr(discord.adapter, "_send_exec_approval_prompt", no_card)
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source("thread")), child)
    child.ask.set()
    _text_prompt_reached(discord, key)
    assert discord.slash(verb, discord.source("thread", user_id="43"))
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is (verb == "approve")
    assert child.ran == ([COMMAND] if verb == "approve" else [])


def test_smart_deny_override_from_a_child_is_asked_in_the_thread_as_one_operation(discord, monkeypatch):
    """Smart mode hands a DENY verdict to the human as a one-operation override: from a background
    child that request reaches the thread too, and it carries no session or permanent scope."""
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "smart")
    monkeypatch.setattr(approval, "_smart_verdict", lambda *a, **k: "deny")
    seen = []
    send_exec_approval = discord.adapter.send_exec_approval

    async def record(**kw):
        seen.append({k: kw.get(k) for k in ("allow_permanent", "allow_session", "smart_denied")})
        return await send_exec_approval(**kw)

    monkeypatch.setattr(discord.adapter, "send_exec_approval", record)
    child = _Child()
    key = _parent_dispatches_child_then_ends(discord.turn(discord.source("thread")), child)
    child.ask.set()
    _wait(lambda: approval.has_blocking_approval(key) and discord.channels[int(THREAD)].sent,
          "the override request must reach the thread")
    assert seen == [{"allow_permanent": False, "allow_session": False, "smart_denied": True}]
    assert {str(cid) for cid, ch in discord.channels.items() if ch.sent} == {THREAD}
    assert discord.slash("deny", discord.source("thread", user_id="43"))
    assert child.done.wait(5)
    assert child.decisions[0]["approved"] is False
    assert child.ran == []
