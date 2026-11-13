"""Who and what may answer a pending command approval once a background child's request can
outlive its parent turn.

* A bare ``yes`` / ``ok`` / ``y`` / ``always`` / thumbs-up answers only a request the running turn is
  itself blocked on. A background child's request (its queue entry carries the child's notify lease as
  ``owner``) is answered by the card's button or an explicit ``/approve`` / ``/deny``.
* The reply to a typed ``/approve`` or ``/deny`` names the command it resolved and says how many
  requests are still pending.
* The slash access policy (``allow_admin_from`` / ``group_allow_admin_from``) decides who may type
  ``/approve`` and ``/deny``; the bare-word route obeys the same policy. With the policy set, another
  bot's ``/approve`` is refused and the admin's still works; with it unset every admitted author
  answers, as before.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.test_discord_background_child_approval import (  # `discord` is a fixture
    COMMAND, THREAD, _Child, _parent_dispatches_child_then_ends, _wait, discord,
)
from tools import approval, async_delegation as ad
from tools.approval_gateway_wait import _ApprovalEntry

SECOND = "rm -rf /tmp/second-victim"
OWNER, OTHER_BOT = "42", "9001"


@pytest.fixture(autouse=True)
def no_standing_approvals():
    """No session or permanent grant leaks between cases: one would answer the next case's request."""
    def clear():
        with approval._lock:
            approval._session_approved.clear()
            approval._permanent_approved.clear()
    clear()
    yield
    clear()


def _bare(bot, text: str, source: SessionSource, key: str) -> bool:
    """A bare-word message that arrives while a turn of the session runs."""
    bot.runner._delivery_adapter_for = lambda _source: bot.adapter
    bot.runner._send_busy_reply = AsyncMock()
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="5003")
    return bot.call(bot.runner._route_plaintext_approval_while_busy(event, key))


def _child_request_pending(bot) -> tuple[str, _Child]:
    child = _Child()
    key = _parent_dispatches_child_then_ends(bot.turn(bot.source("thread")), child)
    child.ask.set()
    _wait(lambda: approval.has_blocking_approval(key) and bot.channels[int(THREAD)].sent, "the child's card")
    return key, child


def _later_turn_asks(bot, key: str, inside) -> dict:
    """A later turn of the same session raises its own approval and blocks on it; ``inside`` runs
    while it is blocked. Returns the turn's decision under ``"decision"``."""
    box: dict = {}

    def run_conversation(message, **_kw):
        ctx = contextvars.copy_context()
        worker = threading.Thread(target=lambda: ctx.run(
            lambda: box.update(decision=approval.check_all_command_guards(SECOND, "local"))))
        worker.start()
        _wait(lambda: len(approval.list_gateway_approvals(key)) == 2, "the turn's own request")
        inside(box)
        worker.join(5)
        return {"final_response": "ok"}

    turn = bot.turn(bot.source("thread"))
    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run_conversation), [], None, None, None)
    return box


# -- bare words ---------------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["yes", "ok", "y", "always", "session", "👍"])
def test_bare_word_during_a_later_turn_is_an_ordinary_message_for_a_childs_request(discord, word):
    """The parent replied, the child's card is open, a later turn runs: ``yes`` answers nothing."""
    key, child = _child_request_pending(discord)
    with approval._lock:
        permanent_before = set(approval._permanent_approved)

    consumed = _bare(discord, word, discord.source("thread", user_id="43"), key)

    assert consumed is False, "the word goes on to the running turn as an ordinary message"
    assert approval.has_blocking_approval(key), "the child's request is still pending"
    assert not child.done.is_set() and child.ran == []
    with approval._lock:
        assert set(approval._permanent_approved) == permanent_before, "no allowlist entry was written"
        assert not approval._session_approved.get(key)
    assert discord.slash("deny", discord.source("thread", user_id="43"))
    assert child.done.wait(5) and child.ran == []


def test_bare_no_does_not_deny_a_childs_request(discord):
    key, child = _child_request_pending(discord)
    assert _bare(discord, "no", discord.source("thread", user_id="43"), key) is False
    assert approval.has_blocking_approval(key) and not child.done.is_set()
    assert discord.slash("approve", discord.source("thread", user_id="43"))
    assert child.done.wait(5) and child.ran == [COMMAND]


def test_bare_yes_answers_the_running_turns_own_request_and_not_an_older_childs(discord):
    """The child's request is older; the running turn is blocked on its own: ``yes`` is for the turn."""
    key, child = _child_request_pending(discord)

    def inside(box):
        box["order"] = [p["command"] for p in approval.list_gateway_approvals(key)]
        box["consumed"] = _bare(discord, "yes", discord.source("thread", user_id="43"), key)

    box = _later_turn_asks(discord, key, inside)

    assert box["order"] == [COMMAND, SECOND], "the child's request is the oldest in the queue"
    assert box["consumed"] is True
    assert box["decision"]["approved"] is True, "the turn's own command was approved"
    assert not child.done.is_set() and child.ran == [], "the child's command was not"
    assert [p["command"] for p in approval.list_gateway_approvals(key)] == [COMMAND]
    assert discord.slash("deny", discord.source("thread", user_id="43"))
    assert child.done.wait(5)


def test_bare_yes_still_answers_a_foreground_request(discord):
    """No child involved: the turn blocked on its own request is answered by ``yes``, as before."""
    source = discord.source("thread")
    key = discord.runner._session_key_for_source(source)
    entry = _ApprovalEntry({"command": SECOND})
    with approval._lock:
        approval._gateway_queues.setdefault(key, []).append(entry)

    assert _bare(discord, "yes", source, key) is True
    assert entry.event.is_set() and entry.result == "once"
    assert not approval.has_blocking_approval(key)
