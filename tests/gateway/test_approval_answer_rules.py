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


# -- the reply names what it resolved -----------------------------------------------------------


def test_typed_approve_names_the_command_and_says_how_many_remain(discord):
    first = _Child()
    second = _Child(guard=lambda: approval.check_all_command_guards(SECOND, "local"))
    turn = discord.turn(discord.source("thread"))
    key, dispatched = turn._ctx.session_key, []

    def run_conversation(message, **_kw):
        for child in (first, second):
            dispatched.append(ad.dispatch_async_delegation(
                goal="g", context=None, toolsets=None, role="leaf", model=None, session_key=key, runner=child))
        return {"final_response": "started two helpers"}

    turn._run_conversation_with_approval(SimpleNamespace(run_conversation=run_conversation), [], None, None, None)
    assert [d["status"] for d in dispatched] == ["dispatched", "dispatched"]
    first.ask.set()
    _wait(lambda: len(approval.list_gateway_approvals(key)) == 1, "first request")
    second.ask.set()
    _wait(lambda: len(approval.list_gateway_approvals(key)) == 2, "second request")

    reply = discord.slash("approve", discord.source("thread", user_id="43"))

    assert first.done.wait(5) and first.ran == [COMMAND], "one /approve takes the oldest request"
    assert not second.done.is_set()
    assert "background-child-victim" in reply, "the reply names the command it approved"
    assert "second-victim" not in reply
    assert "1 more" in reply, "and says one request is still pending"

    reply = discord.slash("deny", discord.source("thread", user_id="43"))

    assert second.done.wait(5) and second.decisions[0]["approved"] is False
    assert "second-victim" in reply, "the deny reply names the command it denied"
    assert "more" not in reply, "nothing is left pending"


def test_answer_preview_is_redacted_and_cut_short():
    from gateway.platforms.base_exec_approval import APPROVAL_ANSWER_PREVIEW_CHARS, approval_answer_detail
    secret = "ghp_" + "X" * 36
    long_command = f"curl -H 'Authorization: token {secret}' https://example.com/" + "a" * 400 + " TAIL"

    detail = approval_answer_detail([{"command": long_command}], 0)

    assert secret not in detail, "the preview is the redacted text the card shows"
    assert "TAIL" not in detail and "curl" in detail
    assert len(detail) < APPROVAL_ANSWER_PREVIEW_CHARS + 40
    assert approval_answer_detail([{"command": "a"}, {"command": "b"}], 0) == "", "several at once: the count says it"
    multiline = approval_answer_detail([{"command": "echo `id`\nrm x"}], 0)
    assert multiline.count("`") == 2 and "\nrm" not in multiline, "one inline-code span, on one line"


# -- who may type the answer: the slash access policy -------------------------------------------


def _policy_runner(extra: dict):
    """A runner whose Discord platform carries ``extra``, driven through ``_handle_message``."""
    from tests.gateway.test_slash_access_dispatch import _make_runner
    runner = _make_runner(platform_extra=extra)
    runner._busy_ack_ts, runner._draining = {}, False
    runner.adapters[Platform.DISCORD]._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    return runner


def _thread_source(user_id: str, *, is_bot: bool = False) -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, user_id=user_id, user_name=f"name-{user_id}", chat_id="777",
                         chat_type="thread", thread_id="777", is_bot=is_bot)


def _pending(runner, source, owner=None) -> _ApprovalEntry:
    entry = _ApprovalEntry({"command": COMMAND})
    entry.owner = owner
    with approval._lock:
        approval._gateway_queues.setdefault(runner._session_key_for_source(source), []).append(entry)
    return entry


@pytest.fixture
def clean_approvals():
    yield
    with approval._lock:
        approval._gateway_queues.clear()
        approval._pending.clear()


_GATED = {"group_allow_admin_from": [OWNER]}


@pytest.mark.parametrize("background", [False, True], ids=["foreground", "background-child"])
@pytest.mark.parametrize("verb", ["approve", "approve always", "approve session", "approve all", "deny"])
def test_with_an_admin_list_another_bots_typed_answer_is_refused(clean_approvals, verb, background):
    runner = _policy_runner(_GATED)
    bot = _thread_source(OTHER_BOT, is_bot=True)
    entry = _pending(runner, bot, owner=object() if background else None)

    reply = asyncio.run(runner._handle_message(MessageEvent(text=f"/{verb}", source=bot, message_id="m1")))

    assert "admin-only" in reply, reply
    assert entry.result is None and not entry.event.is_set(), "the request is still pending"
    assert approval.has_blocking_approval(runner._session_key_for_source(bot))


@pytest.mark.parametrize("verb,result", [("approve", "once"), ("approve always", "always"), ("deny", "deny")])
def test_with_an_admin_list_the_admins_typed_answer_still_resolves(clean_approvals, verb, result):
    runner = _policy_runner(_GATED)
    owner = _thread_source(OWNER)
    entry = _pending(runner, owner)

    asyncio.run(runner._handle_message(MessageEvent(text=f"/{verb}", source=owner, message_id="m1")))

    assert entry.event.is_set() and entry.result == result


@pytest.mark.parametrize("verb,result", [("approve", "once"), ("deny", "deny")])
def test_without_an_admin_list_another_bots_typed_answer_resolves_as_before(clean_approvals, verb, result):
    """The shipped default: no ``group_allow_admin_from`` means no slash gating at all."""
    runner = _policy_runner({})
    bot = _thread_source(OTHER_BOT, is_bot=True)
    entry = _pending(runner, bot)

    asyncio.run(runner._handle_message(MessageEvent(text=f"/{verb}", source=bot, message_id="m1")))

    assert entry.event.is_set() and entry.result == result


def test_the_admin_list_leaves_the_card_button_to_the_user_allowlist():
    """``group_allow_admin_from`` gates typed commands only; the button stays with the allowlist
    unless ``require_admin_for_exec_approval`` is also set (and then it reads ``allow_admin_from``)."""
    from plugins.platforms.discord.adapter_config_readers import _resolve_exec_approval_admin_gate
    assert _resolve_exec_approval_admin_gate(_GATED) == (False, set())
    assert _resolve_exec_approval_admin_gate(
        {"require_admin_for_exec_approval": True, "allow_admin_from": [OWNER]}) == (True, {OWNER})


@pytest.mark.parametrize("user_id,resolved", [(OWNER, True), ("43", False)])
def test_bare_word_obeys_the_slash_access_policy(clean_approvals, user_id, resolved):
    """A bare ``yes`` is a typed ``/approve``: someone the policy refuses ``/approve`` cannot say it."""
    runner = _policy_runner(_GATED)
    source = _thread_source(user_id)
    key = runner._session_key_for_source(source)
    entry = _pending(runner, source)
    runner._delivery_adapter_for = lambda _source: runner.adapters[Platform.DISCORD]
    runner._send_busy_reply = AsyncMock()
    event = MessageEvent(text="yes", message_type=MessageType.TEXT, source=source, message_id="m2")

    consumed = asyncio.run(runner._route_plaintext_approval_while_busy(event, key))

    assert consumed is resolved
    assert entry.event.is_set() is resolved
    assert event.text == ("/approve" if resolved else "yes"), "a refused word stays an ordinary message"
