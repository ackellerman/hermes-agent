"""A bot message that freshly @mentions this bot is a new ask, never ``reply_to_reply``.

Discord message 1571794454132686869: Boba answered a TARS message; 14 s later a different TARS
session posted a new ask in the same thread, opening with ``<@boba>``. The chain guard keys
``replied_to`` on the sender bot id, so it read the new ask as TARS answering Boba's answer and
dropped it, and the drop log carried no message id. The fresh-mention exemption keeps the hop cap,
the duplicate rule and the window budget; a Discord reply to this bot's own message keeps the old
``reply_to_reply`` verdict.
"""

import logging
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings
from gateway.session import Platform, SessionSource

CHAT = "1571780464853647400"
PEER = "1476257859233644656"  # TARS
HUMAN = "alice"


@pytest.fixture
def clock():
    state = {"t": 1_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda s: state.__setitem__("t", state["t"] + s))


def _runner(clock, **overrides):
    from gateway.run import GatewayRunner

    settings = BotLoopGuardSettings(**{"max_events": 1000, "window_seconds": 60, "cooldown_seconds": 600, **overrides})
    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._bot_loop_guard = BotLoopGuard(settings=lambda: settings, clock=clock.now)
    runner._adapter_profile_for_source = lambda source: ""
    return runner


def _event(user, text, *, is_bot=True, mention=False, message_id=None):
    from gateway.platforms.base import MessageEvent

    source = SessionSource(platform=Platform.DISCORD, chat_id=CHAT, chat_type="thread", user_id=user, is_bot=is_bot)
    event = MessageEvent(text=text, message_id=message_id or text, source=source)
    # Set as an attribute (not a constructor kwarg) so this file also runs against the unfixed tree.
    event.fresh_self_mention = mention
    return event


async def _admitted(runner, event) -> bool:
    return await runner._hm_admit_event(event) is not None


@pytest.mark.asyncio
async def test_a_fresh_mention_from_the_bot_just_answered_is_a_new_ask(clock):
    """(a) The incident: TARS asks, we answer, another TARS session @mentions us with a new ask."""
    runner = _runner(clock)
    first = _event(PEER, "roots not set yet", mention=True)
    assert await _admitted(runner, first)
    runner._note_bot_reply(first.source)  # our answer went out
    clock.advance(14)
    assert await _admitted(runner, _event(PEER, "Added scope for X", mention=True, message_id="1571794454132686869"))


@pytest.mark.asyncio
async def test_the_same_sender_without_a_fresh_mention_is_still_reply_to_reply(clock):
    """(b) Only the fresh mention is exempt: an unmentioned answer to our answer still ends it."""
    runner = _runner(clock)
    first = _event(PEER, "question", mention=True)
    assert await _admitted(runner, first)
    runner._note_bot_reply(first.source)
    assert await _admitted(runner, _event(PEER, "new ask", mention=True))
    runner._note_bot_reply(first.source)
    assert not await _admitted(runner, _event(PEER, "thanks!", mention=False))
    assert runner._bot_loop_guard.chain_admit(
        ("", "discord", CHAT), PEER, "thanks again", is_bot=True) == (False, "reply_to_reply")


@pytest.mark.asyncio
async def test_fresh_mentions_are_still_capped_by_max_hops_and_duplicates(clock):
    """(c) A bot that @mentions us in every reply stays bounded by max_hops (and the duplicate rule)."""
    runner = _runner(clock, max_hops=3)
    verdicts = []
    for n in range(6):
        event = _event(PEER, f"ping {n}", mention=True)
        verdicts.append(await _admitted(runner, event))
        runner._note_bot_reply(event.source)
    assert verdicts == [True, True, True, False, False, False]
    assert runner._bot_loop_guard.chain_admit(
        ("", "discord", CHAT), PEER, "ping 9", is_bot=True, fresh_mention=True) == (False, "max_hops")
    # A human message still starts a new chain.
    assert await _admitted(runner, _event(HUMAN, "carry on", is_bot=False))
    assert await _admitted(runner, _event(PEER, "ping 10", mention=True))
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(), clock=clock.now)
    assert guard.chain_admit(CHAT, PEER, "same ask", is_bot=True, fresh_mention=True) == (True, "ok")
    assert guard.chain_admit(CHAT, PEER, "same  ask", is_bot=True, fresh_mention=True) == (False, "duplicate")


@pytest.mark.asyncio
async def test_the_drop_log_names_the_message_id(clock, caplog):
    """(d) Grepping the logs for a Discord message id finds the guard's drop."""
    runner = _runner(clock)
    first = _event(PEER, "question", mention=True)
    assert await _admitted(runner, first)
    runner._note_bot_reply(first.source)
    with caplog.at_level(logging.INFO, logger="gateway.authz_mixin"):
        assert not await _admitted(runner, _event(PEER, "thanks!", message_id="1571794454132686869"))
    drops = [r.getMessage() for r in caplog.records if "Bot loop guard ended a bot exchange" in r.getMessage()]
    assert drops and "1571794454132686869" in drops[0] and "reply_to_reply" in drops[0]


def test_merged_bursts_keep_the_first_messages_verdict():
    """A burst takes the first message's flag: a later fresh @mention merged in (another author in a
    shared thread, or a later chunk) must not turn an answer to our answer into a new ask."""
    from gateway.platforms.base import MessageEvent

    head = MessageEvent(text="<mention stripped> part 1")
    head.fresh_self_mention = True
    tail = MessageEvent(text="part 2")
    head.absorb_reply_expected(tail)
    assert head.fresh_self_mention is True
    answer = MessageEvent(text="thanks!")
    answer.absorb_reply_expected(head)
    assert answer.fresh_self_mention is False
    assert MessageEvent(text="x").fresh_self_mention is False


def _discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    return adapter


def _message(content, *, reference=None, mentions=()):
    return SimpleNamespace(content=content, reference=reference, mentions=list(mentions))


def test_discord_fresh_mention_is_a_typed_mention_not_a_reply_to_this_bot():
    adapter = _discord_adapter()
    own = SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=99)))
    other = SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=7)))
    unresolved = SimpleNamespace(resolved=None)
    assert adapter._is_fresh_self_mention(_message("<@99> Added scope for X"))
    assert adapter._is_fresh_self_mention(_message("<@!99> legacy form"))
    assert adapter._is_fresh_self_mention(_message("<@99> see that", reference=other))
    # A reply to this bot's own message keeps the reply_to_reply verdict, typed mention or not.
    assert not adapter._is_fresh_self_mention(_message("<@99> thanks", reference=own))
    assert not adapter._is_fresh_self_mention(_message("<@99> thanks", reference=unresolved))
    # A streamed answer's overflow chunk replies to the author's own previous chunk: not a new ask.
    chunk = _message("<@99> ...continued", reference=SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=7))))
    chunk.author = SimpleNamespace(id=7)
    assert not adapter._is_fresh_self_mention(chunk)
    # A reply-ping puts us in ``mentions`` without a typed token: not fresh.
    assert not adapter._is_fresh_self_mention(_message("thanks", reference=own, mentions=[adapter._client.user]))
    # Another bot's mention is not ours (multiplexed profiles each judge their own bot id).
    assert not adapter._is_fresh_self_mention(_message("<@100> for the other seat"))
