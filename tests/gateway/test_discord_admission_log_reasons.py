"""Every admission decision is logged with its reason.

Evidence this exists: a bot message dropped by ``allow_bots: none`` produced no log line at all, so
"the bot ignored the other agent" and "the gateway never received the event" looked identical from
the outside. Admission now emits one record per message carrying message id, author id + bot flag,
channel id and the reason.

The INFO record is one per message — never per chunk or attachment — and admits stay at DEBUG so a
busy server does not double its log volume just by being busy.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import discord

import plugins.platforms.discord.adapter as adapter_mod
from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
CHANNEL_ID = 4242
BOT_AUTHOR_ID = 777
OTHER_BOT_ID = 888
HUMAN_AUTHOR_ID = 555

LOGGER = "plugins.platforms.discord.adapter"


def _adapter(*, extra=None, allowed: bool = True) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra=extra or {})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=allowed)
    adapter._get_parent_channel_id = Mock(return_value=None)
    adapter._discord_free_response_channels = Mock(return_value=set())
    adapter._discord_channel_keys = Mock(return_value={str(CHANNEL_ID)})
    adapter._threads = set()
    # Hermetic gates: blank snapshot, no profile home — the host's real .env must not decide this.
    adapter._gate_env_snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    adapter._live_policy_home = None
    return adapter


def _message(*, author_id=HUMAN_AUTHOR_ID, bot=False, message_type=None, content="hello",
             mentions=(), msg_id=1234, author=None):
    return SimpleNamespace(
        id=msg_id,
        author=author or SimpleNamespace(id=author_id, bot=bot, display_name="whoever"),
        channel=SimpleNamespace(id=CHANNEL_ID, name="general", parent_id=None),
        content=content,
        mentions=list(mentions),
        type=discord.MessageType.default if message_type is None else message_type,
        guild=SimpleNamespace(id=1),
        attachments=[],
    )


def _drop_records(caplog):
    return [r for r in caplog.records if "admission: dropping message" in r.getMessage()]


def test_dropped_bot_message_logs_bot_not_allowed(caplog):
    # The exact case that was silent: allow_bots defaults to none, so a sibling agent's message
    # vanished without a trace.
    caplog.set_level(logging.INFO, logger=LOGGER)
    admitted, _ = _adapter()._discord_message_admission(
        _message(author_id=BOT_AUTHOR_ID, bot=True), claim=True)

    assert admitted is False
    records = _drop_records(caplog)
    assert len(records) == 1
    line = records[0].getMessage()
    assert records[0].levelno == logging.INFO
    assert "message 1234" in line
    assert str(BOT_AUTHOR_ID) in line
    assert "bot=True" in line
    assert str(CHANNEL_ID) in line
    assert "reason=bot-not-allowed" in line


def test_one_line_per_message_however_many_attachments(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    message = _message(author_id=BOT_AUTHOR_ID, bot=True)
    message.attachments = [SimpleNamespace(id=n) for n in range(3)]
    _adapter()._discord_message_admission(message, claim=True)
    assert len(_drop_records(caplog)) == 1


def test_each_silent_drop_path_names_its_reason(caplog):
    # The vocabulary the ruling asked for: every early return says why, in one grep-able field.
    caplog.set_level(logging.INFO, logger=LOGGER)

    adapter = _adapter()
    # self-message: the bot's own post (object identity, as discord.py delivers it).
    adapter._discord_message_admission(
        _message(author=adapter._client.user, bot=True), claim=False)
    adapter._discord_message_admission(
        _message(message_type=discord.MessageType.pins_add), claim=False)
    adapter._discord_message_admission(
        _message(content="<@888> ping", mentions=[SimpleNamespace(id=OTHER_BOT_ID, bot=True)]),
        claim=False)
    # Claim then re-run: the second pass is the dedup path both ingress paths can hit.
    adapter._discord_message_admission(_message(msg_id=5555), claim=True)
    adapter._discord_message_admission(_message(msg_id=5555), claim=False)
    _adapter(allowed=False)._discord_message_admission(_message(msg_id=6666), claim=False)

    reasons = [r.getMessage().rsplit("reason=", 1)[-1].rstrip(")") for r in _drop_records(caplog)]
    # ``type:<n>`` — discord.MessageType is stubbed to a mock in this test env, so only the shape of
    # the numeric reason is asserted here (real discord.py reports type:6 for pins_add).
    assert reasons[0] == "self-message"
    assert reasons[1].startswith("type:")
    assert reasons[2:] == ["other-bot-addressed", "dedup", "user-not-allowed"]


def test_admitted_messages_stay_out_of_the_info_log(caplog):
    # The volume guard: a normal (admitted) message must not add an INFO record.
    caplog.set_level(logging.INFO, logger=LOGGER)
    admitted, _ = _adapter()._discord_message_admission(_message(), claim=True)
    assert admitted is True
    assert _drop_records(caplog) == []
