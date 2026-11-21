"""Speech from a bot account in a Discord voice channel follows the bot policy its typed messages do.

The voice source is rebuilt from the ``/voice join`` copy (or built fresh for a programmatic join), and
neither carried the speaker's ``Member.bot``. A bot's speech therefore ran as a person's turn: it ignored
``DISCORD_ALLOW_BOTS=none`` and was never counted by the gateway's bot loop guard.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter

_GUILD, _TEXT, _JOINER, _BOT, _HUMAN, _UNCACHED = 1, 700, 42, 43, 44, 45


def _runner(monkeypatch: pytest.MonkeyPatch, join: str, allow_bots: str | None, inline_mention: bool = True):
    for var in ("DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
                "DISCORD_ALLOWED_ROLES", "DISCORD_ALLOW_BOTS", "DISCORD_BOTS_REQUIRE_INLINE_MENTION"):
        monkeypatch.delenv(var, raising=False)
    # Every speaker is on the user allowlist, so only the bot policy can refuse one.
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{_JOINER},{_BOT},{_HUMAN},{_UNCACHED}")
    if allow_bots is not None:
        monkeypatch.setenv("DISCORD_ALLOW_BOTS", allow_bots)
    if not inline_mention:
        monkeypatch.setenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._allowed_user_ids = {str(_JOINER), str(_BOT), str(_HUMAN), str(_UNCACHED)}
    members = {
        _JOINER: SimpleNamespace(id=_JOINER, display_name="joiner", bot=False, roles=[]),
        _BOT: SimpleNamespace(id=_BOT, display_name="relay-bot", bot=True, roles=[]),
        _HUMAN: SimpleNamespace(id=_HUMAN, display_name="person", bot=False, roles=[]),
    }
    guild = SimpleNamespace(id=_GUILD, name="Hermes Server", get_member=members.get)
    adapter._client = SimpleNamespace(get_guild=lambda _id: guild, get_channel=lambda _id: None,
                                      get_user=lambda _id: None)
    adapter._voice_text_channels = {_GUILD: _TEXT}
    joined = SessionSource(platform=Platform.DISCORD, chat_id=str(_TEXT), chat_type="group",
                           user_id=str(_JOINER), user_name="joiner", guild_id=str(_GUILD))
    adapter._voice_sources = {_GUILD: joined.to_dict()} if join == "bound" else {}
    adapter.handle_message = AsyncMock()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode, runner._session_db, runner.session_store = {}, None, MagicMock()
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("join", ["bound", "programmatic"])
@pytest.mark.parametrize("speaker,allow_bots,inline_mention,dispatched,is_bot", [
    (_BOT, None, True, 0, None),          # DISCORD_ALLOW_BOTS defaults to none: a bot's speech is ignored
    (_BOT, "mentions", False, 0, None),   # an utterance cannot @mention Hermes
    (_BOT, "all", True, 0, None),         # by default even "all" needs a typed inline mention
    (_BOT, "all", False, 1, True),        # admitted, and the gateway sees a bot
    (_HUMAN, None, True, 1, False),       # a person is unaffected
    (_UNCACHED, None, True, 1, False),    # a member cache miss still reads as a person
])
async def test_bot_speech_follows_discord_allow_bots(
    monkeypatch: pytest.MonkeyPatch, join: str, speaker: int, allow_bots: str | None,
    inline_mention: bool, dispatched: int, is_bot: bool | None,
) -> None:
    runner, adapter = _runner(monkeypatch, join, allow_bots, inline_mention)

    await runner._handle_voice_channel_input(_GUILD, speaker, "what broke?", adapter=adapter)

    assert adapter.handle_message.await_count == dispatched
    for call in adapter.handle_message.await_args_list:
        source = call.args[0].source
        assert source.user_id == str(speaker)
        assert source.is_bot is is_bot


@pytest.mark.asyncio
@pytest.mark.parametrize("join", ["bound", "programmatic"])
async def test_bot_speech_is_refused_while_the_loop_guard_cools_down(
    monkeypatch: pytest.MonkeyPatch, join: str,
) -> None:
    """Discord has no budget cooldown (loop-guard-hold): a full window holds the bot's next message at ingress,
    and the voice door itself stays open for it, as it does for a person in the same chat."""
    from gateway.bot_loop_guard import BotLoopGuard, settings_from_config
    from gateway.platforms.base import MessageEvent

    runner, adapter = _runner(monkeypatch, join, "all", inline_mention=False)
    now = [10_000.0]
    block = {"max_events": 1000, "max_hops": 1000}
    guard = runner._bot_loop_guard = BotLoopGuard(
        settings=lambda: settings_from_config({"gateway": {"bot_loop_guard": block}}), clock=lambda: now[0])
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._adapter_for_source = lambda source: None  # a held message stays queued; no pump runs here
    runner._hm_estop_paused = lambda: False
    source = runner._voice_input_source(adapter, _GUILD, _BOT, _TEXT)
    conversation = runner._bot_loop_guard_conversation(source)
    # Fill the window: the bot's first ten messages are admitted.
    for n in range(10):
        assert await runner._hm_admit_event(MessageEvent(text=f"ask {n}", message_id=f"v{n}", source=source)) is not None

    await runner._handle_voice_channel_input(_GUILD, _BOT, "what broke?", adapter=adapter)
    await runner._handle_voice_channel_input(_GUILD, _HUMAN, "what broke?", adapter=adapter)

    # Nothing refused the bot at the voice door; the window holds what it says at ingress.
    dispatched = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert [event.source.user_id for event in dispatched] == [str(_BOT), str(_HUMAN)]
    assert await runner._hm_admit_event(dispatched[0]) is None
    assert guard.held_events(conversation) == [dispatched[0]]
    assert await runner._hm_admit_event(dispatched[1]) is not None  # a person in the same chat is not held
