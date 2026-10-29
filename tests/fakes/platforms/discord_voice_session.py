"""Fakes at the Discord edge for voice-session tests: voice client, voice channel, receiver, client.

Everything Hermes owns runs for real (``DiscordAdapter`` built by its constructor, its join/leave,
timeout, flush, auto-leave handler); only the discord.py objects it talks to are stand-ins.
``connect_capturing_handlers`` runs the real ``DiscordAdapter.connect()`` against a fake bot so the
real ``on_voice_state_update`` closure can be driven.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

# Import order matters: this installs the discord mock before the adapter module is imported.
from tests.gateway.test_discord_connect import FakeBot  # noqa: F401  (also installs the discord mock)

from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD, TEXT, VOICE_A, VOICE_B = 1, 700, 900, 901


class FakeVoiceClient:
    """discord.py ``VoiceProtocol`` surface: ``disconnect()`` ends in ``cleanup()`` as the real one does."""

    def __init__(self, client: Any, channel: Any) -> None:
        self.client, self.channel = client, channel
        self._connected = True
        self.cleanups = 0

    def is_connected(self) -> bool:
        return self._connected

    def is_playing(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def cleanup(self) -> None:
        self.cleanups += 1

    async def disconnect(self, *, force: bool = False) -> None:
        self._connected = False
        self.cleanup()

    async def move_to(self, channel: Any) -> None:
        self.channel = channel

    def closed_by_discord(self) -> None:
        """A final disconnect Discord initiated (kick, channel deleted): discord.py calls cleanup()."""
        self._connected = False
        self.cleanup()


class FakeVoiceChannel:
    def __init__(self, cid: int, guild: Any, *, name: str = "drive", members: Optional[list] = None) -> None:
        self.id, self.guild, self.name = cid, guild, name
        self.members = members if members is not None else []
        self.connect_classes: list = []

    async def connect(self, *, cls: Any = None, **_kw: Any) -> Any:
        self.connect_classes.append(cls)
        return (cls or FakeVoiceClient)(self.guild._client, self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeVoiceChannel) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


class FakeReceiver:
    """``VoiceReceiver`` stand-in: ``pending`` is what the leave's final flush returns."""

    instances: list["FakeReceiver"] = []

    def __init__(self, vc: Any, allowed_user_ids: Any = None) -> None:
        self.vc, self.pending, self.stopped, self._running = vc, [], False, True
        FakeReceiver.instances.append(self)

    @staticmethod
    def pcm_to_wav(pcm: bytes, path: str) -> None:
        """WAV conversion is ffmpeg's job; STT is stubbed by the tests that need a transcript."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped, self._running = True, False

    def flush_pending(self) -> list:
        pending, self.pending = self.pending, []
        return pending

    def check_silence(self) -> list:
        return []

    def refresh_credentials(self, _why: str) -> None:
        pass


def make_guild(client: Any, gid: int = GUILD) -> Any:
    guild = SimpleNamespace(id=gid, name="Garage", _client=client, get_member=lambda uid: None)
    return guild


def make_adapter(*, owner: Optional[str] = None, bot_user_id: int = 999) -> DiscordAdapter:
    """A real ``DiscordAdapter`` on a fake client: one guild, one text channel, voice timers off."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token=f"fake-{bot_user_id}"))
    if owner:
        adapter.set_owner_profile(owner)
    client = SimpleNamespace(user=SimpleNamespace(id=bot_user_id, name=f"bot-{bot_user_id}"))
    guild = make_guild(client)
    text = SimpleNamespace(id=TEXT, name="drive-notes", topic=None, guild=guild, send=AsyncMock())
    client.get_guild = lambda gid: guild if gid == GUILD else None
    client.get_channel = lambda cid: text if cid == TEXT else None
    client.get_user = lambda uid: None
    adapter._client = client
    adapter._voice_timeout_seconds = 0
    adapter._is_allowed_user = lambda uid, guild=None, is_dm=False: True
    adapter.handle_message = AsyncMock()
    adapter._fake_guild, adapter._fake_text = guild, text
    return adapter


def voice_channel(adapter: DiscordAdapter, cid: int = VOICE_A, **kw: Any) -> FakeVoiceChannel:
    return FakeVoiceChannel(cid, adapter._fake_guild, **kw)


async def stop_voice_tasks(adapter: DiscordAdapter) -> None:
    tasks = [*adapter._voice_timeout_tasks.values(), *adapter._voice_listen_tasks.values()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def connect_capturing_handlers(adapter: DiscordAdapter, monkeypatch: Any) -> dict:
    """Run the real ``connect()`` on a fake bot; return its registered event handlers by name.

    The fake bot keeps the adapter's fake guild/channel lookups, so the captured closures act on
    the same objects the rest of the test drives.
    """
    monkeypatch.setattr("gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)
    intents = SimpleNamespace(message_content=False, dm_messages=False, guild_messages=False,
                              members=False, voice_states=False)
    monkeypatch.setattr(adapter_mod.Intents, "default", lambda: intents)
    fake_client = adapter._client
    bots: list = []

    def _bot(**kwargs: Any) -> Any:
        bot = FakeBot(**kwargs)
        bot.user = fake_client.user
        bot.get_guild, bot.get_channel, bot.get_user = (
            fake_client.get_guild, fake_client.get_channel, fake_client.get_user)
        bots.append(bot)
        return bot

    monkeypatch.setattr(adapter_mod.commands, "Bot", _bot)
    monkeypatch.setattr(adapter, "_resolve_allowed_usernames", AsyncMock())
    monkeypatch.setattr(adapter, "_start_liveness_probe", lambda: None)
    assert await adapter.connect() is True
    return bots[-1]._events


def member(uid: int, guild: Any) -> Any:
    return SimpleNamespace(id=uid, guild=guild, display_name=f"user-{uid}")


def voice_state(channel: Any) -> Any:
    return SimpleNamespace(channel=channel)
