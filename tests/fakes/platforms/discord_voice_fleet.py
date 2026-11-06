"""A small Discord voice fleet for policy tests: several real ``DiscordAdapter`` seats in one guild.

Everything Hermes owns runs for real (adapters, their join/leave, the runner's ``/voice join``
handler); only discord.py objects are fakes (``discord_voice_session``). Each seat gets its own fake
guild object, as each bot has its own discord.py cache, but a voice channel id names ONE shared
member list, so a bot that connects shows up in every seat's view of that channel.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import yaml

from tests.fakes.platforms import discord_voice_session as fx  # first: installs the discord mock

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType

GUILD, USER = fx.GUILD, 4242
AUTO_VC, MANUAL_VC, OTHER_VC = 1001, 1002, 1003
HOME_TEXT, ALPHA_TEXT, BETA_TEXT, THREAD = 700, 710, 720, 730
BOT_IDS = {"default": 9000, "alpha": 9001, "beta": 9002}


def write_config(path: Path, cfg: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg if isinstance(cfg, str) else yaml.safe_dump(cfg), encoding="utf-8")


def write_fleet(root: Path, voice_channels: Optional[dict] = None, *, alpha_legacy=(), seats: Optional[dict] = None) -> None:
    """Root config (the fleet map) + alpha/beta profile configs on disk."""
    discord: dict = {}
    if voice_channels is not None:
        discord["voice_channels"] = voice_channels
    write_config(root / "config.yaml", {"discord": discord})
    alpha: dict = {"discord": {}}
    if alpha_legacy:
        alpha["discord"]["voice_auto_join"] = {"enabled": True, "channel_ids": [str(v) for v in alpha_legacy],
                                               "user_ids": [str(USER)]}
    for name, cfg in (("alpha", alpha), ("beta", {"discord": {}})):
        if seats and name in seats:
            cfg = seats[name]
        write_config(root / "profiles" / name / "config.yaml", cfg)


class Fleet:
    def __init__(self) -> None:
        self.members: dict[int, list] = {}
        self.names = {AUTO_VC: "General", MANUAL_VC: "Voice (manual)", OTHER_VC: "Side room"}
        self.seats: dict[str, Any] = {}
        self.user_vc: Optional[int] = None
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner.adapters, runner._profile_adapters, runner._voice_mode = {}, {}, {}
        runner._session_db, runner.session_store = None, MagicMock()
        runner._is_user_authorized = MagicMock(return_value=True)
        self.runner = runner

    def set_voice_mode_path(self, path: Path) -> None:
        self.runner._VOICE_MODE_PATH = path

    def seat(self, name: str) -> Any:
        if name in self.seats:
            return self.seats[name]
        adapter = fx.make_adapter(owner=None if name == "default" else name, bot_user_id=BOT_IDS[name])
        guild = adapter._fake_guild
        texts = {cid: SimpleNamespace(id=cid, name=f"chat-{cid}", topic=None, guild=guild, send=MagicMock())
                 for cid in (HOME_TEXT, ALPHA_TEXT, BETA_TEXT, THREAD)}
        adapter._client.get_channel = texts.get
        channels: dict[int, Any] = {}
        fleet = self

        def vc(cid: int) -> Any:
            if cid not in channels:
                ch = fx.FakeVoiceChannel(cid, guild, name=fleet.names[cid], members=fleet.members.setdefault(cid, []))
                original = ch.connect

                async def connect(*, cls: Any = None, **kw: Any) -> Any:
                    client = await original(cls=cls, **kw)
                    ch.members.append(SimpleNamespace(id=BOT_IDS[name], bot=True, display_name=f"{name}-bot",
                                                      guild=guild))
                    disconnect = client.disconnect

                    async def leave(*, force: bool = False) -> None:
                        ch.members[:] = [m for m in ch.members if m.id != BOT_IDS[name]]
                        await disconnect(force=force)

                    client.disconnect = leave
                    return client

                ch.connect = connect
                channels[cid] = ch
            return channels[cid]

        def get_member(uid: int) -> Any:
            if uid == USER:
                voice = SimpleNamespace(channel=vc(fleet.user_vc)) if fleet.user_vc else None
                return SimpleNamespace(id=USER, bot=False, display_name="operator", voice=voice, guild=guild)
            for members in fleet.members.values():
                for m in members:
                    if m.id == uid:
                        return m
            return None

        guild.get_member = get_member
        adapter._fake_vc = vc
        adapter.gateway_runner = self.runner
        if name == "default":
            self.runner.adapters[Platform.DISCORD] = adapter
        else:
            self.runner._profile_adapters[name] = {Platform.DISCORD: adapter}
        self.seats[name] = adapter
        return adapter

    def operator_in(self, cid: Optional[int]) -> None:
        self.user_vc = cid
        if cid is not None:
            for members in self.members.values():
                members[:] = [m for m in members if m.id != USER]
            self.members.setdefault(cid, []).append(SimpleNamespace(id=USER, bot=False, display_name="operator"))

    def command(self, seat: str, chat_id: int, *, thread: bool = False) -> MessageEvent:
        adapter = self.seat(seat)
        source = adapter.build_source(
            chat_id=str(chat_id), chat_type="thread" if thread else "group", user_id=str(USER),
            user_name="operator", thread_id=str(chat_id) if thread else None, guild_id=str(GUILD),
            parent_chat_id=str(BETA_TEXT if seat == "beta" else HOME_TEXT) if thread else None)
        return MessageEvent(text="/voice join", message_type=MessageType.TEXT, source=source,
                            raw_message=SimpleNamespace(guild_id=GUILD, guild=None))

    async def voice_join(self, seat: str, chat_id: int, **kw: Any) -> str:
        return await self.runner._handle_voice_channel_join(self.command(seat, chat_id, **kw))

    def bots_in(self, cid: int) -> list[int]:
        return sorted(m.id for m in self.members.get(cid, []) if getattr(m, "bot", False))

    async def close(self) -> None:
        for adapter in self.seats.values():
            await fx.stop_voice_tasks(adapter)
