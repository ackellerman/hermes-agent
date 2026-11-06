"""A same-seat rebind (``/voice join`` typed in another chat or thread while this seat is in the VC)
is gated by ``allow_thread_move`` and moves the spoken-reply mode with the binding.

Today's Q5 route: the slash event in a thread carries the thread as its chat, the runner binds it
and stores that source, so spoken turns land in the thread's session. The gate keeps that for VCs
that allow it (every unlisted VC, F11) and refuses it elsewhere. The move goes through
``_bind_voice_text_channel``, which drops the old chat's stored source when no new one is given: a
raw dict write kept it, so turns still went to the OLD chat.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio

from tests.fakes.platforms import discord_voice_fleet as vf  # first: installs the discord mock
from tests.fakes.platforms import discord_voice_session as fx
from gateway.session import build_session_key
from gateway.voice_channel_policy import REPLY_NO_MOVE
from plugins.platforms.discord import adapter as adapter_mod

_JOINED = "Joined voice channel **Voice (manual)**"


@pytest.fixture(autouse=True)
def _fake_receiver():
    with patch.object(adapter_mod, "VoiceReceiver", fx.FakeReceiver):
        yield


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "hroot"
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest_asyncio.fixture
async def fleet(root, tmp_path):
    fleet = vf.Fleet()
    fleet.set_voice_mode_path(tmp_path / "voice_mode.json")
    fleet.operator_in(vf.MANUAL_VC)
    yield fleet
    await fleet.close()


def _key(chat: int) -> str:
    return f"beta:discord:{chat}"


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_channels,seat_cfg", [
    ({str(vf.MANUAL_VC): {"allow_thread_move": True}}, None),
    ({}, None),                                              # unlisted VC: today's behaviour (F11)
])
async def test_a_rebind_into_a_thread_moves_speech_and_spoken_replies(root, fleet, voice_channels, seat_cfg):
    vf.write_fleet(root, voice_channels)
    beta = fleet.seat("beta")
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith(_JOINED)
    thread_cmd = fleet.command("beta", vf.THREAD, thread=True)
    assert (await fleet.runner._handle_voice_channel_join(thread_cmd)).startswith(_JOINED)
    assert beta._voice_text_channels[vf.GUILD] == vf.THREAD
    spoken = fleet.runner._voice_input_source(beta, vf.GUILD, vf.USER, vf.THREAD)
    assert build_session_key(spoken, group_sessions_per_user=True) == build_session_key(
        thread_cmd.source, group_sessions_per_user=True)
    assert fleet.runner._voice_mode == {_key(vf.THREAD): "all"}  # the old chat no longer speaks
    assert str(vf.BETA_TEXT) not in beta._auto_tts_enabled_chats


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_channels,seat_cfg", [
    ({str(vf.MANUAL_VC): {}}, None),                                            # listed: default off
    ({str(vf.MANUAL_VC): {"allow_thread_move": False}}, None),
    ({str(vf.MANUAL_VC): {"allow_thread_move": True}},                          # the seat narrows
     {"discord": {"voice_channels": {str(vf.MANUAL_VC): {"allow_thread_move": False}}}}),
])
async def test_a_rebind_is_refused_where_the_policy_does_not_allow_it(root, fleet, voice_channels, seat_cfg):
    vf.write_fleet(root, voice_channels, seats={"beta": seat_cfg} if seat_cfg else None)
    beta = fleet.seat("beta")
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith(_JOINED)
    before = (dict(beta._voice_sources), dict(fleet.runner._voice_mode))
    assert await fleet.voice_join("beta", vf.THREAD, thread=True) == REPLY_NO_MOVE.format(vc="Voice (manual)")
    assert beta._voice_text_channels[vf.GUILD] == vf.BETA_TEXT
    assert (dict(beta._voice_sources), dict(fleet.runner._voice_mode)) == before


@pytest.mark.asyncio
async def test_a_rebind_without_a_new_source_drops_the_old_chats_source(root, fleet, monkeypatch):
    """``bind: home`` binds a chat the command was not typed in: no source to store for it."""
    vf.write_fleet(root, {str(vf.MANUAL_VC): {"allow_thread_move": True}})
    beta = fleet.seat("beta")
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith(_JOINED)
    vf.write_fleet(root, {str(vf.MANUAL_VC): {"allow_thread_move": True, "bind": "home"}})
    monkeypatch.setattr(beta, "_auto_join_home_text_channel", lambda guild: vf.HOME_TEXT)
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith(_JOINED)
    spoken = fleet.runner._voice_input_source(beta, vf.GUILD, vf.USER, vf.HOME_TEXT)
    assert spoken.chat_id == str(vf.HOME_TEXT)
    assert fleet.runner._voice_mode == {_key(vf.HOME_TEXT): "all"}


@pytest.mark.asyncio
async def test_a_rebind_hands_back_an_auto_join_speak_enable(root, fleet):
    """The old chat's spoken replies were enabled BY the auto-join; moving must not leave them on."""
    vf.write_fleet(root, {str(vf.MANUAL_VC): {"allow_thread_move": True}})
    beta = fleet.seat("beta")
    assert await beta.join_voice_channel(beta._fake_vc(vf.MANUAL_VC), text_channel_id=vf.BETA_TEXT)
    beta._auto_tts_enabled_chats.add(str(vf.BETA_TEXT))
    beta._auto_join_tts_chats.add(str(vf.BETA_TEXT))
    assert (await fleet.voice_join("beta", vf.THREAD, thread=True)).startswith(_JOINED)
    assert str(vf.BETA_TEXT) not in beta._auto_tts_enabled_chats | beta._auto_join_tts_chats
    assert str(vf.THREAD) in beta._auto_tts_enabled_chats
