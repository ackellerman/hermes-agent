"""``/voice join`` follows the fleet voice-channel map: R1, ``never``, ``seats``, ``unlisted``, ``bind``.

R1 (operator requirement): ``/voice join`` must not attach to an auto-join voice channel. The caller
in an auto VC is answered "join a manual voice channel first": no connect, no move, and the
auto-join session keeps its home binding. The auto set comes from CONFIG (the root map's auto
entries plus every served profile's on-disk ``voice_auto_join.channel_ids``), so a seat that is not
connected still counts. Real ``DiscordAdapter`` seats and the real runner handler; discord.py faked.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio

from tests.fakes.platforms import discord_voice_fleet as vf  # first: installs the discord mock
from tests.fakes.platforms import discord_voice_session as fx
from gateway.voice_channel_policy import (REPLY_ALREADY_BOUND, REPLY_INVALID, REPLY_NEVER, REPLY_R1,
                                          REPLY_SEAT, REPLY_UNLISTED, REPLY_NO_HOME)
from plugins.platforms.discord import adapter as adapter_mod

_MAP = {
    str(vf.AUTO_VC): {"join": "auto", "seats": ["default"]},
    str(vf.MANUAL_VC): {"join": "manual", "allow_thread_move": True},
}


@pytest.fixture(autouse=True)
def _fake_receiver():
    with patch.object(adapter_mod, "VoiceReceiver", fx.FakeReceiver):
        yield


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "hroot"
    vf.write_fleet(root, _MAP)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest_asyncio.fixture
async def fleet(root, tmp_path):
    fleet = vf.Fleet()
    fleet.set_voice_mode_path(tmp_path / "voice_mode.json")
    yield fleet
    await fleet.close()


async def _auto_joined_default(fleet):
    """The default seat auto-joined General and bound its home channel (as the auto-join does)."""
    default = fleet.seat("default")
    fleet.operator_in(vf.AUTO_VC)
    assert await default.join_voice_channel(default._fake_vc(vf.AUTO_VC), text_channel_id=vf.HOME_TEXT)
    return default


@pytest.mark.asyncio
async def test_r1_another_seats_chat_cannot_attach_to_an_auto_vc(fleet):
    await _auto_joined_default(fleet)
    beta = fleet.seat("beta")
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == REPLY_R1
    assert vf.GUILD not in beta._voice_clients and fleet.bots_in(vf.AUTO_VC) == [vf.BOT_IDS["default"]]
    assert fleet.seat("default")._voice_text_channels[vf.GUILD] == vf.HOME_TEXT


@pytest.mark.asyncio
async def test_r1_the_auto_seats_own_other_chat_does_not_move_the_binding(fleet):
    """Today's Q4 path: the binding moved from home to the typing chat. R1 refuses instead."""
    default = await _auto_joined_default(fleet)
    vc_client = default._voice_clients[vf.GUILD]
    assert await fleet.voice_join("default", vf.ALPHA_TEXT) == REPLY_R1
    assert default._voice_text_channels[vf.GUILD] == vf.HOME_TEXT
    assert default._voice_clients[vf.GUILD] is vc_client and vc_client.is_connected()
    assert fleet.runner._voice_mode == {}


@pytest.mark.asyncio
async def test_r1_typed_in_the_already_bound_chat_says_so_and_changes_nothing(fleet):
    default = await _auto_joined_default(fleet)
    assert await fleet.voice_join("default", vf.HOME_TEXT) == REPLY_ALREADY_BOUND.format(vc="General")
    assert default._voice_text_channels[vf.GUILD] == vf.HOME_TEXT and fleet.runner._voice_mode == {}


@pytest.mark.asyncio
async def test_r1_opt_in_keeps_the_previous_behaviour(root, fleet):
    vf.write_fleet(root, {**_MAP, str(vf.AUTO_VC): {"join": "auto", "seats": ["default"],
                                                    "allow_attach_from_text": True, "allow_thread_move": True}})
    default = await _auto_joined_default(fleet)
    assert (await fleet.voice_join("default", vf.ALPHA_TEXT)).startswith("Joined voice channel **General**")
    assert default._voice_text_channels[vf.GUILD] == vf.ALPHA_TEXT


@pytest.mark.asyncio
async def test_r1_counts_a_seats_legacy_auto_join_while_that_seat_is_not_connected(root, fleet):
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    fleet.operator_in(vf.OTHER_VC)
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == REPLY_R1
    assert fleet.bots_in(vf.OTHER_VC) == []


@pytest.mark.asyncio
async def test_a_manual_vc_joins_and_binds_the_typing_chat(fleet):
    fleet.operator_in(vf.MANUAL_VC)
    reply = await fleet.voice_join("beta", vf.BETA_TEXT)
    beta = fleet.seat("beta")
    assert reply.startswith("Joined voice channel **Voice (manual)**") and fleet.bots_in(vf.MANUAL_VC) == [vf.BOT_IDS["beta"]]
    assert beta._voice_text_channels[vf.GUILD] == vf.BETA_TEXT
    assert fleet.runner._voice_mode == {f"beta:discord:{vf.BETA_TEXT}": "all"}


@pytest.mark.asyncio
@pytest.mark.parametrize("entry,seat_cfg,reply", [
    ({"join": "never"}, None, REPLY_NEVER.format(vc="Voice (manual)")),
    ({"seats": ["alpha"]}, None, REPLY_SEAT.format(vc="Voice (manual)")),
    ({}, {"discord": {"voice_channels": {str(vf.MANUAL_VC): {"seats": ["alpha"]}}}},
     REPLY_SEAT.format(vc="Voice (manual)")),                                        # the seat narrows
    ({"join": "sometimes"}, None, REPLY_INVALID),                                    # fails closed
])
async def test_never_seats_and_bad_entries_refuse_without_connecting(root, fleet, entry, seat_cfg, reply):
    vf.write_fleet(root, {**_MAP, str(vf.MANUAL_VC): entry}, seats={"beta": seat_cfg} if seat_cfg else None)
    fleet.operator_in(vf.MANUAL_VC)
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == reply
    assert fleet.bots_in(vf.MANUAL_VC) == [] and vf.GUILD not in fleet.seat("beta")._voice_clients


@pytest.mark.asyncio
@pytest.mark.parametrize("unlisted,joined", [(None, True), ("allow", True), ("refuse", False)])
async def test_unlisted_vcs_follow_the_switch(root, fleet, unlisted, joined):
    vf.write_fleet(root, {**_MAP, **({"unlisted": unlisted} if unlisted else {})})
    fleet.operator_in(vf.OTHER_VC)
    reply = await fleet.voice_join("beta", vf.BETA_TEXT)
    assert (fleet.bots_in(vf.OTHER_VC) == [vf.BOT_IDS["beta"]]) is joined
    if not joined:
        assert reply == REPLY_UNLISTED.format(vc="Side room")


@pytest.mark.asyncio
async def test_an_unreadable_served_profile_config_refuses(root, fleet):
    vf.write_config(root / "profiles" / "alpha" / "config.yaml", "discord: [unclosed\n")
    fleet.operator_in(vf.MANUAL_VC)
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == REPLY_INVALID
    assert fleet.bots_in(vf.MANUAL_VC) == []


@pytest.mark.asyncio
async def test_bind_home_binds_the_seats_home_channel(root, fleet, monkeypatch):
    vf.write_fleet(root, {**_MAP, str(vf.MANUAL_VC): {"bind": "home"}})
    fleet.operator_in(vf.MANUAL_VC)
    beta = fleet.seat("beta")
    monkeypatch.setattr(beta, "_auto_join_home_text_channel", lambda guild: None)
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == REPLY_NO_HOME.format(vc="Voice (manual)")
    monkeypatch.setattr(beta, "_auto_join_home_text_channel", lambda guild: vf.HOME_TEXT)
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith("Joined voice channel **Voice (manual)**")
    assert beta._voice_text_channels[vf.GUILD] == vf.HOME_TEXT and vf.GUILD not in beta._voice_sources
    assert fleet.runner._voice_mode == {f"beta:discord:{vf.HOME_TEXT}": "all"}


# --- the receiver a join wires ----------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("allow_bots,inline,drop", [
    (None, None, True),          # default bot policy: other bots' audio is dropped
    ("all", None, True),         # "all" still needs a typed mention, which speech cannot carry
    ("all", "false", False),     # the seat admits bot speech (#143368 predicate): keep it
])
async def test_a_voice_join_wires_the_receivers_bot_exclusion(fleet, monkeypatch, allow_bots, inline, drop):
    """The real ``/voice join`` path builds the receiver with ``drop_bot_audio`` from the seat's bot
    policy and ``fleet_bot_ids`` as the seat's LIVE view of every in-process seat's bot id (a
    callable, so a seat that connects later is still excluded)."""
    for var, value in (("DISCORD_ALLOW_BOTS", allow_bots), ("DISCORD_BOTS_REQUIRE_INLINE_MENTION", inline)):
        monkeypatch.delenv(var, raising=False)
        if value is not None:
            monkeypatch.setenv(var, value)
    fx.FakeReceiver.instances.clear()
    beta = fleet.seat("beta")
    fleet.seat("alpha")
    fleet.operator_in(vf.MANUAL_VC)
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith("Joined voice channel **Voice (manual)**")
    receiver = beta._voice_receivers[vf.GUILD]
    assert receiver is fx.FakeReceiver.instances[-1]
    assert receiver.bot_exclusion["drop_bot_audio"] is drop
    fleet_ids = receiver.bot_exclusion["fleet_bot_ids"]
    assert callable(fleet_ids) and fleet_ids() == {vf.BOT_IDS["alpha"], vf.BOT_IDS["beta"]}
    fleet.seat("default")  # connects after the join: the same callable now names it too
    assert fleet_ids() == set(vf.BOT_IDS.values())


# --- the auto-join path --------------------------------------------------------------------------

async def _drive_auto_join(seat, fleet, monkeypatch, vc_id, legacy):
    handlers = await fx.connect_capturing_handlers(seat, monkeypatch)
    seat._auto_join_cfg = {"enabled": True, "channel_ids": [str(v) for v in legacy], "user_ids": [str(vf.USER)],
                           "join_mode": "automatic", "require_text_opt_in": False}
    monkeypatch.setattr(seat, "_select_auto_join_text_channel", lambda guild, channel: vf.HOME_TEXT)
    fleet.operator_in(vf.MANUAL_VC if vc_id != vf.MANUAL_VC else vf.OTHER_VC)
    operator = seat._fake_guild.get_member(vf.USER)
    await handlers["on_voice_state_update"](operator, fx.voice_state(None), fx.voice_state(seat._fake_vc(vc_id)))
    return vf.GUILD in seat._voice_clients


@pytest.mark.asyncio
async def test_never_stops_a_legacy_auto_join(root, fleet, monkeypatch):
    vf.write_fleet(root, {**_MAP, str(vf.OTHER_VC): {"join": "never"}}, alpha_legacy=[vf.OTHER_VC])
    seat = fleet.seat("alpha")
    try:
        assert await _drive_auto_join(seat, fleet, monkeypatch, vf.OTHER_VC, legacy=[vf.OTHER_VC]) is False
    finally:
        await seat.disconnect()


@pytest.mark.asyncio
async def test_legacy_auto_join_still_joins_with_no_map_entry(root, fleet, monkeypatch):
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    seat = fleet.seat("alpha")
    try:
        assert await _drive_auto_join(seat, fleet, monkeypatch, vf.OTHER_VC, legacy=[vf.OTHER_VC]) is True
    finally:
        await seat.disconnect()


@pytest.mark.asyncio
async def test_a_root_auto_entry_drives_the_named_seats_auto_join(root, fleet, monkeypatch):
    seat = fleet.seat("default")
    try:
        assert await _drive_auto_join(seat, fleet, monkeypatch, vf.AUTO_VC, legacy=[]) is True
    finally:
        await seat.disconnect()
    other = fleet.seat("alpha")  # not in the entry's seats
    try:
        assert await _drive_auto_join(other, fleet, monkeypatch, vf.AUTO_VC, legacy=[]) is False
    finally:
        await other.disconnect()


@pytest.mark.asyncio
async def test_a_root_manual_entry_stops_a_seats_legacy_auto_join(root, fleet, monkeypatch):
    """V-Q2 on the adapter side: the seat's in-memory ``voice_auto_join.channel_ids`` must not
    auto-join a VC the root lists as manual (the policy object alone is not the mechanism)."""
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.MANUAL_VC])
    seat = fleet.seat("alpha")
    try:
        assert await _drive_auto_join(seat, fleet, monkeypatch, vf.MANUAL_VC, legacy=[vf.MANUAL_VC]) is False
        assert str(vf.MANUAL_VC) not in seat._effective_auto_join_channel_ids()
    finally:
        await seat.disconnect()


@pytest.mark.asyncio
async def test_auto_join_rereads_the_policy_changed_after_connect(root, fleet, monkeypatch):
    """The policy resolved at connect is stale by the time a voice event arrives: the join re-reads
    it from disk. Nothing here re-resolves by hand; only the on-disk root map changes."""
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    seat = fleet.seat("alpha")
    handlers = await fx.connect_capturing_handlers(seat, monkeypatch)
    seat._auto_join_cfg = {"enabled": True, "channel_ids": [str(vf.OTHER_VC)], "user_ids": [str(vf.USER)],
                           "join_mode": "automatic", "require_text_opt_in": False}
    monkeypatch.setattr(seat, "_select_auto_join_text_channel", lambda guild, channel: vf.HOME_TEXT)
    fleet.operator_in(vf.OTHER_VC)
    operator = seat._fake_guild.get_member(vf.USER)
    event = (operator, fx.voice_state(None), fx.voice_state(seat._fake_vc(vf.OTHER_VC)))
    try:
        assert str(vf.OTHER_VC) in seat._effective_auto_join_channel_ids()  # as resolved at connect
        vf.write_fleet(root, {**_MAP, str(vf.OTHER_VC): {"join": "never"}}, alpha_legacy=[vf.OTHER_VC])
        await handlers["on_voice_state_update"](*event)
        assert vf.GUILD not in seat._voice_clients
        vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])  # control: allowed again on disk
        seat._auto_join_debounce.clear()
        await handlers["on_voice_state_update"](*event)
        assert vf.GUILD in seat._voice_clients
    finally:
        await seat.disconnect()


@pytest.mark.asyncio
async def test_an_approved_prompt_rechecks_the_policy_before_joining(root, fleet, monkeypatch):
    """``user_prompt`` mode joins when the reaction arrives, possibly after the policy changed. The
    policy is resolved once (as connect does); the later changes are on disk only."""
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    seat = fleet.seat("alpha")
    seat._resolve_voice_channel_policy()
    seat._auto_join_cfg = {"enabled": True, "channel_ids": [str(vf.OTHER_VC)]}
    vf.write_fleet(root, {**_MAP, str(vf.OTHER_VC): {"join": "never"}}, alpha_legacy=[vf.OTHER_VC])
    await seat._do_auto_join(seat._fake_vc(vf.OTHER_VC), vf.HOME_TEXT, vf.GUILD)
    assert vf.GUILD not in seat._voice_clients
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])  # control: allowed again
    await seat._do_auto_join(seat._fake_vc(vf.OTHER_VC), vf.HOME_TEXT, vf.GUILD)
    assert vf.GUILD in seat._voice_clients
