"""Occupancy: a voice channel another seat (or any bot) is in is REFUSED, never handed over.

Ruling V-Q1: ``/voice join`` from a seat's chat while another seat occupies the VC replies
"<Seat> is in <VC>; `/voice leave` there first", and an auto-join into an occupied VC is skipped,
never evicting the occupant. Occupancy = the VC's members that are bots or fleet bot ids, plus the
runner's in-process view (seats mid-connect, under the runner's per-(guild, VC) lock, and seats
already connected whose join the member cache has not caught up with).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio

from tests.fakes.platforms import discord_voice_fleet as vf  # first: installs the discord mock
from tests.fakes.platforms import discord_voice_session as fx
from gateway.voice_channel_policy import REPLY_OCCUPIED
from plugins.platforms.discord import adapter as adapter_mod

_MAP = {str(vf.MANUAL_VC): {"join": "manual", "allow_thread_move": True}}
_OCCUPIED_BY_BETA = REPLY_OCCUPIED.format(seat="beta", vc="Voice (manual)")


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
    for name in ("default", "alpha", "beta"):
        fleet.seat(name)
    fleet.operator_in(vf.MANUAL_VC)
    yield fleet
    await fleet.close()


@pytest.mark.asyncio
async def test_a_second_seat_is_refused_and_joins_after_the_first_leaves(fleet):
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith("Joined voice channel **Voice (manual)**")
    assert await fleet.voice_join("alpha", vf.ALPHA_TEXT) == _OCCUPIED_BY_BETA
    assert fleet.bots_in(vf.MANUAL_VC) == [vf.BOT_IDS["beta"]]
    leave = fleet.command("beta", vf.BETA_TEXT)
    leave.text = "/voice leave"
    await fleet.runner._handle_voice_channel_leave(leave)
    assert (await fleet.voice_join("alpha", vf.ALPHA_TEXT)).startswith("Joined voice channel **Voice (manual)**")  # control
    assert fleet.bots_in(vf.MANUAL_VC) == [vf.BOT_IDS["alpha"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("occupant,label", [
    (SimpleNamespace(id=7777, bot=True, display_name="music-bot"), "music-bot"),          # any bot
    (SimpleNamespace(id=vf.BOT_IDS["beta"], bot=False, display_name="?"), "beta"),        # fleet id, cache says human
])
async def test_a_bot_member_the_runner_did_not_connect_still_occupies(fleet, occupant, label):
    fleet.members[vf.MANUAL_VC].append(occupant)
    assert await fleet.voice_join("alpha", vf.ALPHA_TEXT) == REPLY_OCCUPIED.format(seat=label, vc="Voice (manual)")
    assert vf.GUILD not in fleet.seat("alpha")._voice_clients


@pytest.mark.asyncio
async def test_a_connected_seat_occupies_before_the_member_cache_shows_it(fleet):
    beta = fleet.seat("beta")
    assert await beta.join_voice_channel(beta._fake_vc(vf.MANUAL_VC), text_channel_id=vf.BETA_TEXT)
    fleet.members[vf.MANUAL_VC][:] = [m for m in fleet.members[vf.MANUAL_VC] if m.id != vf.BOT_IDS["beta"]]
    assert await fleet.voice_join("alpha", vf.ALPHA_TEXT) == _OCCUPIED_BY_BETA


@pytest.mark.asyncio
async def test_concurrent_joins_leave_exactly_one_seat_in_the_vc(fleet):
    for name in ("alpha", "beta"):  # the member list only updates after the connect yields
        ch = fleet.seat(name)._fake_vc(vf.MANUAL_VC)
        connect = ch.connect

        async def slow(*, cls=None, _connect=connect, **kw):
            await asyncio.sleep(0.01)
            return await _connect(cls=cls, **kw)

        ch.connect = slow
    replies = await asyncio.gather(fleet.voice_join("alpha", vf.ALPHA_TEXT), fleet.voice_join("beta", vf.BETA_TEXT))
    assert len(fleet.bots_in(vf.MANUAL_VC)) == 1
    assert sum("is in Voice (manual)" in r for r in replies) == 1


@pytest.mark.asyncio
async def test_the_own_bot_is_not_an_occupant(fleet):
    """A seat already in the VC re-typing /voice join elsewhere is a rebind, not an occupied VC."""
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith("Joined voice channel **Voice (manual)**")
    assert (await fleet.voice_join("beta", vf.THREAD, thread=True)).startswith("Joined voice channel **Voice (manual)**")


@pytest.mark.asyncio
async def test_a_rebind_is_not_refused_by_a_bot_that_arrived_after_the_seat(fleet):
    """The seat is already connected in the VC, so a rebind connects nothing: another bot that joined
    since is not an occupant of THIS seat's session. A second seat is still refused (control)."""
    assert (await fleet.voice_join("beta", vf.BETA_TEXT)).startswith("Joined voice channel **Voice (manual)**")
    fleet.members[vf.MANUAL_VC].append(SimpleNamespace(id=7777, bot=True, display_name="music-bot"))
    beta, client = fleet.seat("beta"), fleet.seat("beta")._voice_clients[vf.GUILD]
    assert (await fleet.voice_join("beta", vf.THREAD, thread=True)).startswith("Joined voice channel **Voice (manual)**")
    assert beta._voice_text_channels[vf.GUILD] == vf.THREAD and beta._voice_clients[vf.GUILD] is client
    assert await fleet.voice_join("alpha", vf.ALPHA_TEXT) == _OCCUPIED_BY_BETA


@pytest.mark.asyncio
async def test_a_seat_connected_elsewhere_is_still_checked_for_occupancy(fleet):
    """Only a client connected in THIS VC skips the check: a seat in another VC moving here is a new
    presence in this VC and must not walk past its occupant."""
    beta = fleet.seat("beta")
    assert await beta.join_voice_channel(beta._fake_vc(vf.OTHER_VC), text_channel_id=vf.BETA_TEXT)
    fleet.members[vf.MANUAL_VC].append(SimpleNamespace(id=7777, bot=True, display_name="music-bot"))
    assert await fleet.voice_join("beta", vf.BETA_TEXT) == REPLY_OCCUPIED.format(seat="music-bot", vc="Voice (manual)")
    assert beta._voice_clients[vf.GUILD].channel.id == vf.OTHER_VC


# --- auto-join never evicts ----------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("occupied", [True, False])
async def test_auto_join_skips_an_occupied_vc_and_never_evicts(root, fleet, monkeypatch, occupied):
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    beta, alpha = fleet.seat("beta"), fleet.seat("alpha")
    if occupied:  # beta is in the VC by a programmatic join (no runner path)
        assert await beta.join_voice_channel(beta._fake_vc(vf.OTHER_VC), text_channel_id=vf.BETA_TEXT)
    handlers = await fx.connect_capturing_handlers(alpha, monkeypatch)
    alpha._auto_join_cfg = {"enabled": True, "channel_ids": [str(vf.OTHER_VC)], "user_ids": [str(vf.USER)],
                            "join_mode": "automatic", "require_text_opt_in": False}
    monkeypatch.setattr(alpha, "_select_auto_join_text_channel", lambda guild, channel: vf.HOME_TEXT)
    fleet.operator_in(vf.OTHER_VC)
    operator = alpha._fake_guild.get_member(vf.USER)
    try:
        await handlers["on_voice_state_update"](operator, fx.voice_state(None), fx.voice_state(alpha._fake_vc(vf.OTHER_VC)))
        assert (vf.GUILD in alpha._voice_clients) is (not occupied)
        if occupied:
            assert beta._voice_clients[vf.GUILD].is_connected()
            assert fleet.bots_in(vf.OTHER_VC) == [vf.BOT_IDS["beta"]]
    finally:
        await alpha.disconnect()


@pytest.mark.asyncio
async def test_auto_join_without_a_runner_still_reads_the_member_list(root, fleet):
    alpha = fleet.seat("alpha")
    alpha.gateway_runner = None
    fleet.members.setdefault(vf.OTHER_VC, []).append(SimpleNamespace(id=7777, bot=True, display_name="music-bot"))
    vf.write_fleet(root, _MAP, alpha_legacy=[vf.OTHER_VC])
    alpha._resolve_voice_channel_policy()
    await alpha._do_auto_join(alpha._fake_vc(vf.OTHER_VC), vf.HOME_TEXT, vf.GUILD)
    assert vf.GUILD not in alpha._voice_clients
