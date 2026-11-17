"""``gateway.bot_loop_guard.human_relays``: a listed sender id (a webhook bridge relaying people, e.g. an
in-game chat bridge) is a human to the loop guard on every path: it resets the chain and is never
dropped by ``max_hops`` / ``duplicate`` / ``reply_to_reply``, never counted against the window budget and
never refused by a cooldown. Unlisted bots and webhooks keep the old behaviour. Admission
(``*_ALLOW_BOTS``, bot allowlists, mention gates) is not the guard's business and is unchanged.
"""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings, settings_from_config
from gateway.session import Platform, SessionSource

RELAY = "1571240603155435684"  # the vm-chat bridge's webhook id
OTHER_WEBHOOK = "1571240603155435685"
PEER = "999000999"
HUMAN = "100200300"
THREAD = "1440000000000000001"
GROUP_CHAT = "-1001234567890"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in ("TELEGRAM_ALLOW_BOTS", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
                "TELEGRAM_GROUP_ALLOWED_CHATS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def clock():
    state = {"t": 1_000_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda s: state.__setitem__("t", state["t"] + s))


def _from_config(**block):
    """Settings the way a seat gets them: parsed from its config.yaml block."""
    block = {"max_events": 20, "window_seconds": 60, "cooldown_seconds": 600, "human_relays": [RELAY], **block}
    return settings_from_config({"gateway": {"bot_loop_guard": block}})


@pytest.fixture
def settings():
    """Mutable holder: the guard re-reads settings per call, so a test can edit the config live."""
    return {"value": _from_config()}


def _ingress_runner(clock, settings):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._is_user_authorized_for_source = lambda source, **kw: True  # admission is not under test
    runner._adapter_profile_for_source = lambda source: ""
    runner._bot_loop_guard = BotLoopGuard(settings=lambda: settings["value"], clock=clock.now)
    return runner


_MESSAGE_IDS = itertools.count(1)


def _event(user, text, *, is_bot=True):
    from gateway.platforms.base import MessageEvent

    source = SessionSource(platform=Platform.DISCORD, chat_id=THREAD, chat_type="thread", user_id=user,
                           user_name=f"u{user}", is_bot=is_bot)
    return MessageEvent(text=text, message_id=f"m{next(_MESSAGE_IDS)}", source=source)


async def _admit(runner, event) -> bool:
    """The ingress gate; a delivered reply to a bot is noted as the turn would note it."""
    admitted = await runner._hm_admit_event(event) is not None
    if admitted:
        runner._note_bot_reply(event.source)
    return admitted


# --- chain rules through the ingress gate ----------------------------------------------------


@pytest.mark.asyncio
async def test_listed_relay_is_admitted_past_max_hops_duplicates_and_reply_to_reply(clock, settings):
    runner = _ingress_runner(clock, settings)
    # 12 relay lines, each answered by the seat (reply_to_reply bait), some repeated verbatim (duplicate bait).
    verdicts = [await _admit(runner, _event(RELAY, f"<player> line {n % 3}")) for n in range(12)]
    assert verdicts == [True] * 12


@pytest.mark.asyncio
async def test_unlisted_webhook_is_still_dropped_at_max_hops(clock, settings):
    """Control: the same traffic from a webhook not on the list ends at the 4th hop (reply rule on too)."""
    settings["value"] = _from_config(max_events=1000, bot_replies_expect_reply=True)
    runner = _ingress_runner(clock, settings)
    verdicts = [await _admit(runner, _event(OTHER_WEBHOOK, f"<player> line {n}")) for n in range(8)]
    assert verdicts == [True] * 4 + [False] * 4


@pytest.mark.asyncio
async def test_listed_relay_resets_the_chain_so_a_real_bot_gets_a_fresh_hop_budget(clock, settings):
    settings["value"] = _from_config(max_events=1000, bot_replies_expect_reply=True)
    runner = _ingress_runner(clock, settings)
    first = [await _admit(runner, _event(PEER, f"bot {n}")) for n in range(5)]
    assert first == [True] * 4 + [False]

    assert await _admit(runner, _event(RELAY, "<player> hello")) is True

    second = [await _admit(runner, _event(PEER, f"bot again {n}")) for n in range(5)]
    assert second == [True] * 4 + [False]


@pytest.mark.asyncio
async def test_removing_the_key_restores_the_old_behaviour_without_a_restart(clock, settings):
    """Control: settings are re-read per call, so dropping ``human_relays`` from config applies at once."""
    runner = _ingress_runner(clock, settings)
    assert all([await _admit(runner, _event(RELAY, f"<player> a{n}")) for n in range(6)])

    settings["value"] = _from_config(human_relays=None, bot_replies_expect_reply=True)
    assert getattr(settings["value"], "human_relays", frozenset()) == frozenset()
    verdicts = [await _admit(runner, _event(RELAY, f"<player> b{n}")) for n in range(6)]
    assert verdicts == [True] * 4 + [False] * 2


def test_note_reply_is_not_recorded_for_a_relay(clock, settings):
    runner = _ingress_runner(clock, settings)
    guard = runner._bot_loop_guard
    conversation = runner._bot_loop_guard_conversation(_event(RELAY, "x").source)
    assert guard.chain_admit(conversation, PEER, "bot question", is_bot=True) == (True, "ok")
    runner._note_bot_reply(_event(RELAY, "x").source)
    # Had the relay been noted as the replied-to bot, nothing would change for PEER; check the chain directly.
    assert guard._chains[conversation].replied_to == ""


# --- window budget and cooldown through the authz path ----------------------------------------


def _telegram(user_id, is_bot=True):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_CHAT, chat_type="group", user_id=user_id,
                         user_name=f"u{user_id}", is_bot=is_bot)


@pytest.fixture
def authz_runner(monkeypatch, clock, settings):
    from gateway.run import GatewayRunner

    monkeypatch.setenv("TELEGRAM_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", HUMAN)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", GROUP_CHAT)
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    runner._bot_loop_guard = BotLoopGuard(settings=lambda: settings["value"], clock=clock.now)
    return runner


def _inbound(runner, source) -> bool:
    return runner._is_user_authorized(source) and runner._admit_bot_message(source)


def test_listed_relay_never_trips_the_window_budget(authz_runner):
    assert all(_inbound(authz_runner, _telegram(RELAY)) for _ in range(60))
    assert authz_runner._bot_loop_guard.tracked_conversations == 0
    # Control: an unlisted bot in the same chat is still cut at the budget (20).
    verdicts = [_inbound(authz_runner, _telegram(PEER)) for _ in range(25)]
    assert verdicts == [True] * 20 + [False] * 5


def test_listed_relay_is_not_refused_by_a_cooldown_other_bots_tripped(authz_runner):
    for _ in range(21):
        _inbound(authz_runner, _telegram(PEER))
    assert authz_runner._is_user_authorized(_telegram(PEER)) is False  # the chat is cooling down
    assert authz_runner._is_user_authorized(_telegram(OTHER_WEBHOOK)) is False  # control: unlisted bot refused
    assert _inbound(authz_runner, _telegram(RELAY)) is True


def test_relay_status_does_not_bypass_admission(monkeypatch, authz_runner):
    """The list is a loop-guard setting, not an allowlist: a relay the platform refuses stays refused."""
    monkeypatch.delenv("TELEGRAM_GROUP_ALLOWED_CHATS")
    monkeypatch.delenv("TELEGRAM_ALLOW_BOTS")
    assert authz_runner._is_user_authorized(_telegram(RELAY)) is False


# --- config parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ([RELAY, " 42 ", ""], frozenset({RELAY, "42"})),
    ([1571240603155435684], frozenset({RELAY})),
    (f"{RELAY}, 42 ,,", frozenset({RELAY, "42"})),
    ((RELAY,), frozenset({RELAY})),
    ({"id": RELAY}, frozenset()),
    (42, frozenset()),
    (None, frozenset()),
    (True, frozenset()),
    ([None, True, {"x": 1}, RELAY], frozenset({RELAY})),
], ids=["list", "int-list", "csv", "tuple", "dict-junk", "int-junk", "null", "bool-junk", "junk-items"])
def test_human_relays_parsing(raw, expected):
    cfg = {"gateway": {"bot_loop_guard": {"human_relays": raw}}}
    assert settings_from_config(cfg).human_relays == expected


def test_human_relays_default_is_empty():
    assert BotLoopGuardSettings().human_relays == frozenset()
    assert settings_from_config({}).human_relays == frozenset()
    assert settings_from_config({"gateway": {"bot_loop_guard": {"max_hops": 2}}}).human_relays == frozenset()


def test_guard_counts_as_bot():
    guard = BotLoopGuard(settings=lambda: BotLoopGuardSettings(human_relays=frozenset({RELAY})))
    assert guard.counts_as_bot(RELAY, True) is False
    assert guard.counts_as_bot(f" {RELAY} ", True) is False
    assert guard.counts_as_bot(OTHER_WEBHOOK, True) is True
    assert guard.counts_as_bot(HUMAN, False) is False
    assert guard.counts_as_bot("", True) is True
    assert guard.counts_as_bot(None, True) is True


# --- Bot Mode runtime hop count (agent/bot_chain.py reads the same block) ---------------------


def test_a_relay_authored_turn_starts_a_new_bot_mode_chain(monkeypatch):
    from agent import bot_chain

    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    relays = _from_config(max_hops=2)
    monkeypatch.setattr(bot_chain, "_settings", lambda: relays)
    agent = SimpleNamespace(session_id="s", _turn_author=None)
    for n in range(5):
        author = {"id": RELAY, "name": "bridge", "is_bot": True}
        agent._turn_author = author
        bot_chain.on_turn_start(agent, author, None, f"<player> {n}")
        assert bot_chain.admit_send(agent, ["bot:gamma"], f"relay {n}").refusal is None

    # Control: the same turns from an unlisted bot run past max_hops and the send is refused.
    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    agent = SimpleNamespace(session_id="t", _turn_author=None)
    refusals = []
    for n in range(5):
        author = {"id": OTHER_WEBHOOK, "name": "other", "is_bot": True}
        agent._turn_author = author
        bot_chain.on_turn_start(agent, author, None, f"<player> {n}")
        refusals.append(bot_chain.admit_send(agent, ["bot:gamma"], f"other {n}").refusal)
    assert refusals[-1] is not None and "limit of 2" in refusals[-1]


# --- transport-profile scoping (multiplexed seats): the list is read where the rest of the guard is ---


def _profile_home(tmp_path, name, relays_yaml="", max_events=1000):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n  bot_loop_guard:\n    max_hops: 2\n    bot_replies_expect_reply: true\n"
        f"    max_events: {max_events}\n{relays_yaml}", encoding="utf-8")
    return home


def _transport_runner(transport_home):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._principal_authorized = lambda source, **kw: True  # admission is not under test
    runner._adapter_profile_for_source = lambda source: "transport"
    runner._authorization_home_for_source = lambda source: Path(transport_home)
    runner._bot_loop_guard = BotLoopGuard()  # the real load_settings
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("listed_in, expected", [
    ("transport", [True] * 6),
    ("ambient", [True, True, False, False, False, False]),  # control: an ambient listing does not leak in
])
async def test_relay_list_is_read_under_the_transport_profile(tmp_path, monkeypatch, listed_in, expected):
    relays = f"    human_relays: ['{RELAY}']\n"
    transport = _profile_home(tmp_path, "transport", relays if listed_in == "transport" else "")
    ambient = _profile_home(tmp_path, "ambient", relays if listed_in == "ambient" else "")
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    runner = _transport_runner(transport)
    verdicts = [await _admit(runner, _event(RELAY, f"<player> {n}")) for n in range(6)]
    assert verdicts == expected


def test_cooldown_peek_reads_the_relay_list_under_the_transport_profile(tmp_path, monkeypatch):
    transport = _profile_home(tmp_path, "transport", f"    human_relays: ['{RELAY}']\n", max_events=2)
    monkeypatch.setenv("HERMES_HOME", str(_profile_home(tmp_path, "ambient")))
    runner = _transport_runner(transport)
    for _ in range(3):
        runner._admit_bot_message_for_source(_event(PEER, "x").source)  # trips the chat's cooldown
    assert runner._is_user_authorized_for_source(_event(PEER, "x").source) is False
    relay = _event(RELAY, "y").source
    assert runner._is_user_authorized_for_source(relay) is True
    assert runner._admit_bot_message_for_source(relay) is True


def test_an_unusable_value_is_logged_once(caplog):
    import gateway.bot_loop_guard as module

    module._WARNED_RELAY_VALUES.discard(repr(1571240603155435684))
    cfg = {"gateway": {"bot_loop_guard": {"human_relays": 1571240603155435684}}}
    with caplog.at_level("WARNING", logger="gateway.bot_loop_guard"):
        assert settings_from_config(cfg).human_relays == frozenset()
        assert settings_from_config(cfg).human_relays == frozenset()
    assert sum("human_relays" in r.getMessage() for r in caplog.records) == 1
