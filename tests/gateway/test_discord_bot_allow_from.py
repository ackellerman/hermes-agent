"""``discord.bot_allow_from`` / ``DISCORD_ALLOWED_BOTS``: narrow ``allow_bots`` to named bot user IDs.

Topology under test is a multiplexed gateway: the DEFAULT profile unscoped (its home IS the root,
``HERMES_HOME``), and two profile-scoped secondaries, ``alpha`` and ``beta``, each with its OWN
``config.yaml``. Every adapter is a real ``DiscordAdapter`` built and snapshotted inside its own
profile scope (the shape ``connect()`` has); every config value goes through the real
``apply_yaml_config_fn`` bridge. All IDs are synthetic.

Probe map:
  P1/P1b  per-profile isolation at ingress; no process-env write from a scoped profile
  P2      gateway authz (the busy-path gate) uses the owning profile's list
  P3      a list edit applies on the next message
  P4      ID shapes; ``[]`` = no list; a name never matches (fail closed)
  P5/P5b  history keeps an unlisted bot, tagged ``[unverified]``; allowlist computed once per fetch
  P6      busy path refuses an unlisted bot
  P7      missed-message recovery (``claim=False``) uses the same gate
  P8      real ``load_gateway_config``, all three YAML placements
  P9/P9b  the verdict is the adapter's own profile's, whatever scope the caller is in
  P10     the primary profile with a list; no leak to a secondary; live clear
  P11     a live ``allow_bots`` edit reaches gateway authz (enable and disable); non-str guard
  P12     a bot in the HUMAN allowlist but not the bot list is refused
  P13     snapshot + drift rows; P14 connect INFO line + one typo WARNING
  P15     ``DISCORD_ALLOWED_BOTS`` is a profile gate (scrubbed from children); P16 primary bridge
  P17     authz env fallback (no adapter, or a non-bool verdict) honours the scoped list
  P18     a checker that RAISES refuses the bot (fail closed)
  P19     the refusal marker is falsy, and the caller still refuses
"""
import asyncio
import contextlib
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest

import plugins.platforms.discord.adapter as adapter_mod
from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ALPHA = "910000000000000001"
BOT_BETA = "910000000000000002"
BOT_GAMMA = "910000000000000003"
BOT_DELTA = "910000000000000004"
BOT_OUTSIDER = "910000000000000009"
BOT_DEFAULT = "910000000000000010"
HUMAN = "920000000000000001"
CHANNEL = 4242
ADAPTER_LOGGER = "plugins.platforms.discord.adapter"


@pytest.fixture(autouse=True)
def _hermetic_env():
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("DISCORD_") or key.startswith("GATEWAY_ALLOW"):
            del os.environ[key]
    adapter_mod._LIVE_POLICY_CACHE.clear()
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    yield
    secret_scope.set_multiplex_active(prev)
    os.environ.clear()
    os.environ.update(saved)
    adapter_mod._LIVE_POLICY_CACHE.clear()


def _write_cfg(home, body, env=""):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")
    (home / ".env").write_text(env, encoding="utf-8")
    # Bump mtime so the live-policy signature cache sees the edit even within one clock tick.
    st = os.stat(home / "config.yaml")
    os.utime(home / "config.yaml", ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))


@contextlib.contextmanager
def _scope(home, *, scoped=True):
    if not scoped:
        yield
        return
    tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(tok)


def _load(home, *, self_id, scoped=True):
    import hermes_yaml
    cfg = hermes_yaml.safe_load((home / "config.yaml").read_text()) or {}
    with _scope(home, scoped=scoped):
        seeded = adapter_mod._apply_yaml_config(cfg, cfg.get("discord") or {}) or {}
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake", extra=dict(seeded)))
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=int(self_id), bot=True))
        adapter._text_batch_delay_seconds = 0
        adapter._snapshot_gate_env()
    adapter._probe_home = home
    adapter._probe_scoped = scoped
    return adapter


def _bot_message(adapter, author_id, msg_id):
    """Explicit inline @mention of this bot from a bot author in a guild channel."""
    me = adapter._client.user
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=int(author_id), bot=True, display_name=f"bot{author_id}"),
        channel=SimpleNamespace(id=CHANNEL, name="general", parent_id=None),
        content=f"<@{me.id}> handoff",
        mentions=[me],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
        reference=None,
    )


def _admit(adapter, author_id, msg_id, *, claim=True):
    with _scope(adapter._probe_home, scoped=adapter._probe_scoped):
        return adapter._discord_message_admission(_bot_message(adapter, author_id, msg_id), claim=claim)[0]


def _admit_under(adapter, author_id, msg_id, *, under=None):
    """Same message as _admit, but evaluated with NO scope (``under=None``) or inside ANOTHER profile's."""
    with _scope(under, scoped=under is not None):
        return adapter._discord_message_admission(_bot_message(adapter, author_id, msg_id), claim=True)[0]


@pytest.fixture
def multi(tmp_path, monkeypatch):
    root = tmp_path / "hh"
    # Production: the primary (unscoped) profile's home IS the root. Without this the default
    # adapter binds conftest's home and its rows pass through the env snapshot, untested.
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_cfg(root, "discord:\n  allow_bots: mentions\n")
    alpha = root / "profiles" / "alpha"
    _write_cfg(alpha, f"discord:\n  allow_bots: mentions\n  thread_require_mention: true\n"
                      f"  bot_allow_from: [{BOT_BETA}, {BOT_GAMMA}, {BOT_DELTA}]\n")
    beta = root / "profiles" / "beta"
    _write_cfg(beta, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n")
    # Load order = the multiplexer's: secondaries connect in their scopes, default unscoped.
    adapters = {
        "alpha": _load(alpha, self_id=BOT_ALPHA),
        "beta": _load(beta, self_id=BOT_BETA),
        "default": _load(root, self_id=BOT_DEFAULT, scoped=False),
    }
    assert adapters["default"]._live_policy_home == root, "fixture control: primary must bind the root"
    assert adapters["beta"]._live_policy_home == beta
    return adapters


def _runner_for(adapter):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.DISCORD: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_k: False)
    runner.pairing_stores = {}
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._delivery_adapter_for = lambda _s: adapter
    runner._authorization_adapter = lambda *_a, **_k: adapter
    runner._adapter_profile_for_source = lambda _s: None
    return runner


def _bot_source(user_id):
    return SessionSource(platform=Platform.DISCORD, chat_id=str(CHANNEL), chat_type="group",
                         user_id=user_id, user_name="bot", is_bot=True)


def _authz(adapter, bot, *, home=None, scoped=True):
    runner = _runner_for(adapter)
    with _scope(home if home is not None else adapter._probe_home, scoped=scoped and adapter._probe_scoped):
        return runner._principal_authorized(_bot_source(bot), allow_adapter_delegation=True)


class _Chan:
    id = CHANNEL

    def __init__(self, msgs):
        self._msgs = msgs

    def history(self, *, limit, before, after=None, oldest_first=None):
        msgs = self._msgs

        async def _it():
            for m in msgs:
                yield m
        return _it()


def _hist(msg_id, author, text):
    return SimpleNamespace(id=msg_id, author=author, content=text, attachments=[],
                           type=discord.MessageType.default)


def _bot_author(bot_id, name):
    return SimpleNamespace(id=int(bot_id), display_name=name, name=name, bot=True)


# ── P1: per-profile isolation of the allowlist at ingress ────────────────────────────────────────
def test_p1_each_profile_enforces_its_own_list(multi):
    assert _admit(multi["alpha"], BOT_BETA, 1) is True
    assert _admit(multi["alpha"], BOT_OUTSIDER, 2) is False
    assert _admit(multi["beta"], BOT_ALPHA, 3) is True, "beta must admit alpha (its own list)"
    assert _admit(multi["beta"], BOT_GAMMA, 4) is False
    assert _admit(multi["default"], BOT_OUTSIDER, 5) is True, "default has no list: allow_bots alone decides"


def test_p1b_no_process_env_written_by_a_scoped_profile(multi):
    assert "DISCORD_ALLOWED_BOTS" not in os.environ


# ── P2: gateway authz (also the busy-path gate) uses the owning profile's list ──────────────────
@pytest.mark.parametrize("who,bot,expected", [
    ("beta", BOT_ALPHA, True), ("beta", BOT_GAMMA, False),
    ("alpha", BOT_DELTA, True), ("alpha", BOT_ALPHA, False),
    ("default", BOT_OUTSIDER, True),
])
def test_p2_authz_verdict_per_profile(multi, who, bot, expected):
    assert _authz(multi[who], bot) is expected


# ── P3: live policy — narrowing/widening applies on the next message, like allow_bots ───────────
def test_p3_bot_allow_from_edit_applies_live(multi):
    beta = multi["beta"]
    assert _admit(beta, BOT_OUTSIDER, 10) is False
    _write_cfg(beta._probe_home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}, {BOT_OUTSIDER}]\n")
    assert _admit(beta, BOT_OUTSIDER, 11) is True
    assert _authz(beta, BOT_OUTSIDER) is True, "authz follows the same live list"
    _write_cfg(beta._probe_home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n")
    assert _admit(beta, BOT_OUTSIDER, 12) is False, "narrowing applies live too"
    assert _authz(beta, BOT_OUTSIDER) is False


# ── P4: id shapes and empty-list semantics ──────────────────────────────────────────────────────
@pytest.mark.parametrize("value,author,expected", [
    (f"[{BOT_ALPHA}]", BOT_ALPHA, True),              # YAML int
    (f"['{BOT_ALPHA}']", BOT_ALPHA, True),            # quoted string
    (f"'{BOT_ALPHA}'", BOT_ALPHA, True),              # scalar
    (f"['<@{BOT_ALPHA}>']", BOT_ALPHA, True),         # pasted mention form, like allow_from accepts
    ("[]", BOT_OUTSIDER, True),                       # empty list == no list (allow_bots alone decides)
    ("['alpha']", BOT_ALPHA, False),                  # a name never matches a bot id: fail closed
], ids=["int", "quoted", "scalar", "mention-form", "empty", "name"])
def test_p4_id_shapes(tmp_path, value, author, expected):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: {value}\n")
    adapter = _load(home, self_id=BOT_BETA)
    assert _admit(adapter, author, 20) is expected


def test_p4b_csv_scalar_two_ids(tmp_path):
    """An UNQUOTED two-id scalar (``bot_allow_from: <id1>,<id2>``) loads as ONE str and still names two bots."""
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: {BOT_ALPHA},{BOT_GAMMA}\n")
    beta = _load(home, self_id=BOT_BETA)
    assert _admit(beta, BOT_ALPHA, 30) is True
    assert _admit(beta, BOT_GAMMA, 31) is True
    assert _admit(beta, BOT_OUTSIDER, 32) is False
    assert _authz(beta, BOT_GAMMA) is True
    assert _authz(beta, BOT_OUTSIDER) is False


# ── P5: history backfill keeps a peer's turn (as background) for a narrowed seat ─────────────────
def test_p5_history_keeps_unlisted_bot_as_unverified(multi, monkeypatch):
    beta = multi["beta"]
    monkeypatch.setattr(discord, "Thread", type("T", (), {}), raising=False)
    calls = []
    real = getattr(beta, "_bot_allowlist", None)  # absent on a tree without the feature: the tag asserts decide
    if real is not None:
        monkeypatch.setattr(beta, "_bot_allowlist", lambda: calls.append(1) or real())
    chan = _Chan([
        _hist(4, _bot_author(BOT_GAMMA, "Gamma"), "gamma finding"),
        _hist(3, _bot_author(BOT_DELTA, "Delta"), "delta finding"),
        _hist(2, _bot_author(BOT_ALPHA, "Alpha"), "alpha brief"),
    ])
    with _scope(beta._probe_home):
        out = asyncio.run(beta._fetch_channel_context(chan, before=SimpleNamespace(id=9)))
    assert "[Alpha [bot]] alpha brief" in out and "[unverified] [Alpha" not in out
    assert "[unverified] [Gamma [bot]] gamma finding" in out, "a peer's turn is context, not a trigger"
    assert "[unverified] [Delta [bot]] delta finding" in out
    assert len(calls) == 1, f"the allowlist is resolved once per fetch, not per message ({len(calls)})"


# ── P5b: same, single-profile (no leak in play) ─────────────────────────────────────────────────
def test_p5b_single_profile_history_semantics(tmp_path, monkeypatch):
    secret_scope.set_multiplex_active(False)
    home = tmp_path / "solo"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    beta = _load(home, self_id=BOT_BETA, scoped=False)
    monkeypatch.setattr(discord, "Thread", type("T", (), {}), raising=False)
    chan = _Chan([_hist(3, _bot_author(BOT_GAMMA, "Gamma"), "gamma finding")])
    out = asyncio.run(beta._fetch_channel_context(chan, before=SimpleNamespace(id=9)))
    assert "[unverified] [Gamma [bot]] gamma finding" in out


# ── P6: the busy path (run_busy._handle_active_session_busy_message) refuses an unlisted bot ────
# In production ingress drops the unlisted bot first; this is the second layer, for events that
# reach the busy path some other way (recovery, a routed source).
def test_p6_busy_path_drops_unlisted_bot(multi, caplog):
    from gateway.platforms.base import MessageEvent, MessageType
    beta = multi["beta"]
    runner = _runner_for(beta)
    runner._authorization_home_for_source = lambda _s: None
    runner._draining = False
    ev = MessageEvent(text="steer", message_type=MessageType.TEXT, source=_bot_source(BOT_GAMMA), message_id="b1")
    caplog.set_level(logging.WARNING)
    with _scope(beta._probe_home):
        try:
            handled = asyncio.run(runner._handle_active_session_busy_message(ev, "sk"))
        except Exception as exc:  # downstream machinery absent in the skeleton => the bot got PAST authz
            handled = f"passed-authz ({type(exc).__name__})"
    assert "unauthorized user in active session" in caplog.text, handled


# ── P7: missed-message recovery path (claim=False) uses the same gate ────────────────────────────
def test_p7_recovered_message_path(multi):
    assert _admit(multi["beta"], BOT_GAMMA, 77, claim=False) is False
    assert _admit(multi["beta"], BOT_ALPHA, 78, claim=False) is True


# ── P8: real load_gateway_config for a secondary profile, every YAML placement the fleet uses ───
@pytest.mark.parametrize("body", [
    f"discord:\n  allowed_channels: '1'\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n",
    f"discord:\n  auto_thread: false\nplatforms:\n  discord:\n    allow_bots: mentions\n    bot_allow_from: [{BOT_ALPHA}]\n",
    f"platforms:\n  discord:\n    extra:\n      allow_bots: mentions\n      bot_allow_from: [{BOT_ALPHA}]\n",
], ids=["toplevel", "platforms-discord", "platforms-discord-extra"])
def test_p8_real_loader(tmp_path, monkeypatch, body):
    from gateway.config import load_gateway_config
    root = tmp_path / "hh"
    home = root / "profiles" / "beta"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text("discord:\n  allow_bots: mentions\n")
    (home / "config.yaml").write_text(body)
    (home / ".env").write_text("DISCORD_BOT_TOKEN=fake\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
    try:
        pc = load_gateway_config().platforms[Platform.DISCORD]
        adapter = DiscordAdapter(pc)
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=int(BOT_BETA), bot=True))
        adapter._text_batch_delay_seconds = 0
        adapter._snapshot_gate_env()
        adapter._probe_home, adapter._probe_scoped = home, True
        assert adapter._get_allow_bots() == "mentions"
        assert _admit(adapter, BOT_ALPHA, 1) is True
        assert _admit(adapter, BOT_GAMMA, 2) is False, "an ignored placement fails OPEN"
        assert adapter.bot_admission_mode() == "mentions"
    finally:
        secret_scope.reset_secret_scope(tok)
    assert "DISCORD_ALLOWED_BOTS" not in os.environ


def test_p8b_mentions_without_list_under_platforms_discord(tmp_path, monkeypatch):
    """``allow_bots: mentions`` directly under ``platforms.discord`` (not ``extra``) and NO list: any
    mentioned bot is admitted at ingress and by authz, and nothing is written to process env."""
    from gateway.config import load_gateway_config
    root = tmp_path / "hh"
    home = root / "profiles" / "beta"
    home.mkdir(parents=True)
    (root / "config.yaml").write_text("discord:\n  allow_bots: mentions\n")
    (home / "config.yaml").write_text("discord:\n  auto_thread: false\nplatforms:\n  discord:\n    allow_bots: mentions\n")
    (home / ".env").write_text("DISCORD_BOT_TOKEN=fake\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home), profile_home=str(home))
    try:
        pc = load_gateway_config().platforms[Platform.DISCORD]
        adapter = DiscordAdapter(pc)
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=int(BOT_BETA), bot=True))
        adapter._text_batch_delay_seconds = 0
        adapter._snapshot_gate_env()
        adapter._probe_home, adapter._probe_scoped = home, True
        assert adapter._get_allow_bots() == "mentions"
        assert adapter.bot_admission_mode() == "mentions"
        assert _admit(adapter, BOT_GAMMA, 1) is True
        assert _admit(adapter, BOT_OUTSIDER, 2) is True
        assert _authz(adapter, BOT_GAMMA, home=home) is True
    finally:
        secret_scope.reset_secret_scope(tok)
    assert "DISCORD_ALLOW_BOTS" not in os.environ
    assert "DISCORD_ALLOWED_BOTS" not in os.environ


# ── P9 / P9b: the verdict is bound to the adapter's own profile, not the caller's ambient scope ──
def test_p9_ingress_scope_independent(multi):
    alpha_home = multi["alpha"]._probe_home
    beta = multi["beta"]
    assert _admit_under(beta, BOT_ALPHA, 101) is True
    assert _admit_under(beta, BOT_GAMMA, 102) is False
    assert _admit_under(beta, BOT_ALPHA, 103, under=alpha_home) is True
    assert _admit_under(beta, BOT_GAMMA, 104, under=alpha_home) is False


def test_p9b_authz_scope_independent(multi):
    beta = multi["beta"]
    alpha_home = multi["alpha"]._probe_home
    assert _authz(beta, BOT_ALPHA, home=alpha_home) is True  # a sibling's scope
    assert _authz(beta, BOT_GAMMA, home=alpha_home) is False
    assert _authz(beta, BOT_ALPHA, scoped=False) is True     # no scope at all
    assert _authz(beta, BOT_GAMMA, scoped=False) is False


# ── P10: the PRIMARY (unscoped) profile WITH a list; no leak; live clear ─────────────────────────
def test_p10_primary_with_list(tmp_path, monkeypatch):
    root = tmp_path / "hh"
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_cfg(root, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n")
    beta_home = root / "profiles" / "beta"
    _write_cfg(beta_home, "discord:\n  allow_bots: mentions\n")
    beta = _load(beta_home, self_id=BOT_BETA)
    default = _load(root, self_id=BOT_DEFAULT, scoped=False)
    assert default._live_policy_home == root
    env_written = os.environ.get("DISCORD_ALLOWED_BOTS")
    assert _admit(default, BOT_ALPHA, 201) is True
    assert _admit(default, BOT_OUTSIDER, 202) is False
    assert _admit(beta, BOT_OUTSIDER, 203) is True, f"primary list leaked into secondary (env={env_written!r})"
    assert _authz(beta, BOT_OUTSIDER) is True
    _write_cfg(root, "discord:\n  allow_bots: mentions\n")
    assert _admit(default, BOT_OUTSIDER, 204) is True, "removing the primary's key must clear it live"


# ── P11: a live allow_bots edit reaches gateway authz, not only ingress ──────────────────────────
_HUMAN_ONLY = f"discord:\n  auto_thread: false\nplatforms:\n  discord:\n    allow_from: ['{HUMAN}']\n"
_BOTS_ON = (f"discord:\n  auto_thread: false\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n"
            f"platforms:\n  discord:\n    allow_from: ['{HUMAN}']\n")


def test_p11_live_allow_bots_enable_reaches_authz(tmp_path):
    root = tmp_path / "hh"
    _write_cfg(root, "discord:\n  allow_bots: mentions\n")
    home = root / "profiles" / "beta"
    _write_cfg(home, _HUMAN_ONLY)
    beta = _load(home, self_id=BOT_BETA)
    assert _admit(beta, BOT_ALPHA, 301) is False
    assert _authz(beta, BOT_ALPHA) is False
    _write_cfg(home, _BOTS_ON)
    assert _admit(beta, BOT_ALPHA, 302) is True, "ingress is live"
    assert _authz(beta, BOT_ALPHA) is True, "authz refuses the bot ingress just admitted (allow_bots read at connect)"
    assert _authz(beta, BOT_GAMMA) is False


def test_p11b_live_allow_bots_disable_reaches_authz(tmp_path):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, _BOTS_ON)
    beta = _load(home, self_id=BOT_BETA)
    assert _authz(beta, BOT_ALPHA) is True
    _write_cfg(home, _HUMAN_ONLY.replace("auto_thread: false\n", "auto_thread: false\n  allow_bots: none\n"))
    assert _admit(beta, BOT_ALPHA, 311) is False
    assert _authz(beta, BOT_ALPHA) is False, "a connect-time 'mentions' must not outlive a live 'none'"


@pytest.mark.parametrize("live_mode,expected", [
    (MagicMock(), True),     # not a str: ignored, connect-time extra ('mentions') decides
    (None, True),
    ("none", False),         # a real str verdict wins over the connect-time extra
    ("MENTIONS ", True),
])
def test_p11c_bot_admission_mode_str_guard(live_mode, expected):
    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={"allow_bots": "mentions"}),
        bot_admission_mode=lambda: live_mode,
        bot_sender_allowed=lambda _uid: True,
    )
    runner = _runner_for(adapter)
    assert runner._principal_authorized(_bot_source(BOT_ALPHA), allow_adapter_delegation=True) is expected


# ── P12: a bot listed in the HUMAN allowlist but not bot_allow_from is refused ───────────────────
def test_p12_bot_in_human_allowlist_is_refused(tmp_path):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n",
               env=f"DISCORD_ALLOWED_USERS={HUMAN},{BOT_GAMMA}\n")
    beta = _load(home, self_id=BOT_BETA)
    runner = _runner_for(beta)
    human = SessionSource(platform=Platform.DISCORD, chat_id=str(CHANNEL), chat_type="group",
                          user_id=HUMAN, user_name="human", is_bot=False)
    with _scope(home):
        assert runner._principal_authorized(human, allow_adapter_delegation=True) is True
        assert runner._principal_authorized(_bot_source(BOT_GAMMA), allow_adapter_delegation=True) is False
        assert runner._principal_authorized(_bot_source(BOT_ALPHA), allow_adapter_delegation=True) is True


# ── P13: snapshot row + drift row (a matched pair) ───────────────────────────────────────────────
def test_p13_snapshot_and_drift_rows(multi):
    beta = multi["beta"]
    assert beta._policy_snapshot["bot_allow_from"] == BOT_ALPHA
    _write_cfg(beta._probe_home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_OUTSIDER}, {BOT_ALPHA}]\n")
    rows = {row.name: row for row in beta._config_drift_keys()}
    row = rows["bot_allow_from"]
    assert row.snapshot == BOT_ALPHA
    assert row.disk == f"{BOT_ALPHA} {BOT_OUTSIDER}"
    assert row.reloadable is True
    assert "bot_allow_from" not in {r.name for r in multi["default"]._config_drift_keys()}, \
        "a profile that never named the key has no row"


# ── P14: connect INFO line (the deploy witness) and one WARNING for a typo ──────────────────────
def test_p14_connect_info_and_typo_warning(tmp_path, caplog):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: ['alpha', {BOT_ALPHA}]\n")
    caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER)
    beta = _load(home, self_id=BOT_BETA)
    assert f"bot allowlist: {BOT_ALPHA} (home=beta, applied live)" in caplog.text
    assert _admit(beta, BOT_ALPHA, 401) is True, "valid entries survive a typo beside them"
    assert _admit(beta, BOT_GAMMA, 402) is False
    assert _admit(beta, BOT_ALPHA, 403) is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "not Discord user IDs" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "reason=bot-not-in-allowlist" in caplog.text


# ── P15: the env name is a profile gate (scrubbed from children spawned for another profile) ─────
def test_p15_allowed_bots_is_a_profile_gate_env():
    from tools.environments.local_env_policy import is_profile_gate_env
    assert is_profile_gate_env("DISCORD_ALLOWED_BOTS") is True
    assert "DISCORD_ALLOWED_BOTS" in adapter_mod._GATE_ENV_KEYS
    assert "DISCORD_ALLOWED_BOTS" in adapter_mod._LIVE_POLICY_ENV_KEYS


# ── P16: the primary's YAML→env bridge (single profile, the PR's config-bridge intent) ──────────
def test_p16_primary_bridge_writes_csv(tmp_path, monkeypatch):
    secret_scope.set_multiplex_active(False)
    cfg = {"discord": {"bot_allow_from": [int(BOT_ALPHA), BOT_GAMMA]}}
    seeded = adapter_mod._apply_yaml_config(cfg, cfg["discord"]) or {}
    assert seeded["bot_allow_from"] == f"{BOT_ALPHA},{BOT_GAMMA}"
    assert os.environ.get("DISCORD_ALLOWED_BOTS") == f"{BOT_ALPHA},{BOT_GAMMA}"


# ── P17: authz env fallback — no adapter (bare runner) or a non-bool verdict ────────────────────
def _bare_runner():
    """A runner with no adapters at all: authz can only read the scoped env."""
    runner = _runner_for(None)
    runner.adapters = {}
    # The real resolvers, not the fixture's stand-ins: with no adapters they must find none.
    del runner._delivery_adapter_for, runner._authorization_adapter
    return runner


@pytest.mark.parametrize("env,bot,expected", [
    (f"DISCORD_ALLOW_BOTS=mentions\nDISCORD_ALLOWED_BOTS={BOT_ALPHA}\n", BOT_ALPHA, True),
    (f"DISCORD_ALLOW_BOTS=mentions\nDISCORD_ALLOWED_BOTS={BOT_ALPHA}\n", BOT_GAMMA, False),
    ("DISCORD_ALLOW_BOTS=mentions\n", BOT_GAMMA, True),           # no list: the mode alone decides
    (f"DISCORD_ALLOW_BOTS=none\nDISCORD_ALLOWED_BOTS={BOT_GAMMA}\n", BOT_GAMMA, False),
], ids=["listed", "unlisted", "no-list", "mode-none"])
def test_p17_env_fallback_without_adapter(tmp_path, env, bot, expected):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, "{}\n", env=env)
    runner = _bare_runner()
    assert runner._delivery_adapter_for(_bot_source(bot)) is None, "fixture control: no adapter"
    assert "DISCORD_ALLOWED_BOTS" not in os.environ, "fixture control: the list is in scoped env only"
    with _scope(home):
        assert runner._principal_authorized(_bot_source(bot), allow_adapter_delegation=True) is expected


@pytest.mark.parametrize("bot,expected", [(BOT_ALPHA, True), (BOT_GAMMA, False)])
def test_p17b_env_fallback_on_non_bool_verdict(tmp_path, bot, expected):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, "{}\n", env=f"DISCORD_ALLOW_BOTS=mentions\nDISCORD_ALLOWED_BOTS={BOT_ALPHA}\n")
    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={}),
        bot_admission_mode=lambda: "mentions",
        bot_sender_allowed=lambda _uid: MagicMock(),  # not a bool: must not decide
    )
    runner = _runner_for(adapter)
    with _scope(home):
        assert runner._principal_authorized(_bot_source(bot), allow_adapter_delegation=True) is expected


# ── P18: the adapter has a checker but it raises: refuse (its list may be YAML-only) ─────────────
def test_p18_checker_raising_refuses(tmp_path):
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, "{}\n")  # no env list: before the fix the bot fell through to "no list" = admit

    def _boom(_uid):
        raise RuntimeError("config unreadable")

    adapter = SimpleNamespace(
        config=SimpleNamespace(extra={"allow_bots": "mentions"}),
        bot_admission_mode=lambda: "mentions",
        bot_sender_allowed=_boom,
    )
    runner = _runner_for(adapter)
    with _scope(home):
        assert runner._principal_authorized(_bot_source(BOT_GAMMA), allow_adapter_delegation=True) is False
        assert runner._principal_authorized(_bot_source(BOT_ALPHA), allow_adapter_delegation=True) is False


# ── P19: the refusal marker is falsy, so a caller testing ``if grant:`` cannot grant ─────────────
def test_p19_refusal_marker_is_falsy_and_still_refuses(tmp_path):
    from gateway import authz_mixin
    marker = authz_mixin._BOT_REFUSED
    assert bool(marker) is False
    home = tmp_path / "hh" / "profiles" / "beta"
    _write_cfg(home, f"discord:\n  allow_bots: mentions\n  bot_allow_from: [{BOT_ALPHA}]\n",
               env=f"DISCORD_ALLOWED_USERS={BOT_GAMMA}\n")
    beta = _load(home, self_id=BOT_BETA)
    runner = _runner_for(beta)
    with _scope(home):
        grant = runner._chat_scoped_grant(_bot_source(BOT_GAMMA), None, True, True)
        assert grant is marker, grant
        # GAMMA is in the HUMAN allowlist: only the marker check keeps the ladder from granting.
        assert runner._principal_authorized(_bot_source(BOT_GAMMA), allow_adapter_delegation=True) is False
        assert runner._principal_authorized(_bot_source(BOT_ALPHA), allow_adapter_delegation=True) is True
