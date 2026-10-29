"""Voice auto-join must bind the seat's OWN channel, speak while in the room, and stop.

Three behaviours the upstream auto-join (#96527) does not provide, all seen in production:

1. It bound ``guild.system_channel`` (the server's default text channel) instead of the seat's own
   home channel, so replies landed in a room nobody was watching.
2. It never enabled spoken replies, so the bot joined the voice room and stayed silent.
3. Nothing tore that enablement down, so it would outlive the voice session and turn every later
   TEXT reply in that channel into speech (and, once out of the VC, an audio file).

These exercise the REAL resolver and the REAL teardown, not lambdas standing in for them: an
ablation that makes ``_auto_join_home_text_channel`` raise must fail this file.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter
from gateway import run_voice as _run_voice_mod


class FakeTextChannel:
    """Stands in for ``discord.TextChannel`` so ``isinstance`` works without a live client."""

    def __init__(self, cid, name="chan", *, send=True, guild_id=1):
        self.id = cid
        self.name = name
        self.guild = SimpleNamespace(id=guild_id, me=object())
        self.send = AsyncMock()
        self._send = send

    def permissions_for(self, me):
        return SimpleNamespace(send_messages=self._send, read_messages=True)


class NotATextChannel:
    """A voice-channel lookalike: same shape, deliberately NOT a FakeTextChannel subclass.

    (Subclassing it would satisfy ``isinstance(ch, discord.TextChannel)`` and the test would pass
    for the wrong reason.)
    """

    def __init__(self, cid, name="a-voice-room", *, send=True, guild_id=1):
        self.id = cid
        self.name = name
        self.guild = SimpleNamespace(id=guild_id, me=object())
        self.send = AsyncMock()

    def permissions_for(self, me):
        return SimpleNamespace(send_messages=True, read_messages=True)


class _Stub:
    """Minimal adapter surface for the methods under test."""

    _auto_join_home_text_channel = DiscordAdapter._auto_join_home_text_channel
    _select_auto_join_text_channel = DiscordAdapter._select_auto_join_text_channel
    _do_auto_join = DiscordAdapter._do_auto_join
    leave_voice_channel = DiscordAdapter.leave_voice_channel

    def __init__(self, channels=(), *, env_home=None, config_home=None):
        self._by_id = {ch.id: ch for ch in channels}
        self._client = SimpleNamespace(
            get_channel=lambda cid: self._by_id.get(cid),
            get_guild=lambda gid: None,
        )
        self._env_home = env_home
        self._config_home = config_home
        self._auto_tts_enabled_chats: set[str] = set()
        self._auto_tts_disabled_chats: set[str] = set()
        self._auto_join_tts_chats: set[str] = set()
        # leave_voice_channel surface
        self._voice_locks: dict = {}
        self._voice_receivers: dict = {}
        self._voice_listen_tasks: dict = {}
        self._voice_text_channels: dict = {}
        self._voice_sources: dict = {}
        self._voice_mixers: dict = {}
        self._voice_clients: dict = {}
        self._voice_timeout_tasks: dict = {}
        self.join_voice_channel = AsyncMock(return_value=True)

    def _gate_env(self, key):
        return self._env_home

    def _is_allowed_user(self, uid, guild=None, is_dm=False):
        return True


def _voice_mixin():
    """The gateway object that owns ``_toggle_adapter_auto_tts_set`` (a mixin on the runner)."""
    for name in dir(_run_voice_mod):
        candidate = getattr(_run_voice_mod, name)
        if isinstance(candidate, type) and hasattr(candidate, "_toggle_adapter_auto_tts_set"):
            return candidate
    raise RuntimeError("no runner class exposes _toggle_adapter_auto_tts_set")


def _toggle_auto_tts(adapter, chat_id, on, *, enable):
    """Call the runner's static toggle without letting it rebind as an instance method."""
    return staticmethod(_voice_mixin()._toggle_adapter_auto_tts_set)(adapter, chat_id, on, enable=enable)


def _guild(system=None, text_channels=()):
    return SimpleNamespace(
        id=1, me=object(), system_channel=system, text_channels=list(text_channels)
    )


def _config_with(home_chat_id):
    return {"platforms": {"discord": {"home_channel": {"chat_id": home_chat_id}}}}


@pytest.fixture(autouse=True)
def _fake_text_channel_type():
    """Make ``isinstance(ch, discord.TextChannel)`` meaningful without importing discord."""
    with patch.object(adapter_mod, "discord", SimpleNamespace(TextChannel=FakeTextChannel)):
        yield


# ------------------------------------------------------------------- the resolver itself


def test_resolver_uses_the_home_channel_from_config():
    stub = _Stub([FakeTextChannel(222, "game-servers")], config_home=222)
    with patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        assert stub._auto_join_home_text_channel(_guild()) == 222


def test_resolver_falls_back_to_the_env_home_when_config_has_none():
    stub = _Stub([FakeTextChannel(333, "from-env")], env_home="333")
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        assert stub._auto_join_home_text_channel(_guild()) == 333


def test_resolver_prefers_config_over_env():
    stub = _Stub([FakeTextChannel(222), FakeTextChannel(333)], env_home="333", config_home=222)
    with patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        assert stub._auto_join_home_text_channel(_guild()) == 222


def test_resolver_rejects_a_home_channel_from_another_guild():
    stub = _Stub([FakeTextChannel(222, "elsewhere", guild_id=99)], config_home=222)
    with patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        assert stub._auto_join_home_text_channel(_guild()) is None


def test_resolver_rejects_a_home_that_is_not_a_text_channel():
    stub = _Stub([NotATextChannel(222, "a-voice-room")], config_home=222)
    with patch.object(adapter_mod, "discord", SimpleNamespace(TextChannel=FakeTextChannel)), \
            patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        assert stub._auto_join_home_text_channel(_guild()) is None


def test_resolver_rejects_a_home_the_bot_cannot_post_in():
    stub = _Stub([FakeTextChannel(222, "read-only", send=False)], config_home=222)
    with patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        assert stub._auto_join_home_text_channel(_guild()) is None


def test_resolver_returns_none_when_nothing_is_configured():
    stub = _Stub([])
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        assert stub._auto_join_home_text_channel(_guild()) is None


# ------------------------------------------------------------------- channel selection order


def test_selection_prefers_the_seat_home_over_the_guild_default():
    home = FakeTextChannel(222, "game-servers")
    default = FakeTextChannel(111, "general")
    stub = _Stub([home, default], config_home=222)
    with patch("hermes_cli.config.read_raw_config", return_value=_config_with(222)):
        chosen = stub._select_auto_join_text_channel(
            _guild(default, [default, home]), voice_channel=None
        )
    assert chosen == 222, "auto-join bound the guild default instead of the seat's channel"


def test_selection_falls_back_to_the_guild_default():
    default = FakeTextChannel(111, "general")
    stub = _Stub([default])
    with patch("hermes_cli.config.read_raw_config", return_value={}):
        chosen = stub._select_auto_join_text_channel(_guild(default, [default]), voice_channel=None)
    assert chosen == 111


# ------------------------------------------------------------------- speech on, and off again


@pytest.mark.asyncio
async def test_auto_join_turns_on_spoken_replies_and_records_that_it_did():
    bound = FakeTextChannel(222, "game-servers")
    stub = _Stub([bound], config_home=222)
    stub._voice_text_channels[1] = 222          # join_voice_channel bound this session
    stub._auto_tts_disabled_chats = {"222"}

    await stub._do_auto_join(SimpleNamespace(id=999), 222, 1)

    assert "222" in stub._auto_tts_enabled_chats
    assert "222" not in stub._auto_tts_disabled_chats
    assert "222" in stub._auto_join_tts_chats, "the enable must be attributable so leave can undo it"


@pytest.mark.asyncio
async def test_auto_join_does_not_claim_a_chat_the_user_already_enabled():
    """Otherwise leaving would silently clear a /voice tts the user set by hand."""
    stub = _Stub([FakeTextChannel(222)], config_home=222)
    stub._voice_text_channels[1] = 222
    stub._auto_tts_enabled_chats = {"222"}      # already on, by the user

    await stub._do_auto_join(SimpleNamespace(id=999), 222, 1)

    assert "222" in stub._auto_tts_enabled_chats
    assert "222" not in stub._auto_join_tts_chats


@pytest.mark.asyncio
async def test_auto_join_installs_no_enable_when_the_session_is_already_gone():
    """A leave woken by the released lock can run first; the enable must not outlive it."""
    stub = _Stub([FakeTextChannel(222)], config_home=222)
    stub._voice_text_channels.clear()           # leave_voice_channel already ran

    await stub._do_auto_join(SimpleNamespace(id=999), 222, 1)

    assert not stub._auto_tts_enabled_chats
    assert not stub._auto_join_tts_chats


@pytest.mark.asyncio
async def test_a_failed_join_enables_nothing():
    stub = _Stub([FakeTextChannel(222)], config_home=222)
    stub._voice_text_channels[1] = 222
    stub.join_voice_channel = AsyncMock(return_value=False)

    await stub._do_auto_join(SimpleNamespace(id=999), 222, 1)

    assert not stub._auto_tts_enabled_chats
    assert not stub._auto_join_tts_chats


@pytest.mark.asyncio
async def test_leaving_voice_undoes_the_auto_join_speech_enable():
    """Otherwise every later TEXT reply in that channel is spoken, forever."""
    stub = _Stub([FakeTextChannel(222)], config_home=222)
    stub._voice_text_channels[1] = 222
    stub._auto_tts_enabled_chats = {"222"}
    stub._auto_join_tts_chats = {"222"}

    await stub.leave_voice_channel(1)

    assert "222" not in stub._auto_tts_enabled_chats
    assert "222" not in stub._auto_join_tts_chats


@pytest.mark.asyncio
async def test_leaving_voice_keeps_a_users_explicit_voice_tts():
    """A chat enabled by /voice tts was never recorded by auto-join, so leave must not clear it."""
    stub = _Stub([FakeTextChannel(222)], config_home=222)
    stub._voice_text_channels[1] = 222
    stub._auto_tts_enabled_chats = {"222"}
    stub._auto_join_tts_chats = set()

    await stub.leave_voice_channel(1)

    assert "222" in stub._auto_tts_enabled_chats


def test_explicit_voice_tts_takes_ownership_back_from_a_later_auto_join():
    """auto-join -> /voice tts -> leave must keep speaking: the user's choice outranks the join's."""
    adapter = SimpleNamespace(
        _auto_tts_enabled_chats={"222"},
        _auto_tts_disabled_chats=set(),
        _auto_join_tts_chats={"222"},       # enabled by the auto-join a moment ago
    )
    _toggle_auto_tts(adapter, "222", True, enable=True)

    assert "222" not in adapter._auto_join_tts_chats, (
        "an explicit /voice tts must stop the auto-join owning the enable, or leave clears it"
    )
