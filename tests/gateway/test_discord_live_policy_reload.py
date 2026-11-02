"""``allow_bots`` and ``free_response_channels`` take effect per message, without a gateway restart.

Owner ruling (2026-11-02): gates live, plugins restart-only. Both keys used to be read from the
connect-time env snapshot, so an operator who narrowed or widened the policy had to restart the
gateway to make it real — and had no way to tell whether the edit had landed.

The live read is bound to the adapter's own profile home at connect() and cached on the two files'
stat signature (mtime_ns, size, ino, ctime_ns). A message therefore costs two ``stat`` calls and a
changed file is picked up on the very next message — the tests below never sleep.

Fail-closed is part of the contract: an unparseable value keeps the previous behaviour (no bot
admission, no free-response channels) and says so once, instead of silently opening the gate.
"""

import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import discord
import pytest

import plugins.platforms.discord.adapter as adapter_mod
from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
CHANNEL_ID = 4242
OTHER_USER_ID = 820
OTHER_BOT_ID = 888

LOGGER = "plugins.platforms.discord.adapter"


@pytest.fixture(autouse=True)
def _clean_cache():
    cache = getattr(adapter_mod, "_LIVE_POLICY_CACHE", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def _adapter(home, *, extra=None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra=extra or {})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._get_parent_channel_id = Mock(return_value=None)
    adapter._discord_channel_keys = Mock(return_value={str(CHANNEL_ID)})
    adapter._threads = set()
    # Hermetic gates: the host's real .env must never decide a test (the live read is the file below).
    adapter._gate_env_snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    adapter._live_policy_home = home
    return adapter


def _bot_message(msg_id=4001):
    # Inline `@bot` token: the (independent, restart-only) bots_require_inline_mention gate is on by
    # default, so an unmentioned bot message would be dropped for a reason unrelated to allow_bots.
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=OTHER_BOT_ID, bot=True, display_name="peer"),
        channel=SimpleNamespace(id=CHANNEL_ID, name="general", parent_id=None),
        content=f"<@{BOT_ID}> handoff",
        mentions=[],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
    )


def _third_party_mention(msg_id=5001):
    """A message that only the free-response check can admit (mentions someone else, not us)."""
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=OTHER_USER_ID, bot=False, display_name="someone"),
        channel=SimpleNamespace(id=CHANNEL_ID, name="general", parent_id=None),
        content=f"<@{OTHER_USER_ID}> look at this",
        mentions=[SimpleNamespace(id=OTHER_USER_ID, bot=False)],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
    )


def _write_env(home, **values):
    lines = [f"{key}={value}" for key, value in values.items()]
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Same-second rewrites can land inside one filesystem timestamp tick; nudge so the signature is
    # unambiguous even on a coarse-mtime fs (ext4 ns resolution does not need this).
    os.utime(home / ".env", (time.time() + 1, time.time() + 1))


def test_free_response_channels_edit_admits_in_the_same_process(tmp_path):
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CHANNELS="")
    adapter = _adapter(tmp_path)
    assert adapter._discord_free_response_channels() == set()

    # Not a free-response channel yet: the third-party mention is dropped.
    assert adapter._discord_message_admission(_third_party_mention(), claim=True)[0] is False

    # The operator adds the channel to .env — NO restart, no gateway reload.
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CHANNELS=str(CHANNEL_ID))
    assert adapter._discord_free_response_channels() == {str(CHANNEL_ID)}
    assert adapter._discord_message_admission(_third_party_mention(msg_id=5002), claim=True)[0] is True


def test_free_response_channels_from_config_yaml_applies_live(tmp_path):
    # The sibling case from the incident: the key lived in config.yaml, not .env.
    (tmp_path / "config.yaml").write_text("discord:\n  allow_bots: none\n", encoding="utf-8")
    adapter = _adapter(tmp_path)
    assert adapter._discord_free_response_channels() == set()

    (tmp_path / "config.yaml").write_text(
        f"discord:\n  allow_bots: none\n  free_response_channels: [{CHANNEL_ID}]\n", encoding="utf-8")
    os.utime(tmp_path / "config.yaml", (time.time() + 2, time.time() + 2))
    assert adapter._discord_free_response_channels() == {str(CHANNEL_ID)}


def test_allow_bots_edit_admits_a_bot_in_the_same_process(tmp_path):
    _write_env(tmp_path, DISCORD_ALLOW_BOTS="none")
    adapter = _adapter(tmp_path)
    assert adapter._get_allow_bots() == "none"
    assert adapter._discord_message_admission(_bot_message(), claim=True)[0] is False

    _write_env(tmp_path, DISCORD_ALLOW_BOTS="all")
    assert adapter._get_allow_bots() == "all"
    assert adapter._discord_message_admission(_bot_message(msg_id=4002), claim=True)[0] is True


def test_allow_bots_from_config_yaml_applies_live(tmp_path):
    (tmp_path / "config.yaml").write_text("discord:\n  allow_bots: none\n", encoding="utf-8")
    adapter = _adapter(tmp_path)
    assert adapter._get_allow_bots() == "none"

    (tmp_path / "config.yaml").write_text("discord:\n  allow_bots: mentions\n", encoding="utf-8")
    os.utime(tmp_path / "config.yaml", (time.time() + 2, time.time() + 2))
    assert adapter._get_allow_bots() == "mentions"


def test_unparseable_allow_bots_fails_closed_and_logs_once(tmp_path, caplog):
    # "mentionss" used to match neither admission branch and therefore admit EVERY bot.
    caplog.set_level(logging.WARNING, logger=LOGGER)
    _write_env(tmp_path, DISCORD_ALLOW_BOTS="mentionss")
    adapter = _adapter(tmp_path)

    for index in range(5):
        assert adapter._get_allow_bots() == "none"
        assert adapter._discord_message_admission(_bot_message(msg_id=6000 + index), claim=True)[0] is False

    warnings = [r for r in caplog.records if "Unparseable allow_bots" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "mentionss" in warnings[0].getMessage()


def test_unparseable_free_response_channels_fails_closed(tmp_path, caplog):
    # A truncated JSON list must not be comma-split into junk channel keys.
    caplog.set_level(logging.WARNING, logger=LOGGER)
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CHANNELS='["4242",')
    adapter = _adapter(tmp_path)
    assert adapter._discord_free_response_channels() == set()
    assert adapter._discord_free_response_channels() == set()
    warnings = [r for r in caplog.records if "Unparseable free_response_channels" in r.getMessage()]
    assert len(warnings) == 1


def test_live_read_is_cached_until_the_file_changes(tmp_path):
    # The cache is stat-signature based (not a TTL): an unchanged file is not re-parsed, and a
    # changed one is visible immediately.
    _write_env(tmp_path, DISCORD_ALLOW_BOTS="none")
    first = adapter_mod._read_policy_file_values(tmp_path)
    assert adapter_mod._read_policy_file_values(tmp_path) is first

    _write_env(tmp_path, DISCORD_ALLOW_BOTS="all")
    refreshed = adapter_mod._read_policy_file_values(tmp_path)
    assert refreshed is not first
    assert refreshed["env"]["DISCORD_ALLOW_BOTS"] == "all"


def test_unreadable_config_yaml_keeps_the_previous_policy(tmp_path, caplog):
    # A half-written config.yaml must not flip a gate: the connect-time snapshot stands.
    _write_env(tmp_path, DISCORD_ALLOW_BOTS="none")
    adapter = _adapter(tmp_path)
    assert adapter._get_allow_bots() == "none"

    (tmp_path / "config.yaml").write_text("discord: [unclosed\n", encoding="utf-8")
    os.utime(tmp_path / "config.yaml", (time.time() + 3, time.time() + 3))
    assert adapter._get_allow_bots() == "none"
    assert adapter._discord_free_response_channels() == set()


def test_unconnected_adapter_keeps_the_snapshot_and_reads_nothing(tmp_path, monkeypatch):
    # No profile home bound => the live read stays out entirely (this is what keeps tests and
    # standalone senders off the host's config).
    _write_env(tmp_path, DISCORD_ALLOW_BOTS="all")
    adapter = _adapter(None)
    assert adapter._get_allow_bots() == "none"
    assert adapter._discord_free_response_channels() == set()
