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


# ── free_response_categories: the same live, profile-scoped read (category carry) ─────────────────

CATEGORY_ID = 500


def _forum_post(msg_id=7001, *, category_id=CATEGORY_ID):
    """A post in a forum (thread) whose parent forum sits in a category; mentions someone else."""
    forum = SimpleNamespace(id=700, name="app-forum", category_id=category_id,
                            category=SimpleNamespace(name="Projects") if category_id else None)
    thread = SimpleNamespace(id=7100 + msg_id % 100, name="post", parent_id=700, parent=forum)
    return SimpleNamespace(
        id=msg_id,
        author=SimpleNamespace(id=OTHER_USER_ID, bot=False, display_name="someone"),
        channel=thread,
        content=f"<@{OTHER_USER_ID}> look at this",
        mentions=[SimpleNamespace(id=OTHER_USER_ID, bot=False)],
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
        attachments=[],
    )


def _category_adapter(home, *, extra=None):
    adapter = _adapter(home, extra=extra)
    # Real channel keys for the forum post (the base helper pins them to the test channel).
    adapter._discord_channel_keys = Mock(return_value={"7100", "700"})
    return adapter


def test_free_response_categories_env_edit_admits_in_the_same_process(tmp_path):
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="")
    adapter = _category_adapter(tmp_path)
    assert adapter._discord_free_response_categories() == set()
    assert adapter._discord_message_admission(_forum_post(), claim=True)[0] is False

    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES=str(CATEGORY_ID))
    assert adapter._discord_free_response_categories() == {str(CATEGORY_ID)}
    assert adapter._discord_message_admission(_forum_post(msg_id=7002), claim=True)[0] is True


def test_free_response_categories_from_config_yaml_applies_live(tmp_path):
    (tmp_path / "config.yaml").write_text("discord:\n  allow_bots: none\n", encoding="utf-8")
    adapter = _category_adapter(tmp_path)
    assert adapter._discord_free_response_categories() == set()

    (tmp_path / "config.yaml").write_text(
        f"discord:\n  free_response_categories: [{CATEGORY_ID}]\n", encoding="utf-8")
    os.utime(tmp_path / "config.yaml", (time.time() + 2, time.time() + 2))
    assert adapter._discord_free_response_categories() == {str(CATEGORY_ID)}
    assert adapter._discord_message_admission(_forum_post(msg_id=7003), claim=True)[0] is True


def test_unparseable_free_response_categories_fails_closed_and_logs_once(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES='["500",')
    adapter = _category_adapter(tmp_path)
    for index in range(3):
        assert adapter._discord_free_response_categories() == set()
        assert adapter._discord_message_admission(_forum_post(msg_id=7010 + index), claim=True)[0] is False
    warnings = [r for r in caplog.records if "Unparseable free_response_categories" in r.getMessage()]
    assert len(warnings) == 1


def test_free_response_categories_two_homes_resolve_independently(tmp_path):
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "config.yaml").write_text(
        f"discord:\n  free_response_categories: [{CATEGORY_ID}]\n", encoding="utf-8")
    (home_b / "config.yaml").write_text("discord:\n  require_mention: true\n", encoding="utf-8")
    adapter_a, adapter_b = _category_adapter(home_a), _category_adapter(home_b)
    assert adapter_a._discord_message_admission(_forum_post(msg_id=7020), claim=True)[0] is True
    assert adapter_b._discord_message_admission(_forum_post(msg_id=7021), claim=True)[0] is False

    # An edit to B's own file changes B only.
    (home_b / "config.yaml").write_text(
        "discord:\n  free_response_categories: [\"999\"]\n", encoding="utf-8")
    os.utime(home_b / "config.yaml", (time.time() + 2, time.time() + 2))
    assert adapter_b._discord_free_response_categories() == {"999"}
    assert adapter_a._discord_free_response_categories() == {str(CATEGORY_ID)}


def test_free_response_categories_snapshot_is_profile_scoped(tmp_path, monkeypatch):
    # Another profile's value in process env must not become this adapter's policy: the connect-time
    # snapshot (taken inside this profile's scope) owns the env fallback, as for the channel key.
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CATEGORIES", str(CATEGORY_ID))
    assert "DISCORD_FREE_RESPONSE_CATEGORIES" in adapter_mod._GATE_ENV_KEYS
    adapter = _category_adapter(None)
    assert adapter._discord_free_response_categories() == set()


def test_free_response_categories_change_logs_info_once_per_change(tmp_path, caplog):
    import hermes_logging
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    caplog.set_level(logging.INFO, logger=LOGGER)
    adapter = _category_adapter(tmp_path)
    token = set_hermes_home_override(str(tmp_path))
    try:
        assert adapter._discord_free_response_categories() == set()
        (tmp_path / "config.yaml").write_text(
            f"discord:\n  free_response_categories: [{CATEGORY_ID}]\n", encoding="utf-8")
        os.utime(tmp_path / "config.yaml", (time.time() + 2, time.time() + 2))
        for _ in range(3):
            assert adapter._discord_free_response_categories() == {str(CATEGORY_ID)}
        (tmp_path / "config.yaml").write_text(
            "discord:\n  free_response_categories: [\"999\"]\n", encoding="utf-8")
        os.utime(tmp_path / "config.yaml", (time.time() + 4, time.time() + 4))
        for _ in range(2):
            assert adapter._discord_free_response_categories() == {"999"}
    finally:
        reset_hermes_home_override(token)
    changes = [r for r in caplog.records if "free_response_categories now" in r.getMessage()]
    assert [r.getMessage().split("now ", 1)[1] for r in changes] == [
        "['500'] (applied live)", "['999'] (applied live)"]
    assert all(r.levelno == logging.INFO for r in changes)
    # Emitted on the calling (profile-scoped) path, so hermes_logging routes it to that profile's log.
    assert {getattr(r, "hermes_home", None) for r in changes} == {str(tmp_path.resolve())}


def test_connect_logs_the_resolved_free_response_categories(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    (tmp_path / "config.yaml").write_text(
        f"discord:\n  free_response_categories: [{CATEGORY_ID}]\n", encoding="utf-8")
    adapter = _category_adapter(None)
    adapter._voice_fx_cfg = {}
    monkeypatch.setattr(adapter_mod, "_resolve_profile_home", lambda: tmp_path)
    adapter._snapshot_gate_env()          # what connect() does, inside the profile scope
    lines = [r.getMessage() for r in caplog.records if "free-response categories:" in r.getMessage()]
    assert lines == [f"[Discord] free-response categories: ['500'] (home={tmp_path.name}, applied live)"]
    # The connect line is the baseline: an unchanged read afterwards logs nothing more.
    adapter._discord_free_response_categories()
    assert not [r for r in caplog.records if "free_response_categories now" in r.getMessage()]


# ── clearing a live key applies live too (C3) ─────────────────────────────────────────────────────
# Clearing used to be inert: an empty YAML value counted as "not mentioned", so the getter fell back to
# the connect-time config.extra and kept the OLD list (and the drift row agreed, so nothing warned).
# Contract now: a key present in config.yaml is authoritative even when empty; a key that THIS adapter
# has seen in its files and that is now gone resolves empty as well; an empty .env line is ignored
# (config.yaml decides); a value that only ever came from process env is untouched.

def _write_yaml(home, body, bump):
    (home / "config.yaml").write_text(body, encoding="utf-8")
    os.utime(home / "config.yaml", (time.time() + bump, time.time() + bump))


@pytest.mark.parametrize("key,getter", [
    ("free_response_channels", "_discord_free_response_channels"),
    ("free_response_categories", "_discord_free_response_categories"),
])
@pytest.mark.parametrize("cleared", ['""', "[]", None])
def test_cleared_list_key_applies_live(tmp_path, key, getter, cleared):
    _write_yaml(tmp_path, f"discord:\n  {key}: \"111\"\n", 1)
    adapter = _adapter(tmp_path, extra={key: "111"})      # startup bridge seeded extra from the file
    assert getattr(adapter, getter)() == {"111"}

    _write_yaml(tmp_path, f"discord:\n  {key}: \"111,222\"\n", 2)
    assert getattr(adapter, getter)() == {"111", "222"}   # widening was already live

    body = f"discord:\n  {key}: {cleared}\n" if cleared is not None else "discord:\n  require_mention: true\n"
    _write_yaml(tmp_path, body, 3)
    assert getattr(adapter, getter)() == set()


@pytest.mark.parametrize("cleared", ['""', None])
def test_cleared_allow_bots_applies_live(tmp_path, cleared):
    _write_yaml(tmp_path, "discord:\n  allow_bots: all\n", 1)
    adapter = _adapter(tmp_path, extra={"allow_bots": "all"})
    assert adapter._get_allow_bots() == "all"
    assert adapter._discord_message_admission(_bot_message(msg_id=9001), claim=True)[0] is True

    body = f"discord:\n  allow_bots: {cleared}\n" if cleared is not None else "discord:\n  require_mention: true\n"
    _write_yaml(tmp_path, body, 2)
    assert adapter._get_allow_bots() == "none"
    assert adapter._discord_message_admission(_bot_message(msg_id=9002), claim=True)[0] is False


def test_cleared_key_is_reported_by_the_drift_row(tmp_path):
    _write_yaml(tmp_path, "discord:\n  free_response_categories: [\"500\"]\n", 1)
    adapter = _adapter(tmp_path, extra={"free_response_categories": "500"})
    adapter._voice_fx_cfg = {}
    adapter._capture_policy_snapshot()
    _write_yaml(tmp_path, "discord:\n  free_response_categories: []\n", 2)
    rows = {row.name: row for row in adapter._config_drift_keys()}
    assert rows["free_response_categories"].snapshot == "500"
    assert rows["free_response_categories"].disk == ""
    assert rows["free_response_categories"].drifted()


def test_empty_env_line_falls_through_to_config_yaml(tmp_path):
    # W3b: an empty .env value is ignored, not a clear: config.yaml decides.
    _write_yaml(tmp_path, "discord:\n  free_response_categories: [\"500\"]\n", 1)
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="")
    adapter = _adapter(tmp_path)
    assert adapter._discord_free_response_categories() == {"500"}
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="600")
    assert adapter._discord_free_response_categories() == {"600"}     # non-empty .env wins
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="")
    assert adapter._discord_free_response_categories() == {"500"}     # emptied .env: YAML again


def test_process_env_only_value_is_not_cleared_by_absent_files(tmp_path):
    # A value that never came from this profile's files (container env, unit-file Environment=) has
    # nothing on disk to "remove": the absent key keeps the connect-time snapshot.
    adapter = _adapter(tmp_path)
    adapter._gate_env_snapshot["DISCORD_ALLOW_BOTS"] = "all"
    adapter._gate_env_snapshot["DISCORD_FREE_RESPONSE_CATEGORIES"] = "500"
    _write_yaml(tmp_path, "discord:\n  require_mention: true\n", 1)
    for _ in range(2):
        assert adapter._get_allow_bots() == "all"
        assert adapter._discord_free_response_categories() == {"500"}



# ── the two read-failure cases, pinned (impl review 2026-11-04, blockers 1 and 2) ─────────────────

def test_config_yaml_typo_after_connect_keeps_all_three_live_keys(tmp_path):
    # Since a removed key clears live, the yaml_readable guard is the only thing between a hand-edit
    # typo and a profile silently losing its policy. All three live keys must survive it.
    _write_yaml(tmp_path, (
        "discord:\n  allow_bots: all\n  free_response_channels: \"111\"\n"
        "  free_response_categories: [\"500\"]\n"), 1)
    adapter = _adapter(tmp_path, extra={
        "allow_bots": "all", "free_response_channels": "111", "free_response_categories": "500"})
    assert adapter._get_allow_bots() == "all"
    assert adapter._discord_free_response_channels() == {"111"}
    assert adapter._discord_free_response_categories() == {"500"}

    _write_yaml(tmp_path, "discord: [unclosed\n", 2)
    for _ in range(2):
        assert adapter._get_allow_bots() == "all"
        assert adapter._discord_free_response_channels() == {"111"}
        assert adapter._discord_free_response_categories() == {"500"}


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
def test_unreadable_env_reads_as_empty_and_config_yaml_decides(tmp_path):
    # load_env_file returns {} for an unreadable file, so it cannot be told from an emptied one.
    # Pinned behaviour: the .env value drops out and config.yaml decides, as after a restart.
    _write_yaml(tmp_path, "discord:\n  free_response_categories: [\"600\"]\n", 1)
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="500")
    adapter = _adapter(tmp_path)
    assert adapter._discord_free_response_categories() == {"500"}

    env = tmp_path / ".env"
    env.chmod(0)
    os.utime(env, (time.time() + 5, time.time() + 5))
    try:
        assert adapter._discord_free_response_categories() == {"600"}
    finally:
        env.chmod(0o600)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
def test_unreadable_env_with_no_yaml_key_clears_a_seen_env_value(tmp_path):
    # Same root cause, no config.yaml value to fall back to: the seen key resolves empty (fails
    # closed: a mention is required again), which is also what a restart would give.
    _write_env(tmp_path, DISCORD_FREE_RESPONSE_CATEGORIES="500")
    adapter = _adapter(tmp_path, extra={"free_response_categories": "500"})   # snapshot holds 500 too
    assert adapter._discord_free_response_categories() == {"500"}
    env = tmp_path / ".env"
    env.chmod(0)
    os.utime(env, (time.time() + 5, time.time() + 5))
    try:
        assert adapter._discord_free_response_categories() == set()
    finally:
        env.chmod(0o600)
