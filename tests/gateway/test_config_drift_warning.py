"""Startup config drift: a change on disk that the running process will never see says so.

Owner ruling (2026-11-02): gates live, plugins restart-only. The gate keys are re-read per message
now, but ``plugins.enabled`` and the voice mixer are loaded once at startup — an edit there is inert
until a restart, and nothing said so. Once a minute the gateway compares the values it actually
loaded against the files on disk and logs ONE WARNING naming the keys.

The quiet case is the control: an unchanged config must produce no warning at all, or the warning
becomes noise the operator learns to skip.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import plugins.platforms.discord.adapter as adapter_mod
from gateway.config_drift import ConfigDriftMonitor, DriftKey, format_warning
from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

LOGGER = "plugins.platforms.discord.adapter"


@pytest.fixture(autouse=True)
def _clean_cache():
    cache = getattr(adapter_mod, "_LIVE_POLICY_CACHE", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def _adapter(home) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra={})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._gate_env_snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    adapter._voice_fx_cfg = adapter._load_voice_fx_config({})
    adapter._live_policy_home = home
    adapter._capture_policy_snapshot()
    return adapter


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")
    # Keep the stat signature unambiguous even on a coarse-mtime filesystem.
    import os
    import time
    os.utime(home / "config.yaml", (time.time() + 5, time.time() + 5))


# ── the monitor itself ───────────────────────────────────────────────────

def test_no_drift_is_silent(caplog):
    """The control: identical snapshot and disk produce no record at all."""
    caplog.set_level(logging.DEBUG)
    monitor = ConfigDriftMonitor(lambda: [
        DriftKey(name="plugins.enabled", snapshot="a,b", disk="a,b"),
        DriftKey(name="allow_bots", snapshot="none", disk="none", reloadable=True),
    ], scope="discord")
    assert monitor.check_once() is None
    assert monitor.check_once() is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_one_warning_per_change_not_per_minute(caplog):
    caplog.set_level(logging.DEBUG)
    rows = [DriftKey(name="plugins.enabled", snapshot="a", disk="a")]
    monitor = ConfigDriftMonitor(lambda: rows, scope="discord")

    assert monitor.check_once() is None            # quiet
    rows = [DriftKey(name="plugins.enabled", snapshot="a", disk="a,b")]
    assert monitor.check_once() is not None        # drift -> one warning
    assert monitor.check_once() is None            # same drift -> not repeated
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "plugins.enabled" in warnings[0].getMessage()
    assert "restart required" in warnings[0].getMessage()

    # Converged: the latch re-arms, so a later re-drift is reported again.
    rows = [DriftKey(name="plugins.enabled", snapshot="a,b", disk="a,b")]
    assert monitor.check_once() is None
    rows = [DriftKey(name="plugins.enabled", snapshot="a,b", disk="a")]
    assert monitor.check_once() is not None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_warning_separates_live_keys_from_restart_keys():
    drifted = [
        DriftKey(name="allow_bots", snapshot="none", disk="all", reloadable=True),
        DriftKey(name="plugins.enabled", snapshot="a", disk="a,b", reloadable=False),
    ]
    message = format_warning(drifted, scope="discord")
    assert "allow_bots" in message
    assert "applied live" in message
    assert "plugins.enabled" in message
    assert "restart required" in message
    # A reloadable-only change must NOT claim a restart is needed.
    assert "restart" not in format_warning(drifted[:1], scope="discord")


def test_provider_failure_is_contained(caplog):
    def boom():
        raise RuntimeError("config read exploded")

    caplog.set_level(logging.DEBUG)
    monitor = ConfigDriftMonitor(boom, scope="discord")
    assert monitor.check_once() is None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


# ── the adapter's keys (the real wiring) ─────────────────────────────────

def test_changed_plugins_enabled_warns_once_naming_the_key(tmp_path, caplog):
    _write_config(tmp_path, "plugins:\n  enabled: [discord]\n")
    adapter = _adapter(tmp_path)
    monitor = ConfigDriftMonitor(adapter._config_drift_keys, scope="discord")
    caplog.set_level(logging.INFO, logger="gateway.config_drift")

    assert monitor.check_once() is None             # quiet control
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    _write_config(tmp_path, "plugins:\n  enabled: [discord, memory]\n")
    assert monitor.check_once() is not None
    assert monitor.check_once() is None             # one warning per change

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    line = warnings[0].getMessage()
    assert "plugins.enabled" in line
    assert "restart required" in line
    assert warnings[0].name == "gateway.config_drift"


def test_voice_setting_drift_is_restart_only(tmp_path, caplog):
    _write_config(tmp_path, "discord:\n  voice_fx:\n    enabled: false\n")
    adapter = _adapter(tmp_path)
    monitor = ConfigDriftMonitor(adapter._config_drift_keys, scope="discord")
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert monitor.check_once() is None
    _write_config(tmp_path, "discord:\n  voice_fx:\n    enabled: true\n")
    message = monitor.check_once()
    assert message and "discord.voice_fx" in message and "restart required" in message


def test_gate_drift_is_reported_as_applied_live(tmp_path, caplog):
    (tmp_path / ".env").write_text("DISCORD_ALLOW_BOTS=none\n", encoding="utf-8")
    adapter = _adapter(tmp_path)
    monitor = ConfigDriftMonitor(adapter._config_drift_keys, scope="discord")
    caplog.set_level(logging.INFO, logger=LOGGER)

    assert monitor.check_once() is None                # disk still says none
    (tmp_path / ".env").write_text("DISCORD_ALLOW_BOTS=all\n", encoding="utf-8")
    message = monitor.check_once()
    assert message and "allow_bots" in message and "applied live" in message


def test_connected_adapter_feeds_the_process_wide_monitor(tmp_path, monkeypatch, caplog):
    """The real wiring: connect-time snapshot -> one process-wide monitor -> one WARNING."""
    import gateway.config_drift as config_drift_mod

    _write_config(tmp_path, "plugins:\n  enabled: [discord]\n")
    adapter = _adapter(tmp_path)
    monkeypatch.setattr(adapter_mod, "_resolve_profile_home", lambda: tmp_path)
    adapter._snapshot_gate_env()          # what connect() does, inside the profile scope
    try:
        adapter_mod._start_config_drift_monitor(adapter)
        monitor = config_drift_mod._MONITOR
        assert monitor is not None
        caplog.set_level(logging.INFO, logger="gateway.config_drift")

        assert monitor.check_once() is None            # unchanged on disk: silent
        _write_config(tmp_path, "plugins:\n  enabled: [discord, memory]\n")
        assert monitor.check_once() is not None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "plugins.enabled" in warnings[0].getMessage()
    finally:
        config_drift_mod.reset_monitor()
        adapter_mod._DRIFT_ADAPTERS.discard(adapter)


def test_unconnected_adapter_offers_no_drift_rows(tmp_path):
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._gate_env_snapshot = {key: "" for key in adapter_mod._GATE_ENV_KEYS}
    # Neither the profile home nor the snapshot was captured: nothing to compare, nothing to read.
    assert adapter._config_drift_keys() == []


def test_free_response_categories_drift_is_reported_as_applied_live(tmp_path, caplog):
    _write_config(tmp_path, "discord:\n  require_mention: true\n")
    adapter = _adapter(tmp_path)
    assert adapter._policy_snapshot.get("free_response_categories") == ""
    monitor = ConfigDriftMonitor(adapter._config_drift_keys, scope="discord")
    caplog.set_level(logging.INFO, logger="gateway.config_drift")

    assert monitor.check_once() is None                # disk agrees with the snapshot
    _write_config(tmp_path, "discord:\n  free_response_categories: [\"500\"]\n")
    message = monitor.check_once()
    assert message and "free_response_categories changed on disk (applied live)" in message
    assert "restart" not in message
