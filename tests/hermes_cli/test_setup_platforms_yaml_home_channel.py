"""Setup's home-channel diagnostic reads config.yaml as well as env.

A home channel configured in YAML is a configured home channel: reporting it as missing
sent operators to "fix" a setting that was already correct. Routing is untouched by this
change — only the diagnostic's inputs.
"""

import pytest

from hermes_cli.setup_platforms import _warn_missing_home_channels, _yaml_home_channels

_LEAKY = (
    "DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL",
    "SLACK_BOT_TOKEN", "SLACK_HOME_CHANNEL",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermetic_home = tmp_path / "home"
    hermetic_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermetic_home))
    for var in _LEAKY:
        monkeypatch.delenv(var, raising=False)
    (hermetic_home / ".env").write_text("", encoding="utf-8")
    return hermetic_home


def _run(capsys) -> str:
    _warn_missing_home_channels()
    return capsys.readouterr().out


def test_yaml_home_channel_suppresses_the_warning(home, monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "dummy-token")
    (home / "config.yaml").write_text("discord:\n  home_channel: '123456789'\n", encoding="utf-8")

    assert _yaml_home_channels()["discord"] == "123456789"
    assert "No home channel set for: Discord" not in _run(capsys)


def test_no_home_channel_anywhere_still_warns(home, monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "dummy-token")
    (home / "config.yaml").write_text("discord:\n  enabled: true\n", encoding="utf-8")

    out = _run(capsys)
    assert "No home channel set for: Discord" in out
    assert "hermes config set DISCORD_HOME_CHANNEL" in out


def test_env_home_channel_still_suppresses_the_warning(home, monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "dummy-token")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "987654321")
    (home / "config.yaml").write_text("discord:\n  enabled: true\n", encoding="utf-8")

    assert "No home channel set for: Discord" not in _run(capsys)


def test_gateway_platforms_block_and_dict_form_are_read(home, monkeypatch, capsys):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "dummy-token")
    (home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    discord:\n      home_channel:\n        chat_id: '5555'\n",
        encoding="utf-8")

    assert _yaml_home_channels()["discord"] == "5555"
    assert "No home channel set for: Discord" not in _run(capsys)


def test_unconfigured_platform_is_not_reported(home, capsys):
    (home / "config.yaml").write_text("discord:\n  enabled: true\n", encoding="utf-8")
    assert "Discord" not in _run(capsys)
