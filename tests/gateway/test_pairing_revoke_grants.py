"""P1 RED: a config.yaml allow_from grant survives ``hermes pairing revoke``.

Two rows, both written against the APIs that exist at the base sha so they fail
by ASSERTION, never by import:

1. ``test_yaml_grant_survives_plain_revoke_real_verdict`` — single-profile shape
   (the real YAML→env bridge seeds ``DISCORD_ALLOWED_USERS``), real
   ``PairingStore.revoke``, real ``GatewayAuthorizationMixin`` verdict. On base
   revoke rewrites ``.env`` with the surviving YAML entries (env beats YAML
   first-writer-wins) and the verdict turns False. Required behaviour: a plain
   revoke removes the PAIRING grant only, leaves the operator's ``.env``
   byte-identical, and the surviving YAML grant keeps authorizing so the caller
   can report it.
2. ``test_revoke_effective_flag_is_unknown`` — the ``--effective`` flag does not
   exist on base (argparse exits 2).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.session import SessionSource

PLATFORM = "discord"
USER = "1002"
KEPT = "1001"

_LEAKY_ENV = (
    "DISCORD_ALLOWED_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "DISCORD_ALLOWED_ROLES",
    "DISCORD_GROUP_ALLOWED_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS",
)


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    for var in _LEAKY_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A disposable HERMES_HOME — never a real profile."""
    launch = tmp_path / "home"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch


def _write_home(home, config_yaml: str, env_text: str = "") -> None:
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    (home / ".env").write_text(env_text, encoding="utf-8")


def _adapter_extra_from_loaded_config():
    """``config.extra`` exactly as the plugin/adapter receives it (real loader)."""
    cfg = load_gateway_config()
    platform_cfg = cfg.platforms.get(Platform.DISCORD)
    return dict(getattr(platform_cfg, "extra", {}) or {})


def _verdict(extra: dict, store, *, multiplex: bool = False, chat_type: str = "dm") -> bool:
    """Real ``GatewayAuthorizationMixin`` verdict through a real ``GatewayRunner``."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    adapter = SimpleNamespace(
        platform=Platform.DISCORD,
        config=SimpleNamespace(extra=extra),
        send=AsyncMock(),
        enforces_own_access_policy=False,
        _dm_policy="allowlist",
        _group_policy="allowlist",
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner.pairing_store = store
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1" if chat_type == "group" else "dm-1",
        chat_type=chat_type,
        user_id=USER,
        user_name="synthetic",
        is_bot=False,
    )
    return runner._is_user_authorized(source)


def test_yaml_grant_survives_plain_revoke_real_verdict(home):
    """Plain revoke must not materialise the bridged env mirror into ``.env``.

    Base behaviour (the defect): ``_sync_allowlist_remove`` writes
    ``DISCORD_ALLOWED_USERS=<remaining>`` into the profile ``.env`` for a value
    the YAML→env bridge put in process env. That mirror then outranks
    ``config.yaml`` (env beats YAML, first-writer-wins), so the operator's YAML
    grant is silently narrowed by a command documented never to edit it, and the
    verdict flips to False while the YAML still names the user.
    """
    from gateway.pairing import PairingStore

    _write_home(home, f'discord:\n  allow_from:\n    - "{KEPT}"\n    - "{USER}"\n')
    extra = _adapter_extra_from_loaded_config()
    assert "1002" in str(extra.get("allow_from")), "precondition: YAML grant reaches extra"

    store = PairingStore()
    store._approve_user(PLATFORM, USER, "")
    assert store.revoke(PLATFORM, USER) is True
    assert store.is_approved(PLATFORM, USER) is False, "pairing record is gone"

    env_after = (home / ".env").read_text(encoding="utf-8")
    assert "DISCORD_ALLOWED_USERS" not in env_after, (
        "plain revoke wrote a .env allowlist mirror for a value the YAML→env "
        f"bridge owns; .env is now {env_after!r}"
    )
    assert _verdict(extra, store) is True, (
        "the config.yaml grant must keep authorizing after a plain pairing revoke "
        "so the caller can report it"
    )


def test_revoke_effective_flag_is_unknown():
    """``hermes pairing revoke --effective`` must parse; on base argparse exits 2."""
    import argparse

    from hermes_cli.subcommands.pairing import build_pairing_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_pairing_parser(subparsers, cmd_pairing=lambda args: None)

    args = parser.parse_args(["pairing", "revoke", "--effective", PLATFORM, USER])
    assert args.effective is True
    assert (args.platform, args.user_id) == (PLATFORM, USER)
