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
import contextlib
import os
from pathlib import Path

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


@pytest.fixture(autouse=True)
def _reset_bridge_provenance(monkeypatch):
    """Bridge provenance is process-scoped state; never let it leak between rows."""
    import gateway.config_loader as config_loader
    from gateway.platforms import _shared

    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)
    _shared._BRIDGED_ENV_NAMES.clear()
    yield
    _shared._BRIDGED_ENV_NAMES.clear()


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


# ---------------------------------------------------------------------------
# P2 — enumeration + removal (gateway/access_grants.py), synthetic homes only
# ---------------------------------------------------------------------------


def _store():
    from gateway.pairing import PairingStore

    return PairingStore()


def _find(*, store=None, home=None, platform: str = PLATFORM, user: str = USER):
    from gateway.access_grants import find_user_grants

    return find_user_grants(platform, user, store=store, home=home)


def _remove(sources, *, store=None, home=None):
    from gateway.access_grants import remove_user_grants

    return remove_user_grants(sources, store=store, home=home)


def _locations(sources) -> list[str]:
    return [source.location for source in sources]


def _assert_report_agrees(sources, verdict: bool) -> None:
    """Plan §9.2.7: the offline report must agree with the real authz verdict.

    Any ``authorizes_now=True`` ⇒ the real verdict authorizes; no authorizing source and no
    ``None`` (cannot-evaluate) source ⇒ the real verdict denies. A shape where the report and
    authz disagree fails here — that is the anti-drift gate on this replica.
    """
    authorizing = [s for s in sources if s.authorizes_now is True]
    unknown = [s for s in sources if s.authorizes_now is None]
    if authorizing:
        assert verdict is True, (
            f"report says {authorizing[0].location} authorizes, but the real verdict denied "
            f"(sources={[(s.kind, s.location, s.authorizes_now) for s in sources]})"
        )
    elif not unknown:
        assert verdict is False, (
            "report found no authorizing grant, but the real verdict authorized "
            f"(sources={[(s.kind, s.location, s.authorizes_now) for s in sources]})"
        )


@contextlib.contextmanager
def _secondary_profile(home: Path, name: str = "work"):
    """A multiplexed secondary profile scope: the YAML→env bridge is skipped, no env allowlist."""
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope

    secondary = home / "profiles" / name
    secondary.mkdir(parents=True, exist_ok=True)
    set_multiplex_active(True)
    try:
        with _profile_runtime_scope(secondary, prepared_secret_scope={}):
            yield secondary
    finally:
        set_multiplex_active(False)


def test_multiplex_yaml_grant_is_reported_authorizing(home):
    """The defect precondition: multiplex secondary, no env allowlist → YAML authorizes."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from:\n    - "1001"\n    - "1002"\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        located = [s for s in grants if s.location == "config.yaml discord.allow_from"]
        assert [s.entry for s in located] == ["1002"]
        assert located[0].authorizes_now is True
        assert located[0].removable is True
        # Discord's allow_from is one sender list covering DMs and guild channels (upstream #103743).
        assert located[0].chat_scope == "global(dm+channel)"
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))


def test_effective_removes_multiplex_yaml_grant_and_denies(home):
    """``--effective`` edits config.yaml in place: comments and other entries survive."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            "# operator note\n"
            "discord:\n  enabled: true  # keep me\n  allow_from:\n    - \"1001\"\n    - \"1002\"\n",
            encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        removed = _remove(grants, store=store)

        assert "config.yaml discord.allow_from" in _locations(removed)
        text = (secondary / "config.yaml").read_text(encoding="utf-8")
        assert "1002" not in text
        assert "1001" in text
        assert "# operator note" in text and "# keep me" in text

        fresh_extra = _adapter_extra_from_loaded_config()
        _assert_report_agrees(_find(store=store), _verdict(fresh_extra, store, multiplex=True))
        assert _verdict(fresh_extra, store, multiplex=True) is False


def test_single_profile_yaml_grant_reports_the_bridged_env_source(home):
    """Single-profile: the bridge owns DISCORD_ALLOWED_USERS; report it as non-removable."""
    _write_home(home, 'discord:\n  allow_from:\n    - "1001"\n    - "1002"\n')
    extra = _adapter_extra_from_loaded_config()
    store = _store()
    grants = _find(store=store)

    bridged = [s for s in grants if s.kind == "env"]
    assert [s.location for s in bridged] == ["env DISCORD_ALLOWED_USERS (bridged from config.yaml)"]
    assert bridged[0].removable is False and bridged[0].authorizes_now is True
    # The YAML entry is declared, but env-shape precedence means it is not what authorizes.
    yaml_source = [s for s in grants if s.kind == "yaml"]
    assert yaml_source and yaml_source[0].authorizes_now is False
    _assert_report_agrees(grants, _verdict(extra, store))


def test_gateway_platforms_allow_from_is_enumerated_and_removable(home):
    """F2: ``gateway.platforms.<plat>.allow_from`` authorizes and must be reported/removed."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'gateway:\n  platforms:\n    discord:\n      allow_from:\n        - "1001"\n        - "1002"\n',
            encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        located = [s for s in grants if s.location == "config.yaml gateway.platforms.discord.allow_from"]
        assert [s.entry for s in located] == ["1002"]
        assert located[0].authorizes_now is True
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))

        removed = _remove(grants, store=store)
        assert "config.yaml gateway.platforms.discord.allow_from" in _locations(removed)
        text = (secondary / "config.yaml").read_text(encoding="utf-8")
        assert "1002" not in text and "1001" in text


def test_platforms_extra_allowed_users_is_enumerated_and_removable(home):
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'platforms:\n  discord:\n    extra:\n      allowed_users: ["1001", "1002"]\n',
            encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        located = [s for s in grants if s.location == "config.yaml platforms.discord.extra.allowed_users"]
        assert [s.entry for s in located] == ["1002"]
        assert located[0].authorizes_now is True
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))
        assert "config.yaml platforms.discord.extra.allowed_users" in _locations(_remove(grants, store=store))
        assert "1002" not in (secondary / "config.yaml").read_text(encoding="utf-8")


def test_env_only_grant_is_reported_and_removed(home, monkeypatch):
    """Control row: an authored ``.env`` allowlist is removable and denies afterwards."""
    _write_home(home, "", f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{KEPT},{USER}")  # the CLI loads .env into os.environ
    store = _store()
    grants = _find(store=store)
    env_sources = [s for s in grants if s.kind == "env"]
    assert [s.location for s in env_sources] == [".env DISCORD_ALLOWED_USERS"]
    assert env_sources[0].removable is True and env_sources[0].authorizes_now is True
    _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store))

    assert _locations(_remove(grants, store=store)) == [".env DISCORD_ALLOWED_USERS"]
    # Only the revoked principal is dropped: the operator's other entry survives.
    assert (home / ".env").read_text(encoding="utf-8").strip() == f"DISCORD_ALLOWED_USERS={KEPT}"
    assert os.environ["DISCORD_ALLOWED_USERS"] == KEPT
    assert _find(store=store) == []
    assert _verdict(_adapter_extra_from_loaded_config(), store) is False


def test_pairing_only_grant_is_the_only_source(home):
    _write_home(home, "")
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    grants = _find(store=store)
    assert [s.kind for s in grants] == ["pairing"]
    _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store))


def test_both_sources_plain_revoke_narrows_mirror_and_reports_yaml(home, monkeypatch):
    """Row 5: the pairing revoke narrows the env mirror; the YAML grant is still named."""
    _write_home(home, f'discord:\n  allow_from: ["{KEPT}", "{USER}"]\n',
                f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{KEPT},{USER}")
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    assert store.revoke(PLATFORM, USER) is True
    # Documented pairing behaviour, unchanged: the authored env mirror is narrowed.
    assert (home / ".env").read_text(encoding="utf-8").strip() == f"DISCORD_ALLOWED_USERS={KEPT}"

    after = _find(store=store)
    assert "config.yaml discord.allow_from" in _locations(after)
    _assert_report_agrees(after, _verdict(_adapter_extra_from_loaded_config(), store))
    # The YAML entry is not consulted while ANY env allowlist exists — reported, not hidden.
    assert [s.authorizes_now for s in after if s.kind == "yaml"] == [False]

    removed = _remove(after, store=store)
    assert "config.yaml discord.allow_from" in _locations(removed)
    assert USER not in (home / "config.yaml").read_text(encoding="utf-8")
    assert _verdict(_adapter_extra_from_loaded_config(), store) is False


def test_wildcard_is_reported_but_never_removed(home):
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text('discord:\n  enabled: true\n  allow_from: ["*"]\n',
                                               encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        wildcard = [s for s in grants if s.kind == "wildcard"]
        assert wildcard and wildcard[0].entry == "*"
        assert wildcard[0].removable is False and wildcard[0].authorizes_now is True
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))
        assert _remove(grants, store=store) == []
        assert "*" in (secondary / "config.yaml").read_text(encoding="utf-8")


def test_allow_all_from_yaml_is_declared_not_consulted_in_a_secondary(home):
    """F6: the YAML allow-all flag does not reach authz for a multiplexed secondary."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            "discord:\n  enabled: true\n  allow_all_users: true\n", encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        allow_all = [s for s in grants if s.kind == "allow_all"]
        assert allow_all and allow_all[0].removable is False
        assert allow_all[0].authorizes_now is False
        assert "bridged" in allow_all[0].reason
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))


def test_allow_all_from_yaml_authorizes_in_single_profile(home):
    _write_home(home, "discord:\n  allow_all_users: true\n")
    extra = _adapter_extra_from_loaded_config()  # the bridge seeds DISCORD_ALLOW_ALL_USERS
    store = _store()
    grants = _find(store=store)
    allow_all = [s for s in grants if s.kind == "allow_all"]
    assert allow_all and allow_all[0].authorizes_now is True
    _assert_report_agrees(grants, _verdict(extra, store))


def test_group_allow_from_is_reported_with_group_scope(home):
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  group_allow_from: ["1002"]\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        group = [s for s in grants if s.chat_scope == "group"]
        assert [s.location for s in group] == ["config.yaml discord.group_allow_from"]
        assert group[0].authorizes_now is True
        assert not [s for s in grants if s.kind == "yaml" and s.chat_scope.startswith("global")]
        _assert_report_agrees(
            grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True, chat_type="group"))


def test_mention_form_entry_is_matched_and_removable(home):
    """A ``<@!1002>`` entry names the principal (matched/removed) but authz compares verbatim."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "<@!1002>"]\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        located = [s for s in grants if s.kind == "yaml"]
        assert [s.entry for s in located] == ["<@!1002>"]
        assert located[0].removable is True
        assert located[0].authorizes_now is False and "verbatim" in located[0].reason
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))

        _remove(grants, store=store)
        text = (secondary / "config.yaml").read_text(encoding="utf-8")
        assert "<@!1002>" not in text and "1001" in text


def test_username_entry_cannot_be_evaluated_offline(home, monkeypatch):
    """F3: a username-shaped entry may resolve to this numeric id at connect time."""
    _write_home(home, "", f"DISCORD_ALLOWED_USERS={KEPT},some_user\n")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{KEPT},some_user")
    store = _store()
    grants = _find(store=store)
    unresolved = [s for s in grants if s.kind == "unresolved"]
    assert [s.entry for s in unresolved] == ["some_user"]
    assert unresolved[0].authorizes_now is None and unresolved[0].removable is False
    assert "cannot be evaluated offline" in unresolved[0].reason
    # The unresolved row is what forces exit 3; authz may still deny this exact id offline.
    _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store))


def test_global_allowlist_is_reported_and_never_removed(home, monkeypatch):
    """F17: GATEWAY_ALLOWED_USERS is cross-platform; --effective must not touch it."""
    _write_home(home, "", f"GATEWAY_ALLOWED_USERS={KEPT},{USER}\n")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", f"{KEPT},{USER}")
    store = _store()
    grants = _find(store=store)
    global_source = [s for s in grants if "GATEWAY_ALLOWED_USERS" in s.location]
    assert global_source and global_source[0].authorizes_now is True
    assert global_source[0].removable is False
    assert "cross-platform" in global_source[0].reason
    _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store))
    assert _remove(grants, store=store) == []
    assert "GATEWAY_ALLOWED_USERS" in (home / ".env").read_text(encoding="utf-8")


def test_profile_store_reports_its_own_home(home):
    """F9: the report must read the addressed profile's config.yaml, never the launch home's."""
    from gateway.pairing import PairingStore

    (home / "config.yaml").write_text(f'discord:\n  allow_from: ["{USER}"]\n', encoding="utf-8")
    work = home / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text('discord:\n  allow_from: ["2001"]\n', encoding="utf-8")

    # A profile-scoped store resolves ITS home (PairingStore.__init__'s root logic).
    assert _find(store=PairingStore(profile="work")) == []
    # The unscoped store stays on the launch home.
    assert [s.location for s in _find(store=PairingStore())] == ["config.yaml discord.allow_from"]


def test_unreadable_config_is_an_unresolved_source(home):
    _write_home(home, f'discord:\n  allow_from: ["{USER}"]\n')
    (home / "config.yaml").chmod(0o000)
    try:
        grants = _find(store=_store())
        unresolved = [s for s in grants if s.kind == "unresolved" and s.location == "config.yaml"]
        assert unresolved and unresolved[0].authorizes_now is None and unresolved[0].removable is False
    finally:
        (home / "config.yaml").chmod(0o600)


def test_effective_refuses_unreadable_config_and_leaves_env_byte_identical(home):
    """F11: refuse (exit 1) rather than half-apply — `.env` must not change."""
    from gateway.access_grants import AccessGrantError

    _write_home(home, f'discord:\n  allow_from: ["{USER}"]\n', f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    os.environ["DISCORD_ALLOWED_USERS"] = f"{KEPT},{USER}"
    (home / "config.yaml").chmod(0o000)
    before = (home / ".env").read_bytes()
    try:
        grants = _find(store=_store())  # only the env sources are visible while the file is unreadable
        with pytest.raises(AccessGrantError):
            _remove(grants, store=_store())
        assert (home / ".env").read_bytes() == before
    finally:
        (home / "config.yaml").chmod(0o600)
        os.environ.pop("DISCORD_ALLOWED_USERS", None)


def test_effective_refuses_a_managed_env_key(home, monkeypatch):
    """F4: a managed/pinned .env key must never be reported as removed."""
    import hermes_cli.config as config_mod
    from gateway.access_grants import AccessGrantError

    _write_home(home, "", f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    os.environ["DISCORD_ALLOWED_USERS"] = f"{KEPT},{USER}"
    monkeypatch.setattr(config_mod, "is_managed", lambda: True)
    before = (home / ".env").read_bytes()
    try:
        grants = _find(store=_store())
        with pytest.raises(AccessGrantError, match="managed"):
            _remove(grants, store=_store())
        assert (home / ".env").read_bytes() == before
    finally:
        os.environ.pop("DISCORD_ALLOWED_USERS", None)

