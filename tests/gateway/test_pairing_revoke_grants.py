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
import json
import os
from pathlib import Path

import pytest

try:  # Collection-time fastapi import: pydantic's plugin loader scans distribution entry points
    # of the running interpreter's environment, and doing that inside a test would trip
    # tests/home_io_guard.py for a reason unrelated to the code under test.
    import hermes_cli.web_server_files  # noqa: F401
except ImportError:  # pragma: no cover - fastapi/starlette absent
    pass

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.session import SessionSource

PLATFORM = "discord"
USER = "1002"
KEPT = "1001"
# Group SENDER lists are consulted for group/forum traffic only (authz reads them under
# ``is_group_or_forum``); group CHAT allowlists admit every member, channel traffic included.
GROUP = "group+forum"
CHAT = "group/channel"

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
    """Bridge provenance is process-scoped state; never let it leak between rows.

    ``getattr`` keeps this file importable (and the two RED rows failing by ASSERTION) against a
    tree that predates ``_BRIDGED_ENV_NAMES`` — F9: a base-archive run must reproduce the defect
    instead of erroring in the fixture.
    """
    import gateway.config_loader as config_loader
    from gateway.platforms import _shared

    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False, raising=False)
    recorded = getattr(_shared, "_BRIDGED_ENV_NAMES", None)
    if recorded is not None:
        recorded.clear()
    yield
    if recorded is not None:
        recorded.clear()


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
    """``hermes pairing revoke --effective`` must parse; on base argparse exits 2.

    Wrapped so a bare archive fails by ASSERTION (naming the defect) instead of an escaping
    ``SystemExit`` — F9: the RED row must reproduce the gap, not error out of the test body.
    """
    import argparse

    from hermes_cli.subcommands.pairing import build_pairing_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_pairing_parser(subparsers, cmd_pairing=lambda args: None)

    try:
        args = parser.parse_args(["pairing", "revoke", "--effective", PLATFORM, USER])
    except SystemExit as exc:  # base: `--effective` is an unknown argument (exit 2)
        raise AssertionError(
            f"`pairing revoke --effective` must parse; argparse exited {exc.code}") from None
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


def _assert_report_agrees_per_context(sources, *, dm: bool, group: bool) -> None:
    """F4: every authorizing row must be backed by the verdict for the context its scope claims.

    A ``group``-scoped row is judged by group traffic, so a group list can never pass this gate on a
    DM grant — and no row may be reported authorizing in a context the real verdict denies.
    """
    authorizing = [s for s in sources if s.authorizes_now is True]
    for source in authorizing:
        if source.chat_scope.startswith("group"):
            assert group is True, f"{source.location} claims a group grant; group verdict denied"
        else:
            assert dm is True, f"{source.location} claims a dm/platform grant; dm verdict denied"
    if not authorizing and not [s for s in sources if s.authorizes_now is None]:
        assert dm is False and group is False, (
            "report found no authorizing and no unknown source, but a context verdict authorized")


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
    """Single-profile: the bridge owns DISCORD_ALLOWED_USERS; report it as non-removable.

    F1: the YAML entry is the AUTHORED grant behind that bridged value, so it authorizes now —
    reporting it ``False`` (as the pre-fix replica did when any process env allowlist existed) is
    what let the CLI drop it and exit 0.
    """
    _write_home(home, 'discord:\n  allow_from:\n    - "1001"\n    - "1002"\n')
    extra = _adapter_extra_from_loaded_config()
    store = _store()
    grants = _find(store=store)

    bridged = [s for s in grants if s.kind == "env"]
    assert [s.location for s in bridged] == ["env DISCORD_ALLOWED_USERS (bridged from config.yaml)"]
    assert bridged[0].removable is False and bridged[0].authorizes_now is True
    # The YAML row is the authored source of that bridged value: it authorizes, and it is removable.
    yaml_source = [s for s in grants if s.kind == "yaml"]
    assert yaml_source and yaml_source[0].authorizes_now is True
    assert yaml_source[0].removable is True
    assert "no authored env allowlist" in yaml_source[0].reason
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
    # With an AUTHORED env allowlist present the YAML row cannot be decided offline (F1: None, not
    # False) — reported and never silently dropped, so the exit code stays 3 until it is removed.
    yaml_rows = [s for s in after if s.kind == "yaml"]
    assert [s.authorizes_now for s in yaml_rows] == [None]
    assert "authored env allowlist" in yaml_rows[0].reason

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
    """F6/F1: the YAML allow-all flag does not reach authz for a multiplexed secondary.

    It is never reported ``False`` (F1): with the env flag unset in this process the row is
    ``None`` — "a gateway bridges this to DISCORD_ALLOW_ALL_USERS; inert only for a multiplexed
    secondary" — so the exit code is 3 rather than a silent 0.
    """
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            "discord:\n  enabled: true\n  allow_all_users: true\n", encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        allow_all = [s for s in grants if s.kind == "allow_all"]
        assert allow_all and allow_all[0].removable is False
        assert allow_all[0].authorizes_now is None
        assert "bridged" in allow_all[0].reason and "multiplexed secondary" in allow_all[0].reason
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))


@pytest.mark.parametrize('config', ['allow_all_users: "yes"\n', 'gateway:\n  allow_all_users: 1\n'])
def test_global_allow_all_secondary_is_conservatively_reported(home, config):
    with _secondary_profile(home) as secondary:
        (secondary / 'config.yaml').write_text(config)
        store = _store()
        rows = [row for row in _find(store=store) if row.kind == 'allow_all']
        assert len(rows) == 1 and rows[0].authorizes_now is None
        assert not rows[0].removable
        assert _revoke_cli(store, effective=True) == 3
        assert (secondary / 'config.yaml').read_text() == config


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
        group = [s for s in grants if s.chat_scope == GROUP]
        assert [s.location for s in group] == ["config.yaml discord.group_allow_from"]
        assert group[0].authorizes_now is True
        assert not [s for s in grants if s.kind == "yaml" and s.chat_scope.startswith("global")]
        _assert_report_agrees(
            grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True, chat_type="group"))


def test_mention_form_entry_is_matched_and_removable(home):
    """A ``<@!1002>`` entry names the principal (matched/removed) but authz compares verbatim.

    ``authorizes_now`` is ``None``, not ``False`` (F1: a YAML row is never reported inert): the
    Discord adapter resolves username-shaped entries at connect time, so an offline process cannot
    rule out that this spelling resolves to the principal.
    """
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "<@!1002>"]\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        located = [s for s in grants if s.kind == "yaml"]
        assert [s.entry for s in located] == ["<@!1002>"]
        assert located[0].removable is True
        assert located[0].authorizes_now is None and "verbatim" in located[0].reason
        _assert_report_agrees(grants, _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True))

        _remove(grants, store=store)
        text = (secondary / "config.yaml").read_text(encoding="utf-8")
        assert "<@!1002>" not in text and "1001" in text


def test_one_list_with_two_spellings_of_the_same_principal_removes_cleanly(home):
    """``1002`` and ``<@1002>`` are two sources but ONE list: staging per path, not per source.

    Regression: staging each source in turn made the second pass see the first pass's write and
    refuse with "no longer names" — ``--effective`` could not remove a list that spelled the same
    principal twice, which is exactly the list an operator ends up with after copying a mention.
    """
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002", "<@1002>"]\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        assert sorted(s.entry for s in grants if s.kind == "yaml") == ["1002", "<@1002>"]

        removed = _remove(grants, store=store)
        assert {s.entry for s in removed} == {"1002", "<@1002>"}
        text = (secondary / "config.yaml").read_text(encoding="utf-8")
        assert "1002" not in text and "1001" in text


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
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{KEPT},{USER}")
    monkeypatch.setattr(config_mod, "is_managed", lambda: True)
    before = (home / ".env").read_bytes()
    try:
        grants = _find(store=_store())
        with pytest.raises(AccessGrantError, match="managed"):
            _remove(grants, store=_store())
        assert (home / ".env").read_bytes() == before
    finally:
        os.environ.pop("DISCORD_ALLOWED_USERS", None)


# ---------------------------------------------------------------------------
# P3 — CLI surface, live-adapter purge, dashboard endpoint
# ---------------------------------------------------------------------------


def _revoke_cli(store, *, effective: bool = False, platform: str = PLATFORM, user: str = USER) -> int:
    """Run the real ``_cmd_revoke`` and return its exit code."""
    from hermes_cli.pairing import _cmd_revoke

    with pytest.raises(SystemExit) as exc:
        _cmd_revoke(store, platform, user, effective=effective)
    return int(exc.value.code or 0)


def test_revoke_plain_yaml_only_exits_3_and_names_the_source(home, capsys):
    """Single-profile-shaped secondary: YAML-only grant, no env allowlist → success is a lie, exit 3."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002"]\n', encoding="utf-8")
        store = _store()
        store._approve_user(PLATFORM, USER, "")

        assert _revoke_cli(store) == 3
        out = capsys.readouterr().out
        assert "Revoked pairing access" in out
        assert "config.yaml discord.allow_from" in out
        assert "authorizes now" in out
        assert "--effective" in out
        assert store.is_approved(PLATFORM, USER) is False
        assert USER in (secondary / "config.yaml").read_text(encoding="utf-8")


def test_revoke_effective_exits_0_and_denies(home, capsys):
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002"]\n', encoding="utf-8")
        store = _store()
        store._approve_user(PLATFORM, USER, "")

        assert _revoke_cli(store, effective=True) == 0
        out = capsys.readouterr().out
        assert "Removed config.yaml discord.allow_from" in out
        assert "No other configured source names" in out
        assert USER not in (secondary / "config.yaml").read_text(encoding="utf-8")
        assert _verdict(_adapter_extra_from_loaded_config(), store, multiplex=True) is False


def test_revoke_plain_env_only_exits_0(home, monkeypatch, capsys):
    """Control row: an authored .env allowlist entry is what plain revoke removes."""
    _write_home(home, "", f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", f"{KEPT},{USER}")
    store = _store()
    store._approve_user(PLATFORM, USER, "")

    assert _revoke_cli(store) == 0
    out = capsys.readouterr().out
    assert "No other configured source names" in out
    assert (home / ".env").read_text(encoding="utf-8").strip() == f"DISCORD_ALLOWED_USERS={KEPT}"
    assert _verdict(_adapter_extra_from_loaded_config(), store) is False


def test_revoke_plain_single_profile_yaml_grant_leaves_env_untouched(home, capsys):
    """F12: the single-profile shape is report-only too, and ``.env`` stays byte-identical."""
    _write_home(home, 'discord:\n  allow_from: ["1001", "1002"]\n')
    extra = _adapter_extra_from_loaded_config()  # the bridge seeds the env mirror in-process
    store = _store()
    store._approve_user(PLATFORM, USER, "")

    assert _revoke_cli(store) == 3
    out = capsys.readouterr().out
    assert "env DISCORD_ALLOWED_USERS (bridged from config.yaml)" in out
    assert (home / ".env").read_text(encoding="utf-8") == ""
    assert _verdict(extra, store) is True


def test_revoke_effective_refusal_keeps_pairing_record_and_env(home, capsys):
    """F11: a refusal must not have revoked the pairing record or touched `.env` either."""
    _write_home(home, f'discord:\n  allow_from: ["{USER}"]\n', f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n")
    os.environ["DISCORD_ALLOWED_USERS"] = f"{KEPT},{USER}"
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    before_env = (home / ".env").read_bytes()
    (home / "config.yaml").chmod(0o000)
    try:
        assert _revoke_cli(store, effective=True) == 1
        out = capsys.readouterr().out
        assert "Refused" in out
        assert store.is_approved(PLATFORM, USER) is True
        assert (home / ".env").read_bytes() == before_env
    finally:
        (home / "config.yaml").chmod(0o600)
        os.environ.pop("DISCORD_ALLOWED_USERS", None)


def _live_adapter(extra: dict):
    return SimpleNamespace(
        platform=Platform.DISCORD, config=SimpleNamespace(extra=extra),
        send=AsyncMock(), _dm_allow_from=list(extra.get("allow_from") or []),
    )


def test_effective_purges_live_adapter_snapshot_but_plain_revoke_does_not(home, monkeypatch, capsys):
    """F1: only a removal that changed an authorizing source may purge the in-process snapshot."""
    import gateway.pairing as pairing_mod

    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002"]\n', encoding="utf-8")
        store = _store()
        store._approve_user(PLATFORM, USER, "")

        # Plain revoke of a YAML-only grant: nothing removed, nothing to purge.
        live = _live_adapter({"allow_from": [KEPT, USER]})
        monkeypatch.setattr(pairing_mod, "_iter_live_gateway_adapters", lambda: iter([live]))
        assert _revoke_cli(store) == 3
        assert live.config.extra["allow_from"] == [KEPT, USER]

        # --effective removed the YAML entry, so the in-process snapshot must not keep it.
        assert _revoke_cli(store, effective=True) == 0
        assert live.config.extra["allow_from"] == [KEPT]
        capsys.readouterr()


def _router_client():
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - fastapi/starlette absent
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


def test_dashboard_revoke_reports_a_surviving_yaml_grant(home):
    """F10: the endpoint must not answer a bare ``ok`` (or 404) while config.yaml still grants."""
    _write_home(home, 'discord:\n  allow_from: ["1001", "1002"]\n')
    store = _store()
    store._approve_user(PLATFORM, USER, "")

    response = _router_client().post(
        "/api/pairing/revoke", json={"platform": PLATFORM, "user_id": USER})
    assert response.status_code == 200
    body = response.json()
    assert body["revoked"] is True
    assert body["removed"] and body["removed"][0]["kind"] == "pairing"
    assert [row["location"] for row in body["still_authorized_by"]] == ["config.yaml discord.allow_from"]
    assert body["still_authorized_by"][0]["authorizes_now"] is True


def test_dashboard_revoke_without_a_pairing_record_still_reports_grants(home):
    """A 404 here would tell an operator "nothing to revoke" while the YAML grant stands."""
    _write_home(home, 'discord:\n  allow_from: ["1002"]\n')

    response = _router_client().post(
        "/api/pairing/revoke", json={"platform": PLATFORM, "user_id": USER})
    assert response.status_code == 200
    body = response.json()
    assert body["revoked"] is False and body["removed"] == []
    assert [row["location"] for row in body["still_authorized_by"]] == ["config.yaml discord.allow_from"]


def test_dashboard_revoke_requires_platform_and_user(home):
    _write_home(home, "")
    client = _router_client()
    # A missing field is rejected by the request model (422); empty values reach the handler (400).
    assert client.post("/api/pairing/revoke", json={"platform": PLATFORM}).status_code == 422
    assert client.post(
        "/api/pairing/revoke", json={"platform": "", "user_id": ""}).status_code == 400


# ---------------------------------------------------------------------------
# F1-F4 regression rows: the shape a GATEWAY loads, not this process's env
# ---------------------------------------------------------------------------


def test_revoke_cli_yaml_allow_all_is_never_a_silent_zero(home, capsys):
    """F1a: ``config.yaml discord.allow_all_users: true`` + empty ``.env`` → exit 3, source named.

    CLI-shaped on purpose: nothing calls ``load_gateway_config()`` first, so the YAML→env bridge
    never runs and ``DISCORD_ALLOW_ALL_USERS`` is unset in this process. The gateway DOES bridge it,
    so the flag authorizes there; the offline replica must report it as unknown, never as inert.
    """
    _write_home(home, "discord:\n  allow_all_users: true\n")
    assert os.environ.get("DISCORD_ALLOW_ALL_USERS") is None, "precondition: no bridged flag here"
    store = _store()

    assert _revoke_cli(store) == 3
    out = capsys.readouterr().out
    assert "config.yaml discord.allow_all_users" in out
    assert "cannot be evaluated offline" in out
    assert "No other configured source names" not in out

    # A flag cannot be removed: --effective must say so instead of claiming a removal it did not make.
    assert _revoke_cli(store, effective=True) == 3
    out = capsys.readouterr().out
    assert "Removed" not in out
    assert "config.yaml discord.allow_all_users" in out
    assert "No other configured source names" not in out
    assert (home / "config.yaml").read_text(encoding="utf-8") == "discord:\n  allow_all_users: true\n"


def test_revoke_cli_yaml_allow_from_survives_a_group_env_list(home, monkeypatch, capsys):
    """F1b: ``telegram.allow_from`` + ``TELEGRAM_GROUP_ALLOWED_USERS`` → exit 3, source named.

    The bridge copies ``allow_from`` into ``TELEGRAM_ALLOWED_USERS``, which the union consults — so a
    group-scoped env list must NOT make the YAML row disappear. CLI-shaped: the profile ``.env`` is
    in the process env (as the CLI loads it) and no config load has happened.
    """
    _write_home(home, 'telegram:\n  allow_from:\n    - "1002"\n',
                "TELEGRAM_GROUP_ALLOWED_USERS=5555\n")
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "5555")
    store = _store()

    assert _revoke_cli(store, platform="telegram") == 3
    out = capsys.readouterr().out
    assert "config.yaml telegram.allow_from" in out and "authorizes now" in out
    assert "No other configured source names" not in out
    assert USER in (home / "config.yaml").read_text(encoding="utf-8")

    # --effective DOES remove this one, and only then may it report a plain success.
    assert _revoke_cli(store, platform="telegram", effective=True) == 0
    out = capsys.readouterr().out
    assert "Removed config.yaml telegram.allow_from" in out
    assert "No other configured source names" in out
    assert USER not in (home / "config.yaml").read_text(encoding="utf-8")


_BUZZ_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"


def test_plugin_platform_env_name_and_alias_match_without_a_live_registry(home, monkeypatch):
    """F2: the allowlist env name comes from the registry entry, and matching uses authz's predicate.

    Buzz stores npub or hex; inbound pubkeys are always hex (``_normalize_nostr_allow_entries``). A
    second matcher implementation missed the npub entry, produced no row, and the command exited 0
    while the principal was authorized.
    """
    from gateway.authz_mixin import _npub_to_hex
    from gateway.platform_registry import platform_registry

    hex_id = _npub_to_hex(_BUZZ_NPUB)
    assert hex_id and len(hex_id) == 64, "precondition: the fixture npub decodes"
    monkeypatch.setattr(
        platform_registry, "get",
        lambda name: SimpleNamespace(allowed_users_env="BUZZ_ALLOWED_USERS"))
    _write_home(home, "", f"BUZZ_ALLOWED_USERS={_BUZZ_NPUB}\n")
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", _BUZZ_NPUB)

    grants = _find(platform="buzz", user=hex_id)
    env = [s for s in grants if s.kind == "env"]
    assert [s.location for s in env] == [".env BUZZ_ALLOWED_USERS"]
    assert env[0].entry == _BUZZ_NPUB  # labelled by its authored spelling
    assert env[0].authorizes_now is True and env[0].removable is True


def test_unresolvable_platform_allowlist_env_fails_closed(home, monkeypatch, capsys):
    """F2: no registry entry and no known env name → an ``unresolved`` row (exit 3), never nothing."""
    monkeypatch.delenv("BUZZ_ALLOWED_USERS", raising=False)
    _write_home(home, "")

    grants = _find(platform="buzz", user="deadbeef")
    unresolved = [s for s in grants if s.kind == "unresolved"]
    assert unresolved and unresolved[0].authorizes_now is None
    assert unresolved[0].removable is False
    assert "not resolvable" in unresolved[0].reason

    assert _revoke_cli(_store(), platform="buzz", user="deadbeef") == 3
    assert "No other configured source names" not in capsys.readouterr().out


def test_env_entry_declared_but_not_consulted_is_unknown_not_inert(home):
    """F3: a ``.env`` grant a live profile scope does not consult reads as unknown (restart re-reads)."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text("discord:\n  enabled: true\n", encoding="utf-8")
        (secondary / ".env").write_text(f"DISCORD_ALLOWED_USERS={KEPT},{USER}\n", encoding="utf-8")
        store = _store()

        grants = _find(store=store)
        declared = [s for s in grants if s.kind == "env"]
        assert [s.location for s in declared] == [".env DISCORD_ALLOWED_USERS"]
        assert declared[0].authorizes_now is None
        assert "a restart re-reads it" in declared[0].reason
        assert declared[0].removable is True

        # The exit code follows the row: 3, not 0 while the file names the user.
        assert _revoke_cli(store) == 3


def test_group_allow_from_never_claims_a_dm_grant(home):
    """F4: a group list is reported for GROUP traffic; the DM verdict denies the same principal."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  group_allow_from: ["1002"]\n', encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        extra = _adapter_extra_from_loaded_config()
        dm = _verdict(extra, store, multiplex=True, chat_type="dm")
        group = _verdict(extra, store, multiplex=True, chat_type="group")
        assert dm is False and group is True, "precondition: the two contexts disagree"

        authorizing = [s for s in grants if s.authorizes_now is True]
        assert [s.chat_scope for s in authorizing] == [GROUP]
        assert not [s for s in grants if s.kind == "yaml" and s.chat_scope != GROUP]
        _assert_report_agrees_per_context(grants, dm=dm, group=group)


def test_platform_list_and_group_list_are_judged_in_their_own_context(home):
    """F4: ``allow_from`` covers the platform's DM scope; ``group_allow_from`` covers group traffic."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1002"]\n  group_allow_from: ["1001"]\n',
            encoding="utf-8")
        store = _store()
        grants = _find(store=store)
        extra = _adapter_extra_from_loaded_config()
        dm = _verdict(extra, store, multiplex=True, chat_type="dm")
        group = _verdict(extra, store, multiplex=True, chat_type="group")

        platform_rows = [s for s in grants if s.kind == "yaml" and s.chat_scope != GROUP]
        assert [s.entry for s in platform_rows] == [USER]
        assert platform_rows[0].authorizes_now is True
        _assert_report_agrees_per_context(grants, dm=dm, group=group)


# ---------------------------------------------------------------------------
# Round 3 — structural coverage: the recursive name scan, chat grants, per-group
# sender lists. These are the shapes that used to exit 0 while the gateway said "yes".
# ---------------------------------------------------------------------------

_WECOM_CHAT = "wrABC123"
_WECOM_GROUP_CONFIG = (
    "platforms:\n"
    "  wecom:\n"
    "    extra:\n"
    "      # operator note\n"
    "      groups:\n"
    f"        {_WECOM_CHAT}:\n"
    f"          allow_from: [\"{KEPT}\", \"{USER}\"]\n"
    "      other_setting: true\n"
)


def test_wecom_per_group_sender_list_is_reported_and_removable(home, capsys):
    """F1 (round 3): ``extra.groups.<chat>.allow_from`` names the user — report it, remove it.

    Before this round the key was not in any modelled list, so ``revoke`` exited 0 with
    "No other configured source names user" while ``_adapter_group_has_sender_allowlist`` handed the
    gateway a True verdict for that chat, and ``--effective`` reported a success having written
    nothing.
    """
    _write_home(home, _WECOM_GROUP_CONFIG)
    store = _store()
    grants = _find(store=store, platform="wecom")
    located = [s for s in grants
               if s.location == f"config.yaml platforms.wecom.extra.groups.[redacted].allow_from"]
    assert [s.entry for s in located] == [USER]
    assert located[0].kind == "yaml"
    assert located[0].chat_scope == GROUP  # a sender list: group/forum, never channel
    assert located[0].removable is True and located[0].authorizes_now is True

    assert _revoke_cli(store, platform="wecom") == 3
    out = capsys.readouterr().out
    assert f"platforms.wecom.extra.groups.[redacted].allow_from" in out
    assert "No other configured source names" not in out
    assert USER in (home / "config.yaml").read_text(encoding="utf-8")

    # --effective drops only this principal from that chat's list.
    assert _revoke_cli(store, platform="wecom", effective=True) == 0
    out = capsys.readouterr().out
    assert f"Removed config.yaml platforms.wecom.extra.groups.[redacted].allow_from" in out
    assert "No other configured source names" in out
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert USER not in text and KEPT in text
    assert "# operator note" in text and "other_setting: true" in text, (
        "comments and unrelated keys survive the rewrite")


def test_wecom_sole_entry_removal_leaves_a_chat_grant_not_a_false_success(home, capsys):
    """Removing the only entry empties the list, and an empty list admits EVERY sender in the chat.

    WeCom's ``_is_group_allowed`` returns True when ``allow_from`` holds nothing, so ``--effective``
    must not print "No other configured source names user" after leaving the chat wide open.
    """
    _write_home(home, "platforms:\n  wecom:\n    extra:\n      groups:\n"
                      f"        wrABC123:\n          allow_from: [\"{USER}\"]\n")
    store = _store()
    assert _revoke_cli(store, platform="wecom") == 3

    assert _revoke_cli(store, platform="wecom", effective=True) == 3
    out = capsys.readouterr().out
    assert "Removed config.yaml platforms.wecom.extra.groups.[redacted].allow_from" in out
    assert "No other configured source names" not in out
    assert "no sender entries left" in out and "every sender in that chat is admitted" in out
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert USER not in text


def test_wecom_group_allow_from_is_a_chat_grant_not_a_sender_list(home, capsys):
    """F3 (round 3): on an own-policy host ``group_allow_from`` is a CHAT list — non-removable.

    ``_own_policy_authorizes`` matches that list against the chat id, so it admits every member of
    the chat; reporting it as a sender list (the pre-round model) let a numeric chat id drop out of
    the report entirely.
    """
    _write_home(home, "platforms:\n  wecom:\n    extra:\n      group_allow_from: [\"1234567890\"]\n")
    store = _store()
    grants = _find(store=store, platform="wecom")
    chat_rows = [s for s in grants if s.kind == "chat_grant"]
    assert [s.location for s in chat_rows] == ["config.yaml platforms.wecom.extra.group_allow_from"]
    assert [s.entry for s in chat_rows] == ["1234567890"]
    assert chat_rows[0].chat_scope == CHAT
    assert chat_rows[0].removable is False and chat_rows[0].authorizes_now is None
    assert "1234567890" in chat_rows[0].reason

    assert _revoke_cli(store, platform="wecom") == 3
    assert "No other configured source names" not in capsys.readouterr().out
    # A chat grant is not removable: --effective must not claim a removal it did not make.
    assert _revoke_cli(store, platform="wecom", effective=True) == 3
    out = capsys.readouterr().out
    assert "Removed" not in out
    assert "1234567890" in (home / "config.yaml").read_text(encoding="utf-8")


def test_telegram_group_allowed_chats_is_a_chat_grant(home, capsys):
    """F2 (round 3): ``TELEGRAM_GROUP_ALLOWED_CHATS`` admits every sender in the chat → rc 3."""
    _write_home(home, "", "TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890\n")
    store = _store()
    grants = _find(store=store, platform="telegram")
    chat_rows = [s for s in grants if s.kind == "chat_grant"]
    assert [s.location for s in chat_rows] == [".env TELEGRAM_GROUP_ALLOWED_CHATS"]
    assert chat_rows[0].entry == "-1001234567890" and chat_rows[0].removable is False
    assert chat_rows[0].authorizes_now is None and chat_rows[0].chat_scope == CHAT
    assert "-1001234567890" in chat_rows[0].reason

    assert _revoke_cli(store, platform="telegram") == 3
    out = capsys.readouterr().out
    assert "Still named by .env TELEGRAM_GROUP_ALLOWED_CHATS" in out
    assert "No other configured source names" not in out

    assert _revoke_cli(store, platform="telegram", effective=True) == 3
    assert "TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890" in (home / ".env").read_text(encoding="utf-8")


def test_yaml_group_allowed_chats_and_allowed_chats_are_chat_grants(home):
    """Both YAML chat allowlists are reported, in every block the loader merges."""
    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            "platforms:\n  telegram:\n    extra:\n      group_allowed_chats: [-100555]\n"
            "      allowed_chats: [-100999]\n",
            encoding="utf-8")
        grants = _find(store=_store(), platform="telegram")
        chat_rows = [s for s in grants if s.kind == "chat_grant"]
        assert [s.location for s in chat_rows] == [
            "config.yaml platforms.telegram.extra.group_allowed_chats",
            "config.yaml platforms.telegram.extra.allowed_chats",
        ]
        assert [s.entry for s in chat_rows] == ["-100555", "-100999"]
        assert all(s.removable is False and s.authorizes_now is None for s in chat_rows)


def test_unmodelled_config_key_naming_the_user_is_a_name_scan_hit(home, capsys):
    """The structural net: a plugin key nobody modelled still cannot exit 0."""
    _write_home(home,
                "platforms:\n  discord:\n    extra:\n      owner_ids:\n        - \"1001\"\n"
                "        - \"1002\"\n")
    store = _store()
    grants = _find(store=store)
    scan = [s for s in grants if s.kind == "unresolved"]
    # ``entry`` is blanked on net rows: the same walk sees credentials that contain the principal.
    assert [(s.location, s.entry) for s in scan] == [
        ("config.yaml [unmodeled path redacted]", "")]
    assert scan[0].authorizes_now is None and scan[0].removable is False
    assert "does not model what reads it" in scan[0].reason

    assert _revoke_cli(store) == 3
    out = capsys.readouterr().out
    assert "config.yaml [unmodeled path redacted]" in out
    assert "No other configured source names" not in out
    # Nothing removable: --effective must not invent a success either.
    assert _revoke_cli(store, effective=True) == 3
    assert "Removed" not in capsys.readouterr().out


def test_unmodelled_env_var_naming_the_user_is_a_name_scan_hit(home, capsys):
    """The net covers the profile ``.env`` too, not only ``config.yaml``."""
    _write_home(home, "", "SOME_PLUGIN_OPERATORS=1001,1002\n")
    grants = _find(store=_store())
    scan = [s for s in grants if s.kind == "unresolved"]
    assert [(s.location, s.entry) for s in scan] == [(".env SOME_PLUGIN_OPERATORS", "")]
    assert _revoke_cli(_store()) == 3


def test_name_scan_ignores_a_value_naming_a_different_principal(home, capsys):
    """The control: no unresolved row, no exit-3 noise, when the value names somebody else."""
    _write_home(home,
                "platforms:\n  discord:\n    extra:\n      owner_ids: [\"1001\"]\n",
                "SOME_PLUGIN_OPERATORS=1001\n")
    store = _store()
    assert _find(store=store) == []

    assert _revoke_cli(store) == 0
    out = capsys.readouterr().out
    assert "No other configured source names" in out and "cannot be evaluated" not in out
    assert _revoke_cli(store, effective=True) == 0


def test_per_group_camelcase_spelling_is_modelled_and_removable(home):
    """``groups.<chat>.allowFrom`` is the other documented spelling (WeCom reads both)."""
    _write_home(home,
                "platforms:\n  wecom:\n    extra:\n      groups:\n        g-2:\n"
                f"          allowFrom: [\"{KEPT}\", \"{USER}\"]\n")
    store = _store()
    grants = _find(store=store, platform="wecom")
    located = [s for s in grants
               if s.location == "config.yaml platforms.wecom.extra.groups.[redacted].allowFrom"]
    assert [s.entry for s in located] == [USER]
    assert located[0].removable is True and located[0].chat_scope == GROUP

    assert _remove(grants, store=store)
    text = (home / "config.yaml").read_text(encoding="utf-8")
    assert USER not in text and KEPT in text


def test_name_scan_names_the_owner_of_a_foreign_env_var(home):
    """A value in another platform's env var is still reported, and the row says which command."""
    _write_home(home, "", f"DISCORD_ALLOWED_USERS={USER}\n")
    grants = _find(store=_store(), platform="telegram")
    scan = [s for s in grants if s.kind == "unresolved"]
    assert [s.location for s in scan] == [".env DISCORD_ALLOWED_USERS"]
    assert "discord's env var" in scan[0].reason
    assert "pairing revoke discord" in scan[0].reason


def test_name_scan_is_quiet_on_a_home_that_never_names_the_user(home, capsys):
    """No noise control: a realistic home with paths, tokens and other users stays exit 0."""
    _write_home(
        home,
        "discord:\n"
        "  enabled: true\n"
        "  token: \"abc\"\n"
        "  extra:\n"
        "    require_mention: true\n"
        "    allow_from: [\"1001\", \"1003\"]\n"
        "  home_channel: {platform: discord, chat_id: \"555\"}\n",
        "DISCORD_BOT_TOKEN=abc\nHERMES_HOME=/tmp/example\nTZ=UTC\n")
    store = _store()
    assert _find(store=store) == []
    assert _revoke_cli(store) == 0
    out = capsys.readouterr().out
    assert "No other configured source names" in out and "cannot be evaluated" not in out


def test_non_numeric_principal_gets_the_same_unresolved_net(home):
    """Nit (round 3): ``_unresolved`` is shape-based, not ``isdigit()``-gated."""
    _write_home(home, "", "SIGNAL_ALLOWED_USERS=+15550001111,+15550009999,some_user\n")
    grants = _find(store=_store(), platform="signal", user="+15550001111")
    unresolved = [s for s in grants
                  if s.kind == "unresolved" and s.location == ".env SIGNAL_ALLOWED_USERS"]
    assert [s.entry for s in unresolved] == ["some_user"]
    assert unresolved[0].authorizes_now is None
    # A sibling phone number is decidable: same shape, different principal — no row, no noise.
    assert not [s for s in grants if s.entry == "+15550009999"]
    own = [s for s in grants if s.entry == "+15550001111"]
    assert own and own[0].kind == "env", "the principal's own entry is reported as an env grant"
    assert own[0].authorizes_now is None, "declared in .env but not consulted here: a restart re-reads it"


# ---------------------------------------------------------------------------
# P4 — the net is structural: every authored spelling names the user (review round 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_yaml, expected_path", [
    ('plugin:\n  ids: [1002]\n', "config.yaml [unmodeled path redacted]"),                     # a list of ints
    ('plugin:\n  ids: "1002,1003"\n', "config.yaml [unmodeled path redacted]"),                # one CSV string
    ('plugin:\n  users:\n    "1002": true\n', "config.yaml [unmodeled path redacted]"),      # a mapping KEY
    ('plugin:\n  ids: "*,1003"\n', "config.yaml [unmodeled path redacted]"),                   # wildcard in CSV
])
def test_the_net_catches_non_string_and_composite_spellings(home, config_yaml, expected_path):
    """The net must not depend on a value being a bare string.

    Review round 4: ``ids: [1002]`` (int), ``ids: "1002,1003"`` (CSV), ``users: {"1002": true}``
    (key) and ``"*,1003"`` all exited 0 with "No other configured source names user 1002" — the walk
    skipped non-strings and compared the raw CSV instead of its entries.
    """
    _write_home(home, config_yaml)
    grants = _find(store=_store())
    rows = [s for s in grants if s.kind == "unresolved" and s.location == expected_path]
    assert rows, f"{expected_path} must be reported; got {[s.location for s in grants]}"
    assert rows[0].authorizes_now is None and rows[0].removable is False
    assert _revoke_cli(_store()) == 3


def test_whatsapp_jid_entry_aliases_a_bare_phone_principal(home):
    """A JID is a phone number in a suffix wrapper — not an email shape to be screened out."""
    _write_home(home, 'plugin:\n  ids: ["15551234567@s.whatsapp.net"]\n')
    grants = _find(store=_store(), platform="whatsapp", user="15551234567")
    rows = [s for s in grants if s.location == "config.yaml [unmodeled path redacted]"]
    assert rows and rows[0].kind == "unresolved"


def test_reports_never_echo_an_unrelated_secret(home):
    """A value that merely CONTAINS the principal can be a credential; the report must not carry it."""
    from gateway.access_grants import describe

    secret = "synthetic-secret-token-abc123"
    _write_home(home, f'plugin:\n  credential: "alice,{secret}"\n')
    grants = _find(store=_store(), user="alice")
    assert [s.location for s in grants if s.kind == "unresolved"] == ["config.yaml [unmodeled path redacted]"]
    blob = json.dumps(describe(grants))
    assert secret not in blob and "alice," not in blob
    assert all(row["entry"] == "" for row in describe(grants) if row["kind"] == "unresolved")


def test_a_live_gateway_keeps_the_change_from_reading_as_enforced(home, monkeypatch, capsys):
    """Files can be clean while the running process still enforces the old list — say so, exit 3."""
    import hermes_cli.pairing as pairing_cli

    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002"]\n', encoding="utf-8")
        monkeypatch.setattr(pairing_cli, "_live_gateway_pid", lambda store: 4242)

        assert _revoke_cli(_store(), effective=True) == 3
        out = capsys.readouterr().out
        assert "Removed config.yaml discord.allow_from" in out
        assert "running gateway (pid 4242)" in out
        assert "cannot be evaluated offline" in out


@pytest.mark.parametrize('config', [
    'allow_all_users: true\n',
    'allow_all_users: "true"\n',
    'gateway:\n  allow_all_users: "yes"\n',
    'discord:\n  allow_all_users: "true"\n',
    'discord:\n  allow_all_users: 1\n',
    'platforms:\n  discord:\n    extra:\n      allow_all_users: "true"\n',
])
@pytest.mark.parametrize('effective', [False, True])
def test_allow_all_real_cli_and_gateway(home, config, effective):
    import subprocess
    import sys

    _write_home(home, config, 'DISCORD_ALLOWED_USERS=1002\n' if effective else '')
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, '-m', 'hermes_cli.main', 'pairing', 'revoke',
         *(['--effective'] if effective else []), PLATFORM, USER],
        env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 3, (result.stdout, result.stderr)
    assert 'allow_all_users' in result.stdout
    assert (home / 'config.yaml').read_text() == config
    extra = _adapter_extra_from_loaded_config()
    assert all(_verdict(extra, _store(), chat_type=kind)
               for kind in ('dm', 'group', 'channel', 'forum'))


def test_authored_env_flags_without_process_env(home):
    _write_home(home, '', 'DISCORD_ALLOW_ALL_USERS=yes\nDISCORD_ALLOWED_ROLES=admin\n')
    rows = _find(store=_store())
    assert {'allow_all', 'role'} <= {row.kind for row in rows}
    assert all(row.authorizes_now is None for row in rows if row.kind in {'allow_all', 'role'})


def test_malformed_mapping_is_one_clear_unresolved_row(home):
    _write_home(home, 'discord:\n  allow_from: {"1001": true, "1002": true}\n')
    rows = [r for r in _find(store=_store()) if r.location == 'config.yaml discord.allow_from']
    assert len(rows) == 1 and rows[0].kind == 'unresolved'
    assert 'mapping' in rows[0].reason and not rows[0].removable


def test_block_level_groups_are_not_edited(home):
    config = 'discord:\n  groups:\n    chan-1:\n      allow_from: ["1002"]\n'
    _write_home(home, config)
    assert _revoke_cli(_store(), effective=True) == 3
    assert (home / 'config.yaml').read_text() == config


def test_gateway_detection_failure_is_not_success(home, monkeypatch, capsys):
    import gateway.status
    def fail(_home):
        raise PermissionError('synthetic failure')
    monkeypatch.setattr(gateway.status, 'live_gateway_pid_for_home', fail)
    _write_home(home, 'discord:\n  allow_from: ["1002"]\n')
    assert _revoke_cli(_store(), effective=True) == 3
    assert 'could not determine' in capsys.readouterr().out.lower()


def test_gateway_detection_scopes_served_profiles_to_default_root(home, monkeypatch):
    import gateway.status
    import hermes_constants
    import hermes_cli.gateway_multiplex_served as served
    import hermes_cli.pairing as pairing_cli
    root = home / 'root'
    monkeypatch.setattr(hermes_constants, 'get_default_hermes_root', lambda: root)
    monkeypatch.setattr(gateway.status, 'live_gateway_pid_for_home', lambda _: None)
    monkeypatch.setattr(served, 'live_default_gateway_pid', lambda: 4242)
    monkeypatch.setattr(served, 'recorded_served_profiles', lambda: ['seat'])
    monkeypatch.setenv('HERMES_HOME', str(home / 'other' / 'profiles' / 'seat'))
    assert pairing_cli._live_gateway_pid(SimpleNamespace(profile=None)) == 0
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'seat'))
    assert pairing_cli._live_gateway_pid(SimpleNamespace(profile=None)) == 4242
    monkeypatch.setattr(served, 'recorded_served_profiles', lambda: None)
    assert pairing_cli._live_gateway_pid(SimpleNamespace(profile=None)) == -1


def test_no_live_gateway_means_a_file_removal_is_a_denial(home, monkeypatch, capsys):
    """Control: no live process for this home ⇒ the effective removal exits 0."""
    import hermes_cli.pairing as pairing_cli

    with _secondary_profile(home) as secondary:
        (secondary / "config.yaml").write_text(
            'discord:\n  enabled: true\n  allow_from: ["1001", "1002"]\n', encoding="utf-8")
        monkeypatch.setattr(pairing_cli, "_live_gateway_pid", lambda store: 0)
        assert _revoke_cli(_store(), effective=True) == 0


def test_effective_partial_write_failure_reports_changed_sources(home, monkeypatch, capsys):
    import hermes_cli.config as config
    _write_home(home, 'discord:\n  allow_from: ["1002"]\n', 'DISCORD_ALLOWED_USERS=1002\n')
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    def fail(*args, **kwargs):
        raise OSError("synthetic-secret-error")
    monkeypatch.setattr(config, "remove_env_value", fail)
    assert _revoke_cli(store, effective=True) == 1
    out = capsys.readouterr().out
    assert "config.yaml" in out and "changed" in out.lower()
    assert "nothing was changed" not in out.lower()
    assert "synthetic-secret-error" not in out
    assert store.is_approved(PLATFORM, USER)
    assert "1002" not in (home / "config.yaml").read_text()
    assert "1002" in (home / ".env").read_text()


def test_unmodeled_yaml_secret_keys_are_redacted(home, capsys):
    from gateway.access_grants import describe
    secret = "synthetic-private-key-abcdef"
    _write_home(home, f'plugin:\n  {secret}:\n    nested-secret-123: ["1002"]\n')
    grants = _find(store=_store())
    assert any(s.kind == "unresolved" for s in grants)
    assert secret not in json.dumps(describe(grants))
    assert "nested-secret-123" not in json.dumps(describe(grants))
    assert _revoke_cli(_store()) == 3
    out = capsys.readouterr().out
    assert secret not in out and "nested-secret-123" not in out


def test_failed_env_write_may_have_changed_target(home, monkeypatch, capsys):
    import hermes_cli.config as config
    _write_home(home, 'discord:\n  allow_from: ["1002"]\n', 'DISCORD_ALLOWED_USERS=1002\n')
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    def change_then_fail(*args, **kwargs):
        (home / ".env").write_text("# changed before failure\n")
        raise OSError("private diagnostic")
    monkeypatch.setattr(config, "remove_env_value", change_then_fail)
    assert _revoke_cli(store, effective=True) == 1
    out = capsys.readouterr().out
    assert "config.yaml discord.allow_from" in out
    assert ".env DISCORD_ALLOWED_USERS may also have changed" in out
    assert "private diagnostic" not in out
    assert store.is_approved(PLATFORM, USER)


@pytest.mark.xfail(strict=True, reason="effective revoke does not roll back a completed YAML write")
def test_effective_env_failure_restores_completed_yaml(home, monkeypatch):
    """A failed second write should not silently drop the first explicit grant."""
    import hermes_cli.config as config

    original = 'discord:\n  allow_from: ["1001", "1002"]\n'
    _write_home(home, original, 'DISCORD_ALLOWED_USERS=1001,1002\n')
    store = _store()
    store._approve_user(PLATFORM, USER, "")
    monkeypatch.setattr(config, "save_env_value", lambda *a, **kw: (_ for _ in ()).throw(OSError()))
    assert _revoke_cli(store, effective=True) == 1
    assert (home / "config.yaml").read_text() == original


@pytest.mark.xfail(strict=True, reason="stale staged YAML overwrites concurrent edits to existing lists")
def test_effective_preserves_concurrent_yaml_grant(home, monkeypatch):
    """Inject an independent grant after staging, immediately before the real writer."""
    import hermes_cli.config as config

    _write_home(home, 'discord:\n  allow_from: ["1001", "1002"]\n')
    real_write = config.atomic_config_write

    def concurrent_write(path, data):
        path.write_text('discord:\n  allow_from: ["1001", "1002", "1003"]\n')
        return real_write(path, data)

    monkeypatch.setattr(config, "atomic_config_write", concurrent_write)
    _revoke_cli(_store(), effective=True)
    assert '1003' in (home / "config.yaml").read_text()


@pytest.mark.xfail(strict=True, reason="stale staged env value overwrites concurrent env updates")
def test_effective_preserves_concurrent_env_grant(home, monkeypatch):
    """A cooperating env writer can add a grant after revoke's unlocked read."""
    import hermes_cli.config as config

    _write_home(home, '', 'DISCORD_ALLOWED_USERS=1001,1002\n')
    real_save = config.save_env_value

    def concurrent_save(key, value):
        real_save(key, '1001,1002,1003')
        return real_save(key, value)

    monkeypatch.setattr(config, "save_env_value", concurrent_save)
    _revoke_cli(_store(), effective=True)
    assert '1003' in (home / ".env").read_text()


def test_group_mapping_key_is_redacted_from_reports(home, capsys):
    from gateway.access_grants import describe
    secret = "secret-group-key-123456"
    _write_home(home, f'platforms:\n  wecom:\n    extra:\n      groups:\n        {secret}:\n          allow_from: ["1002", "1001"]\n')
    rows = _find(store=_store(), platform="wecom")
    assert any(s.kind == "yaml" for s in rows)
    assert secret not in json.dumps(describe(rows))
    assert _revoke_cli(_store(), platform="wecom") == 3
    assert secret not in capsys.readouterr().out
