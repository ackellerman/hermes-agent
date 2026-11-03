"""Every grant source that names one principal on one platform — the offline view.

``hermes pairing revoke`` removes a pairing record; it does not edit the operator's
``config.yaml``. But a ``config.yaml`` allow list is an independent grant that can keep the
same principal authorized long after the pairing record is gone, so a revocation that reports
"done" without naming that grant lies to the operator. This module enumerates every source that
names the user, says whether each one authorizes *now* in this deployment shape, and (for
``--effective``) removes the removable ones.

The property this module owes its callers: **exit 0 means no grant names the user in the shape a
gateway would load this profile in.** The model is therefore what the RUNNING gateway consults,
not what this one-shot process happens to see — the CLI never runs the YAML→env bridge, so the
process env is only trusted where it is *authored*:

* **authored** env allowlist = the profile ``.env`` sets it, or the process env sets it and the
  YAML→env bridge did not write it (``bridged_env_var``). A bridged value never suppresses a YAML
  row: the bridge copies ``allow_from`` into the platform list, which the union then consults.
* YAML/``extra`` allow lists (``allow_from`` / ``group_allow_from`` / ``allowed_users``) authorize
  only while the profile has NO authored env allowlist at all — that is the only route to
  ``_adapter_extra_allowlist_authorizes`` (``gateway/authz_mixin.py``). Group-sender, group-chat
  and global lists do NOT suppress a YAML row through that gate; they are judged as their own
  consulted lists, chat-scope aware.
* ``allow_all_users`` authorizes through its bridged env flag only (``<PLATFORM>_ALLOW_ALL_USERS``
  / ``GATEWAY_ALLOW_ALL_USERS``), which does not reach a multiplexed secondary profile.
* ``GATEWAY_ALLOWED_USERS`` and platform/group env entries authorize when a consulted list names
  the principal — matched through authz's own predicate (``_principal_matches_allowlist``), never
  a second implementation.

``authorizes_now`` is ``True`` (authorizes in every context the row covers), ``False`` (evaluable,
and authorizes in none) or ``None`` (cannot be decided offline — a role list, a username entry, an
entry that some alias *might* match, an unreadable config, a value declared but not consulted in
this process). Callers must never print an unqualified success while a source is ``True`` or
``None``. Chat scope is part of the verdict: a group list never claims a DM grant (F4).

Reads are side-effect free: the YAML view is a throwaway ``platforms_data`` filled by the loader's
own ``bridge_platform_shared_keys``, so no process env is written and no plugin hook runs. Writes
(``remove_user_grants``) go through ``hermes_cli.config``'s guarded primitives: validate everything
first, write ``config.yaml``, then ``.env``, then purge in-process adapter snapshots — a refusal
raises ``AccessGrantError`` before the first byte changes.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DM = "dm"
GROUP = "group"
GLOBAL = "global"

# Discord allow_from is one sender list covering DMs and guild channels (upstream PR #103743 keeps
# extending channel traffic to it), so its scope label must not claim "dm".
_DISCORD_GLOBAL_SCOPE = "global(dm+channel)"

# Platforms whose allow-list scope is one sender list covering DMs AND guild channels (upstream PR
# #103743 keeps extending channel traffic to it), so the scope label must not claim "dm".
_DISCORD_PLATFORMS = frozenset({"discord"})

# (key, chat_scope) for the allow lists an operator can author. ``None`` scope = the platform's own
# allow-list scope (global, or Discord's dm+channel).
_ALLOW_KEYS: tuple[tuple[str, Optional[str]], ...] = (
    ("allow_from", None),
    ("group_allow_from", GROUP),
    ("allowed_users", None),
    # Command-authority lists, not access: reported when they name the user, never removed and
    # never counted as an access grant (plan §8).
    ("allow_admin_from", None),
    ("group_allow_admin_from", GROUP),
)

# Every place a platform block may live; the loader merges all three (``merge_platform_sections``
# / ``platform_section`` in ``gateway/config_loader.py``).
_BLOCK_PREFIXES: tuple[tuple[str, ...], ...] = (
    (),
    ("platforms",),
    ("gateway", "platforms"),
)

_REPORTED_FIELDS = ("kind", "location", "chat_scope", "entry", "removable", "reason", "authorizes_now")

_NO_AUTHORED_ENV = "no authored env allowlist is configured, so this list is the one that authorizes"
_ENV_SHAPE_REASON = (
    "an authored env allowlist is configured for this platform; env beats YAML, and whether this "
    "entry still reaches authz depends on the profile shape — cannot be decided offline")
_USERNAME_REASON = ("cannot be evaluated offline: a non-numeric entry that an adapter may resolve to "
                    "a numeric id at connect time")
_MENTION_REASON = ("mention-form entry names the principal but authz compares entries verbatim, so "
                   "this entry alone does not authorize; the adapter may resolve it at connect time")
_DECLARED_NOT_CONSULTED = "declared in `.env`; a restart re-reads it"
_GLOBAL_REASON = ("global allowlist is cross-platform: not removed per platform (it would revoke this "
                  "principal on every other platform too)")
_UNRESOLVED_ENV = ("cannot be evaluated offline: this platform's allowlist env var is not resolvable "
                   "in this process (no registry entry, no known name), so a gateway may consult one "
                   "that names this user")
_AUTHORITY_REASON = "command-authority list, not access: reported, never removed, never counted"


class AccessGrantError(RuntimeError):
    """A grant removal was refused. Raised BEFORE the first write, so nothing half-applies."""


@dataclass(frozen=True)
class GrantSource:
    """One place that names the principal.

    ``_path`` is the authored location (dict keys inside ``config.yaml``, or the env var name) and
    ``_platform`` / ``_user_id`` carry the principal — internal provenance for
    :func:`remove_user_grants`, never part of the reported surface (:func:`describe`).
    """

    kind: str            # "pairing" | "env" | "yaml" | "wildcard" | "allow_all" | "role" | "unresolved"
    location: str        # "config.yaml platforms.discord.allow_from", ".env DISCORD_ALLOWED_USERS", ...
    chat_scope: str      # "dm" | "group" | "global" | "global(dm+channel)"
    entry: str           # matched entry, e.g. "1002" or "*"
    removable: bool
    reason: str = ""
    authorizes_now: Optional[bool] = None
    _path: tuple[str, ...] = ()
    _platform: str = ""
    _user_id: str = ""


def describe(sources: Iterable[GrantSource]) -> list[dict]:
    """JSON-ready rows for a CLI/dashboard surface."""
    return [{name: getattr(source, name) for name in _REPORTED_FIELDS} for source in sources]


# --------------------------------------------------------------------------- readers / matching


def _platform_obj(platform: str):
    from gateway.config import Platform

    with contextlib.suppress(ValueError, AttributeError):
        return Platform(platform)
    return None


class _PlatformShim:
    """A ``.value``-carrying stand-in for a platform the enum has no member for.

    Registry lookups and ``_principal_matches_allowlist`` key on ``platform.value``, so a plugin
    platform whose enum member only exists after its plugin loads still gets the platform-specific
    alias normalisation (Buzz ``npub1…`` ↔ hex) instead of silently missing a grant (F2).
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


def _platform_handle(platform: str):
    return _platform_obj(platform) or _PlatformShim(str(platform or "").strip().lower())


def _source(platform: str, user_id: str, chat_type: str = DM):
    """A ``SessionSource`` for authz's own predicate, in the chat context being judged."""
    from gateway.session import SessionSource

    return SessionSource(
        platform=_platform_handle(platform),  # type: ignore[arg-type]  # shim only when no enum member
        chat_id="dm-1" if chat_type == DM else "group-1",
        chat_type=chat_type,
        user_id=user_id,
    )


def _ordered_entries(raw: Any) -> list[str]:
    """Allow-list entries in authored order (list, JSON literal string, or CSV string)."""
    if raw is None:
        return []
    from gateway.platforms._shared import decode_json_list_literal

    decoded = decode_json_list_literal(raw)
    if isinstance(decoded, (list, tuple, set, frozenset)):
        return [str(part).strip() for part in decoded if str(part).strip()]
    return [part.strip() for part in str(decoded).split(",") if part.strip()]


def _entries(raw: Any) -> list[str]:
    """Parsed entries, de-duplicated and sorted (stable report order)."""
    return sorted(set(_ordered_entries(raw)))


def _mention_id(entry: str) -> Optional[str]:
    """``<@1002>`` / ``<@!1002>`` → ``"1002"``; anything else → None."""
    text = str(entry or "").strip()
    if len(text) > 4 and text.startswith("<@") and text.endswith(">"):
        inner = text[2:-1]
        if inner.startswith("!"):
            inner = inner[1:]
        if inner.isdigit():
            return inner
    return None


def _is_wildcard(entry: Any) -> bool:
    return str(entry or "").strip() == "*"


def _matches(platform: str, entry: Any, user_id: str) -> bool:
    """Whether authz itself would admit ``user_id`` from this single entry.

    Delegates to ``_principal_matches_allowlist`` on a singleton set — the SAME predicate authz
    evaluates — so every platform alias rule (WhatsApp phone↔JID, Buzz ``npub1…``↔hex, SimpleX
    contactId-only) is inherited instead of re-implemented (F2). A ``<@1002>`` mention is NOT a
    verbatim allow-list entry, so it does not match here.
    """
    text = str(entry or "").strip()
    if not text or _is_wildcard(text) or _is_wildcard(user_id):
        return False
    try:
        from gateway.authz_mixin import _coerce_allow_set, _principal_matches_allowlist

        return bool(_principal_matches_allowlist(_source(platform, user_id), user_id, _coerce_allow_set(text)))
    except Exception:  # noqa: BLE001 - fall back to the pairing store's alias rules, never fail open
        from gateway.pairing import _user_ids_match

        return bool(_user_ids_match(platform, text, user_id))


def _mention_matches(entry: Any, user_id: str) -> bool:
    """``<@1002>`` / ``<@!1002>`` names the principal for REPORTING/REMOVAL only."""
    mention = _mention_id(str(entry or ""))
    return mention is not None and mention == str(user_id)


def _unresolved(platform: str, entry: Any, user_id: str) -> bool:
    """An entry whose shape cannot be compared to this principal offline.

    Discord resolves username entries through the API at connect time; any plugin platform may
    normalise its own entry forms. So an entry that neither the predicate nor the mention rule could
    decide, and whose shape differs from the principal's, is reported as cannot-be-evaluated on
    EVERY platform (F2) — exit 3, never a silent 0.
    """
    text = str(entry or "").strip()
    if not text or not str(user_id).isdigit():
        return False
    return not (text.isdigit() or _is_wildcard(text) or _mention_id(text) is not None)


def _platform_allow_env_resolution(platform: str) -> tuple[str, bool]:
    """``(env var, resolved)`` for the platform's own allowlist (F2).

    Order: the platform-registry entry's ``allowed_users_env`` (plugin platforms), then
    ``pairing._PLATFORM_ALLOWLIST_ENV`` (built-ins). An unresolved name fails CLOSED: the caller
    emits an ``unresolved`` row, because a gateway process may still consult a list here.
    """
    platform = str(platform or "").strip().lower()
    if not platform:
        return "", False
    from gateway.authz_mixin import _registry_entry

    entry = _registry_entry(_platform_handle(platform))
    name = str(getattr(entry, "allowed_users_env", "") or "").strip() if entry is not None else ""
    if not name:
        from gateway.pairing import _PLATFORM_ALLOWLIST_ENV

        name = str(_PLATFORM_ALLOWLIST_ENV.get(platform) or "").strip()
    return name, bool(name)


def _platform_allow_env(platform: str) -> str:
    return _platform_allow_env_resolution(platform)[0]


def _allow_all_env(platform: str) -> str:
    env = _platform_allow_env(platform)
    if env.endswith("_ALLOWED_USERS"):
        return env.replace("_ALLOWED_USERS", "_ALLOW_ALL_USERS")
    entry = None
    with contextlib.suppress(Exception):
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get(platform)
    return str(getattr(entry, "allow_all_env", "") or "") if entry is not None else ""


def _roles_env(platform: str) -> str:
    env = _platform_allow_env(platform)
    return env.replace("_ALLOWED_USERS", "_ALLOWED_ROLES") if env.endswith("_ALLOWED_USERS") else ""


def _group_envs(platform: str) -> tuple[str, str]:
    """``(sender allowlist, chat allowlist)`` env names authz actually consults for this platform."""
    from gateway.authz_mixin import _GROUP_CHAT_ENV, _GROUP_USER_ENV

    plat = _platform_obj(platform)
    if plat is None:
        return "", ""
    return _GROUP_USER_ENV.get(plat, "") or "", _GROUP_CHAT_ENV.get(plat, "") or ""


def _gate_env(name: str) -> str:
    """The same per-profile isolated reader authz uses — never a raw ``os.environ`` read."""
    from gateway.platforms._shared import platform_gate_env

    return platform_gate_env(name) if name else ""


def _truthy(name: str) -> bool:
    return bool(name) and _gate_env(name).strip().lower() in {"true", "1", "yes"}


def _authored_entries(home: Path, var: str) -> list[str]:
    """Entries a gateway LOADS for ``var``: the profile ``.env``, plus process env it did not bridge.

    This is the only value the YAML-suppressing test may use (F1). The CLI never runs the YAML→env
    bridge, so a process value that ``bridged_env_var`` reports as this process's own bridge write is
    a replica of ``config.yaml`` — trusting it would make a YAML row disappear precisely when the
    gateway would still authorize it.
    """
    if not var:
        return []
    from agent.secret_scope import load_env_file
    from gateway.platforms._shared import bridged_env_var

    from_file = set(_entries(load_env_file(home / ".env").get(var)))
    process = set() if bridged_env_var(var) else set(_entries(_gate_env(var)))
    return sorted(from_file | process)


def _consulted_or_authored(home: Path, var: str) -> bool:
    """True when ``var`` names anything at all — now (consulted) or after a restart (authored)."""
    if not var:
        return False
    if _entries(_gate_env(var)):
        return True
    from agent.secret_scope import load_env_file

    return bool(_entries(load_env_file(home / ".env").get(var)))


def _scope_for(platform: str) -> str:
    return _DISCORD_GLOBAL_SCOPE if platform in _DISCORD_PLATFORMS else GLOBAL


# --------------------------------------------------------------------------- home / config


def resolve_home(*, store: Any = None, home: Any = None) -> Path:
    """The profile home these grants live in.

    Mirrors ``PairingStore.__init__``'s own root logic so a profile-scoped store reports THAT
    profile's ``config.yaml`` / ``.env``, never the process home (a named profile's grants are
    invisible to the default profile's gateway, and vice versa).
    """
    if home is not None:
        return Path(home)
    profile = getattr(store, "profile", None) if store is not None else None
    if profile:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root()
        return root if profile == "default" else root / "profiles" / profile
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _read_config(home: Path) -> tuple[dict, Optional[str]]:
    """Authored ``config.yaml`` (raw, unexpanded) or ``({}, refusal)`` when it cannot be read."""
    path = home / "config.yaml"
    if not path.exists():
        return {}, None
    from hermes_cli.config import require_readable_config_before_write

    try:
        return require_readable_config_before_write(path), None
    except RuntimeError as exc:  # _refuse_overwrite: unreadable / unparseable / non-mapping
        return {}, str(exc)


def _effective_extra(config: dict, platform: str) -> dict:
    """``config.extra`` as the loader's own precedence would produce it (no env writes)."""
    plat = _platform_obj(platform)
    if plat is None:
        return {}
    from gateway.config_loader import bridge_platform_shared_keys, merge_platform_sections

    gateway_section = config.get("gateway")
    gateway_platforms = gateway_section.get("platforms") if isinstance(gateway_section, dict) else None
    gw_data: dict = {}
    platforms_data = merge_platform_sections(config, gateway_section, gw_data)
    bridge_platform_shared_keys(config, gateway_platforms, gw_data, platforms_data, [plat])
    block = platforms_data.get(platform)
    return dict(block.get("extra") or {}) if isinstance(block, dict) else {}


def _env_allowlist_configured(platform: str, home: Path) -> bool:
    """True when ANY env allowlist authz consults names something — consulted now or authored.

    Only ``GATEWAY_ALLOW_ALL_USERS``'s reachability uses this four-list superset (authz consults a
    group list only in a group context, so the process cannot decide per-context offline).
    """
    group_user, group_chat = _group_envs(platform)
    return any(
        _consulted_or_authored(home, name)
        for name in (_platform_allow_env(platform), group_user, group_chat, "GATEWAY_ALLOWED_USERS")
    )


# --------------------------------------------------------------------------- enumeration


def find_user_grants(
    platform: str, user_id: str, *, store: Any = None, home: Any = None,
) -> list[GrantSource]:
    """Every source that names ``user_id`` on ``platform``.

    ``store`` (a ``PairingStore``) scopes the home AND supplies the pairing record; ``home``
    overrides the home for tests. Nothing is written and no process env is mutated.
    """
    platform = str(platform or "").strip().lower()
    user_id = str(user_id or "").strip()
    if not platform or not user_id:
        return []
    resolved_home = resolve_home(store=store, home=home)

    sources: list[GrantSource] = []
    # The pairing record: the caller owns the store, this module never re-reads a home for it.
    if store is not None and store.is_approved(platform, user_id):
        sources.append(GrantSource(
            kind="pairing", location="pairing store", chat_scope=GLOBAL, entry=user_id,
            removable=True, authorizes_now=True, _platform=platform, _user_id=user_id))

    platform_env, env_known = _platform_allow_env_resolution(platform)
    # F1: the YAML-reaches-extra gate is the AUTHORED platform allowlist only. A bridged value, a
    # group-sender list, a group-chat list and the global list never suppress a YAML row — the
    # bridge copies ``allow_from`` into the platform list, which the union then consults.
    yaml_only_env = not _authored_entries(resolved_home, platform_env)
    sources.extend(_env_sources(platform, user_id, resolved_home, platform_env, env_known))

    config, refusal = _read_config(resolved_home)
    if refusal:
        sources.append(GrantSource(
            kind="unresolved", location="config.yaml", chat_scope=GLOBAL, entry="", removable=False,
            reason=f"config.yaml cannot be read, so its grants cannot be evaluated offline: {refusal}",
            authorizes_now=None, _platform=platform, _user_id=user_id))
        return sources
    sources.extend(_yaml_sources(platform, user_id, config, yaml_only_env, env_known))
    return sources


def _env_location(var: str, from_env_file: str, bridged: bool) -> str:
    """Report the provenance of an env value: authored ``.env``, this process's bridge, or process env."""
    if from_env_file:
        return f".env {var}"
    return f"env {var} (bridged from config.yaml)" if bridged else f"env {var} (process)"


def _env_allow_source(
    platform: str, user_id: str, var: str, scope: str, entry: str, from_env_file: str,
    *, bridged: bool, global_scope: bool,
) -> GrantSource:
    """One ``env`` source, with the provenance that decides removability and the reason text."""
    location = _env_location(var, from_env_file, bridged)
    removable = bool(from_env_file) and not global_scope
    authorizes: Optional[bool] = True if _matches(platform, entry, user_id) else None
    if authorizes is None:
        reason = _MENTION_REASON
    elif removable:
        reason = ""
    elif global_scope:
        reason = _GLOBAL_REASON
    elif bridged:
        reason = "the YAML→env bridge wrote this value; edit config.yaml"
    else:
        reason = "set outside the profile .env (process environment)"
    return GrantSource(
        kind="env", location=location, chat_scope=scope, entry=entry, removable=removable,
        reason=reason, authorizes_now=authorizes, _path=(var,), _platform=platform, _user_id=user_id)


def _env_sources(
    platform: str, user_id: str, home: Path, platform_env: str, env_known: bool,
) -> list[GrantSource]:
    """Env allowlists, provenance-aware: ``.env`` (authored), bridged, process-only."""
    from agent.secret_scope import load_env_file
    from gateway.platforms._shared import bridged_env_var

    authored = load_env_file(home / ".env")
    group_user, _group_chat = _group_envs(platform)
    allow_all = _allow_all_env(platform)
    sources: list[GrantSource] = []
    if not env_known:
        # F2: fail closed. A gateway process may load a plugin that names this platform's allowlist
        # env var, so "I could not find one" is never "there is none".
        sources.append(GrantSource(
            kind="unresolved", location=f"env {platform.upper()} allowlist", chat_scope=GLOBAL,
            entry="", removable=False, reason=_UNRESOLVED_ENV, authorizes_now=None,
            _platform=platform, _user_id=user_id))

    for var, scope in (
        (platform_env, _scope_for(platform)),
        (group_user, GROUP),
        ("GATEWAY_ALLOWED_USERS", GLOBAL),
    ):
        if not var:
            continue
        global_scope = var == "GATEWAY_ALLOWED_USERS"
        bridged = bridged_env_var(var)
        from_env_file = str(authored.get(var) or "").strip()
        consulted = _entries(_gate_env(var))
        matched = [entry for entry in consulted if _names_principal(platform, entry, user_id)]
        for entry in matched:
            sources.append(_env_allow_source(
                platform, user_id, var, scope, entry, from_env_file,
                bridged=bridged, global_scope=global_scope))
        if not matched:
            # F3: declared in the profile's .env but not consulted by THIS process (another profile's
            # secret scope, a stale process env). A restart re-reads it, so it is unknown, never False.
            for entry in _entries(from_env_file):
                if not _names_principal(platform, entry, user_id):
                    continue
                reason = _DECLARED_NOT_CONSULTED if not global_scope else f"{_DECLARED_NOT_CONSULTED} — {_GLOBAL_REASON}"
                sources.append(GrantSource(
                    kind="env", location=f".env {var}", chat_scope=scope, entry=entry,
                    removable=not global_scope, reason=reason,
                    authorizes_now=None, _path=(var,), _platform=platform, _user_id=user_id))


    # Wildcards and username-shaped entries apply to the platform's own sender lists; the global list
    # is cross-platform and resolved per adapter, so it is reported above but not classified here.
    for var, scope in ((platform_env, _scope_for(platform)), (group_user, GROUP)):
        if not var:
            continue
        live_entries = _entries(_gate_env(var))
        var_location = _env_location(var, str(authored.get(var) or "").strip(), bridged_env_var(var))
        if any(_is_wildcard(entry) for entry in live_entries):
            sources.append(GrantSource(
                kind="wildcard", location=var_location, chat_scope=scope, entry="*",
                removable=False, reason="wildcard: admits every sender; never removed by revoke",
                authorizes_now=True, _path=(var,), _platform=platform, _user_id=user_id))
        for entry in live_entries:
            if _unresolved(platform, entry, user_id):
                sources.append(GrantSource(
                    kind="unresolved", location=var_location, chat_scope=scope, entry=entry,
                    removable=False, reason=_USERNAME_REASON, authorizes_now=None,
                    _path=(var,), _platform=platform, _user_id=user_id))

    if _truthy(allow_all):
        sources.append(GrantSource(
            kind="allow_all", location=f"env {allow_all}", chat_scope=GLOBAL, entry="true",
            removable=False, reason="allow-all flag: admits every sender; never removed by revoke",
            authorizes_now=True, _path=(allow_all,), _platform=platform, _user_id=user_id))
    if _truthy("GATEWAY_ALLOW_ALL_USERS"):
        # Consulted only in the no-env-allowlist branch, and only in a context that allowlist does
        # not cover — so with any env allowlist configured the reachability is unknown, never False.
        consulted = not _env_allowlist_configured(platform, home)
        sources.append(GrantSource(
            kind="allow_all", location="env GATEWAY_ALLOW_ALL_USERS", chat_scope=GLOBAL, entry="true",
            removable=False,
            reason=("allow-all flag: admits every sender on every platform (consulted only while no "
                    "env allowlist is configured); never removed by revoke"),
            authorizes_now=True if consulted else None, _path=("GATEWAY_ALLOW_ALL_USERS",),
            _platform=platform, _user_id=user_id))


    roles_env = _roles_env(platform)
    roles = _gate_env(roles_env)
    if roles:
        sources.append(GrantSource(
            kind="role", location=f"env {roles_env}", chat_scope=GLOBAL, entry=",".join(_entries(roles)),
            removable=False,
            reason="role grant cannot be evaluated per user offline; a matching role admits this user",
            authorizes_now=None, _path=(roles_env,), _platform=platform, _user_id=user_id))
    return sources


def _yaml_sources(
    platform: str, user_id: str, config: dict, yaml_only_env: bool, env_known: bool,
) -> list[GrantSource]:
    """Authored ``config.yaml`` blocks, key by key, with the loader's own reachability."""
    effective = _effective_extra(config, platform)
    # The loader needs a ``Platform`` to merge/bridge a block; without one the effective value is
    # unknown, and an unknown effective value must never read as "does not authorize".
    loader_known = _platform_obj(platform) is not None and env_known
    sources: list[GrantSource] = []
    for prefix in _BLOCK_PREFIXES:
        block: Any = config
        for key in (*prefix, platform):
            block = block.get(key) if isinstance(block, dict) else None
        if not isinstance(block, dict):
            continue
        block_path = (*prefix, platform)
        containers: list[tuple[dict, tuple[str, ...]]] = [(block, block_path)]
        extra = block.get("extra")
        if isinstance(extra, dict):
            containers.append((extra, (*block_path, "extra")))
        for container, container_path in containers:
            sources.extend(_yaml_container_sources(
                platform, user_id, container, container_path, effective,
                yaml_only_env=yaml_only_env, loader_known=loader_known))
    return sources


def _yaml_authorizes(*, in_effective: bool, yaml_only_env: bool, loader_known: bool) -> Optional[bool]:
    """``authorizes_now`` for one authored YAML allow row (F1/F4): True or None, NEVER False.

    True only when the entry is in the loader's effective value AND no env allowlist is authored for
    this platform — under that shape the entry authorizes in every context: single-profile/default
    bridges it into the platform list the union consults, and a multiplexed secondary reaches
    ``config.extra`` directly. With an authored env allowlist (or an unresolvable platform) the
    shape decides, and the offline replica must not guess: None keeps the exit code at 3.
    """
    if not loader_known or not in_effective:
        return None
    return True if yaml_only_env else None


def _yaml_container_sources(
    platform: str, user_id: str, container: dict, container_path: tuple[str, ...],
    effective: dict, *, yaml_only_env: bool, loader_known: bool,
) -> list[GrantSource]:
    def effective_entries(key: str) -> set[str]:
        return {str(entry).strip() for entry in _entries(effective.get(key))}

    out: list[GrantSource] = []
    for key, scope in _ALLOW_KEYS:
        if key not in container:
            continue
        path = (*container_path, key)
        location = "config.yaml " + ".".join(path)
        entries = _entries(container.get(key))
        effective_set = effective_entries(key)
        authority = key.endswith("admin_from")
        # A group list speaks for GROUP traffic only; every other list covers the platform's own
        # scope (F4). ``chat_scope`` is on the row, so a group grant never reads as a DM grant.
        scope_value = scope or _scope_for(platform)
        for entry in entries:
            if not _names_principal(platform, entry, user_id):
                continue
            verbatim = _matches(platform, entry, user_id)
            if authority:
                authorizes: Optional[bool] = None
                reason = _AUTHORITY_REASON
            elif not verbatim:
                authorizes = None
                reason = _MENTION_REASON
            else:
                authorizes = _yaml_authorizes(
                    in_effective=str(entry).strip() in effective_set,
                    yaml_only_env=yaml_only_env, loader_known=loader_known)
                reason = _NO_AUTHORED_ENV if authorizes else _ENV_SHAPE_REASON
            out.append(GrantSource(
                kind="yaml", location=location, chat_scope=scope_value, entry=entry,
                removable=not authority, reason=reason, authorizes_now=authorizes,
                _path=path, _platform=platform, _user_id=user_id))
        if authority:
            continue
        if any(_is_wildcard(entry) for entry in entries):
            out.append(GrantSource(
                kind="wildcard", location=location, chat_scope=scope_value, entry="*",
                removable=False, reason="wildcard: admits every sender; never removed by revoke",
                authorizes_now=_yaml_authorizes(
                    in_effective="*" in effective_set, yaml_only_env=yaml_only_env,
                    loader_known=loader_known),
                _path=path, _platform=platform, _user_id=user_id))
        for entry in entries:
            if _unresolved(platform, entry, user_id):
                out.append(GrantSource(
                    kind="unresolved", location=location, chat_scope=scope_value, entry=entry,
                    removable=False, reason=_USERNAME_REASON, authorizes_now=None,
                    _path=path, _platform=platform, _user_id=user_id))

    if container.get("allow_all_users") is True:
        path = (*container_path, "allow_all_users")
        allow_all_env = _allow_all_env(platform)
        # F1: the YAML flag authorizes only through its bridged env flag. This process may never have
        # run the bridge (the CLI does not), so "not set here" is UNKNOWN, not inert: a gateway loads
        # this profile and sets it. None keeps the exit code at 3; False would silently drop the row.
        flag_live = bool(allow_all_env and _truthy(allow_all_env))
        out.append(GrantSource(
            kind="allow_all", location="config.yaml " + ".".join(path), chat_scope=GLOBAL, entry="true",
            removable=False,
            reason=("allow-all flag: admits every sender on this platform (authorizes through the "
                    f"bridged {allow_all_env or 'env flag'}; inert only for a multiplexed secondary "
                    "profile, whose YAML→env bridge is skipped)"),
            authorizes_now=True if flag_live else None,
            _path=path, _platform=platform, _user_id=user_id))

    roles = container.get("allowed_roles")
    if roles:
        path = (*container_path, "allowed_roles")
        out.append(GrantSource(
            kind="role", location="config.yaml " + ".".join(path), chat_scope=GLOBAL,
            entry=",".join(_entries(roles)), removable=False,
            reason="role grant cannot be evaluated per user offline; a matching role admits this user",
            authorizes_now=None, _path=path, _platform=platform, _user_id=user_id))
    return out


# --------------------------------------------------------------------------- removal


def _value_at(node: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_at(node: dict, path: tuple[str, ...], value: Any) -> None:
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _names_principal(platform: str, entry: Any, user_id: str) -> bool:
    """Names the principal at all — by authz's predicate or by mention spelling (removal/reporting)."""
    return _matches(platform, entry, user_id) or _mention_matches(entry, user_id)


def _strip(value: Any, platform: str, user_id: str) -> Any:
    """Drop the principal's entries, preserving the authored container type, order and ``*``."""
    if isinstance(value, str):
        return ",".join(entry for entry in _ordered_entries(value)
                        if not _names_principal(platform, entry, user_id))
    if isinstance(value, (list, tuple)):
        return [entry for entry in value if not _names_principal(platform, entry, user_id)]
    if isinstance(value, (set, frozenset)):
        return {entry for entry in value if not _names_principal(platform, entry, user_id)}
    return value


def _env_write_targets_home(home: Path) -> bool:
    """``save_env_value``/``remove_env_value`` write the PROCESS home's ``.env``."""
    from hermes_cli.config import get_env_path

    with contextlib.suppress(OSError):
        return get_env_path().parent.resolve() == Path(home).resolve()
    return False


def remove_user_grants(
    sources: Iterable[GrantSource], *, store: Any = None, home: Any = None,
) -> list[GrantSource]:
    """Remove every removable source; returns what was actually removed.

    Order: validate everything → write ``config.yaml`` → write ``.env`` → purge live adapters.
    Any refusal (unreadable/pinned ``config.yaml``, managed ``.env`` key, a value that changed
    between the scan and the write) raises :class:`AccessGrantError` before the first write, so the
    command can refuse instead of half-applying (a narrowed ``.env`` with the YAML grant intact).
    """
    sources = list(sources)
    platform = next((source._platform for source in sources if source._platform), "")
    user_id = next((source._user_id for source in sources if source._user_id), "")
    if not sources or not platform or not user_id:
        return []
    resolved_home = resolve_home(store=store, home=home)

    yaml_plan = [s for s in sources if s.kind == "yaml" and s.removable and s._path]
    env_plan = [s for s in sources if s.kind == "env" and s.removable and s._path]
    pairing_plan = [s for s in sources if s.kind == "pairing" and s.removable]

    # An unreadable config.yaml means the grant set is UNKNOWN: writing `.env` would report a
    # success nobody can verify (the file may name the principal), so refuse before any write.
    unreadable = [s for s in sources if s.kind == "unresolved" and s.location == "config.yaml"]
    if unreadable:
        raise AccessGrantError(f"{unreadable[0].reason} Nothing was removed.")

    from hermes_cli.config import (
        atomic_config_write, remove_env_value, require_env_writable,
        require_readable_config_before_write, save_env_value,
    )

    # ---- validate (nothing written yet) ----
    config_path = resolved_home / "config.yaml"
    staged: Optional[dict] = None
    if yaml_plan:
        try:
            staged = require_readable_config_before_write(config_path)
        except RuntimeError as exc:
            raise AccessGrantError(str(exc)) from exc
        # Group by authored path: several entry spellings of the SAME principal (``1002`` and
        # ``<@1002>``) are several sources but ONE list. Staging per path, not per source, keeps the
        # second pass from seeing its own first-pass write and refusing as "changed since reported".
        staged_paths: dict[tuple[str, ...], list] = {}
        for source in yaml_plan:
            staged_paths.setdefault(source._path, []).append(source)
        for path, group in staged_paths.items():
            value = _value_at(staged, path)
            if value is None:
                raise AccessGrantError(
                    f"{group[0].location} no longer exists (the file changed since it was reported); "
                    "nothing was removed.")
            remaining = _strip(value, platform, user_id)
            if _entries(remaining) == _entries(value):
                raise AccessGrantError(
                    f"{group[0].location} no longer names {user_id} (the file changed since it was "
                    "reported); nothing was removed.")
            _set_at(staged, path, remaining)

    env_path = resolved_home / ".env"
    env_writes: list[tuple[str, str]] = []
    env_removed: list[GrantSource] = []
    if env_plan:
        if not _env_write_targets_home(resolved_home):
            from hermes_constants import get_hermes_home

            raise AccessGrantError(
                f"refusing to edit .env for {resolved_home}: the process home is {get_hermes_home()}. "
                "Run the command with `hermes -p <profile>` so the write targets that profile.")
        from agent.secret_scope import load_env_file

        current = load_env_file(env_path)
        for var in sorted({source._path[0] for source in env_plan}):
            try:
                require_env_writable(var, f"revoke {user_id}")
            except ValueError as exc:
                raise AccessGrantError(str(exc)) from exc
            before = str(current.get(var) or "")
            after = _strip(before, platform, user_id)
            if _entries(after) == _entries(before):
                continue
            env_writes.append((var, after))
            env_removed.extend(source for source in env_plan if source._path[0] == var)

    # ---- write: config.yaml first, then .env ----
    removed: list[GrantSource] = []
    if staged is not None and yaml_plan:
        try:
            atomic_config_write(config_path, staged)
        except (RuntimeError, ValueError) as exc:
            raise AccessGrantError(f"config.yaml was not written: {exc}") from exc
        removed.extend(yaml_plan)
    for var, after in env_writes:
        if after:
            save_env_value(var, after)
        else:
            remove_env_value(var)
    removed.extend(env_removed)

    for source in pairing_plan:
        if store is not None and store.revoke(platform, user_id):
            removed.append(source)

    if removed:
        from gateway.pairing import _sync_live_adapter_allowlist_remove

        _sync_live_adapter_allowlist_remove(platform, user_id)
    return removed
