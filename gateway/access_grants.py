"""Every grant source that names one principal on one platform — the offline view.

``hermes pairing revoke`` removes a pairing record; it does not edit the operator's
``config.yaml``. But a ``config.yaml`` allow list is an independent grant that can keep the
same principal authorized long after the pairing record is gone, so a revocation that reports
"done" without naming that grant lies to the operator. This module enumerates every source that
names the user, says whether each one authorizes *now* in this deployment shape, and (for
``--effective``) removes the removable ones.

Precedence replica — the source of truth is ``gateway/authz_mixin.py:645-696``:

* YAML/``extra`` allow lists (``allow_from`` / ``group_allow_from`` / ``allowed_users``) authorize
  only while the profile has NO env allowlist at all (platform, group sender, group chat, global)
  — that branch is the only route to ``_adapter_extra_allowlist_authorizes``;
* ``allow_all_users`` authorizes through its env flag only (the YAML key is bridged to
  ``<PLATFORM>_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOW_ALL_USERS`` outside a multiplexed secondary);
* ``GATEWAY_ALLOWED_USERS`` and platform/group env entries always authorize when they name the
  principal.

``authorizes_now`` is ``True`` (authorizes), ``False`` (declared but not consulted in this shape)
or ``None`` (cannot be evaluated offline — a role list, an unresolved username entry, an
unreadable config). Callers must never print an unqualified success while a source is ``True`` or
``None``.

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

# Platforms whose adapter resolves username-shaped allowlist entries to numeric IDs at connect time
# (``adapter.resolved_allowlist_user_ids``): a non-numeric entry can authorize a numeric principal
# and cannot be evaluated offline.
_USERNAME_RESOLVING_PLATFORMS = frozenset({"discord"})

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

_NOT_CONSULTED = "config.yaml allow list is consulted only while no env allowlist is configured"
_USERNAME_REASON = ("cannot be evaluated offline: a username-shaped entry the adapter resolves to a "
                    "numeric id at connect time")
_MENTION_REASON = ("mention-form entry names the principal but authz compares entries verbatim, so it "
                   "does not authorize; removed like a plain id")
_GLOBAL_REASON = ("global allowlist is cross-platform: not removed per platform (it would revoke this "
                  "principal on every other platform too)")


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
    """Principal match for REPORTING/REMOVAL: the repo's alias rules plus Discord mention syntax."""
    text = str(entry or "").strip()
    if not text or _is_wildcard(text) or _is_wildcard(user_id):
        return False
    from gateway.pairing import _user_ids_match

    if _user_ids_match(platform, text, user_id):
        return True
    mention = _mention_id(text)
    return mention is not None and mention == str(user_id)


def _authorizes_entry(platform: str, entry: Any, user_id: str) -> bool:
    """Whether authz itself would admit ``user_id`` from this entry.

    Authz compares allow-list entries VERBATIM (``_allows`` / ``_principal_matches_allowlist``), so a
    ``<@!1002>`` entry is inert config for the verdict even though it names the principal: it is
    matched for reporting/removal, never counted as an authorizing grant.
    """
    text = str(entry or "").strip()
    if _is_wildcard(text):
        return True
    if not text or _is_wildcard(user_id):
        return False
    from gateway.pairing import _user_ids_match

    return _user_ids_match(platform, text, user_id)


def _unresolved(platform: str, entry: Any, user_id: str) -> bool:
    """A non-numeric entry that could resolve to this numeric principal at connect time."""
    text = str(entry or "").strip()
    if platform not in _USERNAME_RESOLVING_PLATFORMS or not str(user_id).isdigit() or not text:
        return False
    if _is_wildcard(text) or text.isdigit() or _mention_id(text) is not None:
        return False
    return True


def _platform_allow_env(platform: str) -> str:
    from gateway.pairing import _allowlist_env_for_platform

    return _allowlist_env_for_platform(platform) or ""


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


def _scope_for(platform: str) -> str:
    return _DISCORD_GLOBAL_SCOPE if platform in _USERNAME_RESOLVING_PLATFORMS else GLOBAL


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


def _env_allowlist_configured(platform: str) -> bool:
    """True when ANY env allowlist authz consults is non-empty (the YAML-reaches-extra gate)."""
    platform_env = _platform_allow_env(platform)
    group_user, group_chat = _group_envs(platform)
    return any(_gate_env(name) for name in (platform_env, group_user, group_chat, "GATEWAY_ALLOWED_USERS"))


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

    yaml_only_env = not _env_allowlist_configured(platform)
    sources.extend(_env_sources(platform, user_id, resolved_home, yaml_only_env))

    config, refusal = _read_config(resolved_home)
    if refusal:
        sources.append(GrantSource(
            kind="unresolved", location="config.yaml", chat_scope=GLOBAL, entry="", removable=False,
            reason=f"config.yaml cannot be read, so its grants cannot be evaluated offline: {refusal}",
            authorizes_now=None, _platform=platform, _user_id=user_id))
        return sources
    sources.extend(_yaml_sources(platform, user_id, config, yaml_only_env))
    return sources


def _env_allow_source(
    platform: str, user_id: str, var: str, scope: str, entry: str, from_env_file: str,
    *, bridged: bool, global_scope: bool,
) -> GrantSource:
    """One ``env`` source, with the provenance that decides removability and the reason text."""
    location = f".env {var}" if from_env_file else (
        f"env {var} (bridged from config.yaml)" if bridged else f"env {var} (process)")
    removable = bool(from_env_file) and not global_scope
    authorizes = _authorizes_entry(platform, entry, user_id)
    if not authorizes:
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


def _env_sources(platform: str, user_id: str, home: Path, yaml_only_env: bool) -> list[GrantSource]:
    """Env allowlists, provenance-aware: ``.env`` (authored), bridged, process-only."""
    from agent.secret_scope import load_env_file
    from gateway.platforms._shared import bridged_env_var

    authored = load_env_file(home / ".env")
    platform_env = _platform_allow_env(platform)
    group_user, _group_chat = _group_envs(platform)
    allow_all = _allow_all_env(platform)
    sources: list[GrantSource] = []

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
        matched = [entry for entry in _entries(_gate_env(var)) if _matches(platform, entry, user_id)]
        for entry in matched:
            sources.append(_env_allow_source(
                platform, user_id, var, scope, entry, from_env_file,
                bridged=bridged, global_scope=global_scope))
        if not matched:
            # Declared in .env but not consulted by this process (no scope / another profile's value).
            for entry in _entries(from_env_file):
                if not _matches(platform, entry, user_id):
                    continue
                sources.append(GrantSource(
                    kind="env", location=f".env {var}", chat_scope=scope, entry=entry,
                    removable=not global_scope,
                    reason=("declared in .env but not consulted by this process (profile secret scope)"
                            if not global_scope else _GLOBAL_REASON),
                    authorizes_now=False, _path=(var,), _platform=platform, _user_id=user_id))

    # Wildcards and username-shaped entries apply to the platform's own sender lists; the global list
    # is cross-platform and resolved per adapter, so it is reported above but not classified here.
    for var, scope in ((platform_env, _scope_for(platform)), (group_user, GROUP)):
        if not var:
            continue
        live_entries = _entries(_gate_env(var))
        if any(_is_wildcard(entry) for entry in live_entries):
            sources.append(GrantSource(
                kind="wildcard", location=f"env {var}", chat_scope=scope, entry="*",
                removable=False, reason="wildcard: admits every sender; never removed by revoke",
                authorizes_now=True, _path=(var,), _platform=platform, _user_id=user_id))
        for entry in live_entries:
            if _unresolved(platform, entry, user_id):
                sources.append(GrantSource(
                    kind="unresolved", location=f"env {var}", chat_scope=scope, entry=entry,
                    removable=False, reason=_USERNAME_REASON, authorizes_now=None,
                    _path=(var,), _platform=platform, _user_id=user_id))

    if _truthy(allow_all):
        sources.append(GrantSource(
            kind="allow_all", location=f"env {allow_all}", chat_scope=GLOBAL, entry="true",
            removable=False, reason="allow-all flag: admits every sender; never removed by revoke",
            authorizes_now=True, _path=(allow_all,), _platform=platform, _user_id=user_id))
    if _truthy("GATEWAY_ALLOW_ALL_USERS"):
        sources.append(GrantSource(
            kind="allow_all", location="env GATEWAY_ALLOW_ALL_USERS", chat_scope=GLOBAL, entry="true",
            removable=False,
            reason=("allow-all flag: admits every sender on every platform (consulted only while no "
                    "env allowlist is configured); never removed by revoke"),
            authorizes_now=yaml_only_env, _path=("GATEWAY_ALLOW_ALL_USERS",),
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


def _yaml_sources(platform: str, user_id: str, config: dict, yaml_only_env: bool) -> list[GrantSource]:
    """Authored ``config.yaml`` blocks, key by key, with the loader's own reachability."""
    effective = _effective_extra(config, platform)
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
                platform, user_id, container, container_path, effective, yaml_only_env))
    return sources


def _yaml_container_sources(
    platform: str, user_id: str, container: dict, container_path: tuple[str, ...],
    effective: dict, yaml_only_env: bool,
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
        scope_value = scope or _scope_for(platform)
        for entry in entries:
            if not _matches(platform, entry, user_id):
                continue
            verbatim = _authorizes_entry(platform, entry, user_id)
            authorizes = bool(yaml_only_env and verbatim and str(entry).strip() in effective_set)
            if authority:
                reason = "command-authority list, not access: reported, never removed"
            elif not authorizes and not verbatim:
                reason = _MENTION_REASON
            elif authorizes:
                reason = ""
            else:
                reason = _NOT_CONSULTED
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
                authorizes_now=bool(yaml_only_env and "*" in effective_set),
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
        out.append(GrantSource(
            kind="allow_all", location="config.yaml " + ".".join(path), chat_scope=GLOBAL, entry="true",
            removable=False,
            reason=("allow-all flag: admits every sender on this platform (authorizes only through the "
                    f"bridged {allow_all_env or 'env flag'})"),
            authorizes_now=bool(allow_all_env and _truthy(allow_all_env)),
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


def _strip(value: Any, platform: str, user_id: str) -> Any:
    """Drop the principal's entries, preserving the authored container type, order and ``*``."""
    if isinstance(value, str):
        return ",".join(entry for entry in _ordered_entries(value) if not _matches(platform, entry, user_id))
    if isinstance(value, (list, tuple)):
        return [entry for entry in value if not _matches(platform, str(entry), user_id)]
    if isinstance(value, (set, frozenset)):
        return {entry for entry in value if not _matches(platform, str(entry), user_id)}
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
        for source in yaml_plan:
            value = _value_at(staged, source._path)
            if value is None:
                raise AccessGrantError(
                    f"{source.location} no longer exists (the file changed since it was reported); "
                    "nothing was removed.")
            remaining = _strip(value, platform, user_id)
            if _entries(remaining) == _entries(value):
                raise AccessGrantError(
                    f"{source.location} no longer names {user_id} (the file changed since it was "
                    "reported); nothing was removed.")
            _set_at(staged, source._path, remaining)

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
                require_env_writable(var, f"revoke {user_id} from {var}")
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
