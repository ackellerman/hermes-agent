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
``None``.

**Coverage is structural, not a list of known keys** (round 3): after the modelled sources are
built, :func:`_name_scan_sources` walks every authored string in ``config.yaml`` (all keys, nested,
plugin sections included) and every value in the profile ``.env``. Any value that names the
principal at a path no modelled source already covers becomes an ``unresolved`` row. So a surface
added tomorrow cannot silently produce exit 0: either it is modelled (and its row carries a verdict)
or the net reports it. Alongside the net, grants that name a *chat* rather than a person
(``group_allowed_chats``, ``TELEGRAM_GROUP_ALLOWED_CHATS``, ``QQ_GROUP_ALLOWED_USERS``,
``allowed_chats``, an own-policy adapter's ``group_allow_from``) get a non-removable
``chat_grant`` row: they admit every member of that chat and no pairing revoke can remove a person
from a chat. Per-group *sender* lists (``groups.<chat>.allow_from`` / ``allowFrom``) are modelled
as removable group-scoped rows — and a list emptied of its last entry is reported as a chat grant,
because the adapter reads an empty list as "no sender restriction", not as "deny everyone".

Reads are side-effect free: the YAML view is a throwaway ``platforms_data`` filled by the loader's
own ``bridge_platform_shared_keys``, so no process env is written and no plugin hook runs. Writes
(``remove_user_grants``) go through ``hermes_cli.config``'s guarded primitives: validate everything
first, write ``config.yaml``, then ``.env``, then purge in-process adapter snapshots — a refusal
raises ``AccessGrantError`` before the first byte changes.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DM = "dm"
# Group SENDER lists are consulted for group and forum traffic only — authz reads them under
# ``is_group_or_forum`` — so the label must not claim channel traffic it does not cover (round-3 nit).
GROUP = "group+forum"
# Group CHAT allowlists admit every member of an allowlisted chat, channel broadcasts included
# (``_chat_scoped_grant`` tests ``is_group``).
CHAT = "group/channel"
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

# Chat allowlists (``allowed_chats`` is Telegram's DM/guest chat list): they name a CHAT, so they
# admit every member of it. Reported, never removed — see the module docstring.
_CHAT_ALLOW_KEYS: tuple[str, ...] = ("group_allowed_chats", "allowed_chats")

# Per-chat sender lists (WeCom documents ``groups: {<chat>: {allow_from: [...]}}``).
_PER_GROUP_KEYS: tuple[str, ...] = ("allow_from", "allowFrom")

# Own-policy hosts (``OwnAccessPolicyMixin``'s host list): for these ``extra.group_allow_from`` is
# read by ``OwnAccessPolicyMixin._is_group_allowed`` as a group CHAT allowlist, not a sender list.
_OWN_POLICY_PLATFORMS = frozenset({"wecom", "weixin", "qqbot", "whatsapp", "whatsapp_cloud", "yuanbao"})

# Audited platforms where ``extra.group_allow_from`` is a group SENDER list (authz gates them via
# ``_adapter_extra_allowlist_authorizes``). Anything else is undecidable offline and reported
# ``unresolved`` — the fail-closed direction.
_SENDER_GROUP_ALLOW_FROM = frozenset({"discord", "telegram", "slack"})

# Every place a platform block may live; the loader merges all three (``merge_platform_sections``
# / ``platform_section`` in ``gateway/config_loader.py``).
_BLOCK_PREFIXES: tuple[tuple[str, ...], ...] = (
    (),
    ("platforms",),
    ("gateway", "platforms"),
)

_REPORTED_FIELDS = ("kind", "location", "chat_scope", "entry", "removable", "reason", "authorizes_now")

# Kinds whose ``entry`` is a value matched from an allowlist (safe to show). Everything else blanks
# it: the name-scan walk sees every authored scalar, including credentials that happen to contain the
# principal's string, and a report (or the dashboard JSON) must not echo an unrelated secret.
_ENTRY_KINDS = frozenset({"pairing", "env", "yaml"})

# Phone-in-suffix spellings a WhatsApp principal is stored as. Treated as ``numeric`` so a JID entry
# is compared to a bare phone id instead of being screened out as an email shape.
_WHATSAPP_JID_SUFFIXES = ("@s.whatsapp.net", "@c.us", "@g.us")

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
# The structural net: a value naming the user that no modelled source covers.
_UNMODELLED_REASON = ("this value names the user and this command does not model what reads it")
_CHAT_GRANT_REASON = (
    "chat allowlist: every sender in chat {chat} is admitted; a pairing revoke removes a person, "
    "not a chat")
_CHAT_GRANT_EXTRA = {
    "allowed_chats": " (also a DM/guest chat list on Telegram)",
}
_OWN_GROUP_CHAT_REASON = (
    "this adapter enforces its own access policy, so `group_allow_from` is a group CHAT allowlist: "
    "chat {chat} admits every member; a pairing revoke cannot remove a person from a chat")
_UNKNOWN_GROUP_SCOPE_REASON = (
    "cannot be decided offline: `group_allow_from` is a group sender list for most adapters but a "
    "group CHAT allowlist for own-policy adapters (WeCom, Weixin, QQBot, WhatsApp, Yuanbao), and "
    "this platform is neither — either reading admits this entry")
_PER_GROUP_REASON = (
    "per-group sender list for chat {chat}: it names this user, but whether it reaches authz "
    "depends on the platform env-allowlist shape and the loader's merge — cannot be decided offline")
_EMPTY_GROUP_REASON = (
    "chat {chat} has no sender entries left in `groups` — the adapter treats an empty allow_from as "
    "no restriction, so every sender in that chat is admitted; removing a person from an empty list "
    "denies nobody")


class AccessGrantError(RuntimeError):
    """A grant removal was refused before any write."""


class AccessGrantWriteError(AccessGrantError):
    """A write failed; preceding writes may already have changed their targets."""


@dataclass(frozen=True)
class GrantSource:
    """One place that names the principal.

    ``_path`` is the authored location (dict keys inside ``config.yaml``, or the env var name) and
    ``_platform`` / ``_user_id`` carry the principal — internal provenance for
    :func:`remove_user_grants`, never part of the reported surface (:func:`describe`).
    """

    kind: str            # "pairing" | "env" | "yaml" | "wildcard" | "allow_all" | "role"
                         # | "unresolved" | "chat_grant"
    location: str        # "config.yaml platforms.discord.allow_from", ".env DISCORD_ALLOWED_USERS", ...
    chat_scope: str      # "dm" | "group+forum" | "group/channel" | "global" | "global(dm+channel)"
    entry: str           # matched entry, e.g. "1002" or "*"
    removable: bool
    reason: str = ""
    authorizes_now: Optional[bool] = None
    _path: tuple[str, ...] = ()
    _platform: str = ""
    _user_id: str = ""


def describe(sources: Iterable[GrantSource]) -> list[dict]:
    """JSON-ready rows for a CLI/dashboard surface.

    ``entry`` is blanked for every kind whose entry was not matched from an allowlist: the name-scan
    rows walk all authored scalars, so an unrelated credential can contain the principal's string.
    A report must never hand a caller a value it did not ask for.
    """
    rows = []
    for source in sources:
        row = {name: getattr(source, name) for name in _REPORTED_FIELDS}
        if source.kind not in _ENTRY_KINDS:
            row["entry"] = ""
        rows.append(row)
    return rows


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


_HEX64 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _shape_class(entry: Any) -> str:
    """The *form* of an allowlist entry: comparable shapes must agree to be judged offline.

    ``<@1002>`` is a numeric id in a mention wrapper, and ``+1555…`` is a phone number, so both
    classify as ``numeric`` — otherwise a mention or a WhatsApp phone spelling of a DIFFERENT
    principal would read as "cannot be evaluated" and add exit-3 noise to decidable configs.
    """
    text = str(entry or "").strip()
    if not text:
        return ""
    if _is_wildcard(text):
        return "wildcard"
    if _mention_id(text) is not None or text.lstrip("+").isdigit():
        return "numeric"
    lowered = text.lower()
    # A Baileys/Cloud WhatsApp JID is a phone number in a suffix wrapper, and the phone is the same
    # principal a bare-id revoke carries — classifying it as ``email`` hid the alias entirely.
    if any(lowered.endswith(suffix) for suffix in _WHATSAPP_JID_SUFFIXES):
        return "numeric"
    if "@" in text:
        return "email"
    if text.lower().startswith("npub1"):
        return "npub"
    if _HEX64.match(text):
        return "hex64"
    return "opaque"


def _unresolved(platform: str, entry: Any, user_id: str) -> bool:
    """An entry whose *shape* cannot be compared to this principal offline.

    Discord resolves username entries through the API at connect time; any plugin platform may
    normalise its own entry forms. So an entry that neither the predicate nor the mention rule could
    decide, and whose form differs from the principal's, is reported as cannot-be-evaluated on
    EVERY platform — exit 3, never a silent 0. The rule is shape-based, not ``isdigit()``-gated, so
    a non-numeric platform (Signal contact ids, Buzz hex, SimpleX, email) gets the same net (F5).
    """
    entry_shape, user_shape = _shape_class(entry), _shape_class(user_id)
    if entry_shape == "wildcard":
        return False  # already reported as its own wildcard row
    return bool(entry_shape) and bool(user_shape) and entry_shape != user_shape


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


def _group_allow_from_reading(platform: str) -> str:
    """What ``extra.group_allow_from`` means here: ``sender`` | ``chat`` | ``unknown``.

    authz reads the same key twice: as a group SENDER list (``_adapter_extra_allowlist_authorizes``)
    for ordinary adapters, and as a group CHAT allowlist (``OwnAccessPolicyMixin._is_group_allowed``
    through ``_own_policy_authorizes``) for the hosts that gate access themselves. The CLI never
    loads adapters, so the platform decides: an own-policy host is a chat list, an audited sender
    host is a sender list, and anything else is undecidable — fail closed.
    """
    if platform in _OWN_POLICY_PLATFORMS:
        return "chat"
    if platform in _SENDER_GROUP_ALLOW_FROM:
        return "sender"
    return "unknown"


def _chat_grant(
    platform: str, user_id: str, location: str, path: tuple[str, ...], chat: str, key: str,
    *, reason: str = "",
) -> GrantSource:
    """A grant that names a CHAT: every member is admitted, and no revoke can remove one person."""
    return GrantSource(
        kind="chat_grant", location=location, chat_scope=CHAT, entry=str(chat), removable=False,
        reason=reason or _CHAT_GRANT_REASON.format(chat=chat) + _CHAT_GRANT_EXTRA.get(key, ""),
        authorizes_now=None, _path=path, _platform=platform, _user_id=user_id)


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
    # The structural net: whatever the modelled sources did not account for, the authored values
    # themselves are walked. This is what makes exit 0 a promise about the FILE, not about the set
    # of keys this module happens to know today.
    sources.extend(_name_scan_sources(platform, user_id, config, resolved_home, sources))
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

    # Grants that name a CHAT: an allowlisted chat admits every sender in it, which no pairing
    # revoke can undo for one person (F2). Reported, never removed.
    group_chat = _group_envs(platform)[1]
    if group_chat:
        from_env_file = str(authored.get(group_chat) or "").strip()
        location = _env_location(group_chat, from_env_file, bridged_env_var(group_chat))
        chats = sorted(set(_entries(_gate_env(group_chat))) | set(_entries(from_env_file)))
        sources.extend(
            _chat_grant(platform, user_id, location, (group_chat,), chat, group_chat)
            for chat in chats)

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
        # Authored entries count here too: an entry the profile declares but this process does not
        # consult is still a shape a restart will read (F5 — a foreign-shaped entry must not vanish
        # just because it is not in this process's env).
        for entry in sorted(set(live_entries) | set(_entries(str(authored.get(var) or "")))):
            if _unresolved(platform, entry, user_id):
                sources.append(GrantSource(
                    kind="unresolved", location=var_location, chat_scope=scope, entry=entry,
                    removable=False, reason=_USERNAME_REASON, authorizes_now=None,
                    _path=(var,), _platform=platform, _user_id=user_id))

    if _truthy(allow_all) or str(authored.get(allow_all) or "").strip().lower() in {"true", "1", "yes"}:
        live = _truthy(allow_all)
        sources.append(GrantSource(
            kind="allow_all", location=f"{'env' if live else '.env'} {allow_all}", chat_scope=GLOBAL, entry="true",
            removable=False, reason="allow-all flag: admits every sender; never removed by revoke",
            authorizes_now=True if live else None, _path=(allow_all,), _platform=platform, _user_id=user_id))
    if _truthy("GATEWAY_ALLOW_ALL_USERS") or str(authored.get("GATEWAY_ALLOW_ALL_USERS") or "").strip().lower() in {"true", "1", "yes"}:
        # Consulted only in the no-env-allowlist branch, and only in a context that allowlist does
        # not cover — so with any env allowlist configured the reachability is unknown, never False.
        live = _truthy("GATEWAY_ALLOW_ALL_USERS")
        consulted = live and not _env_allowlist_configured(platform, home)
        sources.append(GrantSource(
            kind="allow_all", location=f"{'env' if live else '.env'} GATEWAY_ALLOW_ALL_USERS", chat_scope=GLOBAL, entry="true",
            removable=False,
            reason=("allow-all flag: admits every sender on every platform (consulted only while no "
                    "env allowlist is configured); never removed by revoke"),
            authorizes_now=True if consulted else None, _path=("GATEWAY_ALLOW_ALL_USERS",),
            _platform=platform, _user_id=user_id))


    roles_env = _roles_env(platform)
    roles = _gate_env(roles_env) or str(authored.get(roles_env) or "")
    if roles:
        sources.append(GrantSource(
            kind="role", location=f"{'env' if _gate_env(roles_env) else '.env'} {roles_env}", chat_scope=GLOBAL, entry=",".join(_entries(roles)),
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
    # Inspect both authored locations even if precedence or profile scoping makes one inert.
    for path, container in (((), config), (("gateway",), config.get("gateway"))):
        if isinstance(container, dict) and str(container.get("allow_all_users")).strip().lower() in {"true", "1", "yes"}:
            flag_path = (*path, "allow_all_users")
            sources.append(GrantSource(
                kind="allow_all", location="config.yaml " + ".".join(flag_path),
                chat_scope=GLOBAL, entry="", removable=False, authorizes_now=None,
                reason="allow-all flag may admit every sender on every platform; never removed by revoke",
                _path=flag_path, _platform=platform, _user_id=user_id))
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
        raw = container.get(key)
        if isinstance(raw, dict):
            out.append(GrantSource(
                kind="unresolved", location=location, chat_scope=scope or _scope_for(platform),
                entry="", removable=False, authorizes_now=None,
                reason="allowlist is a mapping, not a list or CSV; cannot evaluate offline",
                _path=path, _platform=platform, _user_id=user_id))
            continue
        entries = _entries(raw)
        effective_set = effective_entries(key)
        authority = key.endswith("admin_from")
        # A group list speaks for GROUP traffic only; every other list covers the platform's own
        # scope (F4). ``chat_scope`` is on the row, so a group grant never reads as a DM grant.
        scope_value = scope or _scope_for(platform)
        if key == "group_allow_from":
            reading = _group_allow_from_reading(platform)
            if reading == "chat":
                # For own-policy hosts the same key is a CHAT list: each entry admits every member
                # of that chat, so it is reported as a chat grant, never as a sender list.
                out.extend(
                    _chat_grant(platform, user_id, location, path, entry, key,
                                reason=_OWN_GROUP_CHAT_REASON.format(chat=entry))
                    for entry in entries)
                continue
            if reading == "unknown":
                out.extend(
                    GrantSource(
                        kind="unresolved", location=location, chat_scope=CHAT, entry=entry,
                        removable=False, reason=_UNKNOWN_GROUP_SCOPE_REASON, authorizes_now=None,
                        _path=path, _platform=platform, _user_id=user_id)
                    for entry in entries)
                continue
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

    if str(container.get("allow_all_users")).strip().lower() in {"true", "1", "yes"}:
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

    out.extend(_chat_allow_sources(platform, user_id, container, container_path))
    out.extend(_per_group_sender_sources(
        platform, user_id, container, container_path, effective,
        yaml_only_env=yaml_only_env, loader_known=loader_known))
    seen_unresolved: set[str] = set()
    deduped = []
    for source in out:
        if source.kind == "unresolved":
            if source.location in seen_unresolved:
                continue
            seen_unresolved.add(source.location)
        deduped.append(source)
    return deduped


def _chat_allow_sources(
    platform: str, user_id: str, container: dict, container_path: tuple[str, ...],
) -> list[GrantSource]:
    """``group_allowed_chats`` / ``allowed_chats``: a chat id admits every member of that chat."""
    out: list[GrantSource] = []
    for key in _CHAT_ALLOW_KEYS:
        if key not in container:
            continue
        path = (*container_path, key)
        location = "config.yaml " + ".".join(path)
        out.extend(_chat_grant(platform, user_id, location, path, chat, key)
                   for chat in _entries(container.get(key)))
    return out


def _per_group_sender_sources(
    platform: str, user_id: str, container: dict, container_path: tuple[str, ...], effective: dict,
    *, yaml_only_env: bool, loader_known: bool,
) -> list[GrantSource]:
    """``groups.<chat>.allow_from`` / ``allowFrom``: a per-chat SENDER list (WeCom, any platform).

    One row per chat, group-scoped and REMOVABLE: ``--effective`` drops this principal from that
    chat's list and leaves the operator's other entries and file comments intact. A list that names
    another sender yields no row — that list DENIES this principal, which is the direction we want.

    An empty list is the opposite: ``_is_group_allowed`` treats ``allow_from`` as absent when it
    holds no entries, so an empty (or keyless) group config admits every sender in that chat. That
    is a chat grant, not a denial — and it is exactly the state ``--effective`` leaves behind when
    the principal was the list's only entry, so it must be reported or the removal would read as a
    success that the gateway contradicts.
    """
    groups = container.get("groups")
    if not isinstance(groups, dict):
        return []
    # ``_adapter_group_has_sender_allowlist`` reads ``config.extra["groups"]``, so only the extra
    # block is consulted; a block-level ``groups`` is authored but reaches nothing (None, not False).
    extra_container = bool(container_path) and container_path[-1] == "extra"
    # ``groups`` is an own-policy concept (WeCom); on an audited sender-list platform it is not
    # consulted, so an empty config there is not a grant and must not add noise.
    own_policy_like = _group_allow_from_reading(platform) != "sender"
    out: list[GrantSource] = []
    for chat, group_cfg in groups.items():
        if not isinstance(group_cfg, dict):
            continue
        entries_by_key = {
            key: _entries(group_cfg.get(key)) for key in _PER_GROUP_KEYS if key in group_cfg}
        for key, entries in entries_by_key.items():
            path = (*container_path, "groups", str(chat), key)
            location = "config.yaml " + ".".join((*container_path, "groups", "[redacted]", key))
            suffix = path[len(container_path):]
            effective_entries = (
                {str(entry).strip() for entry in _entries(_value_at(effective, suffix))}
                if extra_container else set())
            for entry in entries:
                if not _names_principal(platform, entry, user_id):
                    continue
                authorizes = _yaml_authorizes(
                    in_effective=str(entry).strip() in effective_entries,
                    yaml_only_env=yaml_only_env, loader_known=loader_known)
                out.append(GrantSource(
                    kind="yaml", location=location, chat_scope=GROUP, entry=entry, removable=extra_container,
                    reason="" if authorizes else _PER_GROUP_REASON.format(chat="[redacted]"),
                    authorizes_now=authorizes, _path=path, _platform=platform, _user_id=user_id))
        if own_policy_like and not any(entries_by_key.values()):
            path = (*container_path, "groups", str(chat))
            out.append(_chat_grant(
                platform, user_id, "config.yaml " + ".".join((*container_path, "groups", "[redacted]")), path, "[redacted]", "groups",
                reason=_EMPTY_GROUP_REASON.format(chat="[redacted]")))
    return out


# --------------------------------------------------------------------------- the structural net


def _iter_authored_strings(
    node: Any, path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], str]]:
    """Every authored scalar in a config mapping, with the dotted path that holds it.

    A list yields its members at the LIST's path — an allowlist is one grant at one location, which
    is also how the modelled sources address it. A mapping nested inside a list is walked with its
    index, so a plugin section that happens to be a list still reports a usable location.

    Scalars are yielded as ``str(value)`` and mapping KEYS are yielded at their parent's path: a
    grant is often written as ``ids: [1002]`` (an int), ``ids: "1002,1003"`` (CSV, split by the
    caller) or ``users: {"1002": true}`` (a key, not a value). Skipping any of those is how the net
    developed holes, and the shape screen (``_scan_match``) is what keeps the wider walk quiet.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            sub = (*path, str(key))
            if isinstance(key, (str, int)) and not isinstance(key, bool):
                yield path, str(key)  # a mapping key can itself name the principal
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (str, int, float)):
                yield sub, str(value)
            elif isinstance(value, (dict, list, tuple)):
                yield from _iter_authored_strings(value, sub)
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            if isinstance(item, bool) or item is None:
                continue
            if isinstance(item, (str, int, float)):
                yield path, str(item)
            elif isinstance(item, (dict, list, tuple)):
                yield from _iter_authored_strings(item, (*path, str(index)))


def _accounted_for(path: tuple[str, ...], modelled: Iterable[tuple[str, ...]]) -> bool:
    """Whether a modelled source already covers this authored path (its path is a prefix)."""
    return any(
        bool(model) and len(model) <= len(path) and path[:len(model)] == model for model in modelled)


_ALIAS_SHAPE_PAIRS = frozenset({
    # authz normalises these pairs to a common form before comparing
    # (WhatsApp phone ↔ JID is numeric↔numeric, Buzz npub ↔ hex is the other pair),
    # so only an entry whose form can alias the principal needs the real predicate.
    ("numeric", "numeric"), ("email", "email"), ("opaque", "opaque"),
    ("npub", "npub"), ("hex64", "hex64"), ("npub", "hex64"), ("hex64", "npub"),
})


def _scan_match(platform: str, entry: Any, user_id: str, user_shape: str) -> bool:
    """Does this authored value name the principal? Cheap screen, then authz's own predicate.

    The scan walks every string in the file, so the predicate (which builds a ``SessionSource``) is
    reached only for values whose shape can alias the principal's. A value of an unrelated form
    (a tool name, a URL, a path) cannot be compared to a numeric id by any adapter, so it costs one
    set lookup instead of a predicate call — the guarantee is unchanged, only the constant is.
    """
    if _is_wildcard(entry):
        return True
    entry_shape = _shape_class(entry)
    if not entry_shape:
        return False
    if entry_shape != user_shape and (entry_shape, user_shape) not in _ALIAS_SHAPE_PAIRS:
        return False
    return _names_principal(platform, entry, user_id)


def _env_var_owner(var: str) -> str:
    """The platform a known per-platform allowlist env var belongs to (``""`` when unknown).

    Only used to WORD a name-scan row: a value in another platform's env var still names the
    principal (so it is still reported), but the operator should be told which command removes it.
    """
    if not var:
        return ""
    from gateway.authz_mixin import _GROUP_CHAT_ENV, _GROUP_USER_ENV
    from gateway.pairing import _PLATFORM_ALLOWLIST_ENV

    owners: dict[str, str] = {}
    for plat, name in _PLATFORM_ALLOWLIST_ENV.items():
        if not name:
            continue
        owners[name] = plat
        if name.endswith("_ALLOWED_USERS"):
            stem = name[: -len("_ALLOWED_USERS")]
            owners[f"{stem}_ALLOW_ALL_USERS"] = plat
            owners[f"{stem}_ALLOWED_ROLES"] = plat
    for mapping in (_GROUP_USER_ENV, _GROUP_CHAT_ENV):
        for plat, name in mapping.items():
            if name:
                owners[name] = getattr(plat, "value", str(plat))
    return owners.get(var, "")


def _name_scan_sources(
    platform: str, user_id: str, config: dict, home: Path, sources: Iterable[GrantSource],
) -> list[GrantSource]:
    """The net: every authored value that names the principal, at a path nothing models.

    This is the round-3 inversion. Enumerating surfaces is a losing game — each new key (a plugin
    section, a channel override, a per-group list) reopens the same false exit 0 — so the promise is
    turned around: ``exit 0`` now means no authored value in ``config.yaml`` or the profile ``.env``
    names this principal unless a modelled row already reported it. One pass, deterministic order,
    text only (``config.yaml`` and ``.env`` are text; nothing is executed or imported).
    """
    modelled = [source._path for source in sources if source._path]
    rows: list[GrantSource] = []
    seen: set[tuple[str, str]] = set()
    user_shape = _shape_class(user_id)

    def named(entry: Any) -> bool:
        return _scan_match(platform, entry, user_id, user_shape)

    def add(kind: str, location: str, reason: str) -> None:
        # ``entry`` stays EMPTY on the net's rows: the walk sees every authored scalar, including
        # credentials and tokens, so carrying the raw value into a report (and the dashboard JSON)
        # would leak unrelated secrets. Mapping keys can also be secrets, so redact the path.
        if (kind, location) in seen:
            return
        seen.add((kind, location))
        rows.append(GrantSource(
            kind=kind, location=location, chat_scope=GLOBAL, entry="", removable=False,
            reason=reason, authorizes_now=None, _platform=platform, _user_id=user_id))

    for path, value in _iter_authored_strings(config):
        if _accounted_for(path, modelled):
            continue
        if not any(named(entry) for entry in _entries(value)):
            continue
        add("unresolved", "config.yaml [unmodeled path redacted]", _UNMODELLED_REASON)

    from agent.secret_scope import load_env_file

    for var, raw in load_env_file(home / ".env").items():
        if _accounted_for((var,), modelled):
            continue
        if not any(named(entry) for entry in _entries(raw)):
            continue
        owner = _env_var_owner(var)
        reason = _UNMODELLED_REASON
        if owner and owner != platform:
            reason = (f"this value names the user in {owner}'s env var and this command does not "
                      f"model what reads it; `pairing revoke {owner} …` reports and removes it")
        add("unresolved", f".env {var}", reason)

    rows.sort(key=lambda row: row.location)
    return rows


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
    Validation refusals occur before writes. A later write failure raises
    :class:`AccessGrantWriteError` and may leave completed changes in place.
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

    # A write can fail after modifying its target. Do not roll back: without a
    # cross-file lock a rollback could overwrite another writer's changes.
    removed: list[GrantSource] = []
    completed: list[str] = []
    target = "config.yaml"
    try:
        if staged is not None and yaml_plan:
            atomic_config_write(config_path, staged)
            removed.extend(yaml_plan)
            completed.extend(dict.fromkeys(s.location for s in yaml_plan))
        for var, after in env_writes:
            target = f".env {var}"
            if after:
                save_env_value(var, after)
            else:
                remove_env_value(var)
            completed.append(target)
            removed.extend(s for s in env_removed if s._path[0] == var)
    except Exception:
        done = ", ".join(completed) if completed else "none confirmed"
        raise AccessGrantWriteError(
            f"Write failed. Completed changes: {done}. "
            f"The failed target {target} may also have changed; inspect it. "
            "No rollback was attempted; concurrent edits cannot be ruled out. "
            "The pairing record was not touched; a running gateway may retain old grants.") from None

    for source in pairing_plan:
        if store is not None and store.revoke(platform, user_id):
            removed.append(source)

    if removed:
        from gateway.pairing import _sync_live_adapter_allowlist_remove

        _sync_live_adapter_allowlist_remove(platform, user_id)
    return removed
