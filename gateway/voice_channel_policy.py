"""Fleet voice-channel policy (fork-local): which seat may be in which Discord voice channel, how.

One map, in the ROOT ``config.yaml`` (``get_default_hermes_root()``), under ``discord.voice_channels``::

    discord:
      voice_channels:
        unlisted: allow              # allow (default) | refuse: may a seat join a VC with no entry?
        "<voice channel id>":
          join: manual               # auto | manual | never
          bind: caller               # caller | home: where `/voice join` binds speech
          seats: "*"                 # "*" or a list of profile names allowed to connect
          allow_attach_from_text: false   # R1 opt-in: `/voice join` may attach to an auto VC
          allow_thread_move: false   # `/voice join` in another chat may move this seat's binding

Only the root may mark a VC ``auto``. A seat's own ``config.yaml`` may carry the same block, and it can
only NARROW: lower ``join`` (auto > manual > never), drop itself from ``seats``, turn an opt-in off, set
``unlisted: refuse`` or ``bind: home``. Anything wider is ignored with a warning. A seat's legacy
``discord.voice_auto_join.channel_ids`` (with ``enabled: true``) count as ``auto`` for that seat only
for VCs the root map does not list; a root entry wins over them, with a warning.

Unknown keys and values fail CLOSED: a bad entry refuses that VC; a bad ``unlisted`` value refuses
unlisted VCs; an unknown top-level key, an unreadable root config or an unreadable served profile
config refuses every join (F12, F1). A root ``join: auto`` entry without ``seats`` is such a bad entry
(it does not default to ``"*"``). Defaults: a listed entry ``join: manual``, ``bind: caller``,
``seats: "*"``, both opt-ins false (an auto-join always binds the seat's home channel; ``bind`` decides
`/voice join` only); an unlisted VC behaves as before this policy
(``join: manual``, ``bind: caller``, ``allow_thread_move: true``), apart from R1 and occupancy.

Pure logic plus one loader; no discord.py imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

JOIN_VALUES = ("auto", "manual", "never")
BIND_VALUES = ("caller", "home")
UNLISTED_VALUES = ("allow", "refuse")
ENTRY_KEYS = frozenset({"join", "bind", "seats", "allow_attach_from_text", "allow_thread_move"})
_JOIN_RANK = {"never": 0, "manual": 1, "auto": 2}
_AUTO_NEEDS_SEATS = "join=auto needs an explicit seats list (or seats: \"*\")"

# Replies (fork-local; English only, like the rest of the fork's voice-policy surface).
REPLY_R1 = "join a manual voice channel first"
REPLY_NEVER = "Voice is disabled in **{vc}** by the voice channel policy."
REPLY_SEAT = "This seat is not allowed to join **{vc}** (voice channel policy `seats`)."
REPLY_UNLISTED = "**{vc}** is not in the voice channel policy, and unlisted channels are refused."
REPLY_INVALID = "The voice channel policy is invalid or unreadable, so I won't join (see the gateway log)."
REPLY_OCCUPIED = "{seat} is in {vc}; `/voice leave` there first"
REPLY_ALREADY_BOUND = "Already in **{vc}**; speech is bound to this chat."
REPLY_NO_MOVE = ("I'm in **{vc}** with speech bound to another chat; moving it here is off for this channel "
                 "(`allow_thread_move`). Use `/voice leave` first.")
REPLY_NO_HOME = "**{vc}** binds speech to this seat's home channel, and none is configured here."


def _vc_key(vc_id: Any) -> Optional[str]:
    if isinstance(vc_id, bool):
        return None
    if isinstance(vc_id, int):
        return str(vc_id) if vc_id > 0 else None
    if isinstance(vc_id, str) and vc_id.strip().isdigit():
        return vc_id.strip()
    return None


def _validate_entry(raw: Any, *, seat_layer: bool) -> tuple[Optional[dict], Optional[str]]:
    """``(normalized entry, None)`` or ``(None, reason)``. Seat entries keep only the keys they set."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        return None, "entry must be a mapping"
    unknown = sorted(str(k) for k in raw if k not in ENTRY_KEYS)
    if unknown:
        return None, f"unknown key(s) {unknown}"
    out: dict[str, Any] = {}
    if "join" in raw:
        if raw["join"] not in JOIN_VALUES:
            return None, f"join={raw['join']!r} (expected one of {list(JOIN_VALUES)})"
        out["join"] = raw["join"]
    if "bind" in raw:
        if raw["bind"] not in BIND_VALUES:
            return None, f"bind={raw['bind']!r} (expected one of {list(BIND_VALUES)})"
        out["bind"] = raw["bind"]
    if "seats" in raw:
        seats = raw["seats"]
        if seats == "*":
            out["seats"] = "*"
        elif isinstance(seats, (list, tuple)) and all(isinstance(s, str) and s.strip() for s in seats):
            out["seats"] = frozenset(s.strip() for s in seats)
        else:
            return None, f"seats={seats!r} (expected \"*\" or a list of profile names)"
    for key in ("allow_attach_from_text", "allow_thread_move"):
        if key in raw:
            if not isinstance(raw[key], bool):
                return None, f"{key}={raw[key]!r} (expected true or false)"
            out[key] = raw[key]
    if seat_layer:
        return out, None
    join = out.get("join", "manual")
    if join == "auto" and "seats" not in out:
        # TARS ruling (F8): no "*" default for an auto entry, or every seat with auto-join on races for it.
        return None, _AUTO_NEEDS_SEATS
    return {
        "join": join,
        "bind": out.get("bind", "caller"),
        "seats": out.get("seats", "*"),
        "allow_attach_from_text": out.get("allow_attach_from_text", False),
        "allow_thread_move": out.get("allow_thread_move", False),
    }, None


@dataclass
class ParsedMap:
    entries: dict[str, dict] = field(default_factory=dict)
    unlisted: Optional[str] = None
    invalid: dict[str, str] = field(default_factory=dict)  # vc -> reason: that VC fails closed
    errors: list[str] = field(default_factory=list)        # fatal: every VC fails closed
    warnings: list[str] = field(default_factory=list)


def parse_voice_channels(raw: Any, *, where: str, seat_layer: bool = False) -> ParsedMap:
    """Validate one ``discord.voice_channels`` block (root or seat layer)."""
    parsed = ParsedMap()
    if raw is None:
        return parsed
    if not isinstance(raw, Mapping):
        parsed.errors.append(f"{where}: discord.voice_channels must be a mapping")
        return parsed
    for key, value in raw.items():
        if key == "unlisted":
            if value in UNLISTED_VALUES:
                parsed.unlisted = value
            else:
                # Fail closed: an unknown value refuses unlisted VCs rather than allowing them.
                parsed.unlisted = "refuse"
                parsed.warnings.append(f"{where}: unlisted={value!r} is not allow|refuse; refusing unlisted VCs")
            continue
        vc = _vc_key(key)
        if vc is None:
            parsed.errors.append(f"{where}: unknown key {key!r} in discord.voice_channels")
            continue
        entry, reason = _validate_entry(value, seat_layer=seat_layer)
        if entry is None:
            parsed.invalid[vc] = f"{where}: VC {vc}: {reason}"
            if reason == _AUTO_NEEDS_SEATS:
                parsed.warnings.append(f"{where}: VC {vc}: {reason}; refusing that VC (config error)")
        else:
            parsed.entries[vc] = entry
    return parsed


def legacy_auto_join_ids(cfg: Mapping) -> frozenset[str]:
    """``discord.voice_auto_join.channel_ids`` when ``enabled`` — the seat's legacy auto VCs."""
    raw = ((cfg.get("discord") or {}) if isinstance(cfg, Mapping) else {}).get("voice_auto_join") or {}
    if not isinstance(raw, Mapping) or not raw.get("enabled"):
        return frozenset()
    ids = raw.get("channel_ids") or []
    if isinstance(ids, (str, int)):
        ids = [ids]
    if not isinstance(ids, (list, tuple)):
        return frozenset()
    return frozenset(k for k in (_vc_key(i) for i in ids) if k)


def _voice_channels_block(cfg: Mapping) -> Any:
    discord_cfg = cfg.get("discord") if isinstance(cfg, Mapping) else None
    return discord_cfg.get("voice_channels") if isinstance(discord_cfg, Mapping) else None


@dataclass(frozen=True)
class VoiceChannelRule:
    """The resolved policy of one VC for one seat."""

    vc_id: Optional[str]
    listed: bool
    join: str
    bind: str
    seat_allowed: bool
    allow_attach_from_text: bool
    allow_thread_move: bool
    refusal: Optional[str] = None  # "invalid" | "unlisted" | None

    @property
    def joinable(self) -> bool:
        return self.refusal is None and self.join != "never" and self.seat_allowed

    def refusal_reply(self, vc_name: str) -> Optional[str]:
        if self.refusal == "invalid":
            return REPLY_INVALID
        if self.refusal == "unlisted":
            return REPLY_UNLISTED.format(vc=vc_name)
        if self.join == "never":
            return REPLY_NEVER.format(vc=vc_name)
        if not self.seat_allowed:
            return REPLY_SEAT.format(vc=vc_name)
        return None


@dataclass(frozen=True)
class VoicePolicy:
    """The fleet map as one seat sees it, plus the fleet-wide auto set R1 needs."""

    seat: str
    root: ParsedMap
    seat_layer: ParsedMap
    legacy_auto: frozenset[str]
    fleet_auto: frozenset[str]
    errors: tuple[str, ...] = ()

    @property
    def unlisted(self) -> str:
        values = {self.root.unlisted or "allow", self.seat_layer.unlisted or "allow"}
        return "refuse" if "refuse" in values else "allow"

    def rule(self, vc_id: Any) -> VoiceChannelRule:
        vc = _vc_key(vc_id)
        if self.errors or self.root.errors or self.seat_layer.errors or (
                vc is not None and (vc in self.root.invalid or vc in self.seat_layer.invalid)):
            return VoiceChannelRule(vc, False, "never", "caller", False, False, False, refusal="invalid")
        legacy = vc is not None and vc in self.legacy_auto
        base = self.root.entries.get(vc) if vc is not None else None
        if base is None:
            if self.unlisted == "refuse":
                return VoiceChannelRule(vc, False, "never", "caller", False, False, False, refusal="unlisted")
            # A legacy voice_auto_join id counts as auto for this seat only where the root is silent.
            base = {"join": "auto" if legacy else "manual", "bind": "caller",
                    "seats": "*", "allow_attach_from_text": False, "allow_thread_move": True}
            listed = False
        else:
            base = dict(base)  # a root entry wins over the seat's legacy ids (V-Q2: only the root marks auto)
            listed = True
        seats = base["seats"]
        seat_allowed = seats == "*" or self.seat in seats
        join, bind = base["join"], base["bind"]
        attach, move = base["allow_attach_from_text"], base["allow_thread_move"]
        narrow = self.seat_layer.entries.get(vc) if vc is not None else None
        if narrow:
            if "join" in narrow and _JOIN_RANK[narrow["join"]] < _JOIN_RANK[join]:
                join = narrow["join"]
            if "seats" in narrow and narrow["seats"] != "*" and self.seat not in narrow["seats"]:
                seat_allowed = False
            if narrow.get("bind") == "home":
                bind = "home"
            attach = attach and narrow.get("allow_attach_from_text", attach)
            move = move and narrow.get("allow_thread_move", move)
        return VoiceChannelRule(vc, listed, join, bind, seat_allowed, attach, move)

    def is_auto_vc(self, vc_id: Any) -> bool:
        """R1: does ANY seat auto-join this VC (root map auto entries + every served profile's legacy
        auto-join ids)? A broken config reads as auto, so R1 fails closed."""
        if self.errors or self.root.errors:
            return True
        vc = _vc_key(vc_id)
        return vc is not None and (vc in self.fleet_auto or vc in self.root.invalid)

    def auto_join_channels(self) -> frozenset[str]:
        """The VCs THIS seat auto-joins (still gated by ``voice_auto_join`` enabled + user_ids)."""
        candidates = set(self.legacy_auto) | {vc for vc, e in self.root.entries.items() if e["join"] == "auto"}
        out = set()
        for vc in candidates:
            rule = self.rule(vc)
            if rule.joinable and rule.join == "auto":
                out.add(vc)
        return frozenset(out)

    def summary(self) -> str:
        listed = {vc for vc in self.root.entries} | set(self.seat_layer.entries)
        parts = []
        for vc in sorted(listed):
            r = self.rule(vc)
            parts.append(f"{vc}:{'refused' if not r.joinable else r.join}/bind={r.bind}"
                         f"/attach={str(r.allow_attach_from_text).lower()}/move={str(r.allow_thread_move).lower()}")
        problems = list(self.errors) + self.root.errors + self.seat_layer.errors
        problems += list(self.root.invalid.values()) + list(self.seat_layer.invalid.values())
        return (f"seat={self.seat} unlisted={self.unlisted} auto_join={sorted(self.auto_join_channels())} "
                f"fleet_auto={sorted(self.fleet_auto)} listed=[{', '.join(parts)}]"
                + (f" FAIL-CLOSED: {'; '.join(problems)}" if problems else ""))

    @property
    def problems(self) -> list[str]:
        return (list(self.errors) + self.root.errors + self.seat_layer.errors
                + list(self.root.invalid.values()) + list(self.seat_layer.invalid.values()))


def resolve_voice_policy(seat: str, root_cfg: Mapping, seat_cfg: Optional[Mapping],
                         profile_cfgs: Mapping[str, Mapping], *, seat_is_root: bool,
                         errors: Iterable[str] = ()) -> VoicePolicy:
    """Pure resolution. ``profile_cfgs`` is every served profile's config (for the R1 auto set);
    ``seat_is_root`` means the seat's own file IS the root file (no narrowing layer)."""
    root = parse_voice_channels(_voice_channels_block(root_cfg), where="root config.yaml")
    seat_layer = ParsedMap() if seat_is_root else parse_voice_channels(
        _voice_channels_block(seat_cfg or {}), where=f"profile {seat} config.yaml", seat_layer=True)
    if not seat_is_root:
        for vc, entry in seat_layer.entries.items():
            base = root.entries.get(vc)
            if entry.get("join") == "auto" and (base is None or base["join"] != "auto"):
                seat_layer.warnings.append(f"profile {seat}: VC {vc} join=auto ignored (only the root may mark auto)")
            for key in ("allow_attach_from_text", "allow_thread_move"):
                if entry.get(key) is True and base is not None and base[key] is False:
                    seat_layer.warnings.append(f"profile {seat}: VC {vc} {key}=true ignored (seats may only narrow)")
        if seat_layer.unlisted == "allow" and root.unlisted == "refuse":
            seat_layer.warnings.append(f"profile {seat}: unlisted=allow ignored (root refuses unlisted VCs)")
    seat_legacy = legacy_auto_join_ids(root_cfg if seat_is_root else (seat_cfg or {}))
    for vc in sorted(seat_legacy):
        base = root.entries.get(vc)
        if base is not None and (base["join"] != "auto" or (base["seats"] != "*" and seat not in base["seats"])):
            seat_layer.warnings.append(
                f"profile {seat}: voice_auto_join.channel_ids VC {vc} ignored (the root map lists it: "
                f"join={base['join']}, seats={'*' if base['seats'] == '*' else sorted(base['seats'])})")
    # R1's fleet auto set: root auto entries, plus legacy ids only for VCs the root does not list.
    fleet_auto = {vc for vc, e in root.entries.items() if e["join"] == "auto"}
    for cfg in profile_cfgs.values():
        fleet_auto.update(vc for vc in legacy_auto_join_ids(cfg) if vc not in root.entries)
    return VoicePolicy(seat=seat, root=root, seat_layer=seat_layer, legacy_auto=seat_legacy,
                       fleet_auto=frozenset(fleet_auto), errors=tuple(errors))


def voice_channel_occupant(voice_channel: Any, own_bot_id: Any, fleet_bots: Mapping[int, str]) -> Optional[str]:
    """The label of a bot in *voice_channel* other than this seat's own: its seat name when it is a
    fleet bot (``fleet_bots``: bot id -> seat), else its display name. Any bot counts (V-Q1: the
    operator should know a VC is in use); a fleet id counts even when the member cache misses
    ``Member.bot``. ``None`` when no other bot is there."""
    try:
        members = list(getattr(voice_channel, "members", None) or [])
    except TypeError:
        members = []
    for m in members:
        mid = getattr(m, "id", None)
        if mid is None or mid == own_bot_id:
            continue
        if mid in fleet_bots:
            return fleet_bots[mid]
        if getattr(m, "bot", None) is True:
            name = getattr(m, "display_name", None)
            return name if isinstance(name, str) and name else str(mid)
    return None


def _read_config_strict(path: Path) -> tuple[dict, Optional[str]]:
    """``({}, None)`` for a missing file; ``({}, reason)`` for one that cannot be read or parsed."""
    try:
        # The shared C-backed reader (the one config loading uses, same YAML 1.1 rules): this runs on
        # the event loop for every served profile; PyYAML's pure-Python parser cost ~4x as much there.
        from utils import fast_safe_load
        with open(path, encoding="utf-8-sig") as f:
            data = fast_safe_load(f)
    except FileNotFoundError:
        return {}, None
    except Exception as exc:  # unreadable or unparseable: fail closed
        return {}, f"{path}: {type(exc).__name__}"
    if data is None:
        return {}, None
    if not isinstance(data, dict):
        return {}, f"{path}: top level is not a mapping"
    return data, None


def load_voice_policy(seat: Optional[str]) -> VoicePolicy:
    """Read the fleet map from the ROOT config (never the process/profile home) and every served
    profile's on-disk config, independent of which seats are connected (F1). Any read failure is
    recorded as an error, which refuses every join."""
    from hermes_constants import get_default_hermes_root

    seat = (seat or "").strip() or "default"
    errors: list[str] = []
    root_dir = get_default_hermes_root()
    root_cfg, err = _read_config_strict(root_dir / "config.yaml")
    if err:
        errors.append(f"root config unreadable ({err})")
    homes: list[tuple[str, Path]] = []
    try:
        from hermes_cli.profiles import profiles_to_serve
        # Standalone seats are served too (by their own gateway) and must count for R1.
        homes = [(name, Path(home)) for name, home in profiles_to_serve(True, include_standalone=True)]
    except Exception as exc:
        errors.append(f"served profile roster unreadable ({type(exc).__name__})")
    try:
        root_resolved = root_dir.resolve()
    except OSError:
        root_resolved = root_dir
    profile_cfgs: dict[str, Mapping] = {}
    seat_home: Optional[Path] = None
    for name, home in homes:
        try:
            is_root = home.resolve() == root_resolved
        except OSError:
            is_root = False
        if is_root:
            profile_cfgs[name] = root_cfg
        else:
            cfg, err = _read_config_strict(home / "config.yaml")
            if err:
                errors.append(f"profile {name} config unreadable ({err})")
            profile_cfgs[name] = cfg
        if name == seat:
            seat_home = home
    seat_is_root = seat == "default"
    if seat_home is not None:
        try:
            seat_is_root = seat_home.resolve() == root_resolved
        except OSError:
            pass
    if seat_is_root:
        seat_cfg: Mapping = root_cfg
    elif seat in profile_cfgs:
        seat_cfg = profile_cfgs[seat]
    else:
        try:
            from hermes_cli.profiles import get_profile_dir
            seat_cfg, err = _read_config_strict(Path(get_profile_dir(seat)) / "config.yaml")
        except Exception as exc:
            seat_cfg, err = {}, type(exc).__name__
        if err:
            errors.append(f"profile {seat} config unreadable ({err})")
        profile_cfgs[seat] = seat_cfg
    policy = resolve_voice_policy(seat, root_cfg, seat_cfg, profile_cfgs, seat_is_root=seat_is_root,
                                  errors=errors)
    for warning in policy.root.warnings + policy.seat_layer.warnings:
        logger.warning("Voice channel policy: %s", warning)
    return policy
