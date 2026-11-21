"""Bot-authored inbound messages: the Discord moving-window hold, and the budget and chain rules of every other platform.

``{PLATFORM}_ALLOW_BOTS`` only decides admission, so two Hermes profiles replying to each other never stop. This
module is the gateway's answer, one guard object shared by every seat in the process (settings come from each seat's own
config.yaml ``gateway.bot_loop_guard``, re-read on every call).

Discord (loop-guard-hold): a conversation's bot messages are counted in a moving window. Past ``hold_after_messages``
in ``hold_window_seconds`` the next ones are HELD in memory, never dropped, and handed over one at a time on a backoff
(``hold_initial_seconds`` doubling to ``hold_max_seconds``); a runaway pauses until a person writes there. Any person's
message releases what is held; a person's ``/stop`` discards it. ``duplicate`` and ``reply_to_reply`` still drop, judged
when the message arrives. The window, the queues and their pace live here; the runner (gateway/run_bot_loop_hold.py)
owns the pump, the notices and the restart file.

Every other platform keeps the older rules, unchanged: ``admit`` (``max_events`` per ``window_seconds`` with a
``cooldown_seconds`` cooldown) and ``chain_admit`` (``max_hops``, ``duplicate``, ``reply_to_reply``). A human message
starts a new chain; each bot message after it is one hop and ``max_hops`` caps them; a bot repeating the same text
inside one chain is one hop, not a new ask; and with ``bot_replies_expect_reply`` false (the default) a bot answering
the reply this bot delivered to it is not answered again (#137027). A bot message that freshly @mentions this bot (not
a reply to its message) is a new ask, not such an answer.

``human_relays`` lists sender ids (e.g. a webhook bridging people from another chat into this one) the
guard treats as people although the platform flags them as bots: such a message resets the chain and is
never counted, cooled down, held or dropped. Only the guard reads it; admission (``*_ALLOW_BOTS``, bot
allowlists, mention gates) is unchanged. List only relays that never carry this bot's replies back.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Hashable, List, Optional, Tuple

from gateway.bot_loop_hold_text import format_hold_notice

__all__ = [
    "BotLoopGuard", "BotLoopGuardSettings", "HeldEntry", "NoticeJob", "configured_bot_user", "format_hold_notice",
    "load_settings", "settings_from_config",
]

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})

# The hold's own constants (not config keys, spec S4/S10/S2).
_EPS = 1e-6                  # clock comparisons: advancing a float clock in steps never lands exactly on a boundary
_RELEASED_TTL = 600.0        # an entry leaves the released set when its turn starts, or after 10 min
_RECORD_TIMEOUT = 60.0       # a hand-over whose slot is never claimed stops blocking the next one after 60 s
_CLAIMED_RECORD_TIMEOUT = 600.0  # a claimed record whose end was never reported (a crash path) is closed after 10 min
_MAX_HAND_OVERS = 3
_ASKS_KEPT = 50
_LEADING_MENTIONS = re.compile(r"^\s*(?:<@!?\d+>\s*)+")


@dataclass(frozen=True)
class BotLoopGuardSettings:
    enabled: bool = True
    max_events: int = 20
    window_seconds: float = 300.0
    cooldown_seconds: float = 600.0
    max_hops: int = 4
    bot_replies_expect_reply: bool = False
    human_relays: frozenset = frozenset()
    hold_after_messages: int = 10
    hold_window_seconds: float = 120.0
    hold_initial_seconds: float = 15.0
    hold_max_seconds: float = 480.0
    hold_runaway_at_cap: int = 3
    hold_queue_max: int = 50

    def is_human_relay(self, sender) -> bool:
        """Whether ``sender`` is a listed relay of human messages (``human_relays``)."""
        return sender is not None and str(sender).strip() in self.human_relays


def _as_bool(raw, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower() if raw is not None else ""
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def _as_positive(raw, default: float) -> float:
    if isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _as_positive_int(raw, default: int) -> int:
    value = _as_positive(raw, 0.0)
    return int(value) if value >= 1 and value == int(value) else default


_WARNED_RELAY_VALUES: set[str] = set()
_WARNED_HOLD_VALUES: set[tuple] = set()


def _as_id_set(raw) -> frozenset:
    """A list (or comma-separated string) of sender ids. Ids are strings; anything else reads as empty."""
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        if raw is not None and repr(raw) not in _WARNED_RELAY_VALUES:  # settings are re-read per message: warn once
            _WARNED_RELAY_VALUES.add(repr(raw))
            logger.warning("gateway.bot_loop_guard.human_relays must be a list of quoted ids, got %r; ignoring it "
                           "(write human_relays: ['<id>']).", raw)
        return frozenset()
    return frozenset(text for item in raw
                     if isinstance(item, (str, int)) and not isinstance(item, bool) and (text := str(item).strip()))


def _number(raw) -> Optional[float]:
    """``raw`` as a finite number, or None (a bool, a non-number string, NaN and infinity are not numbers here)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
    else:
        return None
    return value if math.isfinite(value) else None


def _warn_hold_value(key: str, raw, expected: str, using, *extra) -> None:
    """One WARNING per bad value: the settings are re-read on every message."""
    marker = (key, repr(raw)) + extra
    if marker in _WARNED_HOLD_VALUES:
        return
    _WARNED_HOLD_VALUES.add(marker)
    logger.warning("gateway.bot_loop_guard.%s must be %s, got %r; using %s.", key, expected, raw, using)


def _hold_number(block: dict, key: str, default, low: float, high: float, *, integer: bool):
    """A ``hold_*`` value inside ``[low, high]`` (an integer for the count keys), else the default and one WARNING."""
    raw = block.get(key)
    if raw is None:
        return default
    value = _number(raw)
    if value is None or not low <= value <= high or (integer and value != int(value)):
        kind = "an integer" if integer else "a number"
        _warn_hold_value(key, raw, f"{kind} from {low:g} to {high:g}", default)
        return default
    return int(value) if integer else value


def settings_from_config(cfg) -> BotLoopGuardSettings:
    """Read ``gateway.bot_loop_guard`` from a loaded config dict. Unusable values keep the default."""
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    if not isinstance(block, dict):
        return BotLoopGuardSettings()
    defaults = BotLoopGuardSettings()
    initial = _hold_number(block, "hold_initial_seconds", defaults.hold_initial_seconds, 1, 600, integer=False)
    # A max below the initial value is the bad one: it falls back to the larger of 480 and the initial value.
    fallback_max = max(defaults.hold_max_seconds, float(initial))
    raw_max = block.get("hold_max_seconds")
    hold_max = fallback_max
    if raw_max is not None:
        value = _number(raw_max)
        if value is None or not initial <= value <= 86400:
            _warn_hold_value("hold_max_seconds", raw_max, f"a number from hold_initial_seconds ({initial:g}) to 86400",
                             fallback_max, initial)
        else:
            hold_max = value
    return BotLoopGuardSettings(
        enabled=_as_bool(block.get("enabled"), defaults.enabled),
        max_events=_as_positive_int(block.get("max_events"), defaults.max_events),
        window_seconds=_as_positive(block.get("window_seconds"), defaults.window_seconds),
        cooldown_seconds=_as_positive(block.get("cooldown_seconds"), defaults.cooldown_seconds),
        max_hops=_as_positive_int(block.get("max_hops"), defaults.max_hops),
        bot_replies_expect_reply=_as_bool(block.get("bot_replies_expect_reply"), defaults.bot_replies_expect_reply),
        human_relays=_as_id_set(block.get("human_relays")),
        hold_after_messages=_hold_number(block, "hold_after_messages", defaults.hold_after_messages, 1, 1000,
                                         integer=True),
        hold_window_seconds=_hold_number(block, "hold_window_seconds", defaults.hold_window_seconds, 10, 3600,
                                         integer=False),
        hold_initial_seconds=initial,
        hold_max_seconds=hold_max,
        hold_runaway_at_cap=_hold_number(block, "hold_runaway_at_cap", defaults.hold_runaway_at_cap, 1, 100,
                                         integer=True),
        hold_queue_max=_hold_number(block, "hold_queue_max", defaults.hold_queue_max, 1, 10000, integer=True),
    )


def configured_bot_user(platform_config, user_id) -> bool:
    """Whether ``user_id`` is in the adapter's ``bot_users`` list (config.yaml platform ``extra``): the
    opt-in that marks a peer bot on a platform whose events carry no bot flag (Mattermost, Matrix,
    Signal, IRC, ...), so this guard and the bot-peer delivery rules engage for it (#137027)."""
    if user_id is None or user_id == "":
        return False
    configured = (getattr(platform_config, "extra", None) or {}).get("bot_users")
    if isinstance(configured, str):
        configured = configured.split(",")
    return isinstance(configured, (list, tuple, set)) and str(user_id) in {str(v).strip() for v in configured}


def load_settings() -> BotLoopGuardSettings:
    """Settings from the live config.yaml. Defaults when the config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly

        return settings_from_config(load_config_readonly())
    except Exception:
        return BotLoopGuardSettings()


class HeldEntry:
    """One held Discord bot message: the adapter's finished ``MessageEvent`` itself, kept in memory as is."""

    __slots__ = (
        "arrived",
        "ask",
        "attempts",
        "conversation",
        "discarded",
        "event",
        "immediate",
        "person_released",
        "ref",
        "seat",
        "wall",
        "warned",
    )

    def __init__(self, event, seat: str, conversation: Hashable, arrived: float, ref: str, ask) -> None:
        self.event = event
        self.seat = seat
        self.conversation = conversation
        self.arrived = arrived          # guard clock, for ages and the timer
        self.wall = time.time()         # wall clock, for the restart file
        self.ref = ref
        self.ask = ask
        self.person_released = False    # a person wrote: no timer, goes by the pace rule alone
        self.immediate = False          # back at the head after a failed hand-over: due at the next wake
        self.attempts = 0
        self.discarded = False
        self.warned = False             # "no live adapter" is warned once per episode

    @property
    def source(self):
        return self.event.source

    @property
    def message_id(self) -> str:
        return str(getattr(self.event, "message_id", "") or "")

    def info(self) -> dict[str, Any]:
        """What the restart file, the discard line and the lost line say about this message (never its text)."""
        source = self.event.source
        return {
            "ref": self.ref,
            "profile": self.seat,
            "channel": str(getattr(source, "chat_id", "") or ""),
            "message": self.message_id or "voice",
            "sender_id": str(getattr(source, "user_id", "") or ""),
            "sender_name": str(getattr(source, "user_name", "") or getattr(source, "user_id", "") or ""),
            "guild": str(getattr(source, "guild_id", "") or ""),
            "arrived": self.wall,
        }


@dataclass
class NoticeJob:
    """One chat line the runner still has to post: ``kind`` is hold, pause, over, discard or lost."""

    kind: str
    conversation: Hashable
    seat: str
    source: Any
    entries: list[dict[str, Any]]
    attempts: int = 0
    lost: Optional[HeldEntry] = None
    sent: int = 0    # messages of this job already posted (a retry resumes after them)


class _Hold:
    """One conversation's window, queue and backoff."""

    __slots__ = (
        "capped",
        "inflight",
        "last_held",
        "level",
        "notice_hold",
        "notice_over",
        "notice_pause",
        "paused",
        "queue",
        "timer_base",
        "window",
    )

    def __init__(self) -> None:
        self.window: deque[float] = deque()
        self.queue: list[HeldEntry] = []
        self.inflight: list[HeldEntry] = []
        self.level = 1
        self.timer_base: Optional[float] = None
        self.last_held = 0.0
        self.capped = 0
        self.paused = False
        self.notice_hold = self.notice_pause = self.notice_over = False

    def timer_waiting(self) -> bool:
        return any(not entry.person_released for entry in self.queue)


class _Record:
    """The open turn record of a hand-over (S4): bound to the handed-over event itself."""

    __slots__ = ("claimed_at", "entry", "opened", "seat")

    def __init__(self, entry: HeldEntry, seat: str, opened: float) -> None:
        self.entry = entry
        self.seat = seat
        self.opened = opened
        self.claimed_at: Optional[float] = None


def _event_key(seat: str, event) -> Optional[tuple]:
    mid = getattr(event, "message_id", None)
    if mid in (None, ""):
        return None
    return (seat or "", str(getattr(event.source, "chat_id", "") or ""), str(mid))


class BotLoopGuard:
    """Per-conversation sliding window with a cooldown once the budget trips (non-Discord), and the Discord hold.
    Thread-safe. Settings are re-read on every call so a config.yaml edit takes effect without a restart."""

    def __init__(
        self,
        settings: Callable[[], BotLoopGuardSettings] = load_settings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._events: dict[Hashable, deque[float]] = {}
        self._cooldown_until: dict[Hashable, float] = {}
        self._last_sweep = 0.0
        self._last_chain_sweep = 0.0
        self._last_hold_sweep = 0.0
        self._chains: dict[Hashable, _Chain] = {}
        # The Discord hold.
        self._holds: dict[Hashable, _Hold] = {}
        self._released: dict[tuple, float] = {}
        self._released_sources: list[tuple[str, Any, float]] = []
        self._records: dict[Hashable, _Record] = {}
        self._jobs: list[NoticeJob] = []
        self._lost: list[HeldEntry] = []
        self._refs: dict[str, bool] = {"dirty": False}
        self._ref_numbers = itertools.count(1)

    @property
    def tracked_conversations(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def tracking_chains(self) -> bool:
        """Whether any bot exchange is open, so a human message has a chain to reset."""
        return bool(self._chains)

    def counts_as_bot(self, sender, is_bot: bool) -> bool:
        """Whether this guard handles a message from ``sender`` as bot traffic: the platform's bot flag,
        unless ``sender`` is listed in ``human_relays``. Every guard call site asks this first."""
        return bool(is_bot) and not self._settings().is_human_relay(sender)

    def blocked(self, conversation: Hashable) -> bool:
        """True while ``conversation`` is cooling down. Reads only, so callers may ask as often as they like."""
        if not self._settings().enabled:
            return False
        with self._lock:
            return self._cooldown_until.get(conversation, 0.0) > self._clock()

    def admit(self, conversation: Hashable) -> tuple[bool, str]:
        """Count one admitted bot-authored message for ``conversation`` (the non-Discord budget).
        Returns ``(allowed, state)``; state is ``disabled``, ``ok``, ``tripped`` (this message started the cooldown) or ``cooldown``."""
        settings = self._settings()
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        with self._lock:
            self._sweep(now, settings)
            if self._cooldown_until.get(conversation, 0.0) > now:
                return False, "cooldown"
            events = self._events.setdefault(conversation, deque())
            cutoff = now - settings.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= settings.max_events:
                self._cooldown_until[conversation] = now + settings.cooldown_seconds
                events.clear()
                return False, "tripped"
            events.append(now)
            return True, "ok"

    def chain_admit(
        self, conversation: Hashable, sender: str, text: str, *, is_bot: bool, fresh_mention: bool = False,
    ) -> tuple[bool, str]:
        """Apply the chain rules to one admitted inbound message (every platform but Discord, whose messages go
        through ``hold_admit``). Returns ``(allowed, state)``; state is
        ``disabled``, ``human`` (chain reset), ``ok``, ``max_hops``, ``duplicate`` or ``reply_to_reply``.
        ``fresh_mention``: the message newly @mentions this bot and is not a reply to its message, so it
        is a new ask and never ``reply_to_reply`` (``replied_to`` is keyed on the sender bot, not on one
        of its sessions); it still counts as a hop and against ``duplicate``."""
        settings = self._settings()
        if not settings.enabled:
            return True, "disabled"
        now = self._clock()
        with self._lock:
            self._sweep(now, settings)
            if not is_bot:
                self._chains.pop(conversation, None)
                return True, "human"
            chain = self._chains.get(conversation)
            if chain is None or now - chain.updated > settings.cooldown_seconds:
                chain = self._chains[conversation] = _Chain()
            chain.updated = now
            ask = (sender, hashlib.sha256(" ".join(str(text or "").split()).encode()).hexdigest())
            if ask in chain.asks:
                return False, "duplicate"
            if not settings.bot_replies_expect_reply and chain.replied_to == sender and not fresh_mention:
                return False, "reply_to_reply"
            if chain.hops >= settings.max_hops:
                return False, "max_hops"
            chain.hops += 1
            chain.asks[ask] = False
            return True, "ok"

    def note_reply(self, conversation: Hashable, recipient: str) -> None:
        """Record that this bot's reply to bot ``recipient`` went out. Only a delivered reply makes the
        recipient's next message an answer to an answer: a message still waiting for its turn, or one
        the turn chose to leave unanswered, does not."""
        if not self._settings().enabled:
            return
        with self._lock:
            chain = self._chains.get(conversation)
            if chain is not None:
                chain.replied_to = recipient

    # ------------------------------------------------------------------------------------------------------------
    # The Discord hold (loop-guard-hold). Every method below that takes ``self._lock`` does its work and returns:
    # nothing is awaited and no callback leaves the guard under the lock except ``take_due``'s ``can_start``, a quick
    # synchronous check of the runner's own state.
    # ------------------------------------------------------------------------------------------------------------

    def hold_admit(self, conversation: Hashable, seat: str, event, *, fresh_mention: bool = False) -> str:
        """Judge one Discord bot message the moment it arrives. Returns ``disabled``, ``released`` (already handed
        over by the hold: not judged again), ``duplicate`` or ``reply_to_reply`` (dropped), ``held`` (queued) or ``ok``
        (admitted and counted)."""
        settings = self._settings()
        if not settings.enabled:
            return "disabled"
        now = self._clock()
        source = event.source
        sender = str(getattr(source, "user_id", "") or "")
        with self._lock:
            self._sweep(now, settings)
            self._prune_released(now)
            if self._is_released_locked(seat, event):
                return "released"
            chain = self._chain_for_hold(conversation, now, settings)
            ask = (sender, hashlib.sha256(" ".join(str(event.text or "").split()).encode()).hexdigest())
            if ask in chain.asks:
                return "duplicate"
            if not settings.bot_replies_expect_reply and chain.replied_to == sender and not fresh_mention:
                return "reply_to_reply"
            hold = self._holds.get(conversation)
            if hold is None:
                hold = self._holds[conversation] = _Hold()
            self._refresh_hold(hold, now, settings)
            if hold.paused or hold.timer_waiting() or len(hold.window) >= settings.hold_after_messages:
                self._hold_locked(hold, conversation, seat, event, ask, chain, now, settings)
                return "held"
            hold.window.append(now)
            self._remember_ask(chain, ask, held=False)
            return "ok"

    def _hold_locked(self, hold, conversation, seat, event, ask, chain, now, settings) -> HeldEntry:
        entry = HeldEntry(event, seat, conversation, now, f"{int(time.time())}-{next(self._ref_numbers)}", ask)
        if not hold.queue:
            hold.timer_base = now
        hold.queue.append(entry)
        hold.last_held = now
        self._remember_ask(chain, ask, held=True)
        self._refs["dirty"] = True
        source = event.source
        logger.info(
            "Bot loop guard holding bot messages in %s chat %s: message %s from %s (release in %ss, %d held; "
            "gateway.bot_loop_guard).", getattr(getattr(source, "platform", None), "value", ""),
            getattr(source, "chat_id", ""), entry.message_id or "voice", getattr(source, "user_id", ""),
            f"{self._wait(hold.level, settings)[0]:g}", len(hold.queue))
        if not hold.notice_hold:
            hold.notice_hold = True
            self._jobs.append(NoticeJob("hold", conversation, seat, source, []))
        if len(hold.queue) > settings.hold_queue_max and not hold.notice_over:
            hold.notice_over = True
            logger.warning(
                "Bot loop guard queue over hold_queue_max (%d) in %s chat %s: %d held messages are still queued "
                "(gateway.bot_loop_guard).", settings.hold_queue_max,
                getattr(getattr(source, "platform", None), "value", ""), getattr(source, "chat_id", ""),
                len(hold.queue))
            self._jobs.append(NoticeJob("over", conversation, seat, source, []))
        return entry

    def held_events(self, conversation: Hashable) -> list:
        """The held events of ``conversation``, oldest first: the very objects held."""
        with self._lock:
            hold = self._holds.get(conversation)
            return [entry.event for entry in hold.queue] if hold else []

    def hold_level(self, conversation: Hashable) -> int:
        """The backoff level: 1 when nothing is backing off (S3)."""
        settings = self._settings()
        with self._lock:
            hold = self._holds.get(conversation)
            if hold is None:
                return 1
            self._refresh_hold(hold, self._clock(), settings)
            return hold.level

    def is_paused(self, conversation: Hashable) -> bool:
        with self._lock:
            hold = self._holds.get(conversation)
            return bool(hold and hold.paused)

    def is_released_event(self, seat: str, event) -> bool:
        """Whether the hold has already handed ``event`` over for ``seat`` (keyed seat, channel, message id; a voice
        utterance, which has no id, by its ``source`` object)."""
        with self._lock:
            self._prune_released(self._clock())
            return self._is_released_locked(seat, event)

    def hold_conversations(self) -> list:
        with self._lock:
            return [key for key, hold in self._holds.items() if hold.queue]

    def peek_head(self, conversation: Hashable) -> Optional[HeldEntry]:
        with self._lock:
            hold = self._holds.get(conversation)
            return hold.queue[0] if hold and hold.queue else None

    def pending_work(self) -> bool:
        """Whether the pump has anything to do: a queue, a turn record to close or a notice to post."""
        with self._lock:
            return bool(self._jobs or self._records or any(hold.queue for hold in self._holds.values()))

    def seconds_to_next_due(self) -> Optional[float]:
        """Seconds until the earliest queue head falls due (0 if one is due), None when no head has a timer."""
        settings = self._settings()
        now = self._clock()
        best: Optional[float] = None
        with self._lock:
            for hold in self._holds.values():
                if not hold.queue or hold.paused:
                    continue
                head = hold.queue[0]
                if head.person_released or head.immediate or not settings.enabled:
                    due = 0.0
                else:
                    wait, _capped = self._wait(hold.level, settings)
                    base = hold.timer_base if hold.timer_base is not None else head.arrived
                    due = max(0.0, base + wait - now)
                best = due if best is None else min(best, due)
        return best

    def take_due(self, conversation: Hashable, can_start: Callable[[HeldEntry], bool]):
        """Pop the queue head of ``conversation`` if it is due and ``can_start(head)`` agrees (no running turn, the
        runner's own view). Opens the turn record, marks the event released and does the accounting. Returns
        ``(entry, reason, age_seconds)`` or None. ``reason`` is ``timer``, ``human``, ``disabled`` or ``retry``."""
        settings = self._settings()
        now = self._clock()
        with self._lock:
            hold = self._holds.get(conversation)
            if hold is None or not hold.queue:
                return None
            self._refresh_hold(hold, now, settings)
            record = self._records.get(conversation)
            if record is not None:
                if not self._record_stale(record, now):
                    return None
                self._close_record(conversation, record, "its slot was never claimed" if record.claimed_at is None
                                   else "its end was never reported")
            head = hold.queue[0]
            enabled = settings.enabled
            if not enabled:
                hold.paused = False
            wait, at_cap = 0.0, False
            if not enabled:
                reason = "disabled"
            elif head.person_released:
                reason = "human"
            elif head.immediate:
                reason = "retry"
            else:
                reason = "timer"
                if hold.paused:
                    return None
                wait, at_cap = self._wait(hold.level, settings)
                base = hold.timer_base if hold.timer_base is not None else head.arrived
                if now + _EPS < base + wait:
                    return None
            if not can_start(head):
                return None
            hold.queue.pop(0)
            hold.inflight.append(head)
            head.immediate = False
            self._refs["dirty"] = True
            chain = self._chains.get(conversation)
            if chain is not None and head.ask in chain.asks:
                chain.asks[head.ask] = False
            if reason == "timer":
                hold.window.append(now)                  # released by the timer: counted (S1)
                hold.level += 1
                hold.last_held = now
                hold.capped = hold.capped + 1 if at_cap else 0
                if hold.capped >= settings.hold_runaway_at_cap and not hold.paused:
                    hold.paused = True
                    source = head.event.source
                    logger.warning(
                        "Bot loop guard paused bot exchange in %s chat %s (%d held; a person's message releases "
                        "it, /stop discards it; gateway.bot_loop_guard).",
                        getattr(getattr(source, "platform", None), "value", ""), getattr(source, "chat_id", ""),
                        len(hold.queue))
                    if not hold.notice_pause:
                        hold.notice_pause = True
                        self._jobs.append(NoticeJob("pause", conversation, head.seat, source, []))
            nxt = hold.queue[0] if hold.queue else None
            hold.timer_base = now if nxt is not None and not (nxt.person_released or nxt.immediate) else None
            self._mark_released(head.seat, head.event, now)
            self._records[conversation] = _Record(head, head.seat, now)
            return head, reason, max(0.0, now - head.arrived)

    def hand_over_done(self, conversation: Hashable, entry: HeldEntry) -> None:
        """``handle_message`` returned normally: the adapter owns the event now (S2)."""
        with self._lock:
            hold = self._holds.get(conversation)
            if hold is not None and entry in hold.inflight:
                hold.inflight.remove(entry)
            self._refs["dirty"] = True

    def hand_over_failed(self, conversation: Hashable, entry: HeldEntry) -> str:
        """``handle_message`` raised. Returns ``discarded`` (a person's /stop got there first: not retried),
        ``lost`` (the third failure: dropped, to be named in a lost line) or ``retry`` (back at the head)."""
        with self._lock:
            hold = self._holds.get(conversation)
            if hold is not None and entry in hold.inflight:
                hold.inflight.remove(entry)
            record = self._records.get(conversation)
            if record is not None and record.entry is entry:
                del self._records[conversation]
            self._unmark_released(entry.seat, entry.event)
            self._refs["dirty"] = True
            if entry.discarded:
                return "discarded"
            entry.attempts += 1
            if hold is None:
                hold = self._holds[conversation] = _Hold()
            if entry.attempts >= _MAX_HAND_OVERS:
                self._lost.append(entry)
                nxt = hold.queue[0] if hold.queue else None
                hold.timer_base = self._clock() if nxt is not None and not (nxt.person_released or nxt.immediate) \
                    else hold.timer_base
                return "lost"
            entry.immediate = True
            hold.queue.insert(0, entry)
            return "retry"

    def lost_sent(self, entry: HeldEntry) -> None:
        """The lost line naming ``entry`` went out: it leaves the restart file."""
        with self._lock:
            if entry in self._lost:
                self._lost.remove(entry)
            self._refs["dirty"] = True

    def human_seen(self, conversation: Hashable, text: str, source=None, seat: str = "") -> tuple[list[HeldEntry], int]:
        """A person wrote in ``conversation`` (S7): clear the window, level, pause, notice flags and asks, then release
        every held message (FIFO, one at a time by the pace rule), or DISCARD them when the text is ``/stop``.
        Returns ``(discarded entries, number released)``."""
        with self._lock:
            self._chains.pop(conversation, None)
            hold = self._holds.get(conversation)
            if hold is None:
                return [], 0
            hold.window.clear()
            hold.level = 1
            hold.capped = 0
            hold.paused = False
            hold.timer_base = None
            hold.last_held = 0.0
            hold.notice_hold = hold.notice_pause = hold.notice_over = False
            if not hold.queue and not hold.inflight:
                return [], 0
            if self.is_stop_text(text):
                discarded = list(hold.queue)
                for entry in discarded + hold.inflight:
                    entry.discarded = True
                hold.queue.clear()
                self._refs["dirty"] = True
                if discarded:
                    first = discarded[0]
                    self._jobs.append(NoticeJob("discard", conversation, first.seat, first.event.source,
                                                [entry.info() for entry in discarded]))
                return discarded, 0
            for entry in hold.queue:
                entry.person_released = True
            return [], len(hold.queue)

    @staticmethod
    def is_stop_text(text) -> bool:
        """The hold's one ``/stop`` rule (S7): strip every leading user mention, then the first word, ``/`` prefix,
        lowercased and with its ``@bot`` suffix cut, is ``stop`` (``MessageEvent.get_command``'s parse)."""
        stripped = _LEADING_MENTIONS.sub("", str(text or ""), count=1).strip()
        parts = stripped.split(maxsplit=1)
        if not parts or not parts[0].startswith("/"):
            return False
        raw = parts[0][1:].lower().split("@", 1)[0]
        return raw == "stop"

    # --- the turn record (S4): the runner calls these at slot claim and slot release -----------------------------

    def has_turn_state(self) -> bool:
        """Whether any hand-over record or released mark is open (the runner skips its claim/release calls if not)."""
        return bool(self._records or self._released or self._released_sources)

    def turn_claimed(self, seat: str, event) -> None:
        """The runner claimed the session slot for ``event``: if a hand-over is bound to this very event its record is
        claimed, and the event leaves the released set (its turn starts)."""
        if not (self._records or self._released or self._released_sources):
            return
        with self._lock:
            now = self._clock()
            for record in self._records.values():
                if record.seat == (seat or "") and self._same_event(record.entry, event):
                    record.claimed_at = now
            self._unmark_released(seat, event)

    def turn_ended(self, seat: str, event) -> None:
        """The runner released the slot ``event`` held: the record bound to it closes."""
        if not self._records:
            return
        with self._lock:
            for key, record in list(self._records.items()):
                if record.seat == (seat or "") and record.claimed_at is not None \
                        and self._same_event(record.entry, event):
                    del self._records[key]

    @staticmethod
    def _same_event(entry: HeldEntry, event) -> bool:
        if entry.event is event:
            return True
        key = _event_key(entry.seat, entry.event)
        if key is not None:
            return key == _event_key(entry.seat, event)
        return entry.event.source is getattr(event, "source", None)

    # --- notices and the restart file ---------------------------------------------------------------------------

    def drain_jobs(self) -> list[NoticeJob]:
        with self._lock:
            jobs, self._jobs = self._jobs, []
            return jobs

    def requeue_job(self, job: NoticeJob) -> None:
        with self._lock:
            self._jobs.append(job)

    def notice_context(self, job: NoticeJob) -> Optional[dict[str, Any]]:
        """What a hold, pause or over notice says right now, or None when there is nothing left to say (the queue
        emptied, or only person-released entries wait)."""
        settings = self._settings()
        with self._lock:
            hold = self._holds.get(job.conversation)
            if hold is None:
                return None
            waiting = [entry for entry in hold.queue if not entry.person_released]
            if not waiting:
                return None
            wait, _capped = self._wait(hold.level, settings)
            return {
                "senders": [str(getattr(e.event.source, "user_name", "") or getattr(e.event.source, "user_id", ""))
                            for e in waiting],
                "held": len(hold.queue),
                "delay": wait,
                "replies_to_replies": not settings.bot_replies_expect_reply,
                "limit": settings.hold_queue_max,
            }

    def refs_snapshot(self) -> list[dict[str, Any]]:
        """Every message still held, in flight or lost-but-not-yet-named, for the restart file."""
        with self._lock:
            entries: list[HeldEntry] = []
            for hold in self._holds.values():
                entries.extend(hold.inflight)
                entries.extend(hold.queue)
            entries.extend(self._lost)
            return [entry.info() for entry in sorted(entries, key=lambda e: e.arrived)]

    def refs_dirty(self) -> bool:
        with self._lock:
            return bool(self._refs["dirty"])

    def refs_written(self) -> None:
        with self._lock:
            self._refs["dirty"] = False

    # --- internals (callers hold the lock) ----------------------------------------------------------------------

    @staticmethod
    def _wait(level: int, settings: BotLoopGuardSettings) -> tuple[float, bool]:
        """``min(initial * 2**(level-1), max)`` and whether the cap bound it (S3)."""
        raw = settings.hold_initial_seconds * (2 ** min(max(level - 1, 0), 40))
        if raw >= settings.hold_max_seconds - _EPS:
            return settings.hold_max_seconds, True
        return raw, False

    def _refresh_hold(self, hold: _Hold, now: float, settings: BotLoopGuardSettings) -> None:
        cutoff = now - settings.hold_window_seconds
        while hold.window and hold.window[0] <= cutoff:
            hold.window.popleft()
        if hold.timer_waiting() or hold.paused:
            return
        # [Q2] The level returns to 1 only after one full window with nothing held.
        if hold.level > 1 and now - hold.last_held >= settings.hold_window_seconds - _EPS:
            hold.level = 1
            hold.capped = 0
        if hold.level == 1 and not hold.inflight:
            hold.notice_hold = hold.notice_pause = hold.notice_over = False  # the episode is over

    def _chain_for_hold(self, conversation: Hashable, now: float, settings: BotLoopGuardSettings) -> _Chain:
        chain = self._chains.get(conversation)
        if chain is None or now - chain.updated > settings.cooldown_seconds:
            fresh = _Chain()
            hold = self._holds.get(conversation)
            if hold is not None:  # a held message's ask is kept while it is held, not expired by cooldown_seconds
                for entry in hold.queue:
                    fresh.asks[entry.ask] = True
            self._chains[conversation] = chain = fresh
        chain.updated = now
        return chain

    @staticmethod
    def _remember_ask(chain: _Chain, ask, *, held: bool) -> None:
        """Keep the last ``_ASKS_KEPT`` unheld asks plus every held ask; the oldest unheld goes first (S10)."""
        chain.asks.pop(ask, None)
        chain.asks[ask] = held
        unheld = [key for key, is_held in chain.asks.items() if not is_held]
        for key in unheld[: max(0, len(unheld) - _ASKS_KEPT)]:
            del chain.asks[key]

    def _record_stale(self, record: _Record, now: float) -> bool:
        if record.claimed_at is None:
            return now - record.opened > _RECORD_TIMEOUT + _EPS
        return now - record.claimed_at > _CLAIMED_RECORD_TIMEOUT + _EPS

    def _close_record(self, conversation: Hashable, record: _Record, why: str) -> None:
        self._records.pop(conversation, None)
        source = record.entry.event.source
        logger.warning("Bot loop guard closed the turn record of held message %s in %s chat %s: %s "
                       "(gateway.bot_loop_guard).", record.entry.message_id or "voice",
                       getattr(getattr(source, "platform", None), "value", ""), getattr(source, "chat_id", ""), why)

    def _mark_released(self, seat: str, event, now: float) -> None:
        key = _event_key(seat, event)
        if key is not None:
            self._released[key] = now + _RELEASED_TTL
        else:
            self._released_sources.append((seat or "", event.source, now + _RELEASED_TTL))

    def _unmark_released(self, seat: str, event) -> None:
        key = _event_key(seat, event)
        if key is not None:
            self._released.pop(key, None)
        else:
            self._released_sources = [item for item in self._released_sources
                                      if not (item[0] == (seat or "") and item[1] is event.source)]

    def _is_released_locked(self, seat: str, event) -> bool:
        key = _event_key(seat, event)
        if key is not None:
            return key in self._released
        source = getattr(event, "source", None)
        return any(item[0] == (seat or "") and item[1] is source for item in self._released_sources)

    def _prune_released(self, now: float) -> None:
        if self._released:
            for key in [k for k, expires in self._released.items() if expires <= now]:
                del self._released[key]
        if self._released_sources:
            self._released_sources = [item for item in self._released_sources if item[2] > now]

    def _sweep(self, now: float, settings: BotLoopGuardSettings) -> None:
        """Drop idle conversations at most once per window so memory stays bounded."""
        if now - self._last_hold_sweep >= settings.hold_window_seconds:
            self._last_hold_sweep = now
            for key, hold in list(self._holds.items()):
                cutoff = now - settings.hold_window_seconds
                while hold.window and hold.window[0] <= cutoff:
                    hold.window.popleft()
                if (not hold.queue and not hold.inflight and not hold.paused and key not in self._records
                        and not hold.window and (hold.level == 1
                                                 or now - hold.last_held >= settings.hold_window_seconds)):
                    del self._holds[key]
        if now - self._last_sweep >= settings.window_seconds:
            self._last_sweep = now
            idle_cutoff = now - settings.window_seconds
            for key in [k for k, dq in self._events.items() if not dq or dq[-1] <= idle_cutoff]:
                del self._events[key]
            for key in [k for k, until in self._cooldown_until.items() if until <= now]:
                del self._cooldown_until[key]
        # An idle chain (no message for cooldown_seconds) is forgotten, so ``tracking_chains`` does not stay true for an
        # exchange that ended; a held message's ask outlives that: ``_chain_for_hold`` rebuilds it from the queue.
        if now - self._last_chain_sweep >= min(settings.window_seconds, settings.cooldown_seconds):
            self._last_chain_sweep = now
            for key in [k for k, chain in self._chains.items() if now - chain.updated > settings.cooldown_seconds]:
                del self._chains[key]


class _Chain:
    """One bot-to-bot exchange since the last human message in a conversation."""
    __slots__ = ("asks", "hops", "replied_to", "updated")

    def __init__(self) -> None:
        self.hops = 0
        self.asks: dict[tuple[str, str], bool] = {}  # ask -> still held (insertion order = age)
        self.replied_to: str = ""
        self.updated = 0.0
