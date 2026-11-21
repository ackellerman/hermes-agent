"""Discord loop-guard hold: the runner's half (loop-guard-hold, spec S2, S4, S6-S8, S13).

``gateway/bot_loop_guard.py`` owns the window, the queues and their pace. This mixin owns what needs the gateway:
the ingress decision for a Discord message (``_admit_discord_hold``), the single pump task that hands held messages back
to the serving adapter's ``handle_message`` one at a time, the chat notices, the restart file
(one per seat: ``<that seat's profile home>/gateway/bot_loop_hold_refs.json``) and the "lost to a restart" lines posted at startup.

Bound through the MRO of ``GatewayRunner``. Test doubles build bare runners with ``object.__new__``, so every attribute
read here has a class-level default or goes through ``getattr``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.bot_loop_guard import BotLoopGuard, HeldEntry, NoticeJob
from gateway.bot_loop_hold_text import (
    format_entry_lines, format_hold_notice, format_listing, format_over_notice, format_pause_notice,
)
from gateway.config import Platform
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")  # log-record parity with the origin module

_REFS_FILE = "bot_loop_hold_refs.json"
_REFS_MAX_ENTRIES = 500
_REFS_COALESCE_SECONDS = 1.0
_PUMP_RECHECK_SECONDS = 5.0
_NOTICE_ATTEMPTS = 3
_UNPOSTABLE = ("unknown channel", "missing access", "forbidden", "404", "403", "not found", "missing permissions")


class GatewayBotLoopHoldMixin:
    # Class-level defaults so partially constructed runners in tests do not blow up.
    _hold_loop: Optional[asyncio.AbstractEventLoop] = None
    _hold_wake_event: Optional[asyncio.Event] = None
    _hold_task: Optional[asyncio.Task] = None
    _hold_report_task: Optional[asyncio.Task] = None
    _hold_refs_ready: bool = True     # False from start() until the previous run's lost messages are reported
    _hold_refs_last_write: float = 0.0

    # --- ingress (called by ``_admit_bot_chain``) ----------------------------------------------------------------

    @staticmethod
    def _bot_loop_hold_applies(source) -> bool:
        """The hold is a Discord rule (S0): every other platform keeps the budget and the chain rules."""
        return getattr(source, "platform", None) == Platform.DISCORD

    def _admit_discord_hold(self, event, guard: BotLoopGuard) -> str:
        """One admitted Discord message, judged under its transport profile. Returns ``human`` (a person or a
        ``human_relays`` sender: reset and released what is held), or the guard's verdict for a bot message:
        ``disabled``, ``released``, ``ok`` (admitted), ``held`` (queued) or ``duplicate`` / ``reply_to_reply`` (dropped)."""
        source = event.source
        return self._under_authorization_profile(source, lambda: self._admit_discord_hold_locked(event, guard))

    def _admit_discord_hold_locked(self, event, guard: BotLoopGuard) -> str:
        source = event.source
        # A person never reads the settings: only a flagged bot needs the relay list.
        if not getattr(source, "is_bot", False) or not self._guard_counts_as_bot(source):
            self._bot_loop_human_apply(source, str(getattr(event, "text", "") or ""))
            return "human"
        seat = self._adapter_profile_for_source(source) or ""
        verdict = guard.hold_admit(self._bot_loop_guard_conversation(source), seat, event,
                                   fresh_mention=bool(getattr(event, "fresh_self_mention", False)))
        if verdict == "held":
            self._bot_loop_hold_wake()
        return verdict

    def _bot_loop_human_seen(self, source, text, *, voice: bool = False) -> None:
        """A person's message reached this gateway on Discord (S7): reset the conversation and release or discard what
        it holds. Called for an admitted person by ``_admit_bot_chain`` and by the Discord adapter's
        ``_report_human_seen`` at each drop site after its allowlist passed. A flagged bot that is not a listed relay
        releases nothing."""
        if not self._bot_loop_hold_applies(source):
            return

        def _apply() -> None:
            if getattr(source, "is_bot", False) and self._guard_counts_as_bot(source):
                return
            self._bot_loop_human_apply(source, str(text or ""))

        self._under_authorization_profile(source, _apply)

    def _bot_loop_human_apply(self, source, text: str) -> None:
        guard = self._bot_loop_guard_instance()
        conversation = self._bot_loop_guard_conversation(source)
        seat = self._adapter_profile_for_source(source) or ""
        discarded, released = guard.human_seen(conversation, text, source, seat)
        if discarded:
            logger.warning("Bot loop guard discarded %d held messages on /stop in %s chat %s (%s; "
                           "gateway.bot_loop_guard).", len(discarded),
                           source.platform.value if source.platform else "", source.chat_id,
                           ", ".join(entry.message_id or "voice" for entry in discarded))
        if discarded or released:
            self._bot_loop_hold_wake()

    # --- the turn record (S4) ------------------------------------------------------------------------------------

    def _bot_loop_hold_turn_claimed(self, event) -> None:
        """The runner claimed the session slot for ``event``."""
        guard = getattr(self, "_bot_loop_guard", None)
        if guard is None or not guard.has_turn_state():
            return
        guard.turn_claimed(self._adapter_profile_for_source(event.source) or "", event)

    def _bot_loop_hold_turn_ended(self, event) -> None:
        """The runner released the slot ``event`` held: the next held message may go."""
        guard = getattr(self, "_bot_loop_guard", None)
        if guard is None or not guard.has_turn_state():
            return
        guard.turn_ended(self._adapter_profile_for_source(event.source) or "", event)
        self._bot_loop_hold_wake()

    # --- the pump (S13) ------------------------------------------------------------------------------------------

    def _hm_estop_paused(self) -> bool:
        """Whether the global emergency stop (``hermes pause``) is on: nothing starts a new turn meanwhile."""
        try:
            from agent.estop import paused_reply
        except ImportError:
            return False
        try:
            return paused_reply() is not None
        except Exception:
            return False

    def _bot_loop_hold_wake(self) -> None:
        """Wake the pump task (from any thread); a no-op until it runs."""
        event, loop = self._hold_wake_event, self._hold_loop
        if event is None or loop is None:
            return
        try:
            loop.call_soon_threadsafe(event.set)
        except RuntimeError:  # the loop is closed
            pass

    async def _bot_loop_hold_pump_once(self) -> None:
        """One pump wake: hand over what is due (one per conversation, never over a running turn), post the pending
        notices and write the restart file when it is due."""
        guard = getattr(self, "_bot_loop_guard", None)
        if guard is None:
            return
        draining = bool(getattr(self, "_draining", False))
        if not (draining or getattr(self, "_startup_restore_in_progress", False) or self._hm_estop_paused()):
            for conversation in guard.hold_conversations():
                await self._bot_loop_hold_hand_over(guard, conversation)
        if not draining:
            await self._bot_loop_hold_send_notices(guard)
        self._bot_loop_hold_flush_refs(only_if_due=True)

    async def _bot_loop_hold_hand_over(self, guard: BotLoopGuard, conversation) -> None:
        head = guard.peek_head(conversation)
        if head is None:
            return
        source = head.event.source
        try:
            adapter = self._adapter_for_source(source)
        except Exception:
            adapter = None
        if adapter is None:
            if not head.warned:
                head.warned = True
                logger.warning("Bot loop guard cannot hand over held message %s in %s chat %s: no live adapter serves "
                               "it; it stays queued (gateway.bot_loop_guard).", head.message_id or "voice",
                               source.platform.value if source.platform else "", source.chat_id)
            return

        def can_start(entry: HeldEntry) -> bool:
            key = self._session_key_for_source(entry.event.source)
            if self._is_session_running(key):
                return False
            active = getattr(adapter, "_active_sessions", None)
            return not (active is not None and key in active)

        taken = self._under_authorization_profile(source, lambda: guard.take_due(conversation, can_start))
        if taken is None:
            return
        entry, reason, age = taken
        entry.warned = False
        logger.info("Bot loop guard released held message %s in %s chat %s (reason=%s, age %.0fs; "
                    "gateway.bot_loop_guard).", entry.message_id or "voice",
                    source.platform.value if source.platform else "", source.chat_id, reason, age)
        try:
            await adapter.handle_message(entry.event)
        except asyncio.CancelledError:
            guard.hand_over_failed(conversation, entry)
            raise
        except Exception as exc:
            outcome = guard.hand_over_failed(conversation, entry)
            logger.warning("Bot loop guard hand-over of held message %s in %s chat %s raised (%s: %s); %s "
                           "(gateway.bot_loop_guard).", entry.message_id or "voice",
                           source.platform.value if source.platform else "", source.chat_id, type(exc).__name__, exc,
                           {"retry": "it goes back to the head of its queue", "discarded": "it was discarded by /stop",
                            "lost": "it is dropped after 3 tries"}[outcome])
            if outcome == "lost":
                guard.requeue_job(NoticeJob("lost", conversation, entry.seat, source, [entry.info()], lost=entry))
        else:
            guard.hand_over_done(conversation, entry)

    async def _bot_loop_hold_pump_loop(self) -> None:
        wake = self._hold_wake_event
        while True:
            try:
                await self._bot_loop_hold_pump_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Bot loop guard hold pump failed; it keeps running")
            guard = getattr(self, "_bot_loop_guard", None)
            timeout = 30.0
            if guard is not None:
                if guard.pending_work():
                    timeout = _PUMP_RECHECK_SECONDS
                    due = guard.seconds_to_next_due()
                    if due is not None:
                        timeout = min(timeout, max(0.05, due))
                if guard.refs_dirty():
                    timeout = min(timeout, _REFS_COALESCE_SECONDS)
            try:
                await asyncio.wait_for(wake.wait(), timeout)
            except TimeoutError:
                pass
            wake.clear()

    def _bot_loop_hold_start(self) -> None:
        """Start the single pump task, and the restart report, on the gateway loop (after the adapters connect)."""
        loop = asyncio.get_running_loop()
        if self._hold_task is not None and not self._hold_task.done():
            return
        self._hold_loop = loop
        self._hold_wake_event = asyncio.Event()
        self._hold_refs_ready = False  # the previous run's file is read first; nothing overwrites it before that
        self._hold_task = loop.create_task(self._bot_loop_hold_pump_loop(), name="bot-loop-hold-pump")
        self._hold_report_task = loop.create_task(self._bot_loop_hold_report_lost(), name="bot-loop-hold-report")

    def _bot_loop_hold_shutdown(self) -> None:
        """Graceful stop: end the pump, then write what is still held so the next start can name it (S8)."""
        for name in ("_hold_task", "_hold_report_task"):
            task = getattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
        try:
            self._bot_loop_hold_flush_refs()
        except Exception:
            logger.warning("Bot loop guard could not write its restart file on stop", exc_info=True)

    # --- notices (S6) --------------------------------------------------------------------------------------------

    def _bot_loop_hold_seat_label(self, job: NoticeJob) -> str:
        try:
            adapter = self._adapter_for_source(job.source)
        except Exception:
            adapter = None
        user = getattr(getattr(adapter, "_client", None), "user", None)
        for attr in ("display_name", "name"):
            value = getattr(user, attr, None)
            if isinstance(value, str) and value:
                return value
        return job.seat or "this bot"

    def _bot_loop_hold_render(self, guard: BotLoopGuard, job: NoticeJob) -> list[str]:
        """The chat messages ``job`` says right now (none when there is nothing left to say)."""
        seat = self._bot_loop_hold_seat_label(job)
        if job.kind in ("hold", "pause", "over"):
            ctx = self._under_authorization_profile(job.source, lambda: guard.notice_context(job))
            if ctx is None:
                return []
            if job.kind == "hold":
                return [format_hold_notice(ctx["senders"], seat, ctx["delay"], ctx["held"], ctx["replies_to_replies"])]
            if job.kind == "pause":
                return [format_pause_notice(ctx["senders"], seat, ctx["held"], ctx["replies_to_replies"])]
            return [format_over_notice(ctx["senders"], seat, ctx["held"], ctx["limit"])]
        lines = format_entry_lines(job.entries)
        if job.kind == "discard":
            lead = f"Bot loop guard: discarded {len(lines)} held message(s) on a person's /stop (not delivered):"
        else:
            lead = ("Bot loop guard: could not hand over this held message after 3 tries; it was dropped "
                    "(held messages live in memory only):")
        return format_listing(lead, lines)[0]

    async def _bot_loop_hold_post(self, source, text: str) -> Optional[bool]:
        """True when ``text`` was posted, False when the send failed, None when no live adapter serves ``source`` (the
        entry's own "no live adapter" warning already says so; the notice waits for the adapter)."""
        try:
            adapter = self._adapter_for_source(source)
        except Exception:
            adapter = None
        sender = getattr(adapter, "send_hold_notice", None)
        if sender is None:
            return None
        try:
            result = await sender(str(source.chat_id), text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Bot loop guard notice send failed in %s chat %s: %s (gateway.bot_loop_guard).",
                           source.platform.value if source.platform else "", source.chat_id, exc)
            return False
        if getattr(result, "success", True) is False:
            logger.warning("Bot loop guard notice send failed in %s chat %s: %s (gateway.bot_loop_guard).",
                           source.platform.value if source.platform else "", source.chat_id,
                           getattr(result, "error", None))
            return False
        return True

    async def _bot_loop_hold_send_notices(self, guard: BotLoopGuard) -> None:
        for job in guard.drain_jobs():
            messages = self._bot_loop_hold_render(guard, job)
            failed, waiting = False, False
            for index in range(job.sent, len(messages)):
                posted = await self._bot_loop_hold_post(job.source, messages[index])
                if posted is True:
                    continue
                job.sent = index
                failed, waiting = posted is False, posted is None
                break
            if waiting:
                guard.requeue_job(job)  # no adapter yet: wait for it, no attempt used
                continue
            if not failed:
                if job.lost is not None:
                    guard.lost_sent(job.lost)
                continue
            job.attempts += 1
            if job.attempts < _NOTICE_ATTEMPTS:
                guard.requeue_job(job)  # retried on the next wake
            else:
                logger.warning("Bot loop guard gave up posting a %s notice in %s chat %s after %d tries "
                               "(gateway.bot_loop_guard).", job.kind,
                               job.source.platform.value if job.source.platform else "", job.source.chat_id,
                               job.attempts)

    # --- the restart file (S8) -----------------------------------------------------------------------------------

    def _bot_loop_hold_seat_homes(self) -> dict[str, Path]:
        """Seat (profile name; ``""`` is the process's own seat) -> that seat's profile home. Without
        ``multiplex_profiles`` there is one seat and one home. Under it, one process serves every profile and each
        seat keeps its own refs file in its own home (S8), never one shared file."""
        from hermes_constants import get_hermes_home

        homes: dict[str, Path] = {"": Path(get_hermes_home())}
        config = getattr(self, "config", None)
        if getattr(config, "multiplex_profiles", False):
            try:
                from gateway.run import _multiplex_profile_homes

                for name, home in _multiplex_profile_homes(config):
                    homes[str(name)] = Path(home)
            except Exception:
                logger.warning("Bot loop guard could not list the served profiles; using the process home only "
                               "(gateway.bot_loop_guard).", exc_info=True)
        return homes

    def _bot_loop_hold_home_for(self, seat: str) -> Optional[Path]:
        """The profile home holding ``seat``'s refs file, or None when it cannot be told."""
        homes = self._bot_loop_hold_seat_homes()
        if seat in homes:
            return homes[seat]
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return homes[""]
        try:
            from hermes_cli.profiles import get_profile_dir

            return Path(get_profile_dir(seat))
        except Exception:
            return None

    def _bot_loop_hold_all_homes(self) -> list[Path]:
        """Every distinct seat home, in a stable order (one entry per file)."""
        seen: list[Path] = []
        for home in self._bot_loop_hold_seat_homes().values():
            if home not in seen:
                seen.append(home)
        return seen

    @staticmethod
    def _bot_loop_hold_refs_path(home: Path) -> Optional[Path]:
        """``<home>/gateway/bot_loop_hold_refs.json``; None when the home does not exist (never re-created)."""
        return home / "gateway" / _REFS_FILE if home.is_dir() else None

    def _bot_loop_hold_write_file(self, home: Path, entries: list[dict[str, Any]], not_listed: int = 0) -> bool:
        """Write ONE seat's file: its 500-entry cap and its count of entries dropped from the list."""
        path = self._bot_loop_hold_refs_path(home)
        if path is None:
            return False
        if not entries and not not_listed and not path.exists():
            return True
        if len(entries) > _REFS_MAX_ENTRIES:
            dropped = len(entries) - _REFS_MAX_ENTRIES
            logger.warning("Bot loop guard restart file %s lists %d messages; the oldest %d are dropped from the "
                           "list (gateway.bot_loop_guard).", path, len(entries), dropped)
            entries, not_listed = entries[-_REFS_MAX_ENTRIES:], not_listed + dropped
        from utils import atomic_write_text

        path.parent.mkdir(mode=0o700, exist_ok=True)
        atomic_write_text(path, json.dumps({"version": 1, "not_listed": not_listed, "entries": entries}, indent=1),
                          mode=0o600)
        self._hold_refs_last_write = time.monotonic()
        return True

    def _bot_loop_hold_flush_refs(self, *, only_if_due: bool = False) -> None:
        """Write what is held (and what the last run lost and this run has not yet named) to each seat's file."""
        guard = getattr(self, "_bot_loop_guard", None)
        if guard is None or not self._hold_refs_ready:
            return
        if only_if_due and (not guard.refs_dirty()
                            or time.monotonic() - self._hold_refs_last_write < _REFS_COALESCE_SECONDS):
            return
        carry: dict[Path, list[dict[str, Any]]] = getattr(self, "_hold_refs_carry", None) or {}
        carry_not_listed: dict[Path, int] = getattr(self, "_hold_refs_not_listed", None) or {}
        per_home: dict[Path, list[dict[str, Any]]] = {home: list(items) for home, items in carry.items()}
        for item in guard.refs_snapshot():
            home = self._bot_loop_hold_home_for(str(item.get("profile") or ""))
            if home is not None:
                per_home.setdefault(home, []).append(item)
        written = getattr(self, "_hold_refs_written_homes", None) or set()
        ok = True
        for home in set(per_home) | set(written) | set(carry_not_listed):
            items = per_home.get(home, [])
            if not self._bot_loop_hold_write_file(home, items, carry_not_listed.get(home, 0)):
                ok = False
        self._hold_refs_written_homes = {home for home, items in per_home.items() if items}
        if ok:
            guard.refs_written()

    async def _bot_loop_hold_report_lost(self) -> None:
        """After the adapters connect: name the held messages the last run lost (its restart file), one message set per
        conversation, then keep whatever could not be named for the next start."""
        try:
            await self._bot_loop_hold_report_lost_inner()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Bot loop guard could not report the previous run's held messages", exc_info=True)
        finally:
            self._hold_refs_ready = True
            try:
                self._bot_loop_hold_flush_refs()
            except Exception:
                logger.warning("Bot loop guard could not write its restart file", exc_info=True)

    async def _bot_loop_hold_report_lost_inner(self) -> None:
        carry: dict[Path, list[dict[str, Any]]] = {}
        carry_not_listed: dict[Path, int] = {}
        try:
            for home in self._bot_loop_hold_all_homes():
                remaining, not_listed = await self._bot_loop_hold_report_file(home)
                if remaining:
                    carry[home] = remaining
                if not_listed:
                    carry_not_listed[home] = not_listed
        finally:
            self._hold_refs_carry = carry
            self._hold_refs_not_listed = carry_not_listed

    async def _bot_loop_hold_report_file(self, home: Path) -> tuple[list[dict[str, Any]], int]:
        """Name what ONE seat's file says its previous run lost; returns what could not be named (kept for the next
        start) and the count of entries the file had already dropped from its list."""
        path = self._bot_loop_hold_refs_path(home)
        if path is None or not path.exists():
            return [], 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data["entries"]
            not_listed = int(data.get("not_listed", 0) or 0)
            if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
                raise ValueError("entries must be a list of objects")
        except Exception as exc:
            aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(path, aside)
            except OSError:
                aside = path
            logger.warning("Bot loop guard restart file %s is unreadable (%s); moved to %s "
                           "(gateway.bot_loop_guard).", path, exc, aside.name)
            return [], 0
        if not entries and not not_listed:
            return [], 0
        remaining: list[dict[str, Any]] = list(entries)
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for item in entries:
            groups.setdefault((str(item.get("profile") or ""), str(item.get("channel") or "")), []).append(item)
        last_group = list(groups)[-1] if groups else None
        logger.warning("Bot loop guard lost %d held messages to a restart (%s; gateway.bot_loop_guard).",
                       len(entries), path)

        def forget(items: list[dict[str, Any]]) -> None:
            for item in items:
                for index, kept in enumerate(remaining):
                    if kept is item:
                        del remaining[index]
                        break

        extra_not_listed = not_listed
        for key, group in groups.items():
            profile, channel = key
            source = SessionSource(platform=Platform.DISCORD, chat_id=channel, chat_type="group",
                                   profile=profile or None)
            adapter = self._bot_loop_hold_report_adapter(source)
            lead = (f"Bot loop guard: {len(group)} held message(s) were lost when this bot restarted "
                    f"(held bot messages live in memory only):")
            messages, counts = format_listing(lead, format_entry_lines(group),
                                              not_listed=extra_not_listed if key == last_group else 0)
            sender = getattr(adapter, "send_hold_notice", None)
            position = 0
            for text, count in zip(messages, counts):
                chunk = group[position: position + count]
                position += count
                ok, unpostable = False, False
                if sender is not None:
                    try:
                        result = await sender(channel, text)
                        ok = getattr(result, "success", True) is not False
                        error = str(getattr(result, "error", "") or "")
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        error = str(exc)
                    unpostable = not ok and any(mark in error.lower() for mark in _UNPOSTABLE)
                if ok:
                    forget(chunk)
                    self._bot_loop_hold_write_file(home, remaining, extra_not_listed if remaining else 0)
                    continue
                logger.warning("Bot loop guard could not post the lost-messages line in chat %s%s "
                               "(gateway.bot_loop_guard).", channel,
                               "; the channel can no longer be posted to, so its entries are removed"
                               if unpostable else "; its entries stay in the restart file")
                if unpostable:
                    forget(group[position - count:])
                break
            if key == last_group and not remaining:
                extra_not_listed = 0
        return remaining, (extra_not_listed if remaining else 0)

    def _bot_loop_hold_report_adapter(self, source) -> Any:
        try:
            adapter = self._adapter_for_source(source)
        except Exception:
            adapter = None
        if adapter is None:
            adapter = (getattr(self, "adapters", None) or {}).get(Platform.DISCORD)
        return adapter
