"""Discord voice-session lifecycle: the per-guild session record, the leave, and the end event.

A voice session starts on a cold ``channel.connect()`` and ends exactly once, when a leave takes its
record under the guild's voice lock: ``/voice leave`` (``command``), auto-leave (``auto_leave`` with
``trigger`` ``no_authorized_users`` or ``inactivity``), a connection Discord closed for good
(``disconnect``, seen through ``VoiceClient.cleanup()``, which discord.py calls only on a final
disconnect, never on a reconnect) and adapter teardown (``shutdown``; the runner reports a teardown
while the gateway keeps running as ``disconnect``). The record is taken after the final receiver
flush (so the last utterance is in it) and before ``vc.disconnect()``; the end is handed to the
runner synchronously from a ``finally``, so neither an exception in the leave nor the inactivity
task cancelling itself loses it. The handler only schedules work; nothing here awaits a plugin.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.i18n import t

logger = logging.getLogger("plugins.platforms.discord.adapter")

VOICE_SESSION_END_REASONS = ("auto_leave", "command", "disconnect", "shutdown")
# disconnect() waits this long for scheduled end handlers, inside the gateway's 5 s adapter budget.
_VOICE_SESSION_END_SHUTDOWN_BUDGET_SECS = 2.0


@dataclass
class VoiceSessionRecord:
    """One voice session of one guild: created on a cold connect, taken once by the leave."""

    guild_id: int
    voice_channel_id: Optional[int]
    voice_session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    participants: list[str] = field(default_factory=list)
    session_keys: list[str] = field(default_factory=list)
    last_session_key: Optional[str] = None


@dataclass(frozen=True)
class VoiceSessionEnd:
    """Immutable end-of-session snapshot handed to the runner (plain data, no SDK handles)."""

    voice_session_id: str
    guild_id: int
    voice_channel_id: Optional[int]
    text_channel_id: Optional[int]
    started_at: float
    ended_at: float
    reason: str
    trigger: Optional[str]
    participants: tuple[str, ...]
    session_keys: tuple[str, ...]
    session_key: Optional[str]


def hermes_voice_client_class(base: type, adapter: Any) -> type:
    """A ``VoiceClient`` subclass whose ``cleanup()`` tells *adapter* the connection is gone for good."""
    adapter_ref = weakref.ref(adapter)

    class HermesVoiceClient(base):  # type: ignore[misc, valid-type]
        def cleanup(self) -> None:
            try:
                super().cleanup()
            finally:
                owner = adapter_ref()
                if owner is not None:
                    owner._on_voice_client_closed(self)

    return HermesVoiceClient


class DiscordVoiceSessionMixin:
    _voice_clients: dict[int, Any]
    _voice_locks: dict[int, asyncio.Lock]
    _voice_text_channels: dict[int, Any]
    _voice_sources: dict[int, dict[str, Any]]
    _voice_timeout_tasks: dict[int, asyncio.Task]
    _voice_receivers: dict[int, Any]
    _voice_listen_tasks: dict[int, asyncio.Task]
    _voice_mixers: dict[int, Any]
    _voice_timeout_limit: Callable[[], int]
    _on_voice_disconnect: Optional[Callable]
    _client: Any
    _is_allowed_user: Any
    _process_voice_input: Any

    # --- the session record -------------------------------------------------------------------

    def _start_voice_session(self, guild_id: int, vc: Any) -> None:
        sessions = getattr(self, "_voice_sessions", None)
        if isinstance(sessions, dict):
            channel_id = getattr(getattr(vc, "channel", None), "id", None)
            sessions[guild_id] = VoiceSessionRecord(guild_id=guild_id, voice_channel_id=channel_id)

    def note_voice_session_speech(self, guild_id: int, user_id: str, session_key: Optional[str]) -> None:
        """Record one dispatched transcript: its speaker and the session key it was dispatched into
        (``None`` when the runner refused to attribute it to this bot's profile)."""
        record = (getattr(self, "_voice_sessions", None) or {}).get(guild_id)
        if record is None:
            return
        if user_id not in record.participants:
            record.participants.append(user_id)
        if session_key:
            if session_key not in record.session_keys:
                record.session_keys.append(session_key)
            record.last_session_key = session_key

    def _take_voice_session(self, guild_id: int, vc: Any, text_channel_id: Any) -> Optional[dict[str, Any]]:
        """Pop the guild's record (call under the voice lock); the snapshot fields known at take time."""
        record = (getattr(self, "_voice_sessions", None) or {}).pop(guild_id, None)
        if record is None:
            return None
        channel_id = getattr(getattr(vc, "channel", None), "id", None) or record.voice_channel_id
        return {
            "voice_session_id": record.voice_session_id, "guild_id": guild_id, "voice_channel_id": channel_id,
            "text_channel_id": text_channel_id, "started_at": record.started_at,
            "participants": tuple(record.participants), "session_keys": tuple(record.session_keys),
            "session_key": record.last_session_key,
        }

    def _emit_voice_session_end(self, taken: dict[str, Any], reason: str, trigger: Optional[str]) -> None:
        """Hand the end to the runner. Synchronous by contract: it only schedules (the runner returns
        a task); a raising or missing handler never fails the leave."""
        end = VoiceSessionEnd(**taken, ended_at=time.time(), reason=reason,
                              trigger=trigger if reason == "auto_leave" else None)
        callback = getattr(self, "_voice_session_end_callback", None)
        if callback is None:
            logger.debug("Voice session %s ended (%s); no end handler bound", end.voice_session_id, reason)
            return
        try:
            task = callback(end)
        except Exception:
            logger.warning("Voice session end handler failed for guild %s", end.guild_id, exc_info=True)
            return
        tasks = getattr(self, "_voice_session_end_tasks", None)
        if isinstance(task, asyncio.Future) and isinstance(tasks, set):
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    async def _await_voice_session_end_tasks(self, budget: float = _VOICE_SESSION_END_SHUTDOWN_BUDGET_SECS) -> None:
        """Teardown: give scheduled end handlers a bounded chance to finish; stragglers keep running."""
        pending = [task for task in getattr(self, "_voice_session_end_tasks", ()) if not task.done()]
        if not pending:
            return
        _done, still = await asyncio.wait(pending, timeout=budget)
        if still:
            logger.info("%d voice-session-end handler(s) still running after %.1fs; not waiting further",
                        len(still), budget)

    # --- a connection Discord closed for good ---------------------------------------------------

    def _voice_client_class(self) -> Optional[type]:
        from plugins.platforms.discord import adapter as adapter_mod
        base = getattr(adapter_mod.discord, "VoiceClient", None) if adapter_mod.DISCORD_AVAILABLE else None
        if not isinstance(base, type):
            return None
        cached = getattr(self, "_voice_client_cls", None)
        if cached is None or cached.__mro__[1] is not base:
            cached = self._voice_client_cls = hermes_voice_client_class(base, self)
        return cached

    def _on_voice_client_closed(self, vc: Any) -> None:
        """``cleanup()`` hook (final disconnects only). Our own leave popped the client first, so this
        is then a no-op; otherwise the session ends as ``disconnect`` and the stale state goes."""
        guild_id = getattr(getattr(getattr(vc, "channel", None), "guild", None), "id", None)
        if guild_id is None or self._voice_clients.get(guild_id) is not vc:
            return
        try:
            task = asyncio.get_running_loop().create_task(self._end_voice_session_on_close(guild_id, vc))
        except RuntimeError:
            return
        tasks = self.__dict__.setdefault("_voice_close_tasks", set())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def _end_voice_session_on_close(self, guild_id: int, vc: Any) -> None:
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            if self._voice_clients.get(guild_id) is not vc:
                return
            text_ch_id = self._voice_text_channels.get(guild_id)
            logger.info("Voice connection in guild %d closed by Discord; ending the voice session", guild_id)
            await self._leave_voice_locked(guild_id, reason="disconnect", trigger=None)
        if self._on_voice_disconnect and text_ch_id:
            try:
                self._on_voice_disconnect(str(text_ch_id))
            except Exception:
                logger.debug("voice disconnect cleanup failed", exc_info=True)

    async def _end_stale_voice_session_locked(self, guild_id: int) -> None:
        """Cold join while a record survives (the previous connection dropped unseen): end it as
        ``disconnect`` first, keeping the text binding for a join that names none."""
        if guild_id not in (getattr(self, "_voice_sessions", None) or {}):
            return
        binding, source = self._voice_text_channels.get(guild_id), self._voice_sources.get(guild_id)
        await self._leave_voice_locked(guild_id, reason="disconnect", trigger=None)
        if binding is not None:
            self._voice_text_channels.setdefault(guild_id, binding)
            if source is not None:
                self._voice_sources.setdefault(guild_id, source)

    # --- the leave ------------------------------------------------------------------------------

    async def leave_voice_channel(self, guild_id: int, *, reason: str = "command", trigger: Optional[str] = None) -> None:
        """Disconnect from the voice channel in a guild, ending its voice session (if any) once."""
        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            await self._leave_voice_locked(guild_id, reason=reason, trigger=trigger)

    async def _leave_voice_locked(self, guild_id: int, *, reason: str, trigger: Optional[str]) -> None:
        taken = None
        captured_for = self._voice_text_channels.get(guild_id)
        try:
            receiver = self._voice_receivers.pop(guild_id, None)
            pending_inputs = []
            if receiver:
                pending_inputs = receiver.flush_pending()
                receiver.stop()
            listen_task = self._voice_listen_tasks.pop(guild_id, None)
            if listen_task:
                listen_task.cancel()
            guild = self._client.get_guild(guild_id) if self._client is not None else None
            for user_id, pcm_data in pending_inputs:
                if self._is_allowed_user(str(user_id), guild=guild, is_dm=False):
                    await self._process_voice_input(guild_id, user_id, pcm_data, captured_for)
            # Tear down the mixer (stops the continuous outgoing stream).
            if getattr(self, "_voice_mixers", None) is not None:
                self._voice_mixers.pop(guild_id, None)
            vc = self._voice_clients.pop(guild_id, None)
            # After the final flush (its speech is recorded), before vc.disconnect() (whose cleanup()
            # would otherwise read as a connection Discord closed).
            taken = self._take_voice_session(guild_id, vc, captured_for)
            if vc and vc.is_connected():
                try:
                    if vc.is_playing():
                        vc.stop()
                except Exception:
                    logger.debug("Stopping voice playback before the leave failed", exc_info=True)
                await vc.disconnect()
            task = self._voice_timeout_tasks.pop(guild_id, None)
            if task:
                task.cancel()
            # Undo the auto-join's implicit "speak in this chat" opt-in. Without this the enabled
            # set outlives the voice session: every later reply in that channel is synthesised,
            # including replies to plain text, and once the bot has left the VC it degrades to an
            # audio attachment on each one. Only chats THIS path enabled are cleared, so a user's
            # explicit /voice tts survives.
            auto_tts_chats = getattr(self, "_auto_join_tts_chats", None)
            if isinstance(auto_tts_chats, set) and captured_for is not None:
                key = str(captured_for)
                if key in auto_tts_chats:
                    auto_tts_chats.discard(key)
                    enabled = getattr(self, "_auto_tts_enabled_chats", None)
                    if isinstance(enabled, set):
                        enabled.discard(key)
            self._voice_text_channels.pop(guild_id, None)
            self._voice_sources.pop(guild_id, None)
        finally:
            if taken is None:  # the leave raised before the take: the session still ends, once
                taken = self._take_voice_session(guild_id, None, captured_for)
            if taken is not None:
                self._emit_voice_session_end(taken, reason, trigger)

    # --- inactivity timeout (moved from adapter.py) ---------------------------------------------

    def _cancel_voice_timeout(self, guild_id: int) -> None:
        task = self._voice_timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()

    def _reset_voice_timeout(self, guild_id: int) -> None:
        """Reset the auto-disconnect inactivity timer."""
        self._cancel_voice_timeout(guild_id)
        timeout = self._voice_timeout_limit()
        if timeout <= 0:
            logger.debug("Voice inactivity timeout disabled (guild=%d)", guild_id)
            return
        self._voice_timeout_tasks[guild_id] = asyncio.ensure_future(
            self._voice_timeout_handler(guild_id, timeout)
        )

    async def _voice_timeout_handler(self, guild_id: int, timeout: Optional[int] = None) -> None:
        """Auto-disconnect after the configured inactivity timeout."""
        timeout = self._voice_timeout_limit() if timeout is None else int(timeout)
        if timeout <= 0:
            return
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        text_ch_id = self._voice_text_channels.get(guild_id)
        # ``/voice off`` keeps the bot in the channel; only the bot's own audio counts as
        # activity, so the timer would fire every VOICE_TIMEOUT and spam "Left voice channel".
        _mode_getter = getattr(self, "_voice_mode_getter", None)
        if text_ch_id is not None and _mode_getter is not None:
            try:
                if _mode_getter(str(text_ch_id)) == "off":
                    return
            except Exception:
                logger.debug("voice mode lookup failed; the inactivity leave proceeds", exc_info=True)
        await self.leave_voice_channel(guild_id, reason="auto_leave", trigger="inactivity")
        # Notify the runner so it can clean up voice_mode state
        if self._on_voice_disconnect and text_ch_id:
            try:
                self._on_voice_disconnect(str(text_ch_id))
            except Exception:
                logger.debug("voice disconnect cleanup failed", exc_info=True)
        if text_ch_id and self._client:
            ch = self._client.get_channel(text_ch_id)
            if ch:
                try:
                    await ch.send(t("platform.discord.voice.left_inactivity"))
                except Exception:
                    logger.debug("Could not post the inactivity-leave notice", exc_info=True)

    def is_in_voice_channel(self, guild_id: int) -> bool:
        """Check if the bot is connected to a voice channel in this guild."""
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()
