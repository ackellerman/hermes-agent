"""Voice-session end for GatewayRunner: fire ``on_voice_session_end`` once, then at most one follow-up.

The Discord adapter hands over an immutable :class:`VoiceSessionEnd` when a voice session ends
(``plugins/platforms/discord/adapter_voice_session.py``). Here, per end:

* the reason is settled: an adapter teardown is ``shutdown`` only while the gateway itself is
  stopping; a recycle or profile removal while it keeps running reports ``disconnect``;
* the hook fires ONCE, under the runtime scope of the profile whose bot left (the adapter owner),
  so only that profile's plugins see it — never the launch profile's by accident;
* the first plugin that returns ``{"followup_turn": str}`` gets one turn in the session the voice
  transcripts were dispatched into (the most recent one), started through the same
  ``inject_plugin_message`` path ``PluginContext.inject_message`` uses — so
  ``plugins.entries.<id>.allow_gateway_injection`` gates it unchanged — as a ``[system]`` internal
  notification whose reply is posted as text and never spoken.

The hook call and the injection (which reads the gate from config) run on a worker thread, never on
the event loop. Every hook call is bounded (the configured ``plugins.hook_callback_timeout``, or
30 s when that is off), a ``shutdown`` end gets a 2 s bound and its follow-up is dropped. An owner
profile that no longer resolves skips the hook rather than dispatching it to the launch profile.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

HOOK = "on_voice_session_end"
_FOLLOWUP_MAX_CHARS = 8000
_SHUTDOWN_HOOK_TIMEOUT_SECS = 2.0
# Used when plugins.hook_callback_timeout is 0 (worker path off): a hung plugin must not pin an
# executor thread per session end for as long as it hangs.
_DEFAULT_HOOK_TIMEOUT_SECS = 30.0
# The last transcript's turn opens its session from a background task; a session that utterance
# created may not be in the store yet when the end is dispatched.
_SESSION_ENTRY_WAIT_SECS = 5.0


def select_voice_followup(results: list[tuple[Optional[str], Any]]) -> Optional[tuple[str, str]]:
    """``(plugin_key, instruction)`` of the first plugin-owned ``{"followup_turn": str}``, or None.

    Later candidates, callbacks no plugin owns (shell hooks: no ``allow_gateway_injection`` gate to
    honour) and oversized instructions are dropped with a log line."""
    winner: Optional[tuple[str, str]] = None
    for owner, result in results:
        text = result.get("followup_turn") if isinstance(result, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        if winner is not None:
            logger.info("%s: follow-up from %s ignored: one follow-up per session end, %s already won",
                        HOOK, owner or "a non-plugin hook", winner[0])
        elif owner is None:
            logger.info("%s: dropping a follow-up from a hook no plugin owns (no injection gate)", HOOK)
        elif len(text) > _FOLLOWUP_MAX_CHARS:
            logger.warning("%s: dropping follow-up from plugin %s: %d chars > %d",
                           HOOK, owner, len(text), _FOLLOWUP_MAX_CHARS)
        else:
            winner = (owner, text.strip())
    return winner


class GatewayVoiceSessionEndMixin:
    def _note_voice_session_speech(self, adapter: Any, guild_id: int, user_id: int, source: Any,
                                   identity: Any) -> None:
        """Record a dispatched transcript on the adapter's session: the speaker, and the session key
        this very source keys into — only when the turn runs in the bot owner's own profile, since the
        follow-up is started by the owner's plugin (a ``profile_routes`` re-home records no key)."""
        note = getattr(adapter, "note_voice_session_speech", None)
        if not callable(note):
            return
        session_key = self._session_key_for_source(source)
        runtime = getattr(identity, "runtime_profile", None)
        if runtime is not None and runtime != getattr(identity, "transport_profile", runtime):
            logger.debug("voice speech routed to profile %s, not recorded for bot owner %s's session end",
                         runtime, identity.transport_profile)
            session_key = None
        note(guild_id, str(user_id), session_key)

    def _on_voice_session_end(self, end: Any, *, adapter: Any) -> Optional[asyncio.Task]:
        """Adapter callback (synchronous, from the leave's ``finally``): settle the reason, schedule
        the dispatch and return its task (``disconnect()`` gives it a bounded wait)."""
        reason = end.reason
        if reason == "shutdown" and getattr(self, "_running", False) and not getattr(self, "_draining", False):
            reason = "disconnect"  # the adapter is torn down while the gateway keeps running
        task = asyncio.get_running_loop().create_task(self._dispatch_voice_session_end(end, reason, adapter))
        tasks = getattr(self, "_background_tasks", None)
        if isinstance(tasks, set):
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        return task

    def _voice_session_end_owner(self, adapter: Any) -> tuple[str, Any]:
        """``(profile name, scope home)`` of the bot that left: ``None`` home = the launch profile."""
        owner = getattr(adapter, "_owner_profile", None)
        if owner:
            return owner, self._routed_profile_home(owner)
        return getattr(self, "_primary_profile_name", None) or self._active_profile_name(), None

    async def _dispatch_voice_session_end(self, end: Any, reason: str, adapter: Any) -> None:
        try:
            from gateway.run import _async_profile_runtime_scope
            from gateway.run_adapters import UNRESOLVED_PROFILE_HOME
            profile, home = self._voice_session_end_owner(adapter)
            if home is UNRESOLVED_PROFILE_HOME:
                logger.warning("%s skipped for voice session %s: the bot owner's profile %s does not "
                               "resolve (not dispatched to another profile's plugins)",
                               HOOK, end.voice_session_id, profile)
                return
            async with self._async_scope_or_null(_async_profile_runtime_scope, home):
                session_id = None
                if end.session_key:
                    entry = await self._voice_session_entry(end.session_key, patient=reason != "shutdown")
                    session_id = getattr(entry, "session_id", None)
                payload = {
                    "platform": "discord", "profile": profile, "session_id": session_id,
                    "session_key": end.session_key, "session_keys": list(end.session_keys),
                    "guild_id": str(end.guild_id),
                    "voice_channel_id": None if end.voice_channel_id is None else str(end.voice_channel_id),
                    "text_channel_id": None if end.text_channel_id is None else str(end.text_channel_id),
                    "started_at": end.started_at, "ended_at": end.ended_at, "reason": reason,
                    "trigger": end.trigger, "participants": list(end.participants),
                    "voice_session_id": end.voice_session_id,
                }
                # contextvars (the profile home override) follow the worker thread.
                await asyncio.to_thread(self._run_voice_session_end_hook, payload)
        except Exception:
            logger.warning("%s dispatch failed for voice session %s", HOOK,
                           getattr(end, "voice_session_id", "?"), exc_info=True)

    async def _voice_session_entry(self, session_key: str, *, patient: bool) -> Any:
        """The store entry of the session the last transcript went into. When that transcript opened
        the session, its turn (a background task) may not have created the entry yet: wait for it,
        bounded (``patient``; never at shutdown). ``None`` when it does not appear."""
        loop = asyncio.get_running_loop()
        deadline, delay = loop.time() + _SESSION_ENTRY_WAIT_SECS, 0.02
        while True:
            entry = await self.async_session_store.lookup_by_session_key(session_key)
            if entry is not None or not patient or loop.time() >= deadline:
                if entry is None and patient:
                    logger.info("%s: session %s was not opened within %.0fs; the hook sees no session_id",
                                HOOK, session_key, _SESSION_ENTRY_WAIT_SECS)
                return entry
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.25)

    @staticmethod
    def _voice_session_end_hook_timeout(reason: str) -> float:
        """Always finite: the configured bound, or the default when it is off; 2 s at shutdown."""
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        configured = _resolve_hook_callback_timeout()
        timeout = configured if configured > 0 else _DEFAULT_HOOK_TIMEOUT_SECS
        return min(timeout, _SHUTDOWN_HOOK_TIMEOUT_SECS) if reason == "shutdown" else timeout

    def _run_voice_session_end_hook(self, payload: dict[str, Any]) -> None:
        """Worker thread, inside the owner's scope: fire the hook, then start the follow-up (if any)."""
        from hermes_cli import lifecycle
        reason = payload["reason"]
        results = lifecycle.invoke_hook_attributed(
            HOOK, timeout=self._voice_session_end_hook_timeout(reason), **payload)
        selected = select_voice_followup(results)
        if selected is None:
            return
        plugin_key, instruction = selected
        if reason == "shutdown":
            logger.info("%s: dropping follow-up from plugin %s (reason=shutdown)", HOOK, plugin_key)
            return
        session_key = payload["session_key"]
        if not session_key:
            logger.info("%s: dropping follow-up from plugin %s: no session to continue "
                        "(no transcript was dispatched for this bot's profile)", HOOK, plugin_key)
            return
        self._inject_voice_followup(plugin_key, instruction, session_key)

    @staticmethod
    def _inject_voice_followup(plugin_key: str, instruction: str, session_key: str) -> bool:
        """Start the follow-up AS the owning plugin, through the same gate ``inject_message`` uses."""
        from gateway.run_notifications import _mark_internal_notification
        from hermes_cli.plugins import PluginContext, get_plugin_manager
        from hermes_cli.plugins_injection import inject_plugin_message
        manager = get_plugin_manager()
        loaded = getattr(manager, "_plugins", {}).get(plugin_key)
        manifest = getattr(loaded, "manifest", None)
        if manifest is None:
            logger.info("%s: dropping follow-up from plugin %s: no longer loaded", HOOK, plugin_key)
            return False
        started = inject_plugin_message(
            PluginContext(manifest, manager), _mark_internal_notification(instruction), "system",
            session_key=session_key, origin=None, suppress_voice_reply=True)
        if started:
            logger.info("%s: follow-up from plugin %s scheduled into session %s", HOOK, plugin_key, session_key)
        else:
            logger.info("%s: follow-up from plugin %s not started (injection denied or unavailable)",
                        HOOK, plugin_key)
        return started
