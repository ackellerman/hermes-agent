"""Spoken PRE-TURN brief for Discord voice-input turns.

Opt-in per profile (``discord.voice_brief.enabled``, default off). Before a voice turn runs,
one bounded auxiliary call for the ``voice_brief`` task asks for one short sentence, in the
model's own words, stating its understanding and what it will do next. The task route is its own
(``auxiliary.voice_brief.*``) and defaults to ``auto``: the turn's routed provider/model, with the
profile's fallback policy behind it. A configured pin is absolute instead — the brief runs on the
pinned provider/model and its own ``fallback_chain``, never on the main agent's model. That
sentence is synthesized with the profile's configured TTS provider, played in the turn's bound
voice channel (awaited), and posted to the bound text channel. Only then does the turn run —
exactly as it does without this feature: no gated tools, no muted streaming, no rewritten history.

Two failure shapes, and the difference is deliberate. A **pinned** brief (``auxiliary.voice_brief``
pins a provider *and* ``task_chain_only``) that cannot be served — its client cannot be built, or
the request fails after its own chain is exhausted — raises ``VoiceBriefBypassed``: the caller posts
one short message naming the provider and failure class, plays nothing, and runs the turn anyway.
Every other failure (TTS, playback, text post, or any failure of an ``auto`` brief) returns visible
failure text instead of running the turn. The brief is never invented: a canned phrase is not a
fallback, and the bypass is not a brief.

Fakes a test needs live only at the edges: the platform adapter (playback / text post), the TTS
provider, and the model endpoint.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import shutil
import tempfile
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_SECTION = "voice_brief"
MAX_BRIEF_CHARS = 200
# Hard bounds. The model call is the one that can hang a voice turn forever.
MODEL_TIMEOUT_S = 15.0
# Thinking off for this call (the same mechanism title generation uses); routes that refuse the
# disable are stepped up to their lowest effort by the auxiliary client's reasoning-floor ladder.
MODEL_REASONING = {"enabled": False}
TTS_TIMEOUT_S = 60.0

# Routes that execute work themselves; the normal loop (and so this seam) never runs there.
_EXTERNAL_EXECUTION_OWNERS = frozenset({"codex_app_server", "proxy"})

FAILED_TEXT = (
    "I couldn't speak my short understanding of your request, so I stopped without running anything."
)
UNSUPPORTED_ROUTE_TEXT = (
    "I can't speak a short understanding of your request on this route: it runs work outside the "
    "normal conversation loop, so I stopped without running anything."
)
# One short line, no credential and no key material: what failed (provider/model) and how.
BYPASS_TEXT = (
    "I couldn't speak my short understanding of your request: the pinned voice-brief provider "
    "{route} failed ({reason}). Running your request anyway."
)

_INSTRUCTIONS = (
    "In one brief, natural sentence, tell the requester what you understood, then what you'll do "
    "now. Address them directly in your own words. Avoid answering the request itself. "
    f"Keep it under {MAX_BRIEF_CHARS} characters."
)


class VoiceBriefError(Exception):
    """Delivery failed; the message is user-visible and the turn must not run."""


class VoiceBriefBypassed(Exception):
    """A PINNED brief could not be served; the turn runs without one and the caller says so.

    ``provider`` is ``provider/model`` (what the operator pinned), ``reason`` a short failure class.
    Neither carries credential material: this message is posted to the chat channel.
    """

    def __init__(self, provider: str, reason: str):
        self.provider = str(provider or "")
        self.reason = str(reason or "")
        super().__init__(f"{self.provider} failed ({self.reason})")


def brief_enabled(config: Optional[dict[str, Any]]) -> bool:
    """True when this profile opted in to ``discord.voice_brief.enabled``."""
    section = ((config or {}).get("discord") or {}).get(CONFIG_SECTION) or {}
    return bool(isinstance(section, dict) and section.get("enabled"))


def route_owns_execution(api_mode: Optional[str]) -> bool:
    """True for routes whose execution never passes through the normal turn loop."""
    return str(api_mode or "").strip().lower() in _EXTERNAL_EXECUTION_OWNERS


def failure_text(reason: str) -> str:
    """Visible refusal text: names the stage, never fabricates a brief."""
    return f"{FAILED_TEXT} ({reason})"


def unsupported_route_text() -> str:
    return UNSUPPORTED_ROUTE_TEXT


def bypassed_text(route: str, reason: str) -> str:
    """One chat line for a bypassed pinned brief: names the provider, never a credential."""
    return BYPASS_TEXT.format(route=route, reason=reason)


def pinned_task_route() -> Optional[str]:
    """``provider/model`` of the brief's OWN route when it is pinned and ``task_chain_only``.

    ``None`` means "the slot is on ``auto``": the brief then follows the turn's route and that
    route's chain, and any failure refuses the turn as it always has. The two halves are the same
    ones the auxiliary client reads at its three doors (``_task_stops_at_chain``), so the seam and
    the client agree on what "pinned" means.
    """
    from agent.auxiliary_client import _resolve_task_provider_model, _task_stops_at_chain

    provider, model, _base_url, _api_key, _api_mode = _resolve_task_provider_model(CONFIG_SECTION)
    if not _task_stops_at_chain(CONFIG_SECTION, provider):
        return None
    return f"{provider}/{model}" if model else str(provider)


def failure_class(exc: Exception) -> str:
    """Short label for the bypass message — the auxiliary client's own fallback vocabulary.

    Never the exception's text: a provider's error body can echo request headers, and this line is
    posted to a chat channel.
    """
    from agent import auxiliary_client

    if isinstance(exc, auxiliary_client.AuxiliaryClientUnavailable):
        return "provider unavailable"
    for predicate, label in auxiliary_client._FALLBACK_REASONS:
        try:
            if predicate(exc):
                return label
        except Exception:
            continue
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "request timed out"
    return type(exc).__name__


def voice_binding(source: Any, adapter: Any) -> Optional[tuple[int, str]]:
    """``(guild_id, text_channel_id)`` bound to THIS turn, or None when the bot is not in voice.

    The voice-input path binds ``_voice_text_channels[guild] = text_channel`` before the turn is
    dispatched, so the guild whose binding names this turn's chat channel is the turn's own.
    """
    binding = getattr(adapter, "_voice_text_channels", None)
    if not isinstance(binding, dict):
        return None
    guild_id = next(
        (gid for gid, chat in binding.items() if str(chat) == str(getattr(source, "chat_id", ""))), None,
    )
    if guild_id is None:
        return None
    in_channel = getattr(adapter, "is_in_voice_channel", None)
    try:
        if not callable(in_channel) or not in_channel(guild_id):
            return None
    except Exception:
        logger.debug("Voice-channel membership check failed", exc_info=True)
        return None
    return int(guild_id), str(getattr(source, "chat_id", ""))


def brief_messages(message: str) -> list[dict[str, str]]:
    """Provider-edge request: spoken words and the one-sentence understanding and next action."""
    return [
        {"role": "system", "content": _INSTRUCTIONS},
        {
            "role": "user",
            "content": f'The user said this by voice:\n"""\n{message}\n"""',
        },
    ]


def clean_brief(raw: Any) -> str:
    """Collapse to one line and unquote without changing the sentence's meaning."""
    text = " ".join(str(raw or "").split())
    for left, right in (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")):
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def _main_runtime(route: dict[str, Any]) -> dict[str, Any]:
    """The turn's live runtime, in the shape the auxiliary router reads for ``provider: auto``.

    Passing it (rather than explicit ``provider``/``model`` arguments) is what makes ``auto`` mean
    "this turn's route" without pinning anything: explicit call args outrank ``auxiliary.<task>.*``,
    so an argument of our own would make every configured pin unreachable. ``api_mode`` travels with
    the pair so the brief speaks over the session's transport.
    """
    return {key: route.get(key) for key in ("provider", "model", "base_url", "api_key", "api_mode")}


def _model_timeout() -> float:
    """One budget for the model call: ``auxiliary.voice_brief.timeout``, else ``MODEL_TIMEOUT_S``.

    The auxiliary client applies the same configured value to the request itself, so the outer
    ``asyncio.wait_for`` reads it from the same place: a constant here would cut a configured longer
    call short, and a shorter configuration would abandon a request that is still running.

    Known invariant: the two fallback defaults agree for every numeric configuration and differ only
    when the block's ``timeout`` is absent or null — 15s here (a brief is one sentence) against the
    router's own 30s. No ``timeout=`` kwarg is passed, precisely so there is only one source of
    truth for the value; the outer bound is the tighter of the two by design.
    """
    from agent.auxiliary_client import _get_task_timeout

    return _get_task_timeout(CONFIG_SECTION, MODEL_TIMEOUT_S)


async def _ask_model(route: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """One auxiliary call on the brief's own task route. No tools are offered at all.

    ``task`` is the routing decision: with the shipped ``provider: auto`` the request lands on
    ``main_runtime`` (the turn's own route) and then that route's fallback chain; a configured
    ``auxiliary.voice_brief`` pin wins over it. No ``provider``/``model``/``base_url``/``api_key``
    of our own, for the reason in ``_main_runtime``.

    No ``max_tokens``: reasoning models (DeepSeek V4 on Ollama Cloud) bill thinking against it, and a
    small cap returned an empty sentence with finish_reason=length. No ``timeout`` either — the
    router applies ``auxiliary.voice_brief.timeout`` once and ``deliver_preturn_brief`` bounds the
    same value from outside; ``MAX_BRIEF_CHARS`` still rejects an over-long sentence.
    """
    from agent.auxiliary_client import async_call_llm

    route_info: dict[str, str] = {}
    response = await async_call_llm(
        task=CONFIG_SECTION,
        main_runtime=_main_runtime(route),
        messages=messages,
        temperature=0.2,
        reasoning_config=dict(MODEL_REASONING),
        route_info=route_info,
    )
    logger.info(
        "Voice brief route: requested %s/%s (turn), served by %s/%s%s",
        route.get("provider") or "auto", route.get("model") or "default",
        route_info.get("provider") or "auto", route_info.get("model") or "default",
        f" via {route_info['fallback_label']}" if route_info.get("fallback_label") else "",
    )
    return _response_text(response)


def _synthesize(text: str, output_path: str) -> str:
    """Configured TTS provider → audio file path (raises ``VoiceBriefError`` on failure)."""
    from tools.tts_tool import text_to_speech_tool

    raw = text_to_speech_tool(text=text, output_path=output_path)
    try:
        result = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):
        result = {}
    path = str(result.get("file_path") or "")
    if not result.get("success") or not path or not os.path.isfile(path):
        logger.warning("Voice brief synthesis failed: %s", result.get("error") or result)
        raise VoiceBriefError(failure_text("synthesis failed"))
    return path


async def deliver_preturn_brief(
    *, adapter: Any, guild_id: int, chat_id: str, message: str, route: dict[str, Any],
) -> str:
    """Model sentence → TTS → awaited playback → text post. Returns the sentence it delivered.

    Order is load-bearing: playback is awaited before the text post, and both before the caller
    starts the turn. Raises ``VoiceBriefError`` (user-visible) on any failure, except a failed
    model call on a PINNED route (``pinned_task_route``), which raises ``VoiceBriefBypassed`` so the
    caller can say so in one line and run the turn.
    """
    pinned = pinned_task_route()
    try:
        raw = await asyncio.wait_for(_ask_model(route, brief_messages(message)), timeout=_model_timeout())
    except VoiceBriefError:
        raise
    except Exception as exc:
        if pinned is not None:
            # The pin and its own chain could not serve: bypass, never hop to the main model.
            reason = failure_class(exc)
            logger.warning("Voice brief bypassed: pinned route %s failed (%s): %s", pinned, reason, exc)
            raise VoiceBriefBypassed(pinned, reason) from exc
        logger.warning("Voice brief model call failed: %s", exc, exc_info=True)
        raise VoiceBriefError(failure_text("model call failed")) from exc
    text = clean_brief(raw)
    if not text:
        logger.warning("Voice brief model returned no sentence")
        raise VoiceBriefError(failure_text("no sentence returned"))

    if len(text) > MAX_BRIEF_CHARS:
        raise VoiceBriefError(failure_text("sentence too long"))

    workdir = tempfile.mkdtemp(prefix="hermes-voice-brief-")
    audio_path = os.path.join(workdir, "brief.mp3")
    lock = threading.Lock()
    state = {"finished": False, "abandoned": False}

    def synthesize_owned() -> str:
        try:
            path = _synthesize(text, audio_path)
            if os.path.commonpath((os.path.realpath(workdir), os.path.realpath(path))) != os.path.realpath(workdir):
                raise VoiceBriefError(failure_text("synthesis returned an unowned path"))
            return path
        finally:
            with lock:
                state["finished"] = True
                if state["abandoned"]:
                    shutil.rmtree(workdir, ignore_errors=True)

    actual: Optional[str] = None
    submitted = False
    try:
        try:
            # run_in_executor does not copy contextvars: without the caller's context the worker
            # loses the seat's profile home and synthesizes with the root config's voice.
            future = asyncio.get_running_loop().run_in_executor(
                None, contextvars.copy_context().run, synthesize_owned,
            )
            submitted = True
            # Consume a late exception even when the waiter has timed out or been cancelled.
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            actual = await asyncio.wait_for(asyncio.shield(future), timeout=TTS_TIMEOUT_S)
        except VoiceBriefError:
            raise
        except Exception as exc:
            logger.warning("Voice brief synthesis error: %s", exc, exc_info=True)
            raise VoiceBriefError(failure_text("synthesis failed")) from exc

        try:
            played = await adapter.play_in_voice_channel(guild_id, actual, strict=True)
        except Exception as exc:
            logger.warning("Voice brief playback error: %s", exc, exc_info=True)
            raise VoiceBriefError(failure_text("playback failed")) from exc
        if not played:
            logger.warning("Voice brief playback did not complete (guild=%s)", guild_id)
            raise VoiceBriefError(failure_text("playback did not complete"))

        try:
            sent = await adapter.send(str(chat_id), text)
        except Exception as exc:
            logger.warning("Voice brief text post error: %s", exc, exc_info=True)
            raise VoiceBriefError(failure_text("text post failed")) from exc
        if not getattr(sent, "success", False):
            logger.warning("Voice brief text post failed (chat=%s)", chat_id)
            raise VoiceBriefError(failure_text("text post failed"))
        return text
    finally:
        with lock:
            state["abandoned"] = True
            if state["finished"] or not submitted:
                shutil.rmtree(workdir, ignore_errors=True)
