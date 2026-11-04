"""Spoken PRE-TURN brief for Discord voice-input turns.

Opt-in per profile (``discord.voice_brief.enabled``, default off). Before a voice turn runs,
one bounded auxiliary call to the turn's OWN routed provider/model asks for one short sentence,
in the model's own words, stating its understanding and what it will do next. That sentence is
synthesized with the profile's configured TTS provider, played in the turn's bound voice
channel (awaited), and posted to the bound text channel. Only then does the turn run — exactly
as it does without this feature: no gated tools, no muted streaming, no rewritten history.

Any failure (model, TTS, playback, text post) returns a visible failure text instead of running
the turn. The brief is never invented: a canned phrase is not a fallback.

Fakes a test needs live only at the edges: the platform adapter (playback / text post), the TTS
provider, and the model endpoint.
"""

from __future__ import annotations

import asyncio
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

_INSTRUCTIONS = (
    "In one brief, natural sentence, tell the requester what you understood and what you are going "
    "to do now. Address them directly in your own words. Avoid answering the request itself. "
    f"Keep it under {MAX_BRIEF_CHARS} characters."
)


class VoiceBriefError(Exception):
    """Delivery failed; the message is user-visible and the turn must not run."""


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


async def _ask_model(route: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """One auxiliary call on the turn's routed provider/model. No tools are offered at all.

    No ``max_tokens``: reasoning models (DeepSeek V4 on Ollama Cloud) bill thinking against it, and a
    small cap returned an empty sentence with finish_reason=length. ``MODEL_TIMEOUT_S`` is the only
    bound on the call; ``MAX_BRIEF_CHARS`` still rejects an over-long sentence.
    """
    from agent.auxiliary_client import async_call_llm

    response = await async_call_llm(
        task=None,
        provider=route.get("provider") or None,
        model=route.get("model") or None,
        base_url=route.get("base_url") or None,
        api_key=route.get("api_key") or None,
        messages=messages,
        temperature=0.2,
        timeout=MODEL_TIMEOUT_S,
        reasoning_config=dict(MODEL_REASONING),
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
    starts the turn. Raises ``VoiceBriefError`` (user-visible) on any failure.
    """
    try:
        raw = await asyncio.wait_for(_ask_model(route, brief_messages(message)), timeout=MODEL_TIMEOUT_S)
    except VoiceBriefError:
        raise
    except Exception as exc:
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
            future = asyncio.get_running_loop().run_in_executor(None, synthesize_owned)
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
