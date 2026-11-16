"""Whisper's silence phrases on the Discord voice path.

"Thank you very much", "thank you so much" and "thanks a lot" are hallucinations Whisper emits on
near-silent audio, and a Discord voice session hits them constantly. The Discord path drops them
without touching the shared set: the CLI, ``tools/transcription_chunking.py`` and
``tools/voice_client_config.py`` read that set and must keep their behaviour.

Every case drives the real ``DiscordAdapter._process_voice_input``: only STT and the PCM->WAV step
are stubbed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord import adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter

GUILD = 111        # stand-in snowflake
CHANNEL = 5        # stand-in snowflake: the text channel the voice input was captured for
USER = 42
PCM = b"\x00" * 96000


def _adapter() -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_text_channels = {GUILD: CHANNEL}
    adapter._voice_input_callback = AsyncMock()
    return adapter


def _stub_stt(monkeypatch, transcript: str) -> None:
    def pcm_to_wav(pcm, path):
        with open(path, "wb") as fh:
            fh.write(b"RIFF")

    monkeypatch.setattr(adapter_mod.VoiceReceiver, "pcm_to_wav", pcm_to_wav)
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio",
                        lambda path: {"success": True, "transcript": transcript})


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", [
    pytest.param("Thank you very much.", id="drop_very_much"),
    pytest.param("Thank you so much!", id="drop_so_much"),
    pytest.param("Thanks a lot", id="drop_a_lot"),
    pytest.param("THANK YOU VERY MUCH", id="drop_caps"),
])
async def test_silence_phrase_is_dropped(monkeypatch, transcript):
    adapter = _adapter()
    _stub_stt(monkeypatch, transcript)

    await adapter._process_voice_input(GUILD, USER, PCM, CHANNEL)

    adapter._voice_input_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_correct_is_kept(monkeypatch):
    adapter = _adapter()
    _stub_stt(monkeypatch, "Correct.")

    await adapter._process_voice_input(GUILD, USER, PCM, CHANNEL)

    adapter._voice_input_callback.assert_awaited_once_with(
        guild_id=GUILD, user_id=USER, transcript="Correct.",
    )


def test_shared_set_is_unchanged():
    """The three phrases live on the Discord path only; the shared set keeps a CLI reader's shape."""
    from tools.voice_mode_transcript import WHISPER_HALLUCINATIONS, is_whisper_hallucination

    assert WHISPER_HALLUCINATIONS == {
        "thank you", "thanks for watching", "subscribe to my channel", "like and subscribe",
        "please subscribe", "thank you for watching", "bye", "you", "the end",
        "продолжение следует", "sous-titres", "sous-titres réalisés par la communauté d'amara.org",
        "sottotitoli creati dalla comunità amara.org", "untertitel von stephanie geiges",
        "amara.org", "www.mooji.org", "ご視聴ありがとうございました",
    }
    assert len(WHISPER_HALLUCINATIONS) == 17
    for phrase in ("thank you very much", "thank you so much", "thanks a lot"):
        assert is_whisper_hallucination(phrase) is False
