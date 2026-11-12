"""Edge TTS runs its event loop in a pool thread; the caller's contextvars (profile secret scope) must
reach the SDK call there, or Communicate sees the root credentials/home."""

from __future__ import annotations

import contextvars
from pathlib import Path
from types import SimpleNamespace

from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
from tools import tts_tool

_MARK: contextvars.ContextVar[str] = contextvars.ContextVar("edge_ctx_mark", default="root")


def test_edge_run_sees_the_callers_context(monkeypatch, tmp_path):
    monkeypatch.setenv("EDGE_CTX_SECRET", "root-secret")
    seen = []

    class _Communicate:
        def __init__(self, text, **kwargs):
            seen.append((_MARK.get(), get_secret("EDGE_CTX_SECRET")))

        async def save(self, path):
            Path(path).write_bytes(b"ID3fake")

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=_Communicate))
    mark_token = _MARK.set("seat")
    secret_token = set_secret_scope({"EDGE_CTX_SECRET": "seat-secret"})
    try:
        tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})
    finally:
        reset_secret_scope(secret_token)
        _MARK.reset(mark_token)

    assert seen == [("seat", "seat-secret")]
    assert (tmp_path / "out.mp3").read_bytes() == b"ID3fake"
