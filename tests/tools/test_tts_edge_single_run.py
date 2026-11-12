"""Edge TTS must synthesize once: a RuntimeError raised inside the worker (the SDK failing) is a
synthesis failure, not a reason to run the generator a second time in the caller thread."""

from __future__ import annotations

import concurrent.futures
import contextvars
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
from tools import tts_tool

_MARK: contextvars.ContextVar[str] = contextvars.ContextVar("edge_run_once_mark", default="root")


def test_worker_runtime_error_propagates_after_a_single_run(monkeypatch, tmp_path):
    constructed = []

    class _Communicate:
        def __init__(self, text, **kwargs):
            constructed.append(text)

        async def save(self, path):
            raise RuntimeError("edge sdk blew up")

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=_Communicate))

    with pytest.raises(RuntimeError, match="edge sdk blew up"):
        tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})

    assert constructed == ["hello"]  # one run only: no duplicate save/network side effect
    assert not (tmp_path / "out.mp3").exists()


def test_inline_fallback_carries_the_callers_context(monkeypatch, tmp_path):
    """No worker thread available -> run inline, which must still see the caller's profile scope."""
    monkeypatch.setenv("EDGE_RUN_ONCE_SECRET", "root-secret")
    seen = []

    class _Communicate:
        def __init__(self, text, **kwargs):
            seen.append((_MARK.get(), get_secret("EDGE_RUN_ONCE_SECRET")))

        async def save(self, path):
            Path(path).write_bytes(b"ID3fake")

    def _no_threads(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=_Communicate))
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _no_threads)
    mark_token = _MARK.set("seat")
    secret_token = set_secret_scope({"EDGE_RUN_ONCE_SECRET": "seat-secret"})
    try:
        tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})
    finally:
        reset_secret_scope(secret_token)
        _MARK.reset(mark_token)

    assert seen == [("seat", "seat-secret")]
    assert (tmp_path / "out.mp3").read_bytes() == b"ID3fake"
