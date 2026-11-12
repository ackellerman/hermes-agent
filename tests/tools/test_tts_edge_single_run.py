"""Edge TTS must synthesize once: a RuntimeError raised inside the worker (the SDK failing) is a
synthesis failure, not a reason to run the generator a second time in the caller thread."""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
from tools import tts_tool

_MARK: contextvars.ContextVar[str] = contextvars.ContextVar("edge_run_once_mark", default="root")


def test_worker_runtime_error_propagates_after_a_single_run(monkeypatch, tmp_path):
    constructed = []
    saves = []

    class _Communicate:
        def __init__(self, text, **kwargs):
            constructed.append(text)

        async def save(self, path):
            saves.append(path)
            raise RuntimeError("edge sdk blew up")

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=_Communicate))

    with pytest.raises(RuntimeError, match="edge sdk blew up"):
        tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})

    assert constructed == ["hello"]  # one run only: no duplicate save/network side effect
    assert saves == [str(tmp_path / "out.mp3")]


def test_inline_fallback_carries_the_callers_context(monkeypatch, tmp_path):
    """No worker thread available -> run inline, which must still see the caller's profile scope."""
    monkeypatch.setenv("EDGE_RUN_ONCE_SECRET", "root-secret")
    seen = []
    saves = []

    class _Communicate:
        def __init__(self, text, **kwargs):
            seen.append((_MARK.get(), get_secret("EDGE_RUN_ONCE_SECRET")))

        async def save(self, path):
            saves.append(path)
            Path(path).write_bytes(b"ID3fake")

    def _no_threads(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=_Communicate))
    mark_token = _MARK.set("seat")
    secret_token = set_secret_scope({"EDGE_RUN_ONCE_SECRET": "seat-secret"})
    try:
        # The pool starts no thread until submit(), so refusing Thread.start makes submit() raise.
        with monkeypatch.context() as m:
            m.setattr(threading.Thread, "start", _no_threads)
            tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})
    finally:
        reset_secret_scope(secret_token)
        _MARK.reset(mark_token)

    assert seen == [("seat", "seat-secret")]  # generator constructed exactly once, in the caller's scope
    assert saves == [str(tmp_path / "out.mp3")]
    assert (tmp_path / "out.mp3").read_bytes() == b"ID3fake"


def test_non_runtime_error_from_submit_still_shuts_the_pool_down(monkeypatch, tmp_path):
    """Only RuntimeError means "no worker, run inline"; anything else propagates, pool shut down."""
    shutdowns = []
    real_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def _spy_shutdown(self, *args, **kwargs):
        shutdowns.append(kwargs.get("wait", args[0] if args else True))
        return real_shutdown(self, *args, **kwargs)

    def _out_of_memory(self):
        raise MemoryError("no memory for a thread")

    monkeypatch.setattr(
        tts_tool, "_import_edge_tts", lambda: pytest.fail("generator must not run inline for a MemoryError")
    )
    with monkeypatch.context() as m:
        m.setattr(concurrent.futures.ThreadPoolExecutor, "shutdown", _spy_shutdown)
        m.setattr(threading.Thread, "start", _out_of_memory)
        with pytest.raises(MemoryError):
            tts_tool._run_edge_tts("hello", str(tmp_path / "out.mp3"), {})

    assert shutdowns == [True]
