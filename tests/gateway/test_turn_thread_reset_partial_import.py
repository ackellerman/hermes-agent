"""reset_reused_thread_state must survive a module that another thread is still importing.

A module's first import puts it in ``sys.modules`` before its body runs. Gateway pid 124722 hit
this at 15:21:26 on 2026-11-16: a parked worker reset while ``tools.terminal_tool_sudo`` was
half-built, raised ``AttributeError`` on ``_sudo_password_cache_lock``, and the executor retired
the worker. A half-built module holds no state for this thread, so the reset skips it and still
clears the rest.
"""

from __future__ import annotations

import sys
import threading
import types

from gateway.turn_thread_reset import reset_reused_thread_state


def _half_built(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _local_with_state() -> threading.local:
    local = threading.local()
    local.value = "turn 1"
    return local


def test_half_built_sudo_and_interrupt_modules_do_not_stop_the_reset(monkeypatch):
    local = _local_with_state()
    monkeypatch.setitem(sys.modules, "tools.terminal_tool_sudo", _half_built("tools.terminal_tool_sudo"))
    monkeypatch.setitem(sys.modules, "tools.interrupt", _half_built("tools.interrupt"))
    monkeypatch.setitem(sys.modules, "agent.local_prompt_progress",
                        _half_built("agent.local_prompt_progress", _reading=local))

    reset_reused_thread_state()  # must not raise

    assert local.__dict__ == {}  # the per-turn locals after the sudo step were still cleared


def test_cache_assigned_but_lock_not_yet_is_skipped(monkeypatch):
    # The seeded password cannot exist in practice: every writer is defined after the lock, and a
    # first import has had no turn to write. It is seeded only to show the half-built branch is a
    # no-op, not a weakened guarantee.
    cache = {f"thread:{threading.get_ident()}": "pw"}
    monkeypatch.setitem(sys.modules, "tools.terminal_tool_sudo",
                        _half_built("tools.terminal_tool_sudo", _sudo_password_cache=cache))

    reset_reused_thread_state()

    assert cache  # no lock yet: nothing touched, nothing raised


def test_a_fully_built_sudo_module_still_loses_this_threads_password(monkeypatch):
    tid = threading.get_ident()
    cache = {f"thread:{tid}": "pw", "thread:other": "keep"}
    monkeypatch.setitem(sys.modules, "tools.terminal_tool_sudo",
                        _half_built("tools.terminal_tool_sudo", _sudo_password_cache=cache,
                                    _sudo_password_cache_lock=threading.Lock()))

    reset_reused_thread_state()

    assert cache == {"thread:other": "keep"}
