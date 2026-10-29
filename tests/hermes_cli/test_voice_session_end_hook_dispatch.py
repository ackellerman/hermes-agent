"""``on_voice_session_end`` hook machinery: registration, owner attribution, per-session timeout keying.

Real discovery of on-disk probe plugins (``tests/fakes/voice_session_end_probe.py``) in a temp home;
the hook reaches a callback only through ``PluginManager`` dispatch.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from hermes_cli import plugins as plugins_mod
from tests.fakes import voice_session_end_probe as probe

HOOK = "on_voice_session_end"


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    yield home
    probe.reset()
    plugins_mod._reset_plugin_managers_for_tests()


def _payload(session: str) -> dict:
    return {"platform": "discord", "voice_session_id": session, "reason": "command"}


def test_the_hook_is_known_and_results_name_their_plugin(home, caplog):
    probe.install_probe(home, "alpha", returns={"followup_turn": "from alpha"})
    probe.install_probe(home, "beta", returns={"followup_turn": "from beta"})
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manager = probe.discover(home)
    assert not [r for r in caplog.records if "unknown hook" in r.getMessage()]
    # A callback no plugin registered (shell hooks append straight into the mapping).
    manager._hooks[HOOK].append(lambda **_kw: "shell")

    attributed = manager.invoke_hook_attributed(HOOK, **_payload("s1"))

    assert sorted(attributed, key=repr) == sorted(
        [("alpha", {"followup_turn": "from alpha"}), ("beta", {"followup_turn": "from beta"}), (None, "shell")],
        key=repr)
    # The unattributed invoker keeps its contract: bare results, same order.
    assert manager.invoke_hook(HOOK, **_payload("s2")) == [ret for _owner, ret in attributed]
    assert {r["plugin"] for r in probe.records(home)} == {"alpha", "beta"}
    assert all(r["telemetry_schema_version"] for r in probe.records(home))


def test_owner_map_follows_unload_of_one_plugin_and_of_all(home):
    probe.install_probe(home, "alpha", returns="a")
    probe.install_probe(home, "beta", returns="b")
    manager = probe.discover(home)
    assert sorted(owner for _cb, owner in manager._hook_owner_keys.values()) == ["alpha", "beta"]

    manager.unload("alpha")
    assert [owner for _cb, owner in manager._hook_owner_keys.values()] == ["beta"]
    assert manager.invoke_hook_attributed(HOOK, **_payload("s1")) == [("beta", "b")]

    manager.unload()
    assert manager._hook_owner_keys == {}


@pytest.mark.asyncio
async def test_concurrent_session_ends_each_reach_the_callback(home):
    probe.install_probe(home, "alpha", returns="done", block=["s1"])
    manager = probe.discover(home)
    first = asyncio.ensure_future(asyncio.to_thread(manager.invoke_hook_attributed, HOOK, **_payload("s1")))
    try:
        await asyncio.to_thread(probe.started(home, "alpha").wait, 10)

        # Another voice session ending while s1's callback still runs is different work.
        assert await asyncio.to_thread(manager.invoke_hook_attributed, HOOK, **_payload("s2")) == [("alpha", "done")]
        # Control: the SAME session again is a duplicate of the running call and is skipped.
        assert await asyncio.to_thread(manager.invoke_hook_attributed, HOOK, **_payload("s1")) == []
    finally:
        probe.gate(home, "alpha").set()
    assert await first == [("alpha", "done")]
    assert [r["voice_session_id"] for r in probe.records(home)] == ["s1", "s2"]


def test_a_timed_out_session_end_does_not_suppress_the_next_one(home, caplog):
    probe.install_probe(home, "alpha", returns="done", block=["s1"])
    manager = probe.discover(home)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            assert manager.invoke_hook_attributed(HOOK, timeout=1.0, **_payload("s1")) == []
        assert any("timed out" in r.getMessage() for r in caplog.records)

        assert manager.invoke_hook_attributed(HOOK, timeout=1.0, **_payload("s2")) == [("alpha", "done")]
    finally:
        probe.gate(home, "alpha").set()
    assert [r["voice_session_id"] for r in probe.records(home)] == ["s1", "s2"]
