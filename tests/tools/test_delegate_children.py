"""Kept delegated children (``tools.delegate_children``): the record, the keep rule, the run fence, expiry,
orphans and prune protection (plan delegate-resume, e1)."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from tools import delegate_children as dc

DAY = 86400.0


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB
    handle = SessionDB(db_path=tmp_path / "state.db")
    yield handle
    handle.close()


def _parent(db, session_id="parent", source="cli", **extra):
    """A parent agent double whose session row exists with *source*."""
    if db.get_session(session_id) is None:
        db.create_session(session_id, source=source)
    return SimpleNamespace(**{"session_id": session_id, "_session_db": db, "platform": source, "_delegate_depth": 0,
                              **extra})


def _child_session(db, child_sid, parent_sid, subagent_id):
    db.create_session(child_sid, source="subagent", parent_session_id=parent_sid,
                      model_config={"_delegate_from": parent_sid, "_subagent_id": subagent_id})


def _child(child_sid, subagent_id, depth=1):
    return SimpleNamespace(session_id=child_sid, _subagent_id=subagent_id, _delegate_depth=depth)


def _spawn(db, parent, child_sid="kid-1", subagent_id="sa-0-aaaa", origin=dc.ORIGIN_MODEL, start=True):
    """Spawn hook + worker start, as ``_build_child_agent`` and ``_ChildRun.await_child`` call them."""
    _child_session(db, child_sid, parent.session_id, subagent_id)
    child = _child(child_sid, subagent_id)
    with dc.spawn_origin(origin):
        kept = dc.note_spawn(parent, child, goal="find the token", context="ctx", toolsets=["file"], role="leaf",
                             model="m", route={"provider": "p", "model": "m", "api_key": "SECRET"})
    future: Future = Future()
    token = dc.worker_started(child, future, task_index=0) if (kept and start) else None
    return child, future, token, kept


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='delegate_children'").fetchone():
            return []
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM delegate_children")]
    finally:
        conn.close()


def _set(path, child_id, **fields):
    conn = sqlite3.connect(path)
    try:
        sets = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE delegate_children SET {sets} WHERE child_id=?", (*fields.values(), child_id))
        conn.commit()
    finally:
        conn.close()


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _finish(future, result=None):
    future.set_result(result if result is not None else {"completed": True, "final_response": "ok"})


# ── schema / record ─────────────────────────────────────────────────────────

def test_schema_created_on_first_use_through_the_parents_db_path(db):
    path = db.db_path
    assert _rows(path) == []  # no table before a kept child
    parent = _parent(db, source="cli")
    _, future, token, kept = _spawn(db, parent)
    assert kept and token
    rows = _rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["child_id"] == "kid-1" and row["subagent_id"] == "sa-0-aaaa"
    assert row["holder_session_id"] == "parent" and row["conversation_id"] == "parent"
    assert row["status"] == "running" and row["run_token"] == token and row["kept"] == 1
    assert row["goal"] == "find the token" and row["context"] == "ctx" and json.loads(row["toolsets"]) == ["file"]
    assert row["latest_session_id"] == "kid-1" and row["delegation_id"] is None and row["task_index"] == 0
    assert row["profile"] == "default" and row["depth"] == 1
    runner = json.loads(row["runner"])
    assert runner["pid"] == os.getpid()
    # Credentials are never stored.
    assert "SECRET" not in json.dumps(row) and json.loads(row["route"]) == {"provider": "p", "model": "m"}
    _finish(future)


def test_schema_current_v32_state_db_gains_the_table_without_a_version_bump(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_common import SCHEMA_VERSION
    assert SCHEMA_VERSION == 32
    path = tmp_path / "state.db"
    seed = SessionDB(db_path=path)
    seed.create_session("parent", source="desktop")
    seed.close()
    assert _rows(path) == []
    assert dc.ChildStore(path).record_run_start(child_id="kid-32", subagent_id="sa-0-v32", holder_session_id="parent",
                                                conversation_id="parent", brief={"goal": "g"})
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 32
    finally:
        conn.close()
    assert [r["child_id"] for r in _rows(path)] == ["kid-32"]


def test_schema_v31_state_db_fixture_opens_and_gains_the_table(tmp_path):
    """An older (v31-stamped) store: our first use adds the table and leaves the version alone."""
    from hermes_state import SessionDB
    path = tmp_path / "state.db"
    seed = SessionDB(db_path=path)
    seed.create_session("parent", source="desktop")
    seed.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = 31")
    conn.commit()
    conn.close()

    store = dc.ChildStore(path)
    assert store.record_run_start(child_id="kid-31", subagent_id="sa-0-v31", holder_session_id="parent",
                                  conversation_id="parent", brief={"goal": "g"})
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 31  # no SCHEMA_VERSION bump
        assert conn.execute("SELECT status FROM delegate_children").fetchone()[0] == "running"
    finally:
        conn.close()
    reopened = SessionDB(db_path=path)  # the ordinary open migrates v31 -> current with our table present
    try:
        assert reopened.get_session("parent")["source"] == "desktop"
    finally:
        reopened.close()
    assert [r["child_id"] for r in _rows(path)] == ["kid-31"]


def test_record_two_profiles_in_one_process_write_to_their_own_dbs(tmp_path):
    from hermes_state import SessionDB
    dbs = []
    try:
        for name in ("alpha", "beta"):
            home = tmp_path / "profiles" / name
            home.mkdir(parents=True)
            dbs.append(SessionDB(db_path=home / "state.db"))
        futures = []
        for name, handle in zip(("alpha", "beta"), dbs):
            parent = _parent(handle, session_id=f"parent-{name}", source="cli")
            _, future, _, kept = _spawn(handle, parent, child_sid=f"kid-{name}", subagent_id=f"sa-0-{name}")
            assert kept
            futures.append(future)
        for name, handle in zip(("alpha", "beta"), dbs):
            rows = _rows(handle.db_path)
            assert [r["child_id"] for r in rows] == [f"kid-{name}"]
            assert rows[0]["profile"] == name
        for future in futures:
            _finish(future)
    finally:
        for handle in dbs:
            handle.close()


# ── keep rule ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source", ["desktop", "discord"])
def test_keep_children_of_desktop_and_discord_parents(db, source):
    parent = _parent(db, source=source)
    _, future, _, kept = _spawn(db, parent)
    assert kept and [r["child_id"] for r in _rows(db.db_path)] == ["kid-1"]
    _finish(future)


@pytest.mark.parametrize("source", ["webhook", "api_server", "cron", "kanban", "oneshot", "msgraph_webhook",
                                    "relay", "local", "acp"])
def test_keep_not_for_non_interactive_parent_sources(db, source):
    parent = _parent(db, source=source)
    _, _, _, kept = _spawn(db, parent)
    assert not kept and _rows(db.db_path) == []


def test_keep_not_for_a_cron_job_agent_even_with_an_interactive_source(db):
    parent = _parent(db, source="cli", **{dc.NEVER_KEEP_ATTR: True})
    _, _, _, kept = _spawn(db, parent)
    assert not kept and _rows(db.db_path) == []


def test_keep_not_for_a_one_shot_q_run(db, monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    parent = _parent(db, source="cli")  # a -q run whose row still says cli
    _, _, _, kept = _spawn(db, parent)
    assert not kept and _rows(db.db_path) == []


@pytest.mark.parametrize("origin", [dc.ORIGIN_REVIEW, dc.ORIGIN_PLUGIN, None])
def test_keep_never_for_review_plugin_or_unknown_origins(db, origin):
    parent = _parent(db, source="desktop")
    _child_session(db, "kid-1", "parent", "sa-0-aaaa")
    child = _child("kid-1", "sa-0-aaaa")
    if origin is None:
        kept = dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
    else:
        with dc.spawn_origin(origin):
            kept = dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
    assert not kept and dc.worker_started(child, Future()) is None and _rows(db.db_path) == []


def test_keep_review_dispatch_carries_the_review_origin(monkeypatch):
    from agent import review_engine
    from tools import delegate_tool
    seen = {}

    def fake_delegate_task(**kwargs):
        seen.update(kwargs)
        return json.dumps({"status": "dispatched", "delegation_id": "d"})

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    monkeypatch.setattr(review_engine, "_load_review_credentials_cfg", lambda: None)
    review_engine.start_review(SimpleNamespace(), [{"role": "user", "content": "hi"},
                                                   {"role": "assistant", "content": "done: wrote the file"}])
    assert seen["origin"] == dc.ORIGIN_REVIEW


def _stub_delegate_task_build(monkeypatch, on_build):
    """``delegate_task`` up to ``_build_children``, which runs *on_build* and stops the call there."""
    from tools import delegate_tool

    def fake_build_children(*_a, **_k):
        on_build()
        return [], "stop here"

    route = SimpleNamespace(error=None, creds={"model": "m", "provider": None}, refused_entry=lambda i: {})
    monkeypatch.setattr(delegate_tool, "_build_children", fake_build_children)
    monkeypatch.setattr(delegate_tool, "_routes_for", lambda *a, **k: ([route], None))
    monkeypatch.setattr(delegate_tool, "create_live_transcripts", lambda *a, **k: (None, [], []), raising=False)
    import tools.delegation_live_log as live
    monkeypatch.setattr(live, "create_live_transcripts", lambda *a, **k: (None, [], []))


def test_keep_delegate_task_builds_children_under_its_origin(monkeypatch):
    """Only the model's own call (``run_agent._dispatch_delegate_task``) builds under ``model``; a direct caller
    that names no origin builds under None (never kept); an internal caller's origin is used as given."""
    from run_agent import AIAgent
    from tools import delegate_tool
    seen = []
    _stub_delegate_task_build(monkeypatch, lambda: seen.append(dc.current_spawn_origin()))
    parent = SimpleNamespace(session_id="p", _delegate_depth=0, _session_db=None)
    AIAgent._dispatch_delegate_task(parent, {"goal": "g"})
    delegate_tool.delegate_task(goal="g", parent_agent=parent)
    delegate_tool.delegate_task(goal="g", parent_agent=parent, origin=dc.ORIGIN_REVIEW)
    assert seen == [dc.ORIGIN_MODEL, None, dc.ORIGIN_REVIEW]


@pytest.mark.parametrize("door", ["registry", "plugin_context"])
def test_keep_registry_and_plugin_dispatch_of_delegate_task_write_no_row(db, monkeypatch, door):
    """The registry handler is reached only by non-model callers (the model goes through
    ``_dispatch_delegate_task``), chiefly a plugin's ``ctx.dispatch_tool("delegate_task", ...)``: never kept."""
    from tools import delegate_tool  # registers delegate_task
    from tools.registry import registry
    parent = _parent(db, source="desktop")
    outcome = {}

    def on_build():
        _child_session(db, "kid-reg", "parent", "sa-0-reg")
        child = _child("kid-reg", "sa-0-reg")
        outcome["origin"] = dc.current_spawn_origin()
        outcome["kept"] = dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
        outcome["token"] = dc.worker_started(child, Future())

    _stub_delegate_task_build(monkeypatch, on_build)
    args = {"goal": "plugin work", "origin": "model"}  # an ``origin`` arg is not a schema field: ignored
    if door == "registry":
        registry.dispatch("delegate_task", args, parent_agent=parent)
    else:
        from hermes_cli.plugins_context import PluginContext
        ctx = SimpleNamespace(_manager=SimpleNamespace(_cli_ref=None, scope_key=None))
        PluginContext.dispatch_tool(ctx, "delegate_task", args, parent_agent=parent)
    assert outcome == {"origin": None, "kept": False, "token": None}
    assert _rows(db.db_path) == []


def test_keep_subagent_lifecycle_launch_writes_no_row(db, monkeypatch):
    from agent import subagent_lifecycle as sl
    from tools import delegate_tool
    parent = _parent(db, source="desktop")
    outcome = {}

    def fake_build(**kwargs):
        _child_session(db, "kid-plugin", "parent", "sa-0-plug")
        child = _child("kid-plugin", "sa-0-plug")
        outcome["origin"] = dc.current_spawn_origin()
        outcome["kept"] = dc.note_spawn(parent, child, goal=kwargs["goal"], context=None, toolsets=None,
                                        role="leaf", model=None)
        child._delegate_role, child.provider, child.model = "leaf", None, None
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", fake_build)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *a, **k: {
        "model": "m", "provider": None, "base_url": None, "api_key": None, "api_mode": None})
    monkeypatch.setattr(sl, "_EXECUTOR", SimpleNamespace(submit=lambda *a, **k: Future()))
    sl.SubagentLifecycleService(lambda: parent).launch({"goal": "plugin work"})
    assert outcome == {"origin": dc.ORIGIN_PLUGIN, "kept": False}
    assert _rows(db.db_path) == []


def test_keep_grandchild_only_with_keep_true(db):
    parent = _parent(db, source="cli", _delegate_depth=1)
    _child_session(db, "grand", "parent", "sa-0-gr")
    grand = _child("grand", "sa-0-gr", depth=2)
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert not dc.note_spawn(parent, grand, goal="g", context=None, toolsets=None, role="leaf", model=None)
        assert dc.note_spawn(parent, grand, goal="g", context=None, toolsets=None, role="leaf", model=None, keep=True)


def test_keep_sources_config_overrides_the_named_default(db):
    assert {"cli", "tui", "desktop", "discord", "telegram", "slack"} <= dc.DEFAULT_KEEP_SOURCES
    assert not {"api_server", "webhook", "cron", "kanban", "oneshot", "relay", "local", "acp"} & dc.DEFAULT_KEEP_SOURCES
    parent = _parent(db, source="discord")
    assert not dc.should_keep(parent, origin=dc.ORIGIN_MODEL, child_depth=1, cfg={"keep_sources": ["cli"]})
    assert dc.should_keep(parent, origin=dc.ORIGIN_MODEL, child_depth=1, cfg={"keep_sources": None})


def test_keep_via_real_build_child_agent(tmp_path, monkeypatch):
    """The spawn hook in ``_build_child_agent`` with real agents: a desktop parent's child is noted, a webhook
    parent's is not."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    handle = SessionDB(db_path=tmp_path / "state.db")
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter", api_mode="chat_completions",
              model="anthropic/claude-sonnet-4.6", quiet_mode=True, skip_context_files=True, skip_memory=True,
              save_trajectories=False, enabled_toolsets=["file"])
    agents = []
    try:
        for sid, platform in (("p-desktop", "desktop"), ("p-webhook", "webhook")):
            parent = AIAgent(session_id=sid, session_db=handle, platform=platform, **kw)
            agents.append(parent)
            parent._ensure_db_session()
            with dc.spawn_origin(dc.ORIGIN_MODEL):
                child = dt._build_child_agent(task_index=0, goal=f"goal of {platform}", context=None,
                                              toolsets=["file"], model=None, max_iterations=4, task_count=1,
                                              parent_agent=parent)
            agents.append(child)
            child._ensure_db_session()
            future: Future = Future()
            token = dc.worker_started(child, future)
            assert (token is not None) == (platform == "desktop")
            _finish(future)
        rows = _rows(tmp_path / "state.db")
        assert [(r["holder_session_id"], r["goal"]) for r in rows] == [("p-desktop", "goal of desktop")]
        assert rows[0]["child_id"] == agents[1].session_id and rows[0]["subagent_id"] == agents[1]._subagent_id
        assert _wait(lambda: _rows(tmp_path / "state.db")[0]["status"] == "idle")
    finally:
        for agent in reversed(agents):
            agent.close()
        handle.close()


# ── run fence / CAS ─────────────────────────────────────────────────────────

def test_fence_done_callback_sets_idle_with_its_reason(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future, {"completed": False, "interrupted": True})
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    row = store.get("kid-1")
    assert row["idle_reason"] == "interrupted" and row["idle_since"] is not None and row["runner"] is None


def test_fence_two_concurrent_takes_give_one_success(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    barrier, results = threading.Barrier(2), []

    def take():
        barrier.wait()
        results.append(dc.ChildStore(db.db_path).take_fence("kid-1", holder_session_id="parent"))

    threads = [threading.Thread(target=take) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(r.ok for r in results) == [False, True]
    loser = next(r for r in results if not r.ok)
    assert loser.steer_receipt
    winner = next(r for r in results if r.ok)
    assert store.get("kid-1")["run_token"] == winner.token


def test_cas_late_write_from_an_older_run_is_ignored(db):
    parent = _parent(db)
    _, future, old_token, _ = _spawn(db, parent)
    store = dc.ChildStore(db.db_path)
    _finish(future)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    fence = store.take_fence("kid-1")
    assert fence.ok and fence.token != old_token
    assert not store.finish_run("kid-1", old_token, "completed")
    assert not store.beat("kid-1", old_token)
    assert store.get("kid-1")["status"] == "running"
    assert store.finish_run("kid-1", fence.token, "completed")


def test_fence_non_holder_refused(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    result = store.take_fence("kid-1", holder_session_id="someone-else")
    assert not result.ok and result.reason == "not_holder"
    assert store.take_fence("nope").reason == "unknown"


def test_fence_stale_verdict_keeps_a_live_worker_running_until_it_exits(db, monkeypatch):
    """The parent's wait ends on the stale verdict while the worker is alive: the row keeps beating, stays
    ``running`` (``wait_abandoned_at``), a resume gets a steer receipt and a claim is refused; once the worker
    exits it is idle and resumable."""
    from tools import delegate_tool

    class Wedged:
        def __init__(self):
            self.tool_progress_callback = None
            self._credential_pool = None
            self._delegate_saved_tool_names = []
            self._delegate_role = "leaf"
            self._delegate_depth = 1
            self._subagent_id = "sa-0-wedge"
            self.session_id = "kid-wedge"
            self.release = threading.Event()

        def run_conversation(self, **_kwargs):
            self.release.wait()
            return {"final_response": "VERDICT", "completed": True, "api_calls": 3, "messages": []}

        def get_activity_summary(self):
            return {"api_call_count": 3, "current_tool": None, "last_activity_ts": 1000.0, "max_iterations": 50}

        def hard_interrupt(self, *_a, **_k):
            pass

        def close(self):
            self.release.set()

    parent = _parent(db, source="desktop")
    parent.__dict__.update(_current_task_id=None, _active_children=[], _active_children_lock=threading.Lock(),
                           _touch_activity=lambda _d: None, _interrupt_requested=False)
    _child_session(db, "kid-wedge", "parent", "sa-0-wedge")
    child = Wedged()
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
    child._subagent_id = None  # keep the live registry out of this test
    monkeypatch.setattr(dc, "HEARTBEAT_INTERVAL_S", 0.02)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_STALE_CYCLES_IDLE", 2)
    monkeypatch.setattr(delegate_tool, "_STALE_RESULT_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    valve = threading.Timer(30.0, child.release.set)
    valve.daemon = True
    valve.start()
    store = dc.ChildStore(db.db_path)
    try:
        entry = delegate_tool._run_single_child(0, "g", child=child, parent_agent=parent)
        assert entry["status"] == "timeout", entry
        row = store.get("kid-wedge")
        assert row["status"] == "running" and row["wait_abandoned_at"] is not None
        first_beat = row["heartbeat_at"]
        assert _wait(lambda: store.get("kid-wedge")["heartbeat_at"] > first_beat)  # still beating
        resume = store.take_fence("kid-wedge", holder_session_id="parent")
        assert not resume.ok and resume.steer_receipt
        assert store.release("kid-wedge", "parent")
        assert not store.claim("kid-wedge", "other-session", "other-conv")  # a running child is never claimable
    finally:
        child.release.set()
        valve.cancel()
    assert _wait(lambda: store.get("kid-wedge")["status"] == "idle")
    assert store.get("kid-wedge")["idle_reason"] == "completed"
    resume = store.take_fence("kid-wedge", holder_session_id="parent")
    assert resume.ok


# ── lock: hold / release / claim / transfer ─────────────────────────────────

def test_cas_release_claim_hold_and_transfer(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    store = dc.ChildStore(db.db_path)
    assert not store.claim("kid-1", "s2", "c2")  # held
    assert store.release("kid-1", "parent")
    assert not store.claim("kid-1", "s2", "c2")  # released but running
    assert store.hold("kid-1", "parent")  # the holder takes it back
    assert store.release("kid-1", "parent")
    _finish(future)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    assert not store.release("kid-1", "not-the-holder")
    assert store.claim("kid-1", "s2", "c2")
    row = store.get("kid-1")
    assert (row["holder_session_id"], row["conversation_id"], row["released"]) == ("s2", "c2", False)
    assert row["prior_conversations"] == ["parent"]
    assert store.transfer("s2", "s3", "c3") == {"moved": 1, "running": 0}
    row = store.get("kid-1")
    assert (row["holder_session_id"], row["conversation_id"]) == ("s3", "c3")
    assert row["prior_conversations"] == ["parent", "c2"]


def test_record_latest_session_refreshed_through_prior_conversations(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    db.end_session("kid-1", "compression")
    db.create_session("kid-1b", source="subagent", parent_session_id="kid-1",
                      model_config={"_delegate_from": "parent", "_subagent_id": "sa-0-aaaa"})
    store.transfer("parent", "fresh", "fresh-conv")  # the transcript still descends from the old conversation
    assert store.refresh_latest("kid-1") == "kid-1b"
    assert store.get("kid-1")["latest_session_id"] == "kid-1b"
    assert store.show("sa-0-aaaa", ["fresh-conv"])["child_id"] == "kid-1"


def test_record_same_subagent_id_in_two_conversations_resolves_separately(db):
    store = dc.ChildStore(db.db_path)
    futures = []
    for conv, kid in (("conv-a", "kid-a"), ("conv-b", "kid-b")):
        parent = _parent(db, session_id=conv, source="cli")
        _, future, _, kept = _spawn(db, parent, child_sid=kid, subagent_id="sa-0-same")
        assert kept
        futures.append(future)
    assert store.resolve("sa-0-same", ["conv-a"]) == "kid-a"
    assert store.resolve("sa-0-same", ["conv-b"]) == "kid-b"
    assert store.resolve("sa-0-same", ["conv-c"]) is None
    assert store.resolve("kid-a", ["conv-b"]) is None  # a child_id is visible only to its holder's conversation
    for future in futures:
        _finish(future)


# ── expiry ──────────────────────────────────────────────────────────────────

def test_expiry_at_seven_days_plus_one_second(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    t0 = 1_000_000.0
    _set(db.db_path, "kid-1", idle_since=t0)
    assert store.get("kid-1", now=t0 + 7 * DAY)["status"] == "idle"
    assert store.get("kid-1", now=t0 + 7 * DAY + 1)["status"] == "expired"
    assert not store.take_fence("kid-1").ok


def test_expiry_show_does_not_reset_idle_since(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    t0 = time.time() - 6 * DAY
    _set(db.db_path, "kid-1", idle_since=t0)
    for _ in range(3):
        assert store.show("kid-1", ["parent"])["idle_since"] == t0
        assert store.show("sa-0-aaaa", ["parent"])["idle_since"] == t0
    assert store.get("kid-1", now=t0 + 7 * DAY + 1)["status"] == "expired"


def test_expiry_running_child_never_expires(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    store = dc.ChildStore(db.db_path)
    later = time.time() + 30 * DAY
    _set(db.db_path, "kid-1", heartbeat_at=later)  # its tick still beats then
    assert store.get("kid-1", now=later)["status"] == "running"
    _finish(future)


def test_expiry_resume_resets_idle_since(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    fence = store.take_fence("kid-1")
    assert fence.ok and store.get("kid-1")["idle_since"] is None
    store.finish_run("kid-1", fence.token, "completed")
    assert store.get("kid-1")["idle_since"] is not None


# ── orphans ─────────────────────────────────────────────────────────────────

def _running_row(db, child_id="kid-o", subagent_id="sa-0-orph"):
    store = dc.ChildStore(db.db_path)
    token = store.record_run_start(child_id=child_id, subagent_id=subagent_id, holder_session_id="parent",
                                   conversation_id="parent", brief={"goal": "g"})
    assert token
    return store, token


def _process_runner(proc):
    from hermes_state_common import _proc_start_ticks
    from hermes_state_pidns import pid_namespace_id
    ticks = _proc_start_ticks(proc.pid)
    assert ticks is not None
    return {"pid": proc.pid, "pidns": pid_namespace_id(), "start_ticks": ticks}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_orphan_dead_runner_becomes_idle_orphaned(db):
    store, _ = _running_row(db)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    runner = _process_runner(proc)
    proc.kill()
    proc.wait()
    _set(db.db_path, "kid-o", runner=json.dumps(runner), heartbeat_at=time.time())
    row = store.get("kid-o")
    assert (row["status"], row["idle_reason"]) == ("idle", "orphaned") and row["orphaned_at"] is not None
    fence = store.take_fence("kid-o")
    assert fence.ok and fence.was_orphaned


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_orphan_runner_alive_in_another_process_stays_running(db):
    store, _ = _running_row(db)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # Well past orphan_after_s: a provably alive runner is not orphaned on the plain heartbeat age.
        _set(db.db_path, "kid-o", runner=json.dumps(_process_runner(proc)), heartbeat_at=time.time() - 1800)
        assert store.get("kid-o")["status"] == "running"
    finally:
        proc.kill()
        proc.wait()


def test_orphan_foreign_pidns_runner_uses_the_heartbeat_age(db):
    store, _ = _running_row(db)
    foreign = {"pid": 1, "pidns": "pid:[1]-foreign", "start_ticks": 12345}
    t0 = 2_000_000.0
    _set(db.db_path, "kid-o", runner=json.dumps(foreign), heartbeat_at=t0)
    assert store.get("kid-o", now=t0 + 60)["status"] == "running"
    row = store.get("kid-o", now=t0 + 301)
    assert (row["status"], row["idle_reason"]) == ("idle", "orphaned")


def test_orphan_runner_without_start_ticks_uses_the_heartbeat_age(db):
    store, _ = _running_row(db)
    unticked = dict(dc.runner_record(), start_ticks=None)
    t0 = time.time()
    _set(db.db_path, "kid-o", runner=json.dumps(unticked), heartbeat_at=t0)
    assert store.get("kid-o", now=t0 + 60)["status"] == "running"
    assert store.get("kid-o", now=t0 + 301)["idle_reason"] == "orphaned"


def test_orphan_replayed_unknown_completion_carries_the_kept_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent = _parent(handle, source="cli")
        _child_session(handle, "kid-r", "parent", "sa-0-rep")
        child = _child("kid-r", "sa-0-rep")
        child._delegation_id = "deleg_replay"
        with dc.spawn_origin(dc.ORIGIN_MODEL):
            assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
        assert dc.worker_started(child, Future(), task_index=0)
        child._delegate_children_tick.stop()
        ad._persist_dispatch({"delegation_id": "deleg_replay", "parent_session_id": "parent",
                              "dispatched_at": time.time(), "goal": "g", "goals": ["g"]})
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        conn = sqlite3.connect(tmp_path / "state.db")
        conn.execute("UPDATE async_delegations SET owner_pid=?, owner_started_at=NULL", (dead.pid,))
        conn.execute("UPDATE delegate_children SET runner=?", (json.dumps({"pid": dead.pid, "pidns": None}),))
        conn.commit()
        conn.close()
        assert ad.recover_abandoned_delegations() == 1
        conn = sqlite3.connect(tmp_path / "state.db")
        event = json.loads(conn.execute("SELECT event_json FROM async_delegations").fetchone()[0])
        conn.close()
        assert event["status"] == "unknown"
        assert event["error"].endswith(dc.ORPHANED_LINE)
        assert dc.ChildStore(tmp_path / "state.db").get("kid-r")["idle_reason"] == "orphaned"
    finally:
        handle.close()


def test_orphan_replay_without_kept_children_is_unchanged(tmp_path):
    conn = sqlite3.connect(":memory:")
    assert dc.orphaned_unit_note(conn, "deleg_x") is None  # no table: nothing to say


# ── prune protection ────────────────────────────────────────────────────────

def _old_lineage(db):
    """parent (cli) -> kid-1 (kept child) --compression--> kid-2, plus an unkept child; all ended."""
    db.create_session("parent", source="cli")
    _child_session(db, "kid-1", "parent", "sa-0-aaaa")
    db.end_session("kid-1", "compression")
    db.create_session("kid-2", source="subagent", parent_session_id="kid-1",
                      model_config={"_delegate_from": "parent", "_subagent_id": "sa-0-aaaa"})
    _child_session(db, "loose", "parent", "sa-0-loose")
    for sid in ("parent", "kid-2", "loose"):
        db.end_session(sid, "user_exit")
    store = dc.ChildStore(db.db_path)
    token = store.record_run_start(child_id="kid-1", subagent_id="sa-0-aaaa", holder_session_id="parent",
                                   conversation_id="parent", brief={"goal": "g"})
    store.finish_run("kid-1", token, "completed")


def _prune_all(db):
    return db.prune_sessions(older_than_days=None, last_active_before=time.time() + 60)


def test_prune_spares_a_kept_childs_sessions_and_continuation(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _old_lineage(db)
    candidates = {r["id"] for r in db.list_prune_candidates(older_than_days=None, whole_lineages=True,
                                                            last_active_before=time.time() + 60)}
    assert {"kid-1", "kid-2"}.isdisjoint(candidates) and {"parent", "loose"} <= candidates
    _prune_all(db)
    assert db.get_session("kid-1") is not None and db.get_session("kid-2") is not None
    assert db.get_session("loose") is None and db.get_session("parent") is None


def test_prune_without_protection_deletes_kept_transcripts(db, tmp_path):
    (tmp_path / "config.yaml").write_text("delegation:\n  keep_protects_transcripts: false\n")  # the db's own home
    _old_lineage(db)
    _prune_all(db)
    assert db.get_session("kid-1") is None and db.get_session("kid-2") is None


def test_prune_expired_child_is_not_protected(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _old_lineage(db)
    _set(db.db_path, "kid-1", status="expired")
    _prune_all(db)
    assert db.get_session("kid-1") is None and db.get_session("kid-2") is None


def test_prune_without_the_table_is_unchanged(db):
    db.create_session("plain", source="cli")
    db.end_session("plain", "user_exit")
    assert _prune_all(db) == 1
    assert _rows(db.db_path) == []  # prune never creates the table


def test_prune_auto_maintenance_spares_kept_children(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _old_lineage(db)
    conn = sqlite3.connect(db.db_path)
    conn.execute("UPDATE sessions SET started_at = started_at - 400 * 86400, ended_at = ended_at - 400 * 86400")
    conn.commit()
    conn.close()
    db.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)
    assert db.get_session("kid-1") is not None and db.get_session("kid-2") is not None
    assert db.get_session("loose") is None


def test_prune_manual_hermes_sessions_prune_spares_kept_children(tmp_path, monkeypatch, capsys):
    """``hermes sessions prune --yes`` (the path live profiles use: ``sessions.auto_prune`` is off by default)."""
    import argparse
    from hermes_state import SessionDB
    from hermes_cli import sessions_cmd
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        _old_lineage(handle)
        conn = sqlite3.connect(handle.db_path)
        conn.execute("UPDATE sessions SET started_at = started_at - 400 * 86400, ended_at = ended_at - 400 * 86400")
        conn.commit()
        conn.close()
        monkeypatch.setattr(sessions_cmd, "_sessions_dir", lambda: tmp_path / "sessions")
        args = argparse.Namespace(yes=True, dry_run=False)
        assert sessions_cmd._cmd_prune_or_archive(handle, args, "prune") is None
        out = capsys.readouterr().out
        assert "Pruned" in out
        assert handle.get_session("kid-1") is not None and handle.get_session("kid-2") is not None
        assert handle.get_session("loose") is None
    finally:
        handle.close()


# ── review round 1 (plan delegate-resume, e1 r2) ────────────────────────────

def _has_table(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name='delegate_children'").fetchone() is not None
    finally:
        conn.close()


def _idle_kid(db, parent=None):
    parent = parent or _parent(db)
    _, future, token, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    return store, token


# finding 2: the prune guard fails closed

def test_prune_fails_closed_when_the_kept_guard_cannot_be_read(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _old_lineage(db)
    real = db._read_one

    def broken(sql, params=()):
        if "sqlite_master" in sql:
            raise sqlite3.OperationalError("database is locked")
        return real(sql, params)

    monkeypatch.setattr(db, "_read_one", broken)
    with pytest.raises(sqlite3.OperationalError):
        _prune_all(db)
    monkeypatch.setattr(db, "_read_one", real)
    assert db.get_session("kid-1") is not None and db.get_session("kid-2") is not None


# finding 3: malformed knobs fail safe and warn

@pytest.mark.parametrize("raw", ["cli", "cli, tui", {"cli": True}, 5])
def test_keep_sources_malformed_value_keeps_nothing_and_warns(db, raw, caplog):
    dc._WARNED.clear()
    with caplog.at_level("WARNING", logger=dc.__name__):
        assert dc.keep_sources({"keep_sources": raw}) == frozenset()
    assert any("keep_sources" in r.getMessage() and repr(raw) in r.getMessage() for r in caplog.records)
    parent = _parent(db, source="cli")
    assert not dc.should_keep(parent, origin=dc.ORIGIN_MODEL, child_depth=1, cfg={"keep_sources": raw})


def test_keep_sources_null_is_the_default_and_a_list_is_normalized():
    assert dc.keep_sources({"keep_sources": None}) == dc.DEFAULT_KEEP_SOURCES
    assert dc.keep_sources({}) == dc.DEFAULT_KEEP_SOURCES
    assert dc.keep_sources({"keep_sources": ["CLI ", ""]}) == frozenset({"cli"})


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), "30", -1, 0, True])
def test_keep_invalid_numbers_fall_back_with_a_warning(raw, caplog):
    dc._WARNED.clear()
    with caplog.at_level("WARNING", logger=dc.__name__):
        assert dc.keep_idle_days({"keep_idle_days": raw}) == dc.DEFAULT_KEEP_IDLE_DAYS
        assert dc.orphan_after_s({"orphan_after_s": raw}) == dc.DEFAULT_ORPHAN_AFTER_S
    messages = [r.getMessage() for r in caplog.records]
    assert any("keep_idle_days" in m for m in messages) and any("orphan_after_s" in m for m in messages)


def test_keep_valid_numbers_and_unset_are_silent(caplog):
    dc._WARNED.clear()
    with caplog.at_level("WARNING", logger=dc.__name__):
        assert dc.keep_idle_days({"keep_idle_days": 30}) == 30.0
        assert dc.orphan_after_s({}) == dc.DEFAULT_ORPHAN_AFTER_S
    assert not caplog.records


# finding 5: reads take no write lock and run no DDL

def test_record_read_without_the_table_returns_empty_and_creates_nothing(db, tmp_path):
    store = dc.ChildStore(db.db_path)
    assert store.get("x") is None and store.held_by("p") == [] and store.resolve("x", ["p"]) is None
    assert store.show("x", ["p"]) is None
    assert not _has_table(db.db_path)
    missing = tmp_path / "nowhere" / "state.db"
    assert dc.ChildStore(missing).get("x") is None and not missing.exists()


def test_record_read_on_a_read_only_db(db):
    _idle_kid(db)
    os.chmod(db.db_path, 0o444)
    try:
        store = dc.ChildStore(db.db_path)
        assert store.get("kid-1")["status"] == "idle"
        assert [r["child_id"] for r in store.held_by("parent")] == ["kid-1"]
    finally:
        os.chmod(db.db_path, 0o644)


def test_record_read_does_not_wait_on_the_writer_lock(db):
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)  # running, runner = this live process: no rule applies
    store = dc.ChildStore(db.db_path)
    blocker = sqlite3.connect(db.db_path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        assert store.get("kid-1")["status"] == "running"
        assert time.monotonic() - started < 2.0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    _finish(future)


def test_record_read_still_applies_a_due_rule(db):
    store, _ = _idle_kid(db)
    _set(db.db_path, "kid-1", idle_since=1_000_000.0)
    assert store.get("kid-1", now=1_000_000.0 + 8 * DAY)["status"] == "expired"
    assert _rows(db.db_path)[0]["status"] == "expired"  # written, not only reported


# finding 6: a finished worker always leaves running

def test_fence_done_callback_retries_a_failed_final_write(db, monkeypatch):
    monkeypatch.setattr(dc, "FINISH_RETRY_S", 0.01)
    real, fails = dc.ChildStore.finish_run, {"n": dc.FINISH_ATTEMPTS - 1}

    def flaky(self, *a, **k):
        if fails["n"]:
            fails["n"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real(self, *a, **k)

    monkeypatch.setattr(dc.ChildStore, "finish_run", flaky)
    _, future, _, _ = _spawn(db, _parent(db))
    _finish(future)
    assert dc.ChildStore(db.db_path).get("kid-1")["status"] == "idle"


def test_fence_tick_retries_the_final_write_after_the_callback_gives_up(db, monkeypatch):
    monkeypatch.setattr(dc, "FINISH_RETRY_S", 0.0)
    monkeypatch.setattr(dc, "HEARTBEAT_INTERVAL_S", 0.02)
    real, fails = dc.ChildStore.finish_run, {"n": dc.FINISH_ATTEMPTS + 3}

    def flaky(self, *a, **k):
        if fails["n"]:
            fails["n"] -= 1
            raise sqlite3.OperationalError("disk I/O error")
        return real(self, *a, **k)

    monkeypatch.setattr(dc.ChildStore, "finish_run", flaky)
    child, future, _, _ = _spawn(db, _parent(db))
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    assert _wait(lambda: not child._delegate_children_tick.thread.is_alive())


def test_fence_tick_start_failure_still_finishes_the_run(db, monkeypatch):
    def no_thread(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(dc._Tick, "start", no_thread)
    _, future, token, kept = _spawn(db, _parent(db))
    assert kept and token
    _finish(future)
    assert dc.ChildStore(db.db_path).get("kid-1")["status"] == "idle"


def test_orphan_alive_runner_whose_tick_died_orphans_after_the_stale_horizon(db):
    """A live runner whose final write was lost: its heartbeat stops and goes stale."""
    store, _ = _running_row(db)  # runner = this process, provably alive
    t0 = time.time()
    _set(db.db_path, "kid-o", heartbeat_at=t0)
    stale = dc.DEFAULT_ORPHAN_AFTER_S * dc.STALE_ALIVE_FACTOR
    assert store.get("kid-o", now=t0 + dc.DEFAULT_ORPHAN_AFTER_S + 1)["status"] == "running"
    assert store.get("kid-o", now=t0 + stale - 1)["status"] == "running"
    row = store.get("kid-o", now=t0 + stale + 1)
    assert (row["status"], row["idle_reason"], row["idle_since"]) == ("idle", "orphaned", t0)


# finding 7: "provably alive" needs matching start ticks

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_orphan_reused_pid_with_other_start_ticks_is_dead(db):
    from hermes_state_pidns import pid_namespace_id
    store, _ = _running_row(db)
    beat = time.time()
    _set(db.db_path, "kid-o", heartbeat_at=beat,
         runner=json.dumps({"pid": 1, "pidns": pid_namespace_id(), "start_ticks": 123456789}))
    row = store.get("kid-o")
    assert (row["status"], row["idle_reason"]) == ("idle", "orphaned")
    assert row["idle_since"] == beat  # finding 15: the run ended at its last heartbeat


def test_orphan_unreadable_runner_falls_to_the_heartbeat_age(db, monkeypatch):
    import hermes_state_common
    from hermes_state_pidns import pid_namespace_id
    store, _ = _running_row(db)
    t0 = time.time()
    _set(db.db_path, "kid-o", heartbeat_at=t0,
         runner=json.dumps({"pid": 424242, "pidns": pid_namespace_id(), "start_ticks": 5}))

    def eperm(_pid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(hermes_state_common.os, "kill", eperm)
    monkeypatch.setattr(hermes_state_common, "_proc_start_ticks", lambda _pid: None)
    assert store.get("kid-o", now=t0 + 60)["status"] == "running"
    assert store.get("kid-o", now=t0 + 301)["idle_reason"] == "orphaned"


# finding 9: the replayed completion is keyed on the async ledger's unit id

@pytest.mark.parametrize("independent", [False, True])
def test_orphan_async_units_stamp_the_ledger_unit_id_on_their_children(monkeypatch, independent):
    """Live-transcript creation failed (no live id): every unit still gets its own ledger id, and its children
    carry it."""
    from tools import delegate_tool_config, delegate_tool_dispatch as dd
    seen = []

    def dispatch(unit, unit_id, _slot_key, _routing):
        seen.append((unit_id, [getattr(c, "_delegation_unit_id", None) for _, _, c in unit.children]))
        return {"status": "dispatched", "delegation_id": unit_id}

    monkeypatch.setattr(dd, "_dispatch_unit", dispatch)
    monkeypatch.setattr(dd, "_resolve_async_wake_sid", lambda *a, **k: "wake")
    monkeypatch.setattr(dd, "_resolve_async_session_key", lambda *a, **k: ("key", "ui"))
    monkeypatch.setattr(delegate_tool_config, "_get_independent_completions", lambda: independent)
    tasks = [{"goal": "a"}, {"goal": "b"}]
    kids = [SimpleNamespace(session_id=f"k{i}") for i in range(2)]
    batch = dd._Batch(tasks, [(i, t, kids[i]) for i, t in enumerate(tasks)], SimpleNamespace(session_id="p"),
                      {"model": "m"}, None, "leaf", 2, None, [None, None], [], "wake", "ui", None, None, False,
                      time.time())
    dd._dispatch_background(batch)
    assert len(seen) == (2 if independent else 1)
    assert all(unit_id and ids == [unit_id] * len(ids) for unit_id, ids in seen)
    assert len({unit_id for unit_id, _ in seen}) == len(seen)


def test_orphan_replay_keyed_on_the_unit_id_without_a_live_delegation_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent = _parent(handle, source="cli")
        _child_session(handle, "kid-u", "parent", "sa-0-unit")
        child = _child("kid-u", "sa-0-unit")
        child._delegation_unit_id = "deleg_unit"  # live transcripts failed: no child._delegation_id
        with dc.spawn_origin(dc.ORIGIN_MODEL):
            assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
        assert dc.worker_started(child, Future(), task_index=0)
        child._delegate_children_tick.stop()
        assert _rows(tmp_path / "state.db")[0]["delegation_id"] == "deleg_unit"
        ad._persist_dispatch({"delegation_id": "deleg_unit", "parent_session_id": "parent",
                              "dispatched_at": time.time(), "goal": "g", "goals": ["g"]})
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        conn = sqlite3.connect(tmp_path / "state.db")
        conn.execute("UPDATE async_delegations SET owner_pid=?, owner_started_at=NULL", (dead.pid,))
        conn.execute("UPDATE delegate_children SET runner=?", (json.dumps({"pid": dead.pid, "pidns": None}),))
        conn.commit()
        conn.close()
        assert ad.recover_abandoned_delegations() == 1
        conn = sqlite3.connect(tmp_path / "state.db")
        event = json.loads(conn.execute("SELECT event_json FROM async_delegations").fetchone()[0])
        conn.close()
        assert event["error"].endswith(dc.ORPHANED_LINE)
    finally:
        handle.close()


# finding 10: idle_reason agrees with the parent-visible status

def _done(result=None, exc=None, cancel=False):
    future: Future = Future()
    if cancel:
        future.cancel()
    elif exc is not None:
        future.set_exception(exc)
    else:
        future.set_result(result)
    return future


@pytest.mark.parametrize(("future", "reason"), [
    (lambda: _done({"completed": True, "final_response": "ok"}), "completed"),
    (lambda: _done({"completed": True, "failed": True}), "interrupted"),
    (lambda: _done({"completed": False, "failed": True}), "interrupted"),
    (lambda: _done({"completed": False}), "interrupted"),  # max_iterations
    (lambda: _done({"completed": True, "interrupted": True}), "interrupted"),
    (lambda: _done("text"), "interrupted"),
    (lambda: _done(exc=RuntimeError("boom")), "interrupted"),
    (lambda: _done(cancel=True), "interrupted"),
])
def test_fence_done_reason_completed_only_for_a_clean_completion(future, reason):
    assert dc._done_reason(future()) == reason


# finding 11: an AsyncSessionDB parent

def test_record_async_session_db_parent_is_unwrapped_for_lineage_reads(db):
    from hermes_state import AsyncSessionDB
    parent = _parent(db)
    parent._session_db = AsyncSessionDB(db)
    _, future, token, kept = _spawn(db, parent)
    assert kept and token
    store = dc.ChildStore.for_agent(parent)
    assert store is not None and store._session_db is db
    _finish(future)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    assert store.resolve("sa-0-aaaa", ["parent"]) == "kid-1"
    assert store.refresh_latest("kid-1") == "kid-1"


# finding 12: the done-callback is one CAS; the lineage read moves to the read

def test_cas_done_callback_does_no_lineage_read(db, monkeypatch):
    calls = []
    monkeypatch.setattr(dc.ChildStore, "refresh_latest", lambda self, child_id: calls.append(child_id))
    store, _ = _idle_kid(db)
    assert calls == []
    store.show("kid-1", ["parent"])
    assert calls == ["kid-1"]


# finding 13: salvage keeps the rows

def test_record_session_recovery_copies_kept_children(tmp_path):
    from hermes_cli.session_recovery import inspect_session_database, recover_session_database
    from hermes_state import SessionDB
    source, output = tmp_path / "state.db", tmp_path / "recovered.db"
    seed = SessionDB(db_path=source)
    seed.create_session("parent", source="cli")
    seed.close()
    assert dc.ChildStore(source).record_run_start(child_id="kid", subagent_id="sa-0-rec", holder_session_id="parent",
                                                  conversation_id="parent", brief={"goal": "g"})
    inspection = inspect_session_database(source, work_dir=tmp_path)
    assert inspection["tables"]["delegate_children"]["rows"] == 1
    report = recover_session_database(source, output, work_dir=tmp_path)
    assert report["copy"]["delegate_children"]["status"] == "complete"
    assert report["copy"]["delegate_children"]["copied_rows"] == 1
    assert report["complete"] is True and report["verified"] is True
    assert [(r["child_id"], r["subagent_id"]) for r in _rows(output)] == [("kid", "sa-0-rec")]


def test_record_session_recovery_without_kept_children_is_complete(tmp_path):
    from hermes_cli.session_recovery import recover_session_database
    from hermes_state import SessionDB
    source = tmp_path / "state.db"
    seed = SessionDB(db_path=source)
    seed.create_session("parent", source="cli")
    seed.close()
    report = recover_session_database(source, tmp_path / "recovered.db", work_dir=tmp_path)
    assert report["complete"] is True and report["verified"] is True


# finding 16: protection follows the pruned db's own profile

def test_prune_protection_follows_the_pruned_dbs_own_profile_config(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    off, on = tmp_path / "profiles" / "off", tmp_path / "profiles" / "on"
    for home in (off, on):
        home.mkdir(parents=True)
    (off / "config.yaml").write_text("delegation:\n  keep_protects_transcripts: false\n")
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {"keep_protects_transcripts": True})  # the launch profile
    handle = SessionDB(db_path=off / "state.db")
    try:
        _old_lineage(handle)
        _prune_all(handle)
        assert handle.get_session("kid-1") is None
    finally:
        handle.close()
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {"keep_protects_transcripts": False})
    handle = SessionDB(db_path=on / "state.db")
    try:
        _old_lineage(handle)
        _prune_all(handle)
        assert handle.get_session("kid-1") is not None and handle.get_session("kid-2") is not None
    finally:
        handle.close()


# finding 17: ACP transport args are never stored

def test_record_route_drops_acp_transport_fields(db):
    parent = _parent(db)
    _child_session(db, "kid-acp", "parent", "sa-0-acp")
    child = _child("kid-acp", "sa-0-acp")
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model="m",
                             route={"provider": "p", "acp_command": "agent", "acp_args": ["--token", "tok-123"]})
    future: Future = Future()
    assert dc.worker_started(child, future)
    row = _rows(db.db_path)[0]
    assert json.loads(row["route"]) == {"provider": "p"} and "tok-123" not in json.dumps(row)
    _finish(future)


# finding 19: worker-process parents

def test_keep_not_for_a_worker_process_parent():
    """A managed worker's parent writes through ``RuntimeSessionStore`` (no local ``state.db`` path): not kept."""
    from agent.runtime_session_store import RuntimeSessionStore
    parent = SimpleNamespace(session_id="parent", _session_db=object.__new__(RuntimeSessionStore), platform="cli",
                             _delegate_depth=0)
    assert dc.parent_db_path(parent) is None
    assert not dc.should_keep(parent, origin=dc.ORIGIN_MODEL, child_depth=1)
    child = _child("kid-w", "sa-0-w")
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert not dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
    assert dc.worker_started(child, Future()) is None


# finding 20 (e7 part): holder and conversation are read when the row is inserted

def test_record_holder_is_read_when_the_worker_starts(db):
    parent = _parent(db, session_id="before-new")
    _child_session(db, "kid-h", "before-new", "sa-0-h")
    child = _child("kid-h", "sa-0-h")
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
    db.create_session("after-new", source="cli")
    parent.session_id = "after-new"  # a /new between the spawn and the worker start
    future: Future = Future()
    assert dc.worker_started(child, future)
    row = _rows(db.db_path)[0]
    assert (row["holder_session_id"], row["conversation_id"]) == ("after-new", "after-new")
    _finish(future)
