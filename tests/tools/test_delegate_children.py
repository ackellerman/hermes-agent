# health: allow FILE_LINES -- the plan (delegate-resume) keeps one test module per element and pins it in every
# element's verify command (``run_tests.sh tests/tools/test_delegate_children.py -k <topic>``, e1-e7); e3's
# list/scope/query tests must be selectable HERE, so a topical split would silence the plan's own verification.
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
    # A running child with a fresh heartbeat is never expired, however far ahead the reader's clock is.
    assert store.get("kid-1", now=time.time() + 30 * DAY)["status"] == "running"
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
        # Even a day-old heartbeat: a provably alive runner is never orphaned on age (D-005 option 2).
        _set(db.db_path, "kid-o", runner=json.dumps(_process_runner(proc)), heartbeat_at=time.time() - DAY)
        assert store.get("kid-o")["status"] == "running"
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_orphan_a_run_that_died_before_keep_idle_days_is_expired_on_the_first_read(db):
    """Expiry is evaluated after the orphan pass, in the same write: one state, not idle-orphaned then expired."""
    store, _ = _running_row(db)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    runner = _process_runner(proc)
    proc.kill()
    proc.wait()
    _set(db.db_path, "kid-o", runner=json.dumps(runner), heartbeat_at=time.time() - 10 * DAY)
    first = store.get("kid-o")
    assert (first["status"], first["idle_reason"]) == ("expired", "orphaned")
    assert store.get("kid-o")["status"] == "expired"  # stable: the next read agrees
    assert store.take_fence("kid-o").reason == "expired"
    assert _rows(db.db_path)[0]["status"] == "expired"


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
        assert event["error"].endswith(dc.orphaned_line(["sa-0-rep"]))
        # The line names the id its own instruction takes: nothing else in a recovered notice lists a kept child.
        assert "subagent_id='sa-0-rep'" in event["error"] and "subagent_id=..." not in event["error"]
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
    # r4/D-003: the delegating parent is on the child's ownership chain, so the dry run spares it too.
    assert {"kid-1", "kid-2", "parent"}.isdisjoint(candidates) and "loose" in candidates
    _prune_all(db)
    assert db.get_session("kid-1") is not None and db.get_session("kid-2") is not None
    assert db.get_session("loose") is None and db.get_session("parent") is not None


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


# finding 10 (INFO, r3): update_brief is guarded by the run token

def test_record_update_brief_refuses_a_stale_run_token(db):
    store, token = _running_row(db)
    store.finish_run("kid-o", token, "completed")
    new = store.take_fence("kid-o").token
    assert new and new != token
    store.update_brief("kid-o", token, workdir="/stale")  # an older run must not fill the new run's row
    assert _rows(db.db_path)[0]["workdir"] is None
    store.update_brief("kid-o", new, workdir="/fresh")
    assert _rows(db.db_path)[0]["workdir"] == "/fresh"


# finding 5: reads take no write lock and run no DDL

def test_record_read_without_the_table_returns_empty_and_creates_nothing(db, tmp_path):
    store = dc.ChildStore(db.db_path)
    assert store.get("x") is None and store.held_by("p") == [] and store.resolve("x", ["p"]) is None
    assert store.show("x", ["p"]) is None
    assert not _has_table(db.db_path)
    missing = tmp_path / "nowhere" / "state.db"
    assert dc.ChildStore(missing).get("x") is None and not missing.exists()


def test_record_read_on_a_read_only_db(db):
    store, _ = _idle_kid(db)
    db.close()  # the open handle would keep -wal/-shm alive and make the file writable anyway
    os.chmod(db.db_path, 0o444)
    try:
        assert store.get("kid-1")["status"] == "idle"
        assert [r["child_id"] for r in store.held_by("parent")] == ["kid-1"]
    finally:
        os.chmod(db.db_path, 0o644)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_record_read_derives_a_due_rule_on_a_read_only_db(tmp_path):
    """e6's roster reads every turn: a due orphan on an unwritable store shows the derived state, never raises."""
    from hermes_state import SessionDB
    handle = SessionDB(db_path=tmp_path / "state.db")
    store = dc.ChildStore(handle.db_path)
    token = store.record_run_start(child_id="kid-ro", subagent_id="sa-0-ro", holder_session_id="parent",
                                   conversation_id="parent", brief={"goal": "g"})
    assert token
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    runner = _process_runner(proc)
    proc.kill()
    proc.wait()
    _set(handle.db_path, "kid-ro", runner=json.dumps(runner), heartbeat_at=time.time() - 60)
    handle.close()
    os.chmod(handle.db_path, 0o444)
    try:
        row = store.get("kid-ro")
        assert (row["status"], row["idle_reason"]) == ("idle", "orphaned")
        assert [r["child_id"] for r in store.held_by("parent")] == ["kid-ro"]
        assert _rows(handle.db_path)[0]["status"] == "running"  # derived for display, nothing written
    finally:
        os.chmod(handle.db_path, 0o644)
    assert store.get("kid-ro")["status"] == "idle"  # the next writable read persists it


def test_record_show_does_not_stall_on_a_writer_lock_when_the_tip_moved(db):
    store, _ = _idle_kid(db)
    db.end_session("kid-1", "compression")
    db.create_session("kid-2", source="subagent", parent_session_id="kid-1",
                      model_config={"_delegate_from": "parent", "_subagent_id": "sa-0-aaaa"})  # the tip moved
    blocker = sqlite3.connect(db.db_path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        row = store.show("kid-1", ["parent"])
        assert time.monotonic() - started < 3.0
        assert (row["child_id"], row["latest_session_id"]) == ("kid-1", "kid-1")  # refresh skipped, shown stale
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert store.refresh_latest("kid-1") == "kid-2"  # the next read lands it
    assert _rows(db.db_path)[0]["latest_session_id"] == "kid-2"


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

def test_fence_done_callback_makes_one_attempt_and_does_not_wait(db, monkeypatch, caplog):
    """A write-locked store must not delay the parent's result: one attempt, no retry loop, no sleep."""
    monkeypatch.setattr(dc, "HEARTBEAT_INTERVAL_S", 60.0)
    calls = []

    def locked(self, *a, **k):
        calls.append(a)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(dc.ChildStore, "finish_run", locked)
    child, future, _, _ = _spawn(db, _parent(db))
    with caplog.at_level("WARNING", logger=dc.__name__):
        started = time.monotonic()
        _finish(future)
    assert time.monotonic() - started < 1.0
    assert len(calls) == 1  # one attempt; the tick retries, not the callback
    assert child._delegate_children_tick.thread.is_alive()
    assert _rows(db.db_path)[0]["status"] == "running"
    assert any("could not mark child kid-1 idle" in r.getMessage() for r in caplog.records)
    child._delegate_children_tick.stop()


def test_fence_tick_lands_the_final_write_after_the_callbacks_attempt_fails(db, monkeypatch):
    monkeypatch.setattr(dc, "HEARTBEAT_INTERVAL_S", 0.02)
    real, fails = dc.ChildStore.finish_run, {"n": 2}  # literal: only a retrying writer gets past two failures

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


def test_fence_tick_survives_an_unexpected_error(db, monkeypatch):
    """The tick is the only liveness signal: an exception that is not a sqlite3 error must not end it."""
    monkeypatch.setattr(dc, "HEARTBEAT_INTERVAL_S", 0.02)
    real, calls = dc.ChildStore.beat, {"n": 0}

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return real(self, *a, **k)

    monkeypatch.setattr(dc.ChildStore, "beat", flaky)
    child, future, _, _ = _spawn(db, _parent(db))
    tick = child._delegate_children_tick
    assert _wait(lambda: calls["n"] >= 3)  # it kept beating after the ValueError
    assert tick.thread.is_alive()
    _finish(future)
    assert _wait(lambda: not tick.thread.is_alive())
    assert dc.ChildStore(db.db_path).get("kid-1")["status"] == "idle"


def test_fence_tick_start_failure_still_finishes_the_run(db, monkeypatch):
    def no_thread(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(dc._Tick, "start", no_thread)
    _, future, token, kept = _spawn(db, _parent(db))
    assert kept and token
    _finish(future)
    assert dc.ChildStore(db.db_path).get("kid-1")["status"] == "idle"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_orphan_alive_runner_whose_tick_died_stays_running_until_its_process_is_gone(db):
    """D-005 option 2: a provably alive runner is never orphaned on age, however stale its heartbeat; once its
    process is gone the same row is idle-orphaned at its last heartbeat."""
    store, _ = _running_row(db)  # runner = this process, provably alive
    t0 = time.time() - DAY
    _set(db.db_path, "kid-o", heartbeat_at=t0)
    assert store.get("kid-o", now=t0 + 30 * DAY)["status"] == "running"
    fence = store.take_fence("kid-o", now=t0 + 30 * DAY)  # no second run behind a running child
    assert (fence.ok, fence.reason, fence.steer_receipt) == (False, "running", True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    runner = _process_runner(proc)
    proc.kill()
    proc.wait()
    _set(db.db_path, "kid-o", runner=json.dumps(runner))  # the owning process is gone (a restart)
    row = store.get("kid-o", now=t0 + 60)
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
        assert event["error"].endswith(dc.orphaned_line(["sa-0-unit"]))
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


# ── review round 4 (plan delegate-resume, e1 r4 / D-003 option 1) ────────────

def _ended(db, sid, *, parent=None, end="user_exit", source="cli", model_config=None):
    """A stored session row that ended (prune only deletes ended rows)."""
    db.create_session(sid, source=source, parent_session_id=parent, model_config=model_config)
    if end:
        db.end_session(sid, end)
    return sid


def _kept(db, child_sid, subagent_id, holder, conversation):
    """The ``delegate_children`` row of an idle (retained) child, as a spawn + worker start leave it."""
    store = dc.ChildStore(db.db_path)
    token = store.record_run_start(child_id=child_sid, subagent_id=subagent_id, holder_session_id=holder,
                                   conversation_id=conversation, brief={"goal": "g"})
    assert token
    assert store.finish_run(child_sid, token, "completed")
    return store


def _age(db, days=400):
    """Push every session row well past a 90-day retention window (the child's own row stays fresh)."""
    conn = sqlite3.connect(db.db_path)
    conn.execute("UPDATE sessions SET started_at = started_at - ? * 86400,"
                 " ended_at = COALESCE(ended_at, started_at) - ? * 86400", (days, days))
    conn.commit()
    conn.close()


def test_prune_chain_keeps_the_parent_and_the_alias_still_resolves(db, monkeypatch):
    """r1 finding 4 / D-003: pruning the delegating parent must not cut the child off from the conversation.

    A prune whose age takes the parent spares the whole chain, so the ``subagent_id`` alias still resolves
    through ``stored_child_session``, ``latest_session_id`` is right and the resume precondition holds."""
    from tools.delegate_tool_sessions import stored_child_session
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "parent")                                    # the holder's conversation, aged past the window
    _child_session(db, "kid-1", "parent", "sa-0-aaaa")       # the kept child's first segment
    db.end_session("kid-1", "compression")
    db.create_session("kid-2", source="subagent", parent_session_id="kid-1",
                      model_config={"_delegate_from": "parent", "_subagent_id": "sa-0-aaaa"})
    db.end_session("kid-2", "user_exit")
    _ended(db, "unrelated")                                 # expired, unprotected: must still go
    store = _kept(db, "kid-1", "sa-0-aaaa", "parent", "parent")
    _age(db)

    assert db.prune_sessions(older_than_days=90) == 1        # the unrelated session only
    assert db.get_session("unrelated") is None
    assert db.get_session("parent") is not None              # spared with the chain it anchors
    assert db.get_session("kid-1")["parent_session_id"] == "parent"  # not orphaned by a partial prune
    assert db.get_session("kid-2")["parent_session_id"] == "kid-1"

    assert stored_child_session(db, "sa-0-aaaa", ["parent"]) == "kid-2"
    assert store.refresh_latest("kid-1") == "kid-2"
    assert store.resolve("sa-0-aaaa", ["parent"]) == "kid-1"
    assert store.show("kid-1", ["parent"])["latest_session_id"] == "kid-2"
    assert store.take_fence("kid-1").ok is True              # resume precondition (e2) holds after the prune


def test_prune_chain_spares_a_compression_ancestor_above_the_delegate_hop(db, monkeypatch):
    """The parent rotated before it delegated: its own compression ancestors are on the child's chain."""
    from tools.delegate_tool_sessions import stored_child_session
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "conv-0", end="compression")                 # the conversation root (holder's lineage)
    _ended(db, "conv-1", parent="conv-0")                   # its rotation: where the child delegated from
    _child_session(db, "kid-1", "conv-1", "sa-0-bbbb")
    db.end_session("kid-1", "user_exit")
    store = _kept(db, "kid-1", "sa-0-bbbb", "conv-1", "conv-0")
    _age(db)

    db.prune_sessions(older_than_days=90)
    assert db.get_session("conv-1") is not None and db.get_session("conv-0") is not None
    assert db.get_session("kid-1")["parent_session_id"] == "conv-1"
    assert stored_child_session(db, "sa-0-bbbb", ["conv-0"]) == "kid-1"
    assert store.refresh_latest("kid-1") == "kid-1"


def test_prune_chain_spares_two_compression_hops_above_the_delegate_hop(db, monkeypatch):
    """Two compression hops above the delegate hop are spared, and the row above the conversation root —
    the pre-/reset session, which the walk stops below — is NOT (r3 finding 1: the parent-sparing step must
    not reach past the root, so the prune actually deletes it)."""
    from tools.delegate_tool_sessions import stored_child_session
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "conv-prev", end="reset")          # above the conversation root: the walk stops below it
    _ended(db, "conv-0", parent="conv-prev", end="compression", model_config={"_reset_from": "conv-prev"})
    _ended(db, "conv-1", parent="conv-0", end="compression")
    _ended(db, "conv-2", parent="conv-1")
    _child_session(db, "kid-1", "conv-2", "sa-0-cccc")
    db.end_session("kid-1", "user_exit")
    store = _kept(db, "kid-1", "sa-0-cccc", "conv-2", "conv-0")
    _age(db)

    guarded = {row["id"] for row in db._read_all(f"SELECT id FROM ({db._kept_children_sql()})", ())}
    assert {"conv-0", "conv-1", "conv-2", "kid-1"} <= guarded
    assert "conv-prev" not in guarded        # the root (conversation_id) ends the walk
    assert db.prune_sessions(older_than_days=90) == 1     # what the prune DID, not only the guard set
    assert db.get_session("conv-prev") is None            # above the root: not spared by anything
    for sid in ("conv-0", "conv-1", "conv-2", "kid-1"):
        assert db.get_session(sid) is not None, sid
    assert db.get_session("conv-1")["parent_session_id"] == "conv-0"  # links intact, not NULLed
    assert stored_child_session(db, "sa-0-cccc", ["conv-0"]) == "kid-1"
    assert store.take_fence("kid-1").ok is True


def test_prune_chain_shared_parent_spared_while_either_child_is_retained(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "parent")
    _child_session(db, "kid-a", "parent", "sa-0-a")
    _child_session(db, "kid-b", "parent", "sa-0-b")
    for sid in ("kid-a", "kid-b"):
        db.end_session(sid, "user_exit")
    _kept(db, "kid-a", "sa-0-a", "parent", "parent")
    _kept(db, "kid-b", "sa-0-b", "parent", "parent")
    _age(db)
    _set(db.db_path, "kid-b", status="expired")   # b's protection is over, a still holds the parent
    _pruned = db.prune_sessions(older_than_days=90)

    assert db.get_session("parent") is not None and db.get_session("kid-a") is not None
    assert db.get_session("kid-b") is None and _pruned == 1

    _set(db.db_path, "kid-a", status="expired")   # nobody is retained any more: an ordinary prune applies
    assert db.prune_sessions(older_than_days=90) == 2
    assert db.get_session("parent") is None and db.get_session("kid-a") is None


def test_prune_chain_releases_the_parent_and_its_ancestors_after_the_last_child_expires(db, monkeypatch):
    """Requirement 3: protection ends with the child; the chain is not kept alive and keep_idle_days is not
    stretched — the child's own row expires on its usual clock and the next prune takes parent and ancestors."""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "conv-0", end="compression")
    _ended(db, "conv-1", parent="conv-0")
    _child_session(db, "kid-1", "conv-1", "sa-0-dd")
    db.end_session("kid-1", "user_exit")
    store = _kept(db, "kid-1", "sa-0-dd", "conv-1", "conv-0")
    _age(db)

    db.prune_sessions(older_than_days=90)
    for sid in ("conv-0", "conv-1", "kid-1"):
        assert db.get_session(sid) is not None, sid
    # Expiry, not the prune, ends protection: 7 days + 1 s after its last run the child is gone.
    assert store.get("kid-1", now=time.time() + 8 * DAY)["status"] == "expired"
    assert db.prune_sessions(older_than_days=90) == 3
    assert [sid for sid in ("conv-0", "conv-1", "kid-1") if db.get_session(sid) is not None] == []


def test_prune_parent_null_step_still_orphans_an_unprotected_child(db, monkeypatch):
    """The parent link survives only where a retained chain needs it: an ordinary out-of-window child of a
    pruned session is still NULLed, not cascade-deleted."""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "old-parent")
    _child_session(db, "kid-1", "old-parent", "sa-0-e")  # the retained child: keeps old-parent through the chain
    db.end_session("kid-1", "user_exit")
    _ended(db, "young-child", parent="old-parent")   # nobody's chain: younger than the window, so spared
    _kept(db, "kid-1", "sa-0-e", "old-parent", "old-parent")
    _age(db)
    conn = sqlite3.connect(db.db_path)
    conn.execute("UPDATE sessions SET started_at = started_at + 399 * 86400,"
                 " ended_at = ended_at + 399 * 86400 WHERE id = 'young-child'")
    conn.commit()
    conn.close()

    assert db.prune_sessions(older_than_days=90) == 0
    assert db.get_session("old-parent") is not None          # spared as the retained child's conversation
    assert db.get_session("young-child")["parent_session_id"] == "old-parent"

    _set(db.db_path, "kid-1", status="expired")              # the chain is released with the last child
    assert db.prune_sessions(older_than_days=90) == 2
    assert db.get_session("old-parent") is None and db.get_session("kid-1") is None
    assert db.get_session("young-child")["parent_session_id"] is None


def test_prune_chain_of_a_grandchild_includes_its_parent_childs_rows(db, monkeypatch):
    """A nested child's walk crosses a delegate hop, the parent child's compression rotation, then another
    delegate hop: only the grandchild is retained, yet the child's own rows are on its chain."""
    from tools.delegate_tool_sessions import stored_child_session
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "parent")
    _child_session(db, "kid-1", "parent", "sa-0-kid")   # the child itself is not kept
    db.end_session("kid-1", "compression")
    db.create_session("kid-2", source="subagent", parent_session_id="kid-1",
                      model_config={"_delegate_from": "parent", "_subagent_id": "sa-0-kid"})
    db.end_session("kid-2", "compression")
    _child_session(db, "grand-1", "kid-2", "sa-0-grand")
    db.end_session("grand-1", "user_exit")
    store = _kept(db, "grand-1", "sa-0-grand", "kid-2", "parent")  # the child itself is not kept
    _age(db)

    db.prune_sessions(older_than_days=90)
    for sid in ("parent", "kid-1", "kid-2", "grand-1"):
        assert db.get_session(sid) is not None, sid
    assert stored_child_session(db, "sa-0-grand", ["parent"]) == "grand-1"
    assert store.take_fence("grand-1").ok is True

    _set(db.db_path, "grand-1", status="expired")   # release: every row on the chain becomes prunable
    assert db.prune_sessions(older_than_days=90) == 4


@pytest.mark.parametrize("door", ["manual", "auto"])
def test_prune_chain_is_honoured_by_the_manual_and_the_auto_maintenance_paths(tmp_path, monkeypatch, capsys, door):
    """``hermes sessions prune`` (the live profiles' main path) and the startup auto-maintenance both spare
    the retained child's chain, not only its own rows."""
    import argparse
    from hermes_state import SessionDB
    from hermes_cli import sessions_cmd
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        _ended(handle, "conv-0", end="compression")
        _ended(handle, "conv-1", parent="conv-0")
        _child_session(handle, "kid-1", "conv-1", "sa-0-ff")
        handle.end_session("kid-1", "user_exit")
        _ended(handle, "gone")  # unrelated and expired: still pruned by both doors
        _kept(handle, "kid-1", "sa-0-ff", "conv-1", "conv-0")
        _age(handle)
        if door == "auto":
            handle.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)
        else:
            monkeypatch.setattr(sessions_cmd, "_sessions_dir", lambda: tmp_path / "sessions")
            assert sessions_cmd._cmd_prune_or_archive(
                handle, argparse.Namespace(yes=True, dry_run=False), "prune") is None
            assert "Pruned" in capsys.readouterr().out
        for sid in ("conv-0", "conv-1", "kid-1"):
            assert handle.get_session(sid) is not None, sid
        assert handle.get_session("gone") is None
    finally:
        handle.close()


def test_prune_chain_survives_a_corrupt_prior_conversations_value(db, monkeypatch):
    """A malformed ``prior_conversations`` must neither break the prune statement nor widen what it spares:
    it is read as "no prior conversation", exactly as the resolver's own decode does."""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "parent")
    _child_session(db, "kid-1", "parent", "sa-0-gg")
    db.end_session("kid-1", "user_exit")
    _kept(db, "kid-1", "sa-0-gg", "parent", "parent")
    _set(db.db_path, "kid-1", prior_conversations="{not json")
    _age(db)
    assert db.prune_sessions(older_than_days=90) == 0
    assert db.get_session("parent") is not None and db.get_session("kid-1") is not None


def test_prune_never_orphans_a_retained_row_whose_stored_parent_is_not_a_valid_hop(db, monkeypatch):
    """Requirement 2, residual case: a kept row whose ``_delegate_from`` names some other session is not on a
    walkable chain, so its parent is not spared by the chain guard — the prune still must not delete it and
    NULL the link the ownership proof needs."""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "stored-parent")
    db.create_session("kid-x", source="subagent", parent_session_id="stored-parent",
                      model_config={"_delegate_from": "elsewhere", "_subagent_id": "sa-0-hh"})
    db.end_session("kid-x", "user_exit")
    _kept(db, "kid-x", "sa-0-hh", "stored-parent", "stored-parent")
    _age(db)

    assert db.prune_sessions(older_than_days=90) == 0
    assert db.get_session("stored-parent") is not None
    assert db.get_session("kid-x")["parent_session_id"] == "stored-parent"


# ── review round 5 (plan delegate-resume, e1 r5: r3 review findings 1 and 2) ───────────────────

def _reset_lineage(db, subagent_id="sa-0-r5"):
    """A /reset conversation root, with a retained child delegated from it:

    ``pre-reset`` --(_reset_from)--> ``reset-root`` (the holder's conversation root, ``conversation_id``)
    --(_delegate_from)--> ``kid``. The ownership walk stops at ``reset-root``, so ``pre-reset`` — an ordinary
    row above the root, the user's previous conversation — is not on the chain.
    """
    _ended(db, "pre-reset", end="reset")
    _ended(db, "reset-root", parent="pre-reset", model_config={"_reset_from": "pre-reset"})
    _child_session(db, "kid", "reset-root", subagent_id)
    db.end_session("kid", "user_exit")
    store = _kept(db, "kid", subagent_id, "reset-root", "reset-root")
    _age(db)
    return store


def _route(db, sid, key="agent:main:ui"):
    """A ``gateway_routing`` row naming *sid*, as the gateway keeps for a live conversation."""
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute("INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at)"
                     " VALUES('main', ?, ?, ?)", (key, json.dumps({"session_id": sid, "source": "cli"}), time.time()))
        conn.commit()
    finally:
        conn.close()


def _routed(db):
    """The session ids the store's ``gateway_routing`` rows still name."""
    conn = sqlite3.connect(db.db_path)
    try:
        return {json.loads(row[0]).get("session_id") for row in conn.execute("SELECT entry_json FROM gateway_routing")}
    finally:
        conn.close()


def _retire_fences(db):
    """The ``gateway.retired_session.v1.*`` markers the store holds: written for ids being deleted."""
    conn = sqlite3.connect(db.db_path)
    try:
        return {row[0].split("gateway.retired_session.v1.", 1)[1] for row in
                conn.execute("SELECT key FROM state_meta WHERE key LIKE 'gateway.retired_session.v1.%'")}
    finally:
        conn.close()


def test_prune_above_a_reset_root_does_not_spare_the_pre_reset_session(db, monkeypatch):
    """r3 finding 1: a /reset root's ``parent_session_id`` names the previous conversation, which the walk
    deliberately stops below. The parent-sparing step must not be seeded with a conversation root, or it
    spares that row too and the prune skips a deletion it should make."""
    from tools.delegate_tool_sessions import stored_child_session
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    store = _reset_lineage(db)

    assert {row["id"] for row in db._read_all(f"SELECT id FROM ({db._kept_children_sql()})", ())} == {"kid", "reset-root"}

    assert db.prune_sessions(older_than_days=90) == 1        # the pre-reset session, not 0
    assert db.get_session("pre-reset") is None               # above the root: not spared
    assert db.get_session("reset-root") is not None          # the root anchors the alias
    assert db.get_session("kid") is not None
    assert db.get_session("kid")["parent_session_id"] == "reset-root"   # the link the proof walks survives

    assert stored_child_session(db, "sa-0-r5", ["reset-root"]) == "kid"
    assert store.take_fence("kid").ok is True


def test_prune_never_retires_a_row_it_spares(db, monkeypatch):
    """r3 finding 2: a row the prune SPARES is never left half-deleted — no ``gateway.retired_session.v1``
    fence for it, its ``gateway.api.settings`` row and its ``gateway_routing`` entry intact, while the row the
    same prune DOES delete carries its fence. (The ordering that a spared row is never fenced because the
    parent-sparing step runs before ``retire_prunable`` is pinned by
    ``test_prune_never_retires_the_stored_parent_it_still_needs``; here the spared root is already outside the
    selection by the chain guard.)"""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _reset_lineage(db)
    _route(db, "reset-root", "agent:main:ui")
    _route(db, "pre-reset", "agent:main:cli")
    db.set_meta("gateway.api.settings.v1.reset-root", '{"model": "m"}')
    db.set_meta("gateway.api.settings.v1.pre-reset", '{"model": "m"}')

    assert db.prune_sessions(older_than_days=90) == 1

    surviving = {"reset-root", "kid"}
    assert {sid for sid in surviving if db.get_session(sid) is None} == set()
    assert not (surviving & _retire_fences(db))              # no surviving row carries a retire fence
    assert _retire_fences(db) == {"pre-reset"}               # the deleted row's fence exists
    assert _routed(db) == {"reset-root"}                     # the spared root keeps its route, pre-reset's is gone
    assert db.get_meta("gateway.api.settings.v1.reset-root") is not None
    assert db.get_meta("gateway.api.settings.v1.pre-reset") is None


def test_prune_never_retires_the_stored_parent_it_still_needs(db, monkeypatch):
    """The step's load-bearing case (r1 finding 4 residual): a kept row whose stored parent is not a valid
    hop keeps that parent — spared, and (r3 finding 2) spared before retirement, so no fence is written for
    it and its routing entry survives."""
    monkeypatch.setattr(dc, "_delegation_cfg", dict)
    _ended(db, "stored-parent")
    db.create_session("kid-x", source="subagent", parent_session_id="stored-parent",
                      model_config={"_delegate_from": "elsewhere", "_subagent_id": "sa-0-hh"})
    db.end_session("kid-x", "user_exit")
    _kept(db, "kid-x", "sa-0-hh", "stored-parent", "stored-parent")
    _route(db, "stored-parent")
    _age(db)

    assert db.prune_sessions(older_than_days=90) == 0
    assert db.get_session("stored-parent") is not None
    assert db.get_session("kid-x")["parent_session_id"] == "stored-parent"   # not NULLed
    assert _retire_fences(db) == set()                       # no fence for a row that survived
    assert _routed(db) == {"stored-parent"}


# ── e2: ids on creation (dispatch handle, completion results, control by child_id) ───────────────


def _dispatch_batch(parent, children):
    """A background ``_Batch`` over *children* (``(task_index, task, child)``), as ``delegate_task`` builds it."""
    from tools import delegate_tool_dispatch as dd
    tasks = [{"goal": f"goal-{i}"} for i, _, _ in children]
    return dd, dd._Batch(
        tasks, list(children), parent, {"model": "m"}, None, "leaf", len(children), None,
        [None] * len(children), [], "wake", "ui", None, None, False, time.time(),
    )


def _stub_dispatch(monkeypatch, dd):
    """Keep the payload path in-process: admission accepts every unit, no worker runs."""
    from tools import delegate_tool_config
    monkeypatch.setattr(dd, "_dispatch_unit",
                        lambda unit, unit_id, _slot, _routing: {"status": "dispatched", "delegation_id": unit_id})
    monkeypatch.setattr(dd, "_resolve_async_wake_sid", lambda *a, **k: "wake")
    monkeypatch.setattr(dd, "_resolve_async_session_key", lambda *a, **k: ("key", "ui"))
    monkeypatch.setattr(delegate_tool_config, "_get_independent_completions", lambda: False)


def test_ids_dispatch_handle_carries_child_id_and_subagent_id_per_kept_child(db, monkeypatch):
    """e2 red test: a kept background child's dispatch handle names it by ``child_id`` next to ``subagent_id``;
    a child that is not kept carries no ``child_id`` key at all (absent, never a fake id)."""
    parent = _parent(db)
    kept, _, _, was_kept = _spawn(db, parent, start=False)  # built, not started: the handle's state
    assert was_kept and kept._delegate_child_id == "kid-1"
    loose = SimpleNamespace(session_id="kid-2", _subagent_id="sa-0-bbbb")
    dd, batch = _dispatch_batch(parent, [(0, {"goal": "a"}, kept), (1, {"goal": "b"}, loose)])
    _stub_dispatch(monkeypatch, dd)

    payload = json.loads(dd._dispatch_background(batch))

    assert payload["children"] == [
        {"task_index": 0, "subagent_id": "sa-0-aaaa", "child_id": "kid-1"},
        {"task_index": 1, "subagent_id": "sa-0-bbbb"},
    ]
    assert payload["subagent_ids"] == ["sa-0-aaaa", "sa-0-bbbb"]
    assert "child_id" not in payload["children"][1]
    assert "child_id" in payload["control_hint"]

    # With no kept child: no ``children`` key, and the ``control_hint`` sentence is the pre-e2 one, byte for byte
    # (it must not advertise a ``child_id`` the call does not have).
    dd, loose_only = _dispatch_batch(parent, [(0, {"goal": "a"}, loose)])
    loose_payload = json.loads(dd._dispatch_background(loose_only))
    assert "children" not in loose_payload
    assert loose_payload["control_hint"] == dd._BACKGROUND_NOTES["control_hint"]
    assert "child_id" not in loose_payload["control_hint"]
    assert "action='steer' with subagent_id + message" in loose_payload["control_hint"]


def test_ids_completion_result_carries_child_id_and_the_resume_hint(db):
    """Every completion result of a kept child carries ``child_id`` and ``subagent_id`` and a one-line
    ``resume_hint``; the child's answer itself is left clean (the hint is parent-facing prose)."""
    from tools.delegate_tool_results import _finalize_child_results
    parent = _parent(db)
    kept, _, _, was_kept = _spawn(db, parent, start=False)
    assert was_kept
    task = {"goal": "g"}
    entries = [{"task_index": 0, "status": "completed", "summary": "the answer"}]
    _finalize_child_results(entries, [task], [(0, task, kept)], parent)

    entry = entries[0]
    assert (entry["child_id"], entry["subagent_id"]) == ("kid-1", "sa-0-aaaa")
    assert entry["resume_hint"] == dc.resume_hint("sa-0-aaaa")
    # The action that exists, with the id it takes (``continue`` is addressed by subagent_id, not child_id).
    assert "action='continue', subagent_id='sa-0-aaaa'" in entry["resume_hint"]
    assert "'resume'" not in entry["resume_hint"] and "kid-1" not in entry["resume_hint"]
    assert entry["summary"] == "the answer"

    loose = SimpleNamespace(session_id="kid-2", _subagent_id="sa-0-bbbb")
    other = [{"task_index": 0, "status": "completed", "summary": "x"}]
    _finalize_child_results(other, [task], [(0, task, loose)], parent)
    assert set(other[0]).isdisjoint({"child_id", "subagent_id", "resume_hint"})


def test_ids_completion_text_ends_with_the_resume_hint(db):
    """The completion message the parent reads ends with the kept child's resume hint."""
    from tools.delegate_tool_results import _finalize_child_results
    from tools.process_registry_notifications import _format_batch_delegation
    parent = _parent(db)
    kept, _, _, _ = _spawn(db, parent, start=False)
    task = {"goal": "g"}
    entries = [{"task_index": 0, "status": "completed", "summary": "the answer"}]
    _finalize_child_results(entries, [task], [(0, task, kept)], parent)
    completed_at = time.time()
    evt = {"delegation_id": "deleg-ids", "goals": ["g"], "results": entries,
           "dispatched_at": completed_at - 1, "completed_at": completed_at}

    text = _format_batch_delegation(evt, "deleg-ids", completed_at)

    assert text.endswith(entries[0]["resume_hint"])
    assert text.count(entries[0]["resume_hint"]) == 1


def _live_child_record(subagent_id, parent, **extra):
    """Register a live child agent under *subagent_id*, owned by *parent*'s durable session."""
    import tools.delegate_tool_registry as dtr

    class _StubChild:
        def __init__(self):
            self.steered = []

        def steer(self, text):
            self.steered.append(text)
            return True

    stub = _StubChild()
    dtr._register_subagent({
        "subagent_id": subagent_id, "agent": stub, "owner_agent_session_id": parent.session_id,
        "started_at": time.time(), "status": "running", "accepting_steer": True, "goal": "g", **extra,
    })
    return dtr, stub


def test_ids_steer_and_stop_accept_child_id_resolved_to_the_live_subagent_id(db, monkeypatch):
    parent = _parent(db)
    child, future, _, _ = _spawn(db, parent)  # the row exists and the run is open
    dtr, stub = _live_child_record("sa-0-aaaa", parent)
    try:
        out = json.loads(dtr._handle_control_action("steer", None, "focus on X", parent, child_id="kid-1"))
        assert (out["status"], out["subagent_id"], out["child_id"]) == ("queued", "sa-0-aaaa", "kid-1")
        assert stub.steered == ["focus on X"]
        interrupted = []
        monkeypatch.setattr(dtr, "request_hard_interrupt",
                            lambda agent, reason: interrupted.append(agent) or True)
        out = json.loads(dtr._handle_control_action("stop", None, None, parent, child_id="kid-1"))
        assert out["status"] == "interrupt_requested" and interrupted == [stub]
    finally:
        dtr._unregister_subagent("sa-0-aaaa")
        child._delegate_children_tick.stop()
        _finish(future)


def test_ids_delegate_task_routes_child_id_to_the_control_plane(db, monkeypatch):
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task
    assert "child_id" in DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    parent = _parent(db)
    child, future, _, _ = _spawn(db, parent)
    dtr, stub = _live_child_record("sa-0-aaaa", parent)
    try:
        out = json.loads(delegate_task(action="steer", child_id="kid-1", message="keep going", parent_agent=parent))
        assert (out["status"], out["subagent_id"]) == ("queued", "sa-0-aaaa")
        assert stub.steered == ["keep going"]
    finally:
        dtr._unregister_subagent("sa-0-aaaa")
        child._delegate_children_tick.stop()
        _finish(future)


def test_ids_unknown_child_id_is_a_clear_error(db):
    import tools.delegate_tool_registry as dtr
    out = dtr._handle_control_action("steer", None, "x", _parent(db), child_id="no-such-child")
    assert "No child with child_id 'no-such-child'" in out


def test_ids_finished_child_id_is_refused_not_a_silent_no_op(db):
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    _, future, _, _ = _spawn(db, parent)
    _finish(future)
    store = dc.ChildStore(db.db_path)
    assert _wait(lambda: store.get("kid-1")["status"] == "idle")
    out = dtr._handle_control_action("stop", None, None, parent, child_id="kid-1")
    assert "not running (status=idle)" in out and "nothing to stop" in out


def test_ids_child_id_held_by_another_conversation_does_not_resolve(db):
    import tools.delegate_tool_registry as dtr
    other = _parent(db, session_id="other")
    _, future, _, _ = _spawn(db, other)
    try:
        out = dtr._handle_control_action("steer", None, "hijack", _parent(db, session_id="parent"),
                                         child_id="kid-1")
        assert "No child with child_id 'kid-1'" in out
    finally:
        _finish(future)


# ── e1 carry-in: the parent-sparing step cannot be called without its roots ──────────────────────

def test_prune_parent_sparing_requires_the_roots_argument():
    """``roots`` has no default: a caller that omits it must fail loudly instead of silently restoring the
    fail-open the root-seeding rule fixed (a conversation root's own parent would be spared again)."""
    import inspect
    from hermes_state_maintenance import SessionMaintenanceMixin
    parameters = inspect.signature(SessionMaintenanceMixin._parents_still_needed).parameters
    assert parameters["roots"].default is inspect.Parameter.empty


# ── e3: list — kept children in the default scope, scope='profile', query, restart ───────────────

def _kept_idle(db, parent, child_sid="kid-1", subagent_id="sa-0-aaaa", goal="find the token"):
    """Spawn a kept child, let its worker finish, and return once its row is idle (kept, not live)."""
    _child_session(db, child_sid, parent.session_id, subagent_id)
    child = _child(child_sid, subagent_id)
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal=goal, context="ctx", toolsets=["file"], role="leaf", model="m",
                             route={"provider": "p", "model": "m"})
    future: Future = Future()
    assert dc.worker_started(child, future, task_index=0)
    _finish(future)
    assert _wait(lambda: (dc.ChildStore(db.db_path).get(child_sid) or {}).get("status") == "idle")
    return child


def _title(db, session_id, title):
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
        conn.commit()
    finally:
        conn.close()


def test_list_returns_a_kept_child_after_it_finished(db):
    """e3 red test: ``action='list'`` still names a child that finished (on e2's tree the count is 0)."""
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    _kept_idle(db, parent)

    payload = dtr._list_payload(parent)  # no arguments: the default shape and scope

    assert payload["count"] == 1
    assert payload["subagents"] == []
    assert payload["scope"] == "session"
    entry = payload["children"][0]
    assert (entry["child_id"], entry["subagent_id"]) == ("kid-1", "sa-0-aaaa")
    assert entry["goal"] == "find the token"
    assert (entry["status"], entry["idle_reason"], entry["resumable"]) == ("idle", "completed", True)
    assert entry["holder_session_id"] == "parent" and entry["read_only"] is False
    assert "context" not in entry and "route" not in entry  # no brief, no credentials in the listing


def test_list_counts_a_running_kept_child_once(db):
    """A kept child running in THIS process is listed as live, not a second time as a kept row."""
    parent = _parent(db)
    child, future, _, _ = _spawn(db, parent)
    dtr, _stub = _live_child_record("sa-0-aaaa", parent)
    try:
        payload = json.loads(dtr._handle_control_action("list", None, None, parent))
        assert [e["subagent_id"] for e in payload["subagents"]] == ["sa-0-aaaa"]
        assert "children" not in payload
        assert payload["count"] == 1
    finally:
        dtr._unregister_subagent("sa-0-aaaa")
        child._delegate_children_tick.stop()
        _finish(future)
    assert _wait(lambda: dc.ChildStore(db.db_path).get("kid-1")["status"] == "idle")
    finished = json.loads(dtr._handle_control_action("list", None, None, parent))
    assert finished["subagents"] == []
    assert [c["child_id"] for c in finished["children"]] == ["kid-1"]  # once, from the row, not twice
    assert finished["count"] == 1


def test_list_default_scope_is_this_sessions_children_only(db):
    """Two sessions of one profile: the plain list is the caller's own children."""
    import tools.delegate_tool_registry as dtr
    mine = _parent(db, session_id="parent")
    other = _parent(db, session_id="other")
    _kept_idle(db, mine, child_sid="kid-mine", subagent_id="sa-0-mine")
    _kept_idle(db, other, child_sid="kid-other", subagent_id="sa-0-other")

    payload = json.loads(dtr._handle_control_action("list", None, None, mine))

    assert payload["count"] == 1
    assert [c["child_id"] for c in payload["children"]] == ["kid-mine"]
    assert payload["children"][0]["read_only"] is False
    assert "profile_children" not in payload


def test_list_profile_scope_adds_the_profiles_other_kept_children_with_their_holder(db):
    import tools.delegate_tool_registry as dtr
    mine = _parent(db, session_id="parent")
    other = _parent(db, session_id="other")
    _kept_idle(db, mine, child_sid="kid-mine", subagent_id="sa-0-mine", goal="mine")
    _kept_idle(db, other, child_sid="kid-other", subagent_id="sa-0-other", goal="theirs")
    _title(db, "other", "the other session")

    assert json.loads(dtr._handle_control_action("list", None, None, mine))["count"] == 1
    payload = json.loads(dtr._handle_control_action("list", None, None, mine, scope="profile"))

    assert payload["scope"] == "profile" and payload["count"] == 2
    assert [c["child_id"] for c in payload["children"]] == ["kid-mine"]
    entry = payload["profile_children"][0]
    assert entry["child_id"] == "kid-other" and entry["goal"] == "theirs"
    assert entry["holder_session_id"] == "other" and entry["holder_label"] == "the other session"
    assert entry["read_only"] is True and entry["resumable"] is False


def test_list_query_filters_within_both_scopes(db):
    """``query`` is a case-insensitive substring over goal, child_id, subagent_id and title, in either scope."""
    import tools.delegate_tool_registry as dtr
    mine = _parent(db, session_id="parent")
    other = _parent(db, session_id="other")
    _kept_idle(db, mine, child_sid="kid-1", subagent_id="sa-0-aaaa", goal="Find the TOKEN")
    _kept_idle(db, mine, child_sid="kid-2", subagent_id="sa-0-bbbb", goal="summarise the report")
    _set(db.db_path, "kid-2", title="Weekly digest")
    _kept_idle(db, other, child_sid="kid-3", subagent_id="sa-0-cccc", goal="token audit")

    def listed(parent, **kwargs):
        return json.loads(dtr._handle_control_action("list", None, None, parent, **kwargs))

    assert listed(mine)["count"] == 2
    assert [c["child_id"] for c in listed(mine, query="token")["children"]] == ["kid-1"]
    assert [c["child_id"] for c in listed(mine, query="TOKEN", scope="profile")["children"]] == ["kid-1"]
    assert [c["child_id"] for c in listed(mine, query="token", scope="profile")["profile_children"]] == ["kid-3"]
    assert listed(mine, query="KID-2")["children"][0]["child_id"] == "kid-2"
    assert listed(mine, query="sa-0-bb")["children"][0]["child_id"] == "kid-2"
    assert listed(mine, query="digest")["children"][0]["child_id"] == "kid-2"
    empty = listed(mine, query="nothing-matches")
    assert empty["count"] == 0 and "children" not in empty
    assert "'nothing-matches'" in empty["note"] and "without a query" in empty["note"]


def test_list_two_profiles_in_one_process_never_see_each_other(tmp_path):
    """Another profile's ``state.db`` is never opened: not in the default scope, not in ``scope='profile'``."""
    from hermes_state import SessionDB
    import tools.delegate_tool_registry as dtr
    alpha = SessionDB(db_path=tmp_path / "profiles" / "alpha" / "state.db")
    beta = SessionDB(db_path=tmp_path / "profiles" / "beta" / "state.db")
    db_a, db_b = alpha._db if hasattr(alpha, "_db") else alpha, beta._db if hasattr(beta, "_db") else beta
    try:
        parent_a = _parent(db_a)
        parent_b = _parent(db_b)
        _kept_idle(db_a, parent_a, child_sid="kid-a", subagent_id="sa-a", goal="alpha work")
        _kept_idle(db_b, parent_b, child_sid="kid-b", subagent_id="sa-b", goal="beta work")

        assert json.loads(dtr._handle_control_action("list", None, None, parent_a))["count"] == 1
        payload = json.loads(dtr._handle_control_action("list", None, None, parent_a, scope="profile"))

        assert payload["count"] == 1
        assert [c["child_id"] for c in payload["children"]] == ["kid-a"]
        assert [c["goal"] for c in payload["children"]] == ["alpha work"]
        assert "profile_children" not in payload
    finally:
        alpha.close()
        beta.close()


def test_list_survives_a_restart(tmp_path):
    """A new process opens the same ``state.db``: the kept child is still listed, with no live registry at all."""
    from hermes_state import SessionDB
    import tools.delegate_tool_registry as dtr
    path = tmp_path / "state.db"
    first = SessionDB(db_path=path)
    try:
        _kept_idle(first, _parent(first))
    finally:
        first.close()  # the owner process ends here
    reopened = SessionDB(db_path=path)
    try:
        assert dtr._active_subagents == {}  # nothing in-process survives; the payload is built from the db
        payload = json.loads(dtr._handle_control_action("list", None, None, _parent(reopened)))
        assert payload["count"] == 1
        entry = payload["children"][0]
        assert (entry["child_id"], entry["status"], entry["resumable"]) == ("kid-1", "idle", True)
    finally:
        reopened.close()


def test_list_profile_routes_child_is_read_only_to_the_session_that_spawned_it(db, tmp_path):
    """A child that ran as another profile (``profile_routes``) is listed, and read-only even for its spawner."""
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    home = tmp_path / "profiles" / "researcher"
    home.mkdir(parents=True)
    _child_session(db, "kid-r", "parent", "sa-0-route")
    child = _child("kid-r", "sa-0-route")
    with dc.spawn_origin(dc.ORIGIN_MODEL):
        assert dc.note_spawn(parent, child, goal="audit the routes", context=None, toolsets=None, role="leaf",
                             model="m", route={"provider": "p", "model": "m", "profile_home": home},
                             profile_home=home)
    future: Future = Future()
    assert dc.worker_started(child, future, task_index=0)
    _finish(future)
    assert _wait(lambda: (dc.ChildStore(db.db_path).get("kid-r") or {}).get("status") == "idle")

    entry = dtr._list_payload(parent)["children"][0]

    assert entry["child_id"] == "kid-r"
    assert entry["profile"] == "researcher"
    assert entry["read_only"] is True and entry["resumable"] is False


def test_list_keeps_the_newest_twenty_kept_children(db):
    """The plan's 20-child cap applies to kept rows; the newest by activity win and the note says so."""
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    for i in range(22):
        _kept_idle(db, parent, child_sid=f"kid-{i:02d}", subagent_id=f"sa-{i:02d}", goal=f"task {i}")

    payload = dtr._list_payload(parent)

    assert len(payload["children"]) == 20 and payload["count"] == 20
    assert payload["children"][0]["child_id"] == "kid-21"
    shown = [c["child_id"] for c in payload["children"]]
    assert "kid-00" not in shown and "kid-01" not in shown
    assert "2 older kept children of this session" in payload["note"]


def test_list_does_not_wait_on_a_writer_lock(db):
    """The roster reads this every turn: a listing takes no write lock and never creates the table."""
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    _kept_idle(db, parent)
    assert dtr._list_payload(parent)["count"] == 1
    writer = sqlite3.connect(db.db_path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        payload = dtr._list_payload(parent, scope="profile")
        assert payload["count"] == 1
        assert time.monotonic() - started < 2.0
    finally:
        writer.rollback()
        writer.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc start ticks")
def test_list_shows_a_dead_runners_child_as_idle_orphaned(db):
    """A row whose runner died (a restart's leftover) is listed by e1's rules: idle-orphaned and resumable."""
    import tools.delegate_tool_registry as dtr
    parent = _parent(db)
    child, _future, _, _ = _spawn(db, parent)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _set(db.db_path, "kid-1", runner=json.dumps({"pid": dead.pid, "pidns": None}))
    try:
        entry = dtr._list_payload(parent)["children"][0]
        assert (entry["status"], entry["idle_reason"], entry["resumable"]) == ("idle", "orphaned", True)
    finally:
        child._delegate_children_tick.stop()


def test_list_delegate_task_routes_scope_and_query(db):
    from tools.delegate_tool import delegate_task
    parent = _parent(db)
    _kept_idle(db, parent)

    plain = json.loads(delegate_task(action="list", parent_agent=parent))
    assert plain["count"] == 1
    assert plain["scope"] == "session"
    searched = json.loads(delegate_task(action="list", parent_agent=parent, scope="profile", query="TOKEN"))
    assert [c["child_id"] for c in searched["children"]] == ["kid-1"]
    refused = delegate_task(action="list", parent_agent=parent, scope="everything")
    assert "scope='session'" in refused and "'everything'" in refused


# ── e2 carry-ins (review round 1) ────────────────────────────────────────────────────────────────

def test_ids_recovered_completion_of_a_finished_child_carries_its_ids(tmp_path, monkeypatch):
    """Carry-in 3: a crash after one child of a unit finished replays that child's entry with the ``child_id``
    and ``resume_hint`` the live completion would have carried (on e2's tree the durable copy is bare)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from tools import async_delegation as ad
    from tools import delegate_tool_dispatch as dd
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        parent = _parent(handle, source="cli")
        _child_session(handle, "kid-u", "parent", "sa-0-unit")
        child = _child("kid-u", "sa-0-unit")
        child._delegation_unit_id = "deleg_unit"
        with dc.spawn_origin(dc.ORIGIN_MODEL):
            assert dc.note_spawn(parent, child, goal="g", context=None, toolsets=None, role="leaf", model=None)
        future: Future = Future()
        assert dc.worker_started(child, future, task_index=0)
        _finish(future)
        assert _wait(lambda: (dc.ChildStore(handle.db_path).get("kid-u") or {}).get("status") == "idle")
        ad._persist_dispatch({"delegation_id": "deleg_unit", "parent_session_id": "parent",
                              "dispatched_at": time.time(), "goal": "g", "goals": ["g", "h"],
                              "is_batch": True})
        # The moment child 0 finishes, the unit is still running: this is the crash-recovery copy.
        entry = {"task_index": 0, "status": "completed", "summary": "the token is 42"}
        dd._record_finished_child(SimpleNamespace(unit_id="deleg_unit", children=[(0, {"goal": "g"}, child)]),
                                  entry, False)
        assert entry["child_id"] == "kid-u"  # the live entry carries the ids too, exactly as finalize would
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        conn = sqlite3.connect(tmp_path / "state.db")
        conn.execute("UPDATE async_delegations SET owner_pid=?, owner_started_at=NULL", (dead.pid,))
        conn.commit()
        conn.close()
        assert ad.recover_abandoned_delegations() == 1
        conn = sqlite3.connect(tmp_path / "state.db")
        event = json.loads(conn.execute("SELECT event_json FROM async_delegations").fetchone()[0])
        conn.close()
        replayed = event["results"][0]
        assert replayed["status"] == "completed" and replayed["summary"] == "the token is 42"
        assert (replayed["child_id"], replayed["subagent_id"]) == ("kid-u", "sa-0-unit")
        assert replayed["resume_hint"] == dc.resume_hint("sa-0-unit")
        assert event["results"][1]["status"] == "unknown"  # the child that never finished is still unknown
    finally:
        handle.close()


def test_ids_child_of_another_profile_never_resolves(tmp_path):
    """Carry-in 4: two profiles, two ``state.db`` files, one process — parent A can never address profile B's
    child by its ``child_id`` (a different mechanism from the other-conversation case)."""
    from hermes_state import SessionDB
    import tools.delegate_tool_registry as dtr
    alpha = SessionDB(db_path=tmp_path / "profiles" / "alpha" / "state.db")
    beta = SessionDB(db_path=tmp_path / "profiles" / "beta" / "state.db")
    db_a, db_b = alpha._db if hasattr(alpha, "_db") else alpha, beta._db if hasattr(beta, "_db") else beta
    try:
        parent_a = _parent(db_a, session_id="parent-a")
        parent_b = _parent(db_b, session_id="parent-b")
        _kept_idle(db_b, parent_b, child_sid="kid-b", subagent_id="sa-b", goal="beta work")
        out = dtr._handle_control_action("steer", None, "hijack", parent_a, child_id="kid-b")
        assert "No child with child_id 'kid-b'" in out
        stop = dtr._handle_control_action("stop", None, None, parent_a, child_id="kid-b")
        assert "No child with child_id 'kid-b'" in stop
    finally:
        alpha.close()
        beta.close()
