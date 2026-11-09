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


def test_keep_cron_scheduler_marks_its_agent_never_keep():
    import inspect
    from cron import scheduler
    assert "NEVER_KEEP_ATTR" in inspect.getsource(scheduler.run_job)


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


def test_keep_delegate_task_builds_children_under_its_origin(monkeypatch):
    """The model's call builds under ``model``; an internal caller's origin replaces it."""
    from tools import delegate_tool
    seen = []

    def fake_build_children(*_a, **_k):
        seen.append(dc.current_spawn_origin())
        return [], "stop here"

    route = SimpleNamespace(error=None, creds={"model": "m", "provider": None}, refused_entry=lambda i: {})
    monkeypatch.setattr(delegate_tool, "_build_children", fake_build_children)
    monkeypatch.setattr(delegate_tool, "_routes_for", lambda *a, **k: ([route], None))
    monkeypatch.setattr(delegate_tool, "create_live_transcripts", lambda *a, **k: (None, [], []), raising=False)
    import tools.delegation_live_log as live
    monkeypatch.setattr(live, "create_live_transcripts", lambda *a, **k: (None, [], []))
    parent = SimpleNamespace(session_id="p", _delegate_depth=0, _session_db=None)
    delegate_tool.delegate_task(goal="g", parent_agent=parent)
    delegate_tool.delegate_task(goal="g", parent_agent=parent, origin=dc.ORIGIN_REVIEW)
    assert seen == [dc.ORIGIN_MODEL, dc.ORIGIN_REVIEW]


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
    store, token = _running_row(db)
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
        # Even a day-old heartbeat: a provably alive runner is never orphaned on age.
        _set(db.db_path, "kid-o", runner=json.dumps(_process_runner(proc)), heartbeat_at=time.time() - DAY)
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
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {})
    _old_lineage(db)
    candidates = {r["id"] for r in db.list_prune_candidates(older_than_days=None, whole_lineages=True,
                                                            last_active_before=time.time() + 60)}
    assert {"kid-1", "kid-2"}.isdisjoint(candidates) and {"parent", "loose"} <= candidates
    _prune_all(db)
    assert db.get_session("kid-1") is not None and db.get_session("kid-2") is not None
    assert db.get_session("loose") is None and db.get_session("parent") is None


def test_prune_without_protection_deletes_kept_transcripts(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {"keep_protects_transcripts": False})
    _old_lineage(db)
    _prune_all(db)
    assert db.get_session("kid-1") is None and db.get_session("kid-2") is None


def test_prune_expired_child_is_not_protected(db, monkeypatch):
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {})
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
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {})
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
    monkeypatch.setattr(dc, "_delegation_cfg", lambda: {})
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
