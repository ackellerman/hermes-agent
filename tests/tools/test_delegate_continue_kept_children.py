"""Upstream's ``delegate_task(action='continue')`` next to the fork's kept children (``tools.delegate_children``).

A kept child has two records in its parent's ``state.db``: the kept row (``delegate_children``, keyed by
``child_id``) and upstream's durable record (``resumable_subagents``, keyed by ``subagent_id``). These tests pin
how the two meet: what the completion text tells the parent to send, that a continue keeps both records telling
one story, that ``action='list'`` still answers with ``child_id``, and that a child that ran as another profile
(a ``profile_routes`` child) is continued by its own parent inside that profile.

Only the model and the delegation backend are scripted; registry, child build, SessionDB, dispatch, the kept
store and the durable store are the real ones.
"""

import json
import os
import sqlite3
import subprocess
import threading
import time
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from hermes_state import SessionDB
from tests.tools.test_delegate_continue import _completion, _parent, _ScriptedModel
from tests.waiting import wait_until
from tools import delegate_children as dc


class _KeptModel(_ScriptedModel):
    """``_ScriptedModel`` whose children read as real agents to the kept-children hooks (a MagicMock answers
    every attribute, so the ones those hooks test for absence are set), and whose turns can be held open."""

    def __init__(self):
        super().__init__()
        self.hold = None  # (started, release) events for the NEXT turn
        self.fail_first = False
        self.leave_work = False  # the first turn leaves an uncommitted file in its cwd

    def __call__(self, **kw):
        from tools.terminal_tool import get_session_cwd
        child = super().__call__(**kw)
        child._delegate_run_token = None
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, conversation_history=None, task_id=None, **rest):
            hold, self.hold = self.hold, None
            # The stamp a real child's session row carries (``_build_child_agent``): what resolves a subagent_id.
            kw["session_db"].create_session(
                child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"),
                model_config={"_delegate_from": kw.get("parent_session_id"), "_subagent_id": child._subagent_id})
            if hold:
                hold[0].set()
                assert hold[1].wait(30)
            result = ok(user_message, conversation_history=conversation_history, **rest)
            self.turns[-1]["cwd"] = cwd = get_session_cwd(task_id)
            self.turns[-1]["saw_work"] = bool(cwd) and os.path.exists(os.path.join(cwd, "work.txt"))
            if cwd and self.leave_work and len(self.turns) == 1:
                with open(os.path.join(cwd, "work.txt"), "w", encoding="utf-8") as fh:
                    fh.write("child work")
            if self.fail_first and len(self.turns) == 1:
                result = {**result, "completed": False, "failed": True, "error": "provider down",
                          "failure_reason": "server_error"}
            return result
        child.run_conversation.side_effect = run_conversation
        return child


def _kept_parent(home, session_id="parent-sess"):
    home.mkdir(parents=True, exist_ok=True)
    parent = _parent(session_id, db=SessionDB(db_path=home / "state.db"))
    # What a real interactive parent has; a MagicMock would answer these with a truthy mock.
    parent._delegate_child_id = parent._subagent_id = parent._inherited_conversation_id = None
    parent._conversation_id_memo = None
    parent._delegate_children_never_keep = parent._oneshot_session = False
    return parent


def _spawn_kept(parent, goal, **task):
    """The model's own spawn (``origin='model'``, the only kept origin): ``(handle, completion event)``."""
    from tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(tasks=[{"goal": goal, **task}], background=True, parent_agent=parent,
                                   origin="model"))
    assert out.get("status") == "dispatched", out
    return out, _completion(out["delegation_id"])


def _continue(parent, sid, message):
    from tools.delegate_tool import delegate_task
    return json.loads(delegate_task(action="continue", subagent_id=sid, message=message, background=True,
                                    parent_agent=parent, origin="model"))


def _kept_rows(parent):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {dc.TABLE} ORDER BY created_at")]
    finally:
        conn.close()


def _durable(parent, sid):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    try:
        row = conn.execute("SELECT state, pinned, record_json FROM resumable_subagents WHERE subagent_id=?",
                           (sid,)).fetchone()
    finally:
        conn.close()
    return None if row is None else {"state": row[0], "pinned": row[1], **json.loads(row[2])}


def _idle(parent):
    return wait_until(lambda: [r for r in _kept_rows(parent) if r["status"] == "idle"] or None, what="kept row idle")


def _forget_process_state():
    """What a restart loses: the in-process registry of finished children."""
    from tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _recent_subagents.clear()


# ── (b) the texts name the action that exists and the id it takes ───────────

def test_completion_text_tells_the_parent_to_continue_with_the_subagent_id(tmp_path):
    """The hint a kept child's completion ends with is an instruction the parent can send as written."""
    from tools.process_registry_notifications import format_process_notification

    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, done = _spawn_kept(parent, "write the parser")
        sid = out["subagent_ids"][0]
        entry = done["results"][0]
        assert entry["child_id"] == out["children"][0]["child_id"] and entry["subagent_id"] == sid
        hint = entry["resume_hint"]
        assert f"delegate_task(action='continue', subagent_id='{sid}', message=...)" in hint, hint
        assert "action='resume'" not in hint and "child_id=" not in hint, hint
        text = format_process_notification(done)
        assert text.endswith(hint) and "action='resume'" not in text

        _idle(parent)
        cont = _continue(parent, sid, "now add tests")  # exactly what the hint says
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])


def test_no_model_facing_delegation_text_offers_a_resume_action():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_dynamic_schema_overrides
    from tools.delegate_tool_dispatch import _BACKGROUND_NOTES

    schema = json.dumps({**DELEGATE_TASK_SCHEMA, **_build_dynamic_schema_overrides()})
    for text in (dc.ORPHANED_LINE, dc.RESUME_HINT, schema, *_BACKGROUND_NOTES.values()):
        assert "action='resume'" not in text and "'resume'" not in text, text
    assert "action='continue'" in dc.ORPHANED_LINE and "subagent_id" in dc.ORPHANED_LINE
    # The schema says which id each action takes: continue is by subagent_id, never by child_id.
    props = json.loads(schema)["parameters"]["properties"]
    assert "continue" in props["subagent_id"]["description"]
    assert "'continue' takes subagent_id" in props["child_id"]["description"]


def test_continue_addressed_by_child_id_alone_says_which_id_to_send(tmp_path):
    from tools.delegate_tool import delegate_task

    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "write the parser")
        child_id = out["children"][0]["child_id"]
        _idle(parent)
        refused = json.loads(delegate_task(action="continue", child_id=child_id, message="more",
                                           parent_agent=parent, origin="model"))
    assert "subagent_id" in refused["error"] and len(model.builds) == 1, refused


# ── (c) one child, one story ────────────────────────────────────────────────

def test_a_continue_keeps_the_kept_row_and_the_durable_record_in_step(tmp_path):
    """While a continued kept child runs both records say running; when it ends both say it ended; the kept
    row is the SAME row (no second row for the same child), and the continue's handle and completion carry the
    same ``child_id`` the first run's did."""
    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "write the parser")
        sid, child_id = out["subagent_ids"][0], out["children"][0]["child_id"]
        first = _idle(parent)[0]
        assert (first["child_id"], first["idle_reason"]) == (child_id, "completed")
        assert _durable(parent, sid)["state"] == "completed"

        started, release = threading.Event(), threading.Event()
        model.hold = (started, release)
        cont = _continue(parent, sid, "now add tests")
        assert cont.get("status") == "dispatched", cont
        try:
            assert started.wait(10)
            # The fence is taken right after the worker is submitted, so the row may trail the first turn.
            running = wait_until(lambda: [r for r in _kept_rows(parent) if r["status"] == "running"] or None,
                                 what="kept row running")
            assert [(r["child_id"], r["status"]) for r in _kept_rows(parent)] == [(child_id, "running")]
            assert running[0]["run_token"] != first["run_token"]
            assert _durable(parent, sid)["state"] == "running"
            assert cont["children"] == [{"task_index": 0, "subagent_id": sid, "child_id": child_id}]
        finally:
            release.set()
        done = _completion(cont["delegation_id"])
        entry = done["results"][0]
        assert (entry["child_id"], entry["subagent_id"]) == (child_id, sid)
        rows = _idle(parent)
        assert [(r["child_id"], r["status"], r["idle_reason"]) for r in _kept_rows(parent)] == [
            (child_id, "idle", "completed")]
        assert rows[0]["idle_since"] > first["idle_since"]  # a new run resets the expiry clock
        assert _durable(parent, sid)["state"] == "completed"


def test_a_continue_of_a_child_that_was_never_kept_creates_no_kept_row(tmp_path):
    """Only the spawn decides whether a child is kept: continuing an unkept one must not start keeping it."""
    from tools.delegate_tool import delegate_task

    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(delegate_task(tasks=[{"goal": "internal work"}], background=True, parent_agent=parent))
        _completion(out["delegation_id"])  # no origin: not kept
        cont = _continue(parent, out["subagent_ids"][0], "more")
        assert cont.get("status") == "dispatched", cont
        done = _completion(cont["delegation_id"])
    assert "children" not in cont and "child_id" not in done["results"][0]
    assert not os.path.exists(parent._session_db.db_path) or not _table(parent) or _kept_rows(parent) == []


def _table(parent):
    conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test reads a temp-home store
    try:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (dc.TABLE,)).fetchone() is not None
    finally:
        conn.close()


# ── (a) action='list' keeps its scope, query and child_id ───────────────────

def test_list_answers_with_child_id_scope_and_query_before_during_and_after_a_continue(tmp_path):
    from tools.delegate_tool import delegate_task

    def listing(**kw):
        return json.loads(delegate_task(action="list", parent_agent=parent, **kw))

    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "write the parser")
        sid, child_id = out["subagent_ids"][0], out["children"][0]["child_id"]
        _idle(parent)
        before = listing()
        assert before["scope"] == "session" and before["subagents"] == []
        assert [(c["child_id"], c["subagent_id"], c["status"], c["resumable"]) for c in before["children"]] == [
            (child_id, sid, "idle", True)]
        assert listing(query="PARSER")["count"] == 1 and listing(query="no such goal")["count"] == 0
        assert listing(scope="profile")["scope"] == "profile"
        assert "error" in listing(scope="galaxy")

        started, release = threading.Event(), threading.Event()
        model.hold = (started, release)
        cont = _continue(parent, sid, "now add tests")
        try:
            assert started.wait(10)
            during = listing()
            # The running child is named once: the live view carries it, the kept list does not repeat it.
            assert [s["subagent_id"] for s in during["subagents"]] == [sid] and during["count"] == 1, during
            assert "children" not in during
        finally:
            release.set()
        _completion(cont["delegation_id"])
        _idle(parent)
        after = listing()
    assert [(c["child_id"], c["status"]) for c in after["children"]] == [(child_id, "idle")] and after["count"] == 1


# ── the operator's case: a parent continues its own profile-route child ─────

class _ProfileRoutes:
    """A ``profile_routes`` backend: declares a per-task ``profile`` and routes the task to that profile."""
    name = "profile_routes"

    def task_parameters(self, config):
        return {"profile": {"type": "string", "enum": ["coder"], "description": "Run as this profile."}}

    def description(self, config):
        return ""

    def route_task(self, task, config):
        return {"profile": task["profile"], "label": task["profile"]} if task.get("profile") else {}


def _write_profile(home, *, host, model, key, **sections):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"},
        **sections,
    }), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\n", encoding="utf-8")


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


@pytest.fixture
def routed(tmp_path, monkeypatch):
    """A launch profile whose delegation isolates children in worktrees, a ``coder`` profile, a repository the
    parent works in, and the ``profile_routes`` backend active."""
    from tools.terminal_tool import record_session_cwd

    launch = tmp_path / "os-home" / ".hermes"
    coder = launch / "profiles" / "coder"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key",
                   delegation={"worktree_isolation": True})
    _write_profile(coder, host="coder.invalid", model="coder-model", key="coder-key")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("PROFILE_KEY", "launch-key")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "a.txt").write_text("a")
    _git("add", "a.txt", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    parent = _kept_parent(tmp_path / "parent-db")
    parent._current_task_id = "parent-task"
    record_session_cwd("parent-task", str(repo))
    with patch("tools.delegate_tool_routes._active_provider", return_value=_ProfileRoutes()):
        yield parent, coder, repo


@pytest.mark.parametrize("restart", [False, True], ids=["same-process", "after-restart"])
def test_parent_continues_its_profile_route_child_that_stopped_on_a_failure(routed, restart):
    """A profile-route child a provider failure stopped: its parent's follow-up runs on the same conversation,
    as the same profile, in the same worktree - from the in-process record, and from the durable one alone."""
    parent, coder, repo = routed
    model = _KeptModel()
    model.fail_first = True
    with patch("run_agent.AIAgent", side_effect=model):
        out, done = _spawn_kept(parent, "work as coder", profile="coder")
        sid, child_id = out["subagent_ids"][0], out["children"][0]["child_id"]
        assert done["results"][0]["status"] != "completed"
        record = _durable(parent, sid)
        tree = model.turns[0]["cwd"]
        assert (record["state"], record["pinned"]) == ("resumable", 1) and record["route_request"] == {"profile": "coder"}
        assert os.path.realpath(tree) != os.path.realpath(repo) and os.path.isdir(tree)  # its own tree, still there
        assert os.path.realpath(tree) in record["pin_paths"]
        _idle(parent)
        if restart:
            _forget_process_state()
        cont = _continue(parent, sid, "and the follow-up")
        assert cont.get("status") == "dispatched", cont
        again = _completion(cont["delegation_id"])

    first, second = model.turns
    assert second["session_id"] == first["session_id"]                                     # its conversation
    assert second["history"][0]["content"].startswith("work as coder")
    assert second["message"] == "and the follow-up"
    assert first["home"] == second["home"] == str(coder)                                   # its profile home
    assert (model.builds[1]["model"], model.builds[1]["api_key"]) == ("coder-model", "coder-key")
    assert os.path.realpath(second["cwd"]) == os.path.realpath(tree)                       # its worktree
    assert again["results"][0]["child_id"] == child_id
    assert [(r["child_id"], r["profile"]) for r in _kept_rows(parent)] == [(child_id, "coder")]
    assert not (coder / "state.db").exists()  # both records live in the PARENT's store, not the child profile's


@pytest.mark.parametrize("restart", [False, True], ids=["same-process", "after-restart"])
def test_parent_continues_its_profile_route_child_that_completed_with_a_clean_tree(routed, restart):
    """A profile-route child that COMPLETED keeps its conversation and profile on a continue. Its worktree does
    not outlive the completion (a completed child's tree is finalized as before the carry: a clean one is
    removed), so the follow-up runs in a NEW isolated tree, made at the same path off the parent's HEAD."""
    parent, coder, repo = routed
    model = _KeptModel()
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "work as coder", profile="coder")
        sid = out["subagent_ids"][0]
        tree = model.turns[0]["cwd"]
        record = _durable(parent, sid)
        assert (record["state"], record["pinned"]) == ("completed", 0) and not os.path.exists(tree)
        _idle(parent)
        if restart:
            _forget_process_state()
        cont = _continue(parent, sid, "and the follow-up")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])

    first, second = model.turns
    assert second["session_id"] == first["session_id"] and second["home"] == str(coder)
    assert second["history"][0]["content"].startswith("work as coder")
    assert os.path.realpath(second["cwd"]) == os.path.realpath(tree) != os.path.realpath(repo)


def test_a_completed_child_that_left_work_is_continued_without_that_work_in_the_parents_checkout(routed):
    """Probe of a gap in the carried feature, pinned so a fix shows up: a COMPLETED child whose isolated tree held
    work has that tree archived and removed with its branch kept. A later continue cannot re-create the tree (the
    branch name is taken), isolation degrades silently, and the follow-up runs in the PARENT's checkout, where
    the first run's work is not present. The conversation and profile are still the child's own."""
    parent, coder, repo = routed
    model = _KeptModel()
    model.leave_work = True
    with patch("run_agent.AIAgent", side_effect=model):
        out, done = _spawn_kept(parent, "work as coder", profile="coder")
        sid = out["subagent_ids"][0]
        tree = model.turns[0]["cwd"]
        worktree = done["results"][0]["worktree"]
        assert worktree["pruned"] is True and worktree["archive"] and not os.path.exists(tree)
        _idle(parent)
        cont = _continue(parent, sid, "and the follow-up")
        assert cont.get("status") == "dispatched", cont
        _completion(cont["delegation_id"])

    first, second = model.turns
    assert second["session_id"] == first["session_id"] and second["home"] == str(coder)
    assert os.path.realpath(second["cwd"]) == os.path.realpath(repo)  # the parent's checkout, not a tree of its own
    assert second["saw_work"] is False                                 # its earlier work is on the kept branch


def test_listing_still_calls_a_profile_route_child_read_only_though_its_parent_can_continue_it(routed):
    """Probe of a disagreement between the two mechanisms, pinned so a change shows up: the kept listing labels a
    child that ran as another profile ``read_only`` / not ``resumable`` even for the session that spawned it,
    while ``action='continue'`` from that same session runs it (the tests above)."""
    from tools.delegate_tool import delegate_task

    parent, _coder, _repo = routed
    model = _KeptModel()
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "work as coder", profile="coder")
        _idle(parent)
        entry = json.loads(delegate_task(action="list", parent_agent=parent))["children"][0]
    assert entry["subagent_id"] == out["subagent_ids"][0] and entry["profile"] == "coder"
    assert entry["read_only"] is True and entry["resumable"] is False, entry


def test_a_kept_row_outlives_the_durable_record_and_the_listing_still_calls_it_resumable(tmp_path):
    """Probe of a second disagreement, pinned so a change shows up: the kept row lives ``keep_idle_days`` (7 d),
    upstream's record ``resumable_ttl_hours`` (48 h). Once the durable record expired and the process forgot the
    child, a continue is refused, yet the kept listing still says ``resumable``."""
    from tools.delegate_tool import delegate_task

    model, parent = _KeptModel(), _kept_parent(tmp_path)
    with patch("run_agent.AIAgent", side_effect=model):
        out, _done = _spawn_kept(parent, "write the parser")
        sid = out["subagent_ids"][0]
        _idle(parent)
        conn = sqlite3.connect(parent._session_db.db_path)  # health: allow HX015 -- test edits a temp-home store
        with conn:
            conn.execute("UPDATE resumable_subagents SET expires_at=? WHERE subagent_id=?", (time.time() - 1, sid))
        conn.close()
        _forget_process_state()
        refused = _continue(parent, sid, "more")
        assert "No finished subagent" in refused.get("error", ""), refused
        entry = json.loads(delegate_task(action="list", parent_agent=parent))["children"][0]
    assert entry["status"] == "idle" and entry["resumable"] is True, entry
