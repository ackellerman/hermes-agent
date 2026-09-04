"""Tests for kanban goal_mode — per-card Ralph-style goal loop.

Covers three layers:

1. DB: goal_mode / goal_max_turns persist through create_task + from_row,
   and a legacy DB (without the columns) migrates cleanly.
2. Spawn: _default_spawn sets the HERMES_KANBAN_GOAL_MODE env vars only
   when the card opts in.
3. Loop: goals.run_kanban_goal_loop continuation / completion / budget
   behaviour, driven entirely through injected callbacks (no live model).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import goals


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------





def test_legacy_db_migrates_goal_columns(tmp_path, monkeypatch):
    """A tasks table created without goal columns must gain them on init."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal legacy schema: tasks table missing goal_mode / goal_max_turns.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    # init_db runs the additive migration.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "goal_mode" in cols
        assert "goal_max_turns" in cols
        task = kb.get_task(conn, "legacy1")
    # Existing row keeps the safe default.
    assert task.goal_mode is False
    assert task.goal_max_turns is None


# ---------------------------------------------------------------------------
# Spawn env
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Goal loop logic (callback-injected, no live model)
# ---------------------------------------------------------------------------

def _patch_judge(monkeypatch, verdicts):
    """Make judge_goal return a scripted sequence of verdicts."""
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        v = seq.pop(0) if seq else "done"
        # 5-tuple contract: verdict, reason, parse failure, wait, transport failure.
        return v, f"scripted:{v}", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


def test_loop_stops_when_worker_already_completed(monkeypatch):
    # Worker called kanban_complete on its first turn — no judging needed.
    _patch_judge(monkeypatch, ["continue"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="done already",
    )
    assert res["outcome"] == "completed_by_worker"
    assert turns == []  # no extra turns


def test_judge_transport_failure_does_not_spend_worker_turns(monkeypatch):
    """Overlay K2. On 2026-09-04 three sessions each burned 19 full worker
    turns in ~10 s because judge_goal returned transport_failed=True (judge
    RateLimitError) and the loop treated it as a plain 'continue', re-sending
    the whole context per turn until the 20-turn budget was gone — then
    sticky-blocked the card as 'budget exhausted', which tripped the block
    loop and triage. A judge that cannot be reached is not evidence about
    the work: the loop must NOT run another worker turn on it, and after
    DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES it must block as transient
    naming the judge, not the worker."""
    calls = {"n": 0}

    def _down_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        calls["n"] += 1
        return "continue", "judge error: RateLimitError", False, None, True

    monkeypatch.setattr(goals, "judge_goal", _down_judge)
    monkeypatch.setattr(goals, "_judge_retry_sleep", lambda s: None)
    turns: list = []
    blocks: list = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "still working",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocks.append(r),
        first_response="first",
        max_turns=20,
    )
    assert turns == [], f"worker turns were spent on an unreachable judge: {len(turns)}"
    assert calls["n"] == goals.DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
    assert res["outcome"] == "blocked_judge_unreachable"
    assert res["turns_used"] == 1  # only the first (real) turn counts
    assert len(blocks) == 1 and "judge" in blocks[0].lower()
    assert "RateLimitError" in blocks[0]


def test_judge_transport_failure_then_recovery_continues_normally(monkeypatch):
    """A transient judge blip (below the limit) is retried at the judge, not
    at the worker: one transport failure then a real 'continue' → exactly
    one worker turn, then 'done' → finalize nudge."""
    seq = [
        ("continue", "judge error: timeout", False, None, True),
        ("continue", "keep going", False, None, False),
        ("done", "looks complete", False, None, False),
    ]

    def _judge(goal, response, subgoals=None, background_processes=None, **_kw):
        return seq.pop(0) if seq else ("done", "done", False, None, False)

    monkeypatch.setattr(goals, "judge_goal", _judge)
    monkeypatch.setattr(goals, "_judge_retry_sleep", lambda s: None)
    turns: list = []
    status = {"s": "running"}

    def _run_turn(p):
        turns.append(p)
        if len(turns) >= 2:
            status["s"] = "done"
        return "progress"

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=_run_turn,
        task_status_fn=lambda: status["s"],
        block_fn=lambda r: pytest.fail(f"should not block: {r}"),
        first_response="first",
        max_turns=20,
    )
    assert res["outcome"] == "completed_by_worker"
    assert len(turns) == 2






# ---------------------------------------------------------------------------
# CLI judge gate tests (hermes kanban complete bypass fix)
# ---------------------------------------------------------------------------

class TestCLIJudgeGate:
    """hermes kanban complete must apply the same goal_mode judge gate as the
    kanban_complete tool (Issue #38367 sibling gap).

    Uses mocks for kb.get_task and kb.complete_task to avoid depending on the
    full kanban_db schema; the gate logic is the unit under test.
    """

    def _run(self, monkeypatch, *, goal_mode=True, judge_available=True,
             verdict="done", reason="", complete_ok=True, summary="done"):
        import argparse
        import types
        from unittest.mock import MagicMock
        from hermes_cli.kanban import _cmd_complete

        fake_task = types.SimpleNamespace(
            goal_mode=goal_mode,
            title="Finish report",
            body="acceptance: criteria",
        )
        fake_conn = MagicMock()
        complete_calls: list = []

        def fake_connect_closing():
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                yield fake_conn
            return _cm()

        def fake_complete_task(conn, tid, **kw):
            complete_calls.append(tid)
            return complete_ok

        monkeypatch.setattr("hermes_cli.kanban.kb.get_task", lambda conn, tid: fake_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.complete_task", fake_complete_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.connect_closing", fake_connect_closing)
        monkeypatch.setattr("hermes_cli.kanban._worker_run_id_for", lambda _: None)

        _aux_client = (object(), "judge-model") if judge_available else (None, None)
        monkeypatch.setattr(
            "agent.auxiliary_client.get_text_auxiliary_client",
            lambda name: _aux_client,
        )
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda **kw: (verdict, reason, False, None, False),
        )

        args = argparse.Namespace(task_ids=["t1"], summary=summary, result=None, metadata=None)
        return _cmd_complete(args), complete_calls

    def test_judge_rejects_premature_completion(self, monkeypatch):
        rc, complete_calls = self._run(
            monkeypatch, verdict="continue", reason="criteria not met"
        )
        assert rc != 0, "judge rejection must produce non-zero exit code"
        assert complete_calls == [], (
            "complete_task must NOT be invoked when the judge rejects"
        )


    def test_non_goal_mode_task_skips_gate(self, monkeypatch):
        """Plain (non-goal_mode) tasks are never sent to the judge."""
        rc, complete_calls = self._run(monkeypatch, goal_mode=False)
        assert rc == 0
        assert complete_calls == ["t1"]

    def test_judge_blocked_verdict_rejects_completion(self, monkeypatch, capsys):
        """#100954: an unachievable goal must not complete silently.

        The judge's ``blocked`` verdict is a refusal, not a completion —
        ``complete_task`` must never run and stderr must steer the user
        toward re-scoping / recording the block.
        """
        rc, complete_calls = self._run(
            monkeypatch,
            verdict="blocked",
            reason="the target repository does not exist",
        )
        err = capsys.readouterr().err
        assert rc != 0, "blocked verdict must reject the completion"
        assert complete_calls == [], "an unachievable goal must never reach complete_task"
        assert "unachievable" in err.lower()
        assert "kanban block" in err.lower()


# ---------------------------------------------------------------------------
# Overlay K5: kanban worker turn budget honours config (card > config > default)
# ---------------------------------------------------------------------------

def test_kanban_worker_max_turns_precedence(monkeypatch):
    """The kanban worker path hard-wired DEFAULT_MAX_TURNS (20) and ignored
    ``goals.max_turns`` in config.yaml; each 'turn' is an unbounded agent
    conversation (248 API calls measured for one 20-turn run on 2026-09-04).
    Precedence must be: card.goal_max_turns > config goals.kanban_max_turns
    > goals.max_turns > goals.DEFAULT_KANBAN_MAX_TURNS."""
    import types
    r = goals.resolve_kanban_max_turns
    assert r(types.SimpleNamespace(goal_max_turns=7), {"goals": {"max_turns": 20}}) == 7
    assert r(types.SimpleNamespace(goal_max_turns=None), {"goals": {"kanban_max_turns": 6, "max_turns": 20}}) == 6
    assert r(types.SimpleNamespace(goal_max_turns=None), {"goals": {"max_turns": 12}}) == 12
    assert r(types.SimpleNamespace(goal_max_turns=None), {}) == goals.DEFAULT_KANBAN_MAX_TURNS
    assert r(types.SimpleNamespace(goal_max_turns=0), {}) == goals.DEFAULT_KANBAN_MAX_TURNS
    assert r(None, {"goals": {"max_turns": "bad"}}) == goals.DEFAULT_KANBAN_MAX_TURNS
    assert goals.DEFAULT_KANBAN_MAX_TURNS < goals.DEFAULT_MAX_TURNS
