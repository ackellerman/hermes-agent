"""SPEC-0032 acceptance tests: config-driven board resolution layers.

Covers AC1-AC8 from docs/spec/2026-09-09-board-resolution-config-layers.md:

- AC1  project binding beats an unrelated env pin (RED falsifier in old code)
- AC2  switch ON preserves the env-dependent behaviour verbatim
- AC3  stale-pin cleanup: boot pin pops the inherited env var (switch off)
- AC4  layer contract: ContextVar beats everything; current-file/DEFAULT tail
- AC5  precedence inside the project layer (projects.db board_slug > workdir)
- AC6  worker exemption (HERMES_KANBAN_DB set -> env honoured unconditionally)
- AC6b worker pin survives boot (the pop is skipped for workers)
- AC7  suite green (asserted by the repo-wide `-k kanban` run; config defaults
       checked here)
- AC8  auto-decompose steering threads board=slug explicitly and never touches
       HERMES_KANBAN_BOARD (structural + behavioral probes)

The ``set_kanban_cfg`` helper monkeypatches ``hermes_cli.config.load_config``,
which both ``kanban_db._board_resolution_config`` and
``main_tui_launch._kanban_env_board_pin_enabled`` reach through a module-level
``from hermes_cli.config import load_config`` at call time, so the patch is
observed by both.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import main_tui_launch
from hermes_cli import projects_db as pdb
from hermes_cli.config import DEFAULT_CONFIG


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "b@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Board Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


def _seed_triage_task(board: str, title: str = "triage me") -> str:
    with kbc.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=title, triage=True)


@pytest.fixture()
def kanban_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a fresh kanban DB + projects.db, no ambient
    board env. Switch defaults OFF (the new default)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.init_db()
    kb.clear_project_board_cache()
    set_kanban_cfg = self_set_kanban_cfg(monkeypatch)
    set_kanban_cfg(env_board_pin=False, default_board="")
    return home


def self_set_kanban_cfg(monkeypatch):
    def _set(env_board_pin: bool = False, default_board: str = "") -> None:
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"kanban": {"env_board_pin": env_board_pin, "default_board": default_board}},
        )
    return _set


@pytest.fixture()
def set_kanban_cfg(monkeypatch):
    return self_set_kanban_cfg(monkeypatch)


def _board(root: str) -> str:
    """Name a test board deterministically from a repo path (for lexicographic
    assertions keep the slug within a fixed prefix)."""
    return root.rsplit("/", 1)[-1].replace(".", "-").replace("_", "-").lower()


# ---------------------------------------------------------------------------
# AC1: project binding beats an unrelated env pin (switch off)
# ---------------------------------------------------------------------------

def test_ac1_project_binding_beats_unrelated_env(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    # An env board that exists but is not keyed to this repo.
    kb.create_board("envboard")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "envboard")

    repo = tmp_path / "repo-b"
    _init_git_repo(repo)
    board_b = _board(str(repo))  # e.g. "repo-b"
    kb.create_board(board_b, default_workdir=str(repo))

    monkeypatch.chdir(repo)
    assert kb.get_current_board() == board_b


# ---------------------------------------------------------------------------
# AC2: switch ON = today's env behaviour (byte-for-byte parity on env paths)
# ---------------------------------------------------------------------------

def test_ac2_switch_on_env_pin_returns_env_board(kanban_env, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=True)
    kb.create_board("wanted")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "wanted")
    assert kb.get_current_board() == "wanted"


# ---------------------------------------------------------------------------
# AC3: stale-pin cleanup (switch off) — inheriting an unrelated board is gone
# ---------------------------------------------------------------------------

def test_ac3_boot_pin_pops_inherited_env_when_switch_off(kanban_env, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "staleboard")
    main_tui_launch._pin_kanban_board_env()
    assert "HERMES_KANBAN_BOARD" not in os.environ


# ---------------------------------------------------------------------------
# AC4: layer contract
# ---------------------------------------------------------------------------

def test_ac4_context_override_beats_everything(kanban_env, tmp_path, monkeypatch):
    kb.create_board("override-board")
    kb.create_board("envboard")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "envboard")
    repo = tmp_path / "repo-ctx"
    _init_git_repo(repo)
    kb.create_board(_board(str(repo)), default_workdir=str(repo))
    monkeypatch.chdir(repo)
    with kb.scoped_current_board("override-board"):
        assert kb.get_current_board() == "override-board"


def test_ac4_current_file_and_default_tail(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    # Out-of-project session: no project row, no workdir match, no profile default.
    set_kanban_cfg(env_board_pin=False, default_board="")
    kb.create_board("curfile-board")
    (kb.current_board_path()).write_text("curfile-board\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # not a repo
    assert kb.get_current_board() == "curfile-board"

    # No current file -> DEFAULT_BOARD.
    kb.current_board_path().unlink()
    assert kb.get_current_board() == "default"


def test_ac4_profile_default_layer(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False, default_board="profile-board")
    kb.create_board("profile-board")
    monkeypatch.chdir(tmp_path)
    assert kb.get_current_board() == "profile-board"


# ---------------------------------------------------------------------------
# AC5: precedence inside the project layer
# ---------------------------------------------------------------------------

def test_ac5_projects_db_board_slug_beats_workdir(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    repo = tmp_path / "repo-p"
    _init_git_repo(repo)
    # Two distinct boards: one the workdir matches, one the project row names.
    workdir_board = _board(str(repo)) + "-wd"
    slug_board = _board(str(repo)) + "-slug"
    kb.create_board(workdir_board, default_workdir=str(repo))
    kb.create_board(slug_board)

    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Repo P", folders=[str(repo)])
        pdb.update_project(conn, pid, board_slug=slug_board)

    monkeypatch.chdir(repo)
    assert kb.get_current_board() == slug_board  # explicit row wins over workdir


def test_ac5_unset_board_slug_falls_to_workdir(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    repo = tmp_path / "repo-q"
    _init_git_repo(repo)
    workdir_board = _board(str(repo))
    kb.create_board(workdir_board, default_workdir=str(repo))

    with pdb.connect_closing() as conn:
        pdb.create_project(conn, name="Repo Q", folders=[str(repo)])  # no board_slug

    monkeypatch.chdir(repo)
    assert kb.get_current_board() == workdir_board


def test_ac5_workdir_match_is_deterministic(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    """Lexicographically first slug among workdir matches wins."""
    set_kanban_cfg(env_board_pin=False)
    repo = tmp_path / "repo-r"
    _init_git_repo(repo)
    for slug in ("zzz-workdir", "aaa-workdir"):
        kb.create_board(slug, default_workdir=str(repo))
    monkeypatch.chdir(repo)
    assert kb.get_current_board() == "aaa-workdir"


# ---------------------------------------------------------------------------
# AC6 / AC6b: worker contract
# ---------------------------------------------------------------------------

def test_ac6_worker_env_honoured_unconditionally(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    claimed = "claimed-board"
    kb.create_board(claimed)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "worker-kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", claimed)
    # cwd inside a DIFFERENT project's repo.
    repo = tmp_path / "other-repo"
    _init_git_repo(repo)
    kb.create_board(_board(str(repo)), default_workdir=str(repo))
    monkeypatch.chdir(repo)
    assert kb.get_current_board() == claimed


def test_ac6b_worker_pin_survives_boot(kanban_env, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    claimed = "claimed-board"
    kb.create_board(claimed)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_env / "worker-kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", claimed)
    main_tui_launch._pin_kanban_board_env()
    assert os.environ.get("HERMES_KANBAN_BOARD") == claimed


# ---------------------------------------------------------------------------
# AC7: config defaults + suite-green (the full `-k kanban` run is the gate)
# ---------------------------------------------------------------------------

def test_ac7_config_defaults():
    kanban_defaults = DEFAULT_CONFIG["kanban"]
    assert kanban_defaults["env_board_pin"] is False
    assert kanban_defaults["default_board"] == ""


# ---------------------------------------------------------------------------
# AC8: auto-decompose steering (structural + behavioral probes)
# ---------------------------------------------------------------------------

def _build_dispatcher():
    from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher
    settings = _DispatcherSettings(
        interval=60.0, max_spawn=None, max_in_progress=8,
        failure_limit=2, stale_timeout_seconds=0, reconcile_orphans=True,
        default_assignee=None, max_in_progress_per_profile=None,
    )
    return _KanbanDispatcher(kb, settings)


def test_ac8_structural_no_env_write_in_source():
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher
    src = inspect.getsource(_KanbanDispatcher.auto_decompose_tick)
    assert "HERMES_KANBAN_BOARD" not in src


def test_ac8_behavioral_two_board_tick(kanban_env, tmp_path, monkeypatch, set_kanban_cfg):
    set_kanban_cfg(env_board_pin=False)
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _init_git_repo(repo_a)
    _init_git_repo(repo_b)
    board_a = _board(str(repo_a))
    board_b = _board(str(repo_b))
    if board_a == board_b:  # slug collision guard
        board_b = board_b + "-2"
    kb.create_board(board_a, default_workdir=str(repo_a))
    kb.create_board(board_b, default_workdir=str(repo_b))

    tid_a = _seed_triage_task(board_a, title="task on A")
    tid_b = _seed_triage_task(board_b, title="task on B")

    from hermes_cli import kanban_decompose as kdc

    def _fake_aux(verb, task_id, *, aux_task, system, user, max_tokens, timeout, log=None):
        blob = {
            "fanout": True,
            "tasks": [{"title": "child", "assignee": "someone"}],
        }
        return json.dumps(blob), ""

    def _fake_routing():
        from hermes_cli.kanban_decompose import _Routing
        return _Routing(
            orchestrator="orch", default_assignee="", auto_promote=True,
            roster=[], valid_names={"someone"},
        )

    monkeypatch.setattr(kdc, "_call_aux", _fake_aux)
    monkeypatch.setattr(kdc, "_load_routing", _fake_routing)

    # Behavioral probe A10: snapshot entire os.environ, run the tick, assert
    # equality after — _isolate_kanban_board_env precedent, catches direct
    # os.environ writes that a __setitem__ monkeypatch cannot.
    snap_before = dict(os.environ)

    disp = _build_dispatcher()
    n = disp.auto_decompose_tick(auto_decompose_per_tick=10)

    assert n >= 2, f"expected both boards decomposed, got {n}"
    assert dict(os.environ) == snap_before, "watcher path wrote HERMES_KANBAN_BOARD"

    # Child rows must land on the passed board's DB — verify each board's DB
    # holds its own decomposed child and its own triage task left triage.
    for board, tid in ((board_a, tid_a), (board_b, tid_b)):
        with kbc.connect_closing(board=board) as conn:
            root = kb.get_task(conn, tid)
            children = [t for t in kb.list_tasks(conn) if t.title == "child"]
            assert root is not None and root.status == "todo", f"{board}: root not promoted"
            assert children, f"{board}: child missing from its own DB"
