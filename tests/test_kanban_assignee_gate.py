"""SPEC-0031: ``create_task`` refuses non-profile assignees at create time.

AC1 (prevention): a create whose assignee is not a real Hermes profile raises
``ValueError`` naming real profiles and creates NO row. AC2 (legitimate lanes
unbroken): creating without ``--assignee`` still works, a real profile name still
works, and ``''`` / ``'-'`` refuse (the empty arm raises the canonical
``profile name cannot be empty`` from ``normalize_profile_name``).

See docs/spec/2026-09-09-kanban-assignee-create-gate.md for the full mechanism:
upstream refuses all non-profile names in ``create_task``; the sweep-heal pass
that rewrites historical role-labeled rows lives fleet-side in hermes-gates.
"""

from __future__ import annotations

import pytest

import hermes_cli.kanban_db as kb
import hermes_cli.kanban_db_connect as kbc
from hermes_cli import profiles


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and one real profile.

    Mirrors ``tests/hermes_cli/test_kanban_db.py``'s fixture of the same name:
    the per-test HERMES_HOME sandbox plus a real ``profiles/alice`` directory so
    the natural (refusing) ``profile_exists`` hermetic stub passes ``alice`` but
    refuses synthetic names.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    alice = home / "profiles" / "alice"
    alice.mkdir(parents=True)
    (alice / "config.yaml").write_text("alias:\n")
    kb.init_db()
    return home


@pytest.fixture
def guarded_profiles(monkeypatch):
    """Explicit control: only ``default`` and the real ``alice`` profile exist.

    This mirrors how the genuinely-assignee-sensitive dispatch tests keep
    explicit control over ``profile_exists`` (see ``tests/hermes_cli/
    test_kanban_host_cap.py``) — the top-level autouse stub also refuses, but an
    explicit predicate makes the gate's accept/refuse arms unambiguous here.
    """

    def _only_real(name: str) -> bool:
        return name == "default" or name == "alice"

    monkeypatch.setattr(profiles, "profile_exists", _only_real)


def _task_count(title: str) -> int:
    with kbc.connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title=?", (title,)
        ).fetchone()[0]


def test_bad_assignee_refuses_and_creates_no_row(kanban_home, guarded_profiles):
    """AC1 falsifier: refusal + no-row for a non-profile (role) labels."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError) as ei:
            kb.create_task(conn, title="x", assignee="spec-reviewer")
        msg = str(ei.value)
        # Repair message: names the real-profiles repair path.
        assert "not a real Hermes profile" in msg
        assert "silently skip" in msg
        assert "alice" in msg  # up-to-~10 real profile names listed
    assert _task_count("x") == 0  # no row landed on ANY board


def test_dash_refuses(kanban_home, guarded_profiles):
    """AC2 falsifier: ``--assignee -`` stores a literal dash at baseline and
    REFUSES post-fix (``profile_exists('-')`` is False — the arm flips)."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="not a real Hermes profile"):
            kb.create_task(conn, title="dash", assignee="-")
    assert _task_count("dash") == 0


def test_empty_assignee_refuses(kanban_home, guarded_profiles):
    """AC2 falsifier: ``--assignee ''`` refuses with the canonical empty-name
    error raised by ``normalize_profile_name`` inside canonicalization."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="profile name cannot be empty"):
            kb.create_task(conn, title="empty", assignee="")


def test_profile_named_create_unbroken(kanban_home, guarded_profiles):
    """AC2 falsifier: a profile-named create must NOT refuse."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ok", assignee="alice")
    assert tid
    assert _task_count("ok") == 1


def test_unassigned_create_unbroken(kanban_home, guarded_profiles):
    """AC2: creating WITHOUT assignee (omitted = real ``None``) still works —
    unassigned cards are legitimate (``kanban.default_assignee`` may claim them)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unassigned")
    assert tid
    assert _task_count("unassigned") == 1


def test_top_level_refusing_stub_is_active(kanban_home):
    """The autouse hermetic ``profile_exists`` stub must refuse by default.

    No explicit patch here: if the top-level refusing stub in ``tests/conftest.py``
    is working, ``create_task`` with a synthetic assignee raises even though the
    test does nothing special. Guards against the fixture strategy silently
    failing to re-expose the gate (falsifier for AC4's stub semantics).
    """
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="not a real Hermes profile"):
            kb.create_task(conn, title="stub-active", assignee="ghost")