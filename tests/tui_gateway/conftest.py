"""tui_gateway test fixtures.

Several files here import ``tui_gateway.server`` inside a ``patch.dict("sys.modules", {"hermes_constants":
MagicMock(...)})`` window so the module binds a fixed home. The server's import graph reaches
``agent.process_bootstrap`` → ``hermes_bootstrap``, which is process boot: PM dependency activation reads
the real install root through ``hermes_constants`` and exits the process when that is a MagicMock.
Importing it once here, before any window opens, keeps boot out of the mocked import.
"""

from __future__ import annotations

import hermes_bootstrap  # noqa: F401
import pytest


@pytest.fixture(autouse=True)
def all_assignees_spawnable(monkeypatch):
    """Re-allow synthetic assignees for tui_gateway kanban tests.

    ``create_task`` refuses non-profile assignees (SPEC-0031), and the
    top-level ``_hermetic_profile_exists`` stub in tests/conftest.py keeps that
    gate ACTIVE by default. tui_gateway tests create cards with synthetic
    assignees, so this autouse fixture re-allows them (overrides the refusing
    stub). Tests that genuinely need refusal patch ``profile_exists``
    themselves.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
