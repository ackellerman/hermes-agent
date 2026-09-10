"""Shared fixtures for tui_gateway kanban tests.

Appendix B7 of SPEC-0031: this tree had no conftest of its own, so the
tui_gateway test file that creates cards with synthetic assignees
(test_kanban_notify_poller.py) was left uncovered by the
``all_assignees_spawnable`` re-allow fixture. ``create_task`` now refuses
non-profile assignees, and the top-level ``_hermetic_profile_exists`` stub in
tests/conftest.py keeps that gate ACTIVE by default — so this tree needs its
OWN autouse re-allow fixture, same semantics as the other sub-conftests.
"""

from __future__ import annotations

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