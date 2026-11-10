"""The dashboard/web surface accepts every auxiliary slot the CLI picker ships.

``auxiliary.voice_brief`` shipped in ``DEFAULT_CONFIG`` and in the ``hermes model`` picker
(``_AUX_TASKS``), but the backend allowlist the dashboard REST surface validates against was
never extended: ``POST /api/model/set`` answered 400 ``unknown auxiliary task: voice_brief``.

Pinned here:

  - ``_AUX_TASK_SLOTS`` carries ``voice_brief`` immediately after ``voice_chat`` (the order
    ``_AUX_TASKS`` lists it in, so the pickers agree)
  - ``_aux_task_slots()`` — what the REST validation at ``_apply_aux_assignment_sync`` reads —
    serves it too
  - the tuple ships no key the defaults do not (the invariant its "keep in sync" comment claims)

The Web ``BUILTIN_AUX_TASKS`` and Desktop ``AUX_TASKS`` mirrors are asserted in the vitest suites
that own them (``web/src/lib/aux-tasks.test.ts``,
``apps/desktop/src/app/settings/aux-task-rows.test.ts``): a Python test reading ``.ts``/``.tsx``
text is banned outright, and it would not run on a JS-only PR where the lane classifier skips
``tests/*.py`` (tests/AGENTS.md "Never read source code in tests", "Placement (CI lanes)";
web/AGENTS.md Rules).

One direction only: the reverse gap is real and deliberately out of scope — ``memory_query_rewrite``,
``tts_audio_tags``, ``group_router``, ``goal_judge``, ``monitor``, ``background_review``,
``moa_reference`` and ``moa_aggregator`` are in ``DEFAULT_CONFIG["auxiliary"]`` but not in the
tuple (plus four non-slot scalars). Filling them changes unrelated GUI rows.
"""

from __future__ import annotations

from hermes_cli.web_server_config import _AUX_TASK_SLOTS, _aux_task_slots


def _default_aux_slots() -> dict:
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return DEFAULT_CONFIG.get("auxiliary") or {}


# ── Backend allowlist ──────────────────────────────────────────────────────


def test_aux_task_slots_carries_voice_brief_after_voice_chat():
    """``voice_brief`` sits one past ``voice_chat``, the position ``_AUX_TASKS`` lists it in.

    The order is what the Web/Desktop pickers render, so a mismatch silently reorders the page.
    """
    assert "voice_brief" in _AUX_TASK_SLOTS, (
        "voice_brief missing from _AUX_TASK_SLOTS (dashboard REST allowlist)"
    )
    assert _AUX_TASK_SLOTS.index("voice_brief") == _AUX_TASK_SLOTS.index("voice_chat") + 1


def test_served_aux_slots_carries_voice_brief():
    """``_aux_task_slots()`` is what ``POST /api/model/set`` validates against (``:927-928``)."""
    served = _aux_task_slots()

    assert "voice_brief" in served, (
        "voice_brief missing from _aux_task_slots() — POST /api/model/set answers "
        "'unknown auxiliary task: voice_brief'"
    )
    assert served.index("voice_brief") == served.index("voice_chat") + 1


def test_every_aux_task_slot_ships_in_the_defaults():
    """No slot the tuple serves that ``DEFAULT_CONFIG["auxiliary"]`` does not ship.

    The "keep in sync with DEFAULT_CONFIG['auxiliary']" comment on the tuple claims this
    direction only. The reverse does not hold and is deliberately not asserted: eight
    defaults-only slots (``memory_query_rewrite``, ``tts_audio_tags``, ``group_router``,
    ``goal_judge``, ``monitor``, ``background_review``, ``moa_reference``, ``moa_aggregator``)
    plus four non-slot scalars have no tuple entry.
    """
    missing = [slot for slot in _AUX_TASK_SLOTS if slot not in _default_aux_slots()]

    assert not missing, (
        f"_AUX_TASK_SLOTS serves slots DEFAULT_CONFIG['auxiliary'] does not ship: {missing!r}"
    )
