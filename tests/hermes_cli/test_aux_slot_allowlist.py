"""The dashboard/web surface accepts every auxiliary slot the CLI picker ships.

``auxiliary.voice_brief`` shipped in ``DEFAULT_CONFIG`` and in the ``hermes model`` picker
(``_AUX_TASKS``), but the three mirror lists that gate the GUI surface were never extended:
``POST /api/model/set`` answered 400 ``unknown auxiliary task: voice_brief`` and the Web/Desktop
Models page had no row to render for the slot.

Pinned here:

  - ``_AUX_TASK_SLOTS`` carries ``voice_brief`` immediately after ``voice_chat`` (the order
    ``_AUX_TASKS`` lists it in, so the pickers agree)
  - ``_aux_task_slots()`` — what the REST validation at ``_apply_aux_assignment_sync`` reads —
    serves it too
  - the two GUI mirrors (``web/src/lib/aux-tasks.ts``, the Desktop ``aux-task-rows.tsx``) have
    the row, and the seven Desktop i18n catalogs have its copy
  - the tuple ships no key the defaults do not (the invariant its "keep in sync" comment claims)

One direction only: ``memory_query_rewrite``, ``tts_audio_tags`` and ``group_router`` are in
``DEFAULT_CONFIG["auxiliary"]`` but not in the tuple, and that pre-existing gap is out of scope —
it changes unrelated GUI rows.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli.web_server_config import _AUX_TASK_SLOTS, _aux_task_slots

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_AUX_TASKS = _REPO_ROOT / "web" / "src" / "lib" / "aux-tasks.ts"
_DESKTOP_AUX_TASK_ROWS = _REPO_ROOT / "apps" / "desktop" / "src" / "app" / "settings" / "aux-task-rows.tsx"
_DESKTOP_I18N = _REPO_ROOT / "apps" / "desktop" / "src" / "i18n"
_AUX_TASK_COPY_LOCALES = ("en", "de", "es", "fr", "ja", "ru", "zh")


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
    direction. The reverse does not hold (``memory_query_rewrite``, ``tts_audio_tags``,
    ``group_router`` are defaults-only) and is deliberately not asserted.
    """
    missing = [slot for slot in _AUX_TASK_SLOTS if slot not in _default_aux_slots()]

    assert not missing, (
        f"_AUX_TASK_SLOTS serves slots DEFAULT_CONFIG['auxiliary'] does not ship: {missing!r}"
    )


# ── GUI mirrors ────────────────────────────────────────────────────────────


def test_web_builtin_aux_tasks_has_a_voice_brief_row():
    """The Web Models page renders ``BUILTIN_AUX_TASKS``; no row = no picker entry."""
    source = _WEB_AUX_TASKS.read_text(encoding="utf-8")

    assert '"voice_brief"' in source, (
        'web/src/lib/aux-tasks.ts BUILTIN_AUX_TASKS has no "voice_brief" row'
    )


def test_desktop_aux_tasks_has_a_voice_brief_row():
    """The Desktop Models page renders ``AUX_TASKS`` (labels come from the i18n catalogs)."""
    source = _DESKTOP_AUX_TASK_ROWS.read_text(encoding="utf-8")

    assert "'voice_brief'" in source, (
        "apps/desktop/src/app/settings/aux-task-rows.tsx AUX_TASKS has no 'voice_brief' row"
    )


def test_desktop_i18n_carries_voice_brief_copy_in_every_locale():
    """A missing i18n key is legal (the type is ``Record<string, AuxTaskCopy>``) but the row then
    renders the raw key, so every catalog needs the entry."""
    missing = []
    for locale in _AUX_TASK_COPY_LOCALES:
        path = _DESKTOP_I18N / f"{locale}_aux_tasks.ts"
        if "voice_brief:" not in path.read_text(encoding="utf-8"):
            missing.append(path.name)

    assert not missing, f"Desktop i18n catalogs without a voice_brief entry: {missing!r}"
