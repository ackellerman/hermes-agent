"""``auxiliary.voice_brief`` ships in the defaults, the picker roster and the docs.

Covers the config-surface slice of the voice-brief auxiliary-task plan:

  - ``DEFAULT_CONFIG["auxiliary"]["voice_brief"]`` is the ``_aux(15)`` shape, carries no
    ``reasoning_effort`` key, and carries ``task_chain_only: true``
  - ``_AUX_TASKS`` has a ``("voice_brief", "Voice brief", <description>)`` row
  - ``_format_aux_current`` renders the fresh block as ``auto`` and a saved pin as
    ``openrouter · google/gemini-3.6-flash``
  - ``_reset_aux_to_auto`` clears only the routing fields and leaves ``timeout`` and
    ``task_chain_only`` untouched
  - ``website/docs/user-guide/configuration.md`` documents the slot and the Discord switch

Pure-function and content tests; the interactive menu loops are not covered here.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli.config import load_config, save_config
from hermes_cli.main_provider_setup import _AUX_TASKS, _format_aux_current, _reset_aux_to_auto, _save_aux_choice

_CONFIGURATION_DOC = (
    Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "configuration.md"
)


def _auxiliary_blocks() -> dict:
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return DEFAULT_CONFIG.get("auxiliary") or {}


def _isolate_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)


# ── DEFAULT_CONFIG block ────────────────────────────────────────────────────


def test_default_voice_brief_block_is_the_aux_15_auto_shape():
    """The slot ships as a plain auto block with a 15s timeout and nothing else set."""
    blocks = _auxiliary_blocks()
    assert "voice_brief" in blocks, "DEFAULT_CONFIG['auxiliary'] has no voice_brief slot"

    block = blocks["voice_brief"]
    assert block["provider"] == "auto"
    assert block["model"] == ""
    assert block["base_url"] == ""
    assert block["api_key"] == ""
    assert block["timeout"] == 15
    assert block["extra_body"] == {}


def test_default_voice_brief_block_omits_reasoning_effort():
    """No ``reasoning_effort`` key: the brief restates one sentence, so the pinned model's
    own default effort applies (same convention as ``memory_query_rewrite``)."""
    blocks = _auxiliary_blocks()
    assert "voice_brief" in blocks, "DEFAULT_CONFIG['auxiliary'] has no voice_brief slot"

    assert "reasoning_effort" not in blocks["voice_brief"]


def test_default_voice_brief_block_is_task_chain_only():
    """``task_chain_only: true`` ships on the block (read by the pinned-route resolver;
    inert while the provider stays ``auto``)."""
    blocks = _auxiliary_blocks()
    assert "voice_brief" in blocks, "DEFAULT_CONFIG['auxiliary'] has no voice_brief slot"

    assert blocks["voice_brief"].get("task_chain_only") is True


# ── Picker roster ──────────────────────────────────────────────────────────


def test_aux_tasks_roster_has_one_voice_brief_row():
    rows = [(key, name, desc) for key, name, desc in _AUX_TASKS if key == "voice_brief"]
    assert len(rows) == 1, f"expected exactly one voice_brief roster row, got {rows!r}"

    key, name, description = rows[0]
    assert key == "voice_brief"
    assert name == "Voice brief"
    assert description, "the roster row needs a short description for the picker"


# ── _format_aux_current ────────────────────────────────────────────────────


def test_format_aux_current_renders_fresh_voice_brief_block_as_auto():
    block = _auxiliary_blocks().get("voice_brief")
    assert block is not None, "DEFAULT_CONFIG['auxiliary'] has no voice_brief slot"

    assert _format_aux_current(block) == "auto"


def test_format_aux_current_renders_saved_voice_brief_pin(tmp_path, monkeypatch):
    """A saved pin renders ``provider · model`` and keeps the shipped ``task_chain_only``."""
    _isolate_home(tmp_path, monkeypatch)

    _save_aux_choice("voice_brief", provider="openrouter", model="google/gemini-3.6-flash")

    saved = load_config()["auxiliary"]["voice_brief"]
    assert saved.get("task_chain_only") is True, (
        "saving a pin must not drop the shipped task_chain_only value"
    )
    assert _format_aux_current(saved) == "openrouter · google/gemini-3.6-flash"


# ── _reset_aux_to_auto ─────────────────────────────────────────────────────


def test_reset_aux_to_auto_clears_voice_brief_routing_but_keeps_task_chain_only(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)

    cfg = load_config()
    entry = (cfg.get("auxiliary") or {}).get("voice_brief")
    assert entry is not None, "DEFAULT_CONFIG['auxiliary'] has no voice_brief slot"

    entry.update({
        "provider": "openrouter",
        "model": "google/gemini-3.6-flash",
        "base_url": "https://brief.example/v1",
        "api_key": "sk-test-not-a-real-key",
        "timeout": 25,
    })
    assert entry.get("task_chain_only") is True
    save_config(cfg)

    assert _reset_aux_to_auto() >= 1

    after = load_config()["auxiliary"]["voice_brief"]
    assert after["provider"] == "auto"
    assert after["model"] == ""
    assert after["base_url"] == ""
    assert after["api_key"] == ""
    # Not routing fields: user-tuned timeout and the shipped policy key survive the reset.
    assert after["timeout"] == 25
    assert after.get("task_chain_only") is True


# ── Docs content ───────────────────────────────────────────────────────────


def test_configuration_docs_document_the_voice_brief_surface():
    """The user guide covers the picker row, the block's keys and the Discord switch.

    Asserted by content because the docs-drift ratchet cannot see ``auxiliary.<task>.*``
    leaves: the ``auxiliary.<task>.*`` placeholder already covers them.
    """
    text = _CONFIGURATION_DOC.read_text(encoding="utf-8")

    for literal in (
        "Voice brief",
        "auxiliary.voice_brief.provider",
        "auxiliary.voice_brief.model",
        "auxiliary.voice_brief.fallback_chain",
        "auxiliary.voice_brief.task_chain_only",
        "voice_brief 15s",
        "discord.voice_brief.enabled",
    ):
        assert literal in text, f"configuration.md does not document {literal!r}"

    # What task_chain_only: true means, in the pinned case and on auto.
    assert "never falls back to the main agent's model" in text
    assert "inert" in text
