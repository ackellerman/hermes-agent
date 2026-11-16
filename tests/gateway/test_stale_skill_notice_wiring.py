"""Gateway wiring: a continuing session whose loaded skill changed on disk carries the stale-skill
notice in the model-facing user message (api_content sidecar), never in the stored transcript.

Harness follows tests/gateway/test_auto_skill_title_input.py (real ``_hmwa_prepare_turn`` → real
``AIAgent``, stopped before the network), with the sidecar arming that ``_run_sync_scoped`` performs
(``arm_surface_turn(agent, runner._consume_pending_turn_sidecar_notes(...))``).
"""

import os
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.turn_context import TurnContext
from run_agent import AIAgent

NOTICE_HEAD = "[Skill changed on disk"
AUTO_HEADER = '[IMPORTANT: The "{}" skill is auto-loaded. Follow its instructions for this session.]'


class _TurnObserved(BaseException):
    """Stop after the real prologue, before any main-model network request."""


def _stale_alpha_history():
    """History holding a REAL auto-load copy of ``alpha``; ``alpha/SKILL.md`` is then edited."""
    from agent.skill_commands import _build_skill_message, _load_skill_payload
    skill_dir = Path(os.environ["HERMES_HOME"]) / "skills" / "alpha"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\nname: alpha\ndescription: test skill\n---\n\n# alpha\n\nStep one.\nStep two.\n",
                        encoding="utf-8")
    payload, loaded_dir, display = _load_skill_payload("alpha", task_id="discord:quick-key")
    row = _build_skill_message(payload, loaded_dir, AUTO_HEADER.format(display)) + "\n\nstart the job"
    skill_md.write_text(skill_md.read_text(encoding="utf-8").replace("Step two.", "Step two (edited)."),
                        encoding="utf-8")
    return [{"role": "user", "content": row}, {"role": "assistant", "content": "On it."}]


async def _run_turn(tmp_path, monkeypatch, history, opened):
    question = "What is the next step?"
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel", user_id="user", user_name="Example")
    event = MessageEvent(text=question, source=source)
    entry = SessionEntry(session_id="stale-skill", session_key="discord:stale-skill",
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._get_proxy_url = lambda: None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, entry, entry.session_key))
    runner._hmwa_open_session = AsyncMock(return_value=opened)
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, load_transcript=AsyncMock(return_value=list(history)))
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=list(history))
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._consume_pending_native_image_paths = lambda key: []
    runner._adapter_for_source = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None
    staged = []
    real_stage = runner._set_pending_turn_sidecar_notes

    def stage(session_key, notes):
        staged.extend(notes)
        real_stage(session_key, notes)

    runner._set_pending_turn_sidecar_notes = stage

    async def propagate_error(exc, *args):
        raise exc
    runner._hmwa_agent_error_reply = propagate_error

    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    agent = AIAgent(session_db=db, model="test-model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                    platform="discord", session_id=entry.session_id, enabled_toolsets=[],
                    quiet_mode=True, skip_memory=True, skip_context_files=True)
    agent.compression_enabled = False
    model_inputs = []

    def observe_model(agent, messages):
        model_inputs.append(messages[-1].get("api_content") or messages[-1]["content"])
        raise _TurnObserved

    monkeypatch.setattr("agent.conversation_loop.begin_iteration", observe_model)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)

    defaults = TurnContext()
    display = SimpleNamespace(**{name: getattr(defaults, name) for name in runner._DISPLAY_TO_TURN_CTX})
    display.platform_key = "discord"
    display.resolve_display_setting = lambda *args: False
    runner._run_agent_display_settings = lambda source: display

    def run_without_delivery(ctx, worker, *args):
        # _run_sync_scoped → _wire_turn_agent_callbacks arms the staged sidecar notes; do the same.
        from gateway.session_surface import arm_surface_turn
        arm_surface_turn(agent, runner._consume_pending_turn_sidecar_notes(ctx.session_key))
        worker._native_image_run_message = lambda: ctx.message
        worker._run_conversation_with_approval(agent, [], None, ctx.persist_user_message, ctx.persist_user_timestamp)

    runner._run_agent_bind_turn_wiring = run_without_delivery
    runner._run_agent = runner._run_agent_inner
    try:
        with pytest.raises(_TurnObserved):
            await runner._handle_message_with_agent(event, source, entry.session_key, 1)
        stored = [row["content"] for row in db.get_messages(entry.session_id) if row["role"] == "user"]
    finally:
        agent.close()
        db.close()
    assert model_inputs and question in model_inputs[0]
    return model_inputs[0], stored, staged


# W1
@pytest.mark.asyncio
async def test_continuing_session_model_input_carries_the_notice(tmp_path, monkeypatch):
    model_input, stored, _staged = await _run_turn(tmp_path, monkeypatch, _stale_alpha_history(), (False, False))
    assert NOTICE_HEAD in model_input
    assert 'skill_view(name="alpha")' in model_input
    assert stored and all(NOTICE_HEAD not in row for row in stored)  # F10: never in the transcript


# W2
@pytest.mark.asyncio
async def test_new_session_gets_no_notice(tmp_path, monkeypatch):
    model_input, _stored, staged = await _run_turn(tmp_path, monkeypatch, _stale_alpha_history(), (False, True))
    assert NOTICE_HEAD not in model_input
    assert not [note for note in staged if NOTICE_HEAD in note]


# W3
@pytest.mark.asyncio
async def test_failing_detector_never_blocks_the_turn(tmp_path, monkeypatch):
    def boom(history):
        raise RuntimeError("detector exploded")

    monkeypatch.setitem(sys.modules, "agent.skill_staleness",
                        types.SimpleNamespace(stale_skill_notice=boom, stale_skill_names=boom))
    model_input, _stored, staged = await _run_turn(tmp_path, monkeypatch, _stale_alpha_history(), (False, False))
    assert NOTICE_HEAD not in model_input
    assert not [note for note in staged if NOTICE_HEAD in note]
