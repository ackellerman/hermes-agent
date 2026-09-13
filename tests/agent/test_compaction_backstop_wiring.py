"""SPEC-0042 review finding F4: backstop_gate / swap_region had ZERO production
callers — the ON path (AC-19b byte-identity) was unreachable in production.

This locks in the class fix: ``agent/compaction_backstop.py`` now calls both
functions on the production overflow seam, and ``conversation_compression.
_run_summary_dispatch`` consults it before the legacy summary runs. Tests:

- ``agent.compaction_backstop`` / ``agent.conversation_compression`` reference
  backstop_gate and swap_region (grep-provable production callers).
- AC-19b byte-identity: with the pipeline enabled but extraction not ready, the
  ``_run_summary_dispatch`` seam produces byte-identical messages to
  ``enabled: false`` (both run the legacy single-call path against the same
  input) and stamps NO new telemetry attribute.
- The swap path is genuinely reachable: with a complete dump + gate-passed
  checkpoint present, the backstop returns a swapped message list.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent.compaction_dump import DumpStore
from agent.conversation_compression import _run_summary_dispatch
from agent.compaction_backstop import (
    TELEMETRY_DEGRADATION_REASON,
    TELEMETRY_DEGRADED,
    CompactionBackstop,
)


class _FakeAgent:
    """Minimal agent carrying the pipeline config attrs the seam reads."""

    def __init__(self, enabled: bool, storage_root: Path, session_id: str = "sess"):
        self.compaction_pipeline_enabled = enabled
        self.compaction_pipeline_storage_root = str(storage_root)
        self.compaction_pipeline_max_wait_seconds = 1
        self.session_id = session_id
        self.db = None
        self._compaction_models_reachable = None
        self.aux_runtime = {"provider": "ollama"}
        self.provider = "ollama"
        self.model = "muse-glimmer:latest"
        self.context_compressor = None  # unused when commit_fence is None


def _alternating_messages(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


# ── production-caller reachability ────────────────────────────────────


def test_backstop_module_calls_gate_and_swap_in_production():
    src = Path("agent/compaction_backstop.py").read_text()
    assert "backstop_gate(" in src, "backstop_gate must have a production caller"
    assert "swap_region(" in src, "swap_region must have a production caller"


def test_conversation_compression_wires_the_backstop_seam():
    src = Path("agent/conversation_compression.py").read_text()
    assert "maybe_backstop_swap" in src, "the overflow dispatch must consult the backstop"


# ── AC-19b byte-identity at the dispatch seam ─────────────────────────


def _stamp_legacy_compress(spy_calls):
    def compress_fn(messages, **_kw):
        spy_calls.append(messages)
        return list(messages)  # the legacy path returns the same messages
    return compress_fn


def test_ac19b_on_degrade_is_byte_identical_to_off(tmp_path):
    messages = _alternating_messages()
    off_calls, on_calls = [], []

    # enabled:false -> legacy_summary, identical input, telemetry stamped.
    off_agent = _FakeAgent(False, tmp_path / "off")
    off_out = _run_summary_dispatch(
        off_agent, messages, _stamp_legacy_compress(off_calls), {}, commit_fence=None,
        attempt_generation=0, hard_cancel_event=None)

    # enabled:true, no ready checkpoint -> degrade, same legacy path, identical bytes.
    on_agent = _FakeAgent(True, tmp_path / "on")
    on_out = _run_summary_dispatch(
        on_agent, messages, _stamp_legacy_compress(on_calls), {}, commit_fence=None,
        attempt_generation=0, hard_cancel_event=None)

    assert off_out == on_out, "ON-degrade output must byte-match OFF output (AC-19b)"
    assert [m for m in off_calls] == [m for m in on_calls], "input must be identical"
    # The legacy path ran in BOTH cases (degrade falls through to it).
    assert len(off_calls) == 1 and len(on_calls) == 1

    off_tel = off_agent._compaction_backstop_telemetry
    on_tel = on_agent._compaction_backstop_telemetry
    # AC-19b: the ON-degrade run introduces NO new telemetry attribute vs OFF.
    assert set(on_tel.keys()) == set(off_tel.keys())
    assert set(off_tel.keys()) == {TELEMETRY_DEGRADED, TELEMETRY_DEGRADATION_REASON}
    assert off_tel[TELEMETRY_DEGRADED] is False and off_tel[TELEMETRY_DEGRADATION_REASON] is None
    assert on_tel[TELEMETRY_DEGRADED] is True  # degraded recorded
    assert on_tel[TELEMETRY_DEGRADATION_REASON] == "extraction_incomplete"


# ── the swap path is reachable (a real swap_region production call) ────


def test_backstop_reaches_swap_when_a_ready_checkpoint_exists(tmp_path):
    messages = _alternating_messages(8)
    agent = _FakeAgent(True, tmp_path)
    root = tmp_path

    # Dump the region messages[4..7] (complete) under root/sess/<dump_id>.
    store = DumpStore(root)
    region = messages[4:8]
    ref = store.write_dump(agent.session_id, region, start_msg=4, end_msg=7, turn=0)
    dump_id = ref.dump_id
    sid = agent.session_id
    # Extraction + gate artifacts the seam scans for readiness.
    ddir = root / sid / dump_id
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "stage_c.json").write_text(json.dumps({
        "decisions": [{"what": "Postgres", "cites": [[dump_id, 5, 6]]}],
        "commitments": "null_reason: none",
        "artifacts": "null_reason: none",
        "world_effects": "null_reason: none",
        "insights": "null_reason: none",
        "open_threads": "null_reason: none",
        "links": "null_reason: none",
        "instructions_and_corrections": "null_reason: none",
        "narrative": "N.", "confidence": 0.9, "coverage": {"complete": True},
    }))
    (ddir / "gate.json").write_text(json.dumps({"swap_eligible": True, "findings": []}))

    action, swapped, telemetry = CompactionBackstop(agent).decide_and_swap(messages)

    assert action == "swap"
    assert swapped is not None and swapped != messages
    assert any(str(m.get("content", "")).startswith("[compaction_checkpoint]")
               for m in swapped), "the swapped list must contain the checkpoint row"
    assert telemetry[TELEMETRY_DEGRADED] is False


def test_backstop_degrades_when_no_ready_region(tmp_path):
    agent = _FakeAgent(True, tmp_path)
    messages = _alternating_messages(8)
    action, swapped, telemetry = CompactionBackstop(agent).decide_and_swap(messages)
    assert action == "degrade"
    assert swapped is None
    assert telemetry[TELEMETRY_DEGRADED] is True
    assert telemetry[TELEMETRY_DEGRADATION_REASON] == "extraction_incomplete"


def test_backstop_disabled_touches_nothing(tmp_path):
    """AC-19: enabled:false never touches pipeline machinery."""
    agent = _FakeAgent(False, tmp_path)
    action, swapped, telemetry = CompactionBackstop(agent).decide_and_swap(_alternating_messages())
    assert action == "legacy_summary"
    assert swapped is None
    assert telemetry[TELEMETRY_DEGRADED] is False