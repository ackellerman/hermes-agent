"""SPEC-0045 W2 (R5) — threshold idle-bypass falsifiers.

A continuously-prompted session never idles, so the idle gate starves the
pipeline and every turn takes the prose backstop. When the rough token estimate
of the live messages is at/over the LIVE compression trigger
(``compressor.threshold_tokens`` — no new knob), ``gates_pass`` waives the
idle-gap check ONLY: cooldown, budget, lock, and enabled checks are NOT waived,
and ``record["bypass"] = True`` marks the pass for monitoring.

Deterministic: stub agents, no model calls.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.compaction_pipeline import IdlePipelinePass, idle_pipeline_sweep


def _messages(n_chars: int) -> list:
    """Alternating messages whose rough estimate crosses a 1000-token
    threshold at ~13 messages (4 chars/token)."""
    out = []
    for i in range(n_chars // 10):
        out.append({"role": "assistant" if i % 2 else "user", "content": "x" * 10})
    return out


def _agent(root, threshold=None, enabled=True, cooldown=0.0, budget=200000,
           spent=0):
    a = SimpleNamespace()
    a.session_id = "sess"
    a.db = None
    a.compaction_pipeline_enabled = enabled
    a.compaction_pipeline_storage_root = str(root)
    a.compaction_pipeline_map_idle_after_seconds = 20.0
    a.compaction_pipeline_map_cooldown_seconds = cooldown
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = budget
    a.compaction_pipeline_max_wait_seconds = 900.0
    a.compaction_pipeline_gate_always_on = True
    a.compaction_pipeline_loss_probe_samples = 4
    a.compaction_pipeline_models = {}
    a._compaction_models_reachable = False
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    # Stub compressor: window calculation only, like the producer tests. When
    # ``threshold`` is given it rides the LIVE value (R5: no new knob).
    a.context_compressor = SimpleNamespace(_compress_window=lambda msgs: None,
                                           threshold_tokens=threshold)
    a._compaction_pipeline_spent_tokens = spent
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    return a


class TestThresholdBypass:
    def test_falsifier_over_threshold_runs_at_zero_gap_under_does_not(self, tmp_path):
        """AC-R5: two agents, identical messages, idle_gap_seconds=0. The one
        whose rough estimate is >= threshold runs a pass; the one under the
        threshold does not (the idle gate still binds)."""
        messages = _messages(10000)  # ~2500 tokens: over the 1000 threshold
        over = IdlePipelinePass(_agent(tmp_path, threshold=1000))
        assert over.gates_pass(0.0, messages=messages) is True, \
            "est >= threshold must waive the idle gate"
        under = IdlePipelinePass(_agent(tmp_path, threshold=10_000_000))
        assert under.gates_pass(0.0, messages=messages) is False, \
            "est < threshold must keep the idle gate closed"
        # End-to-end through the seam entry: only the over-threshold agent ran.
        rec_over = idle_pipeline_sweep(_agent(tmp_path, threshold=1000),
                                       messages, 0.0)
        assert rec_over.get("ran") is True
        assert rec_over.get("bypass") is True, "the waiver must be recorded"
        rec_under = idle_pipeline_sweep(_agent(tmp_path, threshold=10_000_000),
                                        messages, 0.0)
        assert rec_under == {"ran": False, "reason": "gates"}

    def test_idle_pass_is_not_marked_bypass(self, tmp_path):
        """A genuinely idle pass (gap >= idle_after) is NOT a bypass, even when
        the estimate is over threshold — the telemetry distinguishes the two."""
        messages = _messages(10000)
        rec = idle_pipeline_sweep(_agent(tmp_path, threshold=1000), messages, 30.0)
        assert rec.get("ran") is True
        assert "bypass" not in rec, "an idle-gap pass is not a threshold bypass"

    def test_stub_compressor_without_threshold_gets_no_bypass(self, tmp_path):
        """Evals/producer-test stubs are SimpleNamespace compressors without
        ``threshold_tokens``: no bypass, never a raise."""
        messages = _messages(10000)
        a = _agent(tmp_path, threshold=None)
        a.context_compressor = SimpleNamespace(_compress_window=lambda m: None)
        assert IdlePipelinePass(a).gates_pass(0.0, messages=messages) is False

    def test_threshold_bypass_at_exact_boundary(self, tmp_path):
        """est == threshold bypasses (>=, per the operator ruling)."""
        # 1 message of 4000 'x' -> exactly 1000 rough tokens.
        msgs = [{"role": "user", "content": "x" * 4000}]
        assert IdlePipelinePass(_agent(tmp_path, threshold=1000)) \
            .gates_pass(0.0, messages=msgs) is True

    def test_cooldown_still_blocks_a_bypass_pass(self, tmp_path):
        """The waiver is the idle gate ONLY: a live pass cooldown still refuses."""
        import time
        messages = _messages(10000)
        a = _agent(tmp_path, threshold=1000, cooldown=120.0)
        a._compaction_pipeline_last_pass_ts = time.time()
        assert IdlePipelinePass(a).gates_pass(0.0, messages=messages) is False

    def test_budget_exhausted_still_blocks_a_bypass_pass(self, tmp_path):
        """Budget exhaustion is not waived by the threshold bypass."""
        messages = _messages(10000)
        a = _agent(tmp_path, threshold=1000, budget=100, spent=200)
        assert IdlePipelinePass(a).gates_pass(0.0, messages=messages) is False

    def test_disabled_still_blocks_a_bypass_pass(self, tmp_path):
        """AC-19: the bypass lives behind the enabled check."""
        messages = _messages(10000)
        a = _agent(tmp_path, threshold=1000, enabled=False)
        assert IdlePipelinePass(a).gates_pass(0.0, messages=messages) is False