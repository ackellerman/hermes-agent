"""SPEC-0048 W5 — AC-C1/C2/C4 falsifiers: pipeline LLM-call hardening.

AC-C1 (D-A): with a mock stage lane that never responds (simulating the F-A
stall — the ollama-cloud lane held a 1.09 MB payload open for 30+ minutes with
zero bytes), ``between_turns_pass`` returns WITHIN read-timeout + one retry,
logs the failure, releases the lock, and a SECOND pass acquires it
immediately. Pre-fix: hangs forever, the tick thread blocks, the lock is held
for the process lifetime and compaction silently stops.

AC-C2 (D-A/F-C): a lane returning EMPTY content twice parks the region with
the breaker and the error names "empty stage output (transport)" — never
"not JSON". One empty then one good response proceeds normally.

AC-C4 (D-C): the pass record carries ``duration_s``; a pass exceeding
``compaction_pipeline.slow_pass_warn_s`` logs a WARNING naming the stage in
flight, and ``record["slow_pass"]`` is True.

Note on real-lane timeouts (operator correction): the ollama-cloud lane took
688.2s on the real 1.09 MB stage_b payload and COMPLETED VALIDLY — the 900s
default read timeout must not kill slow-but-alive calls, so these tests inject
a small ``call_timeout_s`` and drive the stall through the injected-lane
transport wrapper instead of sleeping real timeout seconds.

Deterministic: stub agents, injected stage llms, no model calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_map import CompactionMap
from agent.compaction_pipeline import (
    DEFAULT_STAGE_CALL_TIMEOUT_S,
    StageTransportError,
    between_turns_pass,
)
from agent.compaction_extract import STAGE_B_PROMPT, parse_stage_json


def _messages(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


def _agent(root, *, call_timeout_s=DEFAULT_STAGE_CALL_TIMEOUT_S,
           slow_pass_warn_s=300.0, models_reachable=True):
    a = SimpleNamespace()
    a.session_id = "sess"
    a.db = None
    a.compaction_pipeline_enabled = True
    a.compaction_pipeline_between_turns_sweep = True
    a.compaction_pipeline_storage_root = str(root)
    a.compaction_pipeline_map_idle_after_seconds = 20.0
    a.compaction_pipeline_map_cooldown_seconds = 0.0
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 200000
    a.compaction_pipeline_max_wait_seconds = 900.0
    a.compaction_pipeline_gate_always_on = True
    a.compaction_pipeline_loss_probe_samples = 4
    a.compaction_pipeline_models = {}
    a.compaction_pipeline_slow_pass_warn_s = slow_pass_warn_s
    a._compaction_models_reachable = models_reachable
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.context_compressor = SimpleNamespace(
        _compress_window=lambda msgs: (0, 5) if len(msgs) > 4 else None,
        threshold_tokens=None)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    a._session_messages = _messages(8)
    return a


def _stage_llms(reason=None, extract=None, gate=None):
    """Deterministic good lanes (mirrors the SPEC-0046 harness)."""

    def _reason(payload):
        if reason is not None:
            return reason(payload)
        return json.dumps({
            "items": [{"map_ref": "ep0", "verdict": "keep", "because": "later work",
                       "cites": [[0, 1]]}],
            "open_questions": [],
            "coverage": {"every_map_item_accounted": True},
        })

    def _extract(payload):
        if extract is not None:
            return extract(payload)
        ckpt = {
            "instructions_and_corrections": "null_reason: none in region",
            "decisions": [{"what": "chose jsonl", "cites": [["dump", 1, 2]],
                           "rejected_alternatives": []}],
            "insights": "null_reason: none in region",
            "commitments": [{"what": "ship friday", "cites": [["dump", 2, 3]]}],
            "open_threads": "null_reason: none in region",
            "artifacts": "null_reason: none in region",
            "world_effects": "null_reason: none in region",
            "links": "null_reason: none in region",
            "narrative": "work.",
            "confidence": 0.9,
            "coverage": {"complete": True},
        }
        return json.dumps(ckpt)

    def _gate(payload):
        if gate is not None:
            return gate(payload)
        return json.dumps({"swap_eligible": True, "findings": []})

    return {"reason": _reason, "extract": _extract, "gate": _gate,
            "gate_question": _gate, "gate_answer": _gate, "gate_grade": _gate}


def _plant_map(root, covers=(0, 7)):
    CompactionMap(root, "sess").save({
        "schema_version": 1, "covers": {"start_msg": covers[0], "end_msg": covers[1]},
        "episodes": [{"start_msg": covers[0], "end_msg": covers[1], "name": "ep0"}],
        "entities": [], "edges": []})


# ── AC-C1: a stalled lane is bounded — the pass returns, lock released ────


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Real SessionDB (same fixture shape as test_compaction_pipeline_lock):
    the AC-C1 lock-release proof needs a genuine lock row."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    _db = SessionDB(db_path=tmp_path / "state.db")
    yield _db
    _db.close()


def _timeout_error():
    """The exception the openai SDK raises when the read timeout elapses."""
    import httpx
    import openai as _openai
    return _openai.APITimeoutError(
        httpx.Request("POST", "https://lane.invalid/v1/chat/completions"))


class TestACC1StalledLaneBounded:
    def test_falsifier_never_responding_lane_returns_and_releases_lock(
            self, tmp_path, db, caplog):
        """AC-C1 verbatim: a mock lane that never responds — it blocks, then
        the read timeout unwinds it exactly the way the real SDK aborts a
        zero-byte stall (F-A: ollama-cloud held a 1.09 MB payload open with
        zero bytes; pre-fix the tick thread blocked FOREVER with the lock
        held). Post-fix: the pass returns within read-timeout + one retry,
        logs the failure, releases the lock, and a SECOND pass acquires it
        immediately."""
        import threading

        a = _agent(tmp_path)
        a.db = db
        _plant_map(tmp_path)
        # The stall: a lane that blocks like a lane with zero bytes received.
        # The "read timeout" is the deadline event: 0.5s, then the SDK-shaped
        # timeout exception unwinds the call (what timeout=(30, 900) does on
        # the aux path).
        deadline = threading.Event()

        def stalled(payload):
            deadline.wait(30)  # would block forever without the deadline
            raise _timeout_error()

        llms = _stage_llms()
        llms["reason"] = stalled
        a._compaction_stage_llms = llms
        threading.Timer(0.5, deadline.set).start()

        t0 = time.time()
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        elapsed = time.time() - t0

        # The pass RETURNED (pre-fix: hangs forever with the lock held).
        assert rec.get("ran") is True, rec  # the pass unwinds, never raises
        assert elapsed < 30, f"the stalled pass must return promptly: {elapsed:.1f}s"
        # The failure is logged, classified as transport (not "not JSON").
        warnings = [r.message for r in caplog.records
                    if r.levelno >= logging.WARNING]
        joined = " | ".join(warnings)
        assert "timed out (transport)" in joined, \
            f"the stall must be logged as a transport failure: {joined}"
        assert "not JSON" not in joined and "not an object" not in joined, joined
        # The region parked through the normal ladder with the transport error.
        assert any("extraction failed" in w or "parked" in w for w in warnings), joined
        # Lock released: a SECOND pass acquires immediately and runs.
        a._compaction_stage_llms = _stage_llms()
        rec2 = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec2.get("ran") is True, \
            f"the lock must be free: the second pass acquires and runs: {rec2}"
        assert rec2.get("reason") != "lock_held", rec2

    def test_aux_lane_read_timeout_bounds_the_call_and_retries_once(self, caplog):
        """The aux-path faithful model: the client HONORS the timeout kwarg —
        create() would block forever, but the read timeout aborts it. Per D-A
        a stalled call costs AT MOST one read timeout (the single in-process
        retry belongs to EMPTY content, AC-C2), then StageTransportError
        unwinds the call so the pass parks instead of hanging the tick."""
        import time as _time

        from agent.compaction_pipeline import _call_stage_llm

        calls = {"n": 0}
        seen_timeouts = []

        class _ZeroByteLane:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls["n"] += 1
                        seen_timeouts.append(kwargs.get("timeout"))
                        _time.sleep(float(kwargs["timeout"][1]))  # honor it
                        raise _timeout_error()

        t0 = time.time()
        with pytest.raises(StageTransportError) as excinfo:
            _call_stage_llm("reason", _ZeroByteLane(), "m",
                            [{"content": "x"}], read_timeout=0.05)
        elapsed = time.time() - t0
        # Bounded: one read timeout, not a forever-block.
        assert elapsed < 10, f"{elapsed:.1f}s"
        assert calls["n"] == 1, \
            f"a timeout costs at most ONE read timeout (no retry): {calls}"
        assert seen_timeouts == [(30, 0.05)], seen_timeouts
        assert "timed out (transport)" in str(excinfo.value)

    def test_injected_lane_timeout_maps_to_transport_error(self, tmp_path):
        """The injected-lane path maps the SDK timeout exception through the
        same transport discipline (the falsifier lane is the test double for
        the real lane)."""
        from agent.compaction_pipeline import IdlePipelinePass

        a = _agent(tmp_path)
        sweep = IdlePipelinePass(a)
        lane = sweep._tracked_stage_llm("reason", lambda p: _raise_timeout())
        with pytest.raises(StageTransportError) as excinfo:
            lane([{"content": "x"}])
        assert "timed out (transport)" in str(excinfo.value)

    def test_default_timeout_is_900_and_configurable(self):
        """The 688.2s ollama-cloud stage_b call completed VALIDLY: the default
        read timeout must be 900s (not 600s), and
        compaction_pipeline.models.call_timeout_s overrides it."""
        assert DEFAULT_STAGE_CALL_TIMEOUT_S == 900


def _raise_timeout():
    raise _timeout_error()


# ── AC-C2: empty content retried once, then parked with its OWN error ─────


class TestACC2EmptyContentTransport:
    def test_falsifier_empty_twice_parks_naming_transport_not_not_json(self, tmp_path, caplog):
        """AC-C2 verbatim: a lane returning "" TWICE parks the region; the
        park/error names "empty stage output (transport)" — NOT "not JSON".
        Pre-fix: the empty string flowed into JSON parsing and the region
        parked with "stage output is not JSON: ''", burning a breaker rung on
        a transient (F-C)."""
        a = _agent(tmp_path)
        _plant_map(tmp_path)
        llms = _stage_llms()
        llms["reason"] = lambda payload: ""  # empty on EVERY call
        a._compaction_stage_llms = llms

        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        warnings = [r.message for r in caplog.records
                    if r.levelno >= logging.WARNING]
        joined = " | ".join(warnings)
        assert "empty stage output (transport)" in joined, \
            f"the transport error string must surface in the log: {joined}"
        assert "not JSON" not in joined and "not an object" not in joined, \
            f"the misleading parse error must NOT appear: {joined}"
        # The region parked: telemetry says so (nothing gate-passed).
        assert rec.get("ran") is True
        assert not rec.get("swapped_messages"), \
            f"an empty-lane pass must not stage a swap: {rec}"

    def test_park_marker_carries_the_transport_error_string(self, tmp_path):
        """The durable park marker (the breaker) records the transport error
        string, so the operator sees 'empty stage output (transport)' in
        extraction.parked.json — the observable of a transient-class failure."""
        a = _agent(tmp_path)
        _plant_map(tmp_path)
        llms = _stage_llms()
        llms["reason"] = lambda payload: ""
        a._compaction_stage_llms = llms
        between_turns_pass(a, llm_call=lambda _p: "{}")
        store = DumpStore(Path(tmp_path))
        parks = [(d, d / "extraction.parked.json")
                 for d in store.dump_dirs("sess")]
        assert parks, "an empty-lane pass must dump + attempt extraction"
        parked = [p for _, p in parks if p.is_file()]
        assert parked, "the empty-content failure must park the region"
        marker = json.loads(parked[0].read_text(encoding="utf-8"))
        assert "empty stage output (transport)" in str(marker.get("reason", "")), marker

    def test_empty_then_good_proceeds(self, tmp_path):
        """One empty response then one good one: the in-process transport
        retry absorbs the transient and the extraction PROCEEDS — no park, no
        breaker rung burned (F-C: 16/16 successes minutes later)."""
        a = _agent(tmp_path)
        _plant_map(tmp_path)
        llms = _stage_llms()
        calls = {"n": 0}

        def flaky_reason(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                return ""  # the transient content=None shape
            return _stage_llms()["reason"](payload)

        llms["reason"] = flaky_reason
        a._compaction_stage_llms = llms
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True
        assert calls["n"] >= 2, "the empty call must be retried once in-process"
        assert rec.get("extracted") or rec.get("queued_drained") or \
            rec.get("bypass_extracted"), \
            f"empty-then-good must proceed to a real extraction: {rec}"

    def test_transport_error_is_its_own_class(self):
        """StageTransportError exists and is a RuntimeError — the pass's
        except ladder can name it distinctly from ValueError parse errors."""
        exc = StageTransportError("empty stage output (transport)")
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "empty stage output (transport)"

    def test_parse_never_sees_the_empty_string_as_not_json(self, tmp_path):
        """The empty content must be classified at the TRANSPORT layer before
        any JSON parsing: parse_stage_json('') raises 'not JSON' only if a
        future regression leaks it through — document the boundary."""
        with pytest.raises(ValueError) as excinfo:
            parse_stage_json("")
        assert "not JSON" in str(excinfo.value)
        # ...while the transport wrapper catches '' first:
        from agent.compaction_pipeline import _stage_output_or_transport
        with pytest.raises(StageTransportError) as excinfo2:
            _stage_output_or_transport("reason", lambda p: "", [])
        assert "empty stage output (transport)" in str(excinfo2.value)


# ── AC-C4: duration telemetry + slow-pass warning ─────────────────────────


class TestACC4DurationTelemetry:
    def test_falsifier_record_carries_duration_s(self, tmp_path):
        """AC-C4: every ran pass record carries ``duration_s``."""
        a = _agent(tmp_path)
        _plant_map(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True
        assert isinstance(rec.get("duration_s"), (int, float)), \
            f"duration_s must ride on the record: {rec}"
        assert rec["duration_s"] >= 0

    def test_falsifier_slow_pass_warns_with_stage_name(self, tmp_path, caplog):
        """A pass exceeding slow_pass_warn_s logs a WARNING naming the stage
        in flight and marks ``slow_pass: true`` on the record."""
        a = _agent(tmp_path, slow_pass_warn_s=0.0)  # everything is "slow"
        _plant_map(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True
        assert rec.get("slow_pass") is True, \
            f"slow_pass must be marked when over threshold: {rec}"
        warnings = [r.message for r in caplog.records
                    if r.levelno >= logging.WARNING]
        assert any("slow compaction pass" in w for w in warnings), \
            f"a slow pass must WARN: {warnings}"
        named = [w for w in warnings
                 if "slow compaction pass" in w
                 and ("stage" in w or "reason" in w or "gate" in w
                      or "extract" in w or "map_update" in w
                      or "None" in w)]
        assert named, f"the warning must name a stage: {warnings}"

    def test_fast_pass_does_not_warn(self, tmp_path, caplog):
        """A fast pass under threshold: no slow_pass flag, no warning."""
        a = _agent(tmp_path, slow_pass_warn_s=300.0)
        _plant_map(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("slow_pass") is None, rec
        assert not any("slow compaction pass" in r.message
                       for r in caplog.records if r.levelno >= logging.WARNING)


# ── D-D: lock-stall watchdog ──────────────────────────────────────────────


class TestDDLockStallWatchdog:
    def test_lock_held_beyond_4x_threshold_warns_naming_holder(self, tmp_path, caplog):
        """A pass holding the lock beyond slow_pass_warn_s * 4 logs a WARNING
        naming the holder — diagnostic only, no behavior change."""
        from agent.compaction_pipeline import IdlePipelinePass, _check_lock_stall

        a = _agent(tmp_path, slow_pass_warn_s=0.0)
        sweep = IdlePipelinePass(a)
        sweep._lock_acquired_at = time.time() - 10.0  # held "forever"
        sweep._current_stage = "reason"
        _check_lock_stall(sweep)
        warnings = [r.message for r in caplog.records
                    if r.levelno >= logging.WARNING]
        assert any("pipeline lock stall" in w and "pipeline:sess" in w
                   for w in warnings), f"lock stall must warn: {warnings}"

    def test_fresh_lock_does_not_warn(self, tmp_path, caplog):
        from agent.compaction_pipeline import IdlePipelinePass, _check_lock_stall

        a = _agent(tmp_path)
        sweep = IdlePipelinePass(a)
        sweep._lock_acquired_at = time.time()  # just acquired
        _check_lock_stall(sweep)
        assert not [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "lock stall" in r.message]

    def test_watchdog_checked_at_stage_boundaries(self, tmp_path, monkeypatch):
        """Each stage call boundary runs the watchdog check (cheap, no new
        thread): the tracked lane stamps the stage then checks."""
        import agent.compaction_pipeline as cp

        a = _agent(tmp_path, slow_pass_warn_s=300.0)
        sweep = cp.IdlePipelinePass(a)
        called = []
        real = cp._check_lock_stall
        monkeypatch.setattr(cp, "_check_lock_stall",
                            lambda s: (called.append(True), real(s)))
        tracked = sweep._tracked_stage_llm("reason", lambda p: "{}")
        assert tracked([{"content": "x"}]) == "{}"
        assert called, "the stage boundary must run the lock-stall check"
        assert sweep._current_stage == "reason"


# ── config plumbing ───────────────────────────────────────────────────────


class TestConfig:
    def test_slow_pass_warn_s_default_300(self, tmp_path):
        a = _agent(tmp_path)
        assert a.compaction_pipeline_slow_pass_warn_s == 300.0

    def test_call_timeout_s_default_900(self):
        from agent.compaction_pipeline import DEFAULT_STAGE_CALL_TIMEOUT_S
        assert DEFAULT_STAGE_CALL_TIMEOUT_S == 900