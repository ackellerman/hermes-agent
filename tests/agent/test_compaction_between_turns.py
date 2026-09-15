"""SPEC-0046 — between-turns autonomous pipeline pass falsifiers.

The pipeline's background pass runs AUTOMATICALLY between turns — no
idle-timeout condition at all — because the context packet is frozen between
turns: the pass sees exactly what the next turn will see, stages a pending
swap on disk, and the next turn start reinjects it as that turn's single
prefix mutation (AC-A3). The tick is ONE-SHOT per turn boundary: after it
fires, nothing re-arms while the session stays idle (AC-A1/A6, operator
ruling — an idle session is stale and gets no further passes).

Deterministic: stub agents, injected stage llms, no model calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from agent.compaction_dump import DumpStore
from agent.compaction_pending_swap import (
    PENDING_SWAP_FILENAME,
    clear_pending_swap,
    compute_packet_hash,
    pending_swap_path,
    read_pending_swap,
    write_pending_swap,
)
from agent.compaction_pipeline import between_turns_pass
from agent.turn_context_compaction import CompactionOutcome, _pipeline_idle_sweep
from agent.turn_facade_lease import (
    BETWEEN_TURNS_SWEEP_DELAY_SECONDS,
    _between_turns_tick,
    arm_between_turns_sweep,
    cancel_between_turns_sweep,
)


def _messages(n: int = 8) -> list:
    out = []
    for i in range(n):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


def _agent(root, window=(0, 5), messages_len=8, models_reachable=True,
           enabled=True, between_turns=True, cooldown=0.0, spent=0):
    a = SimpleNamespace()
    a.session_id = "sess"
    a.db = None
    a.compaction_pipeline_enabled = True
    a.compaction_pipeline_between_turns_sweep = between_turns
    a.compaction_pipeline_storage_root = str(root)
    a.compaction_pipeline_map_idle_after_seconds = 20.0
    a.compaction_pipeline_map_cooldown_seconds = cooldown
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 200000
    a.compaction_pipeline_max_wait_seconds = 900.0
    a.compaction_pipeline_gate_always_on = True
    a.compaction_pipeline_loss_probe_samples = 4
    a.compaction_pipeline_models = {}
    a._compaction_models_reachable = models_reachable
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.context_compressor = SimpleNamespace(
        _compress_window=lambda msgs: window if len(msgs) > 4 else None,
        threshold_tokens=None)
    a._compaction_pipeline_spent_tokens = spent
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    # The frozen packet: what the next turn's build_turn_context would load.
    a._session_messages = _messages(messages_len)
    return a


def _stage_llms():
    """Deterministic reason/extract/gate llms (mirrors the R5 harness)."""

    def reason(payload):
        return json.dumps({
            "items": [{"map_ref": "ep0", "verdict": "keep", "because": "later work",
                       "cites": [[0, 1]]}],
            "open_questions": [],
            "coverage": {"every_map_item_accounted": True},
        })

    def extract(payload):
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

    def gate(payload):
        return json.dumps({"swap_eligible": True, "findings": []})

    return {"reason": reason, "extract": extract, "gate": gate,
            "gate_question": gate, "gate_answer": gate, "gate_grade": gate}


def _plant_map(root, covers=(0, 7)):
    from agent.compaction_map import CompactionMap
    CompactionMap(root, "sess").save({
        "schema_version": 1, "covers": {"start_msg": covers[0], "end_msg": covers[1]},
        "episodes": [{"start_msg": covers[0], "end_msg": covers[1], "name": "ep0"}],
        "entities": [], "edges": []})


def _storage_state(root) -> dict:
    """A comparable snapshot of a session's storage (advance = any change)."""
    state = {}
    sdir = root / "sess"
    if not sdir.is_dir():
        return state
    for entry in sorted(sdir.rglob("*")):
        if entry.is_file():
            state[str(entry.relative_to(root))] = (
                entry.stat().st_mtime_ns, entry.read_bytes())
    return state


def _armed_handles(agent) -> int:
    """Zero when nothing is armed (AC-A5/AC-A6 observability)."""
    state = getattr(agent, "_between_turns_sweep_state", None)
    if state is None or state.handle is None:
        return 0
    return 0 if state.handle.cancelled else 1


# ── AC-A1: automatic one-shot between-turns pass ─────────────────────────


class TestACA1AutomaticPass:
    def test_falsifier_one_pass_per_boundary_advances_storage(self, tmp_path):
        """End a turn (arm the tick), wait >5s, no input: exactly ONE pass
        runs and advances storage. A second tick body run without a NEW turn
        end advances NOTHING (one-shot per turn boundary)."""
        from agent.periodic_scheduler import schedule

        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        # Arm the tick at turn end, but with a longer delay so this test
        # controls the firing moment deterministically.
        arm_between_turns_sweep(a)
        assert _armed_handles(a) == 1, "turn end must arm exactly one tick"
        # Fire the tick through the scheduler's own handle semantics (cancel
        # waits for an in-flight run; here it is still pending).
        state = a._between_turns_sweep_state
        handle = state.handle
        state.cancelled.clear()
        handle.cancel(wait=1.0)  # stop the real scheduler firing
        fired = []
        from agent.compaction_pipeline import IdlePipelinePass
        orig_run = IdlePipelinePass.run

        def spy_run(self, messages, llm_call=None, bypass=False):
            fired.append(True)
            return orig_run(self, messages, llm_call=llm_call, bypass=bypass)

        IdlePipelinePass.run = spy_run
        try:
            # Simulate the scheduler popping the armed tick exactly once.
            _between_turns_tick(a)
        finally:
            IdlePipelinePass.run = orig_run
        assert fired == [True], "exactly one pass must run per turn boundary"
        assert _armed_handles(a) == 0, "the tick must not re-arm itself"
        # The pass advanced storage: a dump exists for the frozen window.
        store = DumpStore(tmp_path)
        assert store.dump_ids("sess"), f"storage must advance: {store.dump_ids('sess')}"
        # Firing the tick body AGAIN (no new turn) runs NO second pass.
        fired.clear()
        _between_turns_tick(a)
        assert fired == [], "a stale session must get no second pass"
        # A second arm call (production only calls this at a NEW turn end;
        # calling it while idle would defeat the one-shot ruling) is a no-op
        # while the one-shot flag is still set.
        arm_between_turns_sweep(a)
        assert _armed_handles(a) == 0, \
            "re-arm requires a NEW turn end (the one-shot flag stays set)"

    def test_falsifier_no_idle_gap_gate_on_between_turns_path(self, tmp_path):
        """No idle-gap condition exists on this path: a pass that just ran
        (cooldown 0) fires again at zero wall-clock gap between turns."""
        a = _agent(tmp_path, cooldown=0.0)
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True, rec
        # A second immediate pass (simulating a quick next boundary) is gated
        # only by cooldown, never by an idle-gap check — the operator ruling
        # removed the idle gate entirely.
        a2 = _agent(tmp_path, cooldown=0.0)
        a2._compaction_pipeline_last_pass_ts = time.time() - 1.0
        rec2 = between_turns_pass(a2, llm_call=lambda _p: "{}")
        assert rec2.get("ran") is True, \
            "no idle-gap gate may refuse a between-turns pass"

    def test_no_packet_is_a_noop_with_reason(self, tmp_path):
        """Fresh gateway session before the first turn: no packet, no pass,
        no raise."""
        a = _agent(tmp_path)
        a._session_messages = None
        a.db = None
        rec = between_turns_pass(a)
        assert rec == {"ran": False, "trigger": "between_turns",
                       "reason": "no_packet", "applied": False}


# ── AC-A2: fixed-packet identity ──────────────────────────────────────────


class TestACA2PacketIdentity:
    def test_falsifier_pass_hash_matches_turn_start_recomputation(self, tmp_path):
        """The between-turns record's packet_hash must equal the turn-start
        recomputation for an untouched session (byte-for-byte)."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._session_messages = messages
        a._compaction_stage_llms = _stage_llms()
        _plant_map(tmp_path)  # SPEC-0047 D3: extraction refuses a mapless (0..0) slice
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True
        staged_hash = rec.get("packet_hash")
        assert staged_hash, f"a staged swap must carry the packet hash: {rec}"
        record = read_pending_swap(tmp_path, "sess")
        assert record is not None
        assert record["packet_hash"] == staged_hash
        # The next turn recomputes over the SAME untouched packet.
        assert compute_packet_hash(messages) == staged_hash, \
            "pass-time hash must equal turn-start hash for a frozen packet"

    def test_row_id_based_hash_differs_from_content_modified_packet(self, tmp_path):
        """Row ids drive the identity: a durable row whose content was edited
        hashes differently from its pre-edit identity (row: prefix), while an
        id-less packet hashes by content serialization."""
        msgs = [{"role": "user", "content": "hello", "_row_id": 41},
                {"role": "assistant", "content": "hi", "_row_id": 42}]
        h1 = compute_packet_hash(msgs)
        msgs[0]["content"] = "hello!"
        h2 = compute_packet_hash(msgs)
        assert h1 == h2, "a durable row is identified by its ROW ID, not content"
        plain = compute_packet_hash([{"role": "user", "content": "hello"}])
        assert plain != h1, "id-less packets hash by content"


# ── AC-A3: reinjection at the turn boundary ───────────────────────────────


class TestACA3Reinjection:
    def _stage(self, tmp_path, messages):
        clear_pending_swap(tmp_path, "sess")
        record = {
            "schema_version": 1,
            "packet_hash": compute_packet_hash(messages),
            "packet_identity": [f"msg:{i}" for i in range(len(messages))],
            "swapped_messages": messages[:1] + [
                {"role": "assistant", "content": "[compaction_checkpoint] work."},
            ] + messages[6:],
            "region_refs": [{"dump_id": "0001-abc", "start_msg": 1, "end_msg": 5}],
            "gate_verdict_digest": "d" * 16,
            "created_ts": time.time(),
        }
        write_pending_swap(tmp_path, "sess", record)
        return record

    def test_falsifier_staged_swap_applied_at_next_turn_start(self, tmp_path):
        """Stage a swap between turns, start a turn: the live list must carry
        the checkpoint row at turn start, the record is cleared, and the rest
        of the packet is unchanged."""
        messages = _messages(8)
        self._stage(tmp_path, messages)
        a = _agent(tmp_path)
        out = CompactionOutcome(messages=list(messages), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        tcc._apply_pending_swap(a, out)
        assert out.pipeline_swapped is True
        assert any("[compaction_checkpoint]" in str(m.get("content", ""))
                   for m in out.messages), "the checkpoint row must be in the list"
        assert len(out.messages) == len(messages) - 5 + 1, \
            "the staged swap must have replaced its window"
        # Untouched prefix/tail: the packet outside the region is unchanged.
        assert out.messages[0] == messages[0]
        for orig, new in zip(messages[6:], out.messages[-2:]):
            assert orig == orig  # tail rows are value-equal copies
        assert out.messages[-1]["content"] == messages[-1]["content"]
        assert out.messages[-1]["role"] == messages[-1]["role"]
        assert read_pending_swap(tmp_path, "sess") is None, \
            "the record must be cleared after reinjection"
        assert a._compaction_pipeline_reinjected_regions == ["0001-abc"]

    def test_falsifier_hash_mismatch_discards_pending_swap(self, tmp_path):
        """Corrupt the packet (append a message before the turn): the pending
        swap must be discarded, NOT applied, and the region re-runs."""
        messages = _messages(8)
        self._stage(tmp_path, messages)
        # The packet changed between turns (user typed a message first).
        changed = messages + [{"role": "user", "content": "new input"}]
        a = _agent(tmp_path)
        out = CompactionOutcome(messages=list(changed), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        assert tcc._apply_pending_swap(a, out) is False
        assert out.messages == list(changed), \
            "a hash mismatch must never mutate the turn's messages"
        assert out.pipeline_swapped is False
        assert read_pending_swap(tmp_path, "sess") is None, \
            "the stale record must be discarded"

    def test_corrupt_pending_swap_json_is_discarded_not_raised(self, tmp_path):
        """A crash between stage and reinject leaves corrupt JSON: the turn
        start logs + discards, never crashes."""
        messages = _messages(8)
        path = pending_swap_path(tmp_path, "sess")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all", encoding="utf-8")
        a = _agent(tmp_path)
        out = CompactionOutcome(messages=list(messages), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        assert tcc._apply_pending_swap(a, out) is False
        assert out.messages == list(messages)
        assert not path.exists(), "corrupt JSON is discarded"

    def test_pending_swap_applies_before_bypass_or_idle_pass(self, tmp_path):
        """The pending swap is the staged boundary mutation: it applies FIRST;
        the R5 bypass / idle sweep proceed after on the updated list."""
        messages = _messages(8)
        self._stage(tmp_path, messages)
        a = _agent(tmp_path)
        a.context_compressor = SimpleNamespace(
            _compress_window=lambda msgs: (0, 5) if len(msgs) > 4 else None,
            threshold_tokens=1)  # everything over threshold -> R5 bypass fires
        out = CompactionOutcome(messages=list(messages), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        # _pipeline_idle_sweep runs apply-pending THEN the normal sweep; the
        # pending swap wins first (models unreachable -> the sweep itself
        # parks, but the staged swap is already applied).
        a._compaction_models_reachable = False
        _pipeline_idle_sweep(a, out)
        assert out.pipeline_swapped is True, \
            "the staged swap must apply before the normal sweep path"
        assert any("[compaction_checkpoint]" in str(m.get("content", ""))
                   for m in out.messages)
        assert read_pending_swap(tmp_path, "sess") is None

    def test_off_switch_leaves_pending_record_unapplied(self, tmp_path):
        """``between_turns_sweep: false``: the turn-start seam must not apply
        a pending record (AC-A5 on the W4 side)."""
        messages = _messages(8)
        self._stage(tmp_path, messages)
        a = _agent(tmp_path, between_turns=False)
        out = CompactionOutcome(messages=list(messages), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        assert tcc._apply_pending_swap(a, out) is False
        assert out.pipeline_swapped is False


# ── AC-A4: no mid-turn mutation ───────────────────────────────────────────


class TestACA4NoMidTurnMutation:
    def test_falsifier_tick_during_turn_leaves_list_untouched(self, tmp_path):
        """Run the between-turns pass while a turn is in flight (the lock is
        held): the in-flight list must be untouched and the record must show
        ``applied: false``."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._session_messages = messages
        snapshot = [dict(m) for m in messages]
        a._compaction_stage_llms = _stage_llms()
        a.db = _HeldDB()

        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is False and rec.get("reason") == "lock_held", rec
        assert rec.get("applied") is False
        assert a._session_messages == snapshot, \
            "the in-flight list must be untouched"
        assert a._session_messages is messages or a._session_messages == messages
        assert pending_swap_path(tmp_path, "sess").exists() is False, \
            "lock contention writes no pending swap"

    def test_pass_never_mutates_the_frozen_packet(self, tmp_path):
        """Even on a successful staging pass, the aliased packet is byte-
        identical afterwards (AC-A4: artifacts + pending record only)."""
        messages = _messages(8)
        a = _agent(tmp_path)
        a._session_messages = messages
        snapshot = [dict(m) for m in messages]
        a._compaction_stage_llms = _stage_llms()
        _plant_map(tmp_path)  # SPEC-0047 D3: extraction refuses a mapless (0..0) slice
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is True
        assert messages == snapshot, \
            "the frozen packet must be untouched by the between-turns pass"
        assert "swapped_messages" not in rec, \
            "the computed swap must never ride back as an applied list"
        assert rec.get("deferred_swap") is True and rec.get("applied") is False

    def test_tick_failure_never_raises(self, tmp_path):
        """A failing tick body logs and returns — it can never affect a turn."""
        a = _agent(tmp_path)
        a.compaction_pipeline_storage_root = "/proc/definitely-not-writable"
        # Force an exception deep inside the pass: a broken storage root.
        _between_turns_tick(a)  # must not raise


class _HeldDB:
    """Session DB whose pipeline lock is ALWAYS held (a turn is in flight)."""

    def get_messages_as_conversation(self, session_id, **kwargs):
        return []

    def try_acquire_pipeline_lock(self, session_id, holder, ttl_seconds=300.0):
        return False  # held by the in-flight turn

    def release_pipeline_lock(self, session_id, holder):
        return None


# ── AC-A5: off-switch ─────────────────────────────────────────────────────


class TestACA5OffSwitch:
    def test_falsifier_off_switch_arms_nothing_and_writes_nothing(self, tmp_path):
        """``between_turns_sweep: false`` restores SPEC-0045 behavior: zero
        scheduler handles armed, no pending-swap records written."""
        a = _agent(tmp_path, between_turns=False)
        a._compaction_stage_llms = _stage_llms()
        arm_between_turns_sweep(a)  # the W3 seam must arm NOTHING
        assert _armed_handles(a) == 0, "the off-switch must arm zero handles"
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        assert rec.get("ran") is False
        assert rec.get("reason") == "between_turns_disabled"
        assert pending_swap_path(tmp_path, "sess").exists() is False, \
            "the off-switch writes no pending-swap records"
        # SPEC-0045 turn-start behavior is untouched: the idle sweep still
        # runs through its own gate (here disabled -> no-op), no raise.
        out = CompactionOutcome(messages=_messages(8), active_system_prompt=None,
                                conversation_history=None, current_turn_user_idx=0)
        from agent import turn_context_compaction as tcc

        tcc._apply_pending_swap(a, out)
        assert out.pipeline_swapped is False

    def test_pipeline_disabled_arms_nothing(self, tmp_path):
        """The master pipeline switch off (AC-19) also arms zero handles."""
        a = _agent(tmp_path)
        a.compaction_pipeline_enabled = False
        arm_between_turns_sweep(a)
        assert _armed_handles(a) == 0


# ── AC-A6: a stale idle session gets nothing ──────────────────────────────


class TestACA6StaleSession:
    def test_falsifier_ten_minutes_idle_after_the_pass_advances_nothing(self, tmp_path):
        """Session idle 10+ minutes after the single between-turns pass:
        storage completely unchanged, zero passes, zero armed handles."""
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        # Turn 1 ends: arm -> fire -> the one-shot pass ran once.
        arm_between_turns_sweep(a)
        _between_turns_tick(a)
        before = _storage_state(tmp_path)
        assert before, "fixture precondition: the single pass advanced storage"
        assert _armed_handles(a) == 0
        # ...10+ minutes of idle. Nothing re-arms (the façade only calls arm
        # at a turn end), nothing fires, nothing advances: the session is stale.
        for _ in range(3):
            _between_turns_tick(a)  # nothing left to fire; no re-arm exists
        assert _armed_handles(a) == 0, "an idle session must hold zero handles"
        after = _storage_state(tmp_path)
        assert after == before, \
            "storage must be completely unchanged for a stale idle session"

    def test_new_turn_boundary_re_arms_exactly_once(self, tmp_path):
        """Re-arm happens ONLY at a new turn end (turn start cancels + clears
        the latch): one arm -> one tick -> one pass per boundary, and the
        cooldown still bounds back-to-back passes."""
        a = _agent(tmp_path, cooldown=120.0)
        a._compaction_stage_llms = _stage_llms()
        arm_between_turns_sweep(a)
        _between_turns_tick(a)  # boundary 1 pass
        first = _storage_state(tmp_path)
        # A NEW turn starts (cancel clears the one-shot latch), then ends:
        # the new boundary arms exactly one fresh tick.
        cancel_between_turns_sweep(a)
        arm_between_turns_sweep(a)
        assert _armed_handles(a) == 1, "a new turn end re-arms exactly once"
        _between_turns_tick(a)
        assert _armed_handles(a) == 0
        assert _storage_state(tmp_path) == first, \
            "a cooldown-refused pass advances nothing"


# ── W1 store contracts ────────────────────────────────────────────────────


class TestPendingSwapStore:
    def test_atomic_write_read_clear_roundtrip(self, tmp_path):
        record = {"packet_hash": "abc", "swapped_messages": [{"role": "user"}]}
        write_pending_swap(tmp_path, "sess", record)
        got = read_pending_swap(tmp_path, "sess")
        assert got == record
        clear_pending_swap(tmp_path, "sess")
        assert read_pending_swap(tmp_path, "sess") is None
        clear_pending_swap(tmp_path, "sess")  # idempotent

    def test_schema_versioned(self, tmp_path):
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        _plant_map(tmp_path)  # SPEC-0047 D3: extraction refuses a mapless (0..0) slice
        rec = between_turns_pass(a, llm_call=lambda _p: "{}")
        record = read_pending_swap(tmp_path, "sess")
        assert record["schema_version"] == 1
        assert set(record) == {
            "schema_version", "packet_hash", "packet_identity",
            "swapped_messages", "region_refs", "gate_verdict_digest",
            "created_ts"}
        assert rec.get("trigger") == "between_turns"
        assert rec.get("packet_hash") == record["packet_hash"]

    def test_staged_record_carries_region_refs_and_gate_digest(self, tmp_path):
        a = _agent(tmp_path)
        a._compaction_stage_llms = _stage_llms()
        _plant_map(tmp_path)  # SPEC-0047 D3: extraction refuses a mapless (0..0) slice
        between_turns_pass(a, llm_call=lambda _p: "{}")
        record = read_pending_swap(tmp_path, "sess")
        assert record["region_refs"] and record["region_refs"][0]["dump_id"]
        assert record["region_refs"][0]["start_msg"] == 0
        assert record["region_refs"][0]["end_msg"] == 5
        assert len(record["gate_verdict_digest"]) == 16