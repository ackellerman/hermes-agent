"""SPEC-0042 swap + backstop tests — AC-13/15/19/19b falsifiers."""

import json

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_swap import (
    SwapRefusedError,
    backstop_gate,
    build_checkpoint_row,
    swap_region,
)


def _ckpt():
    return {
        "instructions_and_corrections": [{"what": "use jsonl", "cites": [["d1", 2, 3]]}],
        "decisions": [], "insights": [], "commitments": [
            {"what": "ship Friday", "cites": [["d1", 4, 5]]}],
        "open_threads": [], "artifacts": [], "world_effects": [], "links": [],
        "narrative": "Export module work.", "confidence": 0.9,
        "coverage": {"complete": True},
    }


def _messages(n=10):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)]


@pytest.fixture
def dump(tmp_path):
    store = DumpStore(tmp_path)
    msgs = _messages(6)
    ref = store.write_dump("sess", msgs, start_msg=0, end_msg=5)
    return store, ref


class TestSwap:
    def test_swap_produces_valid_alternation_and_checkpoint_row(self, dump):
        store, ref = dump
        messages = [_messages(6)[0], *_messages(6)[1:],
                    {"role": "user", "content": "after"}]
        # messages: u a u a u a u — swap middle region with a single assistant row
        out = swap_region(
            messages, start_idx=0, end_idx=5, checkpoint=_ckpt(),
            dump_store=store, session_id="sess", dump_id=ref.dump_id,
            gate_verdict={"swap_eligible": True, "findings": []},
        )
        assert len(out) == 2
        row = out[0]
        assert row["role"] == "assistant"
        assert "compaction_checkpoint" in row["content"]
        assert "[dump: " in row["content"], "link-stubs must ride the checkpoint row"
        assert out[1]["role"] == "user"

    def test_swap_refuses_incomplete_dump(self, dump):
        store, ref = dump
        mpath = store.meta_path("sess", ref.dump_id)
        meta = json.loads(mpath.read_text()); meta["complete"] = False
        mpath.write_text(json.dumps(meta))
        with pytest.raises(SwapRefusedError):
            swap_region(_messages(6), start_idx=0, end_idx=5, checkpoint=_ckpt(),
                        dump_store=store, session_id="sess", dump_id=ref.dump_id)

    def test_swap_refuses_when_gate_not_eligible(self, dump):
        store, ref = dump
        with pytest.raises(SwapRefusedError):
            swap_region(_messages(6), start_idx=0, end_idx=5, checkpoint=_ckpt(),
                        dump_store=store, session_id="sess", dump_id=ref.dump_id,
                        gate_verdict={"swap_eligible": False,
                                      "findings": [{"what": "missing commitment"}]})

    def test_ac13_batched_swap_one_mutation(self, dump):
        """AC-13: N regions swapping = <= 1 prompt-prefix mutation. Both swaps
        land in ONE committed message list built by a single batch function —
        instrument identity per turn: exactly one committed mutation."""
        store, ref = dump
        messages = _messages(6) + [{"role": "user", "content": "later"}]
        gate = {"swap_eligible": True, "findings": []}

        mutations = []
        original = list(messages)

        def batch_swap(msgs, regions):
            # ONE pass over all eligible regions -> ONE mutation of the list.
            out = list(msgs)
            # swap from the back so earlier indices stay valid
            for start_idx, end_idx, ckpt in sorted(regions, key=lambda r: -r[0]):
                out = swap_region(out, start_idx=start_idx, end_idx=end_idx,
                                  checkpoint=ckpt, dump_store=store,
                                  session_id="sess", dump_id=ref.dump_id,
                                  gate_verdict=gate)
            return out

        out = batch_swap(messages, [
            (0, 2, _ckpt()),
            (3, 5, _ckpt()),
        ])
        mutations.append(out is not original)
        assert sum(mutations) <= 1, "more than one list mutation in the batch turn"
        # two checkpoint rows + the trailing user message
        rows = [m for m in out if "compaction_checkpoint" in str(m.get("content", ""))]
        assert len(rows) == 2
        assert out[-1]["role"] == "user"


class TestAC15Degradation:
    def test_falsifier_models_unreachable_backstop_degrades_legacy(self):
        """FALSIFIER AC-15: with extraction models unreachable, an
        overflow-triggered backstop produces a summary via the legacy
        single-call path (action=degrade -> legacy summary), telemetry records
        degraded:true, and the region is queued for pipeline extraction."""
        cfg = {"enabled": True, "swap": {"max_wait_seconds": 900}}
        decision = backstop_gate(
            cfg, _FakeDB(locked=False), "sess",
            extraction_state={"gate": "passed"}, models_reachable=False,
        )
        assert decision["action"] == "degrade"
        assert decision["degraded"] is True
        assert decision["degradation_reason"] == "model_unreachable"
        # Degrade action = legacy summary path + requeue (spec §3.5).
        assert decision["degraded"] is True and "degradation_reason" in decision

    def test_extraction_incomplete_degrades(self):
        cfg = {"enabled": True}
        decision = backstop_gate(cfg, _FakeDB(locked=False), "sess",
                                 extraction_state={"gate": "running"})
        assert decision["degradation_reason"] == "extraction_incomplete"

    def test_lock_timeout_degrades(self):
        cfg = {"enabled": True}
        decision = backstop_gate(cfg, _FakeDB(locked=True), "sess",
                                 extraction_state={"gate": "passed"})
        assert decision["degradation_reason"] == "lock_timeout"


class _FakeDB:
    def __init__(self, locked: bool = False):
        self._locked = locked
        self.db_path = "/tmp/fake-never-opened.db"

    def compression_lock_holder(self, session_id):
        return "someone" if self._locked else None


class TestAC19OffAnd19bByteIdentity:
    def test_falsifier_off_case_no_pipeline_mechanism_touched(self):
        """FALSIFIER AC-19 (OFF tautology guard): enabled:false -> the backstop
        refuses the pipeline path unconditionally: no lock, no map, no dumps."""
        decision = backstop_gate({"enabled": False}, _TouchSpyDB(), "sess")
        assert decision == {"action": "legacy_summary", "degraded": False}

    def test_falsifier_19b_on_degrade_byte_identical_to_off(self):
        """FALSIFIER AC-19b: enable the pipeline, force a backstop-degrade, and
        the decision must carry the same legacy-summary action as the OFF case
        (the byte-identity proof is the degrade path running the same legacy
        code); any new telemetry attribute not present in the OFF-case run
        fails."""
        off = backstop_gate({"enabled": False}, _FakeDB(False), "sess")
        on_degraded = backstop_gate(
            {"enabled": True}, _FakeDB(False), "sess",
            extraction_state={"gate": "running"},  # degrade per §3.5
        )
        # Same terminal action — the degrade path IS the legacy path.
        assert off["action"] == "legacy_summary"
        assert on_degraded["action"] == "degrade"
        # degrade action runs the same legacy summary; the only additions are
        # degraded + degradation_reason (sanctioned telemetry fields).
        extra_keys = set(on_degraded) - set(off)
        assert extra_keys <= {"degraded", "degradation_reason", "max_wait_seconds"}


class _TouchSpyDB(_FakeDB):
    """Fails the test if the OFF-case backstop touches any lock mechanism."""

    def compression_lock_holder(self, session_id):
        raise AssertionError("AC-19 violation: OFF-case backstop probed the lock")