"""SPEC-0042 rehydration tests — AC-16/17/18 falsifiers."""

import json

import pytest

from agent.compaction_dump import DumpNotFoundError, DumpStore
from agent.compaction_rehydrate import Rehydrator, StubRegistry


def _write_dump(tmp_path, n=6):
    store = DumpStore(tmp_path)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)]
    ref = store.write_dump("sess", msgs, start_msg=0, end_msg=n - 1)
    return store, ref, msgs


class TestAC16UnknownId:
    def test_falsifier_unknown_id_machine_readable_error(self, tmp_path):
        """FALSIFIER AC-16 verbatim: unknown id -> must raise with a
        machine-readable error, never silent emptiness."""
        store, ref, msgs = _write_dump(tmp_path)
        rh = Rehydrator(store)
        with pytest.raises(DumpNotFoundError) as excinfo:
            rh.read_dump("no-such-dump")
        payload = json.loads(str(excinfo.value))
        assert payload["error"] == "dump_not_found"
        assert payload["dump_id"] == "no-such-dump"

    def test_valid_id_returns_verbatim_range(self, tmp_path):
        store, ref, msgs = _write_dump(tmp_path)
        rh = Rehydrator(store)
        result = rh.read_dump(ref.dump_id)
        assert result["source"] == "dump"
        assert result["messages"] == msgs


class TestAC17RestartRecovery:
    def test_falsifier_fresh_process_same_content(self, tmp_path):
        """FALSIFIER AC-17: after restart (fresh store/rehydrator objects over
        the same storage root), rehydration returns the same content."""
        store, ref, msgs = _write_dump(tmp_path)
        first = Rehydrator(store).read_dump(ref.dump_id)
        # Fresh process equivalent: brand-new store + rehydrator instances.
        second = Rehydrator(DumpStore(tmp_path)).read_dump(ref.dump_id)
        assert first["messages"] == second["messages"] == msgs


class TestAC18TranscriptFallback:
    def test_falsifier_dump_deleted_transcript_serves_and_logs(self, tmp_path):
        """FALSIFIER AC-18: with the dump deleted but transcript present,
        read_dump serves from the transcript fallback and logs
        source: transcript-fallback."""
        store, ref, msgs = _write_dump(tmp_path)
        transcript = list(msgs)

        def transcript_reader(session_id, start, end):
            if end is None:
                return transcript[start:]
            return transcript[start:end + 1]

        rh = Rehydrator(store, transcript_reader=transcript_reader)
        # delete the dump artifact
        dpath = store.dump_path("sess", ref.dump_id)
        dpath.unlink()
        result = rh.read_dump(ref.dump_id, session_id="sess")
        assert result["source"] == "transcript-fallback"
        assert result["messages"] == msgs
        assert rh.fallbacks and rh.fallbacks[0].served_from == "transcript-fallback"

    def test_no_fallback_no_transcript_still_errors(self, tmp_path):
        store, ref, msgs = _write_dump(tmp_path)
        store.dump_path("sess", ref.dump_id).unlink()
        rh = Rehydrator(store, transcript_reader=None)
        with pytest.raises(DumpNotFoundError):
            rh.read_dump(ref.dump_id, session_id="sess")


class TestStubRegistry:
    def test_register_and_resolve(self):
        reg = StubRegistry()
        reg.register("d-1", 3, 8, "deploy auth decision")
        assert reg.get("d-1")["summary"] == "deploy auth decision"
        assert reg.get("missing") is None