"""SPEC-0042 pipeline tests — stage acceptance per the spec's own FALSIFIERs.

Stage 1 (AC-1/2): dump store. Each stage's tests are appended in the spec's §5
build order; run with ``scripts/run_tests.sh tests/agent/test_compaction_pipeline.py``.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from agent.compaction_dump import (
    DumpIncompleteError,
    DumpNotFoundError,
    DumpStore,
    region_hash8,
    serialize_message,
)


def _mk_messages(n: int) -> list:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} " + "x" * (i % 17)}
        for i in range(n)
]


def _region_bytes(messages: list) -> bytes:
    return b"".join(serialize_message(m).encode("utf-8") + b"\n" for m in messages)


class TestAC1DumpRoundTrip:
    def test_500_message_region_round_trips_byte_identical(self, tmp_path):
        store = DumpStore(tmp_path)
        messages = _mk_messages(500)
        ref = store.write_dump("sess-a", messages, start_msg=0, end_msg=499)
        assert store.round_trip_bytes("sess-a", ref.dump_id) == _region_bytes(messages)

    def test_incomplete_dump_is_never_referenced_by_swap(self, tmp_path):
        """AC-1 second clause: complete is never true unless fsync completed — the
        swap must refuse (exit non-zero) when fsync did not complete."""
        store = DumpStore(tmp_path)
        messages = _mk_messages(10)
        ref = store.write_dump("sess-b", messages, start_msg=0, end_msg=9)
        # Forge an incomplete dump: flip complete back off on disk (crash-equivalent).
        mpath = store.meta_path("sess-b", ref.dump_id)
        meta = json.loads(mpath.read_text())
        meta["complete"] = False
        mpath.write_text(json.dumps(meta))
        with pytest.raises(DumpIncompleteError):
            store.require_complete("sess-b", ref.dump_id)
        with pytest.raises(DumpIncompleteError):
            store.read_messages("sess-b", ref.dump_id)

    def test_falsifier_kill9_between_serialize_and_meta_rename(self, tmp_path):
        """FALSIFIER AC-1 verbatim: kill -9 the process between serialize and
        meta-rename -> next boot sees complete:false and the swap refuses ->
        must refuse, exit non-zero in the recovery check."""
        crasher = textwrap.dedent(
            """
            import os, signal, sys
            import agent.compaction_dump as cd

            class KilledMidWrite(cd.DumpStore):
                def _atomic_write_json(self, path, obj):
                    # First call = the incomplete meta tombstone; proceed.
                    # Second call = the post-fsync complete flip: die before it runs.
                    if getattr(self, "_calls", 0) == 1:
                        os.kill(os.getpid(), signal.SIGKILL)
                    self._calls = getattr(self, "_calls", 0) + 1
                    return super()._atomic_write_json(path, obj)

            store = KilledMidWrite(sys.argv[1])
            store.write_dump("sess-kill", [{"role": "user", "content": "hi"}] * 20,
                             start_msg=0, end_msg=19)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", crasher, str(tmp_path)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        assert proc.returncode == -signal.SIGKILL, "child must die mid-write, not exit cleanly"

        # Next boot: recovery check on a fresh store instance must refuse, exit non-zero.
        recovery = textwrap.dedent(
            """
            import sys
            from agent.compaction_dump import DumpStore, DumpIncompleteError
            store = DumpStore(sys.argv[1])
            try:
                store.require_complete("sess-kill", sys.argv[2])
            except (DumpIncompleteError, Exception):
                sys.exit(3)
            sys.exit(0)  # a swap that proceeds on an incomplete dump must NOT get here
            """
        )
        sdir = tmp_path / "sess-kill"
        dump_ids = [p.name[: -len(".jsonl")] for p in sdir.glob("*.jsonl")]
        assert dump_ids, "crash must leave the content + tombstone meta behind"
        r2 = subprocess.run(
            [sys.executable, "-c", recovery, str(tmp_path), dump_ids[0]],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        assert r2.returncode != 0, "recovery check must exit non-zero on an incomplete dump"


class TestAC2IdempotentMarking:
    def test_two_dumps_same_region_identical_region_hash8(self, tmp_path):
        store = DumpStore(tmp_path)
        messages = _mk_messages(120)
        r1 = store.write_dump("sess-c", messages, start_msg=0, end_msg=119, turn=7)
        r2 = store.write_dump("sess-c", messages, start_msg=0, end_msg=119, turn=7)
        assert r1.dump_id == r2.dump_id, "same region must produce identical dump_id"

    def test_falsifier_dump_twice_compare_ids(self, tmp_path):
        messages = _mk_messages(50)
        assert region_hash8(messages) == region_hash8(list(messages))
        different = [dict(m, content=m["content"] + " altered") for m in messages]
        assert region_hash8(messages) != region_hash8(different)


class TestAC16PreviewErrorShape:
    def test_unknown_dump_id_raises_machine_readable_error(self, tmp_path):
        store = DumpStore(tmp_path)
        with pytest.raises(DumpNotFoundError) as excinfo:
            store.read_messages("nope", "missing-id")
        payload = json.loads(str(excinfo.value))
        assert payload["error"] == "dump_not_found"
        assert payload["dump_id"] == "missing-id"