"""SPEC-0045 W1 (R1b) — flat-dump migration falsifiers.

Pre-D1 sessions hold their dump artifacts FLAT next to ``pipeline_queue.json``
(``<sid>/<dump_id>.jsonl`` + ``<sid>/<dump_id>.meta.json``); the consumers'
directory scans never see them. The one-time adoption moves each flat pair into
``<sid>/<dump_id>/`` and is idempotent (second run: zero moves, mtimes stable).
Deterministic: synthetic copies only — the REAL pre-fix flat dumps in the
operator's compaction store (an external lived path, never named here; set
``HERMES_COMPACTION_STORE`` to assert it went untouched) are never touched.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agent.compaction_dump import DumpStore
from agent.compaction_pipeline import IdlePipelinePass


def _alternating(n: int) -> list:
    return [{"role": "assistant" if i % 2 else "user", "content": f"m{i}"}
            for i in range(n)]


def _write_flat_dump(sdir: Path, dump_id: str, messages: list, window: tuple) -> None:
    """Write a pre-fix FLAT dump pair exactly the deployed pre-D1 producer left
    it: ``<dump_id>.jsonl`` + ``<dump_id>.meta.json`` as siblings of
    ``pipeline_queue.json``, with real meta fields (start/end_msg)."""
    sdir.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(m, ensure_ascii=False, default=str) + "\n" for m in messages)
    (sdir / f"{dump_id}.jsonl").write_text(payload, encoding="utf-8")
    (sdir / f"{dump_id}.meta.json").write_text(json.dumps({
        "session_id": sdir.name,
        "dump_id": dump_id,
        "start_msg": window[0],
        "end_msg": window[1],
        "message_count": len(messages),
        "token_estimate": 12345,
        "created": 1789421921.0,
        "schema_version": 1,
        "storage": "inline",
        "complete": True,
    }), encoding="utf-8")


# The three REAL flat dumps' shapes (SPEC-0044 §1 evidence), replayed as
# synthetic copies — window and count survive the migration.
REAL_FLAT_DUMPS = [
    ("20260914_191430_6eb2a4", "0001-63a95403", (8, 791), 784),
    ("20260914_191430_6eb2a4", "0001-7baf50ac", (8, 543), 536),
    ("20260915_024124_a8fa66", "0001-5c51cb94", (3, 410), 408),
]


class TestFlatDumpAdoption:
    def test_falsifier_flat_pair_adopted_and_discovered(self, tmp_path):
        """A real-shaped flat pair becomes a discoverable, complete dump dir."""
        messages = _alternating(6)
        _write_flat_dump(tmp_path / "sess", "0001-63a95403", messages, (0, 5))
        store = DumpStore(tmp_path)
        # Pre-state: invisible to the consumer's scan (the G1 dead-path shape).
        assert store.dump_dirs("sess") == []
        adopted = store.ensure_layout("sess")
        assert adopted == ["0001-63a95403"]
        ddir = tmp_path / "sess" / "0001-63a95403"
        assert (ddir / "0001-63a95403.jsonl").is_file()
        assert (ddir / "0001-63a95403.meta.json").is_file()
        # Nothing flat remains next to the queue file.
        assert not (tmp_path / "sess" / "0001-63a95403.jsonl").is_file()
        assert not (tmp_path / "sess" / "0001-63a95403.meta.json").is_file()
        # The region is a REAL dump: discovered, complete, byte-identical.
        assert store.dump_ids("sess") == ["0001-63a95403"]
        assert store.is_complete("sess", "0001-63a95403") is True
        assert store.round_trip_bytes("sess", "0001-63a95403") == \
            (ddir / "0001-63a95403.jsonl").read_bytes()
        meta = store.read_meta("sess", "0001-63a95403") or {}
        assert (int(meta["start_msg"]), int(meta["end_msg"])) == (0, 5)

    def test_falsifier_second_run_zero_moves_mtime_stable(self, tmp_path):
        """AC-R1b idempotency: reopen touches nothing."""
        messages = _alternating(6)
        _write_flat_dump(tmp_path / "sess", "0001-abcdef01", messages, (0, 5))
        store = DumpStore(tmp_path)
        assert store.ensure_layout("sess") == ["0001-abcdef01"]
        ddir = store.dump_dir("sess", "0001-abcdef01")
        mtimes = {p.name: p.stat().st_mtime_ns for p in sorted(ddir.iterdir())}
        assert set(mtimes) == {"0001-abcdef01.jsonl", "0001-abcdef01.meta.json"}
        # Second open: nothing flat left -> no-op, artifacts untouched.
        assert store.ensure_layout("sess") == []
        mtimes2 = {p.name: p.stat().st_mtime_ns for p in sorted(ddir.iterdir())}
        assert mtimes2 == mtimes, "idempotent reopen must not rewrite artifacts"

    def test_falsifier_three_real_shaped_flat_dumps_survive(self, tmp_path):
        """All three real flat dumps' shape (window from meta start/end_msg,
        message count) survives migration — synthetic copies, real data
        untouched. Falsifies the §1 probe on the migrated tree: every region
        dir exists, every meta is complete, windows read back exactly."""
        sessions = {}
        for sid, dump_id, window, count in REAL_FLAT_DUMPS:
            messages = _alternating(count)
            _write_flat_dump(tmp_path / sid, dump_id, messages, window)
            sessions.setdefault(sid, []).append((dump_id, window, count))
        store = DumpStore(tmp_path)
        # Pre-state: every flat dump invisible to the consumer's scan (G1).
        for sid in sessions:
            assert store.dump_dirs(sid) == [], "pre-state: flat dumps invisible"
        # Adopt (the real data has two dumps in one session — one call covers both).
        adopted_by_sid = {sid: store.ensure_layout(sid) for sid in sessions}
        assert adopted_by_sid == {
            sid: sorted(d for d, _w, _c in dumps) for sid, dumps in sessions.items()}
        for sid, dumps in sessions.items():
            for dump_id, window, count in dumps:
                meta = store.read_meta(sid, dump_id) or {}
                got = (int(meta["start_msg"]), int(meta["end_msg"]))
                assert got == window, f"window must survive migration: {got} != {window}"
                assert int(meta["message_count"]) == count
                assert store.is_complete(sid, dump_id) is True
                assert len(store.read_messages(sid, dump_id)) == count
        # The real store was never touched (path via env, never named in code).
        real = os.environ.get("HERMES_COMPACTION_STORE")
        if not real:
            pytest.skip("HERMES_COMPACTION_STORE not set: real-store guard not active")
        real = Path(real)
        for sid, dump_id, _w, _c in REAL_FLAT_DUMPS:
            assert (real / sid / f"{dump_id}.jsonl").is_file()
            assert not (real / sid / dump_id).is_dir(), \
                "migration must never touch the real store"

    def test_adoption_runs_at_the_pass_choke_point(self, tmp_path):
        """The idle pass adopts before its directory scans: a pass over a
        session holding flat artifacts discovers and works on the region."""
        messages = _alternating(6)
        _write_flat_dump(tmp_path / "sess", "0001-abcdef02", messages, (0, 5))
        a = SimpleNamespaceAgent(root=tmp_path)
        rec = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}")
        assert rec.get("adopted_dumps") == ["0001-abcdef02"], rec
        store = DumpStore(tmp_path)
        assert store.dump_ids("sess") == ["0001-abcdef02"]
        assert store.is_complete("sess", "0001-abcdef02") is True
        # Second pass: adoption is a no-op — telemetry absent, zero moves.
        mtime = store.dump_path("sess", "0001-abcdef02").stat().st_mtime_ns
        rec2 = IdlePipelinePass(a).run(messages, llm_call=lambda _p: "{}")
        assert "adopted_dumps" not in rec2
        assert store.dump_path("sess", "0001-abcdef02").stat().st_mtime_ns == mtime

    def test_adoption_failure_logs_and_leaves_pair(self, tmp_path, monkeypatch, caplog):
        """A failed move logs and leaves the flat pair in place — never a
        silent skip, never a deletion."""
        messages = _alternating(6)
        _write_flat_dump(tmp_path / "sess", "0001-abcdef03", messages, (0, 5))
        store = DumpStore(tmp_path)
        journal = tmp_path / "sess" / "0001-abcdef03.jsonl"
        meta = tmp_path / "sess" / "0001-abcdef03.meta.json"
        monkeypatch.setattr(store, "_move_preserving_mtime",
                            lambda src, dst: (_ for _ in ()).throw(OSError("disk gone")))
        with caplog.at_level("WARNING", logger="agent.compaction_dump"):
            adopted = store.ensure_layout("sess")
        assert adopted == [], "a failed move must not be reported adopted"
        assert journal.is_file() and meta.is_file(), "flat pair must survive"
        assert not (tmp_path / "sess" / "0001-abcdef03").is_dir() or \
            not any((tmp_path / "sess" / "0001-abcdef03").iterdir()), \
            "no adopted content may remain behind"
        assert any("adoption failed" in r.message for r in caplog.records)

    def test_session_without_flat_artifacts_is_untouched(self, tmp_path):
        """A session with only canonical (post-D1) dumps: zero cost, no moves,
        no directory churn."""
        store = DumpStore(tmp_path)
        ref = store.write_dump("sess", _alternating(4), start_msg=0, end_msg=3, turn=1)
        ddir = store.dump_dir("sess", ref.dump_id)
        before = {p.name: p.stat().st_mtime_ns for p in sorted(ddir.iterdir())}
        assert store.ensure_layout("sess") == []
        after = {p.name: p.stat().st_mtime_ns for p in sorted(ddir.iterdir())}
        assert after == before

    def test_missing_session_dir_is_a_noop(self, tmp_path):
        store = DumpStore(tmp_path)
        assert store.ensure_layout("no-such-session") == []

    def test_journal_without_meta_is_not_a_dump(self, tmp_path):
        """A stray .jsonl with no sidecar is never adopted or deleted."""
        stray = tmp_path / "sess" / "not-a-dump.jsonl"
        stray.parent.mkdir(parents=True)
        stray.write_text("{}\n", encoding="utf-8")
        store = DumpStore(tmp_path)
        assert store.ensure_layout("sess") == []
        assert stray.is_file(), "a journal without its meta is not data to move"


class SimpleNamespaceAgent:
    """Minimal idle-pass agent harness (mirrors test_compaction_producer)."""

    def __init__(self, root: Path, session_id: str = "sess"):
        from types import SimpleNamespace
        self._a = SimpleNamespace()
        a = self._a
        a.session_id = session_id
        a.db = None
        a.compaction_pipeline_enabled = True
        a.compaction_pipeline_storage_root = str(root)
        a.compaction_pipeline_map_idle_after_seconds = 0.0
        a.compaction_pipeline_map_cooldown_seconds = 0.0
        a.compaction_pipeline_extraction_cooldown_seconds = 0.0
        a.compaction_pipeline_max_stage_retries = 2
        a.compaction_pipeline_budget_per_session_tokens = 200000
        a.compaction_pipeline_max_wait_seconds = 900.0
        a.compaction_pipeline_gate_always_on = True
        a.compaction_pipeline_loss_probe_samples = 4
        a.compaction_pipeline_models = {}
        a._compaction_models_reachable = False  # keep the pass short: no drains
        a.aux_runtime = {"provider": "ollama"}
        a.provider = "ollama"
        a.model = "muse-glimmer:latest"
        a.context_compressor = SimpleNamespace(
            _compress_window=lambda msgs: None)  # no window -> no dump stage
        a._compaction_pipeline_spent_tokens = 0
        a._compaction_pipeline_last_pass_ts = 0.0
        a._compaction_pipeline_last_extract_ts = 0.0
        a._compaction_stage_llms = {}

    def __getattr__(self, name):
        return getattr(self._a, name)