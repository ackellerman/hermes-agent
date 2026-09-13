"""SPEC-0042 pipeline-lock tests — AC-14 falsifier: (1) IDENTITY — two distinct
tables; (2) CONTENTION — both directions, hold-one/attempt-other."""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


class TestAC14Identity:
    def test_falsifier_two_distinct_tables_in_schema(self, db):
        """IDENTITY: a sqlite_master query naming both compression_locks and
        compaction_pipeline_locks — same table name fails."""
        with sqlite3.connect(db.db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('compression_locks','compaction_pipeline_locks')"
            ).fetchall()
        names = {r[0] for r in rows}
        assert names == {"compression_locks", "compaction_pipeline_locks"}, (
            "both tables must exist as DISTINCT tables"
        )

    def test_falsifier_locks_are_distinct_rows_not_aliased(self, db):
        """Holding one lock must leave the other acquirable on the same session
        if (and only if) they are genuinely separate; and the row sources differ."""
        assert db.try_acquire_compression_lock("sess", "compressor-a")
        # A different subsystem can hold the pipeline lock concurrently as a ROW —
        # the mutual exclusion is enforced by the callers respecting each other
        # (AC-14 contention tests), while the LOCKS themselves are distinct rows.
        assert db.try_acquire_pipeline_lock("sess", "pipeline-a")
        # Identity: the two locks read from two distinct tables.
        with sqlite3.connect(db.db_path) as conn:
            c = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id='sess'").fetchall()
            p = conn.execute(
                "SELECT holder FROM compaction_pipeline_locks WHERE session_id='sess'").fetchall()
        assert c == [("compressor-a",)] and p == [("pipeline-a",)]
        db.release_pipeline_lock("sess", "pipeline-a")
        db.release_compression_lock("sess", "compressor-a")

    def test_acquire_release_roundtrip(self, db):
        assert db.try_acquire_pipeline_lock("sess2", "p1")
        assert not db.try_acquire_pipeline_lock("sess2", "p2")  # held
        db.release_pipeline_lock("sess2", "p1")
        assert db.try_acquire_pipeline_lock("sess2", "p2")  # reacquired after release


class TestAC14Contention:
    """Both directions: one pass must refuse (wait or degrade) when the other
    holds ITS lock. The refusal is at the caller (pipeline runner / backstop):
    before running, each checks the OTHER subsystem's live lock."""

    def _compression_lock_held(self, db, session_id: str) -> bool:
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ? AND expires_at > ?", (session_id, _now())
            ).fetchone()
        return row is not None

    def _pipeline_lock_held(self, db, session_id: str) -> bool:
        with sqlite3.connect(db.db_path) as conn:
            row = conn.execute(
                "SELECT holder, expires_at FROM compaction_pipeline_locks "
                "WHERE session_id = ? AND expires_at > ?", (session_id, _now())
            ).fetchone()
        return row is not None

    def test_falsifier_hold_compression_lease_pipeline_pass_must_not_proceed(self, db):
        """(a) hold the compression lease -> pipeline pass must wait or degrade."""
        assert db.try_acquire_compression_lock("sess-x", "compressor")
        # Pipeline pass gate: refuses to proceed while the compression lease is live.
        assert self._compression_lock_held(db, "sess-x")
        # And the gate is what production consults (imported from the swap module).
        from agent.compaction_swap import pipeline_pass_blocked
        assert pipeline_pass_blocked(db, "sess-x") is True

    def test_falsifier_hold_pipeline_lock_manual_compress_must_not_proceed(self, db):
        """(b) hold the pipeline lock -> manual /compress must wait or degrade."""
        assert db.try_acquire_pipeline_lock("sess-y", "pipeline")
        assert self._pipeline_lock_held(db, "sess-y")
        from agent.compaction_swap import compression_pass_blocked
        assert compression_pass_blocked(db, "sess-y") is True

    def test_unblocked_when_no_locks_held(self, db):
        from agent.compaction_swap import pipeline_pass_blocked, compression_pass_blocked
        assert pipeline_pass_blocked(db, "sess-free") is False
        assert compression_pass_blocked(db, "sess-free") is False


def _now():
    import time
    return time.time()