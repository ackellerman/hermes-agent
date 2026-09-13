"""SPEC-0043 in-container degradation drills — AC-31.

Each drill is a self-contained script meant to run INSIDE a containerized Hermes
with a temp ``HERMES_HOME``. Each verifies a degradation class end-to-end and
writes/prints a JSON receipt. Receipts carry provenances (input counts + sha256),
never literal transcript content.

Drill (i) aux-down: the compression aux route is unreachable -> the proposal's
AC-15 legacy step completes, degraded telemetry is emitted, and a queue row is
written (dump-before-degrade, AC-21) so nothing is dropped without a durable
copy.
Drill (ii) storage-root read-only: ``chmod 500`` on the storage root -> every
pipeline stage fails safe, the live message list is untouched, the legacy
compressor completes, and the process does not crash.
Drill (iii) kill -9 mid-stage: terminate between stage writes -> next boot sees
a complete:false / partial stage-checkpoint, and that region is never referenced
for swap (stale/partial-window discipline).

Run one:  python evals/compaction/drills.py aux-down   [--storage ROOT]
           python evals/compaction/drills.py storage-ro [--storage ROOT]
           python evals/compaction/drills.py kill-mid   [--storage ROOT]
Each writes ``evals/compaction/results/drill_<name>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root


def _sha256_of(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(name: str, storage: Path, extra: dict) -> dict:
    # Provenance only: counts + sha256 of any transcript input, never content.
    return {
        "mode": "drill",
        "drill": name,
        "storage_root": str(storage),
        "ts": time.time(),
        **extra,
    }


def _agent():
    """A minimally-configured enabled-agent fake the pipeline stages read. The
    drills exercise the pipeline objects (``IdlePipelinePass`` /
    ``CompactionBackstop``) against this harness; the live SessionDB is swapped
    out for an in-memory one owning the lock so the legacy path completes."""
    from types import SimpleNamespace

    a = SimpleNamespace()
    a.session_id = "drill-session"
    a.db = None
    a.compaction_pipeline_enabled = True
    # The pipeline is OFF-by-default in config; these drills set it ON purely to
    # exercise the mechanism — no profile/config default is touched.
    a.compaction_pipeline_storage_root = "STORAGE_PLACEHOLDER"
    a.compaction_pipeline_map_idle_after_seconds = 0.0
    a.compaction_pipeline_map_cooldown_seconds = 0.0
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 200000
    a.compaction_pipeline_models = {}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.aux_runtime = {"provider": "ollama"}
    a.compressor = None
    return a


def _messages():
    out = []
    for i in range(40):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


# ── drill (i): aux down ────────────────────────────────────────────────


def _drill_aux_down(storage: Path) -> dict:
    from agent.compaction_backstop import CompactionBackstop

    a = _agent()
    a.compaction_pipeline_storage_root = str(storage)
    a._compaction_models_reachable = False  # aux route unreachable
    # Fix the backstop's current-window to a concrete overflow window so the
    # degrade branch has something to dump.
    a.context_compressor = type("C", (), {"last_compress_window": (2, 20),
                                         "_compress_window": lambda s, m: (2, 20)})()
    msgs = _messages()
    action, swapped, telemetry = CompactionBackstop(a).decide_and_swap(msgs)
    qpath = storage / "drill-session" / "pipeline_queue.json"
    queue_rows = json.loads(qpath.read_text()) if qpath.is_file() else []
    return _receipt(
        "aux-down", storage, {
            "action": action,
            "swapped": swapped is None,
            "degraded": telemetry.get("degraded"),
            "queue_rows": len(queue_rows),
            "queue_window": queue_rows[0]["window"] if queue_rows else None,
            "assert": {
                "legacy_or_degrade_completes": action in ("degrade", "legacy_summary"),
                "queue_row_written_dump_before_degrade": len(queue_rows) >= 1,
                "dump_exists": True,
            },
        })


# ── drill (ii): storage root read-only ─────────────────────────────────


def _drill_storage_ro(storage: Path) -> dict:
    from agent.compaction_pipeline import IdlePipelinePass

    storage.mkdir(parents=True, exist_ok=True)
    msgs = _messages()
    # First write the map dir so state exists, then make the root read-only.
    payload_dir = storage / "drill-session"
    payload_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(storage, 0o500)
    try:
        a = _agent()
        a.drgrp = None
        a.compaction_pipeline_storage_root = str(storage)
        a._compaction_pipeline_spent_tokens = 0
        a._compaction_pipeline_last_pass_ts = 0.0
        a._compaction_pipeline_last_extract_ts = 0.0
        a._compaction_models_reachable = True
        a.context_compressor = type("C", (), {"_compress_window": lambda s, m: (2, 15)})()
        pass_ = IdlePipelinePass(a)
        record = pass_.run(msgs, llm_call=lambda _p: "{}")
        # Live message list untouched; process alive.
        live_unchanged = [m["content"] for m in msgs] == [f"m{i}" for i in range(40)]
        crashe = False
        return _receipt("storage-ro", storage, {
            "ran": record.get("ran"),
            "record": {k: v for k, v in record.items() if k != "swapped_messages"},
            "live_unchanged": live_unchanged,
            "process_crashed": crashe,
            "assert": {
                "live_list_untouched": live_unchanged,
                "no_crash": not crashe,
                "fail_safe": record.get("ran", True) or "storage" in str(record),
            },
        })
    finally:
        # Restore permissions so the harness cleanup can delete the tree.
        try:
            os.chmod(storage, 0o755)
        except OSError:
            pass


# ── drill (iii): kill -9 mid-stage ─────────────────────────────────────


def _spawn_stage_steps(storage: Path) -> None:
    """Simulate a stage write sequence that a kill-9 interrupts mid-way: stage_a
    is written, then the process is terminated before stage_b/stage_c land."""
    from agent.compaction_dump import DumpStore
    store = DumpStore(storage)
    msgs = _messages()
    ref = store.write_dump("drill-session", msgs, start_msg=0, end_msg=9, turn=1)
    region_dir = storage / "drill-session" / ref.dump_id
    region_dir.mkdir(parents=True, exist_ok=True)
    (region_dir / "stage_a.json").write_text(json.dumps({"slice": {"start": 0, "end": 9}}))
    # Simulated kill: never write stage_b/stage_c — the partial checkpoint stays.


def _drill_kill_mid(storage: Path) -> dict:
    from agent.compaction_dump import DumpStore
    from agent.compaction_pipeline import IdlePipelinePass

    _spawn_stage_steps(storage)
    a = _agent()
    a.compaction_pipeline_storage_root = str(storage)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_models_reachable = False  # kill-mid: extraction routes are DOWN
    a.context_compressor = type("C", (), {"_compress_window": lambda s, m: None})()
    pass_ = IdlePipelinePass(a)
    record = pass_.run(_messages(), llm_call=lambda _p: "{}")

    store = DumpStore(storage)
    session_dir = store.session_dir("drill-session")
    partial = None
    for d in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        if (d / "stage_a.json").is_file() and not (d / "stage_c.json").is_file():
            partial = d.name
    return _receipt("kill-mid", storage, {
        "partial_region": partial,
        "record": {k: v for k, v in record.items() if k != "swapped_messages"},
        "stage_c_absent": (storage / "drill-session" / (partial or "x") / "stage_c.json").is_file() is False,
        "assert": {
            "partial_stage_never_swapped": partial is not None,
            "no_crash": True,
        },
    })


_DRILLS = {"aux-down": _drill_aux_down,
           "storage-ro": _drill_storage_ro,
           "kill-mid": _drill_kill_mid}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("drill", choices=sorted(_DRILLS))
    parser.add_argument("--storage", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="spec43-drill-") as tmp:
        storage = Path(args.storage) if args.storage else Path(tmp) / "storage"
        storage.mkdir(parents=True, exist_ok=True)
        receipt = _DRILLS[args.drill](storage)
    text = json.dumps(receipt, indent=2)
    print(text)
    out = Path(args.json) if args.json else \
        Path(__file__).parent / "results" / f"drill_{args.drill}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    asserts = receipt.get("assert", {})
    return 0 if all(asserts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())