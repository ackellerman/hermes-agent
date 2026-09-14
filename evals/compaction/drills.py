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
import signal
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


def _container_provenance() -> dict | None:
    """Container identity for the receipt, or None for a bare host run (AC-9).

    The container wrapper exports these; nothing is guessed when they are absent,
    because a receipt carrying no container field must FAIL the AC.
    """
    image = os.environ.get("HERMES_EVAL_CONTAINER_IMAGE")
    if not image:
        return None
    return {
        "container_image": image,
        "container_image_id": os.environ.get("HERMES_EVAL_CONTAINER_IMAGE_ID", ""),
        "container_image_digest": os.environ.get("HERMES_EVAL_CONTAINER_IMAGE_DIGEST", ""),
        "container_commit": os.environ.get("HERMES_EVAL_CONTAINER_COMMIT", ""),
        "container_runtime": os.environ.get("HERMES_EVAL_CONTAINER_RUNTIME", "docker"),
        "container_command": os.environ.get("HERMES_EVAL_CONTAINER_COMMAND", ""),
    }


def _receipt(name: str, storage: Path, extra: dict) -> dict:
    # Provenance only: counts + sha256 of any transcript input, never content.
    # ``container`` identifies the image/commit the receipt was produced from
    # (AC-9); None on a bare host run, which is exactly what makes a host-produced
    # receipt fail the AC.
    return {
        "mode": "drill",
        "drill": name,
        "storage_root": str(storage),
        "ts": time.time(),
        "container": _container_provenance(),
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


def _deterministic_stage_llms(dump_id, start, end):
    """Deterministic stage LLMs for the drills (real pipeline code, no live model):
    the gate must PASS so a region's swap-eligibility turns on its on-disk
    artifacts alone — otherwise the drill would be measuring the stub gate, not
    the partial-region discipline it claims to test."""
    def extract_llm(_messages):
        return json.dumps({
            "instructions_and_corrections": [{"what": "use JSONL", "kind": "correction",
                                              "cites": [[dump_id, start, end]]}],
            "decisions": [{"what": "Postgres", "cites": [[dump_id, start, end]]}],
            "commitments": [{"what": "ship Friday", "cites": [[dump_id, start, end]]}],
            "artifacts": [{"what": "audit-summary", "cites": [[dump_id, start, end]],
                           "recoverable": False}],
            "world_effects": [{"what": "staging runs exporter", "cites": [[dump_id, start, end]]}],
            "insights": "null_reason: none", "open_threads": "null_reason: none",
            "links": "null_reason: none", "narrative": "Export service work.",
            "confidence": 0.9, "coverage": {"complete": True}})

    def map_llm(messages):
        data = json.loads(messages[1]["content"])
        chunk = data["chunk"]
        return json.dumps({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": int(chunk["end_msg"])},
            "episodes": [{"name": "ep", "start_msg": int(chunk["start_msg"]),
                          "end_msg": int(chunk["end_msg"]), "topic": "work"}],
            "entities": [], "edges": []})

    def pass_llm(_messages):
        # Review gate + loss probe both pass: the checkpoint cites the dump.
        return json.dumps({"swap_eligible": True, "findings": []})

    return {"reason": extract_llm, "extract": extract_llm, "check": extract_llm,
            "gate": pass_llm, "map": map_llm}


def _run_pass_bounded(pass_, messages, *, seconds: float, map_llm=None) -> dict:
    """Run a pipeline pass under a hard wall-clock bound (drill telemetry must not
    hang the lane). On timeout the drill reports the timeout rather than a result
    it never observed. ``map_llm`` feeds the run-level map update — passing the
    deterministic map stub keeps the receipt free of a spurious ``map_error`` that
    would mask the failure the drill is actually measuring."""
    import threading
    box: dict = {}

    def target():
        try:
            box["record"] = pass_.run(messages, llm_call=map_llm or (lambda _p: "{}"))
        except Exception as exc:  # noqa: BLE001 — surfaced, never swallowed
            box["error"] = repr(exc)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=seconds)
    if t.is_alive():
        return {"ran": False, "timeout_seconds": seconds,
                "reason": "pass exceeded the drill's wall-clock bound"}
    if "error" in box:
        return {"ran": False, "reason": f"pass raised: {box['error']}"}
    return box.get("record", {"ran": False, "reason": "no record"})


def _messages():
    out = []
    for i in range(40):
        out.append({"role": "assistant" if i % 2 else "user", "content": f"m{i}"})
    return out


# ── drill (i): aux down ────────────────────────────────────────────────


def _drill_aux_down(storage: Path) -> dict:
    from agent.compaction_backstop import CompactionBackstop
    from agent.compaction_dump import DumpStore

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

    # CAUSAL: the dump the queue row refers to must actually exist on disk and be
    # complete. A literal ``True`` here would pass even with dump-before-degrade
    # entirely absent, which is the very behaviour AC-21 requires.
    store = DumpStore(storage)
    dumped_ids = store.dump_ids("drill-session")
    complete_dumps = [d for d in dumped_ids if store.is_complete("drill-session", d)]
    queue_window = queue_rows[0]["window"] if queue_rows else None
    window_matches_a_dump = any(
        [int((store.read_meta("drill-session", d) or {}).get("start_msg", -1)),
         int((store.read_meta("drill-session", d) or {}).get("end_msg", -2))] == list(queue_window)
        for d in complete_dumps) if queue_window else False
    return _receipt(
        "aux-down", storage, {
            "action": action,
            "swapped": swapped is None,
            "degraded": telemetry.get("degraded"),
            "queue_rows": len(queue_rows),
            "queue_window": queue_window,
            "dumps_on_disk": dumped_ids,
            "complete_dumps": complete_dumps,
            "assert": {
                "legacy_or_degrade_completes": action in ("degrade", "legacy_summary"),
                "queue_row_written_dump_before_degrade": len(queue_rows) >= 1,
                "dumped_region_exists_on_disk": bool(complete_dumps),
                "queue_row_window_has_a_complete_dump": window_matches_a_dump,
            },
        })


# ── drill (ii): storage root read-only ─────────────────────────────────


def _make_storage_unwritable(storage: Path, session_id: str = "drill-session") -> tuple:
    """Force the storage root into a genuinely unwritable state and PROVE it with a
    real 4096-byte write. Returns ``(mechanism, writable)``.

    Two mechanisms, because one alone is not portable:
    - ``chmod``: 0o500 on the root AND the session subdir. A root-only chmod is a
      no-op here — the store writes into the session subdir, which stays writable
      (this is exactly the hole that made the old drill's assertions pass against
      a writable root).
    - ``non-directory-root``: replace the root with a regular file. Any path under
      it raises ENOTDIR for every uid, so the condition also holds for a
      container running as root, where mode bits are bypassed.
    """
    probe = storage / session_id / ".ro-probe"
    real_bytes = b"x" * 4096

    def _writable() -> bool:
        try:
            probe.write_bytes(real_bytes)
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    for d in (storage, storage / session_id):
        if d.is_dir():
            try:
                os.chmod(d, 0o500)
            except OSError:
                pass
    if not _writable():
        return "chmod", False

    # Root (or a mode-bit-ignoring filesystem): fall back to a condition no uid
    # can defeat.
    for d in (storage / session_id, storage):
        if d.is_dir():
            try:
                os.chmod(d, 0o755)
            except OSError:
                pass
    shutil.rmtree(storage, ignore_errors=True)
    storage.write_text("not a directory", encoding="utf-8")
    return "non-directory-root", _writable()


def _drill_storage_ro(storage: Path) -> dict:
    from agent.compaction_pipeline import IdlePipelinePass

    storage.mkdir(parents=True, exist_ok=True)
    msgs = _messages()
    # First write the map dir so state exists, then force the root unwritable.
    payload_dir = storage / "drill-session"
    payload_dir.mkdir(parents=True, exist_ok=True)
    mechanism, writable = _make_storage_unwritable(storage)
    try:
        if writable:
            # The drill cannot make a causal claim on a writable root: a "pass"
            # would be the pass-side succeeding normally, not the failure path.
            return _receipt("storage-ro", storage, {
                "unwritable_mechanism": mechanism,
                "preflight_root_writable": True,
                "assert": {"preflight_proved_root_read_only": False},
                "note": ("storage root is still writable to this process; the drill "
                         "refuses to claim a read-only result it did not produce"),
            })

        a = _agent()
        a.compaction_pipeline_storage_root = str(storage)
        a._compaction_pipeline_spent_tokens = 0
        a._compaction_pipeline_last_pass_ts = 0.0
        a._compaction_pipeline_last_extract_ts = 0.0
        a._compaction_models_reachable = True
        a._compaction_stage_llms = _deterministic_stage_llms("unused", 2, 15)
        a.context_compressor = type("C", (), {"_compress_window": lambda s, m: (2, 15)})()
        pass_ = IdlePipelinePass(a)
        record = _run_pass_bounded(pass_, msgs, seconds=60,
                                   map_llm=a._compaction_stage_llms["map"])
        # Live message list untouched; process alive.
        live_unchanged = [m["content"] for m in msgs] == [f"m{i}" for i in range(40)]
        # CAUSAL: the pass must have RECORDED a storage failure. "ran" alone is
        # true for a pass that silently did nothing; the failure has to show up in
        # the telemetry for the fail-safe claim to mean anything.
        recorded_failure = bool(record.get("dump_error") or record.get("map_error")
                                or record.get("queue_error") or record.get("storage_error"))
        dumped = bool(record.get("dumped"))
        return _receipt("storage-ro", storage, {
            "unwritable_mechanism": mechanism,
            "preflight_root_writable": writable,
            "ran": record.get("ran"),
            "record": {k: v for k, v in record.items() if k != "swapped_messages"},
            "live_unchanged": live_unchanged,
            "process_crashed": False,
            "assert": {
                "preflight_proved_root_read_only": writable is False,
                "live_list_untouched": live_unchanged,
                "no_crash": True,
                "storage_failure_recorded": recorded_failure,
                "nothing_written_to_readonly_root": not dumped,
            },
        })
    finally:
        # Restore so the harness can clean up (rmtree a file-as-root or a 0o500 dir).
        try:
            if storage.is_file():
                storage.unlink()
            else:
                for d in (storage / "drill-session", storage):
                    if d.is_dir():
                        os.chmod(d, 0o755)
        except OSError:
            pass


# ── drill (iii): kill -9 mid-stage ─────────────────────────────────────


_KILL_MID_CHILD = '''
import json, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from agent.compaction_dump import DumpStore

storage = Path(sys.argv[1])
store = DumpStore(storage)
msgs = [{"role": "assistant" if i % 2 else "user", "content": f"m{i}"} for i in range(40)]
ref = store.write_dump("drill-session", msgs[:10], start_msg=0, end_msg=9, turn=1)
region = store.dump_dir("drill-session", ref.dump_id)
(region / "stage_a.json").write_text(json.dumps({"slice": {"start": 0, "end": 9}}))
# Kill the process between stage writes: stage_b/stage_c never land.
os.kill(os.getpid(), signal.SIGKILL)
'''


def _drill_kill_mid(storage: Path) -> dict:
    from agent.compaction_dump import DumpStore
    from agent.compaction_pipeline import IdlePipelinePass

    # CAUSAL: spawn a REAL child process and SIGKILL it between stage writes.
    # (The previous version only *described* a kill in a docstring and wrote the
    # partial stage from the parent, so no process was ever terminated.)
    proc = subprocess.run(
        [sys.executable, "-c", _KILL_MID_CHILD, str(storage),
         "drill-session", str(Path(__file__).resolve().parents[2])],
        capture_output=True, text=True, timeout=120,
    )
    killed = proc.returncode == -signal.SIGKILL

    store = DumpStore(storage)
    session_dir = store.session_dir("drill-session")
    partial = None
    for d in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        if (d / "stage_a.json").is_file() and not (d / "stage_c.json").is_file():
            partial = d.name
    partial_dir = session_dir / partial if partial else None
    stage_c_absent = partial_dir is not None and not (partial_dir / "stage_c.json").is_file()

    a = _agent()
    a.compaction_pipeline_storage_root = str(storage)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    # Models ARE reachable here on purpose: the only thing stopping a swap must be
    # the absent stage_c / gate artifact, not an unreachable route.
    a._compaction_models_reachable = True
    a._compaction_stage_llms = _deterministic_stage_llms(partial or "unused", 0, 9)
    a.context_compressor = type("C", (), {"_compress_window": lambda s, m: (0, 9)})()
    live = _messages()
    record = _run_pass_bounded(IdlePipelinePass(a), live, seconds=60,
                               map_llm=a._compaction_stage_llms["map"])

    swapped_regions = record.get("swapped_regions") or []
    swapped_contents = [m.get("content") for m in (record.get("swapped_messages") or [])]
    live_untouched = [m["content"] for m in live] == [f"m{i}" for i in range(40)]
    return _receipt("kill-mid", storage, {
        "child_killed_by_signal": killed,
        "child_returncode": proc.returncode,
        "partial_region": partial,
        "record": {k: v for k, v in record.items() if k != "swapped_messages"},
        "assert": {
            "real_process_killed_mid_stage": killed,
            "partial_region_present": partial is not None,
            "partial_region_has_no_stage_c": stage_c_absent,
            "partial_region_never_swapped": partial is not None and partial not in swapped_regions,
            "live_list_untouched": live_untouched,
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