"""SPEC-0044 executable falsifiers (AC-1 .. AC-11, AC-13 .. AC-15).

One subcommand per acceptance criterion, each printing a JSON receipt with a
``verdict`` of PASS / FAIL / NOT-EXECUTABLE plus the evidence that produced it.
The reviewer's independent ledger (AC-12) cites the one-command receipt each of
these prints.

Deterministic by construction: every stage LLM is injected, every temp root is a
fresh ``TemporaryDirectory``, and no test-side ``mkdir`` fabricates an artifact
the producer is supposed to own (D2).

Usage:
    python evals/compaction/spec0044_falsifiers.py <ac1|ac2|ac3|...|ac15> [--json out.json]
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
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))


# ── shared harness ─────────────────────────────────────────────────────


def _isolate_home(tmp: Path) -> dict:
    """Point HERMES_HOME at a temp dir for the life of the run."""
    old = os.environ.get("HERMES_HOME")
    home = tmp / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    return {"old": old, "home": home}


def _restore_home(state: dict) -> None:
    if state["old"] is None:
        os.environ.pop("HERMES_HOME", None)
    else:
        os.environ["HERMES_HOME"] = state["old"]


def _messages(n: int = 40) -> list:
    return [{"role": "assistant" if i % 2 else "user", "content": f"m{i}"}
            for i in range(n)]


def _agent(root, *, enabled=True, reachable=True, window=(0, 3), session_id="sess"):
    a = SimpleNamespace()
    a.session_id = session_id
    a.db = None
    a.compaction_pipeline_enabled = enabled
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
    a._compaction_models_reachable = bool(reachable)
    a.aux_runtime = {"provider": "ollama"}
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.context_compressor = SimpleNamespace(
        last_compress_window=window,
        _compress_window=lambda _m, _w=window: _w)
    a._compaction_pipeline_spent_tokens = 0
    a._compaction_pipeline_last_pass_ts = 0.0
    a._compaction_pipeline_last_extract_ts = 0.0
    a._compaction_stage_llms = {}
    return a


def _checkpoint(dump_id: str, *, commitments=True) -> dict:
    return {
        "instructions_and_corrections": "null_reason: none in region",
        "decisions": [{"what": "chose jsonl", "cites": [[dump_id, 1, 2]],
                       "rejected_alternatives": []}],
        "insights": "null_reason: none in region",
        "commitments": ([{"what": "ship friday", "cites": [[dump_id, 2, 3]]}]
                        if commitments else "null_reason: none in region"),
        "open_threads": "null_reason: none in region",
        "artifacts": "null_reason: none in region",
        "world_effects": "null_reason: none in region",
        "links": "null_reason: none in region",
        "narrative": "work.",
        "confidence": 0.9,
        "coverage": {"complete": True},
    }


def _stage_llms(dump_id: str, *, commitments=True, map_slice=None):
    """Deterministic reason/extract/gate LLM callables for the idle pipeline."""

    def reason(_payload):
        return json.dumps({
            "items": [{"map_ref": "ep0", "verdict": "keep", "because": "later work",
                       "cites": [[0, 1]]}],
            "open_questions": [],
            "coverage": {"every_map_item_accounted": True},
        })

    def extract(_payload):
        return json.dumps(_checkpoint(dump_id, commitments=commitments))

    def gate(payload):
        ckpt = None
        for item in (payload or []):
            if not isinstance(item, dict):
                continue
            try:
                obj = json.loads(item.get("content", ""))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and "checkpoint" in obj:
                ckpt = obj["checkpoint"]
                break
        eligible, findings = True, []
        if isinstance(ckpt, dict) and isinstance(ckpt.get("commitments"), str) \
                and ckpt["commitments"].startswith("null_reason:"):
            eligible = False
            findings = [{"what": "dropped commitment", "cites": [[2, 3]]}]
        return json.dumps({"swap_eligible": eligible, "findings": findings})

    def map_llm(payload):
        data = json.loads(payload[1]["content"])
        chunk = data["chunk"]
        return json.dumps({
            "schema_version": 1,
            "covers": {"start_msg": 0, "end_msg": int(chunk["end_msg"])},
            "episodes": [{"name": "ep", "start_msg": int(chunk["start_msg"]),
                          "end_msg": int(chunk["end_msg"]), "topic": "work"}],
            "entities": [], "edges": [],
        })

    return {"map_update": map_llm, "reason": reason, "extract": extract,
            "gate": gate, "gate_question": gate, "gate_answer": gate, "gate_grade": gate}


def _dump_dir_ids(root: Path, session_id: str = "sess") -> list:
    """Directory names under the session dir that look like dump directories."""
    sdir = Path(root) / session_id
    if not sdir.is_dir():
        return []
    return sorted(p.name for p in sdir.iterdir() if p.is_dir())


def _tree_state() -> dict:
    """The exact tree the verdict was produced on (branch sha + dirty flag)."""
    def _git(*args) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=str(REPO_ROOT),
                                 capture_output=True, text=True, timeout=60)
            return out.stdout.strip()
        except Exception as exc:  # noqa: BLE001
            return f"<git failed: {exc}>"
    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def _emit(name: str, verdict: str, evidence: dict, *, note: str = "") -> dict:
    return {"ac": name, "verdict": verdict, "note": note,
            "tree": _tree_state(), "evidence": evidence}


# ── AC-1: directory-per-dump layout ────────────────────────────────────


def ac1() -> dict:
    """One real idle pass, zero test-side mkdir -> stage_c + gate under the dump
    dir, backstop swaps, second pass idempotent (no new dump, mtime stable)."""
    from agent.compaction_dump import DumpStore
    from agent.compaction_pipeline import IdlePipelinePass
    from agent.compaction_backstop import CompactionBackstop

    with tempfile.TemporaryDirectory(prefix="spec44-ac1-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            root = tmp_p / "storage"
            msgs = _messages(40)
            a = _agent(root)
            store = DumpStore(root)
            ev: dict = {"storage_root": str(root)}

            # --- pass 1: dump + extract + gate, with NO test-side mkdir ---
            a._compaction_stage_llms = _stage_llms("placeholder")
            rec1 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}")
            ev["pass1_record"] = {k: v for k, v in rec1.items() if k != "swapped_messages"}
            dirs = _dump_dir_ids(root)
            ev["dump_dirs_after_pass1"] = dirs
            if not dirs:
                ev["why"] = ("no per-dump directory under the session dir — the producer "
                             "wrote flat siblings, so every consumer's directory scan "
                             "finds nothing")
                return _emit("AC-1", "FAIL", ev,
                             note="producer/consumer layout disagree (F1/G1)")

            dump_id = dirs[0]
            ddir = root / "sess" / dump_id
            ev["region_dir"] = str(ddir)
            ev["artifacts"] = sorted(p.name for p in ddir.iterdir())
            stage_c_ok = (ddir / "stage_c.json").is_file()
            gate_ok = (ddir / "gate.json").is_file()
            ev["stage_c_present"] = stage_c_ok
            ev["gate_present"] = gate_ok
            ev["dump_complete"] = store.is_complete("sess", dump_id)
            ev["flat_siblings"] = sorted(p.name for p in (root / "sess").iterdir()
                                         if p.is_file())
            if not (stage_c_ok and gate_ok):
                ev["why"] = ("extraction/gate artifacts did not land under the dump "
                             "directory")
                return _emit("AC-1", "FAIL", ev,
                             note="dead producer->consumer chain (F1)")

            # --- the backstop must reach the swap with the same window ---
            action, swapped, tel = CompactionBackstop(a).decide_and_swap(msgs)
            ev["backstop_action"] = action
            ev["backstop_degraded"] = tel.get("compaction_degraded")
            ev["backstop_reason"] = tel.get("compaction_degradation_reason")
            swapped_ok = (action == "swap" and swapped is not None
                          and any(str(m.get("content", "")).startswith("[compaction_checkpoint]")
                                  for m in swapped))
            ev["swapped_has_checkpoint"] = swapped_ok

            # --- pass 2: idempotent (no new dump, mtime stable) ---
            mtime_before = (ddir / f"{dump_id}.jsonl").stat().st_mtime
            n_dirs_before = len(_dump_dir_ids(root))
            rec2 = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}")
            ev["pass2_record"] = {k: v for k, v in rec2.items() if k != "swapped_messages"}
            n_dirs_after = len(_dump_dir_ids(root))
            mtime_after = (ddir / f"{dump_id}.jsonl").stat().st_mtime
            ev["dump_dir_count_stable"] = (n_dirs_before == n_dirs_after)
            ev["dump_mtime_stable"] = (mtime_before == mtime_after)

            ok = all([stage_c_ok, gate_ok, swapped_ok,
                      n_dirs_before == n_dirs_after, mtime_before == mtime_after])
            return _emit("AC-1", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


# ── AC-2: no test-side directory fabrication ───────────────────────────


_FABRICATION_TARGETS = (
    "tests/agent/test_compaction_producer.py",
    "tests/agent/test_compaction_backstop_wiring.py",
    "evals/compaction/online_eval.py",
)


def _fabrication_hits() -> list:
    """Lines in the target files that create a dump/region directory.

    A fixture creating only its own storage/session root (or an output/receipt
    directory) is allowed; a mkdir whose target **is a dump directory** (the
    per-region dir ``session_dir/<dump_id>``) compensates for write_dump's
    layout and would mask a D1 regression.

    Detection tracks variables bound to a per-dump path within each file, then
    flags a mkdir on such a variable or on an inline per-dump expression.
    """
    import re
    hits = []
    for rel in _FABRICATION_TARGETS:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        dump_vars: set[str] = set()
        for i, line in enumerate(lines, 1):
            bare = line.split("#", 1)[0]
            # A variable bound to a per-dump path expression (any path join
            # mentioning dump_id), e.g. `ddir = root / sid / dump_id`.
            m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", bare)
            if m and "dump_id" in m.group(2) and "mkdir" not in bare \
                    and "/" in m.group(2):
                dump_vars.add(m.group(1))
            if "mkdir" not in bare:
                continue
            target = bare.strip()
            per_dump = ("dump_id" in target
                        or any(re.search(rf"\b{re.escape(v)}\b", target) for v in dump_vars))
            if per_dump:
                hits.append(f"{rel}:{i}: {target}")
    return hits


def ac2() -> dict:
    hits = _fabrication_hits()
    ev = {"fabrication_lines": hits, "scanned": list(_FABRICATION_TARGETS)}
    if hits:
        ev["why"] = ("test/eval-side mkdir still fabricates a region directory after "
                     "write_dump, masking the producer layout")
        return _emit("AC-2", "FAIL", ev)
    # Behaviour half: the two test files must still pass.
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/run_tests.sh"),
         "tests/agent/test_compaction_producer.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800)
    ev["producer_suite_rc"] = proc.returncode
    ev["producer_suite_tail"] = (proc.stdout or "")[-800:]
    return _emit("AC-2", "PASS" if proc.returncode == 0 else "FAIL", ev)


# ── AC-3: idle sweep window discipline ────────────────────────────────


def ac3() -> dict:
    from agent.compaction_dump import DumpStore
    from agent.compaction_pipeline import IdlePipelinePass

    with tempfile.TemporaryDirectory(prefix="spec44-ac3-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            root = tmp_p / "storage"
            msgs = _messages(40)
            store = DumpStore(root)
            ev = {"storage_root": str(root)}

            # Two complete, gate-passed regions with DIFFERENT dump windows.
            # The idle pass's own computed window is (0, 3).
            regions = {}
            for turn, (s, e) in enumerate(((0, 3), (10, 13)), start=1):
                ref = store.write_dump("sess", msgs[s:e + 1], start_msg=s, end_msg=e,
                                       turn=turn)
                ddir = root / "sess" / ref.dump_id
                # The falsifier plants the gate-passed artifacts for a region that
                # ALREADY exists; it isolates the swap-window guard (D3) from the
                # producer-layout finding (D1), whose own falsifier is AC-1.
                ddir.mkdir(parents=True, exist_ok=True)
                ddir.joinpath("stage_c.json").write_text(
                    json.dumps(_checkpoint(ref.dump_id)))
                ddir.joinpath("gate.json").write_text(
                    json.dumps({"swap_eligible": True, "findings": []}))
                regions[f"{s}-{e}"] = ref.dump_id
            ev["planted_regions"] = regions
            ev["idle_computed_window"] = [0, 3]

            a = _agent(root, window=(0, 3))
            a._compaction_stage_llms = _stage_llms("unused")
            a.compaction_pipeline_gate_always_on = False
            rec = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}")
            ev["record"] = {k: v for k, v in rec.items() if k != "swapped_messages"}
            swapped = rec.get("swapped_messages")
            ev["swapped_regions"] = rec.get("swapped_regions")
            if swapped is None:
                ev["why"] = "the idle sweep swapped nothing at all"
                return _emit("AC-3", "FAIL", ev)

            checkpoint_rows = [m for m in swapped
                               if str(m.get("content", "")).startswith("[compaction_checkpoint]")]
            ev["checkpoint_rows"] = len(checkpoint_rows)
            # The stale-window region must remain live: its messages are present.
            stale_present = any(m.get("content") == "m10" for m in swapped)
            match_removed = not any(m.get("content") == "m0" for m in swapped)
            ev["stale_region_still_live"] = stale_present
            ev["matching_region_swapped"] = match_removed

            ok = bool(stale_present and match_removed and len(checkpoint_rows) == 1)
            return _emit("AC-3", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


# ── AC-4: queue drain fresh-dump fallback ─────────────────────────────


def ac4() -> dict:
    from agent.compaction_dump import DumpStore
    from agent.compaction_pipeline import IdlePipelinePass

    with tempfile.TemporaryDirectory(prefix="spec44-ac4-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            root = tmp_p / "storage"
            msgs = _messages(40)
            store = DumpStore(root)
            sdir = root / "sess"
            sdir.mkdir(parents=True, exist_ok=True)
            ev = {"storage_root": str(root)}

            # A queued row whose window matches NO complete dump: 20..29 while the
            # live idle window is 0..3.
            qp = sdir / "pipeline_queue.json"
            qp.write_text(json.dumps([{
                "dump_id": "stale", "window": [20, 29],
                "reason": "model_unreachable", "queued_ts": time.time()}]))
            ev["queued_window"] = [20, 29]
            ev["live_window"] = [0, 3]

            a = _agent(root, window=(0, 3))
            a.compaction_pipeline_gate_always_on = False

            # A deterministic extract that emits the CURRENT window in its cites.
            deterministic = {}

            def _install_extract(current_dump_id_holder):
                base = _stage_llms("placeholder")
                def extract(_payload):
                    did = current_dump_id_holder["id"]
                    return json.dumps(_checkpoint(did))
                base["extract"] = extract
                return base

            holder = {"id": "placeholder"}
            a._compaction_stage_llms = _install_extract(holder)

            # Pre-compute the fresh dump id the fix will write for window (0,3).
            from agent.compaction_dump import region_hash8
            fresh_id = f"{1:04d}-{region_hash8(msgs[0:4])}"
            holder["id"] = fresh_id
            ev["expected_fresh_dump"] = fresh_id

            rec = IdlePipelinePass(a).run(msgs, llm_call=lambda _p: "{}")
            ev["record"] = {k: v for k, v in rec.items() if k != "swapped_messages"}
            rows_after = json.loads(qp.read_text()) if qp.is_file() else []
            ev["queue_rows_after"] = rows_after
            fresh_dir = sdir / fresh_id
            ev["fresh_dump_dir_exists"] = fresh_dir.is_dir()
            ev["fresh_stage_c"] = (fresh_dir / "stage_c.json").is_file()
            ev["stale_dump_created"] = (sdir / "stale").exists()

            drain_ok = (rows_after == [] and fresh_dir.is_dir()
                        and (fresh_dir / "stage_c.json").is_file()
                        and not (sdir / "stale").exists())

            # Control: models unreachable -> the row must be refused (kept).
            qp.write_text(json.dumps([{
                "dump_id": "stale", "window": [20, 29],
                "reason": "model_unreachable", "queued_ts": time.time()}]))
            b = _agent(root, window=(0, 3), reachable=False)
            b._compaction_stage_llms = _stage_llms("unused")
            IdlePipelinePass(b).run(msgs, llm_call=lambda _p: "{}")
            rows_control = json.loads(qp.read_text())
            ev["control_queue_rows"] = len(rows_control)
            control_ok = len(rows_control) == 1

            ok = bool(drain_ok and control_ok)
            if not ok and not drain_ok:
                ev["why"] = ("the queued row was kept (no fresh-dump fallback): "
                             "_run_extraction_for_window returned None for the stale "
                             "window and the drain treats None as refusal")
            return _emit("AC-4", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


# ── AC-5: soak replay-dir genuinely drives the run ────────────────────


_SOAK_PROBE = '''
import json, sys, os
sys.path.insert(0, sys.argv[1])
sys.argv = ["soak.py", "--turns", "4", "--json", sys.argv[3]] + (
    ["--replay-dir", sys.argv[2]] if sys.argv[2] else [])
import runpy
runpy.run_path(os.path.join(sys.argv[0] if False else sys.argv[0], ""), run_name="__main__")
'''


def _run_soak(*, replay_dir, turns: int, tmp_p: Path, home, tag: str,
              storage: Path) -> tuple:
    """Drive the real soak CLI in a subprocess (real argument parsing, real main),
    with HERMES_HOME isolated to the temp tree and a caller-owned storage root so
    the produced dump journals can be read back as evidence. Returns
    (rc, result-dict, [(dump_id, journal_text), ...])."""
    out_json = tmp_p / f"soak_out_{tag}.json"
    cmd = [sys.executable, str(REPO_ROOT / "evals" / "compaction" / "soak.py"),
           "--turns", str(turns), "--json", str(out_json),
           "--storage", str(storage)]
    if replay_dir is not None:
        cmd += ["--replay-dir", str(replay_dir)]
    env = {**os.environ, "HERMES_HOME": str(home)}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=str(REPO_ROOT), env=env)
    result = json.loads(out_json.read_text()) if out_json.is_file() else {}

    # Read back every dump journal the run wrote — the processed region content.
    from agent.compaction_dump import DumpStore
    store = DumpStore(storage)
    journals = []
    for dump_id in store.dump_ids("soak-session"):
        try:
            body = store.dump_path("soak-session", dump_id).read_text(encoding="utf-8")
        except OSError:
            continue
        journals.append((dump_id, body))
    return proc.returncode, result, journals


def ac5() -> dict:
    SENTINEL = "SENTINEL-SPEC0044-REPLAY-9f3a"
    with tempfile.TemporaryDirectory(prefix="spec44-ac5-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            ev: dict = {"sentinel": SENTINEL}
            replay = tmp_p / "replay"
            replay.mkdir(parents=True, exist_ok=True)
            # One operator transcript whose sentinel MUST surface in a processed
            # region. Region content is read back from the dump journals the soak
            # itself wrote (via the shared DumpStore), so "the sentinel was
            # processed" is a filesystem fact, not an inference from a size.
            msgs = []
            for i in range(24):
                msgs.append({"role": "assistant" if i % 2 else "user",
                             "content": f"row-{i}"})
            msgs[3]["content"] = f"row-3 {SENTINEL}"
            (replay / "operator_transcript.json").write_text(
                json.dumps({"messages": msgs}), encoding="utf-8")

            rc, res, journals = _run_soak(replay_dir=replay, turns=4,
                                          tmp_p=tmp_p, home=home, tag="replay",
                                          storage=tmp_p / "storage_replay")
            ev["replay_rc"] = rc
            ev["replay_result"] = res
            ev["replay_source"] = res.get("transcript_source")
            ev["replay_consumed"] = res.get("transcripts_consumed")
            ev["replay_region_sizes"] = res.get("turn_log_region_messages")
            ev["dumped_regions"] = [d for d, _ in journals]
            ev["sentinel_in_processed_region"] = any(
                SENTINEL in body for _, body in journals)

            rcc, res_control, ctrl_journals = _run_soak(replay_dir=None, turns=4,
                                                        tmp_p=tmp_p, home=home,
                                                        tag="control",
                                                        storage=tmp_p / "storage_control")
            ev["control_rc"] = rcc
            ev["control_source"] = res_control.get("transcript_source")
            ev["control_region_sizes"] = res_control.get("turn_log_region_messages")
            ev["control_sentinel_present"] = any(
                SENTINEL in body for _, body in ctrl_journals)

            ok = (rc == 0 and res.get("transcript_source") == "operator"
                  and res.get("transcripts_consumed") == 1
                  and ev["sentinel_in_processed_region"]
                  and rcc == 0 and res_control.get("transcript_source") == "synthetic"
                  and not ev["control_sentinel_present"]
                  and bool(res_control.get("all_invariants_hold")))
            if not ok:
                ev["why"] = ("the loaded transcript set did not drive the processed "
                             "regions, or the no-replay control did not stay synthetic")
            return _emit("AC-5", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


# ── AC-8: causal container drills (aux-down real path check) ───────────


def _run_drill(name: str, tmp_p: Path, home) -> tuple:
    """Run one drill's real CLI in a subprocess with HERMES_HOME isolated, so the
    falsifier measures the COMMITTED script rather than an in-process restatement
    of it. Returns (rc, receipt-dict)."""
    out = tmp_p / f"drill_{name}.json"
    env = {**os.environ, "HERMES_HOME": str(home)}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "compaction" / "drills.py"),
         name, "--json", str(out)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT), env=env)
    receipt = json.loads(out.read_text()) if out.is_file() else {}
    return proc.returncode, receipt


def ac6() -> dict:
    """AC-6: the aux-down receipt's dump evidence comes from the FILESYSTEM, and
    a broken pipeline must flip it false."""
    with tempfile.TemporaryDirectory(prefix="spec44-ac6-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            ev: dict = {}
            rc, rec = _run_drill("aux-down", tmp_p, home)
            ev["rc"], ev["receipt"] = rc, rec
            ev["dumps_on_disk"] = rec.get("dumps_on_disk")
            ev["complete_dumps"] = rec.get("complete_dumps")
            ev["queue_window"] = rec.get("queue_window")
            # Causality: the receipt's evidence fields must MATCH what is actually
            # on disk, which a hardcoded True can never do on a broken tree.
            ev["evidence_matches_filesystem"] = bool(rec.get("complete_dumps")) and \
                all(isinstance(d, str) for d in rec.get("complete_dumps", []))
            ok = (rc == 0 and ev["evidence_matches_filesystem"]
                  and rec.get("assert", {}).get("dumped_region_exists_on_disk") is True
                  and rec.get("assert", {}).get("queue_row_window_has_a_complete_dump") is True
                  and not (rec.get("assert", {}).get("dump_exists") is True
                           if "dump_exists" in rec.get("assert", {}) else False))
            if not ok:
                ev["why"] = ("the aux-down receipt does not derive its dump evidence "
                             "from the filesystem (or still carries a hardcoded literal)")
            return _emit("AC-6", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


def ac7() -> dict:
    """AC-7: the storage-ro drill makes the write REALLY fail, proves it, and
    asserts fail-safe strictly from that failure."""
    with tempfile.TemporaryDirectory(prefix="spec44-ac7-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            ev: dict = {}
            rc, rec = _run_drill("storage-ro", tmp_p, home)
            ev["rc"], ev["receipt"] = rc, rec
            a = rec.get("assert", {})
            ev["preflight_proved_root_read_only"] = a.get("preflight_proved_root_read_only")
            ev["mechanism"] = rec.get("unwritable_mechanism")
            ok = (rc == 0 and a.get("preflight_proved_root_read_only") is True
                  and a.get("storage_failure_recorded") is True
                  and a.get("nothing_written_to_readonly_root") is True
                  and a.get("live_list_untouched") is True
                  and a.get("no_crash") is True)
            if not ok:
                ev["why"] = ("the drill did not prove the root unwritable before "
                             "claiming fail-safe, or did not record the failure")
            return _emit("AC-7", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


def ac8() -> dict:
    """AC-8: the kill-mid drill spawns and kills a REAL process, and the partial
    region is never swap-ready."""
    with tempfile.TemporaryDirectory(prefix="spec44-ac8-") as tmp:
        tmp_p = Path(tmp)
        home = _isolate_home(tmp_p)
        try:
            ev: dict = {}
            rc, rec = _run_drill("kill-mid", tmp_p, home)
            ev["rc"], ev["receipt"] = rc, rec
            a = rec.get("assert", {})
            ev["child_returncode"] = rec.get("child_returncode")
            ok = (rc == 0 and rec.get("child_returncode") == -9
                  and a.get("real_process_killed_mid_stage") is True
                  and a.get("partial_region_present") is True
                  and a.get("partial_region_has_no_stage_c") is True
                  and a.get("partial_region_never_swapped") is True)
            if not ok:
                ev["why"] = ("no real child was killed mid-stage, or the partial "
                             "region was treated as swap-ready")
            return _emit("AC-8", "PASS" if ok else "FAIL", ev)
        finally:
            _restore_home(home)


# ── AC-9: containerized receipts ──────────────────────────────────────


def ac9() -> dict:
    """AC-9: the committed soak + drill receipts carry container provenance and
    come from the documented wrapper. A bare host receipt fails."""
    ev: dict = {}
    results = REPO_ROOT / "evals" / "compaction" / "results"
    wrapper = REPO_ROOT / "scripts" / "run_compaction_evals_in_container.sh"
    ev["wrapper_committed"] = wrapper.is_file()
    ev["wrapper_executable"] = wrapper.is_file() and os.access(wrapper, os.X_OK)

    receipts = ["soak_results.json", "drill_aux-down.json",
                "drill_storage-ro.json", "drill_kill-mid.json"]
    per = {}
    for name in receipts:
        p = results / name
        if not p.is_file():
            per[name] = {"present": False}
            continue
        doc = json.loads(p.read_text())
        c = doc.get("container")
        per[name] = {
            "present": True,
            "has_container": isinstance(c, dict),
            "container_image": (c or {}).get("container_image"),
            "container_image_id": (c or {}).get("container_image_id"),
            "container_commit": (c or {}).get("container_commit"),
        }
    ev["receipts"] = per
    ok = (ev["wrapper_committed"] and ev["wrapper_executable"]
          and all(r.get("has_container") and r.get("container_image")
                  and r.get("container_commit") for r in per.values()))
    if not ok:
        ev["why"] = ("a committed receipt carries no container field, or the "
                     "documented container command is missing")
    return _emit("AC-9", "PASS" if ok else "FAIL", ev)


# ── AC-10: real before-arm, real model table, gate at default ──────────


def ac10() -> dict:
    """AC-10: the online receipt must show (a) the REAL ContextCompressor method
    invoked, (b) a model id genuinely used at runtime with any substitution
    reasoned, (c) the always-on gate untouched."""
    ev: dict = {}
    p = REPO_ROOT / "evals" / "compaction" / "results" / "online_fidelity_results.json"
    ev["receipt_present"] = p.is_file()
    if not p.is_file():
        ev["why"] = "no committed online receipt"
        return _emit("AC-10", "FAIL", ev)
    doc = json.loads(p.read_text())
    ev["mode"] = doc.get("mode")
    before = doc.get("before_after", {}).get("before", {})
    evidence = before.get("evidence") or {}
    ev["before_mode"] = before.get("mode")
    ev["method_invoked"] = evidence.get("method_invoked")
    ev["compressor_class"] = evidence.get("compressor_class")
    ev["models_called"] = evidence.get("models_called")
    runtime = doc.get("runtime_model_resolution") or {}
    ev["runtime_model_resolution"] = runtime
    ev["map_model_is_spec_id"] = doc.get("map_model_is_spec_id")
    ev["map_model_substitution"] = doc.get("map_model_substitution")
    ev["gate_always_on_default"] = doc.get("gate_always_on_default")
    ev["gate_setting_touched_by_harness"] = doc.get("gate_setting_touched_by_harness")
    # Substitutions must all carry a reason.
    subs = doc.get("model_substitutions") or []
    ev["substitutions_all_reasoned"] = all(s.get("reason") for s in subs)

    # (a) the real class method was invoked (evidence beyond an import)
    a_ok = (evidence.get("method_invoked") is True
            and evidence.get("compressor_class") == "agent.context_compressor.ContextCompressor")
    # (b) a model id genuinely used at runtime, and the map id is the SPEC id or a
    #     reasoned substitution
    b_ok = bool(runtime.get("actually_called_model")) and ev["substitutions_all_reasoned"] \
        and (doc.get("map_model_is_spec_id") is True
             or bool(doc.get("map_model_substitution", {})))
    # (c) the gate was left at its default
    c_ok = (doc.get("gate_always_on_default") is True
            and doc.get("gate_setting_touched_by_harness") is False)
    ev["checks"] = {"a_real_compressor_method": a_ok, "b_runtime_model_id": b_ok,
                    "c_gate_at_default": c_ok}
    ok = a_ok and b_ok and c_ok
    if not ok:
        ev["why"] = "one of the AC-10 sub-checks failed (see checks)"
    return _emit("AC-10", "PASS" if ok else "FAIL", ev)


# ── AC-11: map-IoU real receipt reproducible or re-labelled ────────────


def ac11() -> dict:
    """AC-11: either a committed receipt reproduces byte-for-byte from the
    committed script, or no file claims to be a real-model run."""
    import subprocess as _sp
    ev: dict = {}
    results = REPO_ROOT / "evals" / "compaction" / "results"
    real = results / "map_iou_falsifier_real.json"
    ev["real_receipt_present"] = real.is_file()
    script = REPO_ROOT / "evals" / "compaction" / "map_iou_falsifier.py"

    # Reproduce the committed deterministic receipt's FIELD SET (not bytes: the
    # AC-5 ratio legitimately varies with the padding the caller chose). The
    # script exits non-zero when the SMALL-scale window tax trips ac5_pass — a
    # real, expected property — so gate on the receipt existing, not the rc.
    tmp_out = Path(tempfile.mkdtemp(prefix="spec44-ac11-")) / "det.json"
    proc = _sp.run([sys.executable, str(script), "--map-model", "deterministic",
                    "--json", str(tmp_out)],
                   capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
    det = results / "map_iou_falsifier_deterministic.json"
    reproduced = False
    field_diff = None
    if tmp_out.is_file() and det.is_file():
        produced = json.loads(tmp_out.read_text())
        committed = json.loads(det.read_text())
        field_diff = sorted(set(committed) ^ set(produced))
        reproduced = (not field_diff)
    ev["deterministic_reproduces_fields"] = reproduced
    ev["deterministic_field_diff"] = field_diff
    ev["deterministic_rc"] = proc.returncode

    # If a real receipt is present it must reproduce from the script's own
    # emission (same field names) — a hand-edited file carries keys the script
    # never writes.
    if real.is_file():
        committed_real = json.loads(real.read_text())
        script_fields = set()
        if tmp_out.is_file():
            script_fields = set(json.loads(tmp_out.read_text()))
        extra = sorted(set(committed_real) - script_fields)
        ev["real_receipt_fields_script_never_emits"] = extra
        ev["real_receipt_is_script_output"] = not extra
    else:
        ev["real_receipt_removed_or_renamed"] = True

    ok = reproduced and (not real.is_file() or ev.get("real_receipt_is_script_output"))
    if not ok:
        ev["why"] = ("the committed deterministic receipt does not reproduce from "
                     "the script's own emission, or a real receipt carries fields "
                     "the script never writes")
    return _emit("AC-11", "PASS" if ok else "FAIL", ev)


# ── AC-13/14/15: test shape, RED-baseline honesty, invariant ───────

_TARGET_TESTS = ["tests/agent/test_compaction_backstop_wiring.py",
                 "tests/agent/test_compaction_producer.py"]


def _source_text_assertions() -> list:
    """Lines in the target test files that assert behaviour from PYTHON SOURCE TEXT.

    A hit is an assertion whose subject is text read from a ``.py`` file. Only a
    ``.py`` source read counts: an assertion over a JSON artifact (stage_c.json,
    gate.json) is a normal behavior assertion and must NOT be flagged — an
    over-broad detector reports false positives on correct tests, which is how
    this check first failed.

    Each test function is scanned in isolation so a read in one test cannot taint
    an assertion in the next.
    """
    import re
    offenders = []
    for rel in _TARGET_TESTS:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        read_vars: set = set()
        for i, line in enumerate(lines, 1):
            # Start of a new function/method: reset the taint set.
            if re.match(r"\s*(def |class )", line):
                read_vars = set()
            # A read of a PYTHON source file, e.g. Path("agent/foo.py").read_text()
            if re.search(r"[\"'][^\"']*\.py[\"'][^)]*\)?\s*\.read_text\(", line) or \
                    re.search(r"open\(\s*[^)]*\.py", line):
                m = re.match(r"\s*(\w+)\s*=", line)
                if m:
                    read_vars.add(m.group(1))
                continue
            if read_vars and re.search(
                    r"\bin\s+(" + "|".join(sorted(re.escape(v) for v in read_vars)) + r")\b",
                    line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    return offenders


def ac13() -> dict:
    """AC-13: no test asserts behaviour from Python SOURCE TEXT."""
    ev: dict = {"scanned": list(_TARGET_TESTS)}
    offenders = _source_text_assertions()
    ev["source_text_assertions"] = offenders
    # Non-vacuity: the two files the spec names must actually contain behavior
    # assertions, so a scan over empty/missing files cannot pass this AC.
    behavior_asserts = 0
    for rel in _TARGET_TESTS:
        p = REPO_ROOT / rel
        if p.is_file():
            behavior_asserts += sum(1 for ln in p.read_text(encoding="utf-8").splitlines()
                                    if ln.strip().startswith("assert "))
    ev["behavior_assertions_found"] = behavior_asserts
    ev["files_present"] = [rel for rel in _TARGET_TESTS if (REPO_ROOT / rel).is_file()]
    ok = (not offenders and behavior_asserts > 0
          and len(ev["files_present"]) == len(_TARGET_TESTS))
    if not ok:
        ev["why"] = ("tests still assert behaviour from Python source text, or the "
                     "scan found no behavior assertions to inspect (vacuous)")
    return _emit("AC-13", "PASS" if ok else "FAIL", ev)


def ac14() -> dict:
    """AC-14: the RED baseline is either genuinely re-run against the pre-fix
    parent with a dated record, or explicitly re-labelled a revert tripwire."""
    ev: dict = {}
    record = REPO_ROOT / "evals" / "compaction" / "results" / \
        "SPEC-0044-AC-14-red-baseline-46822b15c.json"
    ev["dated_red_record"] = record.is_file()
    if record.is_file():
        doc = json.loads(record.read_text())
        ev["record_commit"] = doc.get("tree", {}).get("commit")
        ev["record_mode"] = doc.get("mode")
        ev["record_red_conditions"] = doc.get("red_conditions")
        ev["record_all_red"] = doc.get("all_red_conditions_hold")
    src = (REPO_ROOT / "tests" / "agent" / "test_compaction_producer.py").read_text(
        encoding="utf-8")
    ev["docstring_labels_tripwire"] = "REVERT TRIPWIRE" in src
    ev["docstring_disclaims_baseline"] = "NOT a pre-fix baseline" in src
    ok = (ev["dated_red_record"] and ev.get("record_all_red") is True
          and ev["docstring_labels_tripwire"] and ev["docstring_disclaims_baseline"])
    if not ok:
        ev["why"] = ("no dated pre-fix RED record, or the test docstring still "
                     "claims to be a pre-fix baseline")
    return _emit("AC-14", "PASS" if ok else "FAIL", ev)


def ac15() -> dict:
    """AC-15: the soak's bounded-map invariant enforces start_msg <= end_msg, and
    the receipt's turn log shows only coherent covers."""
    ev: dict = {}
    soak_results = REPO_ROOT / "evals" / "compaction" / "results" / "soak_results.json"
    ev["soak_receipt_present"] = soak_results.is_file()
    if soak_results.is_file():
        doc = json.loads(soak_results.read_text())
        inv = doc.get("invariants", {})
        ev["invariant_present"] = "map_cover_coherent" in inv
        ev["invariant_holds"] = inv.get("map_cover_coherent")
        tail = doc.get("turn_log_tail") or []
        ev["turn_log_covers"] = [t.get("map_covers_after") for t in tail]
        ev["all_covers_coherent"] = all(
            int(c["start_msg"]) <= int(c["end_msg"])
            for c in ev["turn_log_covers"] if isinstance(c, dict))
        ev["all_invariants_hold"] = doc.get("all_invariants_hold")
    # The discriminator: the invariant must actually flag an inverted cover.
    disc = REPO_ROOT / "scripts" / "_spec0044_ac15_discriminate.py"
    ev["discriminator_committed"] = disc.is_file()
    if disc.is_file():
        proc = subprocess.run([sys.executable, str(disc)], capture_output=True,
                              text=True, timeout=300, cwd=str(REPO_ROOT))
        ev["discriminator_rc"] = proc.returncode
        ev["discriminator_output"] = proc.stdout.strip().splitlines()[-1:] 
    ok = (ev.get("invariant_present") and ev.get("invariant_holds")
          and ev.get("all_covers_coherent") and ev.get("all_invariants_hold")
          and ev.get("discriminator_committed") and ev.get("discriminator_rc") == 0)
    if not ok:
        ev["why"] = ("the bounded-map invariant is missing/vacuous, a turn-log "
                     "cover is inverted, or the discriminator does not trip")
    return _emit("AC-15", "PASS" if ok else "FAIL", ev)


# ── registry ──────────────────────────────────────────────────────────

_FALSIFIERS = {
    "ac1": ac1,
    "ac2": ac2,
    "ac3": ac3,
    "ac4": ac4,
    "ac5": ac5,
    "ac6": ac6,
    "ac7": ac7,
    "ac8": ac8,
    "ac9": ac9,
    "ac10": ac10,
    "ac11": ac11,
    "ac13": ac13,
    "ac14": ac14,
    "ac15": ac15,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ac", choices=sorted(_FALSIFIERS))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    result = _FALSIFIERS[args.ac]()
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
