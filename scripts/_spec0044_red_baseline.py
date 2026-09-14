"""SPEC-0044 AC-14(a) RED-baseline probe — pre-fix parent 46822b15c.

Runs the SAME two behaviours the fixed tree's tests assert, against the pre-fix
code, and writes a dated RED record. Run it with a python whose sys.path resolves
the PRE-FIX tree (the /tmp/spec44-base worktree), e.g.:

    cd <pre-fix worktree> && python scripts/_spec0044_red_baseline.py OUT.json

It must FAIL (the pre-fix tree lacks the directory-per-dump layout and the
AC-21 dump-before-degrade wiring), which is what makes the fixed tree's green a
real improvement rather than a tautology.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "<unknown>"


def probe_directory_per_dump_layout(tmp: Path) -> dict:
    """AC-1 pre-fix condition: after write_dump, is there a per-dump DIRECTORY?"""
    from agent.compaction_dump import DumpStore
    store = DumpStore(tmp)
    ref = store.write_dump("sess", [{"role": "user", "content": "hello"}],
                           start_msg=0, end_msg=0, turn=1)
    sdir = store.session_dir("sess")
    dirs = [p.name for p in sdir.iterdir() if p.is_dir()]
    flat = [p.name for p in sdir.iterdir() if p.is_file()]
    return {"dump_id": ref.dump_id, "per_dump_dirs": dirs, "flat_siblings": flat,
            "has_per_dump_dir": ref.dump_id in dirs,
            "journal_inside_dir": (sdir / ref.dump_id / f"{ref.dump_id}.jsonl").is_file()}


def probe_region_extractor_fabrication(tmp: Path) -> dict:
    """AC-1 pre-fix condition: does RegionExtractor CREATE the region dir?"""
    from agent.compaction_extract import RegionExtractor
    target = tmp / "sess" / "never-produced"
    created = False
    raised = None
    try:
        RegionExtractor(tmp, "sess", "never-produced")
        created = target.is_dir()
    except FileNotFoundError as exc:
        raised = f"FileNotFoundError: {exc}"
    return {"extractor_created_dir": created, "extractor_raised": raised,
            "asserts_instead_of_fabricating": raised is not None}


def probe_dump_before_degrade(tmp: Path) -> dict:
    """AC-21 pre-fix condition: does the degrade branch write a dump + enqueue?"""
    from types import SimpleNamespace
    from agent.compaction_backstop import CompactionBackstop
    from agent.compaction_dump import DumpStore

    a = SimpleNamespace()
    a.session_id = "sess"
    a.db = None
    a.compaction_pipeline_enabled = True
    a.compaction_pipeline_storage_root = str(tmp)
    a.compaction_pipeline_map_idle_after_seconds = 0.0
    a.compaction_pipeline_map_cooldown_seconds = 0.0
    a.compaction_pipeline_extraction_cooldown_seconds = 0.0
    a.compaction_pipeline_max_stage_retries = 2
    a.compaction_pipeline_budget_per_session_tokens = 200000
    a.compaction_pipeline_models = {}
    a._compaction_models_reachable = False
    a.provider = "ollama"
    a.model = "muse-glimmer:latest"
    a.aux_runtime = {"provider": "ollama"}
    a.context_compressor = SimpleNamespace(last_compress_window=(2, 5),
                                           _compress_window=lambda m: (2, 5))
    messages = [{"role": "assistant" if i % 2 else "user", "content": f"m{i}"}
                for i in range(8)]
    action, swapped, tel = CompactionBackstop(a).decide_and_swap(messages)
    # Read discovery the pre-fix way (flat), tolerating either layout.
    store = DumpStore(tmp)
    sdir = store.session_dir("sess")
    dumped_flat = sorted(p.name for p in sdir.iterdir() if p.name.endswith(".jsonl")) \
        if sdir.is_dir() else []
    dumped_dirs = store.dump_ids("sess") if hasattr(store, "dump_ids") else []
    qpath = sdir / "pipeline_queue.json"
    rows = json.loads(qpath.read_text()) if qpath.is_file() else []
    return {"action": action, "degraded": tel.get("degraded"),
            "flat_journals": dumped_flat, "dump_dirs": dumped_dirs,
            "queue_rows": len(rows),
            "dumped_before_degrade": bool(dumped_flat or dumped_dirs),
            "enqueued": bool(rows)}


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    expect_fixed = "--expect-fixed" in sys.argv
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    record: dict = {
        "spec": "SPEC-0044",
        "ac": "AC-14(a) — RED baseline against the pre-fix parent commit",
        "purpose": ("Re-run the fixed tree's two behaviours against the PRE-FIX code. "
                    "Every probe below must report the RED condition; a probe that "
                    "shows the fixed behaviour here would mean the fix was already "
                    "present and the fixed-tree green proves nothing."),
        "tree": {"commit": _sha(), "path": str(REPO)},
    }
    with tempfile.TemporaryDirectory(prefix="spec44-red-") as tmp:
        tmp_p = Path(tmp)
        for name, fn in (("AC-1/per-dump-layout", probe_directory_per_dump_layout),
                         ("AC-1/extractor-fabrication", probe_region_extractor_fabrication),
                         ("AC-21/dump-before-degrade", probe_dump_before_degrade)):
            sub = tmp_p / name.replace("/", "_")
            sub.mkdir(parents=True, exist_ok=True)
            try:
                record[name] = fn(sub)
            except Exception as exc:  # noqa: BLE001 — a pre-fix crash IS a RED result
                record[name] = {"raised": f"{type(exc).__name__}: {exc}"}

    layout = record.get("AC-1/per-dump-layout", {})
    fabric = record.get("AC-1/extractor-fabrication", {})
    degrade = record.get("AC-21/dump-before-degrade", {})
    record["red_conditions"] = {
        "no_per_dump_directory": layout.get("has_per_dump_dir") is False,
        "journal_not_inside_dir": layout.get("journal_inside_dir") is False,
        "extractor_fabricated_the_dir": fabric.get("extractor_created_dir") is True,
        "degrade_did_not_dump": degrade.get("dumped_before_degrade") is False,
    }
    record["all_red_conditions_hold"] = all(record["red_conditions"].values())
    record["mode"] = "fixed-tree-control" if expect_fixed else "pre-fix-red-baseline"
    text = json.dumps(record, indent=2, default=str)
    print(text)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    if expect_fixed:
        # On the fixed tree EVERY red condition must be absent, which is what
        # proves the probes discriminate rather than always reporting RED.
        return 0 if not any(record["red_conditions"].values()) else 1
    return 0 if record["all_red_conditions_hold"] else 1


if __name__ == "__main__":
    sys.exit(main())
