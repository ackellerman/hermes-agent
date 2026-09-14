"""Temporary probe (SPEC-0044 AC-15): show the soak's cover invariant is not vacuous.

Reverts the retire normalization in-process (monkeypatch) and runs a soak-shaped
retire sequence, so the inverted cover is produced and the invariant must trip.
Restores nothing on disk — the source tree is untouched.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.compaction_map import CompactionMap


def _legacy_retire(self, start_msg, end_msg):
    """The PRE-FIX retire: only advances covers.start_msg (the inverted cover)."""
    m = self.load()
    lo, hi = int(start_msg), int(end_msg)
    kept = []
    for ep in m.get("episodes", []):
        s, e = int(ep.get("start_msg", 0)), int(ep.get("end_msg", 0))
        if e <= hi:
            continue
        if s <= hi:
            ep = dict(ep, start_msg=hi + 1)
        kept.append(ep)
    m["episodes"] = kept
    covers = m.get("covers", {})
    covers["start_msg"] = max(int(covers.get("start_msg", 0)), hi + 1)
    m["covers"] = covers
    self.save(m)
    return m


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cm = CompactionMap(Path(tmp), "s")
        # The soak's shape: one episode over a 16-message region, then retire it.
        cm.save({"schema_version": 1, "covers": {"start_msg": 0, "end_msg": 15},
                 "episodes": [{"name": "ep", "start_msg": 0, "end_msg": 15}],
                 "entities": [], "edges": []})

        cm.retire(0, 15)
        fixed = cm.load()["covers"]
        print("FIXED retire covers:", json.dumps(fixed))
        fixed_coherent = not (int(fixed["start_msg"]) > int(fixed["end_msg"]))

        # Same state, reverted retire.
        cm.save({"schema_version": 1, "covers": {"start_msg": 0, "end_msg": 15},
                 "episodes": [{"name": "ep", "start_msg": 0, "end_msg": 15}],
                 "entities": [], "edges": []})
        _legacy_retire(cm, 0, 15)
        old = cm.load()["covers"]
        print("PRE-FIX retire covers:", json.dumps(old))
        old_incoherent = int(old["start_msg"]) > int(old["end_msg"])
        print("invariant_flags_pre_fix:", old_incoherent,
              "| invariant_passes_fixed:", fixed_coherent)
        ok = old_incoherent and fixed_coherent
        print("AC-15 DISCRIMINATES:", ok)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
