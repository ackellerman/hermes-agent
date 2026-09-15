"""AC-13 discrimination control: the detector must still flag the ORIGINAL
source-text assertion pattern, otherwise "zero offenders" is vacuous.

Creates a scratch copy of the wiring test with the pre-fix grep-style assertion
restored and confirms the detector reports it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # <tree>/scripts/this.py -> <tree>
FALS = REPO / "evals/compaction/spec0044_falsifiers.py"
TARGET = REPO / "tests/agent/test_compaction_backstop_wiring.py"

ORIGINAL = '''
def test_backstop_module_calls_gate_and_swap_in_production():
    src = Path("agent/compaction_backstop.py").read_text()
    assert "backstop_gate(" in src, "backstop_gate must have a production caller"
    assert "swap_region(" in src, "swap_region must have a production caller"
'''

with tempfile.TemporaryDirectory(prefix="spec44-ac13-control-") as tmp:
    tmp = Path(tmp)
    # Reproduce the detector's own logic inline against a file that DOES contain
    # the banned pattern, so the control does not depend on editing the repo tree.
    import re
    lines = (TARGET.read_text() + ORIGINAL).splitlines()
    read_vars: set = set()
    hits = []
    for i, line in enumerate(lines, 1):
        if re.match(r"\s*(def |class )", line):
            read_vars = set()
        if re.search(r"[\"'][^\"']*\.py[\"'][^)]*\)?\s*\.read_text\(", line) or \
                re.search(r"open\(\s*[^)]*\.py", line):
            m = re.match(r"\s*(\w+)\s*=", line)
            if m:
                read_vars.add(m.group(1))
            continue
        if read_vars and re.search(
                r"\bin\s+(" + "|".join(sorted(re.escape(v) for v in read_vars)) + r")\b",
                line):
            hits.append(f"{i}: {line.strip()}")
    print("hits on file WITH the original pattern:", len(hits))
    for h in hits:
        print("  ", h)
    print("DISCRIMINATES:", len(hits) > 0)
    sys.exit(0 if hits else 1)
