#!/usr/bin/env python3
"""SPEC-0042 AC-19c model-sizing probe: measures per-stage input token sizes on
the maximal fixture class (500-message region, heavy tool use) against the
model-table slots.

Verdict rules (spec §3.7): map-update input must fit the llama-small 64K slot
with >= 20% headroom; Stage B/C inputs must fit the qwen3.8-v3 256K slot with
>= 20% headroom. Either measurement failing means the model-table defaults for
that stage must be revised before merge — the measured numbers gate, not the
projected O(map+window) argument.

Run: python evals/compaction/sizing_probe.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAP_SLOT_TOKENS = 64_000        # llama-small 64K slot
BC_SLOT_TOKENS = 256_000       # qwen3.8-v3 256K slot
HEADROOM = 0.20                # >= 20% headroom required (AC-19c)
REGION_MESSAGES = 500         # maximal fixture class (AC-1 class)
CHUNK_MESSAGES = 25            # idle-cadence chunk (20s idle at ~1-2 msg/s)


def build_maximal_region(n: int = REGION_MESSAGES) -> list:
    """500-message session with heavy tool use: alternating user / assistant
    tool-call / tool-result triples with realistic payload sizes."""
    msgs = []
    for i in range(n):
        phase = i % 3
        if phase == 0:
            msgs.append({"role": "user", "content": (
                f"Task {i}: refactor module {i % 40} and run its tests. "
                + "context " * 40)})
        elif phase == 1:
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{i}", "function": {
                             "name": "terminal", "arguments": json.dumps({
                                 "command": f"pytest tests/module_{i % 40}.py -x"})}}]})
        else:
            msgs.append({"role": "tool", "tool_call_id": f"c{i-1}",
                         "content": ("PASSED module tests\n" + "output line\n" * 60)})
    return msgs


def approx_tokens(obj) -> int:
    """Same rough estimator family as DumpStore.estimate_tokens (chars/4)."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return len(text) // 4


def measure():
    region = build_maximal_region()
    region_tokens = approx_tokens(region)
    results = {"region_messages": len(region), "region_tokens_estimate": region_tokens}

    # ── Map update input: current map (realistic episode count, retired
    # entries removed -> O(live)) + one idle-cadence chunk ──
    realistic_map = {
        "schema_version": 1,
        "covers": {"start_msg": 200, "end_msg": 499},
        "episodes": [
            {"name": f"module {j} refactor and tests", "start_msg": 200 + j * 6,
             "end_msg": 205 + j * 6, "state": "complete"}
            for j in range(50)
        ],
        "entities": [f"module_{j}" for j in range(40)],
        "edges": [{"from": [200 + j * 6, 205 + j * 6],
                   "to": [206 + j * 6, 211 + j * 6],
                   "kind": "depends-on"} for j in range(40)],
    }
    chunk = region[REGION_MESSAGES - CHUNK_MESSAGES:]
    map_input = approx_tokens(realistic_map) + approx_tokens(chunk)
    results["map_update_input_tokens"] = int(map_input)
    results["map_slot_tokens"] = MAP_SLOT_TOKENS
    results["map_headroom_pct"] = round(100 * (1 - map_input / MAP_SLOT_TOKENS), 1)
    results["map_pass"] = bool(map_input <= MAP_SLOT_TOKENS * (1 - HEADROOM))

    # ── Stage B input: sliced map + targeted re-read (sampled dump excerpts) ──
    stage_b_input = approx_tokens(realistic_map) + approx_tokens(chunk) * 2
    results["stage_b_input_tokens"] = int(stage_b_input)

    # ── Stage C input: map + B's structured verdict list (smaller than map) ──
    stage_b_output = {"items": [
        {"map_ref": ep["name"], "verdict": "keep", "because": "dep",
         "cites": [[ep["start_msg"], ep["end_msg"]]]}
        for ep in realistic_map["episodes"]]}
    stage_c_input = approx_tokens(realistic_map) + approx_tokens(stage_b_output)
    results["stage_c_input_tokens"] = int(stage_c_input)
    results["bc_slot_tokens"] = BC_SLOT_TOKENS
    results["bc_headroom_pct"] = round(
        100 * (1 - max(stage_b_input, stage_c_input) / BC_SLOT_TOKENS), 1)
    results["bc_pass"] = bool(max(stage_b_input, stage_c_input) <= BC_SLOT_TOKENS * (1 - HEADROOM))

    results["ac19c_pass"] = bool(results["map_pass"] and results["bc_pass"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="optional path to write the results JSON")
    args = parser.parse_args()
    results = measure()
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text)
    return 0 if results["ac19c_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())