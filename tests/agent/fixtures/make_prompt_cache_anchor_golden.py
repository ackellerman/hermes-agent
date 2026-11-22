"""Generate ``prompt_cache_anchor_golden.json`` from the BASE cache planner.

Run this in the worktree BEFORE any edit to ``agent/prompt_caching.py`` (or
``agent/prompt_cache_boundary.py``)::

    python3 tests/agent/fixtures/make_prompt_cache_anchor_golden.py

It pins the planner's request-local output over the plan's five shapes — plain,
tool runs, a skill scaffold, a system prompt with a static prefix, an ephemeral
tail — plus the direct native tool-cache layout of the tool-runs history.

``test_no_anchor_output_unchanged`` replays every case with the anchor registry
empty and requires the output to equal the committed golden exactly, so a cache
anchor can never move the bytes of a request that carries none.
"""

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.prompt_cache_boundary import register_stable_prefix
from agent.prompt_caching import (
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "prompt_cache_anchor_golden.json"

STATIC = "You are Hermes.\n"
SESSION = STATIC + "## Skills\n- alpha: does a thing\n"
EPHEMERAL = "\n\nYou are a focused subagent.\nCONTEXT:\ntask-specific text"
SCAFFOLD = "Stable skill scaffold: inspect the report and preserve the instructions.\n\n"
TOOLS = [
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
]


def _plain_history():
    return [
        {"role": "system", "content": "You are Hermes, an agent."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def _tool_history():
    return [
        {"role": "system", "content": "You are Hermes, an agent."},
        {"role": "user", "content": "look at the tree"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "alpha\nbeta"},
        {"role": "tool", "tool_call_id": "call_1", "content": "gamma"},
        {"role": "assistant", "content": "three entries"},
        {"role": "user", "content": "and now the rest"},
    ]


def _scaffold_history():
    return [
        {"role": "system", "content": "You are Hermes, an agent."},
        {"role": "user", "content": SCAFFOLD + "ticket=one time=10:00"},
        {"role": "assistant", "content": "working on ticket one"},
        {"role": "user", "content": "continue"},
    ]


def _static_prefix_history():
    return [
        {"role": "system", "content": STATIC + "## Tools\n- terminal\n- read_file\n"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def _ephemeral_tail_history():
    return [
        {"role": "system", "content": SESSION + EPHEMERAL},
        {"role": "user", "content": "goal"},
    ]


CASES = [
    {"name": "plain", "entry": "messages", "stable_prefix": None, "kwargs": {},
     "messages": _plain_history()},
    {"name": "tool_runs", "entry": "messages", "stable_prefix": None,
     "kwargs": {"native_anthropic": True}, "messages": _tool_history()},
    {"name": "tool_runs/tool-cache-plan", "entry": "plan", "stable_prefix": None,
     "kwargs": {"native_anthropic": True, "direct_native_tool_cache": True},
     "messages": _tool_history(), "tools": TOOLS},
    {"name": "skill_scaffold", "entry": "messages", "stable_prefix": SCAFFOLD, "kwargs": {},
     "messages": _scaffold_history()},
    {"name": "static_prefix", "entry": "messages", "stable_prefix": None,
     "kwargs": {"static_system_prefix": STATIC}, "messages": _static_prefix_history()},
    {"name": "ephemeral_tail", "entry": "messages", "stable_prefix": None,
     "kwargs": {"static_system_prefix": STATIC, "session_system_prompt": SESSION},
     "messages": _ephemeral_tail_history()},
]


def main() -> int:
    golden = {
        "generated_by": "tests/agent/fixtures/make_prompt_cache_anchor_golden.py",
        "note": "base apply_anthropic_cache_control / build_prompt_cache_plan output, no anchor registered",
        "cases": [],
    }
    for case in CASES:
        if case["stable_prefix"]:
            register_stable_prefix(case["stable_prefix"])
        messages = copy.deepcopy(case["messages"])
        if case["entry"] == "plan":
            plan = build_prompt_cache_plan(messages, copy.deepcopy(case["tools"]), **case["kwargs"])
            output = {"messages": plan.messages, "tools": plan.tools}
        else:
            output = apply_anthropic_cache_control(messages, **case["kwargs"])
        golden["cases"].append({**case, "output": output})
        print(f"case {case['name']}: {len(json.dumps(output))} chars")
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=False) + "\n")
    print(f"wrote {GOLDEN_PATH.relative_to(REPO_ROOT)} cases={len(golden['cases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
