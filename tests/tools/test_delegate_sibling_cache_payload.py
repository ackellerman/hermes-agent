"""What two sibling subagents put on the wire, on a route that takes cache breakpoints.

Two children of one parent get different goals and different contexts. On a marker route (an OpenAI-wire relay
with ``prompt_caching: true``) each child's first request is recorded and compared block by block: how many
leading system blocks are byte-identical, whether the tools are, where the per-task text first appears and
where the ``cache_control`` breakpoints sit. The block that holds the skills index has to end at a breakpoint
both siblings share, or every child writes the index again.

The measured numbers are recorded as the ``sibling_metrics`` test property (``--junitxml=<file>`` writes it
out), so the same file gives the before/after reading on a tree without the change, where its assertions fail.
"""

import json
import os
from pathlib import Path

TASKS = [
    {"goal": "GOAL-ALPHA count the python files", "context": "CONTEXT-ALPHA the parent is auditing numbers."},
    {"goal": "GOAL-BETA find the flaky test", "context": "CONTEXT-BETA the parent is fixing the gateway."},
]
SKILL_MARK = "sibling-probe-skill"
MAX_BREAKPOINTS = 4  # Anthropic's limit per request


def _blocks(message):
    """A message's content as (text, has cache_control) blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return [(content, False)]
    return [(p.get("text", ""), "cache_control" in p) for p in content or [] if isinstance(p, dict)]


def sibling_metrics(first: dict, second: dict) -> dict:
    """Compare two sibling first requests. Blocks are listed in wire order: every system block, then every
    block of each later message."""
    def flat(req):
        return [(i, j, text, marked) for i, m in enumerate(req["messages"]) for j, (text, marked) in enumerate(_blocks(m))]

    one, two = flat(first), flat(second)
    system_one = [b for b in one if first["messages"][b[0]]["role"] == "system"]
    system_two = [b for b in two if second["messages"][b[0]]["role"] == "system"]
    shared_system = 0
    for a, b in zip(system_one, system_two):
        if a[2] != b[2]:
            break
        shared_system += 1
    shared_all = 0
    for a, b in zip(one, two):
        if a[2] != b[2]:
            break
        shared_all += 1
    text_one, text_two = "".join(b[2] for b in system_one), "".join(b[2] for b in system_two)
    common = next((k for k, (x, y) in enumerate(zip(text_one, text_two)) if x != y), min(len(text_one), len(text_two)))
    task_marks = [t[key].split()[0] for t in TASKS for key in ("goal", "context")]
    first_task_block = next((k for k, b in enumerate(one) if any(mark in b[2] for mark in task_marks)), None)
    skills_block = next((k for k, b in enumerate(one) if SKILL_MARK in b[2]), None)
    markers = [k for k, b in enumerate(one) if b[3]]
    return {
        "system_blocks": len(system_one),
        "leading_identical_system_blocks": shared_system,
        "leading_identical_blocks_incl_messages": shared_all,
        "identical_system_chars": common, "system_chars": [len(text_one), len(text_two)],
        "tools_identical": first.get("tools") == second.get("tools"), "tools": len(first.get("tools") or []),
        "first_block_with_task_text": first_task_block,
        "task_text_in_system": first_task_block is not None and first_task_block < len(system_one),
        "skills_index_block": skills_block,
        "skills_block_ends_at_breakpoint": skills_block is not None and one[skills_block][3],
        "skills_block_shared": skills_block is not None and skills_block < shared_all,
        "breakpoint_blocks": markers, "breakpoints": [json.dumps(r).count('"cache_control"') for r in (first, second)],
        "breakpoints_inside_shared_prefix": sum(1 for k in markers if k < shared_all),
    }


def _sibling_first_requests():
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
    from tools.delegate_tool import delegate_task

    home = Path(os.environ["HERMES_HOME"])
    skill = home / "skills" / "probe" / SKILL_MARK
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {SKILL_MARK}\ndescription: Use when probing the skills index.\n---\n\n# Probe\n")
    with FakeLLMServer(lambda _req: Text("done")) as srv:
        route = (f"custom_providers:\n  - name: cacherelay\n    base_url: {srv.base_url}\n    api_key: sk-fake-e2e\n"
                 "    models:\n      fake-model:\n        prompt_caching: true\n")
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW + route
                          + "prompt_caching:\n  cache_ttl: auto\n")
        parent = AIAgent(provider="custom:cacherelay", base_url=srv.base_url, api_key="sk-fake-e2e", model="fake-model",
                         session_db=SessionDB(db_path=home / "state.db"), session_id="parent", quiet_mode=True,
                         platform="cli", enabled_toolsets=["terminal", "file", "skills", "delegation"])
        parent._ensure_db_session()
        assert parent._use_prompt_caching, "the marker route is not active"
        out = json.loads(delegate_task(tasks=[dict(t) for t in TASKS], parent_agent=parent))
        parent_ttl = parent._cache_ttl
        parent.close()
        requests = srv.main_requests()
    assert len(out["results"]) == 2 and len(requests) == 2, (out, len(requests))
    by_goal = {goal: next(r for r in requests if goal in json.dumps(r)) for goal in ("GOAL-ALPHA", "GOAL-BETA")}
    return by_goal["GOAL-ALPHA"], by_goal["GOAL-BETA"], parent_ttl


def test_siblings_share_every_block_before_their_task_text_and_the_skills_index_ends_at_a_breakpoint(record_property):
    first, second, parent_ttl = _sibling_first_requests()
    metrics = sibling_metrics(first, second)
    ttls = sorted({part["cache_control"].get("ttl", "5m") for req in (first, second) for msg in req["messages"]
                   if isinstance(msg.get("content"), list) for part in msg["content"] if "cache_control" in part})
    metrics["child_cache_ttls"], metrics["parent_cache_ttl"] = ttls, parent_ttl
    record_property("sibling_metrics", json.dumps(metrics, sort_keys=True))

    assert metrics["skills_index_block"] is not None, "the probe skill is not in the child's prompt"
    # 1. The siblings share the prefix up to the per-task text: every system block and the tools.
    assert metrics["leading_identical_system_blocks"] == metrics["system_blocks"], metrics
    assert metrics["tools_identical"] and not metrics["task_text_in_system"], metrics
    assert metrics["first_block_with_task_text"] == metrics["leading_identical_blocks_incl_messages"], metrics
    # 2. The skills index block is shared and ends at a breakpoint, so the second child reads it.
    assert metrics["skills_block_shared"] and metrics["skills_block_ends_at_breakpoint"], metrics
    # 3. No request spends more breakpoints than the provider allows.
    assert max(metrics["breakpoints"]) <= MAX_BREAKPOINTS, metrics
    # A child stays on the 5m tier under ``cache_ttl: auto`` while its interactive parent is on 1h.
    assert parent_ttl == "1h" and ttls == ["5m"], (parent_ttl, ttls)
