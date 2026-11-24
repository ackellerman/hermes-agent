"""Edge cases on single rules of the cache anchor.

Kept apart from ``test_prompt_cache_anchor.py`` because the plan fixes that file at exactly eight
module-level tests (the close check compares the names): each case here pins one neighbour of a
rule the main file exercises as a whole — the tool-cache layout's endpoint budget, a part whose
type is not ``text``, and a ``role: tool`` message on a route that carries no tool part markers.
"""

import copy

import pytest

from agent import prompt_cache_boundary
from agent.prompt_cache_boundary import clear_cache_anchors, register_cache_anchor
from agent.prompt_caching import (
    _count_cache_markers,
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
)

STATIC = "You are Hermes.\n"
SYSTEM = STATIC + "## Tools\n- terminal\n"
ANCHOR = "[ctxpkg references ep-0001..ep-0003]"
TOOLS = [
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
]


@pytest.fixture(autouse=True)
def _isolated_registries():
    """Both registries are process-local globals; no test may inherit either."""
    clear_cache_anchors()
    with prompt_cache_boundary._lock:
        prompt_cache_boundary._prefixes.clear()
    yield
    clear_cache_anchors()
    with prompt_cache_boundary._lock:
        prompt_cache_boundary._prefixes.clear()


def _markers_in(msg):
    """(part index or None for a top-level marker) of every cache breakpoint in one message."""
    found = [None] if "cache_control" in msg else []
    content = msg.get("content")
    if isinstance(content, list):
        found += [i for i, part in enumerate(content)
                  if isinstance(part, dict) and "cache_control" in part]
    return found


def _marked(msg):
    return bool(_markers_in(msg))


def _total(messages, tools=()):
    return _count_cache_markers(messages, list(tools))


def test_tool_layout_endpoints_skip_anchor():
    """The tool layout spends its endpoint slots after the anchor; the anchor's index is skipped."""
    register_cache_anchor(ANCHOR)
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "q"},
               {"role": "assistant", "content": "a"},
               {"role": "user", "content": ANCHOR}]

    plan = build_prompt_cache_plan(copy.deepcopy(history), copy.deepcopy(TOOLS), native_anthropic=True,
                                   direct_native_tool_cache=True)

    # Three slots: the tools marker, the ended turn before the anchor, and the anchor itself.
    assert _marked(plan.messages[2]) and _marked(plan.messages[3])
    assert not _marked(plan.messages[0]) and not _marked(plan.messages[1])
    assert _total(plan.messages, plan.tools) == 3

    # Two ended turns before the anchor: all four slots go out, and the one nearest the anchor
    # is not the one that falls off the window (which is what an unskipped anchor does).
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "q1"},
               {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "q2"},
               {"role": "assistant", "content": "a2"},
               {"role": "user", "content": ANCHOR}]

    plan = build_prompt_cache_plan(copy.deepcopy(history), copy.deepcopy(TOOLS), native_anthropic=True,
                                   direct_native_tool_cache=True)

    assert [_marked(message) for message in plan.messages] == [False, False, True, False, True, True]
    assert _total(plan.messages, plan.tools) == 4


def test_non_text_part_is_not_an_anchor():
    """A part whose type is not ``text`` never joins into an anchor's text, even with the same text."""
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": [{"type": "thinking", "text": ANCHOR}]},
               {"role": "assistant", "content": "Noted."},
               {"role": "user", "content": "tail question"},
               {"role": "assistant", "content": "tail answer"}]

    baseline = apply_anthropic_cache_control(copy.deepcopy(history), native_anthropic=True)
    register_cache_anchor(ANCHOR)
    planned = apply_anthropic_cache_control(copy.deepcopy(history), native_anthropic=True)

    # The thinking part is not a text block: the plan is the no-anchor plan, slot for slot.
    assert not _marked(planned[1])
    assert [_marked(message) for message in planned] == [_marked(message) for message in baseline]
    assert _total(planned) == _total(baseline) == 4
    assert planned[0]["content"] == baseline[0]["content"]


def test_tool_role_message_is_not_an_anchor():
    """On a route without tool part markers a role:tool message carries no breakpoint, so it is no anchor."""
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "q"},
               {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "terminal", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": ANCHOR},
               {"role": "assistant", "content": "a"}]

    baseline = apply_anthropic_cache_control(copy.deepcopy(history), static_system_prefix=STATIC,
                                             tool_part_markers=False)
    # Without an anchor the static prefix splits the system prompt and both parts carry a marker.
    assert len(baseline[0]["content"]) == 2 and len(_markers_in(baseline[0])) == 2
    assert not _marked(baseline[3])
    assert _total(baseline) == 4

    register_cache_anchor(ANCHOR)
    planned = apply_anthropic_cache_control(copy.deepcopy(history), static_system_prefix=STATIC,
                                            tool_part_markers=False)

    # The tool message cannot hold a breakpoint, so it cannot be the anchor either: the system
    # prompt takes its normal two-part split rather than the anchor's single end marker.
    assert len(planned[0]["content"]) == 2, "a role:tool message took the anchor path"
    assert len(_markers_in(planned[0])) == 2
    assert not _marked(planned[3])
    assert _total(planned) == _total(baseline)
