"""A cache anchor for a context engine's stable block.

A compaction rewrites every byte after the head, so the old prefix is read from cache only
if a breakpoint sits at its end. The engine registers the text of its newest reference block
here and the planner spends one of its four breakpoints on that message, whatever else moves
after it. With no anchor registered every request must plan exactly as before — that is what
``prompt_cache_anchor_golden.json`` (generated at base) pins.

Anchor matching is exact text: a string content, or the joined text of plain text parts. A
message carrying any non-text part (an image, a tool_result envelope) is never an anchor.
"""

import copy
import json
from pathlib import Path

import pytest

from agent import prompt_cache_boundary
from agent.prompt_cache_boundary import (
    clear_cache_anchors,
    is_cache_anchor,
    register_cache_anchor,
)
from agent.prompt_caching import (
    _count_cache_markers,
    apply_anthropic_cache_control,
    build_prompt_cache_plan,
    strip_anthropic_cache_control,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "prompt_cache_anchor_golden.json"

STATIC = "You are Hermes.\n"
SYSTEM = STATIC + "## Tools\n- terminal\n"
SESSION = SYSTEM + "## Skills\n- alpha: does a thing\n"
EPHEMERAL_TAIL = SESSION + "\n\nYou are a focused subagent.\nCONTEXT:\ntask text"
ANCHORS = (
    "[ctxpkg references ep-0001..ep-0003]",
    "[ctxpkg references ep-0004..ep-0006]",
    "[ctxpkg references ep-0007..ep-0009]",
)
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


def _history(*, anchor_count=0, anchor_last=False):
    """system, one user message per anchor (answered), then two ordinary turns.

    With ``anchor_last`` the list ends ON the newest anchor's user message and the tail is
    dropped, so the anchor is the last message the planner sees.
    """
    messages = [{"role": "system", "content": SYSTEM}]
    for index in range(anchor_count):
        messages.append({"role": "user", "content": ANCHORS[index]})
        if not (anchor_last and index == anchor_count - 1):
            messages.append({"role": "assistant", "content": "Noted."})
    if not anchor_last:
        messages.append({"role": "user", "content": "tail question"})
        messages.append({"role": "assistant", "content": "tail answer"})
    return messages


def test_no_anchor_output_unchanged():
    """Every shape the plan names, replayed with no anchor registered, equals the base output."""
    golden = json.loads(GOLDEN.read_text())
    names = {case["name"] for case in golden["cases"]}
    assert {"plain", "tool_runs", "skill_scaffold", "static_prefix", "ephemeral_tail"} <= names

    for case in golden["cases"]:
        if case["stable_prefix"]:
            prompt_cache_boundary.register_stable_prefix(case["stable_prefix"])
        messages = copy.deepcopy(case["messages"])
        if case["entry"] == "plan":
            plan = build_prompt_cache_plan(messages, copy.deepcopy(case["tools"]), **case["kwargs"])
            got = {"messages": plan.messages, "tools": plan.tools}
        else:
            assert case["entry"] == "messages"
            got = apply_anthropic_cache_control(messages, **case["kwargs"])
        assert json.dumps(got, sort_keys=True) == json.dumps(case["output"], sort_keys=True), case["name"]


def test_anchor_message_marked():
    """The newest anchor's own message takes a breakpoint, and the tail keeps its own."""
    register_cache_anchor(ANCHORS[0])
    history = _history(anchor_count=1)

    planned = apply_anthropic_cache_control(copy.deepcopy(history), native_anthropic=True)

    assert _marked(planned[1]), "the anchor message carries no breakpoint"
    assert planned[1]["content"][0]["text"] == ANCHORS[0]
    # A breakpoint at its end, not at the start of its content: the block text is the prefix.
    assert planned[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # The tail still gets the spare breakpoints, so this is not a prefix-only plan.
    assert _marked(planned[-1]) and _marked(planned[-2])
    assert _total(planned) == 4
    # The system prompt is still marked, at its end.
    assert _marked(planned[0])

    # Same text, but the message carries a non-text part: not an anchor (its joined text is
    # only the text parts, and marking around an image block would cache the wrong bytes).
    mixed = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": [{"type": "text", "text": ANCHORS[0]},
                                          {"type": "image", "source": {"type": "base64", "data": "AAAA"}}]},
             {"role": "assistant", "content": "Noted."},
             {"role": "user", "content": "tail question"},
             {"role": "assistant", "content": "tail answer"}]
    planned = apply_anthropic_cache_control(copy.deepcopy(mixed), native_anthropic=True)
    assert not _marked(planned[1])
    assert _total(planned) == 4


def test_last_anchor_wins():
    """Two registered blocks in history: only the LAST by position is marked.

    The registration order is the reverse of the message order, so a planner that marks the
    most recently registered anchor, or every anchor, fails here.
    """
    register_cache_anchor(ANCHORS[1])
    register_cache_anchor(ANCHORS[0])

    planned = apply_anthropic_cache_control(copy.deepcopy(_history(anchor_count=2)), native_anthropic=True)

    # history: 0 system, 1 anchor A, 2 ack, 3 anchor B, 4 ack, 5 tail question, 6 tail answer
    assert planned[3]["content"][0]["text"] == ANCHORS[1]
    assert _marked(planned[3])
    assert not _marked(planned[1]), "the older anchor was marked too"
    assert _total(planned) == 4
    assert sum(1 for message in planned if _marked(message)) == 4


def test_never_more_than_four_markers():
    """0-3 anchors x static prefix x both layouts x anchor-is-last: never 5, never a double."""
    for anchor_count in range(4):
        for index in range(anchor_count):
            register_cache_anchor(ANCHORS[index])
        for static_prefix in (False, True):
            prefix_kwargs: dict = {"static_system_prefix": STATIC} if static_prefix else {}
            for anchor_last in (False, True):
                for native in (True, False):
                    history = _history(anchor_count=anchor_count, anchor_last=anchor_last)
                    label = (f"anchors={anchor_count} prefix={static_prefix} last={anchor_last} "
                             f"native={native}")

                    messages = apply_anthropic_cache_control(
                        copy.deepcopy(history), native_anthropic=native, **prefix_kwargs)
                    assert _total(messages) <= 4, label
                    # The system prompt may split into up to three parts, each marked; what must
                    # never happen is one non-system message taking two markers.
                    assert all(len(_markers_in(message)) <= 1 for message in messages
                               if message.get("role") != "system"), label
                    if anchor_count and anchor_last:
                        # Nothing cacheable follows the anchor, so the plan spends exactly two
                        # breakpoints (system end, block). A tail pass that does not skip the
                        # anchor's own index spends two more on the ack before it.
                        assert _total(messages) == 2, label

                    plan = build_prompt_cache_plan(
                        copy.deepcopy(history), copy.deepcopy(TOOLS), native_anthropic=True,
                        direct_native_tool_cache=True, **prefix_kwargs)
                    assert _total(plan.messages, plan.tools) <= 4, label
                    assert all(len(_markers_in(message)) <= 1 for message in plan.messages
                               if message.get("role") != "system"), label
                    assert all(len(_markers_in(tool)) <= 1 for tool in plan.tools), label


def test_system_single_end_marker_with_anchor():
    """With an anchor the system prompt is one block with one marker at its end, never split.

    The ephemeral-tail shape would otherwise spend three breakpoints inside the system prompt
    (static tier, session-prompt end, ephemeral tail) and leave one for the block.
    """
    history = [{"role": "system", "content": EPHEMERAL_TAIL},
               {"role": "user", "content": ANCHORS[0]}]
    kwargs = {"static_system_prefix": STATIC, "session_system_prompt": SESSION}

    # Without an anchor this first-call shape splits the system prompt into three marked parts.
    before = apply_anthropic_cache_control(copy.deepcopy(history), **kwargs)
    assert len(before[0]["content"]) == 3 and len(_markers_in(before[0])) == 3

    register_cache_anchor(ANCHORS[0])
    planned = apply_anthropic_cache_control(copy.deepcopy(history), **kwargs)

    parts = planned[0]["content"]
    assert len(parts) == 1, parts
    assert parts[0]["text"] == EPHEMERAL_TAIL
    assert parts[0]["cache_control"] == {"type": "ephemeral"}
    assert _marked(planned[1])
    assert _total(planned) == 2


def test_tool_cache_layout_anchor():
    """The tool-cache layout: the anchor marker replaces the system-side one, tools stay."""
    register_cache_anchor(ANCHORS[0])
    history = _history(anchor_count=1)

    plan = build_prompt_cache_plan(copy.deepcopy(history), copy.deepcopy(TOOLS), native_anthropic=True,
                                   static_system_prefix=STATIC, direct_native_tool_cache=True)

    # Without an anchor the static prefix would carry this slot; the anchor takes it instead.
    assert not _marked(plan.messages[0])
    assert plan.messages[0]["content"] == history[0]["content"]
    assert _marked(plan.messages[1])
    assert plan.messages[1]["content"][0]["text"] == ANCHORS[0]
    # The tools array marker and the last two end-of-turn markers stay: the four slots are
    # anchor, tools, the ack that closes the anchor's turn, and the last assistant turn.
    assert _marked(plan.tools[-1]) and _marked(plan.tools[0]) is False
    assert _marked(plan.messages[2]) and _marked(plan.messages[-1])
    assert not _marked(plan.messages[3])
    assert _total(plan.messages, plan.tools) == 4


def test_strip_restores_anchor_bytes():
    """A failover strip gives an anchor message back its exact bytes, string or text parts."""
    register_cache_anchor(ANCHORS[0])

    as_string = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ANCHORS[0]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "tail question"},
        {"role": "assistant", "content": "tail answer"},
    ]
    planned = apply_anthropic_cache_control(copy.deepcopy(as_string), native_anthropic=True)
    assert _marked(planned[1])
    stripped = strip_anthropic_cache_control(planned)
    assert stripped == as_string
    assert stripped[1]["content"] == ANCHORS[0]

    # The anchor may also be the joined text of plain-text parts; the strip must not rejoin it.
    register_cache_anchor(ANCHORS[1])
    as_parts = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": "[ctxpkg references ep-0004"},
                                     {"type": "text", "text": "..ep-0006]"}]},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "tail question"},
        {"role": "assistant", "content": "tail answer"},
    ]
    assert "".join(part["text"] for part in as_parts[1]["content"]) == ANCHORS[1]

    planned = apply_anthropic_cache_control(copy.deepcopy(as_parts), native_anthropic=True)
    assert _markers_in(planned[1]) == [1]

    stripped = strip_anthropic_cache_control(planned)
    assert stripped == as_parts
    assert isinstance(stripped[1]["content"], list) and len(stripped[1]["content"]) == 2

    # Planning is copy-on-write in both layouts, for both shapes: the engine keeps its own
    # message objects in history, and a breakpoint leaked into them would rewrite the stored
    # prefix (the whole point of the anchor is that those bytes never change).
    for stored in (copy.deepcopy(as_string), copy.deepcopy(as_parts)):
        before = copy.deepcopy(stored)
        assert _marked(apply_anthropic_cache_control(stored, native_anthropic=True)[1])
        assert stored == before, "the message layout wrote a breakpoint into the caller's messages"
        plan = build_prompt_cache_plan(stored, copy.deepcopy(TOOLS), native_anthropic=True,
                                       direct_native_tool_cache=True)
        assert _marked(plan.messages[1])
        assert stored == before, "the tool-cache layout wrote a breakpoint into the caller's messages"


def test_registry_bounded():
    """At most 64 anchors: 64 kept, the 65th evicts the oldest, a lookup keeps one warm."""
    limit = 64
    for index in range(limit):
        register_cache_anchor(f"[ctxpkg references ep-{index:04d}..ep-{index:04d}]")
    assert all(is_cache_anchor(f"[ctxpkg references ep-{index:04d}..ep-{index:04d}]")
               for index in range(limit))

    # A hit refreshes the entry, so the oldest registered one is no longer the first out.
    assert is_cache_anchor("[ctxpkg references ep-0000..ep-0000]")
    register_cache_anchor("[ctxpkg references ep-0064..ep-0064]")

    assert is_cache_anchor("[ctxpkg references ep-0064..ep-0064]")
    assert is_cache_anchor("[ctxpkg references ep-0000..ep-0000]"), "the refreshed entry was evicted"
    assert not is_cache_anchor("[ctxpkg references ep-0001..ep-0001]"), "the 65th did not evict the oldest"
    assert not is_cache_anchor("[ctxpkg references ep-9999..ep-9999]")
    assert not is_cache_anchor("")

    clear_cache_anchors()
    assert not any(is_cache_anchor(f"[ctxpkg references ep-{index:04d}..ep-{index:04d}]")
                   for index in range(limit))

    # A re-registration refreshes the entry's position: a block a later compaction emits again
    # is the newest, not the one next to fall out.
    clear_cache_anchors()
    a_text = "[ctxpkg references ep-0100..ep-0100]"
    b_text = "[ctxpkg references ep-0101..ep-0101]"
    register_cache_anchor(a_text)
    register_cache_anchor(b_text)
    register_cache_anchor(a_text)
    for index in range(63):
        register_cache_anchor(f"[ctxpkg references ep-{200 + index:04d}..ep-{200 + index:04d}]")
    assert is_cache_anchor(a_text), "re-registering an anchor did not refresh its position"
    assert not is_cache_anchor(b_text), "the stale oldest entry survived the 65th registration"

    # An empty text is nobody's message text: it must never match, and, registering it first or
    # last, it must never hold a slot in the cap.
    clear_cache_anchors()
    register_cache_anchor("")
    real = [f"[ctxpkg references ep-{300 + index:04d}..ep-{300 + index:04d}]" for index in range(limit)]
    for text in real:
        register_cache_anchor(text)
    assert not is_cache_anchor("")
    assert all(is_cache_anchor(text) for text in real), "an empty registration consumed a slot"
    register_cache_anchor("")
    assert all(is_cache_anchor(text) for text in real), "an empty registration evicted a real anchor"
    assert not is_cache_anchor("")

    # Matching is the whole text: a block's message is the block byte for byte, so neither a
    # longer text that carries the block nor a text that is the block cut short is an anchor.
    clear_cache_anchors()
    register_cache_anchor(ANCHORS[0])
    assert not is_cache_anchor(ANCHORS[0] + "x")
    assert not is_cache_anchor(ANCHORS[0][:-1])

    near_miss = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": ANCHORS[0] + " tail"},
                 {"role": "assistant", "content": "Noted."},
                 {"role": "user", "content": "tail question"},
                 {"role": "assistant", "content": "tail answer"}]
    baseline = apply_anthropic_cache_control(copy.deepcopy(near_miss), native_anthropic=True)
    planned = apply_anthropic_cache_control(copy.deepcopy(near_miss), native_anthropic=True)
    assert not _marked(planned[1]), "a message carrying the anchor plus a tail was marked as the anchor"
    assert [_marked(message) for message in planned] == [_marked(message) for message in baseline]
