"""SPEC-0034 AC6: the conversation loop latches after a terminal kanban tool.

A turn that already fired ``kanban_request_review`` / ``kanban_block`` /
``kanban_complete`` / ``kanban_request_changes`` must make no further API
call: the loop head consults ``session_called_kanban_terminal`` and breaks
before the next iteration. Falsifies the 2026-09-12 zombie run (implementer
handoff at 20:29:50; the process kept issuing API calls #30-#34 for another
hour because nothing refused the loop).
"""

from __future__ import annotations

from agent.conversation_loop import _loop_continues
from agent.kanban_stop import session_called_kanban_terminal


def _terminal_messages():
    return [
        {"role": "user", "content": "do the work"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "rr1",
                    "function": {"name": "kanban_request_review", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_request_review",
            "content": "review requested",
            "tool_call_id": "rr1",
        },
    ]


class TestLoopTerminalLatch:
    def test_loop_head_stops_after_terminal_tool(self):
        """AC6 falsifier: a terminal session must not continue the loop even
        with API-call budget remaining."""
        messages = _terminal_messages()
        assert session_called_kanban_terminal(messages) is True
        assert _loop_continues(api_call_count=0, max_iterations=60, budget_remaining=100, messages=messages) is False

    def test_loop_head_continues_on_live_session(self):
        """Polarity guard: no terminal tool called -> loop continues on budget."""
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "working"},
        ]
        assert session_called_kanban_terminal(messages) is False
        assert _loop_continues(api_call_count=0, max_iterations=60, budget_remaining=100, messages=messages) is True

    def test_budget_exhaustion_still_stops(self):
        """The pre-existing stop conditions keep their polarity."""
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "working"},
        ]
        assert _loop_continues(api_call_count=60, max_iterations=60, budget_remaining=0, messages=messages) is False

    def test_terminal_tool_and_budget_exhausted_also_stops(self):
        """Both stop conditions true -> still stops (no resurrection)."""
        messages = _terminal_messages()
        assert _loop_continues(api_call_count=60, max_iterations=60, budget_remaining=0, messages=messages) is False
