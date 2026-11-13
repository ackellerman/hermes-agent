"""The ``delegate_task`` a gateway session sees when it joins its delegations in the turn
(``delegation.gateway_mode`` inline, or ``auto`` on a chat without interim assistant messages).

It is the one-shot variant (results return in the call; nothing about background delivery) plus
``action='continue'``. A one-shot process exits after its reply, so nothing there can follow up on a child. A
gateway session lives on: a child that finished inside one call can be given its next instruction by a later
call, and the joined result already carries the ``subagent_id`` to name it. Only the controls that need a
RUNNING child (list/steer/stop) have nothing to reach, so they stay unadvertised (the handler still answers
them). The description keeps the one-shot delivery sentence: dispatch recognises the joined variant by it
(``tools.delegate_tool.session_joins_delegation``).
"""
from __future__ import annotations

from typing import Any

_JOINED_CONTROL = {
    "action": {
        "type": "string", "enum": ["spawn", "continue"],
        "description": (
            "Default 'spawn'. 'continue' = give a FINISHED child of this conversation a follow-up (subagent_id + "
            "message): it resumes its own conversation with its history, and its result returns in this tool call "
            "like a spawn. goal/tasks are ignored unless spawning."),
    },
    "subagent_id": {
        "type": "string",
        "description": "Required for action='continue': the subagent_id from that child's entry in an earlier result.",
    },
    "message": {"type": "string", "description": "For action='continue': the child's next instruction."},
}


def joined_delegate_function(function: dict) -> dict:
    """A built ``delegate_task`` definition for a session that joins its children and outlives the call."""
    from tools.delegate_tool import oneshot_delegate_function
    joined = oneshot_delegate_function(function)
    params = joined["parameters"]
    return {**joined, "parameters": {**params, "properties": {**params["properties"], **_JOINED_CONTROL}}}


def joined_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*tools* with ``delegate_task`` swapped for its joined gateway variant."""
    return [
        {**tool, "function": joined_delegate_function(tool["function"])}
        if (tool.get("function") or {}).get("name") == "delegate_task" else tool
        for tool in tools
    ]
