"""The resolved route of one auxiliary call, and the shapes its recovery ladder hands back.

``_resolve_call_client`` decides which client serves a task; ``_ResolvedAuxRoute`` is what it
returns, ``_PreparedAuxRequest`` the request built on top of it, and ``_LadderStep`` /
``_LadderRoute`` / ``_rung`` the vocabulary the ordered recovery ladder speaks. They live here
rather than in ``agent/auxiliary_client.py`` so that facade keeps its line cap; ``auxiliary_client``
re-exports every name below, because callers and tests reach them as
``agent.auxiliary_client.<name>``.

``_task_stops_at_chain`` is the one predicate every route-escape door reads, so it sits with the
route record it belongs to.

``_get_auxiliary_task_config`` is imported from its own module (not through ``auxiliary_client``)
for the same reason: the config read is not part of the client facade.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

from agent.auxiliary_task_config import _get_auxiliary_task_config


class _ResolvedAuxRoute(NamedTuple):
    client: Any
    final_model: Optional[str]
    resolved_provider: str
    effective_provider: str


def _task_stops_at_chain(task: Optional[str], resolved_provider: Optional[str]) -> bool:
    """``auxiliary.<task>.task_chain_only`` AND a non-``auto`` resolved provider.

    The key means "an explicit pin is absolute" (D-124): a task whose route the operator pinned
    uses its own configured ``fallback_chain`` as the whole fallback policy, and never the main
    agent's model or the built-in discovery chain. It is read at the four doors a pinned task can
    leave its route through — the ``openrouter``/``custom`` auto re-probe and the vision ``auto``
    re-probe in ``_resolve_call_client``, the request-time main-agent net in
    ``_ladder_provider_fallback`` and the quarantine walk's discovery rung in
    ``_next_fallback_after_quarantine`` — so the gate is the same at each of them.

    The non-``auto`` half is not a convenience: ``auto`` already means "follow the main route and
    its chain", so the key must be inert there. The shipped ``auxiliary.voice_brief`` block sets
    ``task_chain_only: true`` with ``provider: auto``, and an auto brief answers exactly as it
    does without the key.
    """
    if not task:
        return False
    provider = (resolved_provider or "").strip().lower()
    if not provider or provider == "auto":
        return False
    return bool(_get_auxiliary_task_config(str(task)).get("task_chain_only"))


class _PreparedAuxRequest(NamedTuple):
    client: Any
    final_model: Optional[str]
    kwargs: dict[str, Any]
    resolved_provider: str
    request_provider: str
    resolved_model: Optional[str]
    resolved_base_url: Optional[str]
    resolved_api_key: Optional[str]
    resolved_api_mode: Optional[str]
    effective_timeout: float
    effective_extra_body: dict[str, Any]
    base_info: str


class _LadderStep(NamedTuple):
    """A provider request the ladder asks its driver to perform. kind: "call" (client, kwargs) |
    "retry_same_provider" (provider, model) | "fallback" (fb_client, fb_model, fb_label)."""
    kind: str
    args: tuple


def _rung(step: _LadderStep, accept: Callable[[Exception], bool]):
    """One ladder rung: perform ``step``; yields ``(response, None)`` on success,
    ``(None, exc)`` when ``accept(exc)`` lets the next rung handle it, else re-raises."""
    try:
        result = yield step
    except Exception as exc:
        if not accept(exc):
            raise
        return None, exc
    return result, None


# Immutable route context shared by the recovery rungs.
class _LadderRoute(NamedTuple):
    client: Any
    task: Optional[str]
    tag: str
    async_mode: bool
    base_info: str
    resolved_provider: str
    resolved_model: Optional[str]
    resolved_base_url: Optional[str]
    resolved_api_key: Optional[str]
    resolved_api_mode: Optional[str]
    final_model: Optional[str]
    main_runtime: Optional[dict[str, Any]]
    route_info: Optional[dict[str, str]]
    timeout: Optional[float]
