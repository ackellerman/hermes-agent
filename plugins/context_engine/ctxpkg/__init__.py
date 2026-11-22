"""``ctxpkg``: structured driver context — episodes on disk, a reference block, and ``recall``.

Each compaction cuts the compacted span into episodes (one per user turn), writes each one whole to
``<HERMES_HOME>/ctxpkg/<lineage>/ep-<NNNN>.md`` and keeps only a reference line in the prompt. ``recall(ref)``
brings an episode back in full as an ordinary tool result, and that use keeps it warm for
``ctxpkg.warm_turns`` turns. Step 3 builds the engine, the store and the tool; the compaction itself, the
extend-only reference prefix and warm protection land in step 4.

The engine subclasses the built-in ``ContextCompressor`` so the host's token accounting, micro-compaction
settings and ``should_compress`` keep working. The host's ``compression.*`` knobs reach it through
``apply_compression_settings`` (``agent/agent_init.py``'s ``_build_context_engine`` calls it for a plugin engine),
and ``supports_prepared_compaction = False`` keeps the host from attaching a ``PreparedCompaction`` that this
engine would never consume.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.context_compressor import ContextCompressor, _PRUNE_MIN_CHARS

from .episodes import message_text
from .store import MAX_PART_CHARS, RECALL_REF_RE, EpisodeStore, split_parts

logger = logging.getLogger(__name__)

_RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Read a compacted episode back in full. The context lists episodes as `ep-<NNNN> · turns a-b · k chars · "
        "summary · ids: ...`; call this with its ref when you need the detail. Text comes back in parts of at "
        "most 24000 chars, split on line boundaries. Using an episode keeps it warm for a few turns."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Episode ref, exactly ep-<four digits> (e.g. ep-0007)."},
            "part": {"type": "integer", "description": "1-based part number; default 1.", "minimum": 1},
        },
        "required": ["ref"],
    },
}


class CtxpkgEngine(ContextCompressor):
    """Episode store and ``recall`` for a compacted context (compaction itself: step 4)."""

    @property
    def name(self) -> str:
        return "ctxpkg"

    # The host pre-builds a summary for engines that ask for it; this engine's compaction does not read one.
    supports_prepared_compaction = False

    def __init__(self, model: str = ""):
        super().__init__(model=model)

    # ── host settings ───────────────────────────────────────────────────────

    def apply_compression_settings(self, cs: Any, max_tokens: int | None = None) -> None:
        """Take the ``compression.*`` settings the host would pass to the built-in constructor.

        Runs before ``update_model``, which re-derives the trigger through ``_derive_trigger`` from
        ``_config_threshold_percent`` (``agent/context_compressor.py:2648-2660, 2688-2690``) — so the percent is
        set from ``cs.threshold`` here, not left for ``update_model`` to read a default into.
        """
        self._config_threshold_percent = float(cs.threshold)
        self._configured_threshold_percent = float(cs.threshold)
        self.threshold_tokens_cap = self._coerce_threshold_tokens_cap(cs.threshold_tokens)
        self.protect_first_n = int(cs.protect_first)
        self.protect_last_n = int(cs.protect_last)
        self.summary_target_ratio = self._clamp_summary_target_ratio(cs.target_ratio)
        self.abort_on_summary_failure = bool(cs.abort_on_summary_failure)
        self.max_tokens = self._coerce_max_tokens(max_tokens)
        self.proactive_prune_tokens = int(cs.proactive_prune_tokens or 0)
        self.proactive_prune_min_result_chars = max(_PRUNE_MIN_CHARS,
                                                    int(cs.proactive_prune_min_chars or 8000))
        self.proactive_prune_min_reclaim_tokens = max(0, int(cs.proactive_prune_min_reclaim or 0))
        self.min_tail_user_messages = int(cs.min_tail_users)
        self.tail_mode = cs.tail_mode if cs.tail_mode in ("legacy", "lean") else "lean"

    @staticmethod
    def _clamp_summary_target_ratio(ratio: float) -> float:
        """The clamp ``ContextCompressor.__init__`` applies to ``summary_target_ratio`` (:2841)."""
        return max(0.10, min(float(ratio), 0.80))

    # ── store ───────────────────────────────────────────────────────────────

    @property
    def store(self) -> EpisodeStore:
        """The episode store for this engine's session (resolved per call; cheap and rotation-stable)."""
        return EpisodeStore.for_session(self._session_id, self._session_db)

    # ── the recall tool ─────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(_RECALL_TOOL)]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        if name != "recall":
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        ref = args.get("ref")
        if not isinstance(ref, str) or not RECALL_REF_RE.fullmatch(ref):
            # Refused before any path is built: a ref is never interpolated into one.
            return json.dumps({"error": f"ref must be exactly 'ep-' and four digits, got {ref!r}"})
        raw_part = args.get("part")
        try:
            # An absent part is 1; an explicit 0 is not silently promoted, it is out of the 1-based range.
            part = 1 if raw_part is None else int(raw_part)
        except (TypeError, ValueError):
            return json.dumps({"error": "part must be a positive integer", "ref": ref})
        if part < 1:
            return json.dumps({"error": "part must be a positive integer", "ref": ref})
        try:
            store = self.store
            text = store.read(ref)
            if text is None:
                return json.dumps({"error": f"unknown episode ref {ref}", "known": store.nearest(ref, limit=10)})
            parts = split_parts(text, MAX_PART_CHARS)
            if part > len(parts):
                return json.dumps({"error": f"{ref} has {len(parts)} part(s), not {part}", "ref": ref,
                                   "parts": len(parts)})
            store.renew(ref)
        except OSError as exc:
            logger.debug("ctxpkg recall of %s failed", ref, exc_info=True)
            return json.dumps({"error": f"could not read {ref}: {exc}"})
        return json.dumps({"ref": ref, "part": part, "parts": len(parts), "text": parts[part - 1]})

    def on_turn_complete(self, messages: list[dict[str, Any]], usage: dict[str, Any] | None = None,
                         **kwargs: Any) -> None:
        """Advance the persisted turn counter and renew every known ref the turn's assistant text names."""
        try:
            store = self.store
            turn = store.bump_turn()
            text = "\n".join(message_text(message) for message in messages or []
                             if message.get("role") == "assistant")
            for ref in dict.fromkeys(RECALL_REF_RE.findall(text)):
                if store.exists(ref):
                    store.renew(ref, turn)
        except Exception:
            # Best-effort observation hook: a store failure must not fail the finished turn.
            logger.debug("ctxpkg warmth update failed", exc_info=True)


def register(ctx) -> None:
    """Plugin entry point (``plugins/plugin_loader.py`` calls ``register``; its class fallback passes no args)."""
    ctx.register_context_engine(CtxpkgEngine(model=""))
