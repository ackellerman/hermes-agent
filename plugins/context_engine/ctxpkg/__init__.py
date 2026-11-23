"""``ctxpkg``: structured driver context — episodes on disk, a reference block, and ``recall``.

Each compaction cuts the compacted span into episodes (one per user turn), writes each one whole to
``<HERMES_HOME>/ctxpkg/<lineage>/ep-<NNNN>.md`` and keeps only a reference line in the prompt. ``recall(ref)``
brings an episode back in full as an ordinary tool result, and that use keeps it warm for
``ctxpkg.warm_turns`` turns. Compaction (``compaction.py``) appends one reference block per compaction and
leaves every earlier block byte-identical, so the prefix up to the newest block stays a prompt-cache read.

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

from .compaction import VIEW_STRUCTURED, CtxpkgCompactionMixin, load_settings
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


class CtxpkgEngine(CtxpkgCompactionMixin, ContextCompressor):
    """Episode store, ``recall`` tool and episode compaction for a compacted context."""

    @property
    def name(self) -> str:
        return "ctxpkg"

    # The host pre-builds a summary for engines that ask for it; this engine's compaction does not read one.
    supports_prepared_compaction = False

    def __init__(self, model: str = ""):
        super().__init__(model=model)
        # The host sets these before every ``select_context`` (``agent/conversation_loop.py``); the defaults
        # are the not-allowed state, so an engine driven directly (a test, an older host) serves nothing.
        self.view_allowed = False
        self.request_uids: list[str | None] | None = None
        self.prefill_count = 0
        self.cache_ttl_seconds: int | None = None
        self._view_tool_ids: set[str] | None = None
        self._view_anchor_registered = False
        self._view_off_logged = False
        self._rough_tokens = 0
        self._structured_latch = False
        try:
            self._view_settings = load_settings()
        except Exception:  # a config read must never break constructing the engine
            logger.debug("ctxpkg settings unreadable at construction", exc_info=True)
            self._view_settings = None
        if self._view_settings is None or self._view_settings.view != VIEW_STRUCTURED:
            # ``view: off`` keeps today's behaviour byte for byte: the host's override check reads this
            # attribute, sees a non-callable and skips the hook — the per-request clone is not even paid.
            self.select_context = None

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

    # ── the structured view (``ctxpkg.view: structured``) ───────────────────

    def select_context(self, request_messages: list[dict[str, Any]], *,
                       conversation_messages: list[dict[str, Any]] | None = None,
                       incoming_message: dict[str, Any] | None = None,
                       budget_tokens: int = 0, **kwargs: Any) -> list[dict[str, Any]] | None:
        """Serve the request as the structured view, or None to leave it exactly as assembled.

        Returns None unless the host allowed the view for this request (``view_allowed``), and unless the
        request aligns with the uid list the host recorded. Between passes nothing here moves: the view is a
        function of the request, the uid list and ``view.json``, so a new request only appends to the tail.
        """
        if not self.view_allowed:
            return None
        from . import view as view_module

        try:
            store = self.store
        except Exception:
            logger.debug("ctxpkg view: no store for this session; serving the request unchanged", exc_info=True)
            return None
        now = view_module.now_epoch()
        state = view_module.ViewState.load(store)
        self._structured_latch = True

        _offset, reason = view_module.alignment(request_messages, self.request_uids, self.prefill_count)
        if _offset is None:
            view_module.append_row(store, view_module.PASSES_NAME,
                                   view_module.view_off_row(state, reason, now))
            if not self._view_off_logged:
                self._view_off_logged = True
                logger.info("ctxpkg structured view off for this request: %s (session=%s)",
                            reason, self._session_id or "-")
            view_module.stamp_request(store, state, now)
            return None

        self._record_reruns(request_messages, state, store, now)
        self._register_newest_anchor(state)
        self._rough_tokens = view_module.rough(conversation_messages)
        if view_module.should_pass(state, self.cache_ttl_seconds, now):
            view_module.run_pass(
                request_messages, self.request_uids or [], conversation_messages, state, store,
                self._judge(), now, settings=self._view_settings,
                prefill_count=int(self.prefill_count or 0), ttl_s=self.cache_ttl_seconds,
            )
        view_module.stamp_request(store, state, now)
        return view_module.build_view(request_messages, self.request_uids, self.prefill_count, state)

    def _judge(self) -> Any:
        """The pass judge: ``keep_all`` in this step; step 5 swaps in the flash judge."""
        from . import view as view_module

        return view_module.keep_all

    def _record_reruns(self, request_messages: list[dict[str, Any]], state: Any, store: Any,
                       now: float) -> None:
        """One ``reruns.jsonl`` row per call this request makes again whose result a pass stubbed."""
        from . import view as view_module

        try:
            current = view_module.tool_call_ids(request_messages)
            previous, self._view_tool_ids = self._view_tool_ids, current
            if previous is None:
                return  # the first allowed request only records the set
            for row in view_module.rerun_rows(request_messages, current - previous, state, now):
                view_module.append_row(store, view_module.RERUNS_NAME, row)
        except Exception:
            logger.debug("ctxpkg rerun detection failed", exc_info=True)

    def _register_newest_anchor(self, state: Any) -> None:
        """Re-register the newest stored block once per engine instance (the registry is process-local)."""
        if self._view_anchor_registered:
            return
        self._view_anchor_registered = True
        if state.blocks:
            from . import view as view_module

            view_module.register_anchor(state.blocks[-1])

    # ── compaction and the backstop (the latch) ─────────────────────────────

    def should_compress_info(self, prompt_tokens: int | None = None) -> tuple[bool, str | None]:
        """Judge the larger of the provider figure and the raw history's rough size, once latched.

        The served view can be a fraction of the history, so the provider figure alone would never fire
        again; ``select_context`` records the raw estimate on every allowed request.
        """
        if not self._structured_latch:
            return super().should_compress_info(prompt_tokens)
        base = prompt_tokens if prompt_tokens is not None else getattr(self, "last_prompt_tokens", 0)
        try:
            tokens = max(int(base or 0), int(self._rough_tokens or 0))
        except (TypeError, ValueError):
            tokens = base
        return super().should_compress_info(tokens)

    def compress(self, messages: list[dict[str, Any]], current_tokens: int | None = None,
                 focus_topic: str | None = None, force: bool = False, memory_context: str = "",
                 bypass_cooldown: bool = False) -> list[dict[str, Any]]:
        """The built-in handoff compaction once the view latched; the estimate resets after a compaction."""
        if not self._structured_latch:
            return super().compress(messages, current_tokens, focus_topic, force, memory_context,
                                    bypass_cooldown)
        result = ContextCompressor.compress(self, messages, current_tokens, focus_topic, force,
                                            memory_context, bypass_cooldown)
        self._rough_tokens = 0
        return result

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        super().on_session_start(session_id, **kwargs)
        self._rough_tokens = 0
        self._view_tool_ids = None
        self._view_anchor_registered = False
        self._view_off_logged = False

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._rough_tokens = 0

    # ── the recall tool ─────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(_RECALL_TOOL)]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        if name != "recall":
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        response = self._recall(args)
        self._record_recall(args, response)
        return response

    def _recall(self, args: dict[str, Any]) -> str:
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

    def _record_recall(self, args: dict[str, Any], response: str) -> None:
        """One ``recalls.jsonl`` row for a recall of a ref a pass wrote (only while the view is allowed)."""
        if not self.view_allowed:
            return
        from . import view as view_module

        try:
            ref = args.get("ref")
            if not isinstance(ref, str):
                return
            try:
                payload = json.loads(response)
            except (TypeError, ValueError):
                payload = {}
            status = "ok" if isinstance(payload, dict) and "text" in payload else "error"
            store = self.store
            row = view_module.recall_row(view_module.ViewState.load(store), ref, status,
                                         view_module.now_epoch())
            if row is not None:
                view_module.append_row(store, view_module.RECALLS_NAME, row)
        except Exception:
            logger.debug("ctxpkg recall row failed", exc_info=True)

    def on_turn_complete(self, messages: list[dict[str, Any]], usage: dict[str, Any] | None = None,
                         **kwargs: Any) -> None:
        """Advance the persisted turn counter and renew every known ref the turn's assistant text names.

        While the view is allowed this also advances ``view_turn`` and records the first-seen turn and epoch
        of every uid the finished turn carries — the clock the never-candidate rules and the live gate use.
        """
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

        if not self.view_allowed:
            return
        from . import view as view_module

        try:
            store = self.store
            state = view_module.ViewState.load(store)
            state.view_turn += 1
            epoch = view_module.now_epoch()
            for message in messages or []:
                uid = message.get("message_uid") if isinstance(message, dict) else None
                if isinstance(uid, str) and uid and uid not in state.seen:
                    state.seen[uid] = [int(state.view_turn), float(epoch)]
            view_module.save_state(store, state)
        except Exception:
            logger.debug("ctxpkg view turn update failed", exc_info=True)


def register(ctx) -> None:
    """Plugin entry point (``plugins/plugin_loader.py`` calls ``register``; its class fallback passes no args)."""
    ctx.register_context_engine(CtxpkgEngine(model=""))
