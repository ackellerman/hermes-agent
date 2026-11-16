"""Stale-skill notice for continuing sessions.

A skill's text reaches a conversation either as a gateway auto-load payload inside a user row or
as a ``skill_view`` tool result. When the SKILL.md on disk changes later, that copy is out of date
and nothing tells the agent. :func:`stale_skill_notice` derives, from the history plus disk and
with no state of its own, one short notice naming every loaded skill whose newest in-context copy
no longer matches its file. The notice repeats each turn until the agent re-reads the skill (the
re-read then becomes the newest copy and matches).

Detection never renders the skill: rendering runs inline-shell snippets and ``skill_view`` also
installs deps and read-marks. Instead the current raw SKILL.md is compiled to a pattern that
accepts every rendering it could have produced (template tokens substituted or literal, inline
shell expanded or literal). Read-only and never raises: any failure means no notice, logged at
debug.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Newest distinct load names examined per turn (bounds file lookups and reads).
MAX_CHECKED_NAMES = 8
#: More pattern wildcards than this and the file is skipped (bounds regex backtracking).
MAX_WILDCARDS = 8
_SKILL_DIR_BLOCK = "\n\n[Skill directory: "
_AUTO, _TOOL, _UNKNOWN = "auto", "tool", "unknown"


@dataclass(frozen=True)
class _Load:
    """The newest copy of a skill in history: ``kind`` auto/tool, or unknown when that copy is gone
    (compacted, failed, spilled), in which case no notice is possible."""
    name: str
    kind: str
    order: tuple[int, int]
    text: str = ""
    offset: int = 0


def _skill_view_file_path_call_ids(history: list[Any]) -> set[str]:
    """Ids of ``skill_view`` calls that asked for a linked file (``file_path``), not SKILL.md."""
    from agent.context_compressor import _json_dict, _tc_get
    ids: set[str] = set()
    for msg in history:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = _tc_get(tc, "function", {})
            if _tc_get(fn, "name") == "skill_view" and _json_dict(_tc_get(fn, "arguments")).get("file_path"):
                ids.add(str(_tc_get(tc, "id") or ""))
    return ids


def _tool_load(name: str, content: Any, order: tuple[int, int]) -> Optional[_Load]:
    """A ``skill_view`` result row as a load; None for a dedup stub (the earlier copy stays the
    reference); UNKNOWN for anything that no longer carries the full text."""
    try:
        parsed = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        if parsed.get("dedup") or parsed.get("status") == "unchanged":
            return None
        if parsed.get("success") is True and isinstance(parsed.get("content"), str):
            return _Load(name, _TOOL, order, parsed["content"])
    return _Load(name, _UNKNOWN, order)


def _latest_loads(history: list[Any]) -> dict[str, _Load]:
    """``load name → newest _Load`` over the history's ``content`` (never ``api_content``)."""
    from agent.context_compressor import _content_text_for_contains
    from agent.context_compressor_tail import skill_view_names_by_call_id
    from agent.skill_commands import _AUTO_LOAD_HEADER_RE
    rows = [m for m in history if isinstance(m, dict)]
    view_names = skill_view_names_by_call_id([m for m in rows if m.get("role") == "assistant"])
    linked_file_calls = _skill_view_file_path_call_ids(rows)
    latest: dict[str, _Load] = {}
    for idx, msg in enumerate(rows):
        role = msg.get("role")
        if role == "user":
            text = _content_text_for_contains(msg.get("content"))
            # SEARCH, not startswith: inbound sender/channel prefixes precede the header.
            hits = list(_AUTO_LOAD_HEADER_RE.finditer(text))
            for n, hit in enumerate(hits):
                end = hits[n + 1].start() if n + 1 < len(hits) else len(text)
                # Only a real payload carries the directory block; a typed or quoted header does not.
                if _SKILL_DIR_BLOCK in text[hit.end():end]:
                    latest[hit.group(1)] = _Load(hit.group(1), _AUTO, (idx, n), text, hit.end())
        elif role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            name = view_names.get(call_id)
            if not name or call_id in linked_file_calls:
                continue
            if (load := _tool_load(name, msg.get("content"), (idx, 0))) is not None:
                latest[name] = load
    return latest


def _body_pattern(raw: str, skill_dir: Path) -> Optional[str]:
    """Regex source matching every rendering of *raw* (``None`` = more than MAX_WILDCARDS)."""
    from agent.skill_preprocessing import _INLINE_SHELL_RE, _SKILL_TEMPLATE_RE
    wildcards = 0

    def tokens(text: str) -> str:
        nonlocal wildcards
        out, pos = [], 0
        for m in _SKILL_TEMPLATE_RE.finditer(text):
            out.append(re.escape(text[pos:m.start()]))
            value = re.escape(str(skill_dir)) if m.group(1) == "HERMES_SKILL_DIR" else r"\S+"
            out.append(f"(?:{re.escape(m.group(0))}|{value})")
            wildcards += 1
            pos = m.end()
        out.append(re.escape(text[pos:]))
        return "".join(out)

    out, pos = [], 0
    for m in _INLINE_SHELL_RE.finditer(raw):  # template substitution runs first, inside commands too
        out.append(tokens(raw[pos:m.start()]))
        out.append(f"(?:!`{tokens(m.group(1))}`|[\\s\\S]*?)")
        wildcards += 1
        pos = m.end()
    out.append(tokens(raw[pos:]))
    return None if wildcards > MAX_WILDCARDS else "".join(out)


def _is_current(load: _Load, raw: str, skill_dir: Path) -> bool:
    """Whether *load*'s copy is a rendering of the current *raw* SKILL.md."""
    from agent.skill_preprocessing import _INLINE_SHELL_RE, _SKILL_TEMPLATE_RE
    if load.kind == _TOOL:  # skill_view's ``content`` is the whole rendered file
        body, tail, at = raw, "", None
    else:
        from agent.memory_context import sanitize_context
        # The payload is ``header + "\n\n" + body.strip() + "\n\n[Skill directory: ..."``; user rows
        # are sanitized on load, so mirror that on the disk body.
        body, tail, at = sanitize_context(raw.strip()), _SKILL_DIR_BLOCK, load.offset + 2
    if not (_SKILL_TEMPLATE_RE.search(body) or _INLINE_SHELL_RE.search(body)):
        return load.text == body if at is None else load.text.startswith(body + tail, at)
    pattern = _body_pattern(body, skill_dir)
    if pattern is None:
        logger.debug("Stale-skill check: %r has more than %d wildcards; skipped", load.name, MAX_WILDCARDS)
        return True
    if at is None:
        return re.fullmatch(pattern, load.text) is not None
    return re.compile(pattern + re.escape(tail)).match(load.text, at) is not None


def _read_raw(skill_md: Path) -> str:
    from tools.skills_tool import _read_skill_text
    return _read_skill_text(skill_md)


def stale_skill_names(history: list[Any], limit: int = MAX_CHECKED_NAMES) -> list[str]:
    """Load names (oldest first) of skills whose newest in-history copy no longer matches disk."""
    from tools import skills_tool
    loads = sorted(_latest_loads(history).values(), key=lambda load: load.order, reverse=True)[:limit]
    newest_by_file: dict[Path, tuple[_Load, Path]] = {}
    for load in loads:  # newest first: the first copy seen for a file wins
        try:
            resolved = skills_tool.resolve_skill_md(load.name)
            if resolved is None:
                logger.debug("Stale-skill check: %r does not resolve to a skill file; skipped", load.name)
                continue
            skill_dir, skill_md, _resolved_name = resolved
            newest_by_file.setdefault(skill_md.resolve(), (load, skill_dir))
        except Exception:
            logger.debug("Stale-skill check: resolving %r failed", load.name, exc_info=True)
    stale: list[_Load] = []
    for skill_md, (load, skill_dir) in newest_by_file.items():
        if load.kind == _UNKNOWN:
            continue
        try:
            if not _is_current(load, _read_raw(skill_md), skill_dir):
                stale.append(load)
        except Exception:
            logger.debug("Stale-skill check: comparing %r failed", load.name, exc_info=True)
    return [load.name for load in sorted(stale, key=lambda load: load.order)]


def _format_notice(names: list[str]) -> str:
    quoted = ", ".join(f'"{n}"' for n in names)
    calls = ", ".join(f'skill_view(name="{n}")' for n in names)
    if len(names) == 1:
        return (f"[Skill changed on disk: {quoted} changed after you loaded it earlier in this conversation; "
                f"the copy in your context is out of date. Before acting on it, re-read it with {calls}.]")
    return (f"[Skill changed on disk: {quoted} changed after you loaded them earlier in this conversation; "
            f"the copies in your context are out of date. Before acting on them, re-read each with {calls}.]")


def stale_skill_notice(history: list[Any]) -> Optional[str]:
    """One notice naming every loaded skill whose in-context copy is out of date, or None. Never raises."""
    try:
        names = stale_skill_names(history)
        return _format_notice(names) if names else None
    except Exception:
        logger.debug("Stale-skill notice failed", exc_info=True)
        return None
