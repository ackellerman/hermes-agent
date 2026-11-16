"""Stale-skill notice: a skill whose loaded copy in history no longer matches its SKILL.md on disk.

History rows come from the REAL builders (``_load_skill_payload`` + ``_build_skill_message`` for
gateway auto-load rows, ``skill_view`` JSON for tool rows), so a format change in a builder breaks
these tests instead of silently disabling detection.
"""

import itertools
import json
import logging
import os
import shutil
from pathlib import Path

import pytest

AUTO_HEADER = '[IMPORTANT: The "{}" skill is auto-loaded. Follow its instructions for this session.]'
DISCORD_PREFIX = "[Discord channel context: backfill]\n\n[New message]\n[Alice] "
NOTICE_HEAD = "[Skill changed on disk"
_CALL_IDS = itertools.count()


def _skills_root() -> Path:
    return Path(os.environ["HERMES_HOME"]) / "skills"


def _write_config(text: str) -> None:
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(text, encoding="utf-8")


def _skill(name, body="Step one.\nStep two.\nStep three.\n", category=None, root=None, fm_name=None):
    root = root or _skills_root()
    skill_dir = root / category / name if category else root / name
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {fm_name or name}\ndescription: test skill {name}\n---\n\n# {name}\n\n{body}",
        encoding="utf-8")
    return skill_dir


def _edit(skill_dir, old="Step two.", new="Step two (edited)."):
    path = skill_dir / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _auto_payload(name):
    from agent.skill_commands import _build_skill_message, _load_skill_payload
    loaded = _load_skill_payload(name, task_id="discord:quick-key")
    assert loaded, name
    payload, skill_dir, display = loaded
    return _build_skill_message(payload, skill_dir, AUTO_HEADER.format(display))


def _auto_row(*names, prefix="", user_text="what's next?"):
    """The user row a new gateway session stores: payload(s), the user's text, then inbound prefixes."""
    return {"role": "user", "content": prefix + "\n\n".join([*map(_auto_payload, names), user_text])}


def _tool_rows(name, result=None, file_path=None, task_id="20261116_120000_abcd1234"):
    """An assistant ``skill_view`` call plus its tool result (real ``skill_view`` JSON by default)."""
    from tools.skills_tool import skill_view
    call_id = f"call_{next(_CALL_IDS)}"
    args = {"name": name, **({"file_path": file_path} if file_path else {})}
    if result is None:
        result = skill_view(name, file_path=file_path, task_id=task_id)
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "skill_view", "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "tool_name": "skill_view", "content": result},
    ]


def _notice(history):
    from agent.skill_staleness import stale_skill_notice
    return stale_skill_notice(history)


def _assert_names(notice, stale, fresh=()):
    assert notice is not None
    assert notice.count(NOTICE_HEAD) == 1
    for forbidden in ('[IMPORTANT: The "', "SKILL_PRUNED", "Skill directory"):  # never looks like a load
        assert forbidden not in notice
    for name in stale:
        assert f'skill_view(name="{name}")' in notice
    for name in fresh:
        assert f'"{name}"' not in notice


# U1
def test_auto_load_copy_unchanged_disk_gives_no_notice():
    _skill("alpha")
    assert _notice([_auto_row("alpha"), {"role": "assistant", "content": "ok"}]) is None


# U2 (+ notice text guards)
def test_auto_load_copy_then_edit_gives_notice():
    alpha = _skill("alpha")
    history = [_auto_row("alpha"), {"role": "assistant", "content": "ok"}]
    _edit(alpha)
    notice = _notice(history)
    _assert_names(notice, ["alpha"])
    assert notice.startswith(NOTICE_HEAD)


# U3
def test_prefixed_auto_load_row_is_still_a_load():
    alpha = _skill("alpha")
    history = [_auto_row("alpha", prefix=DISCORD_PREFIX)]
    assert not history[0]["content"].startswith("[IMPORTANT")
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])


# U4
def test_skill_view_copy_goes_stale_then_reload_stops_notice():
    alpha = _skill("alpha")
    history = _tool_rows("alpha")
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])
    history += _tool_rows("alpha")  # the agent re-reads the current text
    assert _notice(history) is None


# U5
def test_linked_file_view_is_not_a_skill_md_load():
    alpha = _skill("alpha")
    (alpha / "references").mkdir()
    (alpha / "references" / "x.md").write_text("# Reference x\n", encoding="utf-8")
    history = _tool_rows("alpha", file_path="references/x.md")
    assert json.loads(history[1]["content"])["success"] is True
    _edit(alpha)
    assert _notice(history) is None


# U6
def test_dedup_stub_does_not_reset_the_reference():
    from tools.skills_tool import _skill_view_with_bump
    alpha = _skill("alpha")
    full = _skill_view_with_bump({"name": "alpha"}, task_id="dedup-task")
    stub = _skill_view_with_bump({"name": "alpha"}, task_id="dedup-task")
    assert json.loads(stub).get("dedup") is True
    history = _tool_rows("alpha", result=full) + _tool_rows("alpha", result=stub)
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])


def _demoted(name, full):
    from agent.context_compressor import _skill_pruned_marker
    from agent.context_compressor_tail import _lean_recovery_stub
    return _lean_recovery_stub("skill_view", len(full), "sid-1") + " " + _skill_pruned_marker(name)


def _pruned(name, full):
    from agent.context_compressor import _skill_pruned_marker
    return _skill_pruned_marker(name)


def _failed(name, full):
    from tools.skills_tool_plugin import _fail
    return _fail(f"Failed to read skill '{name}': boom")


def _spilled(name, full):
    from tools.tool_result_storage import _build_persisted_message
    return _build_persisted_message(full[:200], True, len(full), "/tmp/spill/skill_view.txt")


# U7
@pytest.mark.parametrize("replace", [_demoted, _pruned, _failed, _spilled], ids=lambda f: f.__name__.strip("_"))
def test_latest_load_replaced_by_non_load_row_gives_no_notice(replace):
    from tools.skills_tool import skill_view
    alpha = _skill("alpha")
    history = [_auto_row("alpha")]  # an OLDER full copy that will go stale
    history += _tool_rows("alpha", result=replace("alpha", skill_view("alpha", task_id="t")))
    _edit(alpha)
    assert _notice(history[:1]) is not None  # the older copy alone IS stale
    assert _notice(history) is None


# U8
def test_one_notice_names_only_the_edited_skills():
    alpha, beta = _skill("alpha"), _skill("beta")
    history = [_auto_row("alpha", "beta")]
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"], fresh=["beta"])
    _edit(beta)
    _assert_names(_notice(history), ["alpha", "beta"])


def _delete(skill_dir):
    shutil.rmtree(skill_dir)


def _rename(skill_dir):
    moved = skill_dir.with_name("omega")
    skill_dir.rename(moved)
    _edit(moved, "name: alpha", "name: omega")


def _disable(skill_dir):
    _write_config("skills:\n  disabled:\n    - alpha\n")


# U9
@pytest.mark.parametrize("gone", [_delete, _rename, _disable], ids=lambda f: f.__name__.strip("_"))
def test_missing_renamed_or_disabled_skill_is_a_clean_skip(gone, caplog):
    alpha = _skill("alpha")
    history = [_auto_row("alpha")] + _tool_rows("alpha")
    _edit(alpha)
    assert _notice(history) is not None
    gone(alpha)
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert _notice(history) is None
    assert not [r for r in caplog.records if r.exc_info or r.levelno >= logging.WARNING], caplog.records


TOKEN_BODY = ("Run ${HERMES_SKILL_DIR}/scripts/run.sh for session ${HERMES_SESSION_ID}.\n"
              "Step two.\n")


# U10
@pytest.mark.parametrize("kind", ["auto", "tool"])
def test_template_tokens_match_both_renderings(kind):
    alpha = _skill("alpha", body=TOKEN_BODY)
    history = [_auto_row("alpha")] if kind == "auto" else _tool_rows("alpha", task_id="sess-42")
    text = history[-1]["content"]
    assert str(alpha) in text
    assert ("${HERMES_SESSION_ID}" in text) if kind == "auto" else ("sess-42" in text)
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])


# U11
@pytest.mark.parametrize("kind", ["auto", "tool"])
def test_inline_shell_output_matches_unchanged_skill(kind):
    _write_config("skills:\n  inline_shell: true\n")
    alpha = _skill("alpha", body="Greeting: !`echo hi`\nStep two.\n")
    history = [_auto_row("alpha")] if kind == "auto" else _tool_rows("alpha")
    text = history[-1]["content"]
    assert "Greeting: hi" in text and "!`echo hi`" not in text
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])


# U12
def test_typed_header_quote_is_not_a_load():
    _skill("alpha")
    history = _tool_rows("alpha") + [{"role": "assistant", "content": "loaded"}, {
        "role": "user", "content": f"What does {AUTO_HEADER.format('alpha')} mean here?"}]
    assert _notice(history) is None


# U13
def test_odd_history_rows_never_raise():
    _skill("alpha")
    odd = [None, "text", 3, {"role": "user", "content": None},
           {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}]},
           {"role": "assistant", "tool_calls": [{"id": "c9", "type": "function",
                                                 "function": {"name": "skill_view", "arguments": "{not json"}}]},
           {"role": "tool", "tool_call_id": "c9", "content": "{}"},
           {"role": "tool", "tool_call_id": "no-such-call", "content": "x"},
           {"role": "tool", "content": None}]
    assert _notice(odd) is None


def test_one_failing_resolution_does_not_hide_another(monkeypatch, caplog):
    import tools.skills_tool as st
    alpha, beta = _skill("alpha"), _skill("beta")
    history = [_auto_row("alpha", "beta")]
    _edit(alpha)
    _edit(beta)
    real = st.resolve_skill_md

    def flaky(name):
        if name == "alpha":
            raise RuntimeError("resolver exploded")
        return real(name)

    monkeypatch.setattr(st, "resolve_skill_md", flaky)
    with caplog.at_level(logging.DEBUG, logger="agent.skill_staleness"):
        notice = _notice(history)
    _assert_names(notice, ["beta"], fresh=["alpha"])
    assert [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info and "alpha" in r.getMessage()]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads mode-000 files")
def test_one_unreadable_skill_does_not_hide_another(caplog):
    alpha, beta = _skill("alpha"), _skill("beta")
    history = [_auto_row("alpha", "beta")]
    _edit(alpha)
    _edit(beta)
    (alpha / "SKILL.md").chmod(0)
    try:
        with caplog.at_level(logging.DEBUG, logger="agent.skill_staleness"):
            notice = _notice(history)
    finally:
        (alpha / "SKILL.md").chmod(0o644)
    _assert_names(notice, ["beta"], fresh=["alpha"])
    assert [r for r in caplog.records if r.exc_info and "alpha" in r.getMessage()]


def test_public_entry_point_swallows_internal_failure(monkeypatch, caplog):
    import agent.skill_staleness as ss
    _skill("alpha")

    def boom(history, limit=8):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr(ss, "stale_skill_names", boom)
    with caplog.at_level(logging.DEBUG, logger="agent.skill_staleness"):
        assert ss.stale_skill_notice([_auto_row("alpha")]) is None
    assert [r for r in caplog.records if r.exc_info]


# U14
def test_notice_names_the_eight_most_recent_loads():
    names = [f"skill-{i:02d}" for i in range(10)]
    dirs = [_skill(n) for n in names]
    history = []
    for name in names:
        history += [_auto_row(name), {"role": "assistant", "content": "ok"}]
    for skill_dir in dirs:
        _edit(skill_dir)
    _assert_names(_notice(history), names[2:], fresh=names[:2])


# Added: one skill loaded under two names collapses to one file (the newest copy wins).
def test_same_file_under_two_names_uses_the_newest_copy():
    board = _skill("spark-board", category="devops")
    history = [_auto_row("spark-board")]
    _edit(board)
    _assert_names(_notice(history), ["spark-board"])
    history += _tool_rows("devops/spark-board")  # re-read after the edit, by path
    assert _notice(history) is None


# Added: an UNKNOWN newest copy also joins the path collapse.
def test_unknown_newest_copy_under_another_name_hides_older_copy():
    from agent.context_compressor import _skill_pruned_marker
    board = _skill("spark-board", category="devops")
    history = [_auto_row("spark-board")] + _tool_rows(
        "devops/spark-board", result=_skill_pruned_marker("devops/spark-board"))
    _edit(board)
    assert _notice(history[:1]) is not None
    assert _notice(history) is None


# Added: more than 8 template wildcards → the file is skipped (bounded regex cost).
@pytest.mark.parametrize("tokens, expect_notice", [(8, True), (9, False)])
def test_wildcard_cap(tokens, expect_notice):
    alpha = _skill("alpha", body="".join(f"id {i}: ${{HERMES_SESSION_ID}}\n" for i in range(tokens)) + "Step two.\n")
    history = _tool_rows("alpha", task_id="sess-7")
    _edit(alpha)
    assert (_notice(history) is not None) is expect_notice


# Added: an edit at the END of the file is still a change: appending a line, or deleting the last
# line (then the disk text is a strict prefix of the stale copy; the copy must end where the file ends).
@pytest.mark.parametrize("kind", ["auto", "tool"])
@pytest.mark.parametrize("body", ["Step one.\nStep two.\n", TOKEN_BODY], ids=["plain", "tokens"])
@pytest.mark.parametrize("edit", [("Step two.\n", "Step two.\nStep three: new rule.\n"), ("\nStep two.\n", "\n")],
                         ids=["append", "truncate"])
def test_edit_at_end_of_file_is_detected(kind, body, edit):
    alpha = _skill("alpha", body=body)
    history = [_auto_row("alpha")] if kind == "auto" else _tool_rows("alpha")
    assert _notice(history) is None
    _edit(alpha, *edit)
    _assert_names(_notice(history), ["alpha"])


# Added: user rows are sanitized on load; a skill that documents <memory-context> must still match.
def test_memory_context_skill_matches_after_session_db_replay(tmp_path):
    from hermes_state import SessionDB
    alpha = _skill("alpha", body="Wrap recall in <memory-context>old recall</memory-context> blocks.\nStep two.\n")
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s-mem", "discord")
        db.append_message("s-mem", "user", content=_auto_row("alpha")["content"])
        db.append_message("s-mem", "assistant", content="ok")
        history = db.get_messages_as_conversation("s-mem", repair_alternation=True)
    finally:
        db.close()
    assert "<memory-context>" not in history[0]["content"]  # sanitize_context ran on load
    assert _notice(history) is None
    _edit(alpha)
    _assert_names(_notice(history), ["alpha"])
