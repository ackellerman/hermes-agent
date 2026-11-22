"""Subagents never inherit the 1h prompt-cache tier: with ``prompt_caching.cache_ttl: 1h`` the parent
keeps 1h, the child is built at 5m, and the child's wire markers carry no ``ttl``. A disabled cache
stays disabled (the child is not re-enabled to 5m). The one opt-in: a profile-routed child whose OWN
profile sets a literal ``cache_ttl: 1h`` keeps 1h (``auto`` there is not an opt-in)."""
from types import SimpleNamespace
from agent.prompt_caching import apply_anthropic_cache_control
from tools.delegate_tool import _apply_child_cache_ttl


def _markers(messages):
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            out += [b["cache_control"] for b in c if isinstance(b, dict) and "cache_control" in b]
        if "cache_control" in m:
            out.append(m["cache_control"])
    return out


def test_child_1h_becomes_5m_and_wire_markers_drop_ttl():
    child = SimpleNamespace(_cache_ttl="1h")
    _apply_child_cache_ttl(child)
    assert child._cache_ttl == "5m"
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go"}]
    wire = apply_anthropic_cache_control([dict(m) for m in msgs], cache_ttl=child._cache_ttl)
    marks = _markers(wire)
    assert marks and all("ttl" not in m for m in marks), marks
    # and the parent's 1h layout really is different, so this is not a vacuous check
    parent_marks = _markers(apply_anthropic_cache_control([dict(m) for m in msgs], cache_ttl="1h"))
    assert any(m.get("ttl") == "1h" for m in parent_marks)


def test_disabled_and_5m_are_left_alone():
    for ttl in (None, "5m"):
        child = SimpleNamespace(_cache_ttl=ttl)
        _apply_child_cache_ttl(child)
        assert child._cache_ttl == ttl


def test_real_spawn_path_applies_it(tmp_path, monkeypatch):
    """Through ``_build_child_agent`` with a real AIAgent: parent configured 1h, child ends at 5m, and the
    parent is untouched."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "prompt_caching:\n  cache_ttl: 1h\nmodel:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dtc, "_load_config", dict)
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    assert parent._cache_ttl == "1h", "fixture: the operator's 1h must actually be in effect"
    child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                  max_iterations=4, task_count=1, parent_agent=parent)
    try:
        assert child._cache_ttl == "5m"
        assert parent._cache_ttl == "1h"
    finally:
        child.close()
        parent.close()


# --- opt-in: a profile-routed child whose OWN profile pins ``cache_ttl: 1h`` keeps 1h -------------------


def test_explicit_opt_in_keeps_1h_and_nothing_else_moves():
    child = SimpleNamespace(_cache_ttl="1h")
    _apply_child_cache_ttl(child, profile_pins_1h=True)
    assert child._cache_ttl == "1h"
    for ttl in (None, "5m"):  # the opt-in never raises or enables a tier
        child = SimpleNamespace(_cache_ttl=ttl)
        _apply_child_cache_ttl(child, profile_pins_1h=True)
        assert child._cache_ttl == ttl


def _profile_spawn(tmp_path, monkeypatch, *, launch_ttl, profile_ttl):
    """Real ``_build_child_agent`` spawn of a profile-routed child (``profile_home`` = a reviewer profile
    whose config.yaml says ``prompt_caching.cache_ttl: <profile_ttl>``) from an interactive parent whose
    launching profile says ``<launch_ttl>``. Returns (parent ttl, child ttl)."""
    launch = tmp_path / "launch"
    reviewer = tmp_path / "launch" / "profiles" / "reviewer"
    reviewer.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    model = "model:\n  default: anthropic/claude-sonnet-4.6\n"
    (launch / "config.yaml").write_text(f"prompt_caching:\n  cache_ttl: {launch_ttl}\n{model}", encoding="utf-8")
    (reviewer / "config.yaml").write_text(
        (f"prompt_caching:\n  cache_ttl: {profile_ttl}\n" if profile_ttl is not None else "") + model,
        encoding="utf-8")
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", dict)
    monkeypatch.setattr(dtc, "_load_config", dict)
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                  max_iterations=4, task_count=1, parent_agent=parent, profile_home=reviewer)
    try:
        return parent._cache_ttl, child._cache_ttl
    finally:
        child.close()
        parent.close()


def test_profile_child_with_explicit_1h_keeps_1h(tmp_path, monkeypatch):
    assert _profile_spawn(tmp_path, monkeypatch, launch_ttl="5m", profile_ttl="1h") == ("5m", "1h")


def test_profile_child_with_auto_is_still_5m(tmp_path, monkeypatch):
    # The launching parent is interactive and on 1h; 'auto' is "the source decides", and a child is
    # machine-paced, so it is NOT an opt-in.
    assert _profile_spawn(tmp_path, monkeypatch, launch_ttl="1h", profile_ttl="auto") == ("1h", "5m")


def test_profile_child_inherits_no_opt_in_from_the_launching_profile(tmp_path, monkeypatch):
    # Only the child's OWN profile can opt in: the launcher's 1h (and an unset profile key) stays clamped.
    assert _profile_spawn(tmp_path, monkeypatch, launch_ttl="1h", profile_ttl="5m") == ("1h", "5m")
    assert _profile_spawn(tmp_path / "unset", monkeypatch, launch_ttl="1h", profile_ttl=None) == ("1h", "5m")


def test_profile_child_with_disabled_cache_stays_disabled(tmp_path, monkeypatch):
    assert _profile_spawn(tmp_path, monkeypatch, launch_ttl="1h", profile_ttl="off") == ("1h", None)
