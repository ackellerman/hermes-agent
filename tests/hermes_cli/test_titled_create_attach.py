"""Element e2: a create reports whether it made the row, and only that invocation attaches unguarded.

`hermes chat -c <title> --create-if-missing -m X --provider P` mints its session and then resumes it
in the same invocation. The resume validates the requested flags against the route frozen at
creation; that guard must keep refusing a genuine change on every *existing* session, including a
title race that handed back another creator's row — so instead of loosening the comparison, the
create says who made the row and only the invocation that made it skips the check:

1. **The gateway reports what the call did.** `session.create` with a title is resolve-or-create
   (`gateway/session_controls.py` `_create`), and its reply now carries `info["created"]`: true when
   this call staged and committed a new row, false when it resolved an existing titled row (the bare
   title and its `#N` continuation alike). `describe` advertises the observation, so an older
   gateway that cannot report it is distinguishable; no create parameter and no frozen-request byte
   changes.

2. **The client marks only its own row.** `create_titled_session` writes
   `args.resume_created_here = <id>` on the run object it already receives — only for a
   `created: true` row, and with no signature change (a caller that sets nothing is simply not
   exempt). `main` copies it onto the CLI as `cli._created_session_id`, and the classic call site
   — `GatewaySessionStore.open_session` → `_check_resume_launch`, the path every board round takes —
   passes it as `created_here=`. The shared comparison, `check_resume_policy`, takes the same
   parameter (default False); its skip is a whole-comparison early return, and it holds only when
   the id is the session being resumed.

3. **Nothing else is exempt.** `created: false` (an existing row, including a concurrent title
   race), a second invocation on the same id, a marker for a different session, and a gateway that
   reports no `created` at all (an older process) all check exactly as before.

Every module under test is imported inside its test function: on the base tree this file fails on
the behaviour, not on a collection error. No socket, no network, no vendor credential.
"""

import pytest


def _snapshot(sid, info):
    return {"session_id": sid, "stored_session_id": sid, "messages": [], "revision": 1,
            "execution_generation": 0, "subscription_id": "sub-" + sid, "info": dict(info)}


class _ResumeGateway:
    """A gateway connection with no socket: `session.resume` answers one frozen snapshot."""

    final = False
    launch_api_key = None
    launch_api_key_typed = False

    def __init__(self, snapshot):
        self.frozen = snapshot
        self.calls = []

    def state(self, session_id):
        import types
        return types.SimpleNamespace(adopt_snapshot=lambda snapshot: None)

    def call(self, method, **params):
        self.calls.append((method, params))
        return self.frozen if method == "session.resume" else {}

    def resume(self, call, session_id, *, typed=False):
        self.calls.append(("session.resume", {"session_id": session_id, "typed": typed}))
        return self.frozen


class _CreateStore:
    """`create_titled_session`'s view of the gateway: one `session.create` answer, recorded."""

    def __init__(self, info):
        self.info = dict(info)
        self.calls = []

    def client_env(self):
        return {}

    def call(self, method, **params):
        self.calls.append((method, params))
        return _snapshot("local-made", self.info)


def _owner(tmp_path, titles=()):
    """A real SessionDB (titles resolve as the owner resolves them) and the authority `_create` reads."""
    import types

    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    for index, title in enumerate(titles):
        sid = f"local-held-{index}"
        db.create_session(sid, source="cli")
        db.set_session_title(sid, title)
    return db, types.SimpleNamespace(db=db, epoch=1, profile_id=str(tmp_path), instance_id="inst-1",
                                     runner=None, events={})


def _staged_writes(monkeypatch, db, ids):
    """`_create`'s staged write without the storage layer: the receipt becomes a real, titled row.

    Only the three collaborators that write the ledger are replaced; the title resolution above,
    `title_new_session` and everything `_create` decides stay the shipped code.
    """
    import types

    import gateway.session_local_recovery as recovery
    import gateway.session_local_migration as migration
    import gateway.session_route_recovery as route_recovery
    import hermes_state_local as state_local
    from gateway import session_local

    receipts = []

    def stage(authority, actor, params, skills_prompt=None, client_local=False, desktop_local=False):
        receipt = {"session_id": next(ids), "params": dict(params)}
        receipts.append(receipt)
        return receipt

    def commit(database, *, epoch, receipt):
        database.create_session(receipt["session_id"], source="cli")
        return receipt

    def restore(authority, sid):
        return types.SimpleNamespace(session_id=sid)

    monkeypatch.setattr(session_local, "stage_local_session", stage)
    monkeypatch.setattr(state_local, "commit_local_session", commit)
    monkeypatch.setattr(recovery, "restore_local_session", restore)
    # Title resolution's transport checks: this store has no lanes and no lineage receipts.
    monkeypatch.setattr(route_recovery, "handed_off_ref", lambda authority, sid: None)
    monkeypatch.setattr(migration, "resolve_local_target",
                        lambda authority, actor, sid: types.SimpleNamespace(session_id=sid))
    return receipts


def _controls(authority, *, launch_request=None):
    """A real `AuthorityConnection` on a stand-in authority, with a scripted `resume`.

    The connection is the shipped object (its dispatch, capabilities and describe), so only the
    authority's storage and the resume reply are fakes.
    """
    import types

    from gateway.session_controls import AuthorityConnection

    identity = {"user_id": "owner-1", "profile_id": authority.profile_id,
                "instance_id": authority.instance_id, "provider": "local", "native_bootstrap": True}
    connection = AuthorityConnection(authority, types.SimpleNamespace(), identity)

    async def resume(ref, params):
        return _snapshot(ref.session_id, {"model": "wire-model", "cwd": "/w",
                                          "launch_request": dict(launch_request or {})})

    connection.resume = resume
    return connection


def test_a_titled_create_reports_whether_it_made_the_row(tmp_path, monkeypatch):
    """`session.create` with a title says whether *this* call staged and committed the row:

    - a title nobody holds: created, the row is the one the call made;
    - the same title again: an existing row, nothing created;
    - the `#N` continuation the bare title now resolves to: an existing row too.

    The observation is additive: no create parameter carries it, and the frozen request the reply
    echoes is byte-for-byte what `resume` returned.
    """
    import asyncio

    db, authority = _owner(tmp_path)
    try:
        receipts = _staged_writes(monkeypatch, db, iter(("local-made-1", "local-made-2")))
        controls = _controls(authority, launch_request={"model": "typed-model", "provider": "anthropic"})

        async def drive():
            first = await controls.create(None, {"title": "fresh"})
            again = await controls.create(None, {"title": "fresh"})
            numbered = await controls.create(None, {"title": "fresh #2"})
            # The bare title now resolves the latest `#N` continuation, which is an existing row.
            continuation = await controls.create(None, {"title": "fresh"})
            return first, again, numbered, continuation

        first, again, numbered, continuation = asyncio.run(drive())

        assert first["info"]["created"] is True
        assert first["stored_session_id"] == "local-made-1"
        assert again["info"]["created"] is False
        assert again["stored_session_id"] == "local-made-1"
        assert numbered["info"]["created"] is True
        assert numbered["stored_session_id"] == "local-made-2"
        assert continuation["info"]["created"] is False
        assert continuation["stored_session_id"] == "local-made-2"

        # One new observation in `info`, and the frozen request it echoes is untouched.
        assert set(first["info"]) == {"model", "cwd", "launch_request", "created"}
        assert first["info"]["launch_request"] == {"model": "typed-model", "provider": "anthropic"}
        # Nothing new rides the create: the two staged rows saw only the create's own parameters.
        assert [r["session_id"] for r in receipts] == ["local-made-1", "local-made-2"]
        assert all("created" not in r["params"] for r in receipts)
    finally:
        db.close()


def test_describe_advertises_the_created_report(monkeypatch):
    """`describe` names the observation, so a client can tell a gateway that cannot report `created`
    (an older process) from one that can — and the create's parameter contract is unchanged."""
    import asyncio
    import types

    monkeypatch.setattr("gateway.session_skin.resolved_skin", lambda: "default")
    controls = _controls(types.SimpleNamespace(instance_id="inst-1", profile_id="profile-1",
                                              epoch=1, events={}))

    described = asyncio.run(controls.describe(None, {}))

    assert "session-create-created-v1" in described["capabilities"]
    assert "created" not in described["session_create"]["parameters"]
    assert "title" in described["session_create"]["parameters"]


def test_create_titled_session_marks_only_the_row_it_created(monkeypatch):
    """The client writes `args.resume_created_here` for a `created: true` row and for nothing else.

    A resolved row and a gateway that reports no `created` (an older process) leave the run object
    alone: today's behaviour, unchanged. The signature is the one main and the existing tests call.
    """
    import inspect
    import types

    import hermes_cli.gateway_session_store as store_mod

    created = _CreateStore({"created": True})
    monkeypatch.setattr(store_mod, "session_store", lambda: created)
    args = types.SimpleNamespace(continue_last="T")
    assert store_mod.create_titled_session("T", args) == "local-made"
    assert args.resume_created_here == "local-made"
    # The marker is an observation on the reply, never a create parameter.
    assert "created" not in created.calls[-1][1]
    assert created.calls[-1][1]["title"] == "T"

    resolved = types.SimpleNamespace()
    monkeypatch.setattr(store_mod, "session_store", lambda: _CreateStore({"created": False}))
    assert store_mod.create_titled_session("T", resolved) == "local-made"
    assert not hasattr(resolved, "resume_created_here")

    older = types.SimpleNamespace()
    monkeypatch.setattr(store_mod, "session_store", lambda: _CreateStore({}))
    assert store_mod.create_titled_session("T", older) == "local-made"
    assert not hasattr(older, "resume_created_here")

    # No signature change: main's TUI branch calls it with the title alone, and
    # tests/hermes_cli/test_chat_c_fail_loudly.py pins the `lambda title, run=None` shape.
    assert list(inspect.signature(store_mod.create_titled_session).parameters) == ["title", "args"]
    assert store_mod.create_titled_session("T") == "local-made"


def test_a_title_race_row_gets_no_exemption(monkeypatch):
    """A concurrent creator won the title and this call resolved *its* row (created false).

    Control: the marker is not written, so the resume that follows still refuses a route that
    differs from that row's frozen policy. This holds on the base tree too — it is what an
    unconditional marker would break.
    """
    import types

    import hermes_cli.gateway_session_store as store_mod
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import GatewaySessionStore

    monkeypatch.setattr(store_mod, "session_store", lambda: _CreateStore({"created": False}))
    args = types.SimpleNamespace()
    assert store_mod.create_titled_session("contested", args) == "local-made"
    assert not hasattr(args, "resume_created_here")

    # The row someone else made froze another route; this invocation's flags differ.
    gateway = _ResumeGateway(_snapshot("local-made", {"launch_request": {"model": "theirs"}}))
    cli = types.SimpleNamespace(session_id="local-made", _resumed=True,
                                _created_session_id=getattr(args, "resume_created_here", None))
    with pytest.raises(GatewayClientError, match="-m/--model"):
        GatewaySessionStore(gateway).open_session(cli, {"model": "mine"})


def test_the_classic_open_session_skips_the_check_only_for_its_own_new_row():
    """The path every board round takes: `open_session` → `_check_resume_launch`.

    The exemption needs both the marker and the id match. The row's frozen policy is the one this
    process just sent, so the resume that follows a create is not compared — while the same id
    without a marker (a second invocation), a marker for another session, and a plain
    `_check_resume_launch` call all refuse a genuine difference, exactly as before.
    """
    import types

    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import GatewaySessionStore, _check_resume_launch

    frozen = _snapshot("local-made", {"launch_request": {"model": "claude-opus-5-5",
                                                         "provider": "anthropic"}})
    # What the credentials ladder resolved: a different route than the create froze.
    changed = {"model": "deepseek-v4.1-flash:cloud", "provider": "ollama-cloud"}

    _check_resume_launch(changed, frozen, created_here="local-made")  # the create's own resume

    with pytest.raises(GatewayClientError, match="-m/--model"):
        _check_resume_launch(changed, frozen)  # a second invocation on the same id
    with pytest.raises(GatewayClientError, match="-m/--model"):
        _check_resume_launch(changed, frozen, created_here="local-other")  # another session's marker

    def open_with(marker):
        cli = types.SimpleNamespace(session_id="local-made", _resumed=True)
        if marker is not None:
            cli._created_session_id = marker
        store = GatewaySessionStore(_ResumeGateway(frozen))
        return store.open_session(cli, changed)

    assert open_with("local-made") == "local-made"
    for marker in (None, "local-other"):
        with pytest.raises(GatewayClientError, match="-m/--model"):
            open_with(marker)


def test_the_own_new_row_skips_the_classic_only_flags_too():
    """The classic-only half of the guard (`--run-budget`, `-v`) is skipped for the created row only.

    Those two flags are compared against the row's frozen `launch_request` below the created-here
    skip, so a created row — whose frozen request is exactly what this process just sent — resumes
    without that comparison either. The same id with no marker (a second invocation) and a marker
    naming another session still refuse a run-budget/verbose difference, naming both flags.
    """
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_session_store import _check_resume_launch

    frozen = _snapshot("local-made", {"launch_request": {"model": "wire-model", "run_budget": 900.0}})
    launch = {"model": "wire-model", "run_budget": 60.0, "verbose": True}

    _check_resume_launch(launch, frozen, created_here="local-made")  # the create's own resume

    with pytest.raises(GatewayClientError, match="--run-budget, -v/--verbose differ"):
        _check_resume_launch(launch, frozen)  # a second invocation on the same id
    with pytest.raises(GatewayClientError, match="--run-budget, -v/--verbose differ"):
        _check_resume_launch(launch, frozen, created_here="local-other")  # another session's marker


def test_the_shared_comparison_takes_created_here(monkeypatch, tmp_path):
    """`check_resume_policy` carries the same parameter (default False) for the shared comparison.

    The skip is a whole-comparison early return, so with a marker that names another session every
    earlier rule still runs — including the typed-vs-frozen spelling rule, which sits between the
    `model_requested` mapping and the re-resolve branch.
    """
    import argparse

    from hermes_cli.gateway_chat import check_resume_policy
    from hermes_cli.gateway_client import GatewayClientError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    frozen = _snapshot("local-made", {"launch_request": {"model": "claude-opus-5-5",
                                                         "provider": "anthropic"}})
    changed = argparse.Namespace(model="deepseek-v4.1-flash:cloud", provider="ollama-cloud")

    check_resume_policy(changed, frozen, created_here="local-made")
    with pytest.raises(GatewayClientError, match="-m/--model"):
        check_resume_policy(changed, frozen)
    with pytest.raises(GatewayClientError, match="-m/--model"):
        check_resume_policy(changed, frozen, created_here="local-other")

    legacy = _snapshot("local-old", {"launch_request": {"model": "claude-sonnet-4-6"}})
    check_resume_policy(argparse.Namespace(model="claude-sonnet-4.6"), legacy, created_here="local-made")
    with pytest.raises(GatewayClientError, match="-m/--model"):
        check_resume_policy(argparse.Namespace(model="claude-sonnet-7-0"), legacy,
                            created_here="local-made")


def test_main_hands_the_created_marker_to_the_cli(monkeypatch, tmp_path):
    """`main` resolves the titled create on `args` and hands its marker to the CLI process it
    starts: the mark rides the run, so `open_session` can see it. Without a create it is None."""
    import argparse
    import sys
    import types

    import hermes_cli.main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda value: value)
    monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda args: None)
    monkeypatch.setattr(main_mod, "_apply_user_config_bypass", lambda args: None)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda args: None)
    monkeypatch.setattr(main_mod, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr(main_mod, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(main_mod, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_consent.offer_consent_before_chat",
                        lambda args: None)

    handed = {}

    def fake_cli_main(**kwargs):
        handed.clear()
        handed.update(kwargs)

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=fake_cli_main))

    def chat_args(**overrides):
        base = {"continue_last": None, "resume": None, "model": None, "provider": None,
                "toolsets": None, "skills": None, "tui": False, "tui_dev": False, "verbose": None,
                "quiet": True, "query": "hello", "image": None, "worktree": False,
                "checkpoints": False, "pass_session_id": False, "max_turns": None,
                "ignore_rules": False, "ignore_user_config": False, "safe_mode": False,
                "compact": False, "source": None, "yolo": False, "accept_hooks": False,
                "no_restore_cwd": False, "create_if_missing": False, "reasoning": None,
                "run_budget": None, "query_file": None, "output_format": "text", "in_dir": None}
        base.update(overrides)
        return argparse.Namespace(**base)

    main_mod.cmd_chat(chat_args(resume_created_here="local-made"))
    assert handed["created_session_id"] == "local-made"

    main_mod.cmd_chat(chat_args())
    assert handed["created_session_id"] is None


def test_the_cli_keeps_the_marker_it_was_handed(monkeypatch):
    """The transport from `cli.main` to the object `open_session` is given: `cli.py` declares the
    parameter on its entry point, `_build_cli_from_args` forwards it, and `HermesCLI` keeps it as
    `_created_session_id` — absent (None) on a run that created nothing."""
    import inspect

    import cli as cli_mod

    assert inspect.signature(cli_mod.main).parameters["created_session_id"].default is None
    shipped_cli = cli_mod.HermesCLI

    recorded = {}

    class _Host:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(cli_mod, "HermesCLI", _Host)
    monkeypatch.setattr("agent.coding_context.coding_selection", lambda **_: None)
    monkeypatch.setattr("agent.skill_commands.resolve_auto_load_skills", lambda *a, **k: [])
    built = cli_mod._build_cli_from_args(None, None, None, None, None, None, None, None, None, False,
                                         None, False, False, False, None,
                                         created_session_id="local-made")
    assert recorded["created_session_id"] == "local-made"
    assert built is not None

    # The real constructor keeps it, and a run with no create carries None.
    for attribute in ("_init_display_options", "_init_model_routing", "_init_runtime_state"):
        monkeypatch.setattr(shipped_cli, attribute, lambda *a, **k: None)
    assert shipped_cli(created_session_id="local-made")._created_session_id == "local-made"
    assert shipped_cli()._created_session_id is None
