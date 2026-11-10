"""Element e3: `provider: auto` is "no pick" on the resume side too.

`provider: auto` is the create side's own sentinel: `gateway.session_policy._without_auto_pick`
drops it before a launch policy is built, so a session created with `--provider auto` freezes no
provider at all. The thin client's resume comparison read the same word literally, so re-running
the identical command line — a `-c <title>` continuation, or the workboard's argv, which carries
`--provider auto` — raised `--provider differs from this session's launch` on a session whose
frozen launch has no provider to differ from.

The comparison now shares the sentinel instead of spelling a second `== "auto"`: a requested
`provider` of `auto` is dropped there, and every other requested flag keeps its comparison
(`api_key` stays exempt as before, a different `-m` still refuses naming `-m/--model`). The
create side is untouched: the `session.create` request still carries the word, and the owner
folds it with the same sentinel.

Every module under test is imported inside its test function: on the base tree this file fails on
the behaviour, not on a collection error.
"""

import pytest


def _refusal(call, *args, **kwargs):
    """The `GatewayClientError` *call* raises, or None when it accepts the resume."""
    from hermes_cli.gateway_client import GatewayClientError

    with pytest.raises(GatewayClientError) as raised:
        call(*args, **kwargs)
    return str(raised.value)


def test_a_retyped_provider_auto_resumes_an_existing_titled_session():
    """The board's shape: a session frozen from `--provider auto` records no provider, and the
    same argv again — model plus `auto` — attaches instead of raising `--provider differs`."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "deepseek-v4.1-flash:cloud"}}}
    gateway_chat.check_resume_policy(
        argparse.Namespace(resume="local-abc", model="deepseek-v4.1-flash:cloud", provider="auto"), frozen)
    # `auto` is not a pick on its own either: nothing else typed leaves nothing to compare.
    gateway_chat.check_resume_policy(argparse.Namespace(resume="local-abc", provider="auto"), frozen)


def test_the_auto_pick_is_no_pick_against_an_explicitly_frozen_provider():
    """A resume that types only `auto` asks for nothing, so a session frozen with an explicit
    provider keeps it: the sentinel is dropped unconditionally, not only when the freeze is empty."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-a", "provider": "openrouter"}}}
    gateway_chat.check_resume_policy(
        argparse.Namespace(resume="s", model="model-a", provider="auto"), frozen)


def test_a_different_model_with_provider_auto_still_refuses_naming_the_model():
    """Dropping the sentinel drops the provider comparison only: a different `-m` is still a
    genuine override and still refuses, with `-m/--model` named and `--provider` not."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-a"}}}
    refusal = _refusal(gateway_chat.check_resume_policy,
                       argparse.Namespace(resume="s", model="model-b", provider="auto"), frozen)
    assert "-m/--model" in refusal
    assert "--provider" not in refusal


def test_only_the_auto_value_is_the_sentinel_not_every_spelling():
    """The sentinel is `_without_auto_pick`'s own rule (strip + case-fold to `auto`, whole value),
    so the shared spelling `AUTO`/` auto ` is no pick while a provider that merely contains or
    starts with the word — `automatic`, `auto-pick` — is a pick and still differs."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-a"}}}
    for spelling in ("AUTO", " auto ", " Auto "):
        gateway_chat.check_resume_policy(
            argparse.Namespace(resume="s", model="model-a", provider=spelling), frozen)
    for pick in ("automatic", "auto-pick", "openrouter"):
        refusal = _refusal(gateway_chat.check_resume_policy,
                           argparse.Namespace(resume="s", model="model-a", provider=pick), frozen)
        assert "--provider" in refusal


def test_the_alias_provenance_still_decides_the_model_with_the_auto_pick():
    """The owner-stamped `model_requested` still maps the alias a client re-types, whatever
    provider the client adds: the same alias name resumes, the client's own resolved id and an
    unrelated model still refuse (the shape `test_resume_alias_remote_config.py` pins)."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "owner-model",
                                          "model_requested": "fast", "provider": "custom"}}}
    gateway_chat.check_resume_policy(
        argparse.Namespace(resume="s", model="fast", provider="auto"), frozen)
    for model in ("client-model", "other"):
        refusal = _refusal(gateway_chat.check_resume_policy,
                           argparse.Namespace(resume="s", model=model, provider="auto"), frozen)
        assert "-m/--model" in refusal


def test_api_key_stays_exempt_and_every_other_flag_still_compares():
    """A re-typed key is never a policy change; another changed flag beside an `auto` provider is
    still refused, naming that flag alone."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-a", "reasoning": "low"}}}
    gateway_chat.check_resume_policy(
        argparse.Namespace(resume="s", model="model-a", provider="auto", reasoning="low", api_key="sk-retyped"),
        frozen)
    refusal = _refusal(gateway_chat.check_resume_policy,
                       argparse.Namespace(resume="s", model="model-a", provider="auto",
                                          reasoning="high", api_key="sk-retyped"), frozen)
    assert "--reasoning" in refusal
    assert "api_key" not in refusal and "-m/--model" not in refusal


def test_the_create_policy_still_carries_the_auto_pick():
    """The sentinel is applied on the comparison side only: `session.create` still sends the word
    the user typed, and the owner folds it with the same helper when it builds the policy."""
    import argparse

    from hermes_cli import gateway_chat

    policy = gateway_chat._requested_policy(argparse.Namespace(model="model-a", provider="auto"))
    assert policy == {"model": "model-a", "provider": "auto"}


def test_the_create_then_attach_branch_checks_no_policy():
    """A thin-client create adopts the snapshot it just made without a policy check: the only two
    comparison call sites in `run_gateway_chat` are the resume branches, both ahead of
    `session.create`. (That branch is e2's, and this element leaves it alone.)"""
    import ast
    import inspect

    from hermes_cli import gateway_chat

    module = ast.parse(inspect.getsource(gateway_chat))
    create = next(node for node in ast.walk(module)
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_gateway_chat")
    checks = [node.lineno for node in ast.walk(create)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "check_resume_policy"]
    creates = [node.lineno for node in ast.walk(create)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "rpc" and node.args
               and isinstance(node.args[0], ast.Constant) and node.args[0].value == "session.create"]
    assert len(checks) == 2 and creates, (checks, creates)
    assert max(checks) < min(creates)


def test_a_bare_continue_typing_no_flags_is_unchanged():
    """A bare `-c` resume types no provider and no model: nothing is compared, exactly as before."""
    import argparse

    from hermes_cli import gateway_chat

    frozen = {"info": {"launch_request": {"source": "cli", "model": "model-a", "provider": "openrouter"}}}
    gateway_chat.check_resume_policy(argparse.Namespace(resume="s"), frozen)
