"""Regression test for #45759.

An all-exhausted credential pool holds entries but no *usable* credential.
``list_authenticated_providers`` must not treat such a provider as
authenticated -- otherwise an aggregator whose quota is spent gets matched
during no-provider ``/model`` resolution, wins the model name, and sticks as
the session provider (the "sticky provider fallback pollution" bug).
"""

import pytest


class _FakePool:
    def __init__(self, available: bool):
        self._available = available

    def has_credentials(self) -> bool:
        # The pool still holds entries...
        return True

    def has_available(self, **_kwargs) -> bool:
        # ...but none of them are usable when exhausted/dead.
        return self._available


def _patch_opencode_pool(monkeypatch, *, available: bool):
    """Make the opencode-go aggregator look configured but with a pool whose
    only credential is (un)available, depending on ``available``."""
    import hermes_cli.auth as auth
    import agent.credential_pool as cp

    monkeypatch.setattr(
        auth,
        "_load_auth_store",
        lambda: {
            "version": 1,
            "providers": {},
            "active_provider": None,
            "credential_pool": {"opencode-go": {"entries": [{"id": "x"}]}},
        },
    )
    monkeypatch.setattr(
        cp,
        "load_pool",
        lambda provider: _FakePool(available if provider == "opencode-go" else True),
    )


@pytest.fixture(autouse=True)
def _strip_provider_env(monkeypatch):
    """Don't let real provider keys in the environment authenticate providers
    through a different code path than the pool gate under test."""
    import os

    for key in list(os.environ):
        if "OPENCODE" in key or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)


def test_exhausted_pool_provider_is_not_authenticated(monkeypatch):
    """The fix: an exhausted pool is NOT authenticated. Fails on main, where
    the gate accepted any stored pool entry regardless of usability."""
    from hermes_cli.model_switch import get_authenticated_provider_slugs

    _patch_opencode_pool(monkeypatch, available=False)
    slugs = get_authenticated_provider_slugs(current_provider="alibaba")
    assert "opencode-go" not in slugs


def test_opaque_legacy_pool_value_stays_visible(monkeypatch):
    """Legacy token-style auth-store values have no parsed pool entries."""
    from hermes_cli.model_switch_providers import _credential_pool_is_usable

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda _provider: type(
            "EmptyPool",
            (),
            {
                "has_credentials": lambda self: False,
                "has_available": lambda self: False,
            },
        )(),
    )

    assert _credential_pool_is_usable("opencode-go", raw_pool_present=True)


def test_picker_shows_exhausted_pool_provider(monkeypatch):
    """The interactive picker must include providers whose credential pool
    entries are all exhausted, so the user can still switch to a different
    model under the same provider."""
    from hermes_cli.model_switch_providers import list_picker_providers

    _patch_opencode_pool(monkeypatch, available=False)
    providers = list_picker_providers(
        current_provider="alibaba",
        user_providers={},
        custom_providers=[],
    )
    slugs = [p["slug"] for p in providers]
    assert "opencode-go" in slugs, (
        "Picker must show exhausted-pool providers so the user can select "
        "a different model under the same provider"
    )


def test_model_options_payload_shows_exhausted_pool_provider(monkeypatch, tmp_path):
    """The REAL picker payload must honour the same contract as the test above.

    ``build_model_options_payload`` is what RPC ``model.options`` (TUI ``/model``), the dashboard
    ``ModelPickerDialog`` and the gateway's ``/api/model/options`` all serve. It never passed
    ``for_picker`` down to ``list_authenticated_providers``, so ``_overlay_has_creds`` skipped its
    cooldown-tolerance branch and the provider was dropped (or degraded to an empty canonical
    skeleton) even though its pool still holds credentials — the user lost the provider entirely.

    ``list_picker_providers`` (asserted above) always passed the flag, which is why the divergence
    went unnoticed: the contract was specified and tested, but not on the surface users hit.
    """
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_opencode_pool(monkeypatch, available=False)

    ctx = load_picker_context()
    row = next((p for p in build_model_options_payload(ctx)["providers"]
                if p["slug"] == "opencode-go"), None)
    assert row is not None, "exhausted-pool provider vanished from the picker payload"
    assert row["authenticated"] is True, "cooldown pool reported as unauthenticated"
    assert row["models"], "picker row carries no models"


def test_model_options_payload_skeleton_is_not_mistaken_for_unconfigured(monkeypatch, tmp_path):
    """On ``include_unconfigured=True`` the provider must keep models, not degrade to a skeleton.

    ``_apply_picker_hints`` marks a row ``authenticated=False`` when its source is ``canonical``
    with no models — which the picker renders as "(needs setup)" (#needs-setup). A cooldown pool
    must not take that path.
    """
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_opencode_pool(monkeypatch, available=False)

    ctx = load_picker_context()
    rows = build_model_options_payload(ctx, include_unconfigured=True)["providers"]
    row = next((p for p in rows if p["slug"] == "opencode-go"), None)
    assert row is not None
    assert not (row.get("source") == "canonical" and not row.get("models")), \
        "provider degraded to an empty canonical skeleton"
    assert row["authenticated"] is True


def test_available_pool_provider_still_authenticated(monkeypatch, tmp_path):
    """Guard: a healthy pool is unaffected by enabling the flag."""
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _patch_opencode_pool(monkeypatch, available=True)

    ctx = load_picker_context()
    row = next((p for p in build_model_options_payload(ctx)["providers"]
                if p["slug"] == "opencode-go"), None)
    assert row is not None
    assert row["authenticated"] is True

