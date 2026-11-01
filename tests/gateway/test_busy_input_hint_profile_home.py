"""The one-shot busy-input hint latches in the home its read consults.

Complements upstream #119493's ``test_hint_flag_lands_in_the_routed_profile_config`` (which
stubs the loader): these use the real ``_load_gateway_config`` and cover the reported repeat
(global home already flagged, profile home clean), the flag-present case, and the cache/overlay
read-back.

``_compose_busy_ack_message`` reads ``onboarding.seen.busy_input_prompt`` through
``_load_gateway_config()`` (the context-local profile home when a multiplexer sets the override,
else the global home). The flag must be written to that same file; writing it to the global
home while the read consults the profile home means the hint can never latch for that profile.
"""

import time
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from agent.onboarding import BUSY_INPUT_FLAG, busy_input_hint_gateway, is_seen, mark_seen
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _event():
    return MessageEvent(
        text="are you there?",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u1"),
        message_id="m1",
    )


def _runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


def _compose(runner) -> str:
    return runner._compose_busy_ack_message(
        _event(), time.time(), None, None,
        is_steer_mode=False, is_queue_mode=False, is_redirect_mode=False,
        demoted_for_subagents=False, demoted_for_compression=False,
    )


def _write_cfg(home, data):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(data))


def _read_cfg(home):
    path = home / "config.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else None


_HINT = busy_input_hint_gateway("interrupt")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A global home and a profile home; the profile home is the active override."""
    import gateway.run as gr

    global_home = tmp_path / "global"
    profile_home = tmp_path / "global" / "profiles" / "p1"
    global_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(gr, "_hermes_home", global_home)
    token = set_hermes_home_override(profile_home)
    try:
        yield SimpleNamespace(global_home=global_home, profile_home=profile_home)
    finally:
        reset_hermes_home_override(token)


class TestReadWriteSameFile:
    """The brief's second possible divergence: effective-config read vs raw write-back."""

    def test_mark_seen_on_override_home_is_visible_to_gateway_read(self, homes):
        from gateway.run import _gateway_config_home, _load_gateway_config

        _write_cfg(homes.profile_home, {"display": {"busy_ack_detail": True}})
        assert _gateway_config_home() == homes.profile_home
        # Warm the effective-config cache first so a stale cache entry would show up.
        assert not is_seen(_load_gateway_config(), BUSY_INPUT_FLAG)
        assert mark_seen(_gateway_config_home() / "config.yaml", BUSY_INPUT_FLAG)
        assert is_seen(_load_gateway_config(), BUSY_INPUT_FLAG)
        # The pre-existing user key survives the raw round-trip.
        assert _read_cfg(homes.profile_home)["display"] == {"busy_ack_detail": True}

    def test_unrelated_managed_overlay_does_not_hide_the_written_flag(self, homes, tmp_path, monkeypatch):
        from gateway.run import _gateway_config_home, _load_gateway_config
        from hermes_cli import managed_scope

        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(yaml.safe_dump({"display": {"busy_ack_detail": False}}))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        managed_scope.invalidate_managed_cache()
        try:
            assert not is_seen(_load_gateway_config(), BUSY_INPUT_FLAG)
            assert mark_seen(_gateway_config_home() / "config.yaml", BUSY_INPUT_FLAG)
            effective = _load_gateway_config()
            assert is_seen(effective, BUSY_INPUT_FLAG)
            assert effective["display"]["busy_ack_detail"] is False  # the overlay did apply
        finally:
            managed_scope.invalidate_managed_cache()


class TestBusyInputHintLatch:

    def test_flag_present_in_override_home_no_hint(self, homes):
        _write_cfg(homes.profile_home, {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}})
        assert _HINT not in _compose(_runner())

    def test_global_flagged_profile_clean_latches_in_profile_home(self, homes):
        """The reported regression: the global home already carries the flag, the profile home
        has none. The hint is shown once for the profile (its read consults the profile home) and
        must then latch THERE — before the fix it was written to the already-true global file and
        repeated on every busy message."""
        _write_cfg(homes.global_home, {"onboarding": {"seen": {BUSY_INPUT_FLAG: True}}})
        global_before = (homes.global_home / "config.yaml").read_bytes()
        runner = _runner()
        shown = [(_HINT in _compose(runner)) for _ in range(3)]
        assert shown == [True, False, False]
        assert _read_cfg(homes.profile_home)["onboarding"]["seen"][BUSY_INPUT_FLAG] is True
        assert (homes.global_home / "config.yaml").read_bytes() == global_before
