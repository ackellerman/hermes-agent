"""The fleet voice-channel map: one block in the ROOT config, seats only narrow, bad config fails closed."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from gateway.voice_channel_policy import load_voice_policy, resolve_voice_policy

_AUTO_VC, _MANUAL_VC, _OTHER_VC, _LEGACY_VC = "1001", "1002", "1003", "1004"


def _cfg(voice_channels=None, legacy=None, enabled=True) -> dict:
    discord: dict = {}
    if voice_channels is not None:
        discord["voice_channels"] = voice_channels
    if legacy is not None:
        discord["voice_auto_join"] = {"enabled": enabled, "channel_ids": legacy, "user_ids": ["7"]}
    return {"discord": discord}


_ROOT = _cfg({
    _AUTO_VC: {"join": "auto", "seats": ["default"]},
    _MANUAL_VC: {"join": "manual", "allow_thread_move": True},
})


def _policy(seat="alpha", seat_cfg=None, root=_ROOT, profiles=None, seat_is_root=False):
    profiles = profiles if profiles is not None else {"default": root, seat: seat_cfg or {}}
    return resolve_voice_policy(seat, root, seat_cfg or {}, profiles, seat_is_root=seat_is_root)


def test_listed_entries_take_their_defaults_and_unlisted_vcs_behave_as_before():
    p = _policy()
    manual, auto, unlisted = p.rule(_MANUAL_VC), p.rule(_AUTO_VC), p.rule(_OTHER_VC)
    assert (manual.listed, manual.join, manual.bind, manual.allow_attach_from_text, manual.allow_thread_move) == (
        True, "manual", "caller", False, True)
    assert (auto.join, auto.bind, auto.seat_allowed) == ("auto", "caller", False)  # seats: [default]
    assert (unlisted.listed, unlisted.join, unlisted.bind, unlisted.allow_thread_move) == (
        False, "manual", "caller", True)  # F11: today's behaviour
    assert unlisted.joinable and p.unlisted == "allow"
    assert _policy(root=_cfg({_MANUAL_VC: {}})).rule(_MANUAL_VC).allow_thread_move is False  # listed default


def test_a_seat_may_narrow_but_never_widen():
    seat = _cfg({
        _MANUAL_VC: {"join": "never"},
        _OTHER_VC: {"join": "auto", "allow_attach_from_text": True},   # cannot raise
        _AUTO_VC: {"allow_thread_move": True},
    })
    p = _policy(seat_cfg=seat)
    assert p.rule(_MANUAL_VC).join == "never" and not p.rule(_MANUAL_VC).joinable
    assert p.rule(_OTHER_VC).join == "manual" and p.rule(_OTHER_VC).allow_attach_from_text is False
    assert p.rule(_AUTO_VC).allow_thread_move is False
    assert any("only the root may mark auto" in w for w in p.seat_layer.warnings)


def test_a_seat_can_drop_itself_from_seats_and_refuse_unlisted_vcs():
    p = _policy(seat_cfg=_cfg({"unlisted": "refuse", _MANUAL_VC: {"seats": ["beta"]}}))
    assert p.rule(_MANUAL_VC).seat_allowed is False
    assert p.rule(_OTHER_VC).refusal == "unlisted"
    widen = _policy(root=_cfg({"unlisted": "refuse"}), seat_cfg=_cfg({"unlisted": "allow"}))
    assert widen.rule(_OTHER_VC).refusal == "unlisted"


@pytest.mark.parametrize("block,vc,refusal", [
    ({_MANUAL_VC: {"join": "manual", "prompt": True}}, _MANUAL_VC, "invalid"),   # unknown key
    ({_MANUAL_VC: {"join": "prompt"}}, _MANUAL_VC, "invalid"),                   # cut value
    ({_MANUAL_VC: {"bind": "1234"}}, _MANUAL_VC, "invalid"),                     # cut: bind <text id>
    ({_MANUAL_VC: {"allow_attach_from_text": "yes"}}, _MANUAL_VC, "invalid"),
    ({"occupancy": "handover"}, _OTHER_VC, "invalid"),                           # cut top-level key
    ({"unlisted": "maybe"}, _OTHER_VC, "unlisted"),                              # bad value refuses
])
def test_unknown_keys_and_values_fail_closed(block, vc, refusal):
    p = _policy(root=_cfg(block))
    assert p.rule(vc).refusal == refusal and not p.rule(vc).joinable
    assert p.is_auto_vc(vc) is (refusal == "invalid")  # R1 also fails closed on a broken entry


def test_a_bad_seat_entry_refuses_that_vc_for_that_seat_only():
    p = _policy(seat_cfg=_cfg({_MANUAL_VC: {"join": "sometimes"}}))
    assert p.rule(_MANUAL_VC).refusal == "invalid"
    assert _policy(seat="beta").rule(_MANUAL_VC).joinable


def test_legacy_auto_join_counts_as_auto_for_that_seat_and_for_r1_everywhere():
    alpha = _cfg(legacy=[int(_LEGACY_VC)])
    profiles = {"default": _ROOT, "alpha": alpha, "beta": {}}
    p_alpha = resolve_voice_policy("alpha", _ROOT, alpha, profiles, seat_is_root=False)
    p_beta = resolve_voice_policy("beta", _ROOT, {}, profiles, seat_is_root=False)
    assert p_alpha.rule(_LEGACY_VC).join == "auto" and _LEGACY_VC in p_alpha.auto_join_channels()
    assert p_beta.rule(_LEGACY_VC).join == "manual" and _LEGACY_VC not in p_beta.auto_join_channels()
    assert p_beta.is_auto_vc(_LEGACY_VC) and p_beta.is_auto_vc(_AUTO_VC)
    assert not p_beta.is_auto_vc(_MANUAL_VC)
    disabled = {"default": _ROOT, "alpha": _cfg(legacy=[_LEGACY_VC], enabled=False)}
    assert not resolve_voice_policy("beta", _ROOT, {}, disabled, seat_is_root=False).is_auto_vc(_LEGACY_VC)


def test_root_never_beats_a_legacy_auto_join():
    root = _cfg({_LEGACY_VC: {"join": "never"}})
    alpha = _cfg(legacy=[_LEGACY_VC])
    p = resolve_voice_policy("alpha", root, alpha, {"default": root, "alpha": alpha}, seat_is_root=False)
    assert _LEGACY_VC not in p.auto_join_channels() and not p.is_auto_vc(_LEGACY_VC)


def test_a_root_manual_entry_beats_a_seats_legacy_auto_join():
    """V-Q2: only the root marks auto. A seat's legacy ``voice_auto_join.channel_ids`` count as auto
    only for VCs the root map does not list; a root entry wins, with a WARNING."""
    alpha = _cfg(legacy=[_MANUAL_VC])
    profiles = {"default": _ROOT, "alpha": alpha, "beta": {}}
    p_alpha = resolve_voice_policy("alpha", _ROOT, alpha, profiles, seat_is_root=False)
    assert p_alpha.rule(_MANUAL_VC).join == "manual" and p_alpha.rule(_MANUAL_VC).joinable
    assert _MANUAL_VC not in p_alpha.auto_join_channels()
    assert any(_MANUAL_VC in w and "root" in w for w in p_alpha.seat_layer.warnings)


def test_r1_does_not_count_a_legacy_id_the_root_lists_as_manual():
    """The fleet R1 set: alpha's legacy id on a root-manual VC must not make /voice join refuse there
    for every seat (beta here). An unlisted legacy VC still counts (control)."""
    alpha = _cfg(legacy=[_MANUAL_VC, _LEGACY_VC])
    profiles = {"default": _ROOT, "alpha": alpha, "beta": {}}
    p_beta = resolve_voice_policy("beta", _ROOT, {}, profiles, seat_is_root=False)
    assert not p_beta.is_auto_vc(_MANUAL_VC)
    assert p_beta.is_auto_vc(_LEGACY_VC)


def test_a_root_auto_entry_for_other_seats_is_not_widened_by_a_legacy_id():
    """The root entry's ``seats`` decide who auto-joins it; a seat outside them listing it in its
    legacy ids does not auto-join (and it stays auto for R1 through the root entry)."""
    alpha = _cfg(legacy=[_AUTO_VC])
    profiles = {"default": _ROOT, "alpha": alpha}
    p_alpha = resolve_voice_policy("alpha", _ROOT, alpha, profiles, seat_is_root=False)
    assert _AUTO_VC not in p_alpha.auto_join_channels() and not p_alpha.rule(_AUTO_VC).seat_allowed
    assert p_alpha.is_auto_vc(_AUTO_VC)


def test_the_loader_warns_when_a_root_entry_overrides_a_legacy_id(fleet, caplog):
    _write(fleet / "profiles" / "alpha" / "config.yaml", _cfg(legacy=[_MANUAL_VC]))
    with caplog.at_level(logging.WARNING, logger="gateway.voice_channel_policy"):
        p = load_voice_policy("alpha")
    assert p.rule(_MANUAL_VC).join == "manual"
    assert any(r.levelno == logging.WARNING and _MANUAL_VC in r.getMessage() for r in caplog.records)


def test_a_root_auto_entry_without_seats_is_a_config_error_that_fails_closed(fleet, caplog):
    """TARS ruling (F8): ``join: auto`` with no ``seats`` must not default to "*" (every seat with
    auto-join on would race for the VC). It is an invalid entry: that VC is refused for every seat,
    auto-joined by none, and counts as auto for R1; the loader warns. Explicit ``seats: "*"`` is
    still accepted (control), and other VCs are unaffected."""
    root = _cfg({_AUTO_VC: {"join": "auto"}, _MANUAL_VC: {"join": "manual"}, _OTHER_VC: {"join": "auto", "seats": "*"}})
    for seat in ("default", "alpha"):
        p = _policy(seat=seat, root=root, seat_is_root=seat == "default")
        rule = p.rule(_AUTO_VC)
        assert rule.refusal == "invalid" and not rule.joinable and p.is_auto_vc(_AUTO_VC)
        assert p.auto_join_channels() == frozenset({_OTHER_VC})
        assert p.rule(_MANUAL_VC).joinable and p.problems
    _write(fleet / "config.yaml", root)
    with caplog.at_level(logging.WARNING, logger="gateway.voice_channel_policy"):
        load_voice_policy("default")
    assert any(r.levelno == logging.WARNING and _AUTO_VC in r.getMessage() and "seats" in r.getMessage()
               for r in caplog.records)


def test_root_auto_entries_drive_the_named_seats_auto_join():
    p = resolve_voice_policy("default", _ROOT, _ROOT, {"default": _ROOT}, seat_is_root=True)
    assert p.auto_join_channels() == frozenset({_AUTO_VC})
    assert _policy(seat="alpha").auto_join_channels() == frozenset()  # not in seats


# --- the loader --------------------------------------------------------------------------------

def _write(path: Path, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg if isinstance(cfg, str) else yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture
def fleet(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "hroot"
    _write(root / "config.yaml", _ROOT)
    _write(root / "profiles" / "alpha" / "config.yaml",
           _cfg({_MANUAL_VC: {"allow_thread_move": False}}, legacy=[_LEGACY_VC]))
    _write(root / "profiles" / "beta" / "config.yaml", _cfg())
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def test_loader_reads_the_map_from_the_root_even_inside_a_profile_home(fleet, monkeypatch):
    """The process home of a profile-scoped gateway is the PROFILE dir; the map lives in the root."""
    monkeypatch.setenv("HERMES_HOME", str(fleet / "profiles" / "beta"))
    p = load_voice_policy("beta")
    assert p.rule(_AUTO_VC).join == "auto" and p.rule(_MANUAL_VC).allow_thread_move is True
    assert p.is_auto_vc(_LEGACY_VC)  # alpha's legacy auto-join, with alpha not connected
    assert load_voice_policy("alpha").rule(_MANUAL_VC).allow_thread_move is False  # alpha narrows
    assert not p.problems


def test_an_unreadable_served_profile_config_refuses_every_join(fleet):
    _write(fleet / "profiles" / "beta" / "config.yaml", "discord: [unclosed\n")
    p = load_voice_policy("alpha")
    assert p.problems and not p.rule(_MANUAL_VC).joinable and p.is_auto_vc(_MANUAL_VC)
    assert p.auto_join_channels() == frozenset()


def test_the_loader_parses_through_the_shared_fast_reader(fleet, monkeypatch):
    """Every /voice join and seat connect parses every served profile's config on the event loop:
    for the live fleet (33 configs) the pure-Python PyYAML reader measured ~233 ms, the shared
    C-backed reader ~60 ms. Pin the mechanism: each config file goes through ``utils.fast_safe_load``."""
    import utils
    calls = []
    real = utils.fast_safe_load

    def spy(stream):
        calls.append(getattr(stream, "name", stream))
        return real(stream)

    monkeypatch.setattr(utils, "fast_safe_load", spy)
    p = load_voice_policy("alpha")
    assert sorted(Path(c).relative_to(fleet).as_posix() for c in calls) == [
        "config.yaml", "profiles/alpha/config.yaml", "profiles/beta/config.yaml"]
    assert p.rule(_MANUAL_VC).allow_thread_move is False and p.is_auto_vc(_LEGACY_VC) and not p.problems


def test_a_config_with_a_utf8_bom_still_parses(fleet):
    (fleet / "config.yaml").write_bytes(b"\xef\xbb\xbf" + yaml.safe_dump(_ROOT).encode())
    p = load_voice_policy("alpha")
    assert not p.problems and p.rule(_AUTO_VC).join == "auto"


def test_the_summary_names_the_seat_and_its_resolved_policy(fleet):
    text = load_voice_policy("alpha").summary()
    assert text.startswith("seat=alpha unlisted=allow")
    assert f"auto_join=['{_LEGACY_VC}']" in text and f"{_MANUAL_VC}:manual/bind=caller" in text


def test_the_adapter_logs_one_policy_line_per_seat_at_connect(fleet, caplog):
    pytest.importorskip("discord")
    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._owner_profile = "alpha"
    with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
        adapter._resolve_voice_channel_policy()
    lines = [r.getMessage() for r in caplog.records if "Voice channel policy" in r.getMessage()]
    assert len(lines) == 1 and "seat=alpha" in lines[0]
    assert adapter._voice_channel_policy.seat == "alpha"
