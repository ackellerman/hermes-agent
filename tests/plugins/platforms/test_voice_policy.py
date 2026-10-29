"""
Tests for plugins/platforms/discord/voice_policy.py — pure logic, no mocking.

Every test exercises the dataclass and evaluator with plain Python values;
there are no discord.py imports anywhere in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import voice_policy.
_src = Path(__file__).resolve().parents[3]  # tests/ -> repo root
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest

from plugins.platforms.discord.voice_policy import AutoJoinPolicy, VoicePolicyEvaluator


# ---------------------------------------------------------------------------
# AutoJoinPolicy structural tests
# ---------------------------------------------------------------------------

class TestAutoJoinPolicy:
    """AutoJoinPolicy dataclass — construction, defaults, immutability."""

    def test_defaults_are_disabled(self):
        """Default-constructed policy has every safety-gate off."""
        p = AutoJoinPolicy()
        assert p.enabled is False
        assert p.channel_ids == frozenset()
        assert p.user_ids == frozenset()
        assert p.join_mode == "user_prompt"
        assert p.require_text_opt_in is True

    def test_explicit_construction(self):
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"123", "456"}),
            user_ids=frozenset({"789"}),
            join_mode="automatic",
            require_text_opt_in=False,
        )
        assert p.enabled is True
        assert p.channel_ids == frozenset({"123", "456"})
        assert p.user_ids == frozenset({"789"})
        assert p.join_mode == "automatic"
        assert p.require_text_opt_in is False

    def test_immutable_frozensets(self):
        """channel_ids and user_ids are frozenset (immutable)."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"1"}),
            user_ids=frozenset({"2"}),
        )
        with pytest.raises(AttributeError):
            p.channel_ids = frozenset()  # frozen dataclass

    def test_immutable_dataclass(self):
        p = AutoJoinPolicy(enabled=True)
        with pytest.raises(AttributeError):
            p.enabled = False


# ---------------------------------------------------------------------------
# activation_error tests
# ---------------------------------------------------------------------------

class TestActivationError:
    """VoicePolicyEvaluator.activation_error — all failure modes."""

    def test_disabled_returns_error(self):
        p = AutoJoinPolicy(
            enabled=False,
            channel_ids=frozenset({"123"}),
            user_ids=frozenset({"456"}),
        )
        err = VoicePolicyEvaluator.activation_error(p)
        assert err is not None
        assert "disabled" in err.lower()

    def test_empty_channel_ids_returns_error(self):
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset(),
            user_ids=frozenset({"456"}),
        )
        err = VoicePolicyEvaluator.activation_error(p)
        assert err is not None
        assert "channel_ids" in err.lower()

    def test_empty_user_ids_returns_error(self):
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"123"}),
            user_ids=frozenset(),
        )
        err = VoicePolicyEvaluator.activation_error(p)
        assert err is not None
        assert "user_ids" in err.lower()

    def test_both_empty_returns_first_error(self):
        """Both channel_ids and user_ids empty — first error wins."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset(),
            user_ids=frozenset(),
        )
        err = VoicePolicyEvaluator.activation_error(p)
        # Should report channel_ids first (checked before user_ids)
        assert err is not None
        assert "channel_ids" in err.lower()

    def test_all_prerequisites_met_returns_none(self):
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"123"}),
            user_ids=frozenset({"456"}),
        )
        assert VoicePolicyEvaluator.activation_error(p) is None


# ---------------------------------------------------------------------------
# should_join tests
# ---------------------------------------------------------------------------

class TestShouldJoin:
    """VoicePolicyEvaluator.should_join — allow/deny decisions."""

    FULL_POLICY = AutoJoinPolicy(
        enabled=True,
        channel_ids=frozenset({"111", "222"}),
        user_ids=frozenset({"aaa", "bbb"}),
        join_mode="user_prompt",
        require_text_opt_in=True,
    )

    # --- Activation gate ---

    def test_disabled_policy_denies(self):
        p = AutoJoinPolicy(enabled=False)
        ok, reason = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is False
        assert "disabled" in reason

    def test_empty_channel_ids_denies(self):
        p = AutoJoinPolicy(enabled=True, channel_ids=frozenset(), user_ids=frozenset({"aaa"}))
        ok, reason = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is False
        assert "channel_ids" in reason

    def test_empty_user_ids_denies(self):
        p = AutoJoinPolicy(enabled=True, channel_ids=frozenset({"111"}), user_ids=frozenset())
        ok, reason = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is False
        assert "user_ids" in reason

    # --- Channel allowlist ---

    def test_channel_not_in_allowlist_denies(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY, channel_id="999", member_id="aaa",
        )
        assert ok is False
        assert "channel_ids" in reason

    def test_channel_in_allowlist_allows(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY, channel_id="111", member_id="aaa",
        )
        assert ok is True
        assert reason == ""

    def test_second_channel_in_allowlist_allows(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY, channel_id="222", member_id="bbb",
        )
        assert ok is True
        assert reason == ""

    # --- User allowlist ---

    def test_user_not_in_allowlist_denies(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY, channel_id="111", member_id="zzz",
        )
        assert ok is False
        assert "user_ids" in reason

    def test_user_in_allowlist_allows(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY, channel_id="111", member_id="bbb",
        )
        assert ok is True
        assert reason == ""

    # --- require_text_opt_in ---

    def test_require_text_opt_in_no_access_denies(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY,
            channel_id="111",
            member_id="aaa",
            member_has_text_access=False,
        )
        assert ok is False
        assert "require_text_opt_in" in reason

    def test_require_text_opt_in_with_access_allows(self):
        ok, reason = VoicePolicyEvaluator.should_join(
            self.FULL_POLICY,
            channel_id="111",
            member_id="aaa",
            member_has_text_access=True,
        )
        assert ok is True
        assert reason == ""

    def test_require_text_opt_in_false_skips_gate(self):
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"111"}),
            user_ids=frozenset({"aaa"}),
            require_text_opt_in=False,
        )
        ok, reason = VoicePolicyEvaluator.should_join(
            p,
            channel_id="111",
            member_id="aaa",
            member_has_text_access=False,  # would deny if gate were on
        )
        assert ok is True
        assert reason == ""

    # --- Empty-vs-set allowlists ---

    def test_both_allowlists_populated_allows_correct_combinations(self):
        """Cross-product: every valid (channel, user) pair is allowed."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"100", "200"}),
            user_ids=frozenset({"u1", "u2"}),
        )
        for ch in ("100", "200"):
            for uid in ("u1", "u2"):
                ok, _ = VoicePolicyEvaluator.should_join(
                    p, channel_id=ch, member_id=uid,
                )
                assert ok is True, f"expected allow for ch={ch} uid={uid}"

    def test_both_allowlists_populated_denies_wrong_combinations(self):
        """Cross-product: every invalid channel or user is denied."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"100", "200"}),
            user_ids=frozenset({"u1", "u2"}),
        )
        # Wrong channel
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="999", member_id="u1",
        )
        assert ok is False
        # Wrong user
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="100", member_id="u9",
        )
        assert ok is False
        # Both wrong
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="999", member_id="u9",
        )
        assert ok is False

    def test_disabled_policy_with_full_allowlists_still_denies(self):
        """enabled=False overrides everything else."""
        p = AutoJoinPolicy(
            enabled=False,
            channel_ids=frozenset({"111"}),
            user_ids=frozenset({"aaa"}),
        )
        ok, reason = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is False
        assert "disabled" in reason


# ---------------------------------------------------------------------------
# join_mode normalization (tested through the dataclass, not evaluator)
# ---------------------------------------------------------------------------

class TestJoinMode:
    """join_mode is stored as-is by AutoJoinPolicy; normalization happens
    in the adapter's _load_voice_auto_join_config, not in voice_policy."""

    def test_valid_modes_accepted(self):
        for mode in ("user_prompt", "automatic"):
            p = AutoJoinPolicy(enabled=True, join_mode=mode)
            assert p.join_mode == mode

    def test_invalid_mode_stored_as_given(self):
        """voice_policy does NOT validate join_mode; the adapter does."""
        p = AutoJoinPolicy(enabled=True, join_mode="invalid")
        assert p.join_mode == "invalid"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary and edge-case checks."""

    def test_channel_id_as_int_string(self):
        """channel_ids are strings — '111' matches '111'."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"111"}),
            user_ids=frozenset({"aaa"}),
        )
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is True

    def test_string_vs_int_distinct(self):
        """String '111' is NOT frozenset({111}) — must compare as strings."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"111"}),
            user_ids=frozenset({"aaa"}),
        )
        # member_id as int (won't match 'aaa' string)
        ok, reason = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is True  # 'aaa' string matches

    def test_member_has_text_access_default_true(self):
        """Default value for member_has_text_access is True (compatible)."""
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=frozenset({"111"}),
            user_ids=frozenset({"aaa"}),
            require_text_opt_in=False,
        )
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="111", member_id="aaa",
        )
        assert ok is True

    def test_large_allowlists(self):
        """Large frozensets work correctly."""
        channels = frozenset(str(i) for i in range(100))
        users = frozenset(f"u{i}" for i in range(100))
        p = AutoJoinPolicy(
            enabled=True,
            channel_ids=channels,
            user_ids=users,
        )
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="50", member_id="u50",
        )
        assert ok is True
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="200", member_id="u50",
        )
        assert ok is False
        ok, _ = VoicePolicyEvaluator.should_join(
            p, channel_id="50", member_id="u200",
        )
        assert ok is False