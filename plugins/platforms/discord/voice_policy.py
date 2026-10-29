"""
Opt-in Discord voice auto-join policy — pure logic, zero discord.py imports.

SAFETY RATIONALE
----------------
Voice auto-join is OFF by default for good reason: once the bot joins a voice
channel it starts transcribing everyone who speaks.  Automatic transcription
of users who did not opt in is a privacy risk.  Therefore the policy requires
**both** allowlists to be non-empty before it will activate:

  * ``channel_ids`` — which voice channels the bot may auto-join.
  * ``user_ids`` — which Discord users may trigger an auto-join.

Without both lists populated, activation_error() returns a descriptive string
and should_join() returns False.  There is deliberately NO "allow all channels"
or "allow all users" fallthrough — that would transcribe everyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class AutoJoinPolicy:
    """Immutable auto-join policy for a Discord guild.

    Both ``channel_ids`` and ``user_ids`` must be non-empty for the
    policy to activate (safety gate — see module docstring).
    """

    enabled: bool = False
    channel_ids: FrozenSet[str] = field(default_factory=frozenset)
    user_ids: FrozenSet[str] = field(default_factory=frozenset)
    join_mode: str = "user_prompt"  # "user_prompt" | "automatic"
    require_text_opt_in: bool = True


class VoicePolicyEvaluator:
    """Stateless evaluator for :class:`AutoJoinPolicy`.

    All methods are pure functions — no I/O, no imports from ``discord``.
    """

    @staticmethod
    def activation_error(policy: AutoJoinPolicy) -> Optional[str]:
        """Return an error string if ``policy`` cannot activate, else None.

        Checks (in order):
        1. ``enabled`` is False.
        2. ``channel_ids`` is empty (no voice channels allowlisted).
        3. ``user_ids`` is empty (no users allowlisted — privacy gate).
        """
        if not policy.enabled:
            return "voice_auto_join is disabled"
        if not policy.channel_ids:
            return "voice_auto_join.channel_ids must list at least one voice channel ID"
        if not policy.user_ids:
            return "voice_auto_join.user_ids must list at least one user ID"
        return None

    @staticmethod
    def should_join(
        policy: AutoJoinPolicy,
        *,
        channel_id: str,
        member_id: str,
        member_has_text_access: bool = True,
    ) -> Tuple[bool, str]:
        """Decide whether the bot should auto-join for this member+channel.

        Returns ``(True, "")`` when every check passes, or
        ``(False, "reason string")`` on the first failure.

        Checks (in order):
        1. Policy is activation-valid (see :meth:`activation_error`).
        2. ``channel_id`` is in ``policy.channel_ids``.
        3. ``member_id`` is in ``policy.user_ids``.
        4. If ``policy.require_text_opt_in`` then ``member_has_text_access``
           must be True.
        """
        err = VoicePolicyEvaluator.activation_error(policy)
        if err is not None:
            return False, err

        # Channel allowlist
        if channel_id not in policy.channel_ids:
            return False, "voice channel is not in voice_auto_join.channel_ids"

        # User allowlist
        if member_id not in policy.user_ids:
            return False, "user is not in voice_auto_join.user_ids"

        # Text-opt-in gate
        if policy.require_text_opt_in and not member_has_text_access:
            return False, "user lacks text-channel access (require_text_opt_in)"

        return True, ""
