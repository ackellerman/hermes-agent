"""``hermes pairing`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_pairing_parser(subparsers, *, cmd_pairing: Callable) -> None:
    """Attach the ``pairing`` subcommand to ``subparsers``."""
    pairing_parser = subparsers.add_parser(
        "pairing", help="Manage DM pairing codes for user authorization",
        description="Approve or revoke user access via pairing codes")
    pairing_sub = pairing_parser.add_subparsers(dest="pairing_action")

    pairing_sub.add_parser("list", help="Show pending + approved users")

    pairing_approve_parser = pairing_sub.add_parser("approve", help="Approve a pairing request")
    pairing_approve_parser.add_argument(
        "platform", help="Platform name (telegram, discord, slack, whatsapp)")
    pairing_approve_parser.add_argument(
        "code", metavar="request-id|code",
        help="Request ID from 'pairing list', or the code the bot DM'd the user")

    pairing_revoke_parser = pairing_sub.add_parser(
        "revoke",
        help="Revoke user access (pairing record; --effective also removes config.yaml grants)",
        description=(
            "Remove the user's pairing record. A config.yaml allow list is an INDEPENDENT grant: "
            "plain revoke reports it and exits 3 while it still authorizes. --effective also "
            "removes removable config.yaml/.env entries, so the files stop granting the user (a "
            "separately running gateway applies the change when it next loads its config). "
            "Exit codes: 0 nothing names the user and no live gateway still enforces the old list, "
            "3 a named grant survives, cannot be evaluated, or a running gateway has not reloaded, "
            "1 refused (config.yaml unreadable/pinned, managed .env) — nothing written."))
    pairing_revoke_parser.add_argument("platform", help="Platform name")
    pairing_revoke_parser.add_argument("user_id", help="User ID to revoke")
    pairing_revoke_parser.add_argument(
        "--effective", action="store_true",
        help="Also remove removable config.yaml/.env grants for this user (never wildcards, "
             "allow-all flags, role lists or the cross-platform global allowlist)")

    pairing_sub.add_parser("clear-pending", help="Clear all pending codes")
    pairing_parser.set_defaults(func=cmd_pairing)
