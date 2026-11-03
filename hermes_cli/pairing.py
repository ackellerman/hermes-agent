"""CLI commands for the DM pairing system.

``revoke`` has two documented modes. It ALWAYS removes the pairing record; a ``config.yaml``
allow list is an independent grant the command must not edit silently:

* plain ``revoke`` — pairing record only. Every other source that still names the user is
  reported, and the exit code says whether one still authorizes.
* ``revoke --effective`` — also removes the removable sources (``config.yaml`` entries, the
  profile ``.env`` allowlist), so the FILES stop granting the user. A caller inside the gateway
  process also drops the live adapter snapshot; a separately running gateway applies the change
  when it next loads its config. Wildcards, allow-all
  flags, role lists, the cross-platform global allowlist and anything that cannot be evaluated
  offline are reported and left alone.

Exit codes: ``0`` no known grant names the user; ``3`` completed, but a named grant survives or
cannot be evaluated offline; ``1`` refused (unreadable/pinned config.yaml, managed ``.env``) —
nothing was written.
"""


def pairing_command(args):
    """Handle hermes pairing subcommands."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    handlers = {
        "list": lambda: _cmd_list(store),
        "approve": lambda: _cmd_approve(store, args.platform, args.code),
        "revoke": lambda: _cmd_revoke(
            store, args.platform, args.user_id, effective=bool(getattr(args, "effective", False))),
        "clear-pending": lambda: _cmd_clear_pending(store),
    }
    handler = handlers.get(getattr(args, "pairing_action", None))
    if handler is None:
        print("Usage: hermes pairing {list|approve|revoke|clear-pending}")
        print("Run 'hermes pairing --help' for details.")
    else:
        handler()

def _cmd_list(store):
    """List all pending and approved users."""
    pending = store.list_pending()
    approved = store.list_approved()
    if not pending and not approved:
        print("No pairing data found. No one has tried to pair yet~")
        return

    if pending:
        print(f"\n  Pending Pairing Requests ({len(pending)}):")
        print(f"  {'Platform':<12} {'Request ID':<18} {'User ID':<20} {'Name':<20} {'Age'}")
        print(f"  {'--------':<12} {'----------':<18} {'-------':<20} {'----':<20} {'---'}")
        for p in pending:
            print(
                f"  {p['platform']:<12} {(p.get('request_id') or '-'):<18} {p['user_id']:<20} "
                f"{(p.get('user_name') or ''):<20} {p['age_minutes']}m ago"
            )
        print("\n  Approve with: hermes pairing approve <platform> <request-id>")
        print("  The code the bot DM'd the user also works if they relay it.")
    else:
        print("\n  No pending pairing requests.")

    if approved:
        print(f"\n  Approved Users ({len(approved)}):")
        print(f"  {'Platform':<12} {'User ID':<20} {'Name':<20}")
        print(f"  {'--------':<12} {'-------':<20} {'----':<20}")
        for a in approved:
            print(f"  {a['platform']:<12} {a['user_id']:<20} {(a.get('user_name') or ''):<20}")
    else:
        print("\n  No approved users.")

    print()


def _cmd_approve(store, platform: str, code: str):
    """Approve a pairing request id (from ``pairing list``) or a DM'd code."""
    platform = platform.lower().strip()
    code = code.strip()

    if store.looks_like_request_id(code):
        result = store.approve_request(platform, code)
    else:
        result = store.approve_code(platform, code.upper())
    if result:
        uid, name = result["user_id"], result.get("user_name") or ""
        display = f"{name} ({uid})" if name else uid
        print(f"\n  Approved! User {display} on {platform} can now use the bot~")
        print("  They'll be recognized automatically on their next message.\n")
    elif store._is_locked_out(platform):
        # approve_code returns None for both invalid codes and lockout — say which.
        # Tell the operator it's lockout so they don't chase a "wrong code" rabbit hole (#10195).
        import time as _time
        lockout_until = store._load_json(store._rate_limit_path()).get(f"_lockout:{platform}", 0)
        mins = max(0, int(lockout_until - _time.time())) // 60
        print(f"\n  Platform '{platform}' is locked out after too many failed approval attempts.")
        print(f"  Lockout clears in ~{mins} minute(s).")
        print(f"  To reset sooner, delete the '_lockout:{platform}' entry from ~/.hermes/platforms/pairing/_rate_limits.json\n")
    else:
        print(f"\n  Pairing request or code '{code}' not found or expired for platform '{platform}'.")
        print("  Run 'hermes pairing list' to see pending requests.\n")


def _print_surviving(sources) -> None:
    """One line per source that still names the user (or cannot be evaluated offline)."""
    for source in sources:
        state = "authorizes now" if source.authorizes_now is True else "cannot be evaluated offline"
        detail = f" — {source.reason}" if source.reason else ""
        print(f"  Still named by {source.location} [{source.chat_scope}] ({state}){detail}")


def _cmd_revoke(store, platform: str, user_id: str, effective: bool = False):
    """Revoke a user's access. See the module docstring for the two modes and the exit codes."""
    from gateway.access_grants import AccessGrantError, find_user_grants, remove_user_grants

    platform = platform.lower().strip()
    removed = []
    try:
        grants = find_user_grants(platform, user_id, store=store)
        if effective:
            # Config/env removals validate BEFORE writing anything, and the pairing record is not
            # touched until they succeed — a refusal must leave the whole state as it was.
            removed = remove_user_grants(
                [source for source in grants if source.kind != "pairing"], store=store)
    except AccessGrantError as exc:
        print(f"\n  Refused: {exc}\n")
        print("  Nothing was changed: no config.yaml entry, no .env key, pairing record intact.\n")
        raise SystemExit(1)

    revoked = store.revoke(platform, user_id)
    grants = find_user_grants(platform, user_id, store=store)

    print()
    if revoked:
        print(f"  Revoked pairing access for user {user_id} on {platform}.")
    else:
        print(f"  No pairing record for user {user_id} on {platform}.")

    seen_locations: list[str] = []
    for source in removed:
        if source.location in seen_locations:
            continue  # one line per list, even when it held two spellings of the same principal
        seen_locations.append(source.location)
        print(f"  Removed {source.location} [{source.chat_scope}]")

    surviving = [source for source in grants if source.authorizes_now is not False]
    if surviving:
        _print_surviving(surviving)
        print()
        if not effective:
            print("  Revoking the pairing record does not edit your config.yaml.")
            print(f"  To deny this user across every configured source: "
                  f"hermes pairing revoke --effective {platform} {user_id}")
    else:
        print(f"  No other configured source names user {user_id} on {platform}.")
    print()

    raise SystemExit(3 if surviving else 0)


def _cmd_clear_pending(store):
    """Clear all pending pairing codes."""
    count = store.clear_pending()
    if count:
        print(f"\n  Cleared {count} pending pairing request(s).\n")
    else:
        print("\n  No pending requests to clear.\n")
