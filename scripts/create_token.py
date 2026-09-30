"""
Personal access token management (Step 3a). Talks directly to drift.db via
db.crud -- there is no HTTP endpoint or UI for token management (by design,
per the 2026-09-30 instruction: no UI until the React frontend exists).

Usage:
    python scripts/create_token.py create --email alice@example.com --name "ci-pipeline" \
        --project my_project_1 --project my_project_2
    python scripts/create_token.py create --email alice@example.com --name "admin-script" \
        --all-projects --expires-days 90
    python scripts/create_token.py create --email alice@example.com --name "long-lived" \
        --all-projects --no-expiry
    python scripts/create_token.py list --email alice@example.com
    python scripts/create_token.py revoke --email alice@example.com --id <token-id>

Default expiry is 90 days (2026-10-01 cleanup) -- pass --no-expiry for a
token that never expires; that's an explicit opt-in now, not the silent
default.

The full token is printed ONCE, at creation time, and is never stored or
recoverable afterward -- only its SHA-256 hash persists in api_tokens.
"""

import argparse
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from db import crud  # noqa: E402
from auth.tokens import generate_token  # noqa: E402


def cmd_create(args):
    if not args.all_projects and not args.project:
        print("ERROR: specify --project (repeatable) or --all-projects.", file=sys.stderr)
        sys.exit(1)

    project_scope = ["*"] if args.all_projects else list(args.project)
    full_token, prefix, token_hash = generate_token()
    token_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    expires_at = (
        None if args.no_expiry
        else (datetime.now(timezone.utc) + timedelta(days=args.expires_days)).isoformat()
    )

    crud.init_db()
    crud.create_api_token(
        token_id=token_id, user_email=args.email, name=args.name, prefix=prefix,
        token_hash=token_hash, project_scope=project_scope,
        created_at=created_at, expires_at=expires_at,
    )

    print("Token created. This is the ONLY time the full token is shown -- store it now.")
    print()
    print(f"  Token:       {full_token}")
    print(f"  Token ID:    {token_id}")
    print(f"  User:        {args.email}")
    print(f"  Name:        {args.name}")
    print(f"  Scope:       {'ALL projects owned by this user' if args.all_projects else ', '.join(project_scope)}")
    print(f"  Expires:     {expires_at or 'never'}")


def cmd_list(args):
    crud.init_db()
    tokens = crud.list_api_tokens(args.email)
    if not tokens:
        print(f"No tokens for {args.email}.")
        return
    for t in tokens:
        status_str = "REVOKED" if t["revoked"] else "active"
        scope_str = "ALL" if t["project_scope"] == ["*"] else ", ".join(t["project_scope"] or [])
        print(f"  id={t['id']}  name={t['name']!r}  prefix={t['prefix']}  status={status_str}  "
              f"scope=[{scope_str}]  created={t['created_at']}  expires={t['expires_at'] or 'never'}  "
              f"last_used={t['last_used_at'] or 'never'}")


def cmd_revoke(args):
    crud.init_db()
    ok = crud.revoke_api_token(args.id, args.email)
    if ok:
        print(f"Revoked token {args.id} for {args.email}.")
    else:
        print(f"No active token with id={args.id} found for {args.email} (already revoked, "
              f"wrong owner, or doesn't exist).", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Mint a new personal access token.")
    p_create.add_argument("--email", required=True, help="Owning user's email.")
    p_create.add_argument("--name", required=True, help="Human-readable label for this token.")
    p_create.add_argument("--project", action="append", default=[],
                           help="A project_id this token may access. Repeatable.")
    p_create.add_argument("--all-projects", action="store_true",
                           help="Scope this token to all of this user's projects, present and future.")
    p_create.add_argument("--expires-days", type=int, default=90,
                           help="Days until expiry (default 90, 2026-10-01 cleanup). Pass --no-expiry "
                                "for a token that never expires -- that's now an explicit opt-in, not "
                                "the silent default.")
    p_create.add_argument("--no-expiry", action="store_true",
                           help="Token never expires. Overrides --expires-days.")
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List a user's tokens.")
    p_list.add_argument("--email", required=True)
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke", help="Revoke a token by id.")
    p_revoke.add_argument("--email", required=True, help="Must match the token's owner.")
    p_revoke.add_argument("--id", required=True, help="Token id (from `list`).")
    p_revoke.set_defaults(func=cmd_revoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
