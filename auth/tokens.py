"""
Personal access token (PAT) generation, hashing, and verification.

Token format: dm_<prefix>_<secret>
  prefix: a short, random, non-secret identifier (safe to store and
          display in cleartext) used for O(1) DB lookup by prefix,
          before any hash comparison.
  secret: secrets.token_urlsafe(32) -- the actual credential.

Only a SHA-256 hash of the FULL presented token (prefix and secret both,
exactly as the caller sends it in the Authorization header) is ever
persisted -- the plaintext token is shown to the caller once, at creation
time, and never stored or logged anywhere after that. Verifying a
presented token: look up the row by its prefix, then compare hashes with
hmac.compare_digest (constant-time, avoids a timing side-channel on the
hash comparison itself).
"""

import hashlib
import hmac
import secrets
from typing import Optional, Tuple

TOKEN_PREFIX = "dm"


def generate_token() -> Tuple[str, str, str]:
    """Returns (full_token, prefix, token_hash). full_token is shown to
    the caller exactly once; only prefix and token_hash are persisted.

    The prefix uses token_hex, not token_urlsafe: token_urlsafe's alphabet
    includes '_' and '-', so an underscore-delimited prefix generated with
    it could itself contain an underscore and break split("_", 2) parsing
    (parse_prefix would then split at the WRONG point and never find a
    match, even for a valid token). token_hex's alphabet is 0-9a-f only,
    so the prefix can never collide with the '_' delimiter."""
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    full_token = f"{TOKEN_PREFIX}_{prefix}_{secret}"
    return full_token, prefix, hash_token(full_token)


def hash_token(full_token: str) -> str:
    return hashlib.sha256(full_token.encode("utf-8")).hexdigest()


def parse_prefix(full_token: str) -> Optional[str]:
    """Extracts the prefix segment from a presented token, or None if it
    doesn't match the dm_<prefix>_<secret> shape at all -- callers use
    this to decide whether to attempt PAT lookup vs. falling back to
    session-JWT decoding."""
    if not isinstance(full_token, str) or not full_token.startswith(f"{TOKEN_PREFIX}_"):
        return None
    parts = full_token.split("_", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1]


def verify_token_hash(full_token: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_token(full_token), stored_hash)
