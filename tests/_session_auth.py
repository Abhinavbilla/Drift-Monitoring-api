"""
Shared helper for validation/test scripts: mints a session token the same
way dashboard.py's mint_session_token() does. Backend auth is derived from
Google login rather than a static API key (see main.py's verify_access),
and these scripts have no browser/OAuth flow of their own — but since they
already have direct access to COOKIE_KEY (the same secret shared with the
backend), they can legitimately mint their own token the same way.
"""

import os
import time

import jwt


def mint_session_token(email: str, name: str = "Validation Suite") -> str:
    cookie_key = os.getenv("COOKIE_KEY")
    if not cookie_key:
        raise ValueError("COOKIE_KEY not found in environment. Please set it in .env file.")

    payload = {
        "email": email,
        "name": name,
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, cookie_key, algorithm="HS256")
