"""
Email + password sign-in, alongside Google.

Projects are keyed by email, so the one rule that matters most: an account
gets a session only after its owner proves they control the address (a
6-digit code sent there). Otherwise anyone could type someone else's email
and open their projects. A Google user who later creates a password for the
same address goes through the same check and then sees the same projects.

- Passwords: scrypt (stdlib) with a per-user random salt; constant-time compare.
- Codes: 6 digits, stored only as an HMAC (keyed with COOKIE_KEY), valid for
  15 minutes, at most 5 tries, at most one new code per minute.
- Sign-in: 5 wrong passwords lock the account for 15 minutes.
- Responses never say whether an account exists (no email enumeration).
- Email is sent over SMTP (ALERT_EMAIL / ALERT_PASSWORD, SMTP_HOST, SMTP_PORT).
  With no SMTP configured, email sign-in is refused -- unless
  DRIFT_DEV_PRINT_EMAIL_CODES=1, which prints codes to the server log for
  local development only.
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Dict, Optional

from db import crud

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8
CODE_TTL_MINUTES = 15
MAX_CODE_ATTEMPTS = 5
RESEND_COOLDOWN_SECONDS = 60
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
WRONG_CREDENTIALS = "That email and password don't match. Check them and try again."


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- validation and hashing

def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AuthError(422, "Please enter a valid email address.")
    return email


def validate_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(422, f"Use at least {MIN_PASSWORD_LENGTH} characters for your password.")
    if len(password) > 256:
        raise AuthError(422, "That password is too long.")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    enc = lambda b: base64.b64encode(b).decode()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${enc(salt)}${enc(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


_dummy_hash: Optional[str] = None


def _burn_time(password: str) -> None:
    """Same scrypt cost as a real check, so response time doesn't reveal whether the account exists."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_hex(16))
    verify_password(password, _dummy_hash)


# ---------------------------------------------------------------- email

def email_configured() -> bool:
    return bool(os.getenv("ALERT_EMAIL") and os.getenv("ALERT_PASSWORD")) or os.getenv("DRIFT_DEV_PRINT_EMAIL_CODES") == "1"


def send_email(to: str, subject: str, body: str) -> None:
    sender, password = os.getenv("ALERT_EMAIL"), os.getenv("ALERT_PASSWORD")
    if not (sender and password):
        if os.getenv("DRIFT_DEV_PRINT_EMAIL_CODES") == "1":
            print(f"[dev email] to={to} subject={subject!r}\n{body}")
            return
        raise AuthError(503, "Email sign-in isn't set up on this server yet. Please use Google sign-in.")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, f"Drift Sentinel <{sender}>", to
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "465")), timeout=15) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
    except (smtplib.SMTPException, OSError):
        raise AuthError(503, "We couldn't send the email right now. Please try again in a few minutes.")


# ---------------------------------------------------------------- one-time codes

def _code_hash(email: str, purpose: str, code: str) -> str:
    key = os.getenv("COOKIE_KEY", "").encode()
    return hmac.new(key, f"{purpose}:{email}:{code}".encode(), hashlib.sha256).hexdigest()


def issue_code(email: str, purpose: str) -> None:
    existing = crud.get_email_code(email, purpose)
    if existing and _now() - datetime.fromisoformat(existing["created_at"]) < timedelta(seconds=RESEND_COOLDOWN_SECONDS):
        raise AuthError(429, "We just sent you a code. Please wait a minute before asking for another.")
    code = f"{secrets.randbelow(10 ** 6):06d}"
    crud.save_email_code(email, purpose, _code_hash(email, purpose, code),
                         (_now() + timedelta(minutes=CODE_TTL_MINUTES)).isoformat())
    action = "confirm your email address" if purpose == "verify" else "reset your password"
    send_email(email, f"Your Drift Sentinel code: {code}",
               f"Hi,\n\nUse this code to {action}:\n\n    {code}\n\nIt expires in {CODE_TTL_MINUTES} minutes. "
               f"If you didn't ask for it, you can ignore this email.\n\n— Drift Sentinel")


def _check_code(email: str, purpose: str, code: str) -> None:
    row = crud.get_email_code(email, purpose)
    if row is None or _now() > datetime.fromisoformat(row["expires_at"]):
        raise AuthError(400, "That code has expired or was already used. Ask for a new one.")
    if row["attempts"] >= MAX_CODE_ATTEMPTS:
        crud.delete_email_code(email, purpose)
        raise AuthError(429, "Too many wrong codes. Ask for a new one.")
    if not hmac.compare_digest(row["code_hash"], _code_hash(email, purpose, (code or "").strip())):
        crud.bump_email_code_attempts(email, purpose)
        raise AuthError(400, "That code isn't right. Check the email and try again.")
    crud.delete_email_code(email, purpose)


# ---------------------------------------------------------------- flows

def _issue_quietly(email: str, purpose: str) -> None:
    """For flows that must answer the same for unknown addresses: a cooldown
    "please wait" would otherwise reveal that the account exists."""
    try:
        issue_code(email, purpose)
    except AuthError as e:
        if e.status != 429:
            raise


def register(email: str, password: str, name: Optional[str]) -> None:
    """Starts sign-up by emailing a code. The same answer is given whether or
    not the address already has an account; an existing owner just gets a heads-up."""
    email = normalize_email(email)
    validate_password(password)
    user = crud.get_user(email)
    if user and user["verified"]:
        send_email(email, "Someone tried to create a Drift Sentinel account with your email",
                   "Hi,\n\nYou already have an account, so nothing changed. If this was you, sign in instead, "
                   "or use “Forgot password”.\n\n— Drift Sentinel")
        return
    crud.upsert_unverified_user(email, (name or "").strip() or email.split("@")[0], hash_password(password))
    issue_code(email, "verify")


def verify_email(email: str, code: str) -> Dict:
    email = normalize_email(email)
    _check_code(email, "verify", code)
    crud.update_user(email, verified=1)
    return crud.get_user(email)


def login(email: str, password: str) -> Dict:
    email = normalize_email(email)
    user = crud.get_user(email)
    if user is None:
        _burn_time(password or "")
        raise AuthError(401, WRONG_CREDENTIALS)
    if user["locked_until"] and _now() < datetime.fromisoformat(user["locked_until"]):
        raise AuthError(429, "Too many attempts. Please wait a few minutes, or reset your password.")
    if not verify_password(password or "", user["password_hash"]):
        failed = user["failed_logins"] + 1
        if failed >= MAX_FAILED_LOGINS:
            crud.update_user(email, failed_logins=0, locked_until=(_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat())
        else:
            crud.update_user(email, failed_logins=failed)
        raise AuthError(401, WRONG_CREDENTIALS)
    crud.update_user(email, failed_logins=0, locked_until=None)
    if not user["verified"]:
        try:
            issue_code(email, "verify")
        except AuthError:
            pass  # a code was sent moments ago; the message below still applies
        raise AuthError(403, "Please confirm your email first — we've sent you a code.")
    return user


def resend_verification(email: str) -> None:
    email = normalize_email(email)
    user = crud.get_user(email)
    if user and not user["verified"]:
        _issue_quietly(email, "verify")


def forgot_password(email: str) -> None:
    email = normalize_email(email)
    if crud.get_user(email):
        _issue_quietly(email, "reset")


def reset_password(email: str, code: str, new_password: str) -> Dict:
    email = normalize_email(email)
    validate_password(new_password)
    _check_code(email, "reset", code)
    # Receiving the code proves ownership, so this also verifies an unverified account.
    crud.update_user(email, password_hash=hash_password(new_password), verified=1, failed_logins=0, locked_until=None)
    return crud.get_user(email)
