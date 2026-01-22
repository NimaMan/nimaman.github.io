"""
Authentication module for nimamanafcom.

Handles password verification using bcrypt and session creation/validation.
Sessions use signed cookies via itsdangerous.

Algorithm:
1. Password verification:
   - Take plaintext password from form
   - Compare against bcrypt hash from PASSWORD_HASH env var
   - Return True/False (no timing attack info)

2. Session management:
   - Create signed session token with timestamp
   - Store in httponly, samesite=strict cookie
   - Validate signature and expiry on protected routes

Security notes:
- bcrypt handles timing-safe comparison internally
- Sessions are signed, not encrypted (tamper-proof, not secret)
- Session only contains timestamp, no sensitive data
"""

import time

import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from app.config import get_settings


def verify_password(password: str) -> bool:
    """
    Verify a plaintext password against the stored bcrypt hash.

    Args:
        password: Plaintext password from login form

    Returns:
        True if password matches, False otherwise
    """
    settings = get_settings()

    if not settings.password_hash:
        return False

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            settings.password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        # Invalid hash format or other bcrypt error
        return False


def create_session_token() -> str:
    """
    Create a signed session token.

    Token contains only the creation timestamp. Signature prevents tampering.

    Returns:
        Signed session token string
    """
    settings = get_settings()
    serializer = URLSafeTimedSerializer(settings.secret_key)

    return serializer.dumps({"created": int(time.time())})


def validate_session_token(token: str) -> bool:
    """
    Validate a session token's signature and expiry.

    Args:
        token: Session token from cookie

    Returns:
        True if token is valid and not expired, False otherwise
    """
    settings = get_settings()

    if not token or not settings.secret_key:
        return False

    serializer = URLSafeTimedSerializer(settings.secret_key)
    max_age_seconds = settings.session_timeout_hours * 3600

    try:
        serializer.loads(token, max_age=max_age_seconds)
        return True
    except (BadSignature, SignatureExpired):
        return False


# Cookie configuration constants
SESSION_COOKIE_NAME = "session"
SESSION_COOKIE_CONFIG = {
    "httponly": True,
    "samesite": "strict",
    "secure": False,  # Tailscale handles HTTPS, local app is HTTP
}
