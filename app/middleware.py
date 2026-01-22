"""
Session validation middleware for nimamanafcom.

Protects routes requiring authentication by checking session cookies.
Redirects to /login if session is invalid or expired.

Algorithm:
1. Check if request path matches protected patterns
2. If protected:
   a. Extract session cookie
   b. Validate token signature and expiry
   c. If invalid: redirect to /login with 303 (See Other)
   d. If valid: proceed to route handler
3. If not protected: proceed without check

Protected paths:
- /dashboard and /dashboard/*
- /terminal and /terminal/*
- /ralph and /ralph/*
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse

from app.auth import SESSION_COOKIE_NAME, validate_session_token


# Paths that require authentication
PROTECTED_PREFIXES = ["/dashboard", "/terminal", "/ralph"]


def is_protected_path(path: str) -> bool:
    """Check if a path requires authentication."""
    for prefix in PROTECTED_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates session cookies on protected routes.

    For protected paths, checks the session cookie. If missing or invalid,
    redirects to /login. Otherwise, allows the request to proceed.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Check if this is a protected route
        if is_protected_path(path):
            # Get session cookie
            session_token = request.cookies.get(SESSION_COOKIE_NAME)

            # Validate token
            if not validate_session_token(session_token):
                # Invalid or missing session - redirect to login
                return RedirectResponse(url="/login", status_code=303)

        # Not protected or valid session - proceed
        return await call_next(request)
