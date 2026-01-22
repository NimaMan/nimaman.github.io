"""
Main FastAPI application for nimamanafcom.

Private personal dashboard accessible ONLY via Tailscale network.
Provides terminal access and Ralph AI management.

Algorithm:
1. Create FastAPI app instance
2. Configure templates and static files
3. Include routers for different sections (public, dashboard, terminal)
4. Mount Ralph app as sub-application at /ralph (NIMA-008)
5. Add health check endpoint
6. App binds to 127.0.0.1 only (Tailscale serve handles external routing)

Security Model:
- Layer 1: Tailscale network (primary - only your devices can connect)
- Layer 2: Password auth (secondary - defense in depth)
- Layer 3: Signed session cookies (prevents tampering)

Run with: uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import platform
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.auth import (
    SESSION_COOKIE_CONFIG,
    SESSION_COOKIE_NAME,
    create_session_token,
    verify_password,
)
from app.config import get_settings
from app.middleware import SessionMiddleware
from app.ralph_integration import get_ralph_static_dir, ralph_router
from app.terminal import terminal_websocket

# App metadata
APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

# Create FastAPI app
app = FastAPI(
    title="nimamanafcom",
    description="Private personal dashboard - Tailscale access only",
    version="1.0.0",
    docs_url=None,  # Disable Swagger UI in production
    redoc_url=None,  # Disable ReDoc in production
)

# Add session validation middleware for protected routes
app.add_middleware(SessionMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount Ralph's static files at /ralph/static (for Ralph dashboard)
app.mount("/ralph/static", StaticFiles(directory=get_ralph_static_dir()), name="ralph-static")

# Include Ralph router (protected by session middleware since /ralph/* is protected)
app.include_router(ralph_router)

# Configure templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/health")
async def health_check():
    """Health check endpoint. Returns 200 if the app is running."""
    settings = get_settings()
    missing = settings.validate()

    return {
        "status": "healthy" if not missing else "degraded",
        "app": "nimamanafcom",
        "host": settings.host,
        "port": settings.port,
        "config_missing": missing if missing else None,
    }


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """
    Landing page with personal info.

    No authentication required - user is already on Tailscale network.
    Displays name, brief bio, and link to dashboard.
    """
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Login page with password form.

    GET shows the login form. No error message on initial load.
    """
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    """
    Process login form submission.

    POST validates password against bcrypt hash from .env.
    Success: Creates session cookie, redirects to /dashboard.
    Failure: Shows login form again with generic error message.
    """
    if verify_password(password):
        # Create session and redirect to dashboard
        token = create_session_token()
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            **SESSION_COOKIE_CONFIG,
        )
        return response
    else:
        # Show error, no info leakage about what's wrong
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid password"},
            status_code=401,
        )


@app.get("/logout")
async def logout():
    """
    Clear session and redirect to landing page.

    Deletes the session cookie regardless of whether it exists.
    """
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Dashboard navigation hub.

    Requires valid session (checked by middleware).
    Shows system info and navigation to Terminal and Ralph.
    """
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "hostname": socket.gethostname(),
            "platform": platform.system() + " " + platform.release(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    )


@app.get("/terminal", response_class=HTMLResponse)
async def terminal_page(request: Request):
    """
    Terminal page with xterm.js client.

    Requires valid session (checked by middleware).
    Frontend connects to WebSocket at /ws/terminal.
    """
    return templates.TemplateResponse("terminal.html", {"request": request})


@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    """
    WebSocket endpoint for terminal access.

    Validates session cookie, spawns PTY, bridges to xterm.js.
    Session validation happens inside the handler (not middleware).
    """
    await terminal_websocket(websocket)
