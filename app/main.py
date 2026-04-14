"""
Main FastAPI application for nimamanafcom.

The app now serves two roles:
- a public-facing personal website with homepage, posts, and CV pages
- a protected dashboard with terminal access and Ralph integration

Run with: uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import platform
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
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
from app.site_data import (
    CV_EDUCATION,
    CV_EXPERIENCE,
    CV_HIGHLIGHTS,
    CV_PROFILE,
    CV_SKILLS,
    EXPERTISE,
    HOME_PAGE,
    LIFE_TIMELINE,
    SELECTED_WORK,
    SOCIAL_LINKS,
    get_post_by_slug,
    get_posts,
)
from app.terminal import terminal_websocket

# App metadata
APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

# Create FastAPI app
app = FastAPI(
    title="nimamanafcom",
    description="Personal website with a protected dashboard",
    version="1.0.0",
    docs_url=None,  # Disable Swagger UI in production
    redoc_url=None,  # Disable ReDoc in production
)

# Add session validation middleware for protected routes
app.add_middleware(SessionMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount Ralph's static files at /ralph/static when the checkout is available.
ralph_static_dir = get_ralph_static_dir()
if ralph_static_dir.exists():
    app.mount("/ralph/static", StaticFiles(directory=ralph_static_dir), name="ralph-static")

# Include Ralph router (protected by session middleware since /ralph/* is protected)
app.include_router(ralph_router)

# Configure templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def public_template_response(
    template_name: str,
    request: Request,
    *,
    active_page: str,
    **context,
):
    """Render a public website template with shared navigation/footer context."""
    return templates.TemplateResponse(
        request=request,
        name=template_name,
        context={
            "request": request,
            "active_page": active_page,
            "home_page": HOME_PAGE,
            "social_links": SOCIAL_LINKS,
            **context,
        },
    )


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
    Public homepage for the personal website.
    """
    posts = get_posts()
    return public_template_response(
        "home.html",
        request,
        active_page="home",
        featured_posts=posts[:3],
        expertise=EXPERTISE,
        life_timeline=LIFE_TIMELINE,
        selected_work=SELECTED_WORK,
    )


@app.get("/posts", response_class=HTMLResponse)
async def posts_page(request: Request):
    """List public posts imported from the old Hugo site."""
    return public_template_response(
        "posts.html",
        request,
        active_page="posts",
        posts=get_posts(),
    )


@app.get("/posts/{slug}", response_class=HTMLResponse)
async def post_page(request: Request, slug: str):
    """Render a single public post page."""
    post = get_post_by_slug(slug)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return public_template_response(
        "post_detail.html",
        request,
        active_page="posts",
        post=post,
    )


@app.get("/cv", response_class=HTMLResponse)
async def cv_page(request: Request):
    """Structured CV page for the public website."""
    return public_template_response(
        "cv.html",
        request,
        active_page="cv",
        cv_profile=CV_PROFILE,
        cv_experience=CV_EXPERIENCE,
        cv_education=CV_EDUCATION,
        cv_skills=CV_SKILLS,
        cv_highlights=CV_HIGHLIGHTS,
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Login page with password form.

    GET shows the login form. No error message on initial load.
    """
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "error": None},
    )


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
            request=request,
            name="login.html",
            context={"request": request, "error": "Invalid password"},
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
        request=request,
        name="dashboard.html",
        context={
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
    return templates.TemplateResponse(
        request=request,
        name="terminal.html",
        context={"request": request},
    )


@app.websocket("/ws/terminal")
async def websocket_terminal(websocket: WebSocket):
    """
    WebSocket endpoint for terminal access.

    Validates session cookie, spawns PTY, bridges to xterm.js.
    Session validation happens inside the handler (not middleware).
    """
    await terminal_websocket(websocket)
