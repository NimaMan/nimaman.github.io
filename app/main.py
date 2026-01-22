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

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings

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

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
