"""
Ralph dashboard integration for nimamanafcom.

Mounts the Ralph FastAPI app as a sub-application at /ralph.
Handles static file path adjustments for the mounted context.

Algorithm:
1. Temporarily swap sys.modules to load Ralph's app module
2. Import Ralph's routers
3. Restore nimamanafcom's app module
4. Create modified routes with /ralph prefix

Note: We use module swapping to avoid conflicts between nimamanafcom's
'app' package and Ralph's 'app' package. Both projects use 'app' as their
main package name.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

# Ralph paths
RALPH_ROOT = Path("/home/nima/code/ralph")
RALPH_APP_DIR = RALPH_ROOT / "app"


def _load_ralph_routers():
    """
    Load Ralph's routers by temporarily swapping the 'app' module.

    This is necessary because both nimamanafcom and Ralph use 'app' as
    their package name. We save the current app modules, load Ralph's,
    import what we need, then restore the original modules.
    """
    # Save current app-related modules
    saved_modules = {}
    app_module_names = [k for k in sys.modules if k == "app" or k.startswith("app.")]
    for name in app_module_names:
        saved_modules[name] = sys.modules.pop(name)

    # Add Ralph to path
    sys.path.insert(0, str(RALPH_ROOT))

    try:
        # Now import Ralph's routers (app will resolve to Ralph's app)
        from app.routers import projects as ralph_projects
        from app.routers import session as ralph_session
        from app.routers import stories as ralph_stories
        from app.routers import suggestions as ralph_suggestions
        from app.routers import terminal as ralph_terminal
        from app.routers import validation as ralph_validation

        return {
            "projects": ralph_projects,
            "session": ralph_session,
            "stories": ralph_stories,
            "suggestions": ralph_suggestions,
            "terminal": ralph_terminal,
            "validation": ralph_validation,
        }

    finally:
        # Remove Ralph's app modules
        ralph_app_modules = [k for k in sys.modules if k == "app" or k.startswith("app.")]
        for name in ralph_app_modules:
            sys.modules.pop(name, None)

        # Restore nimamanafcom's app modules
        sys.modules.update(saved_modules)

        # Remove Ralph from path
        if str(RALPH_ROOT) in sys.path:
            sys.path.remove(str(RALPH_ROOT))


# Load Ralph routers
try:
    _ralph_modules = _load_ralph_routers()
    RALPH_IMPORT_SUCCESS = True
    RALPH_IMPORT_ERROR = None
except Exception as e:
    import traceback
    RALPH_IMPORT_SUCCESS = False
    RALPH_IMPORT_ERROR = f"{e}\n{traceback.format_exc()}"
    _ralph_modules = {}

# Create router for Ralph integration
ralph_router = APIRouter(prefix="/ralph", tags=["ralph"])


def get_ralph_static_dir() -> Path:
    """Get the path to Ralph's static files directory."""
    return RALPH_APP_DIR / "static"


@ralph_router.get("/", response_class=HTMLResponse)
async def ralph_dashboard(request: Request):
    """
    Serve Ralph dashboard with adjusted static paths.

    The original template uses /static/... which becomes /ralph/static/...
    when accessed through the mounted app.
    """
    if not RALPH_IMPORT_SUCCESS:
        return HTMLResponse(
            content=f"<h1>Ralph Import Error</h1><pre>{RALPH_IMPORT_ERROR}</pre>",
            status_code=500
        )

    # Read the original template and modify static paths
    template_path = RALPH_APP_DIR / "templates" / "index.html"
    with open(template_path, "r") as f:
        html = f.read()

    # Replace /static/ with /ralph/static/ for local static files
    # Keep CDN URLs unchanged (they start with http or //)
    html = html.replace('href="/static/', 'href="/ralph/static/')
    # For JS files, use a custom path that will be handled by our route
    html = html.replace('src="/static/js/', 'src="/ralph/js/')
    html = html.replace('src="/static/', 'src="/ralph/static/')

    # Also fix API endpoints to use /ralph prefix
    # The JavaScript makes fetch calls to /api/... which need to be /ralph/api/...
    html = html.replace("fetch('/api/", "fetch('/ralph/api/")
    html = html.replace("fetch(`/api/", "fetch(`/ralph/api/")
    html = html.replace('"/api/', '"/ralph/api/')

    # Fix WebSocket URLs
    html = html.replace("'/ws/", "'/ralph/ws/")
    html = html.replace('"/ws/', '"/ralph/ws/')

    return HTMLResponse(content=html)


@ralph_router.get("/test", response_class=HTMLResponse)
async def ralph_test():
    """Test page to verify Ralph integration works."""
    error_msg = RALPH_IMPORT_ERROR if RALPH_IMPORT_ERROR else "None"
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Ralph Test</title></head>
    <body style="background: #111; color: #fff; font-family: sans-serif; padding: 2rem;">
        <h1>Ralph Integration Test</h1>
        <p>Import success: {RALPH_IMPORT_SUCCESS}</p>
        <p>Error: <pre>{error_msg}</pre></p>
        <p><a href="/ralph/" style="color: #3b82f6;">Go to Ralph Dashboard</a></p>
        <p><a href="/dashboard" style="color: #3b82f6;">Back to Main Dashboard</a></p>
    </body>
    </html>
    """


@ralph_router.get("/js/app.js")
async def ralph_app_js():
    """
    Serve Ralph's app.js with modified API paths.

    The original app.js uses `/api${endpoint}` which needs to be
    `/ralph/api${endpoint}` when mounted at /ralph.
    """
    js_path = RALPH_APP_DIR / "static" / "js" / "app.js"
    with open(js_path, "r") as f:
        js_content = f.read()

    # Modify the api() function to use /ralph/api instead of /api
    js_content = js_content.replace(
        "await fetch(`/api${endpoint}`",
        "await fetch(`/ralph/api${endpoint}`"
    )

    # Also modify WebSocket URLs if any
    js_content = js_content.replace("'/ws/", "'/ralph/ws/")
    js_content = js_content.replace('"/ws/', '"/ralph/ws/')
    js_content = js_content.replace("`/ws/", "`/ralph/ws/")

    return Response(content=js_content, media_type="application/javascript")


# Include all Ralph API routers (if imports succeeded)
if RALPH_IMPORT_SUCCESS:
    ralph_router.include_router(_ralph_modules["projects"].router)
    ralph_router.include_router(_ralph_modules["stories"].router)
    ralph_router.include_router(_ralph_modules["session"].router)
    ralph_router.include_router(_ralph_modules["validation"].router)
    ralph_router.include_router(_ralph_modules["suggestions"].router)
    ralph_router.include_router(_ralph_modules["terminal"].router)
