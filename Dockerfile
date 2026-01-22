# nimamanafcom - Private Dashboard
# Runs on Tailscale network only, never exposed to public internet
#
# Features:
# - Landing page with personal info
# - Password-protected dashboard
# - Web terminal (xterm.js + PTY)
# - Ralph AI project management integration

FROM python:3.11-slim

# Install system dependencies for PTY and terminal
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Port for the application
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/health')" || exit 1

# Run with uvicorn
# Note: user is set via docker-compose for proper host filesystem access
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
