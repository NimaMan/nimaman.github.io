"""
Configuration module for nimamanafcom.

Loads settings from .env file:
- SECRET_KEY: Used for signing session cookies
- PASSWORD_HASH: Bcrypt hash of the login password
- SESSION_TIMEOUT_HOURS: How long sessions last (default: 24)
- HOST: Bind address (default: 127.0.0.1)
- PORT: Port number (default: 8000)

Algorithm:
1. Load .env file from project root using python-dotenv
2. Provide Settings class with typed fields and defaults
3. Validate that required fields (SECRET_KEY, PASSWORD_HASH) exist
"""

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Required settings
        self.secret_key: str = os.getenv("SECRET_KEY", "")
        self.password_hash: str = os.getenv("PASSWORD_HASH", "")

        # Optional settings with defaults
        self.session_timeout_hours: int = int(
            os.getenv("SESSION_TIMEOUT_HOURS", "24")
        )
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))

        # Ralph integration path
        self.ralph_path: str = os.getenv(
            "RALPH_PATH", "/home/nima/code/ralph"
        )

    def validate(self) -> list[str]:
        """Validate required settings. Returns list of missing fields."""
        missing = []
        if not self.secret_key:
            missing.append("SECRET_KEY")
        if not self.password_hash:
            missing.append("PASSWORD_HASH")
        return missing


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
