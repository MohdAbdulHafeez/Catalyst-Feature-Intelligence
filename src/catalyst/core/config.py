from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class Settings:
    """Minimal application-independent settings for the ML core."""

    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
