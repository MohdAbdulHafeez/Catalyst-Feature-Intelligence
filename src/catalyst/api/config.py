from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApiSettings:
    title: str = "CATALYST API"
    version: str = "0.1.0"
    environment: str = "development"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    @classmethod
    def from_environment(cls) -> ApiSettings:
        origins_value = os.getenv(
            "CATALYST_CORS_ORIGINS",
            "http://localhost:3000",
        )
        origins = tuple(origin.strip() for origin in origins_value.split(",") if origin.strip())
        return cls(
            title=os.getenv("CATALYST_API_TITLE", "CATALYST API"),
            version=os.getenv("CATALYST_API_VERSION", "0.1.0"),
            environment=os.getenv("CATALYST_ENV", "development"),
            cors_origins=origins or ("http://localhost:3000",),
        )
