from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ApiSettings:
    title: str = "CATALYST API"
    version: str = "0.1.0"
    environment: str = "development"
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    data_dir: Path = Path(".catalyst/data/datasets")
    max_upload_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> ApiSettings:
        origins_value = os.getenv("CATALYST_CORS_ORIGINS", "http://localhost:3000")
        origins = tuple(origin.strip() for origin in origins_value.split(",") if origin.strip())
        data_dir = Path(os.getenv("CATALYST_DATA_DIR", str(Path(".catalyst/data/datasets"))))
        max_upload_bytes = int(os.getenv("CATALYST_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        if max_upload_bytes <= 0:
            raise ValueError("CATALYST_MAX_UPLOAD_BYTES must be positive.")
        return cls(
            title=os.getenv("CATALYST_API_TITLE", "CATALYST API"),
            version=os.getenv("CATALYST_API_VERSION", "0.1.0"),
            environment=os.getenv("CATALYST_ENV", "development"),
            cors_origins=origins or ("http://localhost:3000",),
            data_dir=data_dir,
            max_upload_bytes=max_upload_bytes,
        )
