from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from catalyst.api.config import ApiSettings
from catalyst.api.errors import (
    dataset_ingestion_error_handler,
    http_exception_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from catalyst.api.ingestion import DatasetIngestionError, LocalDatasetStore
from catalyst.api.middleware import RequestIdMiddleware
from catalyst.api.routes.datasets import router as datasets_router
from catalyst.api.routes.health import router as health_router


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_environment()

    app = FastAPI(
        title=resolved_settings.title,
        version=resolved_settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = resolved_settings
    app.state.dataset_store = LocalDatasetStore(
        root=resolved_settings.data_dir,
        max_upload_bytes=resolved_settings.max_upload_bytes,
    )

    app.add_middleware(RequestIdMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        DatasetIngestionError,
        dataset_ingestion_error_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_error_handler,
    )

    app.include_router(
        health_router,
        prefix="/api/v1",
    )

    app.include_router(
        datasets_router,
        prefix="/api/v1",
    )

    return app
