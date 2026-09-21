from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from catalyst.api.contracts import ApiError
from catalyst.api.ingestion import DatasetIngestionError


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    error = ApiError(
        code="http_error",
        message=str(exc.detail) if exc.detail else "Request failed.",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(mode="json"),
    )


async def dataset_ingestion_error_handler(
    request: Request,
    exc: DatasetIngestionError,
) -> JSONResponse:
    error = ApiError(
        code="dataset_ingestion_error",
        message=str(exc),
    )

    return JSONResponse(
        status_code=422,
        content=error.model_dump(mode="json"),
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error = ApiError(
        code="validation_error",
        message="Request validation failed.",
    )

    return JSONResponse(
        status_code=422,
        content=error.model_dump(mode="json"),
    )


async def unhandled_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    error = ApiError(
        code="internal_server_error",
        message="An unexpected server error occurred.",
    )

    return JSONResponse(
        status_code=500,
        content=error.model_dump(mode="json"),
    )
