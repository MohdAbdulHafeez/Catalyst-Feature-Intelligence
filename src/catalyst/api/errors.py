from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from catalyst.api.contracts import ApiError


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
