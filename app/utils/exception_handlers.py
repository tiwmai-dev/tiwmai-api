"""Custom exception handlers."""

import traceback
from datetime import datetime

import sentry_sdk
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.exceptions import BaseAPIException
from app.core.logging import app_logger
from app.models.schemas import ErrorResponse


async def base_api_exception_handler(
    request: Request, exc: BaseAPIException
) -> JSONResponse:
    """Handle custom API exceptions."""
    request_id = getattr(request.state, "request_id", None)

    app_logger.error(
        f"[{request_id}] API Exception: {exc.message} " f"(Status: {exc.status_code})"
    )

    error_response = ErrorResponse(
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code, content=error_response.model_dump()
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)

    app_logger.warning(
        f"[{request_id}] HTTP Exception: {exc.detail} " f"(Status: {exc.status_code})"
    )

    error_response = ErrorResponse(
        error="HTTPException", message=exc.detail, request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code, content=error_response.model_dump()
    )


async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", None)

    # Get request body for debugging
    try:
        request_body = await request.body()
        request_body_str = request_body.decode("utf-8") if request_body else "No body"
    except Exception:
        request_body_str = "Unable to read request body"

    # Format validation errors
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        errors.append(
            {"field": field_path, "message": error["msg"], "type": error["type"]}
        )

    app_logger.error(f"[{request_id}] === VALIDATION ERROR DEBUG ===")
    app_logger.error(f"[{request_id}] Request URL: {request.method} {request.url}")
    app_logger.error(f"[{request_id}] Request Body: {request_body_str}")
    app_logger.error(f"[{request_id}] Validation Errors: {errors}")
    app_logger.error(f"[{request_id}] Full Exception: {exc}")

    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed",
        details={"validation_errors": errors},
        request_id=request_id,
    )

    return JSONResponse(status_code=422, content=error_response.model_dump())


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", None)

    sentry_sdk.capture_exception(exc)

    # Log full traceback for debugging
    app_logger.error(
        f"[{request_id}] Unexpected Exception: {str(exc)}\n"
        f"Traceback: {traceback.format_exc()}"
    )

    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        details={"exception_type": exc.__class__.__name__},
        request_id=request_id,
    )

    return JSONResponse(status_code=500, content=error_response.model_dump())


def setup_exception_handlers(app):
    """Setup all custom exception handlers."""
    app.add_exception_handler(BaseAPIException, base_api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
