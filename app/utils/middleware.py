"""Custom middleware for the application."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import app_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]

        # Add request ID to request state
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        app_logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- Client: {request.client.host if request.client else 'unknown'}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            app_logger.info(
                f"[{request_id}] {response.status_code} " f"- {process_time:.3f}s"
            )

            # Add request ID and processing time to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            app_logger.error(f"[{request_id}] Error - {process_time:.3f}s - {str(e)}")
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Only add HSTS in production
        settings = get_settings()
        if not settings.debug:
            response.headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains"

        return response


def setup_cors_middleware(app):
    """Setup CORS middleware with configuration."""
    settings = get_settings()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=settings.allowed_methods_list,
        allow_headers=settings.allowed_headers_list,
    )


def setup_custom_middleware(app):
    """Setup all custom middleware."""
    # Add middleware in reverse order (last added is executed first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    setup_cors_middleware(app)
