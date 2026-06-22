"""FastAPI application entry point."""

import sentry_sdk
from fastapi import FastAPI

from app.api.student_auth_endpoints import router as student_auth_router
from app.api.student_endpoints import router as student_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.utils.exception_handlers import setup_exception_handlers
from app.utils.middleware import setup_custom_middleware

# Initialize logging
setup_logging()

# Get settings
settings = get_settings()

# Initialize error monitoring before creating the FastAPI application.
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        send_default_pii=settings.sentry_send_default_pii,
        environment=settings.sentry_environment,
    )

# Create FastAPI application
app = FastAPI(
    title="Tiwmai Student API",
    description="Student-facing API for Tiwmai learning experiences",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Setup middleware
setup_custom_middleware(app)

# Setup exception handlers
setup_exception_handlers(app)

# Include only student-facing API routes for this Vercel deployment.
app.include_router(student_auth_router, prefix="/api/v1")
app.include_router(student_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Tiwmai Student API",
        "description": "Student-facing API for Tiwmai learning experiences",
        "version": "1.0.0",
        "docs_url": "/api/docs",
        "health_check": "/api/v1/health",
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    from app.core.logging import app_logger

    app_logger.info("Starting Tiwmai Student API")
    app_logger.info(f"Debug mode: {settings.debug}")
    app_logger.info(f"Allowed origins: {settings.allowed_origins}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    from app.core.logging import app_logger

    app_logger.info("Shutting down Tiwmai Student API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
