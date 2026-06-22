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

def create_app(app_settings):
    """Create the FastAPI application with environment-specific exposure."""
    app_instance = FastAPI(
        title="Tiwmai Student API",
        description="Student-facing API for Tiwmai learning experiences",
        version="1.0.0",
        docs_url="/api/docs" if app_settings.debug else None,
        redoc_url="/api/redoc" if app_settings.debug else None,
        openapi_url="/api/openapi.json" if app_settings.debug else None,
    )

    setup_custom_middleware(app_instance)
    setup_exception_handlers(app_instance)

    # Include only student-facing API routes for this Vercel deployment.
    app_instance.include_router(student_auth_router, prefix="/api/v1")
    app_instance.include_router(student_router, prefix="/api/v1")

    @app_instance.get("/")
    async def root():
        """Root endpoint with API information."""
        if not app_settings.debug:
            return {"ok": True}
        return {
            "message": "Tiwmai Student API",
            "description": "Student-facing API for Tiwmai learning experiences",
            "version": "1.0.0",
            "docs_url": "/api/docs",
            "health_check": "/api/v1/health",
        }

    @app_instance.on_event("startup")
    async def startup_event():
        """Application startup tasks."""
        from app.core.logging import app_logger

        app_logger.info("Starting Tiwmai Student API")
        app_logger.info(f"Debug mode: {app_settings.debug}")
        app_logger.info(f"Allowed origins: {app_settings.allowed_origins}")

    @app_instance.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown tasks."""
        from app.core.logging import app_logger

        app_logger.info("Shutting down Tiwmai Student API")

    return app_instance


app = create_app(settings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
