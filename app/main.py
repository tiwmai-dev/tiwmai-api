"""FastAPI application entry point."""

from pathlib import Path

import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.auth_endpoints import router as auth_router
from app.api.chat_endpoints import router as chat_router
from app.api.endpoints import router as legacy_router
from app.api.invitation_endpoints import router as invitation_router
from app.api.job_endpoints import router as job_router
from app.api.student_auth_endpoints import router as student_auth_router
from app.api.student_endpoints import router as student_router
from app.api.tutor_endpoints import router as tutor_canonical_router
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
    """Create the FastAPI application with student, tutor, and admin routes."""
    app_instance = FastAPI(
        title="Tiwmai API",
        description="Combined API for Tiwmai student, tutor, and admin apps",
        version="1.0.0",
        docs_url="/api/docs" if app_settings.debug else None,
        redoc_url="/api/redoc" if app_settings.debug else None,
        openapi_url="/api/openapi.json" if app_settings.debug else None,
    )

    setup_custom_middleware(app_instance)
    setup_exception_handlers(app_instance)

    # Legacy flat routes used by tutor/admin web apps during migration.
    app_instance.include_router(legacy_router, prefix="/api/v1", tags=["Legacy"])
    app_instance.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])
    app_instance.include_router(
        tutor_canonical_router, prefix="/api/v1", tags=["Tutor"]
    )
    app_instance.include_router(chat_router, prefix="/api/v1", tags=["Chat Assistant"])
    app_instance.include_router(
        invitation_router, prefix="/api/v1", tags=["Course Invitations"]
    )
    app_instance.include_router(job_router, prefix="/api/v1", tags=["Jobs"])

    # Student-prefixed route surface used by tiwmai-student-web.
    app_instance.include_router(student_auth_router, prefix="/api/v1")
    app_instance.include_router(student_router, prefix="/api/v1")

    uploads_dir = Path(app_settings.upload_folder)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app_instance.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

    @app_instance.get("/")
    async def root():
        """Root endpoint with API information."""
        if not app_settings.debug:
            return {"ok": True}
        return {
            "message": "Tiwmai API",
            "description": "Combined API for Tiwmai learning experiences",
            "version": "1.0.0",
            "docs_url": "/api/docs",
            "health_check": "/api/v1/health",
        }

    @app_instance.on_event("startup")
    async def startup_event():
        """Application startup tasks."""
        from app.core.logging import app_logger

        app_logger.info("Starting Tiwmai API")
        app_logger.info(f"Debug mode: {app_settings.debug}")
        app_logger.info(f"Upload folder: {app_settings.upload_folder}")
        app_logger.info(f"Allowed origins: {app_settings.allowed_origins}")

    @app_instance.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown tasks."""
        from app.core.logging import app_logger

        app_logger.info("Shutting down Tiwmai API")

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
