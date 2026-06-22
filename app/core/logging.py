"""Logging configuration."""

import os
import sys

from loguru import logger

from app.core.config import get_settings


def setup_logging():
    """Configure logging with loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    logger.add(
        sys.stdout,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Vercel's function bundle is read-only; rely on stdout/stderr there.
    if not settings.debug and not os.getenv("VERCEL"):
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            format=settings.log_format,
            level=settings.log_level,
            rotation="1 day",
            retention="30 days",
            compression="zip",
        )

    return logger


# Initialize logger
app_logger = setup_logging()
