"""Logging configuration."""

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

    # Add file handler for production
    if not settings.debug:
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
