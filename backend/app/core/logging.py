"""
Structured Logging Setup
─────────────────────────
Uses structlog for JSON-formatted, context-aware logging.
Integrates with Sentry for error tracking in production.
"""

import logging
import sys
from typing import Any

import structlog
from app.core.config import settings


def setup_logging():
    """Configure structlog + stdlib logging. Call once at app startup."""

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Processors differ for dev vs production
    if settings.DEBUG:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Sentry integration (production only)
    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                CeleryIntegration(),
            ],
            traces_sample_rate=0.1,
            environment=settings.ENVIRONMENT,
            release=settings.APP_VERSION,
            before_send=_before_send_sentry,
        )

    # Silence noisy third-party loggers
    for noisy in ["uvicorn.access", "sqlalchemy.engine", "botocore", "boto3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _before_send_sentry(event: dict, hint: dict) -> Any:
    """Filter out non-critical errors before sending to Sentry."""
    # Don't send validation errors — those are client mistakes, not bugs
    exc_info = hint.get("exc_info")
    if exc_info:
        exc_type = exc_info[0]
        if exc_type and exc_type.__name__ in {"ValidationError", "RequestValidationError"}:
            return None
    return event
