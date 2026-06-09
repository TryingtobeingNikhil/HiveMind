"""
app/core/logging.py
───────────────────
Structured logging setup for Open Deep Research.

- Development: coloured, human-readable output via standard formatter
- Production:  JSON-formatted lines for log aggregation pipelines

Usage:
    from app.core.logging import get_logger, configure_logging
    logger = get_logger(__name__)
    logger.info("Server starting", extra={"port": 8000})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings


# ── JSON log formatter ────────────────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """
    Emit log records as single-line JSON objects.

    Each record includes:
      timestamp, level, logger, message, and any ``extra`` fields.
    """

    RESERVED_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Use a local variable — do NOT mutate record.message directly,
        # as some Python versions/loggers protect this attribute.
        message = record.getMessage()
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)

        log_record: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Attach exception info if present
        if record.exc_text:
            log_record["exception"] = record.exc_text

        # Attach any extra fields passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                log_record[key] = value

        return json.dumps(log_record, default=str)


# ── Coloured text formatter (development) ─────────────────────────────────────

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
}
_RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    """Human-readable coloured formatter for development use."""

    FMT = "%(asctime)s  %(levelcolour)s%(levelname)-8s%(reset)s  %(name)s  %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelcolour = colour
        record.reset = _RESET
        formatter = logging.Formatter(self.FMT, datefmt="%H:%M:%S")
        return formatter.format(record)


# ── Public API ────────────────────────────────────────────────────────────────


def configure_logging(settings: Settings) -> None:
    """
    Configure the root logger once at application startup.

    All subsequent ``get_logger()`` calls inherit this configuration
    automatically through Python's logging hierarchy.
    """
    level = logging.getLevelName(settings.log_level)

    handler = logging.StreamHandler(sys.stdout)

    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ColourFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers in production
    if settings.is_production:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    startup_logger = get_logger(__name__)
    startup_logger.info(
        "Logging configured",
        extra={
            "log_level": settings.log_level,
            "log_format": settings.log_format,
            "app_env": settings.app_env,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance inheriting root configuration.
    """
    return logging.getLogger(name)
