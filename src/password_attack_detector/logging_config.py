"""Structured logging configuration for the Password Attack Detector.

This module is intentionally named ``logging_config`` (not ``logging``) to
avoid shadowing Python's standard-library ``logging`` module.

Typical usage::

    from password_attack_detector.logging_config import setup_logging, get_logger

    setup_logging(log_level="INFO", environment="development")
    log = get_logger(__name__)
    log.info("system started", version="0.1.0")

In development the output is human-readable and colorised (when stderr is a
TTY).  In production the output is newline-delimited JSON suitable for
ingestion by log aggregators.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import structlog
from structlog.types import FilteringBoundLogger

__all__ = ["get_logger", "reset_logging", "setup_logging"]

_configured: bool = False

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.TimeStamper(fmt="iso"),
]


def setup_logging(
    log_level: str,
    environment: str,
    *,
    force: bool = False,
) -> None:
    """Configure structlog and the stdlib ``logging`` root logger.

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    environment:
        Active environment name.  ``production`` selects JSON output; all
        other values select the human-readable console renderer.
    force:
        When ``True``, reconfigure even if logging was already configured.
        Use in tests that need to change the log level mid-suite.
    """
    global _configured

    if _configured and not force:
        return

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    if environment == "production":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*_SHARED_PROCESSORS, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging so third-party libraries that use logging.getLogger
    # also emit structured output at the correct level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
        force=True,
    )

    _configured = True


def reset_logging() -> None:
    """Reset the configuration guard.

    Intended for use in test fixtures only.  After calling this, the next
    :func:`setup_logging` call will fully reconfigure structlog regardless of
    previous state.
    """
    global _configured
    _configured = False


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a bound structlog logger for *name*.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    FilteringBoundLogger
        A structlog bound logger ready for use.
    """
    return cast(FilteringBoundLogger, structlog.get_logger(name))
