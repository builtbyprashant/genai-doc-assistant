"""Structured JSON logging to stdout.

Every log line is a single JSON object so any aggregator can parse it later with
no custom regex. A trace/request id is attached per request by the API layer.

Query-hash sanitisation (never logging the raw query) is added in Task 9 — this
module is the skeleton it builds on.
"""

from __future__ import annotations

import json
import logging
import sys

from app.core.config import get_settings


class JsonFormatter(logging.Formatter):
    """Render a log record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Structured fields are passed as logger.info(msg, extra={"context": {...}}).
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)
        return json.dumps(payload, default=str)


def get_logger(name: str = "rag") -> logging.Logger:
    """Return a logger that writes JSON to stdout, configured once."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already set up — avoid duplicate handlers on reload
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log through the root logger
    return logger
