"""LLM call retry policy.

Phase 1 keeps it simple: one blanket retry on any failure (no error-type
classification — that's a Phase 2 refinement). The first attempt gets the full
timeout, the retry gets half, so the user waits at most ~1.5x the timeout rather
than 2x. After the retry is exhausted we raise LLMUnavailableError, which the API
turns into a structured 503.
"""

from __future__ import annotations

from typing import Callable

from app.core.config import get_settings
from app.core.errors import LLMUnavailableError
from app.utils.logging import get_logger


def call_with_retry(fn: Callable, *args, agent: str | None = None, **kwargs):
    """Call `fn(*args, timeout=t, **kwargs)`, retrying once at half timeout."""
    logger = get_logger()
    base = get_settings().llm_timeout_seconds
    timeouts = [base, max(1, base // 2)]  # e.g. 20s then 10s

    for attempt, timeout in enumerate(timeouts):
        try:
            return fn(*args, timeout=timeout, **kwargs)
        except Exception as exc:  # blanket retry — Phase 1 (Phase 2 classifies errors)
            if attempt < len(timeouts) - 1:
                logger.warning("llm_retry", extra={"context": {
                    "event": "llm_retry", "attempt": attempt + 1,
                    "timeout": timeout, "agent": agent, "error": str(exc)[:100],
                }})
            else:
                raise LLMUnavailableError(str(exc)[:200], failed_at_agent=agent)
