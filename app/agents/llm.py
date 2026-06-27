"""Thin wrapper around the Anthropic SDK.

Every LLM agent (Planner, Reasoner, Validator) calls `complete()` rather than the
SDK directly. Keeping the call in one place means:
  - the system prompt is cached consistently (prompt caching),
  - tests mock a single function instead of the whole SDK,
  - Task 9 can wrap this with the retry/timeout policy in one spot.
"""

from __future__ import annotations

import json
from typing import Optional

import anthropic

from app.core.config import get_settings, require_api_key

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Lazily build the SDK client so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=require_api_key())
    return _client


def complete(system: str, user: str, max_tokens: int = 800, timeout: Optional[float] = None) -> str:
    """Send one system+user turn to Claude and return the text response.

    The system prompt is marked cacheable — it is identical across calls for a
    given agent, so Anthropic caches it after the first use.
    """
    settings = get_settings()
    client = get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of a model response.

    Models sometimes wrap JSON in prose or code fences, so we slice from the first
    '{' to the last '}'. Raises ValueError if there is nothing parseable — callers
    fall back to a safe default.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    return json.loads(raw[start : end + 1])
