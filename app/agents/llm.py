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
from app.utils.logging import get_logger
from app.utils.retry import call_with_retry

_client: Optional[anthropic.Anthropic] = None
_logger = get_logger()

_EPHEMERAL = {"type": "ephemeral"}


def _cache_kwargs(system: str, cache_mode: str) -> dict:
    """Build the `system` (and maybe top-level `cache_control`) kwargs for a request,
    per the configured prompt-caching strategy (DECISIONS: D-12).

    - block  → the system prompt is its own cacheable block (block-level caching).
               Best for this app: the system prompt is identical across queries, so
               it is cache-read after the first call for each agent.
    - prompt → a top-level `cache_control` marker caches the whole prompt prefix (the
               SDK applies it to the last block). Useful when a long prefix repeats
               (e.g. multi-turn); for single-turn the changing user content limits reuse.
    - off    → no caching.
    """
    if cache_mode == "prompt":
        return {"system": system, "cache_control": _EPHEMERAL}
    if cache_mode == "off":
        return {"system": system}
    return {"system": [{"type": "text", "text": system, "cache_control": _EPHEMERAL}]}


def _build_request(system: str, user: str, cache_context: Optional[str], cache_mode: str) -> dict:
    """Build the `system` + `messages` kwargs for a request (DECISIONS: D-16).

    When `cache_context` is given (and caching is on), the large, stable CONTEXT becomes
    its own `cache_control` block at the head of the user message, with the agent-specific
    task trailing it *uncached*. Two calls that share the same system + context — e.g. the
    Reasoner then the Validator on the same retrieved chunks — produce an identical cached
    prefix, so the second call reads it instead of re-processing the context. (This is the
    only structure that actually caches anything here: the system prompts are ~100 tokens,
    far below the model's minimum cacheable length, so block-level system caching never
    fires — D-12. The context block is where the tokens, and the reuse, actually are.)

    Falls back to the original per-system strategy when there is no large context to cache
    (e.g. the Planner) or caching is off.
    """
    if cache_context and cache_mode != "off":
        return {
            "system": system,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": cache_context, "cache_control": _EPHEMERAL},
                {"type": "text", "text": user},
            ]}],
        }
    return {"messages": [{"role": "user", "content": user}], **_cache_kwargs(system, cache_mode)}


def _log_usage(agent: Optional[str], usage, cache_mode: str) -> None:
    """Log token usage incl. cache hits/writes so caching effectiveness is observable.

    `cache_read_input_tokens` > 0 means the prompt prefix was served from cache;
    `cache_creation_input_tokens` > 0 means it was written to the cache this call."""
    if usage is None:
        return
    _logger.info("llm_usage", extra={"context": {
        "event": "llm_usage",
        "agent": agent,
        "cache_mode": cache_mode,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
    }})


# ── per-query token usage + cost estimate ─────────────────────────────────────

_USAGE_KEYS = ("input_tokens", "output_tokens",
               "cache_read_input_tokens", "cache_creation_input_tokens")

# USD per 1M tokens: (input, output). Cache reads bill ~0.1× input, writes ~1.25×.
_PRICING = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}


def new_usage() -> dict:
    """A fresh per-query token accumulator. Threaded through the agent calls so the
    count survives both the batch path and the streaming generator (a contextvar
    would be lost across the stream's thread contexts)."""
    return {k: 0 for k in _USAGE_KEYS}


def _accumulate(sink: Optional[dict], usage) -> None:
    if sink is None or usage is None:
        return
    for k in _USAGE_KEYS:
        sink[k] += getattr(usage, k, 0) or 0


def _price_for(model: str) -> tuple:
    for prefix, price in _PRICING.items():
        if model.startswith(prefix):
            return price
    return _PRICING["claude-haiku-4-5"]  # default to Haiku pricing


def summarize_usage(sink: dict, model: str) -> dict:
    """Total tokens consumed + an estimated USD cost (cache discounts applied)."""
    in_price, out_price = _price_for(model)
    fresh_in = sink.get("input_tokens", 0)
    cache_read = sink.get("cache_read_input_tokens", 0)
    cache_write = sink.get("cache_creation_input_tokens", 0)
    out = sink.get("output_tokens", 0)
    billable_in = fresh_in + cache_write * 1.25 + cache_read * 0.1
    cost = (billable_in * in_price + out * out_price) / 1_000_000
    return {
        "tokens": fresh_in + cache_read + cache_write + out,
        "cost_usd": round(cost, 6),
        "cache_read_tokens": cache_read,
    }


def get_client() -> anthropic.Anthropic:
    """Lazily build the SDK client so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=require_api_key())
    return _client


def complete(system: str, user: str, max_tokens: int = 800, agent: Optional[str] = None,
             usage: Optional[dict] = None, cache_context: Optional[str] = None) -> str:
    """Send one system+user turn to Claude, with the blanket retry policy.

    `agent` labels the call so a failure reports which agent it failed at. Pass
    `cache_context` (the large, stable retrieved CONTEXT) to cache it as a prefix block so
    a later call sharing the same system + context reads it from cache (D-16). Cache +
    token usage is logged; if `usage` (a `new_usage()` dict) is passed, this call's tokens
    are added to it for the per-query total.
    """
    text, u = call_with_retry(_raw_complete, system, user, max_tokens,
                              agent=agent, cache_context=cache_context)
    _log_usage(agent, u, get_settings().cache_mode)
    _accumulate(usage, u)
    return text


def _raw_complete(system: str, user: str, max_tokens: int, timeout: Optional[float] = None,
                  cache_context: Optional[str] = None):
    settings = get_settings()
    client = get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        **_build_request(system, user, cache_context, settings.cache_mode),
    )
    return response.content[0].text, response.usage


def stream_text(system: str, user: str, max_tokens: int = 800, agent: Optional[str] = None,
                usage: Optional[dict] = None, cache_context: Optional[str] = None):
    """Yield answer text deltas as the model generates them.

    Unlike `complete()`, there is no retry wrapper — a stream can't be transparently
    re-driven once bytes have been sent to the client. Connection errors propagate to
    the caller. Caching follows the same strategy as `complete()` (D-16); cache/token
    usage is logged once the stream completes.
    """
    settings = get_settings()
    client = get_client()
    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        **_build_request(system, user, cache_context, settings.cache_mode),
    ) as stream:
        yield from stream.text_stream
        try:
            final = stream.get_final_message().usage
            _log_usage(agent, final, settings.cache_mode)
            _accumulate(usage, final)
        except Exception:  # usage logging must never break the response
            pass


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
