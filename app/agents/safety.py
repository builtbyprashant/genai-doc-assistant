"""SafetyGuard — cheap, rule-based input validation. No LLM, no DB. (DECISIONS: C-A)

This is the first pipeline step and the cheapest line of defence: length bounds and
a scan for obvious prompt-injection phrasing. It only catches coded patterns —
a determined attacker can rephrase — so it is one layer among several, not the
whole defence.
"""

from __future__ import annotations

import re

MIN_QUERY_CHARS = 3
MAX_QUERY_CHARS = 2000

# Common prompt-injection phrasings. Deliberately conservative to avoid false positives.
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(above|previous|prior)",
    r"forget (everything|all|your instructions)",
    r"reveal your (instructions|system prompt|prompt)",
    r"you are now",
    r"system prompt",
    r"developer mode",
]


class SafetyError(Exception):
    """Input rejected by a guardrail. `code` maps to an API error type."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def check(question: str) -> None:
    """Raise SafetyError if the question is too short/long or looks like injection."""
    text = question.strip()

    if len(text) < MIN_QUERY_CHARS:
        raise SafetyError("query-too-short", "Query is too short. Please ask a complete question.")
    if len(text) > MAX_QUERY_CHARS:
        raise SafetyError("query-too-long", "Query exceeds the maximum length of 2000 characters.")

    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise SafetyError("safety-guardrail", "Query was flagged by the safety guardrail.")
