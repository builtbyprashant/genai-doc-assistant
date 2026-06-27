"""Error vocabulary shared by the whole app.

`PROBLEM_TYPES` is the single registry of error codes → (title, HTTP status),
matching the API contract. `problem_detail` builds an RFC-7807 body from a code so
the API layer never hand-rolls error JSON.
"""

from __future__ import annotations

from typing import Optional

# code -> (human title, HTTP status)
PROBLEM_TYPES: dict[str, tuple[str, int]] = {
    "validation-error": ("Validation error", 422),
    "file-too-large": ("File too large", 413),
    "unsupported-file-type": ("Unsupported file type", 415),
    "duplicate-content": ("Duplicate content", 409),
    "password-protected": ("Password-protected PDF", 422),
    "scanned-pdf": ("Scanned PDF detected", 422),
    "empty-file": ("Empty file", 422),
    "document-not-found": ("Document not found", 404),
    "max-documents-reached": ("Maximum documents reached", 409),
    "filter-not-found": ("Filter not found", 404),
    "safety-guardrail": ("Safety guardrail", 400),
    "query-too-short": ("Query too short", 400),
    "query-too-long": ("Query too long", 400),
    "llm-unavailable": ("LLM service unavailable", 503),
    "llm-timeout": ("LLM timeout", 503),
    "internal-error": ("Internal server error", 500),
}


def status_for(code: str) -> int:
    return PROBLEM_TYPES.get(code, ("Error", 500))[1]


def problem_detail(
    code: str,
    detail: str,
    *,
    instance: Optional[str] = None,
    request_id: Optional[str] = None,
    **extra,
) -> dict:
    """Build an RFC-7807 problem-details body for an error code."""
    title, status = PROBLEM_TYPES.get(code, ("Error", 500))
    body = {
        "type": f"https://rag-api/errors/{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
        "request_id": request_id,
    }
    body.update(extra)
    return body


class LLMUnavailableError(Exception):
    """Raised when an LLM call fails after its retry. Maps to a 503. (C-F sibling)"""

    def __init__(self, message: str, failed_at_agent: Optional[str] = None):
        super().__init__(message)
        self.code = "llm-unavailable"
        self.failed_at_agent = failed_at_agent
        self.retry_after = 30
