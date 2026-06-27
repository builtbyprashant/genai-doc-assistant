"""Central configuration.

Every environment variable the app uses is read and validated here — and nowhere
else. Other modules import `get_settings()` and read fields off the result; they
never touch `os.environ` directly. This keeps config in one auditable place and
makes the whole app easy to test (swap env, clear the cache, reload).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# AGENT_MODE here is only the *default*. The pipeline always takes the mode as a
# parameter, so a per-request override (POST /query) never reads this value.
# (DECISIONS: C-D, D-6d)
VALID_AGENT_MODES = ("custom", "llama_index", "compare")
VALID_CACHE_MODES = ("block", "prompt", "off")


@dataclass(frozen=True)
class Settings:
    # LLM
    anthropic_api_key: str
    anthropic_model: str
    # Prompt-caching strategy (DECISIONS: D-12):
    #   block  → cache the (stable) system prompt as its own block — best for this
    #            app's single-turn pattern; the system prompt repeats across queries.
    #   prompt → top-level cache_control marks the whole prompt prefix (last block).
    #   off    → no caching.
    cache_mode: str

    # Agent mode (default only — see note above)
    agent_mode: str

    # Models (local sentence-transformers, no API key). The embedding model MUST
    # output 384-dim vectors to match the ChromaDB collection — changing it means
    # re-indexing. The reranker is any cross-encoder (scores, not vectors — no
    # dimension constraint).
    embedding_model: str
    reranker_model: str

    # Retrieval
    similarity_threshold: float
    top_k_retrieval: int
    top_k_rerank: int

    # Chunking
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int

    # Storage
    chroma_persist_path: str
    upload_size_limit_mb: int
    max_documents: int

    # API / logging
    backend_url: str
    log_level: str

    # Reliability
    llm_timeout_seconds: int
    query_hash_algo: str

    # Reserved for Phase 2 — read so it round-trips through config, but unused in
    # Phase 1. (DECISIONS: D-3)
    grounding_threshold: float


def load_settings() -> Settings:
    """Build a Settings object from the current environment and validate it."""
    settings = Settings(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        cache_mode=os.environ.get("ANTHROPIC_CACHE_MODE", "block"),
        agent_mode=os.environ.get("AGENT_MODE", "custom"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        reranker_model=os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        similarity_threshold=float(os.environ.get("SIMILARITY_THRESHOLD", "0.4")),
        top_k_retrieval=int(os.environ.get("TOP_K_RETRIEVAL", "10")),
        top_k_rerank=int(os.environ.get("TOP_K_RERANK", "5")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "200")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "25")),
        min_chunk_chars=int(os.environ.get("MIN_CHUNK_CHARS", "20")),
        chroma_persist_path=os.environ.get("CHROMA_PERSIST_PATH", "/data/chroma"),
        upload_size_limit_mb=int(os.environ.get("UPLOAD_SIZE_LIMIT_MB", "10")),
        max_documents=int(os.environ.get("MAX_DOCUMENTS", "20")),
        backend_url=os.environ.get("BACKEND_URL", "http://backend:8000"),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        llm_timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "20")),
        query_hash_algo=os.environ.get("QUERY_HASH_ALGO", "sha256"),
        grounding_threshold=float(os.environ.get("GROUNDING_THRESHOLD", "0.6")),
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    """Catch misconfiguration at load time rather than deep inside a request."""
    if s.agent_mode not in VALID_AGENT_MODES:
        raise ValueError(
            f"AGENT_MODE must be one of {VALID_AGENT_MODES}, got '{s.agent_mode}'"
        )
    if s.cache_mode not in VALID_CACHE_MODES:
        raise ValueError(
            f"ANTHROPIC_CACHE_MODE must be one of {VALID_CACHE_MODES}, got '{s.cache_mode}'"
        )
    if not 0.0 <= s.similarity_threshold <= 1.0:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    if s.top_k_rerank > s.top_k_retrieval:
        raise ValueError("TOP_K_RERANK cannot exceed TOP_K_RETRIEVAL")
    if s.chunk_overlap >= s.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    for name in ("top_k_retrieval", "top_k_rerank", "chunk_size", "max_documents"):
        if getattr(s, name) <= 0:
            raise ValueError(f"{name.upper()} must be a positive integer")

    # ANTHROPIC_API_KEY is required at runtime but intentionally NOT enforced here,
    # so tests and tooling can import config without a key. The API layer checks it
    # before the first LLM call (see require_api_key).


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated settings singleton (cached after first call).

    Tests that change the environment should call `get_settings.cache_clear()`.
    """
    return load_settings()


def require_api_key() -> str:
    """Return the Anthropic API key, raising a clear error if it is missing.

    Called by the app at startup / before LLM calls — never at import time.
    """
    key = get_settings().anthropic_api_key
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file before running."
        )
    return key
