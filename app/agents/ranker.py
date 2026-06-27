"""RankerAgent — re-orders retrieved chunks by answer relevance.

This is NOT an LLM agent. It uses a cross-encoder (`ms-marco-MiniLM-L-6-v2`) that
reads the query and a chunk together and scores how well the chunk answers the
query — more precise than the bi-encoder similarity used for retrieval, but it
makes no Anthropic call. (DECISIONS: C-A, C-B)
"""

from __future__ import annotations

import math
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings


@lru_cache(maxsize=4)
def _model(name: str) -> CrossEncoder:
    """Load (and cache) a cross-encoder by name. Keyed on name so RERANKER_MODEL
    can change without a stale cache."""
    return CrossEncoder(name)


def ranker_agent(question: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """Score chunks against the question and return the best `top_k`, ranked."""
    if not chunks:
        return []
    settings = get_settings()
    top_k = top_k or settings.top_k_rerank

    scores = _model(settings.reranker_model).predict([(question, c["text"]) for c in chunks])
    # Cross-encoder outputs an unbounded logit; squash to 0-1 for the API contract.
    ranked = [{**chunk, "rerank_score": _sigmoid(float(s))} for chunk, s in zip(chunks, scores)]

    # Stable sort: equal scores keep their original retrieval order. (Requirements §3.3)
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
