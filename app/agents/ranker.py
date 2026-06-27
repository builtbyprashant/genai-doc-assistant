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

RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(RANKER_MODEL)


def ranker_agent(question: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """Score chunks against the question and return the best `top_k`, ranked."""
    if not chunks:
        return []
    top_k = top_k or get_settings().top_k_rerank

    scores = _model().predict([(question, c["text"]) for c in chunks])
    # Cross-encoder outputs an unbounded logit; squash to 0-1 for the API contract.
    ranked = [{**chunk, "rerank_score": _sigmoid(float(s))} for chunk, s in zip(chunks, scores)]

    # Stable sort: equal scores keep their original retrieval order. (Requirements §3.3)
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
