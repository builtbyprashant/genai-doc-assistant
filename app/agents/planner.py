"""PlannerAgent — understands intent and rewrites the query for retrieval.

An LLM call. If the model returns malformed JSON we fall back to a safe default
plan (the original query) so the pipeline keeps going rather than failing.
(Requirements §3.3)
"""

from __future__ import annotations

from app.agents import llm
from app.core.config import get_settings

PLANNER_SYSTEM = """You analyse a user's question to improve document retrieval.

Return ONLY a JSON object, no other text:
{
  "intent": "<short description of what the user is asking for>",
  "retrieval_strategy": "semantic",
  "rewritten_query": "<query rewritten for dense semantic search>"
}"""


def planner_agent(question: str) -> dict:
    """Return a retrieval plan: intent, retrieval_strategy, top_k, rewritten_query."""
    top_k = get_settings().top_k_retrieval
    raw = llm.complete(PLANNER_SYSTEM, f"QUESTION: {question}", max_tokens=300, agent="PlannerAgent")
    return _safe_plan(raw, question, top_k)


def _safe_plan(raw: str, question: str, top_k: int) -> dict:
    default = {
        "intent": "answer user question",
        "retrieval_strategy": "semantic",
        "top_k": top_k,
        "rewritten_query": question,
    }
    try:
        data = llm.extract_json(raw)
    except ValueError:
        return default

    return {
        "intent": data.get("intent") or default["intent"],
        "retrieval_strategy": data.get("retrieval_strategy") or "semantic",
        "top_k": top_k,  # Phase 1 keeps top_k fixed from config, not model-chosen
        "rewritten_query": data.get("rewritten_query") or question,
    }
