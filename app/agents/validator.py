"""ValidatorAgent — independent hallucination check on the final answer.

Independence is the whole value here: the Validator sees ONLY the question, the
answer text, and the retrieved chunks — never the Reasoner's reasoning, the
Planner output, or any other pipeline state. A fresh check with no memory of how
the answer was produced avoids self-consistency bias. (DECISIONS: D-5)

Malformed JSON falls back to a permissive default so a parser hiccup never blocks
a response. (Requirements §3.3)
"""

from __future__ import annotations

from app.agents import llm

VALIDATOR_SYSTEM = """You judge whether an ANSWER is fully supported by the CONTEXT.

You are given only the question, the answer, and the context — not how the answer
was generated. Judge solely on whether the context backs up the answer.

Return ONLY a JSON object, no other text:
{
  "is_valid": true,
  "issues": ["..."],
  "hallucination_risk": "low|medium|high",
  "suggested_action": "none|review|regenerate"
}"""


def validator_agent(question: str, answer: str, chunks: list[dict]) -> dict:
    """Validate an answer against its context. Returns the validation dict."""
    context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in chunks)
    user = f"QUESTION: {question}\n\nANSWER: {answer}\n\nCONTEXT:\n{context}"

    raw = llm.complete(VALIDATOR_SYSTEM, user, max_tokens=300, agent="ValidatorAgent")
    return _safe_validation(raw)


def _safe_validation(raw: str) -> dict:
    default = {
        "is_valid": True,
        "issues": [],
        "hallucination_risk": "unknown",
        "suggested_action": "none",
    }
    try:
        data = llm.extract_json(raw)
    except ValueError:
        return default

    return {
        "is_valid": bool(data.get("is_valid", True)),
        "issues": data.get("issues") or [],
        "hallucination_risk": data.get("hallucination_risk") or "unknown",
        "suggested_action": data.get("suggested_action") or "none",
    }
