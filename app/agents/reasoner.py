"""ReasonerAgent — generates a grounded answer from retrieved context.

The whole point of the system is that answers come from the documents, not the
model's training data. That guarantee lives in the ONLY rule in the system prompt
below, backed up by the SimilarityThreshold gate and the Validator. (DECISIONS: C-A)
"""

from __future__ import annotations

from app.agents import llm

REASONER_MAX_TOKENS = 800

# Shared grounding system used by BOTH the Reasoner and the Validator. Keeping it identical
# (and short) is deliberate: the cached prefix is `system + CONTEXT block`, so a shared
# system lets the Validator read the context the Reasoner just cached (DECISIONS: D-16). The
# critical ONLY-rule guardrail stays at the system level; each agent's specific task trails
# the cached context in the user message.
GROUNDING_SYSTEM = (
    "You work strictly from the CONTEXT provided in the user message. Use ONLY that "
    "context — never outside or prior knowledge — and follow the task instructions exactly."
)

REASONER_TASK = """TASK: Answer the QUESTION using only the CONTEXT above.
- If the CONTEXT does not contain the answer, say you cannot answer from the provided
  documents — do not guess.
- Keep the answer concise and factual.

Respond in EXACTLY this format:
ANSWER: <your answer>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASON: <one short sentence on why>
SOURCES: <comma-separated source filenames you used>"""


def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Return `(cache_context, user)`. `cache_context` is the cacheable CONTEXT block —
    byte-identical to the Validator's for the same chunks, so the Validator call reads it
    from cache (D-16). Used by both the streaming and non-streaming Reasoner paths so they
    build the exact same prefix."""
    context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in chunks)
    return f"CONTEXT:\n{context}", f"QUESTION: {question}\n\n{REASONER_TASK}"


def reasoner_agent(question: str, chunks: list[dict], usage: dict | None = None) -> dict:
    """Return a grounded answer dict: answer, confidence, confidence_reason, sources_used."""
    available = [c["filename"] for c in chunks]
    cache_context, user = build_prompt(question, chunks)

    raw = llm.complete(GROUNDING_SYSTEM, user, max_tokens=REASONER_MAX_TOKENS,
                       agent="ReasonerAgent", usage=usage, cache_context=cache_context)
    return _parse_response(raw, available)


def _parse_response(raw: str, available: list[str]) -> dict:
    """Parse the structured response, with a safe fallback if the format slips."""
    answer = raw.strip()
    confidence = "medium"
    reason = ""
    sources: list[str] = []

    if "ANSWER:" in raw:
        after = raw.split("ANSWER:", 1)[1]
        answer = after.split("CONFIDENCE:")[0].strip() if "CONFIDENCE:" in after else after.strip()

    if "CONFIDENCE:" in raw:
        conf = raw.split("CONFIDENCE:", 1)[1].split("\n", 1)[0].upper()
        confidence = "high" if "HIGH" in conf else "low" if "LOW" in conf else "medium"

    if "REASON:" in raw:
        reason = raw.split("REASON:", 1)[1].split("\n", 1)[0].split("SOURCES:")[0].strip()

    if "SOURCES:" in raw:
        names = raw.split("SOURCES:", 1)[1].split("\n", 1)[0].split(",")
        # Only trust filenames we actually passed in — the model can't invent sources.
        sources = [n.strip() for n in names if n.strip() in available]

    if not answer:
        answer = raw.strip()
    if not sources:
        sources = sorted(set(available))

    return {
        "answer": answer,
        "confidence": confidence,
        "confidence_reason": reason,
        "sources_used": sources,
    }
