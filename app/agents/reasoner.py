"""ReasonerAgent — generates a grounded answer from retrieved context.

The whole point of the system is that answers come from the documents, not the
model's training data. That guarantee lives in the ONLY rule in the system prompt
below, backed up by the SimilarityThreshold gate and the Validator. (DECISIONS: C-A)
"""

from __future__ import annotations

from app.agents import llm

REASONER_MAX_TOKENS = 800

REASONER_SYSTEM = """You answer questions strictly from the provided CONTEXT.

Rules:
- Use ONLY the information in the CONTEXT. Never use outside or prior knowledge.
- If the CONTEXT does not contain the answer, say you cannot answer from the
  provided documents — do not guess.
- Keep the answer concise and factual.

Respond in EXACTLY this format:
ANSWER: <your answer>
CONFIDENCE: <HIGH|MEDIUM|LOW>
REASON: <one short sentence on why>
SOURCES: <comma-separated source filenames you used>"""


def reasoner_agent(question: str, chunks: list[dict]) -> dict:
    """Return a grounded answer dict: answer, confidence, confidence_reason, sources_used."""
    available = [c["filename"] for c in chunks]
    context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in chunks)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    raw = llm.complete(REASONER_SYSTEM, user, max_tokens=REASONER_MAX_TOKENS)
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
