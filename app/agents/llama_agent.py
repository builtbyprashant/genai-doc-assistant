"""llama_index mode — a lightweight ReAct-style loop on the Anthropic SDK.

Phase 1 does NOT use LlamaIndex (DECISIONS: D-7). This is a small search→answer
loop that gives the dynamic, variable-LLM-call behaviour the `llama_index` and
`compare` modes are meant to showcase, using our own `llm.complete` + retriever.
Full LlamaIndex ReActAgent integration is deferred to Phase 2.
"""

from __future__ import annotations

import time

from app.agents import llm
from app.core.config import get_settings

LLAMA_SYSTEM = """You are a research assistant answering strictly from a document store.

On each turn reply with EXACTLY one action:
  SEARCH: <query>   — to fetch more context from the store
  FINAL: <answer>   — when the context is enough to answer

After FINAL, add a line:  CONFIDENCE: HIGH | MEDIUM | LOW

Use only the provided context. If it is still insufficient after searching, give
your best FINAL answer and note the gap."""

# Used on the last turn to force a direct answer from what was gathered, rather
# than exhausting the search budget and giving up.
FINAL_SYNTHESIS_SYSTEM = (
    "Answer the question using ONLY the provided context. Be direct and concise. "
    "If the context genuinely does not contain the answer, say so briefly. "
    "End with a line:  CONFIDENCE: HIGH | MEDIUM | LOW"
)

MAX_STEPS = 4


def run_llama_agent(question: str, store, filter_filenames=None, top_k=None) -> dict:
    """Run the loop. Returns answer, sources_used, llm_calls, chunks, trace."""
    top_k = top_k or get_settings().top_k_retrieval

    seen: list[dict] = []
    seen_ids: set[str] = set()
    trace: list[dict] = []
    llm_calls = 0

    def remember(chunks):
        for chunk in chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                seen.append(chunk)

    remember(store.retrieve(question, top_k=top_k, filter_filenames=filter_filenames))

    answer = ""
    confidence = "medium"
    for step in range(1, MAX_STEPS + 1):
        context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in seen)
        t = time.perf_counter()

        if step == MAX_STEPS:
            # Out of search budget — force a direct answer from everything gathered
            # instead of returning a "couldn't conclude" message.
            raw = llm.complete(
                FINAL_SYNTHESIS_SYSTEM,
                f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\n"
                "Answer the question directly using only this context.",
                agent="LlamaReAct",
            )
            llm_calls += 1
            answer, confidence = _split_confidence(raw)
            trace.append(_step(step, "final", int((time.perf_counter() - t) * 1000)))
            break

        raw = llm.complete(
            LLAMA_SYSTEM,
            f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nYour next action:",
            agent="LlamaReAct",
        )
        step_ms = int((time.perf_counter() - t) * 1000)
        llm_calls += 1

        if "FINAL:" in raw:
            answer, confidence = _split_confidence(raw.split("FINAL:", 1)[1])
            trace.append(_step(step, "final", step_ms))
            break
        if "SEARCH:" in raw:
            query = raw.split("SEARCH:", 1)[1].splitlines()[0].strip()
            remember(store.retrieve(query, top_k=top_k, filter_filenames=filter_filenames))
            trace.append(_step(step, "search", step_ms, query=query))
            continue
        # No recognised action → treat the whole reply as the answer.
        answer, confidence = _split_confidence(raw)
        trace.append(_step(step, "final", step_ms))
        break

    if not answer:
        answer = "Unable to reach a conclusion from the available documents."

    return {
        "answer": answer,
        "confidence": confidence,
        "sources_used": sorted({c["filename"] for c in seen}),
        "llm_calls": llm_calls,
        "chunks": seen,
        "trace": trace,
    }


def _step(step: int, action: str, duration_ms: int = 0, **extra) -> dict:
    return {"agent": "LlamaReAct", "status": "completed",
            "details": {"step": step, "action": action, **extra}, "duration_ms": duration_ms}


def _split_confidence(text: str) -> tuple[str, str]:
    """Split a final reply into (answer, confidence). Confidence defaults to medium
    if the model didn't emit a CONFIDENCE line."""
    confidence = "medium"
    if "CONFIDENCE:" in text:
        body, tail = text.split("CONFIDENCE:", 1)
        text = body
        token = tail.split("\n", 1)[0].upper()
        confidence = "high" if "HIGH" in token else "low" if "LOW" in token else "medium"
    return text.strip(), confidence
