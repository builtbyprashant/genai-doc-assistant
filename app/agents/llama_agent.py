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

# Per-turn instructions live in the user message, AFTER the cached CONTEXT block (D-16), so
# the [system + QUESTION + CONTEXT] prefix is identical across turns and the growing context
# is read from cache each turn. The whole loop — including the forced answer — uses
# LLAMA_SYSTEM, so the synthesis turn shares the cached context with the search turns (rather
# than a separate synthesis system, which couldn't).
NEXT_ACTION = "Your next action:"
FORCE_FINAL = (
    "You have used all available search turns. Reply now with your FINAL answer using ONLY "
    "the context above; be direct and concise, and if the context does not contain the answer "
    "say so briefly.\nFINAL: <answer>\nCONFIDENCE: HIGH | MEDIUM | LOW"
)

MAX_STEPS = 4

# Diminishing-returns guard: if a SEARCH surfaces this fraction (or more) of chunks the
# loop has already seen, the model is re-asking the same thing — stop searching and answer
# from what we have instead of burning the rest of the search budget on redundant retrievals.
# (A constant alongside MAX_STEPS; adjust to make the loop more/less eager to keep searching.)
OVERLAP_STOP_RATIO = 0.7


def run_llama_agent(question: str, store, filter_filenames=None, top_k=None, usage=None) -> dict:
    """Run the loop. Returns answer, sources_used, llm_calls, chunks, trace."""
    top_k = top_k or get_settings().top_k_retrieval

    seen: list[dict] = []
    seen_ids: set[str] = set()
    trace: list[dict] = []
    llm_calls = 0

    def remember(chunks) -> int:
        """Add only chunks we haven't seen; return how many were newly added."""
        added = 0
        for chunk in chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                seen.append(chunk)
                added += 1
        return added

    def context_text() -> str:
        return "\n\n".join(f"[{c['filename']}] {c['text']}" for c in seen)

    def cache_block() -> str:
        # The cacheable prefix: question + accumulated context. It grows append-only as
        # chunks are remembered, so each turn's prefix extends the previous turn's and the
        # already-seen context is read from cache instead of re-processed (D-16).
        return f"QUESTION: {question}\n\nCONTEXT:\n{context_text()}"

    def ask(instruction: str) -> str:
        return llm.complete(LLAMA_SYSTEM, instruction, agent="LlamaReAct",
                            usage=usage, cache_context=cache_block())

    remember(store.retrieve(question, top_k=top_k, filter_filenames=filter_filenames))

    answer = ""
    confidence = "medium"
    for step in range(1, MAX_STEPS + 1):
        t = time.perf_counter()

        if step == MAX_STEPS:
            # Out of search budget — force a direct answer from everything gathered.
            answer, confidence = _extract_final(ask(FORCE_FINAL))
            llm_calls += 1
            trace.append(_step(step, "final", _ms(t), reason="max_steps"))
            break

        raw = ask(NEXT_ACTION)
        step_ms = _ms(t)
        llm_calls += 1

        if "FINAL:" in raw:
            answer, confidence = _extract_final(raw)
            trace.append(_step(step, "final", step_ms))
            break
        if "SEARCH:" in raw:
            query = raw.split("SEARCH:", 1)[1].splitlines()[0].strip()
            fetched = store.retrieve(query, top_k=top_k, filter_filenames=filter_filenames)
            added = remember(fetched)
            overlap = 1.0 - added / len(fetched) if fetched else 1.0
            trace.append(_step(step, "search", step_ms, query=query,
                               new_chunks=added, overlap=round(overlap, 2)))
            if added == 0 or overlap >= OVERLAP_STOP_RATIO:
                # Diminishing returns — synthesize now rather than search again.
                t2 = time.perf_counter()
                answer, confidence = _extract_final(ask(FORCE_FINAL))
                llm_calls += 1
                trace.append(_step(step, "final", _ms(t2), reason="diminishing_returns"))
                break
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


def _ms(t: float) -> int:
    return int((time.perf_counter() - t) * 1000)


def _extract_final(raw: str) -> tuple[str, str]:
    """(answer, confidence) from a FINAL reply — tolerant of the `FINAL:` prefix being
    present (a chosen or forced final) or absent (the model deviated from the protocol)."""
    body = raw.split("FINAL:", 1)[1] if "FINAL:" in raw else raw
    return _split_confidence(body)


def _step(step: int, action: str, duration_ms: int = 0, **extra) -> dict:
    # Label each step by its action so the trace reads as the ReAct loop it is
    # (Search → Search → … → Synthesize) instead of a wall of identical "LlamaReAct".
    label = {"search": "Search", "final": "Synthesize"}.get(action, action.title())
    return {"agent": f"LlamaReAct · {label}", "status": "completed",
            "details": {"step": step, **extra}, "duration_ms": duration_ms}


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
