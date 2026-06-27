"""llama_index mode — a lightweight ReAct-style loop on the Anthropic SDK.

Phase 1 does NOT use LlamaIndex (DECISIONS: D-7). This is a small search→answer
loop that gives the dynamic, variable-LLM-call behaviour the `llama_index` and
`compare` modes are meant to showcase, using our own `llm.complete` + retriever.
Full LlamaIndex ReActAgent integration is deferred to Phase 2.
"""

from __future__ import annotations

from app.agents import llm
from app.core.config import get_settings

LLAMA_SYSTEM = """You are a research assistant answering strictly from a document store.

On each turn reply with EXACTLY one action:
  SEARCH: <query>   — to fetch more context from the store
  FINAL: <answer>   — when the context is enough to answer

Use only the provided context. If it is still insufficient after searching, give
your best FINAL answer and note the gap."""

MAX_STEPS = 4


def run_llama_agent(question: str, store, filter_filenames=None) -> dict:
    """Run the loop. Returns answer, sources_used, llm_calls, chunks, trace."""
    settings = get_settings()
    top_k = settings.top_k_retrieval

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
    for step in range(1, MAX_STEPS + 1):
        context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in seen)
        raw = llm.complete(
            LLAMA_SYSTEM,
            f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nYour next action:",
            agent="LlamaReAct",
        )
        llm_calls += 1

        if "FINAL:" in raw:
            answer = raw.split("FINAL:", 1)[1].strip()
            trace.append(_step(step, "final"))
            break
        if "SEARCH:" in raw:
            query = raw.split("SEARCH:", 1)[1].splitlines()[0].strip()
            remember(store.retrieve(query, top_k=top_k, filter_filenames=filter_filenames))
            trace.append(_step(step, "search", query=query))
            continue
        # No recognised action → treat the whole reply as the answer.
        answer = raw.strip()
        trace.append(_step(step, "final"))
        break
    else:
        answer = answer or "Unable to reach a conclusion from the available documents."

    return {
        "answer": answer,
        "sources_used": sorted({c["filename"] for c in seen}),
        "llm_calls": llm_calls,
        "chunks": seen,
        "trace": trace,
    }


def _step(step: int, action: str, **extra) -> dict:
    return {"agent": "LlamaReAct", "status": "completed", "details": {"step": step, "action": action, **extra}, "duration_ms": 0}
