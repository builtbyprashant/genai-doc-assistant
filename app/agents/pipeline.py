"""The query pipeline orchestrator.

Routes a question through one of three modes (DECISIONS: C-D). `agent_mode` is a
*parameter* — the pipeline never reads the AGENT_MODE env var itself; the API
resolves the default and passes it in. (DECISIONS: D-6d)

Custom mode is the 7 named steps (C-H):
  SafetyGuard → PlannerAgent → RetrieverAgent → SimilarityThreshold
              → RankerAgent → ReasonerAgent → ValidatorAgent
with two short-circuits (empty store, below threshold) that return HTTP 200 with
`short_circuit: true` rather than an error. (DECISIONS: C-F)
"""

from __future__ import annotations

import time

from app.agents import llama_agent, planner, ranker, reasoner, safety, validator
from app.core.config import get_settings
from app.services.vector_store import get_vector_store

_NEUTRAL_VALIDATION = {
    "is_valid": True, "issues": [], "hallucination_risk": "n/a", "suggested_action": "none",
}


class FilterNotFoundError(Exception):
    """A query filter named documents that aren't indexed → API 404 filter-not-found."""

    def __init__(self, filenames: list[str]):
        super().__init__(", ".join(filenames))
        self.code = "filter-not-found"
        self.filenames = filenames


def run_pipeline(
    question: str,
    *,
    filter_filenames: list[str] | None = None,
    include_chunks: bool = False,
    include_trace: bool = False,
    agent_mode: str = "custom",
    store=None,
) -> dict:
    """Entry point. Raises SafetyError / FilterNotFoundError for 4xx cases."""
    settings = get_settings()
    store = store if store is not None else get_vector_store()

    safety.check(question)  # SafetyError → API 400
    started = time.perf_counter()

    if agent_mode == "compare":
        return _run_compare(question, filter_filenames, include_chunks, include_trace, store, settings, started)

    if agent_mode == "llama_index":
        core = _llama_core(question, filter_filenames, store)
    else:
        core = _custom_core(question, filter_filenames, store, settings)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _single_response(question, core, elapsed_ms, include_chunks, include_trace)


# ── custom pipeline ───────────────────────────────────────────────────────────

def _custom_core(question, filter_filenames, store, settings) -> dict:
    trace = [_step("SafetyGuard", "passed", "Input passed all safety checks.")]

    if store.chunk_count() == 0:
        trace.append(_step("RetrieverAgent", "skipped", "no_documents_indexed"))
        return _short_circuit("no_documents_indexed", trace,
                              "No documents are indexed. Cannot retrieve context.")

    _check_filter(filter_filenames, store)

    plan = planner.planner_agent(question)
    query = plan["rewritten_query"]
    trace.append(_step("PlannerAgent", "completed", {
        "intent": plan["intent"], "retrieval_strategy": plan["retrieval_strategy"],
        "top_k": plan["top_k"], "rewritten_query": query,
    }))

    retrieved = store.retrieve(query, top_k=settings.top_k_retrieval, filter_filenames=filter_filenames)
    top_score = retrieved[0]["similarity_score"] if retrieved else 0.0
    passed = bool(retrieved) and top_score >= settings.similarity_threshold
    trace.append(_step("RetrieverAgent", "completed", {
        "chunks_retrieved": len(retrieved), "top_similarity_score": top_score,
        "threshold_passed": passed,
    }))

    if not passed:
        trace.append(_step("SimilarityThreshold", "failed", {
            "top_score": top_score, "threshold": settings.similarity_threshold, "result": "short_circuit",
        }))
        return _short_circuit(
            "similarity_threshold_not_met", trace,
            f"Top retrieved chunk scored {top_score:.2f}, below the threshold of "
            f"{settings.similarity_threshold:.2f}. No relevant content found.",
        )
    trace.append(_step("SimilarityThreshold", "passed", {
        "top_score": top_score, "threshold": settings.similarity_threshold, "result": "continue",
    }))

    ranked = ranker.ranker_agent(query, retrieved, top_k=settings.top_k_rerank)
    trace.append(_step("RankerAgent", "completed", {
        "model": "ms-marco-MiniLM-L-6-v2", "chunks_in": len(retrieved), "chunks_out": len(ranked),
    }))

    result = reasoner.reasoner_agent(question, ranked)
    trace.append(_step("ReasonerAgent", "completed", {
        "confidence": result["confidence"], "sources_used": result["sources_used"],
    }))

    validation = validator.validator_agent(question, result["answer"], ranked)
    trace.append(_step("ValidatorAgent", "completed", {
        "is_valid": validation["is_valid"], "hallucination_risk": validation["hallucination_risk"],
    }))

    return {
        "success": True, "short_circuit": False, "short_circuit_reason": None,
        "answer": result["answer"], "confidence": result["confidence"],
        "confidence_reason": result["confidence_reason"], "sources_used": result["sources_used"],
        "validation": validation, "chunks": ranked, "trace": trace,
        "llm_calls": 3,  # Planner + Reasoner + Validator
    }


# ── llama_index mode ──────────────────────────────────────────────────────────

def _llama_core(question, filter_filenames, store) -> dict:
    if store.chunk_count() == 0:
        trace = [_step("LlamaReAct", "skipped", "no_documents_indexed")]
        return _short_circuit("no_documents_indexed", trace,
                              "No documents are indexed. Cannot retrieve context.", llm_calls=0)

    _check_filter(filter_filenames, store)

    run = llama_agent.run_llama_agent(question, store, filter_filenames)
    return {
        "success": True, "short_circuit": False, "short_circuit_reason": None,
        "answer": run["answer"], "confidence": "n/a", "confidence_reason": "",
        "sources_used": run["sources_used"], "validation": dict(_NEUTRAL_VALIDATION),
        "chunks": run["chunks"], "trace": run["trace"], "llm_calls": run["llm_calls"],
    }


# ── compare mode (C-E) ────────────────────────────────────────────────────────

def _run_compare(question, filter_filenames, include_chunks, include_trace, store, settings, started) -> dict:
    custom_started = time.perf_counter()
    custom = _custom_core(question, filter_filenames, store, settings)
    custom_ms = int((time.perf_counter() - custom_started) * 1000)

    # The llama side runs regardless of whether custom short-circuited. (C-E)
    llama_started = time.perf_counter()
    llama = _llama_core(question, filter_filenames, store)
    llama_ms = int((time.perf_counter() - llama_started) * 1000)

    short_circuit = custom["short_circuit"]
    # Validation targets the custom answer normally; if custom short-circuited it
    # has no answer, so validate the llama answer instead. (C-E)
    if short_circuit:
        validation = (
            validator.validator_agent(question, llama["answer"], llama["chunks"])
            if llama["answer"] else dict(_NEUTRAL_VALIDATION)
        )
    else:
        validation = custom["validation"]

    return {
        "mode": "compare",
        "processing_time_ms": int((time.perf_counter() - started) * 1000),
        "short_circuit": short_circuit,
        "question": question,
        "custom": _compare_side(custom, custom_ms, include_chunks, include_trace),
        "llama_index": _compare_side(llama, llama_ms, include_chunks, include_trace),
        "validation": validation,
    }


def _compare_side(core, duration_ms, include_chunks, include_trace) -> dict:
    return {
        "answer": core["answer"], "confidence": core["confidence"],
        "sources_used": core["sources_used"], "llm_calls": core["llm_calls"],
        "duration_ms": duration_ms,
        "chunks": _public_chunks(core["chunks"]) if include_chunks else [],
        "trace": core["trace"] if include_trace else [],
    }


# ── shared helpers ────────────────────────────────────────────────────────────

def _single_response(question, core, elapsed_ms, include_chunks, include_trace) -> dict:
    response = {
        "processing_time_ms": elapsed_ms,
        "success": core["success"],
        "short_circuit": core["short_circuit"],
        "short_circuit_reason": core["short_circuit_reason"],
        "question": question,
        "answer": core["answer"],
        "confidence": core["confidence"],
        "confidence_reason": core["confidence_reason"],
        "sources_used": core["sources_used"],
        "validation": core["validation"],
    }
    if include_chunks:
        response["chunks"] = _public_chunks(core["chunks"])
    if include_trace:
        response["trace"] = core["trace"]
    return response


def _short_circuit(reason, trace, issue, llm_calls=0) -> dict:
    return {
        "success": False, "short_circuit": True, "short_circuit_reason": reason,
        "answer": "", "confidence": "n/a", "confidence_reason": "", "sources_used": [],
        "validation": {"is_valid": False, "issues": [issue], "hallucination_risk": "n/a", "suggested_action": "none"},
        "chunks": [], "trace": trace, "llm_calls": llm_calls,
    }


def _check_filter(filter_filenames, store) -> None:
    if filter_filenames:
        unknown = store.unknown_filenames(filter_filenames)
        if unknown:
            raise FilterNotFoundError(unknown)


def _public_chunks(chunks) -> list[dict]:
    return [
        {
            "filename": c["filename"], "chunk_index": c["chunk_index"],
            "similarity_score": c.get("similarity_score", 0.0),
            "rerank_score": c.get("rerank_score", 0.0), "text": c["text"],
        }
        for c in chunks
    ]


def _step(agent, status, details, duration_ms=0) -> dict:
    return {"agent": agent, "status": status, "details": details, "duration_ms": duration_ms}
