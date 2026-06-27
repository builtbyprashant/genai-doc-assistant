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

from app.agents import llama_agent, llm, planner, ranker, reasoner, safety, validator
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
    top_k_override: int | None = None,
    store=None,
) -> dict:
    """Entry point. Raises SafetyError / FilterNotFoundError for 4xx cases."""
    settings = get_settings()
    store = store if store is not None else get_vector_store()
    top_k = top_k_override or settings.top_k_retrieval

    safety_start = time.perf_counter()
    safety.check(question)  # SafetyError → API 400
    safety_ms = _ms(safety_start)
    started = time.perf_counter()

    if agent_mode == "compare":
        return _run_compare(question, filter_filenames, include_chunks, include_trace,
                            store, settings, top_k, safety_ms, started)

    if agent_mode == "llama_index":
        core = _llama_core(question, filter_filenames, store, top_k)
    else:
        core = _custom_core(question, filter_filenames, store, settings, top_k, safety_ms)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _single_response(question, core, elapsed_ms, include_chunks, include_trace, agent_mode)


# ── custom pipeline ───────────────────────────────────────────────────────────

def _custom_core(question, filter_filenames, store, settings, top_k, safety_ms=0) -> dict:
    trace = [_step("SafetyGuard", "passed", "Input passed all safety checks.", safety_ms)]

    if store.chunk_count() == 0:
        trace.append(_step("RetrieverAgent", "skipped", "no_documents_indexed"))
        return _short_circuit("no_documents_indexed", trace,
                              "No documents are indexed. Cannot retrieve context.")

    _check_filter(filter_filenames, store)

    t = time.perf_counter()
    plan = planner.planner_agent(question)
    query = plan["rewritten_query"]
    trace.append(_step("PlannerAgent", "completed", {
        "intent": plan["intent"], "retrieval_strategy": plan["retrieval_strategy"],
        "top_k": top_k, "rewritten_query": query,
    }, _ms(t)))

    t = time.perf_counter()
    retrieved = store.retrieve(query, top_k=top_k, filter_filenames=filter_filenames)
    retrieve_ms = _ms(t)
    top_score = retrieved[0]["similarity_score"] if retrieved else 0.0
    passed = bool(retrieved) and top_score >= settings.similarity_threshold
    trace.append(_step("RetrieverAgent", "completed", {
        "chunks_retrieved": len(retrieved), "top_similarity_score": top_score,
        "threshold_passed": passed,
    }, retrieve_ms))

    if not passed:
        trace.append(_step("SimilarityThreshold", "failed", {
            "top_score": top_score, "threshold": settings.similarity_threshold, "result": "short_circuit",
        }))
        return _short_circuit(
            "similarity_threshold_not_met", trace,
            f"Top retrieved chunk scored {top_score:.2f}, below the threshold of "
            f"{settings.similarity_threshold:.2f}. No relevant content found.",
            llm_calls=1,  # the Planner already ran before the threshold gate
        )
    trace.append(_step("SimilarityThreshold", "passed", {
        "top_score": top_score, "threshold": settings.similarity_threshold, "result": "continue",
    }))

    t = time.perf_counter()
    ranked = ranker.ranker_agent(query, retrieved, top_k=settings.top_k_rerank)
    trace.append(_step("RankerAgent", "completed", {
        "model": settings.reranker_model, "chunks_in": len(retrieved), "chunks_out": len(ranked),
    }, _ms(t)))

    t = time.perf_counter()
    result = reasoner.reasoner_agent(question, ranked)
    trace.append(_step("ReasonerAgent", "completed", {
        "confidence": result["confidence"], "sources_used": result["sources_used"],
    }, _ms(t)))

    t = time.perf_counter()
    validation = validator.validator_agent(question, result["answer"], ranked)
    trace.append(_step("ValidatorAgent", "completed", {
        "is_valid": validation["is_valid"], "hallucination_risk": validation["hallucination_risk"],
    }, _ms(t)))

    return {
        "success": True, "short_circuit": False, "short_circuit_reason": None,
        "answer": result["answer"], "confidence": result["confidence"],
        "confidence_reason": result["confidence_reason"], "sources_used": result["sources_used"],
        "validation": validation, "chunks": ranked, "trace": trace,
        "llm_calls": 3,  # Planner + Reasoner + Validator
    }


# ── llama_index mode ──────────────────────────────────────────────────────────

def _llama_core(question, filter_filenames, store, top_k) -> dict:
    if store.chunk_count() == 0:
        trace = [_step("LlamaReAct", "skipped", "no_documents_indexed")]
        return _short_circuit("no_documents_indexed", trace,
                              "No documents are indexed. Cannot retrieve context.", llm_calls=0)

    _check_filter(filter_filenames, store)

    run = llama_agent.run_llama_agent(question, store, filter_filenames, top_k=top_k)
    return {
        "success": True, "short_circuit": False, "short_circuit_reason": None,
        "answer": run["answer"], "confidence": run["confidence"], "confidence_reason": "",
        "sources_used": run["sources_used"], "validation": dict(_NEUTRAL_VALIDATION),
        "chunks": run["chunks"], "trace": run["trace"], "llm_calls": run["llm_calls"],
    }


# ── compare mode (C-E) ────────────────────────────────────────────────────────

def _run_compare(question, filter_filenames, include_chunks, include_trace, store, settings, top_k, safety_ms, started) -> dict:
    custom_started = time.perf_counter()
    custom = _custom_core(question, filter_filenames, store, settings, top_k, safety_ms)
    custom_ms = int((time.perf_counter() - custom_started) * 1000)

    # The llama side runs regardless of whether custom short-circuited. (C-E)
    llama_started = time.perf_counter()
    llama = _llama_core(question, filter_filenames, store, top_k)
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

def _single_response(question, core, elapsed_ms, include_chunks, include_trace, agent_mode) -> dict:
    response = {
        "mode": agent_mode,
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


def _ms(start) -> int:
    return int((time.perf_counter() - start) * 1000)


# ── streaming (custom mode) ───────────────────────────────────────────────────

class _AnswerExtractor:
    """Pull just the answer text out of a streamed structured reply.

    The Reasoner replies as `ANSWER: ... CONFIDENCE: ... REASON: ... SOURCES: ...`,
    so while streaming we hide the `ANSWER:` prefix and stop emitting at `CONFIDENCE:`.
    A few trailing chars are held back each step so a partially-arrived stop marker
    never leaks into the visible answer; `flush()` releases the remainder at the end.
    """

    def __init__(self, start: str, stop: str):
        self.start, self.stop = start, stop
        self.emitted = 0

    def _region(self, raw: str):
        after = raw
        if self.start:
            if self.start not in raw:
                return None, False
            after = raw.split(self.start, 1)[1]
        if self.stop in after:
            return after.split(self.stop, 1)[0], True
        return after, False

    def feed(self, raw: str) -> str:
        region, stopped = self._region(raw)
        if region is None:
            return ""
        end = len(region) if stopped else max(self.emitted, len(region) - len(self.stop))
        out = region[self.emitted:end]
        self.emitted = end
        return out

    def flush(self, raw: str) -> str:
        region, _ = self._region(raw)
        if region is None:
            region = raw
        out = region[self.emitted:]
        self.emitted = len(region)
        return out


def run_pipeline_stream(
    question: str,
    *,
    filter_filenames: list[str] | None = None,
    agent_mode: str = "custom",
    top_k_override: int | None = None,
    store=None,
):
    """Generator yielding ('token', text) for answer deltas then ('done', result).

    Only custom mode is token-streamed; other modes run batch and emit their answer
    in one piece so the streaming endpoint behaves uniformly. Raises SafetyError /
    FilterNotFoundError exactly like `run_pipeline` (the endpoint pre-checks those
    before any byte is sent, so they surface as clean 4xx responses).
    """
    settings = get_settings()
    store = store if store is not None else get_vector_store()
    top_k = top_k_override or settings.top_k_retrieval

    safety_start = time.perf_counter()
    safety.check(question)
    safety_ms = _ms(safety_start)
    started = time.perf_counter()

    if agent_mode == "custom":
        yield from _custom_stream(question, filter_filenames, store, settings, top_k, safety_ms, started)
        return

    result = run_pipeline(
        question, filter_filenames=filter_filenames, include_chunks=True,
        include_trace=True, agent_mode=agent_mode, top_k_override=top_k_override, store=store,
    )
    if result.get("answer"):
        yield ("token", result["answer"])
    yield ("done", result)


def _custom_stream(question, filter_filenames, store, settings, top_k, safety_ms, started):
    trace = [_step("SafetyGuard", "passed", "Input passed all safety checks.", safety_ms)]

    def done(core):
        return ("done", _single_response(question, core, _ms(started), True, True, "custom"))

    if store.chunk_count() == 0:
        trace.append(_step("RetrieverAgent", "skipped", "no_documents_indexed"))
        yield done(_short_circuit("no_documents_indexed", trace,
                                  "No documents are indexed. Cannot retrieve context."))
        return

    _check_filter(filter_filenames, store)

    t = time.perf_counter()
    plan = planner.planner_agent(question)
    query = plan["rewritten_query"]
    trace.append(_step("PlannerAgent", "completed", {
        "intent": plan["intent"], "retrieval_strategy": plan["retrieval_strategy"],
        "top_k": top_k, "rewritten_query": query,
    }, _ms(t)))

    t = time.perf_counter()
    retrieved = store.retrieve(query, top_k=top_k, filter_filenames=filter_filenames)
    retrieve_ms = _ms(t)
    top_score = retrieved[0]["similarity_score"] if retrieved else 0.0
    passed = bool(retrieved) and top_score >= settings.similarity_threshold
    trace.append(_step("RetrieverAgent", "completed", {
        "chunks_retrieved": len(retrieved), "top_similarity_score": top_score,
        "threshold_passed": passed,
    }, retrieve_ms))

    if not passed:
        trace.append(_step("SimilarityThreshold", "failed", {
            "top_score": top_score, "threshold": settings.similarity_threshold, "result": "short_circuit",
        }))
        yield done(_short_circuit(
            "similarity_threshold_not_met", trace,
            f"Top retrieved chunk scored {top_score:.2f}, below the threshold of "
            f"{settings.similarity_threshold:.2f}. No relevant content found.",
            llm_calls=1,
        ))
        return
    trace.append(_step("SimilarityThreshold", "passed", {
        "top_score": top_score, "threshold": settings.similarity_threshold, "result": "continue",
    }))

    t = time.perf_counter()
    ranked = ranker.ranker_agent(query, retrieved, top_k=settings.top_k_rerank)
    trace.append(_step("RankerAgent", "completed", {
        "model": settings.reranker_model, "chunks_in": len(retrieved), "chunks_out": len(ranked),
    }, _ms(t)))

    # Stream the Reasoner's answer token-by-token, hiding the structured scaffolding.
    available = [c["filename"] for c in ranked]
    context = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in ranked)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    t = time.perf_counter()
    raw = ""
    extractor = _AnswerExtractor(start="ANSWER:", stop="CONFIDENCE:")
    for delta in llm.stream_text(reasoner.REASONER_SYSTEM, user,
                                 max_tokens=reasoner.REASONER_MAX_TOKENS, agent="ReasonerAgent"):
        raw += delta
        emit = extractor.feed(raw)
        if emit:
            yield ("token", emit)
    tail = extractor.flush(raw)
    if tail:
        yield ("token", tail)
    parsed = reasoner._parse_response(raw, available)
    trace.append(_step("ReasonerAgent", "completed", {
        "confidence": parsed["confidence"], "sources_used": parsed["sources_used"],
    }, _ms(t)))

    t = time.perf_counter()
    validation = validator.validator_agent(question, parsed["answer"], ranked)
    trace.append(_step("ValidatorAgent", "completed", {
        "is_valid": validation["is_valid"], "hallucination_risk": validation["hallucination_risk"],
    }, _ms(t)))

    yield done({
        "success": True, "short_circuit": False, "short_circuit_reason": None,
        "answer": parsed["answer"], "confidence": parsed["confidence"],
        "confidence_reason": parsed["confidence_reason"], "sources_used": parsed["sources_used"],
        "validation": validation, "chunks": ranked, "trace": trace, "llm_calls": 3,
    })
