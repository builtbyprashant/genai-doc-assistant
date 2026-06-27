"""FastAPI backend — all routes for the RAG system.

Every response carries an X-Request-ID (tied to logs) and X-Response-Time-Ms.
Errors are returned as RFC-7807 problem details via a small set of exception
handlers, so no route hand-rolls error JSON. Short-circuits are NOT errors — they
return 200 with `short_circuit: true`. (DECISIONS: C-F)
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agents import safety
from app.agents.pipeline import (
    FilterNotFoundError,
    _check_filter,
    run_pipeline,
    run_pipeline_stream,
)
from app.agents.safety import SafetyError
from app.core.config import get_settings
from app.core.errors import LLMUnavailableError, problem_detail, status_for
from app.services.chunk_service import chunk_document
from app.services.document_loader import DocumentError, load_document
from app.services.vector_store import get_vector_store
from app.utils.logging import get_logger

app = FastAPI(title="Agentic RAG Knowledge System", version="1.0.0")
logger = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ── request/response context ──────────────────────────────────────────────────

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Response-Time-Ms"] = str(int((time.perf_counter() - start) * 1000))
    return response


# ── error handlers (RFC 7807) ─────────────────────────────────────────────────

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None)


def _problem_response(request: Request, code: str, detail: str, **extra) -> JSONResponse:
    body = problem_detail(code, detail, instance=str(request.url.path),
                          request_id=_request_id(request), **extra)
    return JSONResponse(status_code=status_for(code), content=body)


@app.exception_handler(SafetyError)
async def _handle_safety(request: Request, exc: SafetyError):
    return _problem_response(request, exc.code, exc.message)


@app.exception_handler(FilterNotFoundError)
async def _handle_filter(request: Request, exc: FilterNotFoundError):
    return _problem_response(
        request, "filter-not-found",
        f"The following filenames were not found in the index: {', '.join(exc.filenames)}.",
    )


@app.exception_handler(DocumentError)
async def _handle_document(request: Request, exc: DocumentError):
    return _problem_response(request, exc.code, exc.message)


@app.exception_handler(LLMUnavailableError)
async def _handle_llm(request: Request, exc: LLMUnavailableError):
    return _problem_response(
        request, "llm-unavailable",
        "The language model service is unavailable. Please try again in a moment.",
        failed_at_agent=exc.failed_at_agent, retry_after=exc.retry_after,
    )


@app.exception_handler(RequestValidationError)
async def _handle_validation(request: Request, exc: RequestValidationError):
    return _problem_response(request, "validation-error", "Request validation failed.",
                             errors=exc.errors())


# ── models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    filter_filenames: list[str] = []
    include_chunks: bool = False
    include_trace: bool = False
    top_k_override: Optional[int] = Field(default=None, ge=1, le=20)
    agent_mode: Optional[Literal["custom", "llama_index", "compare"]] = None


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Agentic RAG Knowledge System", "version": "1.0.0", "phase": "1",
        "docs": "/docs", "health": "/health", "status": "running",
    }


@app.get("/health")
def health(request: Request):
    settings = get_settings()
    stats = {
        "documents_indexed": 0, "total_chunks": 0,
        "max_documents": settings.max_documents, "documents_remaining": settings.max_documents,
        "similarity_threshold": settings.similarity_threshold,
    }
    status = "ok"
    try:
        store = get_vector_store()
        docs = store.list_documents()
        stats["documents_indexed"] = len(docs)
        stats["total_chunks"] = store.chunk_count()
        stats["documents_remaining"] = settings.max_documents - len(docs)
    except Exception:  # degrade gracefully if ChromaDB is unreachable
        status = "degraded"
    return {"status": status, "version": "1.0.0", "request_id": _request_id(request),
            "timestamp": _now(), "stats": stats}


@app.post("/documents/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    settings = get_settings()
    store = get_vector_store()
    known_hashes = {d["content_hash"]: d["filename"] for d in store.list_documents()}
    known_names = store.existing_filenames()

    results, succeeded, failed = [], 0, 0
    for upload in files:
        content = await upload.read()
        filename = upload.filename
        try:
            is_new = filename not in known_names
            if is_new and len(known_names) >= settings.max_documents:
                raise DocumentError(
                    "max-documents-reached",
                    f"Maximum document limit of {settings.max_documents} has been reached. "
                    "Please delete an existing document before uploading a new one.",
                )

            doc = load_document(filename, content, existing_hashes=known_hashes)
            chunks = chunk_document(doc["text"], filename, content)

            action = "indexed" if is_new else "replaced"
            if not is_new:
                store.delete_document(filename)  # replace = delete then re-index

            doc_id = str(uuid.uuid4())
            store.index_chunks(doc_id, filename, chunks,
                               content_hash=doc["content_hash"], size_bytes=doc["size_bytes"])

            known_hashes[doc["content_hash"]] = filename
            known_names.add(filename)
            results.append({
                "filename": filename, "status": "success", "doc_id": doc_id,
                "action": action, "chunks_indexed": len(chunks),
                "total_chars": len(doc["text"]), "warnings": doc["warnings"],
            })
            succeeded += 1
        except DocumentError as exc:
            results.append({
                "filename": filename, "status": "failed", "doc_id": None, "action": None,
                "chunks_indexed": 0, "total_chars": 0,
                "error": problem_detail(exc.code, exc.message, instance="/documents/upload",
                                        request_id=_request_id(request)),
            })
            failed += 1

    return JSONResponse(status_code=207, content={
        "request_id": _request_id(request), "timestamp": _now(),
        "summary": {"total": len(files), "succeeded": succeeded, "failed": failed},
        "results": results,
    })


@app.get("/documents")
def list_documents(request: Request):
    store = get_vector_store()
    docs = store.list_documents()
    return {"request_id": _request_id(request), "timestamp": _now(),
            "count": len(docs), "documents": docs}


@app.delete("/documents/{filename}")
def delete_document(request: Request, filename: str):
    store = get_vector_store()
    doc = next((d for d in store.list_documents() if d["filename"] == filename), None)
    if doc is None:
        return _problem_response(request, "document-not-found",
                                 f"No document with filename '{filename}' exists in the index.")
    removed = store.delete_document(filename)
    return {"request_id": _request_id(request), "timestamp": _now(),
            "filename": filename, "doc_id": doc["doc_id"],
            "chunks_removed": removed, "status": "deleted"}


@app.post("/query")
def query(request: Request, body: QueryRequest):
    mode = body.agent_mode or get_settings().agent_mode
    result = run_pipeline(
        body.question,
        filter_filenames=body.filter_filenames,
        include_chunks=body.include_chunks,
        include_trace=body.include_trace,
        agent_mode=mode,
        top_k_override=body.top_k_override,
    )
    result["request_id"] = _request_id(request)
    result["timestamp"] = _now()
    return result


# Streamed answer (custom mode streams token-by-token; other modes emit in one piece).
# The body is the answer text, then a 0x1E (record separator) byte, then a JSON blob
# with the full metadata (confidence, sources, validation, trace, timing).
STREAM_META_SEP = "\x1e"


@app.post("/query/stream")
def query_stream(request: Request, body: QueryRequest):
    mode = body.agent_mode or get_settings().agent_mode
    # Pre-check safety and filters up front so they surface as clean RFC-7807 4xx
    # responses rather than blowing up mid-stream after headers are already sent.
    safety.check(body.question)
    store = get_vector_store()
    if body.filter_filenames:
        _check_filter(body.filter_filenames, store)

    request_id, timestamp = _request_id(request), _now()

    def generate():
        meta = None
        for kind, payload in run_pipeline_stream(
            body.question, filter_filenames=body.filter_filenames,
            agent_mode=mode, top_k_override=body.top_k_override, store=store,
        ):
            if kind == "token":
                yield payload
            else:
                meta = payload
        if meta is not None:
            meta["request_id"] = request_id
            meta["timestamp"] = timestamp
            yield STREAM_META_SEP + json.dumps(meta)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
