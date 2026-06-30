# Agentic RAG Knowledge System: API Contract

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Global Conventions](#global-conventions)
- [Error Response Format (RFC 7807 Problem Details)](#error-response-format-rfc-7807-problem-details)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [POST /documents/upload](#post-documentsupload)
  - [GET /documents](#get-documents)
  - [DELETE /documents/{filename}](#delete-documentsfilename)
  - [POST /query](#post-query)
  - [POST /query/stream](#post-querystream)
  - [GET /](#get-)
- [Endpoint Summary](#endpoint-summary)
- [HTTP Status Code Usage](#http-status-code-usage)
- [Streamlit UI: API Usage Map](#streamlit-ui-api-usage-map)
- [Future Version Changes (Forward Reference)](#future-version-changes-forward-reference)

---

## Overview

This document defines the complete REST API contract for the backend (FastAPI).
It is the authoritative reference for both backend implementation and frontend (Streamlit) integration.
Every endpoint, request shape, response shape, status code, and error format is defined here.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Base URL

| Environment | Base URL |
|---|---|
| Local (outside Docker) | `http://localhost:8000` |
| Inside Docker network | `http://backend:8000` |
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Global Conventions

### HTTP Methods
- `GET`, read-only, no side effects
- `POST`, create or trigger an operation
- `DELETE`, remove a resource

### Request Headers
All `POST` requests with a JSON body must include:
```
Content-Type: application/json
```
File upload requests use:
```
Content-Type: multipart/form-data
```

### Response Headers
All responses include:
```
Content-Type: application/json
X-Request-ID: <uuid>       ← present on every response, ties to trace logs
X-Response-Time-Ms: <int>  ← total server-side processing time in milliseconds
```

### Timestamps
All timestamps are ISO 8601 UTC strings:
```
"2024-06-25T14:32:10.123Z"
```

### Boolean flags
All boolean fields use JSON `true` / `false`. Never `"true"` / `"false"` as strings.

### Pagination
There is currently no pagination. All list endpoints return complete results.
Document this as a known limitation, a future version adds cursor-based pagination on `/documents`.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Error Response Format (RFC 7807 Problem Details)

All error responses follow RFC 7807. Every non-2xx response uses this shape:

```json
{
  "type": "https://rag-api/errors/document-not-found",
  "title": "Document not found",
  "status": 404,
  "detail": "No document with filename 'report.pdf' exists in the index.",
  "instance": "/documents/report.pdf",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string (URI) | A URI identifying the error type. Stable across versions. |
| `title` | string | Short human-readable summary. Does not change per occurrence. |
| `status` | integer | HTTP status code. Same as the HTTP response status. |
| `detail` | string | Human-readable explanation specific to this occurrence. Safe to show in UI. |
| `instance` | string | The URI of the request that caused the error. |
| `request_id` | string (UUID) | Ties this error to backend logs. Always include when reporting issues. |

### Error Type Registry

| Error type URI | HTTP status | When used |
|---|---|---|
| `https://rag-api/errors/validation-error` | 422 | Request body or params failed validation |
| `https://rag-api/errors/file-too-large` | 413 | Upload exceeds UPLOAD_SIZE_LIMIT_MB |
| `https://rag-api/errors/unsupported-file-type` | 415 | File extension or MIME type not supported |
| `https://rag-api/errors/duplicate-content` | 409 | Uploaded file content matches an existing document |
| `https://rag-api/errors/password-protected` | 422 | PDF is password protected |
| `https://rag-api/errors/scanned-pdf` | 422 | PDF has no extractable text layer |
| `https://rag-api/errors/empty-file` | 422 | Uploaded file is 0 bytes |
| `https://rag-api/errors/document-not-found` | 404 | Requested filename not in index |
| `https://rag-api/errors/max-documents-reached` | 409 | Index is at MAX_DOCUMENTS capacity |
| `https://rag-api/errors/filter-not-found` | 404 | filter_filenames contains unknown filename(s) |
| `https://rag-api/errors/safety-guardrail` | 400 | Query flagged by input safety check |
| `https://rag-api/errors/query-too-short` | 400 | Query under minimum length |
| `https://rag-api/errors/query-too-long` | 400 | Query over maximum length |
| `https://rag-api/errors/llm-unavailable` | 503 | Anthropic API unreachable or rate limited |
| `https://rag-api/errors/llm-timeout` | 503 | LLM call exceeded LLM_TIMEOUT_SECONDS |
| `https://rag-api/errors/internal-error` | 500 | Unexpected server error |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Endpoints

---

### GET /health

Returns system health and current operational stats. Used by Docker Compose health checks and the Streamlit sidebar status indicator.

**Request:** No body, no parameters.

**Response: 200 OK**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "stats": {
    "documents_indexed": 5,
    "total_chunks": 342,
    "max_documents": 20,
    "documents_remaining": 15,
    "similarity_threshold": 0.3
  }
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` or `"degraded"`. Degraded if ChromaDB is unreachable. |
| `version` | string | Application version from environment. |
| `stats.documents_indexed` | integer | Unique documents currently in the index. |
| `stats.total_chunks` | integer | Total chunks across all documents. |
| `stats.max_documents` | integer | Value of MAX_DOCUMENTS env var. |
| `stats.documents_remaining` | integer | max_documents - documents_indexed. |
| `stats.similarity_threshold` | float | Active SIMILARITY_THRESHOLD value. |

**Error responses:** None expected. Returns 200 even if stats are partially unavailable, degrade gracefully.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### POST /documents/upload

Upload one or more documents for ingestion and indexing.
Accepts multipart form data with multiple files in a single request.

**Request: multipart/form-data**

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | file[] | Yes | One or more files. Supported: PDF, TXT, MD, CSV, .xlsx, .xls, JSON, YAML, DOCX. |

**Example (curl):**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@report.pdf" \
  -F "files=@data.csv" \
  -F "files=@policy.txt"
```

**Response: 207 Multi-Status**

207 is used because each file is processed independently, some may succeed while others fail.

```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "summary": {
    "total": 3,
    "succeeded": 2,
    "failed": 1
  },
  "results": [
    {
      "filename": "report.pdf",
      "status": "success",
      "doc_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "action": "indexed",
      "chunks_indexed": 87,
      "total_chars": 43210,
      "warnings": []
    },
    {
      "filename": "data.csv",
      "status": "success",
      "doc_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
      "action": "replaced",
      "chunks_indexed": 34,
      "total_chars": 8920,
      "warnings": [
        "No column headers detected. Generic column names (Col_1, Col_2...) were assigned."
      ]
    },
    {
      "filename": "scan.pdf",
      "status": "failed",
      "doc_id": null,
      "action": null,
      "chunks_indexed": 0,
      "total_chars": 0,
      "error": {
        "type": "https://rag-api/errors/scanned-pdf",
        "title": "Scanned PDF detected",
        "status": 422,
        "detail": "This PDF appears to be scanned or image-based. Text could not be extracted. Please use a PDF with a text layer or convert it using OCR first.",
        "instance": "/documents/upload",
        "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
      }
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `summary.total` | integer | Total files submitted in this request. |
| `summary.succeeded` | integer | Files successfully indexed. |
| `summary.failed` | integer | Files that failed processing. |
| `results[].status` | string | `"success"` or `"failed"` |
| `results[].action` | string | `"indexed"` (new) or `"replaced"` (same filename, new content) or null if failed |
| `results[].chunks_indexed` | integer | Number of chunks added to the vector store. |
| `results[].warnings` | string[] | Non-fatal issues (e.g. missing headers). Empty array if none. |
| `results[].error` | object | RFC 7807 error object. Present only when status is `"failed"`. |

**Possible per-file failure types:**
- `file-too-large`, exceeds UPLOAD_SIZE_LIMIT_MB
- `unsupported-file-type`, extension or MIME type not supported
- `duplicate-content`, content hash matches existing document under different filename
- `password-protected`, PDF requires a password
- `scanned-pdf`, PDF has no extractable text
- `empty-file`, file is 0 bytes
- `max-documents-reached`, index is full (only on `"indexed"` action, not `"replaced"`)

**Top-level error responses (whole request fails before processing):**

| Status | Type | When |
|---|---|---|
| 400 | `validation-error` | No files included in the request |
| 500 | `internal-error` | Unexpected server-side failure |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### GET /documents

Returns a list of all documents currently indexed in the vector store.
Used by the Streamlit UI to populate the document filter multiselect.

**Request:** No body, no parameters.

**Response: 200 OK**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "count": 2,
  "documents": [
    {
      "doc_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "filename": "report.pdf",
      "chunks": 87,
      "indexed_at": "2024-06-25T14:30:00.000Z",
      "size_bytes": 524288,
      "content_hash": "d41d8cd98f00b204e9800998ecf8427e"
    },
    {
      "doc_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
      "filename": "data.csv",
      "chunks": 34,
      "indexed_at": "2024-06-25T14:31:00.000Z",
      "size_bytes": 9134,
      "content_hash": "7215ee9c7d9dc229d2921a40e899ec5f"
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `count` | integer | Total number of indexed documents. |
| `documents[].doc_id` | string (UUID) | Internal identifier. Used for deletion. |
| `documents[].filename` | string | Original uploaded filename. Used for filtering in /query. |
| `documents[].chunks` | integer | Number of chunks in the index for this document. |
| `documents[].indexed_at` | string (ISO 8601) | When this document was last indexed or replaced. |
| `documents[].size_bytes` | integer | Original file size in bytes. |
| `documents[].content_hash` | string | SHA-256 hash of file content (first 32 hex chars). Used internally for duplicate detection. |

**Error responses:** None expected. Returns empty list if store is empty.

```json
{
  "request_id": "...",
  "timestamp": "...",
  "count": 0,
  "documents": []
}
```

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### DELETE /documents/{filename}

Remove a document and all its chunks from the vector store.
Uses filename (not doc_id) because that is what users know.

**Path parameter:**

| Parameter | Type | Description |
|---|---|---|
| `filename` | string | Exact filename as returned by GET /documents. URL-encoded if it contains spaces. |

**Example:**
```bash
curl -X DELETE "http://localhost:8000/documents/report.pdf"
curl -X DELETE "http://localhost:8000/documents/my%20report.pdf"  # URL-encoded
```

**Response: 200 OK**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "filename": "report.pdf",
  "doc_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "chunks_removed": 87,
  "status": "deleted"
}
```

**Error responses:**

| Status | Type | When |
|---|---|---|
| 404 | `document-not-found` | Filename does not exist in the index |
| 500 | `internal-error` | ChromaDB deletion failed |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### POST /query

Submit a natural language question. Runs the full 7-step agent pipeline synchronously
(3 LLM agents + 4 non-LLM steps in custom mode) and returns the complete JSON when the
pipeline finishes (~2–4s typical on the Haiku default, D-9). For a streamed answer that
shows the first words in ~2–4s instead of waiting for the whole response, use
`POST /query/stream` (custom mode, D-10).

> **Future version note:** As query complexity and document corpus grow, response times may exceed
> acceptable thresholds for synchronous calls. A future version will introduce an asynchronous query
> pattern: `POST /query` returns a `request_id` immediately, and the client polls
> `GET /query/{request_id}` for the result. The result store will use
> **AWS ElastiCache (Redis)** with a TTL of 1 hour. The response shape will remain identical,
> only the delivery mechanism changes.

**Request body:**
```json
{
  "question": "What is the patient discharge policy for ICU transfers?",
  "filter_filenames": ["policy.pdf", "icu-guidelines.txt"],
  "include_chunks": true,
  "include_trace": true,
  "top_k_override": null,
  "agent_mode": "custom"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | Yes | n/a | Natural language question. Min 3 chars, max 2000 chars. |
| `filter_filenames` | string[] | No | `[]` | Filenames to restrict search to. Empty array = search all documents. |
| `include_chunks` | boolean | No | `false` | Include retrieved chunks in response. |
| `include_trace` | boolean | No | `false` | Include agent-by-agent trace in response. |
| `top_k_override` | integer or null | No | `null` | Override TOP_K_RETRIEVAL for this query only. Must be 1–20. Null uses env var default. |
| `agent_mode` | string or null | No | `null` | `"custom"`, `"llama_index"`, or `"compare"`. Overrides the `AGENT_MODE` env var for this query only. Null uses the env var default. See "Compare mode response" below for the `compare` response shape. |

**Response: 200 OK (pipeline completed)**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "processing_time_ms": 5240,
  "success": true,
  "short_circuit": false,
  "short_circuit_reason": null,
  "question": "What is the patient discharge policy for ICU transfers?",
  "answer": "According to the ICU guidelines document, patients may be transferred from ICU when they meet the following criteria...",
  "confidence": "high",
  "confidence_reason": "Multiple source chunks consistently support the answer with high similarity scores.",
  "sources_used": [
    "icu-guidelines.txt",
    "policy.pdf"
  ],
  "tokens": 2847,
  "cost_usd": 0.0041,
  "cache_read_tokens": 1536,
  "validation": {
    "is_valid": true,
    "issues": [],
    "hallucination_risk": "low",
    "suggested_action": "none"
  },
  "chunks": [
    {
      "filename": "icu-guidelines.txt",
      "chunk_index": 12,
      "similarity_score": 0.87,
      "rerank_score": 0.94,
      "text": "Patients may be transferred from ICU when hemodynamic stability is maintained for a minimum of 4 hours..."
    },
    {
      "filename": "policy.pdf",
      "chunk_index": 7,
      "similarity_score": 0.81,
      "rerank_score": 0.88,
      "text": "ICU transfer criteria include: attending physician sign-off, stable vitals for 4 hours, and bed availability..."
    }
  ],
  "trace": [
    {
      "agent": "SafetyGuard",
      "status": "passed",
      "details": "Input passed all safety checks.",
      "duration_ms": 2
    },
    {
      "agent": "PlannerAgent",
      "status": "completed",
      "details": {
        "intent": "Find ICU patient discharge and transfer policy",
        "retrieval_strategy": "targeted",
        "top_k": 10,
        "rewritten_query": "ICU transfer criteria patient discharge policy hemodynamic stability"
      },
      "duration_ms": 890
    },
    {
      "agent": "Retriever",
      "status": "completed",
      "details": {
        "chunks_retrieved": 10,
        "top_similarity_score": 0.87,
        "threshold_passed": true
      },
      "duration_ms": 120
    },
    {
      "agent": "SimilarityThreshold",
      "status": "passed",
      "details": {
        "top_score": 0.87,
        "threshold": 0.3,
        "result": "continue"
      },
      "duration_ms": 1
    },
    {
      "agent": "ReRanker",
      "status": "completed",
      "details": {
        "model": "ms-marco-MiniLM-L-6-v2",
        "chunks_in": 10,
        "chunks_out": 5
      },
      "duration_ms": 1340
    },
    {
      "agent": "ReasonerAgent",
      "status": "completed",
      "details": {
        "confidence": "high",
        "sources_used": ["icu-guidelines.txt", "policy.pdf"]
      },
      "duration_ms": 2100
    },
    {
      "agent": "ValidatorAgent",
      "status": "completed",
      "details": {
        "is_valid": true,
        "hallucination_risk": "low"
      },
      "duration_ms": 787
    }
  ]
}
```

**Response fields, always present:**

| Field | Type | Description |
|---|---|---|
| `request_id` | string (UUID) | Unique ID for this request. Ties to backend logs. |
| `processing_time_ms` | integer | Total server-side time from request receipt to response. |
| `success` | boolean | `true` if pipeline completed and produced an answer. `false` if short-circuited or errored. |
| `short_circuit` | boolean | `true` if pipeline was stopped early (empty store or below threshold). |
| `short_circuit_reason` | string or null | `"no_documents_indexed"`, `"similarity_threshold_not_met"`, or null. |
| `question` | string | The original question as submitted. |
| `answer` | string | The generated answer. Empty string if short_circuit is true. |
| `confidence` | string | `"high"`, `"medium"`, `"low"`, or `"n/a"` (if short-circuited). |
| `confidence_reason` | string | One sentence explaining the confidence level. |
| `sources_used` | string[] | Filenames the Reasoner drew from. Empty if short-circuited. |
| `tokens` | integer | Total Anthropic tokens consumed by this query (input + output + cache). `0` if short-circuited before any LLM call. (D-14) |
| `cost_usd` | number | Estimated USD cost, model-aware pricing with prompt-cache discounts (reads ~0.1×, writes ~1.25×). (D-14) |
| `cache_read_tokens` | integer | Tokens served from the prompt cache (D-12); counted within `tokens`, surfaced so cache savings are visible. |
| `validation` | object | Validator agent output. See below. |

**Validation object, always present:**

| Field | Type | Description |
|---|---|---|
| `is_valid` | boolean | Whether the answer passed validation. |
| `issues` | string[] | List of issues found. Empty if none. |
| `hallucination_risk` | string | `"low"`, `"medium"`, `"high"`, `"unknown"`, or `"n/a"`. `"n/a"` is used only on short-circuit responses. |
| `suggested_action` | string | `"none"`, `"review"`, or `"regenerate"`. |

**Chunks array, present only when `include_chunks: true`:**

| Field | Type | Description |
|---|---|---|
| `filename` | string | Source document filename. |
| `chunk_index` | integer | Position of this chunk within its document. |
| `similarity_score` | float | Cosine similarity score from initial retrieval (0.0–1.0). |
| `rerank_score` | float | Score assigned by the ReRanker (0.0–1.0). |
| `text` | string | The chunk text content. |

**Trace array, present only when `include_trace: true`:**

| Field | Type | Description |
|---|---|---|
| `agent` | string | Agent name. |
| `status` | string | `"passed"`, `"completed"`, `"skipped"`, `"failed"`. |
| `details` | object or string | Agent-specific output. Structure varies per agent. |
| `duration_ms` | integer | Time taken by this agent step in milliseconds. |

---

**Response: 200 OK (`agent_mode: "compare"`)**

When `agent_mode` is `"compare"`, both pipelines run on the same question and the response uses a
side-by-side envelope instead of the single-pipeline shape above. Each side carries its own answer,
sources, LLM-call count, duration, chunks, and trace.

```json
{
  "mode": "compare",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "question": "What is the patient discharge policy for ICU transfers?",
  "processing_time_ms": 6300,
  "short_circuit": false,
  "custom": {
    "answer": "According to the ICU guidelines document...",
    "confidence": "high",
    "sources_used": ["icu-guidelines.txt", "policy.pdf"],
    "llm_calls": 3,
    "duration_ms": 2400,
    "tokens": 2847,
    "cost_usd": 0.0041,
    "short_circuit": false,
    "note": "",
    "chunks": [],
    "trace": []
  },
  "llama_index": {
    "answer": "Patients may be transferred from ICU when...",
    "confidence": "high",
    "sources_used": ["icu-guidelines.txt"],
    "llm_calls": 4,
    "duration_ms": 6100,
    "tokens": 21686,
    "cost_usd": 0.0228,
    "short_circuit": false,
    "note": "",
    "chunks": [],
    "trace": []
  },
  "validation": {
    "is_valid": true,
    "issues": [],
    "hallucination_risk": "low",
    "suggested_action": "none"
  },
  "validated_pipeline": "Custom pipeline"
}
```

| Field | Type | Description |
|---|---|---|
| `mode` | string | Always `"compare"` for this response shape. |
| `processing_time_ms` | integer | Total wall-clock time for **both** pipelines combined. |
| `short_circuit` | boolean | `true` if the **custom** pipeline short-circuited (empty store or below threshold). |
| `custom` | object | Result from the custom 7-step pipeline. Fields: `answer`, `confidence`, `sources_used`, `llm_calls`, `duration_ms`, `tokens`, `cost_usd`, `chunks`, `trace`. When it short-circuits it also carries `short_circuit: true` and `note` (the human-readable reason, e.g. "…scored 0.39, below 0.30…") with an empty `answer`. |
| `llama_index` | object | Result from the LlamaIndex ReAct loop. Same fields. `confidence` is self-rated (`"high"`/`"medium"`/`"low"`), the loop emits its own `CONFIDENCE` line, there is **no** separate Validator on this side (D-5). Runs and returns normally even when the custom pipeline short-circuits. Trace steps are action-labelled (`LlamaReAct · Search` / `· Synthesize`); search steps carry `new_chunks` + `overlap`, and the final step a `reason` (`diminishing_returns` or `max_steps`), see D-15. |
| `validation` | object | Validator output. Runs on the **custom** answer normally; if the custom pipeline short-circuited, runs on the **`llama_index`** answer instead. |
| `validated_pipeline` | string or null | Which side's answer `validation` ran on, `"Custom pipeline"`, `"LlamaIndex ReAct"`, or `null` if neither produced an answer. Lets the UI label the validation line accurately (D-5). |

`chunks` and `trace` inside each side are populated only when `include_chunks` / `include_trace` are true,
following the same rules as the single-pipeline response.

---

**Response: 200 OK (short-circuited, empty store)**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "processing_time_ms": 8,
  "success": false,
  "short_circuit": true,
  "short_circuit_reason": "no_documents_indexed",
  "question": "What is the discharge policy?",
  "answer": "",
  "confidence": "n/a",
  "confidence_reason": "",
  "sources_used": [],
  "validation": {
    "is_valid": false,
    "issues": ["No documents are indexed. Cannot retrieve context."],
    "hallucination_risk": "n/a",
    "suggested_action": "none"
  },
  "chunks": [],
  "trace": [
    {
      "agent": "SafetyGuard",
      "status": "passed",
      "details": "Input passed all safety checks.",
      "duration_ms": 2
    },
    {
      "agent": "Retriever",
      "status": "skipped",
      "details": "no_documents_indexed",
      "duration_ms": 3
    }
  ]
}
```

**Response: 200 OK (short-circuited, below similarity threshold)**
```json
{
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2024-06-25T14:32:10.123Z",
  "processing_time_ms": 310,
  "success": false,
  "short_circuit": true,
  "short_circuit_reason": "similarity_threshold_not_met",
  "question": "What is the weather like in Paris?",
  "answer": "",
  "confidence": "n/a",
  "confidence_reason": "",
  "sources_used": [],
  "validation": {
    "is_valid": false,
    "issues": ["Top retrieved chunk scored 0.18, below the threshold of 0.30. No relevant content found."],
    "hallucination_risk": "n/a",
    "suggested_action": "none"
  },
  "chunks": [],
  "trace": [
    {
      "agent": "Retriever",
      "status": "completed",
      "details": { "chunks_retrieved": 10, "top_similarity_score": 0.18, "threshold_passed": false },
      "duration_ms": 118
    },
    {
      "agent": "SimilarityThreshold",
      "status": "failed",
      "details": { "top_score": 0.18, "threshold": 0.3, "result": "short_circuit" },
      "duration_ms": 1
    }
  ]
}
```

> Note: Short-circuit responses return HTTP 200, not 4xx. The pipeline ran and made a
> deliberate decision. `success: false` in the body communicates the outcome.
> HTTP 4xx is reserved for bad requests (invalid input, safety violations).

**Error responses (HTTP 4xx/5xx):**

| Status | Type | When |
|---|---|---|
| 400 | `safety-guardrail` | Prompt injection or unsafe pattern detected |
| 400 | `query-too-short` | Question under 3 characters |
| 400 | `query-too-long` | Question over 2000 characters |
| 404 | `filter-not-found` | filter_filenames contains unknown filename(s) |
| 422 | `validation-error` | Malformed request body |
| 503 | `llm-unavailable` | Anthropic API down or rate limited |
| 503 | `llm-timeout` | LLM call exceeded LLM_TIMEOUT_SECONDS |
| 500 | `internal-error` | Unexpected server error |

**503 error response example (LLM failure mid-pipeline):**
```json
{
  "type": "https://rag-api/errors/llm-unavailable",
  "title": "LLM service unavailable",
  "status": 503,
  "detail": "The Anthropic API returned an error during the Reasoner agent step. Please try again in 30 seconds.",
  "instance": "/query",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "failed_at_agent": "ReasonerAgent",
  "retry_after": 30
}
```

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### POST /query/stream

Same request body as `POST /query`, but the answer is **streamed** so the first words reach the
UI in ~2–4 s instead of after the whole answer is generated. Implemented (D-10).

- **Scope:** `custom` mode streams the answer token-by-token. `llama_index` and `compare` are not
  token-streamed, for those this endpoint runs the pipeline batch and emits the whole answer in
  one piece (the frontend uses plain `POST /query` for them).
- **Response:** `200 OK`, `Content-Type: text/plain; charset=utf-8`. The body is the **answer text**,
  followed by a single **`0x1E`** (ASCII record separator) byte, followed by a **JSON metadata frame**
  with the same fields as the `/query` response (`answer`, `confidence`, `confidence_reason`,
  `sources_used`, `validation`, `trace`, `chunks`, `processing_time_ms`, `mode`, `short_circuit`,
  `request_id`, `timestamp`). Clients split on the `0x1E` byte: everything before it is the answer,
  everything after is the JSON.
- The Reasoner's `ANSWER:`/`CONFIDENCE:`/`REASON:`/`SOURCES:` scaffolding is stripped during
  streaming, only the answer text is emitted; the structured fields arrive in the metadata frame.
- **Errors:** safety and filter checks run **before** streaming begins, so `safety-violation`,
  `query-too-short/long`, and `filter-not-found` still return normal RFC-7807 4xx JSON (not a stream).
- **Short-circuit** (empty store / below threshold): no answer text is streamed; the metadata frame
  carries `short_circuit: true` and the reason, exactly like `/query`.

> **Future version note:** A future version may switch the wire format to Server-Sent Events (SSE) and add
> llama-mode streaming; the batch `POST /query` remains available either way.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

### GET /

Root endpoint. Returns application identity. Used to confirm the API is reachable.

**Response: 200 OK**
```json
{
  "app": "Agentic RAG Knowledge System",
  "version": "1.0.0",
  "phase": "1",
  "docs": "/docs",
  "health": "/health",
  "status": "running"
}
```

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Endpoint Summary

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/` | API identity and status | None |
| `GET` | `/health` | System health and stats | None |
| `POST` | `/documents/upload` | Upload and index documents | None |
| `GET` | `/documents` | List all indexed documents | None |
| `DELETE` | `/documents/{filename}` | Remove a document | None |
| `POST` | `/query` | Submit a question, run pipeline (batch JSON) | None |
| `POST` | `/query/stream` | Submit a question, stream the answer (custom mode) + metadata frame | None |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## HTTP Status Code Usage

| Code | Meaning in this API |
|---|---|
| 200 | Success. Also used for short-circuit responses (pipeline ran, made a decision). |
| 207 | Multi-status. Used only on /documents/upload when processing multiple files. |
| 400 | Bad request. Input failed validation or safety guardrail. |
| 404 | Resource not found. Document does not exist. |
| 409 | Conflict. Duplicate content or max documents reached. |
| 413 | Payload too large. File exceeds size limit. |
| 415 | Unsupported media type. File format not accepted. |
| 422 | Unprocessable entity. File parsed but content is unusable (scanned PDF, empty file). |
| 500 | Internal server error. Unexpected failure. |
| 503 | Service unavailable. Anthropic API unreachable or timed out. |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Streamlit UI: API Usage Map

How the Streamlit frontend uses each endpoint:

| UI action | API call |
|---|---|
| App startup, sidebar status | `GET /health` |
| App startup, populate doc filter multiselect | `GET /documents` |
| User uploads files | `POST /documents/upload` (all files in one request) |
| User deletes a document | `DELETE /documents/{filename}` |
| User submits a question (`custom` mode) | `POST /query/stream`, answer rendered live via `st.write_stream`, then metadata frame for confidence/sources/trace |
| User submits a question (`llama_index` / `compare`) | `POST /query` with `include_chunks=true`, `include_trace=true` |
| Any API error | Streamlit catches non-200, displays human-readable message. Raw error shown in expandable trace section only. |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Future Version Changes (Forward Reference)

The following changes are planned for a future version. The current API contract is designed to
accommodate them without breaking changes where possible.

| Change | Impact on contract |
|---|---|
| Async query via ElastiCache/Redis | `POST /query` returns 202 Accepted + `request_id`. New endpoint `GET /query/{request_id}` added. Response shape identical. |
| Cursor-based pagination on GET /documents | Adds `cursor`, `limit`, `next_cursor` fields. Backward compatible with default limit = all. |
| API key authentication | Adds `Authorization: Bearer <key>` header requirement. New 401 error type. |
| Multi-turn conversation | Adds optional `session_id` and `conversation_history` fields to POST /query body. |
| Streaming responses | **Done (D-10)**, `POST /query/stream` streams the `custom`-mode answer (text + `0x1E` + JSON metadata). A future version may move it to Server-Sent Events and add `llama_index` streaming. Existing `/query` remains. |

<!-- refinement -->
