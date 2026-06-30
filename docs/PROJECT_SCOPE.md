# Agentic RAG Knowledge System: Project Scope

## Table of Contents

- [Project Goal](#project-goal)
- [Problem Statement](#problem-statement)
- [Who Uses It](#who-uses-it)
- [What Makes It Agentic](#what-makes-it-agentic)
- [What the System Does](#what-the-system-does)
- [Non-Functional Requirements](#non-functional-requirements)
- [Engineering Practices](#engineering-practices)
- [Tech Stack](#tech-stack)
- [Three Models in the System](#three-models-in-the-system)
- [Supported Document Formats](#supported-document-formats)
- [Network Design](#network-design)
- [Task Mapping](#task-mapping)
- [Reliability and Safety Controls](#reliability-and-safety-controls)
- [Key Design Principle](#key-design-principle)
- [Prompt Caching Strategy (D-12)](#prompt-caching-strategy-d-12)
- [Environment Variables](#environment-variables)
- [Project Folder Structure](#project-folder-structure)
- [Docker Compose Design](#docker-compose-design)
- [Limitations](#limitations)
- [Extensibility Hooks (built in now)](#extensibility-hooks-built-in-now)
- [Scope for Future Versions](#scope-for-future-versions)

---

## Project Goal

Build an AI-powered document intelligence system that allows users to upload enterprise documents in multiple formats and ask natural language questions. The system retrieves relevant content using semantic search and generates grounded, cited answers using a multi-agent reasoning pipeline.

This is delivered as both a **web interface** (Streamlit) for human users and a **REST API** (FastAPI) for programmatic access. The entire system runs with a single command via Docker Compose on any machine.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Problem Statement

Knowledge locked in documents (PDFs, spreadsheets, text reports) is hard to query. A user must manually read and search to find answers. This system makes any document collection **conversational**: upload your documents, ask questions in plain English, get grounded answers with source citations.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Who Uses It

Any user, technical or non-technical, who needs to extract insights from documents without manually reading them. Designed as a general-purpose system with no domain restrictions.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## What Makes It Agentic

Rather than a single LLM call, the system uses a **7-step agent pipeline** with **3 true LLM agents** and **4 non-LLM pipeline steps** that collaborate in sequence:

**3 LLM Agents (make Anthropic API calls):**

| Agent | Role |
|---|---|
| PlannerAgent | Understands query intent, rewrites query for optimal retrieval |
| ReasonerAgent | Generates a grounded, cited answer from reranked context |
| ValidatorAgent | Checks the answer for hallucination risk before returning it |

**4 Non-LLM Pipeline Steps (rule-based, encoder, or DB calls, no LLM):**

| Step | Role |
|---|---|
| SafetyGuard | Input validation, prompt injection detection, length checks. Rule-based, no LLM call. |
| Retriever | Fetches top-k chunks from ChromaDB using cosine similarity. DB call, no LLM. |
| SimilarityThreshold | Checks if top chunk clears minimum score. Rule-based, no LLM call. |
| ReRanker | Reorders retrieved chunks by answer relevance using cross-encoder model (ms-marco-MiniLM-L-6-v2). Encoder, no LLM call. |

Each step has a specific responsibility. The output of each step feeds the next. A step failing or short-circuiting stops the pipeline immediately, no wasted LLM calls downstream.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## What the System Does

**Document ingestion:**
- Upload documents via the web UI or API
- Supported formats: PDF, TXT, MD (Markdown), CSV, Excel (.xlsx, .xls), JSON, YAML, DOCX
- Ingestion flow: **save upload bytes to a temp path → validate the raw bytes (MIME check, size, SHA-256 dedup, scanned-PDF and password-PDF detection) → only if validation passes, direct per-format parsers (pypdf, python-docx, pandas, pyyaml, json, chardet) parse the file → chunk → embed → store**
- The per-format parsers are a **thin parsing layer only**, all validation happens on raw bytes *before* any parser touches the file. A file that failed validation is never parsed (D-2).
- Documents are parsed, chunked into overlapping segments, and indexed in a vector database

**Question answering, three modes via AGENT_MODE flag:**
- `custom` (default), 7-step sequential pipeline, 3 LLM calls; the answer is **streamed** token-by-token (D-10), so the first words appear in ~2–4s on the Haiku default (D-9)
- `llama_index`, ReAct search→answer loop (built using python, calls the Anthropic API; see D-7), dynamic 1–4 LLM calls (batch); stops early on diminishing returns rather than burning the search budget on redundant retrievals (D-15)
- `compare`, runs both modes on the same query, returns side-by-side results for evaluation (batch)
- Optionally filters to specific uploaded documents
- Returns: grounded answer, confidence level, source citations, retrieved chunks with similarity scores, full pipeline trace

**Observability:**
- Every request returns a full agent trace (which agent ran, what it decided, status)
- Structured JSON logging with trace IDs on every request, plus a per-query `query_completed`
  log line carrying the same per-step timings as the trace, and a per-LLM-call `llm_usage` line
  with token counts incl. prompt-cache hits (`cache_read_input_tokens`, D-12). Raw queries are
  never logged, only a hash.
- `/health` endpoint with system stats (chunks indexed, status); the UI polls it at most once
  per 30s, matching the Docker healthcheck interval

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Non-Functional Requirements

| Requirement | How it is met |
|---|---|
| Scalable | Stateless FastAPI backend; ChromaDB abstracted behind an interface, swap to managed DB without changing application code |
| Traceable | Every API response includes a full agent trace; every log line carries a trace ID |
| Observable | Structured JSON logs to stdout, compatible with any log aggregator (Loki, CloudWatch) in future |
| Deployable | Docker Compose, one command, runs on any PC, any OS |
| Portable | All config via environment variables (.env); no hardcoded URLs, paths, or secrets |
| Safe | Four input safety layers (cheapest first); defence in depth output controls (ONLY rule + threshold + validator); blanket retry with half timeout; log sanitisation for PII |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Engineering Practices

The project is built and verified to production standards (full detail in the README's "Industry-Standard Best Practices" section).

- **CI/CD (GitHub Actions).** Linting (`ruff`) and the full test suite run on every push and pull request; coverage is reported on each run. Docker images are built only when an image-affecting file changes on `main`, with build-layer caching, and a scheduled job keeps the dependency and model caches warm.
- **Test-driven development.** 105 tests across five suites (vector store, chunking, document loading, the agent pipeline, the API), written alongside each service and shipped in the same commit as the code.
- **Code quality and coverage.** `ruff` enforces a consistent style; `pytest-cov` measures coverage of the `app/` package on every CI run.
- **Security.** The Anthropic API key is read only from the host environment, never written to `.env` or committed; `.env` is gitignored and only `.env.example` is tracked, and git history is verified free of secrets before release. Configuration is environment-only (no hardcoded secrets, URLs, or paths). Input is validated on raw bytes, a SafetyGuard screens for prompt injection, raw queries are never logged (only a SHA-256 hash), and answers are grounded with an independent Validator check.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Streamlit (Python) | Pure Python, no JS needed, built-in file upload and chat UI |
| Backend API | FastAPI (Python) | Fast, async, auto-generates /docs Swagger UI |
| Vector database | ChromaDB (embedded) | No separate service needed, persists to disk, easy to swap |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | 384 dimensions, 256 token limit (~192 words), free, runs locally on CPU, no API key required |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) | Reads query + chunk jointly, precise answer relevance scoring, runs locally on CPU, no API key required |
| LLM | claude-haiku-4-5 via Anthropic API (default, for speed; configurable via `ANTHROPIC_MODEL`, D-9) | Used by the four python agents that call the Anthropic API: PlannerAgent, ReasonerAgent, ValidatorAgent (3 LLM calls per query in custom mode) and llama_agent (1–4 calls in llama_index mode); answer is streamed in custom mode (D-10) |
| Document loading | Direct per-format parsers (D-2) | pypdf, python-docx, pandas (+openpyxl/xlrd), pyyaml, json, chardet for PDF, TXT, MD, CSV, Excel, JSON, YAML, DOCX |
| Agent loop (`llama_index` mode) | `llama_agent` (python, Anthropic API) | Alternative pipeline mode via AGENT_MODE=llama_index, a ReAct search→answer loop with no framework dependency (D-7) |
| PDF parsing | pypdf (direct) | Actively maintained, handles text-layer PDFs |
| Caching | Anthropic prompt caching (configurable: block / prompt / off, D-12) | Reduces token cost and latency on repeated agent system prompts; cache hits logged via `llm_usage` |
| Deployment | Docker Compose + Python venv | Docker for production, venv for local development |
| Logging | Structured JSON to stdout | Compatible with any log shipper in future phases |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Three Models in the System

The system uses three models (two local encoders and one cloud LLM) and no RAG framework:

| Component | Type | Used by | Purpose | API key? |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | Bi-encoder | ChromaDB | Embed chunks + queries → 384-dim vectors for semantic search | No, local |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder | ReRanker | Score query + chunk jointly for precise answer relevance | No, local |
| `claude-haiku-4-5` (default, configurable, D-9) | LLM (Anthropic) | Planner, Reasoner, Validator, llama_agent | Query rewriting, answer generation, validation, ReAct loop | Yes |

`all-MiniLM-L6-v2`, encodes text independently into vectors. Fast, pre-computed at indexing time. Broad semantic similarity. Cannot understand query-chunk relationship jointly.

`ms-marco-MiniLM-L-6-v2`, reads query and chunk together in one pass. Slower but more precise, understands the specific relationship between question and chunk. Used by ReRanker in all modes.

`claude-haiku-4-5` (default), full LLM for reasoning and generation. 3 calls per query in custom mode. Variable 1–4 calls in llama_index mode (ReAct loop, D-7; stops early on diminishing returns, D-15). Requires Anthropic API key. We switched the default from `claude-sonnet-4-6` to Haiku to improve response time (~3× faster generation, D-9); the model stays configurable via `ANTHROPIC_MODEL` (set `claude-sonnet-4-6` for maximum quality). In custom mode the answer is streamed to the UI (D-10).

No RAG framework is used: both modes are built from scratch in python (direct per-format parsers for ingestion, D-2; a hand-written ReAct loop for `llama_index` mode, D-7), so every component is explicit and observable rather than abstracted away.

**Both sentence-transformer models (~80MB each) are pre-downloaded in the Docker image.**

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Supported Document Formats

All formats handled by direct per-format parsers (D-2):

| Format | Extension | Notes |
|---|---|---|
| PDF | .pdf | Text-layer only. Scanned PDFs rejected with clear error. |
| Text | .txt | UTF-8 and common encodings auto-detected. |
| Markdown | .md | Parsed as plain text. Formatting stripped. |
| CSV | .csv | Rows converted to text strings. Headers auto-generated if missing. |
| Excel | .xlsx, .xls | All sheets indexed. Charts and images ignored. |
| JSON | .json | Keys and values converted to readable text. Nested objects flattened. |
| YAML | .yaml, .yml | Parsed to key-value text. Configuration files fully supported. |
| Word | .docx | Text extracted from paragraphs. Tables converted to text. |

---

The system is organized in four layers:

```
User (browser)
      │
      ▼
Streamlit frontend          ← upload UI + question input + doc filter + chunk viewer + trace display
      │
      ▼  HTTP
FastAPI backend             ← REST API, routing, logging, input validation
      │
      ├── Ingestion path    ← parse → chunk → embed → store in ChromaDB
      │
      └── Query path        ← 7-step pipeline (see below)
                                    │                      │
                                    ▼                      ▼
                              Anthropic API           ChromaDB
                              (LLM + caching)         (vector store)

7-step query pipeline (3 LLM agents + 4 non-LLM steps):

  Step 1: SafetyGuard, input validation, prompt injection check (rule-based, no LLM)
        ↓
  Step 2: PlannerAgent, intent analysis + query rewrite (LLM call)
        ↓
  Step 3: Retriever, top-k semantic search with optional doc filter (ChromaDB, no LLM)
        ↓
  Step 4: SimilarityThreshold, if top score < SIMILARITY_THRESHOLD → short-circuit (rule-based, no LLM)
        ↓
  Step 5: ReRanker, reorder chunks by answer relevance
        │                        uses cross-encoder/ms-marco-MiniLM-L-6-v2 (no LLM call)
        ↓
  Step 6: ReasonerAgent, grounded answer from reranked context (LLM call, prompt cached)
        ↓
  Step 7: ValidatorAgent, hallucination risk scoring (LLM call)
        ↓
  Response, answer + confidence + sources + chunks + scores + trace

LLM calls per query:    3 (Planner + Reasoner + Validator)
Encoder calls per query: 1 (ReRanker, ms-marco-MiniLM-L-6-v2, not an LLM call)
Total model calls:       4 per query
```

See architecture diagram (PHASE1_ARCHITECTURE.png) for the visual version.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Network Design

```
Browser
  ├── localhost:8501  →  Streamlit container   (web UI)
  └── localhost:8000  →  FastAPI container     (API + /docs)

Docker internal network (rag_network):
  Streamlit  →  http://backend:8000  (internal service name, not localhost)

ChromaDB runs embedded inside the FastAPI container.
No separate database container required.

External:
  FastAPI container  →  https://api.anthropic.com  (HTTPS, outbound only)

Config:
  All secrets and settings via .env file (never hardcoded)
```

See network diagram (PHASE1_NETWORK.png) for the visual version.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Task Mapping

| Task | What gets built |
|---|---|
| 1. Project foundation | Repo structure, .env, docker-compose.yml, requirements.txt |
| 2. User interaction layer | Streamlit UI (upload, question, doc filter, chunk viewer, trace) + FastAPI routes |
| 3. Document ingestion | storage.py (save_upload) + raw-byte validation → document_loader.py (direct per-format parsers handle validated files only, D-2) |
| 4. Data preparation | chunk_text(), overlapping word-based chunking |
| 5. Vector knowledge store | vector_store.py, ChromaDB index and upsert |
| 6. Intelligent retrieval | retrieve(), cosine similarity search with doc metadata filter. Collection created with `hnsw:space=cosine`; `similarity = 1 - distance` applied to every result. |
| 7. RAG pipeline | reasoner_agent(), context-grounded LLM generation with prompt caching |
| 8. Agent-based reasoning | pipeline.py, 7-step orchestrator: SafetyGuard → Planner → Retriever → Threshold → ReRanker → Reasoner → Validator. 3 LLM agents + 4 non-LLM steps. |
| 9. Reliability and safety | Input guardrails, similarity threshold short-circuit, output validation, retry pattern, log sanitisation |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Reliability and Safety Controls

### Input Safety: Four Layers, Cheapest First

Order is deliberate, short-circuit evaluation, reject as early as possible:

```
Layer 1: Pydantic validation, zero business logic, rejects malformed requests instantly
                                   wrong type / missing field / length violation = HTTP 422
                                   runs before any application code

Layer 2: SafetyGuard, string scan for prompt injection patterns
                                   no DB call, no LLM call, cheap
                                   coverage gap: only catches coded patterns, attackers rephrase

Layer 3: Empty store check, one ChromaDB count() call
                                   pointless to validate filenames if store is empty

Layer 4: Filter validation, one ChromaDB list_documents() call
                                   most expensive input check, only runs if store is non-empty
                                   rejects unknown filenames with descriptive error
```

**Principle:** Order by cost. Never run an expensive check if a cheaper check would have caught the problem.

---

### Output Safety: Defence in Depth

No single control prevents all hallucinations. Three independent layers:

```
ONLY rule (prompt guardrail):
    Prevents: context ignorance, Reasoner answering from training knowledge
    How: system instruction explicitly forbids using knowledge outside retrieved context
    Cannot catch: Reasoner misreading context (follows rule but gets content wrong)

SimilarityThreshold (retrieval guardrail):
    Prevents: irrelevant context hallucination, Reasoner receiving noise and fabricating
    How: blocks pipeline before Reasoner sees bad context
    Cannot catch: subtly wrong or incomplete context that clears the threshold

ValidatorAgent (output guardrail):
    Prevents: output hallucination, claims beyond what context explicitly states
    How: independent LLM call with no memory of generation (fresh context = no self-consistency bias)
    Cannot catch: systematic bias shared by both Reasoner and Validator (same model type)
```

**Validator input contract (independence is enforced by what it receives):**

The Validator receives ONLY:
- the original question
- the final answer text (string only)
- the top retrieved chunks (context sample)

The Validator does NOT receive:
- the Reasoner's internal reasoning or chain-of-thought
- the Planner output
- any other pipeline state

A fresh context with no memory of *how* the answer was generated is what makes the
Validator an independent check rather than a rubber stamp. Passing the Reasoner's reasoning
into the Validator would reintroduce self-consistency bias and defeat the purpose.

**Defence in depth principle:** If one control fails, others backstop. No single control is sufficient. Layer multiple independent controls, each catching different failure modes.

---

### LLM Call Reliability: Blanket Retry Pattern

Every LLM call (Planner, Reasoner, Validator) follows the same retry pattern:

```python
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", 20))

def call_llm_with_retry(fn, *args):
    """
    One retry on any failure.
    First attempt:  LLM_TIMEOUT_SECONDS (default 20s)
    Retry attempt:  LLM_TIMEOUT_SECONDS // 2 (default 10s)
    No distinction between error types, blanket retry.
    Total max wait: 30s (20s + 10s)
    """
    timeouts = [LLM_TIMEOUT_SECONDS, LLM_TIMEOUT_SECONDS // 2]

    for attempt, timeout in enumerate(timeouts):
        try:
            return fn(*args, timeout=timeout)
        except Exception as e:
            if attempt < len(timeouts) - 1:
                logger.warning({
                    "event":   "llm_retry",
                    "attempt": attempt + 1,
                    "timeout": timeout,
                    "error":   str(e)[:100]
                })
            else:
                raise   # exhausted retries, structured error returned to caller
```

**Why half timeout on retry:**
If the first call timed out at 20s and the retry also waits 20s before failing, user waits 40+ seconds. Reducing to 10s caps total wait at 30s. If the service is healthy it responds in 1-2s, 10s is still generous.

**Why blanket retry (no error classification):**
The system keeps it simple. All failures retry once. A future version introduces retryable vs non-retryable error classification.

**After exhausted retries, structured 503 response:**
```json
{
  "type": "https://rag-api/errors/llm-timeout",
  "status": 503,
  "detail": "Service timed out after 1 retry. Please try again in a moment.",
  "failed_at_agent": "ReasonerAgent",
  "retry_after": 30,
  "request_id": "..."
}
```

---

### Log Sanitisation: Query Hashing

Every log line uses a hash of the user query instead of the raw query text:

```python
import hashlib

def hash_query(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:8]
    # "a3f8c21b", 8 hex chars, deterministic, no PII

logger.info({
    "trace_id":     request_id,
    "step":         "ReasonerAgent",
    "status":       "completed",
    "duration_ms":  2100,
    "query_hash":   hash_query(query),   # ← hash, never raw query
    "query_length": len(query),          # ← length for context
})
```

**Why hash and not truncate (`query[:100]`):**

`query[:100]` provides false confidence. PII appears anywhere in a query:
```
"Hi, Prashant here, tell me about cats!"
#     ^^^^^^^^
#     PII at position 4: truncation does nothing
```

A hash eliminates PII risk entirely regardless of where it appears in the query.

**Three benefits of query hashing:**

1. **Zero PII risk**, hash cannot be reversed to recover the original query
2. **Query identification**, same query from any user produces the same hash. Ties related log lines together. Identifies repeated queries across sessions.
3. **Future cache key**, if response caching is added later, `query_hash` is already the natural cache key. Infrastructure built now at zero extra cost.

**Why SHA-256 not MD5:**
MD5 is not cryptographically secure. SHA-256 is the correct habit even for non-security use cases. First 8 hex chars (`[:8]`) gives 4 billion possible values, negligible collision risk at our scale.

**Env var:** `QUERY_HASH_ALGO=sha256`, algorithm is configurable, default sha256.
| 10. Deploy and document | Docker Compose deployment + README + architecture docs |

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Key Design Principle

**Tasks 2–9 are application layer. They are fully decoupled from infrastructure.**

The same codebase runs locally via Docker Compose and on cloud infrastructure without any application code changes. Only environment variables and infrastructure definitions differ between environments. This follows 12-factor app principles and makes the system portable and maintainable.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Prompt Caching Strategy (D-12)

Anthropic prompt caching is configurable via `ANTHROPIC_CACHE_MODE` (`block` | `prompt` | `off`):

1. **`block` (default), system-prompt block caching.** The 3 LLM agents (Planner, Reasoner, Validator) each have a long system prompt that is identical across calls, marked with `cache_control` so Anthropic caches it after the first call. This is the only caching that reliably helps this single-turn app, the retrieved context + question change every query, so they are not reusable across calls (caching them would never hit). The 4 non-LLM steps make no LLM calls, so no caching applies. The Validator's *system prompt* is cached, which is fine, a static system prompt does not affect its independence (it still sees a fresh answer + context each query).

2. **`prompt`, prompt-level caching.** A top-level `cache_control` marker caches the whole prompt prefix (it is applied to the last block of the prompt). Useful when a long prefix repeats verbatim, e.g. multi-turn conversation (a future-version direction). For single-turn RAG its reuse is limited.

3. **`off`, no caching** (debugging / comparison).

Cache effectiveness is observable in the logs: every LLM call emits an `llm_usage` event with `cache_read_input_tokens` (served from cache) and `cache_creation_input_tokens` (written to cache).

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Environment Variables

Non-secret config lives in a local `.env` (gitignored; `.env.example` is the committed
template). `config.py` reads `os.environ`, and Docker Compose loads `.env` into the backend
(D-11). The **API key is read from the host shell/OS environment, never from `.env`**, keeping
the secret out of the project. Production supplies the same vars differently (task-def env +
Secrets Manager, D-11). No settings are hardcoded.

```bash
# ── LLM ────────────────────────────────────────────────────────────────────
# ANTHROPIC_API_KEY: set in your shell/OS env, NOT in .env (D-11). Required.
ANTHROPIC_MODEL=claude-haiku-4-5      # Model for all agents. Default Haiku for speed (D-9);
                                      # set claude-sonnet-4-6 for maximum answer quality.
ANTHROPIC_CACHE_MODE=block            # Prompt caching (D-12): block (default) | prompt | off.
                                      # Cache hits show in the "llm_usage" log event.

# ── Agent mode ──────────────────────────────────────────────────────────────
AGENT_MODE=custom
# custom      → 7-step sequential pipeline (default)
#               3 LLM calls: Planner + Reasoner + Validator
#               Predictable, fast, fully observable trace
# llama_index → ReAct search→answer loop built using python on the Anthropic API (D-7, no LlamaIndex)
#               Variable 1-4 LLM calls depending on query complexity
#               Better for multi-step reasoning across multiple documents
# compare     → Runs BOTH modes on the same query
#               Returns side-by-side results for pipeline evaluation
#               Use to benchmark and tune the custom pipeline

# ── Re-ranker ───────────────────────────────────────────────────────────────
# ReRanker always uses cross-encoder/ms-marco-MiniLM-L-6-v2 (encoder mode).
# No RERANKER_MODE env var: encoder is fixed. Not an LLM call.
# Model is pre-downloaded in Docker image. Runs on CPU. No API key required.

# ── Retrieval ───────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD=0.3
# Minimum cosine similarity score for the top retrieved chunk (D-13; was 0.4).
# If the best match is below this value, the pipeline short-circuits
# and returns "insufficient context" without calling any LLM.
# Range: 0.0–1.0. Tune based on your documents and query types.
# IMPORTANT: the ChromaDB collection MUST be created with hnsw:space=cosine.
# ChromaDB returns a distance, not a similarity: vector_store.py converts every
# result with similarity = 1 - distance. This threshold only makes sense in cosine space.

TOP_K_RETRIEVAL=10
# Number of chunks fetched from ChromaDB before re-ranking.
# Re-ranker then selects the best TOP_K_RERANK from these.

TOP_K_RERANK=5
# Number of chunks passed to the Reasoner after re-ranking.

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE=200                        # Words per chunk. Set to stay within all-MiniLM-L6-v2's
                                      # 256 token limit (~192 words for typical English prose).
CHUNK_OVERLAP=25                      # Overlapping words between chunks (12.5% of chunk size).
MIN_CHUNK_CHARS=20                    # Chunks with fewer non-whitespace characters are discarded.

# ── Storage ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_PATH=/data/chroma      # Where ChromaDB stores its index.
UPLOAD_SIZE_LIMIT_MB=10               # Max file size per upload.
MAX_DOCUMENTS=20                      # Max unique documents allowed in the index.
                                      # Enforced at upload time (on "indexed" action, not "replaced").

# ── API ─────────────────────────────────────────────────────────────────────
BACKEND_URL=http://backend:8000       # Used by Streamlit to reach FastAPI.
                                      # Change to localhost:8000 for local dev without Docker.
LOG_LEVEL=INFO                        # DEBUG / INFO / WARNING / ERROR

# ── Reliability ──────────────────────────────────────────────────────────────
LLM_TIMEOUT_SECONDS=20               # First attempt timeout per LLM call.
                                      # Retry timeout = LLM_TIMEOUT_SECONDS // 2 (10s).
                                      # Total max wait per LLM call = 30s (20s + 10s).
QUERY_HASH_ALGO=sha256               # Algorithm for query hashing in logs.
                                      # Hash replaces raw query, zero PII risk.
                                      # Same query always produces same hash, cache key ready.
                                      # First 8 hex chars used: 4 billion possible values.

# ── Reserved for a future version (present now, unused) ────────────────────
GROUNDING_THRESHOLD=0.6              # RESERVED FOR A FUTURE VERSION, not read by current code.
                                      # Future-version deterministic grounding check: answers scoring
                                      # below this trigger a conditional ValidatorAgent LLM call.
```

**ReRanker always uses the cross-encoder model, no feature flag, no LLM call for ranking.**

The switch requires only a `.env` change and container restart, no code changes.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Project Folder Structure

A conventional project structure, with our file names preserved:

```
genai-doc-assistant/
├── app/
│   ├── api/
│   │   └── main.py              ← FastAPI app and all routes
│   ├── agents/
│   │   ├── pipeline.py          ← 7-step orchestrator + AGENT_MODE routing
│   │   ├── safety.py            ← SafetyGuard (rule-based, no LLM)
│   │   ├── planner.py           ← PlannerAgent (LLM call)
│   │   ├── ranker.py            ← ReRanker (cross-encoder, no LLM)
│   │   ├── reasoner.py          ← ReasonerAgent (LLM call, prompt cached)
│   │   ├── validator.py         ← ValidatorAgent (LLM call)
│   │   └── llama_agent.py       ← python ReAct loop (AGENT_MODE=llama_index)
│   ├── services/
│   │   ├── storage.py           ← save_upload(), isolates file saving (S3 swap in a future version)
│   │   ├── document_loader.py   ← direct per-format parsers (validate, then parse)
│   │   ├── chunk_service.py     ← chunking router (by extension) → word_chunker() today
│   │   └── vector_store.py      ← ChromaDB interface, ONLY file that imports ChromaDB
│   ├── core/
│   │   └── config.py            ← loads and validates all env vars in one place
│   └── utils/
│       └── logging.py           ← structured JSON logging + query hashing
├── frontend/
│   └── app.py                   ← Streamlit UI
├── data/                        ← uploaded documents (gitignored)
├── docs/                        ← all design documentation
├── requirements.txt             ← all Python dependencies
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example                 ← all env vars with defaults, no secrets
├── .gitignore
├── setup_venv.sh                ← venv setup for local dev without Docker
└── README.md
```

**Virtual environment setup (local development without Docker):**

```bash
# Create and activate
python -m venv genai-doc-assistant
source genai-doc-assistant/bin/activate        # Mac/Linux
genai-doc-assistant\Scripts\Activate.bat       # Windows

# Install and configure
pip install -r requirements.txt
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# NOTE: Docker is the primary, supported run path. Local venv is secondary.
# requirements.txt pins python-magic (Linux/Docker). On Windows, the libmagic
# binary is not bundled: Windows venv users must instead run:
#     pip install python-magic-bin
# (setup_venv.sh carries this same note as a comment.)

# Run (two terminals)
uvicorn app.api.main:app --reload --port 8000
streamlit run frontend/app.py
```

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Docker Compose Design

### Three Key Concepts

**1. Port mapping `HOST:CONTAINER`**

```yaml
ports:
  - "8000:8000"   # your PC's port 8000 → container's port 8000
  - "8501:8501"   # your PC's port 8501 → container's port 8501
```

You access the system at `localhost:8000` (API) and `localhost:8501` (UI) in your browser.

**2. Named volumes, how ChromaDB survives restarts**

```yaml
volumes:
  - chroma_data:/data/chroma   # named volume → container path
```

Docker manages `chroma_data` on your host filesystem. When the backend container stops and restarts, `/data/chroma` is still there. Without this volume mount, ChromaDB resets to empty on every restart.

```bash
docker compose down       # stops containers, volume preserved  ✓
docker compose down -v    # stops containers, volume DELETED    ✗ data gone
```

Document `-v` clearly in the README, users who run it accidentally lose all indexed documents.

**3. Service name as internal DNS**

Inside the Docker network, each service is reachable by its service name:

```yaml
# Streamlit reaches FastAPI using the service name, not localhost
environment:
  - BACKEND_URL=http://backend:8000
```

`localhost` inside the Streamlit container refers to the Streamlit container itself, not the backend. `http://backend:8000` works because Docker gives each service a DNS entry equal to its name within the shared network.

**4. `depends_on` with health check, startup order**

```yaml
frontend:
  depends_on:
    backend:
      condition: service_healthy   # wait for /health to return 200
```

Without this, Streamlit starts before FastAPI is ready. The first UI load fails. With it, Docker waits until the backend `/health` endpoint returns 200 before starting Streamlit. Requires a `healthcheck` defined on the backend service.

```yaml
backend:
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Limitations

- Single-turn Q&A only (no conversation history)
- ChromaDB is embedded, not suitable for concurrent multi-user load at scale
- No user authentication or access control
- Document storage is local container filesystem (not persistent across container recreation without a volume mount)
- Streaming is custom-mode only (D-10): `custom` streams the answer token-by-token; `llama_index` and `compare` are returned after full generation
- File size limit: 10MB per document
- Query response time is ~2–4 seconds typical on the Haiku default (3 LLM calls: Planner + Reasoner + Validator, plus 1 encoder call for ReRanker); in custom mode the answer streams, so the first words appear in ~2–4s rather than the full wait (D-9, D-10)
- Embedding model (all-MiniLM-L6-v2) has a 256 token (~192 word) limit, chunk size set to 200 words to respect this limit
- CSV and Excel documents with many columns or long cell values produce lower quality retrieval, row-aware chunking deferred to a future version
- Embedding model is optimised for general English, technical, medical, or non-English documents may produce lower retrieval quality
- **Original uploaded files are not persisted**, only chunks are stored in ChromaDB. If the ChromaDB volume is lost or deleted, all indexed content is gone and documents must be re-uploaded. There is no way to recover without the original files.
- `vector_store.py` is tightly coupled to ChromaDB, swapping to another vector database requires rewriting this file. Backend abstraction layer is deferred to a future version.

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Extensibility Hooks (built in now)

These seams are added now at near-zero cost so a future version is an addition, not a rewrite. They do not change current behaviour.

**a) Vector store isolation.** `vector_store.py` is the **only** module that imports ChromaDB. All other code calls it through a narrow internal interface (`index_chunks`, `retrieve`, `list_documents`, `delete_document`, `chunk_count`). A future version adds a `VectorStore` ABC + factory on top without touching callers.

**b) File-saving isolation.** All file persistence goes through `save_upload()` in `app/services/storage.py`. The system writes to the local filesystem; a future version swaps the body for S3, one function, no other code changes.

**c) Chunking dispatch.** `chunk_service.py` routes by file extension even though every type currently calls the same `word_chunker()`. A future version adds `row_chunker()` (CSV/Excel) and `page_chunker()` (PDF) to the same router without touching callers.

**d) `agent_mode` plumbed through the request.** `pipeline.py` never reads the `AGENT_MODE` env var directly. `config.py` reads it once as the default; the value is passed as a parameter down the call chain. The optional `agent_mode` field on `POST /query` overrides it per request (required for `compare` mode).

<div align="right"><a href="#table-of-contents">&#8593; back to top</a></div>

---

## Scope for Future Versions

The current version is a self-contained local system. The items below are the planned direction for future
versions, building on the extensibility seams already in place (see *Extensibility Hooks (built in now)*).
These are production-hardening and feature additions; none of them change the core retrieval and
grounding behaviour that the current version establishes.

### Deployment and infrastructure
- **AWS deployment via Terraform.** Move from local Docker Compose to a reproducible cloud
  deployment (ECS for the services, S3 for storage, ElastiCache/Redis for a background ingestion
  queue), defined as infrastructure-as-code so an environment can be stood up or torn down
  predictably.
- **Vector store backend abstraction.** The current version couples directly to embedded ChromaDB. A future
  version introduces a narrow vector-store interface with a factory, so the backend can be swapped
  (for example to a managed service such as Pinecone) through an environment variable with no
  application-code change. The embedding model stays the same; only the storage layer changes.
- **Original file persistence and lifecycle.** The current version discards the original bytes after parsing. A
  future version stores originals in S3 so the index can be rebuilt from source, with lifecycle
  policies for TTL-based auto-expiry.

### Observability and reliability
- **External observability.** Ship the existing structured logs to a dashboarding stack (CloudWatch
  plus Loki into Grafana) so per-query timing, cost, and error rates are visible over time, not only
  in raw logs. The per-request correlation id and the `query_completed` / `llm_usage` events from
  the current version feed this directly.
- **Error classification for retries.** The current version uses a single blanket retry. A future version
  classifies API errors so only transient failures (timeout, rate limit, 5xx) are retried, while
  permanent ones (bad request, auth) fail fast with a precise reason for the operator.
- **PII redaction in logs.** The current version already hashes the raw query for correlation. A future version
  adds field-level redaction of any personal data that could appear in logged content (for example
  via a redaction service), keeping logs debuggable but privacy-safe.

### Retrieval and ingestion
- **Format-aware chunking.** The current version chunks every format with a fixed word window. A future version
  adds row-aware chunking for CSV and Excel and better handling of complex PDFs (multi-column
  layouts, tables), so document structure is preserved for retrieval.
- **Multilingual support.** The current embedding model is primarily English. Because the embedding
  model is already a configuration knob (subject to the re-index rule), a future version can swap in
  a multilingual model to raise quality on non-English documents.
- **OCR and protected PDFs.** Scanned or image-only PDFs (no text layer) and password-protected PDFs are rejected today; both were deferred as heavy or niche dependencies not needed for the current scope. A future version can add OCR (for example Tesseract) and password-unlock support.

### Product features
- **Multi-turn conversation.** The current version is single-turn. A future version adds conversation history
  (for example session state in a managed store) so follow-up questions can build on prior turns.
- **Authentication and multi-tenant isolation.** The current version assumes a single trusted user. A future
  version adds authentication (starting with API-key middleware, with room for SSO / federated
  login) and per-user document isolation, so one deployment can serve multiple users safely.
- **Document versioning and history.** The current version treats each upload independently with user-managed deletion. A future version adds document versioning and update history alongside the S3 lifecycle policies above.
- **Concurrent-write safety at scale.** Embedded ChromaDB has a single-writer limitation. The managed
  vector backend above removes it, allowing safe concurrent ingestion.

### Pipeline efficiency
- **Fewer LLM calls per query.** A future version reduces the custom pipeline to at most two LLM
  calls per query by replacing the Validator's LLM call with a deterministic grounding check on the
  normal path, reserving the LLM Validator for higher-stakes queries. This lowers cost and latency
  without weakening the grounding guarantees.

<!-- refinement -->
