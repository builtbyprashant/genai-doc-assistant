# Agentic RAG Knowledge System: Requirements and Assumptions

## Purpose

This document records every explicit design decision, functional assumption, acceptable behaviour, and known limitation for Phase 1. It exists to make implicit choices visible, provide rationale for reviewers and evaluators, and serve as the authoritative reference during build.

Every decision here was made consciously. Where multiple options existed, the chosen option and the reason for choosing it are both documented.

---

## 1. Design Decisions

### 1.1 Document Query Scope

**Options considered:**
- A, Search only documents uploaded in the current session (resets on restart)
- B, Search all documents ever uploaded (always persistent, no user control)
- C, Search all documents by default; user can optionally filter to specific documents per query

**Decision: Option C**

**Rationale:** Persistence via Docker volume costs nothing extra. Filtering by filename is already supported in the vector store interface. For a general-purpose system, the ability to say "answer only from this policy document" is a meaningful and demonstrable feature. It also reflects real enterprise usage where users work with a known set of documents.

**Behaviour:**
- Default (no filter specified): query searches all indexed documents across all sessions
- Filter specified: query searches only the named documents
- Filter with unknown filename: rejected with a descriptive error, not silently ignored

---

### 1.2 Re-ranker Mode

**Options considered:**
- LLM re-ranker, Claude scores and reorders retrieved chunks (transparent, no extra dependency)
- Cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2` runs locally (faster, free, industry standard)

**Decision: cross-encoder/ms-marco-MiniLM-L-6-v2, fixed, no feature flag**

The ReRanker always uses the cross-encoder model. The LLM reranker option was removed in favour of a simpler, faster, and cheaper design. Encoder reranking is more appropriate for production use, no API cost, no extra LLM latency, deterministic results.

**Model delivery:** The cross-encoder model (~80MB) is pre-downloaded into the Docker image during build. No internet dependency after the image is built. Image size is approximately 1.5GB (reduced from ~2GB by removing the dual-mode complexity).

---

### 1.3 Duplicate Document Handling

**Options considered:**
- Allow duplicates, index both, return both in results
- Replace on same filename, delete old, index new
- Reject identical content, hash-based deduplication

**Decision: Two-layer logic**

**Layer 1, Content hash check (at upload):**
Compute a SHA-256 hash of raw file bytes at upload time (first 32 hex chars stored as `content_hash`). If the same hash already exists under a different filename, reject the upload with: "This document appears to be identical to an already indexed file: [existing filename]. Uploading duplicate content degrades answer quality." Response quality is the system's responsibility, duplicate chunks dilute the context window and cause the Reasoner to cite the same content twice under different names.

**Layer 2, Filename replace (same filename, different content):**
If the same filename is uploaded with different content (different hash), treat as a replace: delete all existing chunks for that filename, index the new version. This matches the user's mental model, uploading report.pdf again means "use the new version."

**Failure handling:** Replace is delete-then-index. If indexing fails after deletion, the document is unindexed. The upload response will indicate failure. The user must retry, retrying the same filename will trigger the replace logic again cleanly.

---

### 1.4 API Response Flags

**Decision:** Both retrieved chunks and agent trace are optional in the query response, controlled by request flags.

| Flag | Default | When true |
|---|---|---|
| `include_chunks` | false | Response includes retrieved chunks with similarity scores and filenames |
| `include_trace` | false | Response includes full agent-by-agent trace log |

**Rationale:** Cleaner API design. Programmatic consumers rarely need the trace. The Streamlit UI sets both flags to true by default so the learning experience is always transparent. API consumers can request only what they need.

---

### 1.5 API Versioning

**Decision:** Flat routes for Phase 1 (`/query`, `/documents/upload`, etc.). No `/api/v1/` prefix.

**Rationale:** Phase 1 is a single-version system. Adding versioning prefixes now adds complexity with no benefit. Phase 2 introduces versioning when breaking changes become a real concern. Document the current routes clearly so Phase 2 knows exactly what to version.

---

### 1.6 User Sessions and Multi-Browser Behaviour

**Decision:** No session isolation. All browser tabs share the same document pool.

**Rationale:** There is no authentication in Phase 1. ChromaDB is a single shared instance. Multiple tabs opening the same `localhost:8501` each get their own Streamlit UI state (selected filters, typed questions) but share the same backend document index.

**Acceptable behaviour:** Tab A uploading a document makes it immediately visible to Tab B. Tab A deleting a document affects Tab B. This is documented as a known single-tenant characteristic, not a bug. Multi-tenant isolation is a Phase 2 concern.

**Concurrency risk:** Embedded ChromaDB is not designed for concurrent writes. Simultaneous uploads from multiple tabs can corrupt the index. Mitigation: a file-based write lock during indexing causes a second upload to wait rather than corrupt. Document as a known limitation, sequential uploads are assumed for Phase 1.

---

## 2. Functional Assumptions

### 2.1 User and Usage Model

- One primary user at a time, possibly across multiple browser tabs
- All tabs share the same document pool with no isolation
- Sequential document uploads assumed, concurrent uploads from multiple tabs are handled via write lock but not guaranteed safe at high concurrency
- No authentication, no access control, no user accounts
- The user is assumed to upload documents in good faith, no adversarial input beyond the prompt injection guardrail

### 2.2 Document Assumptions

- Documents are in one of the supported formats: PDF, TXT, MD (Markdown), CSV, Excel (.xlsx, .xls), JSON, YAML (.yaml, .yml), DOCX
- PDFs have a text layer (not scanned/image-based)
- TXT files contain human-readable prose or structured text
- CSV files have or can operate without headers
- Excel files contain tabular data in cell values, charts and images are not expected to be searchable
- Documents are in a language supported by the embedding model (primarily English)
- Documents contain content relevant to the questions being asked, the system cannot answer questions about topics not present in any uploaded document

### 2.3 Infrastructure Assumptions

- Docker and Docker Compose are installed on the host machine
- The host machine has internet access during `docker build` (for model and package downloads)
- After image build, no internet is required except for Anthropic API calls
- The Anthropic API key (read from the host shell/OS environment, not `.env`, D-11) is valid and has sufficient quota
- The host machine has at least 4GB RAM available for the Docker environment
- The ChromaDB volume mount path is writable by the Docker process

---

## 3. Acceptable Behaviours by Edge Case

### 3.1 Document Ingestion

| Edge case | Acceptable behaviour |
|---|---|
| Scanned PDF (no text layer) | Reject. Error: "This PDF appears to be scanned or image-based. Text could not be extracted. Please use a PDF with a text layer or convert it using OCR first." OCR is out of scope for Phase 1. |
| Password protected PDF | Reject. Error: "This PDF is password protected and cannot be read. Please remove the password protection before uploading." |
| MIME type mismatch (e.g. .pdf that is not a PDF) | Reject. Validate actual MIME type using `python-magic`, not just extension. Error: "File content does not match the declared file type." |
| CSV with no headers | Auto-generate column names (Col_1, Col_2...) and proceed. Include warning in upload response: "No column headers detected. Generic column names were assigned." |
| CSV with empty rows | Skip empty rows silently during chunk generation. |
| Excel with merged cells | pandas reads the top-left value of a merged cell. Accept this behaviour. Note in upload response: "Charts, images, and complex formatting were ignored. Only cell values were indexed." |
| Excel with multiple sheets | Index all sheets. Prefix each chunk with the sheet name for traceability. |
| Excel with formulas | pandas reads computed values, not formula text. This is correct behaviour, document it. |
| Non UTF-8 TXT file | Auto-detect encoding using `chardet`. Fall back to `errors='replace'` if detection fails. Note in upload response if non-UTF-8 encoding was detected. Never crash. |
| Same filename, new content | Replace: delete all existing chunks for that filename, index new content. If indexing fails, document is unindexed, user must retry. |
| Same content, different filename | Reject. Error: "This document appears to be identical to an already indexed file: [existing filename]. Uploading duplicate content degrades answer quality." Content hash (SHA-256, first 32 hex chars) is checked at upload time. |
| Same filename, same content | Treat as replace (same behaviour). Result is identical index state. |
| File larger than UPLOAD_SIZE_LIMIT_MB | Reject before processing. Error: "File size exceeds the [limit]MB limit. Please upload a smaller file or split the document." |
| Empty file (0 bytes) | Reject. Error: "The uploaded file is empty." |
| Document shorter than one chunk | Produce exactly one chunk containing the full document content. Never return zero chunks from a non-empty document. |
| Chunk is whitespace or punctuation only | Discard silently. Any chunk where stripped content is less than MIN_CHUNK_CHARS characters is dropped. |
| Upload when MAX_DOCUMENTS limit reached | Reject. Error: "Maximum document limit of [MAX_DOCUMENTS] has been reached. Please delete an existing document before uploading a new one." |

---

### 3.2 Retrieval

| Edge case | Acceptable behaviour |
|---|---|
| Query with empty vector store | Short-circuit before touching ChromaDB. Return HTTP 200 with `short_circuit: true`, `short_circuit_reason: "no_documents_indexed"`. Message: "No documents have been indexed yet. Please upload at least one document before asking questions." Not a 4xx error, the pipeline ran and made a deliberate decision. |
| Query with filter_filenames containing unknown filename | Reject with descriptive error: "The following filenames were not found in the index: [list]. Please check the filename or upload the document first." |
| Query with filter_filenames where no chunks clear similarity threshold | Return insufficient context response naming the filtered documents: "No relevant content was found in the selected documents [list] for this question. Try broadening your document selection or rephrasing your question." |
| TOP_K_RETRIEVAL larger than total chunks in store | Use min(TOP_K_RETRIEVAL, total_chunk_count) silently. Log at DEBUG level. No user impact. |
| TOP_K_RERANK larger than chunks retrieved | Return all retrieved chunks. No error. min() guard applied internally. |
| Top chunk similarity score below SIMILARITY_THRESHOLD | Short-circuit pipeline immediately. No LLM calls made. Response: `{ "success": false, "short_circuit": true, "reason": "similarity_threshold_not_met", "threshold": 0.4, "top_score": [actual score] }` |
| Two documents with identical content returned as sources | Prevented at upload time via content hash check. Cannot occur in normal operation. |

---

### 3.3 Agent Pipeline

| Edge case | Acceptable behaviour |
|---|---|
| Planner returns malformed JSON | Fall back to a safe default plan: `{ "intent": "answer user question", "retrieval_strategy": "semantic", "top_k": TOP_K_RETRIEVAL, "rewritten_query": [original query] }`. Log raw LLM output at WARNING level. Pipeline continues. |
| Validator returns malformed JSON | Fall back to: `{ "is_valid": true, "issues": [], "hallucination_risk": "unknown", "suggested_action": "none" }`. Log raw LLM output at WARNING level. Pipeline continues. |
| Anthropic API unavailable or rate limited | Return HTTP 503 with structured body: `{ "success": false, "error": "LLM service unavailable", "failed_at_agent": "[agent name]", "retry_after": 30 }`. Streamlit UI catches all non-200 responses and displays a human-readable message without crashing. Raw errors never shown to the user. |
| Anthropic API timeout | Each LLM call uses LLM_TIMEOUT_SECONDS (default 20s) for first attempt. One automatic retry at half timeout (10s). Total max wait per LLM call: 30s. After exhausted retries, return structured 503 response. Blanket retry, no error type classification in Phase 1. |
| Reasoner generates very long answer | max_tokens set on Reasoner call (default 800 tokens). Answers are bounded. Document as a known constraint, very long documents may warrant a higher limit via env var. |
| All reranked chunks have identical scores | Preserve original retrieval order (by cosine similarity) as tiebreaker. Deterministic, no user impact. |
| Encoder mode: cross-encoder model unavailable | Will not occur in normal operation, model is pre-downloaded during Docker image build. If image was built without internet, build fails explicitly, not a runtime failure. |

---

### 3.4 Input Guardrails

| Check | Behaviour on failure |
|---|---|
| Query length < 3 characters | Reject. Error: "Query is too short. Please ask a complete question." |
| Query length > 2000 characters | Reject. Error: "Query exceeds the maximum length of 2000 characters." |
| Prompt injection patterns detected | Reject. Error: "Query was flagged by the safety guardrail." Log the flagged pattern at WARNING level. |
| Question submitted with no documents indexed | Short-circuit at the Retriever step (status `skipped`). Return HTTP 200 with `short_circuit: true`, `short_circuit_reason: "no_documents_indexed"`. Message: "No documents have been indexed yet." (Same as empty store check, not a 4xx error.) |

---

### 3.5 UI Behaviour

| Situation | Acceptable behaviour |
|---|---|
| Any FastAPI non-200 response | Streamlit catches and displays a human-readable error message. Raw HTTP errors, stack traces, and JSON error bodies are never shown to the user directly. Detailed error info is available in the expandable trace section. |
| Query in progress | UI shows a spinner. Submit button is disabled during processing to prevent duplicate submissions. |
| Empty question submitted | Client-side validation in Streamlit before API call. Error displayed inline: "Please enter a question before submitting." |
| filter_filenames multiselect empty | Treated as "search all documents." UI label makes this clear: "Searching all documents." |
| include_chunks=true but pipeline short-circuited | chunks field is an empty array. Response includes `short_circuit: true` and `reason` field. UI displays the reason clearly rather than an empty chunks panel. |
| Document deleted while query is in progress | Query completes normally (chunks already retrieved). Deleted document may still appear in sources for that response. This is acceptable, the delete took effect for future queries. |

---

### 3.6 Operational

| Situation | Acceptable behaviour |
|---|---|
| Backend container crashes | Docker Compose `restart: always` policy restarts it automatically. ChromaDB index persists via volume mount, no data loss. |
| Frontend container crashes | Docker Compose `restart: always` restarts it. No state loss, Streamlit is stateless server-side. |
| ChromaDB volume deleted manually | All indexed documents are lost. Uploaded files (if any were saved locally) are unaffected. User must re-upload documents. Documented as an operational risk, treat the volume as the primary data store. |
| Original file bytes after upload | Original file bytes are discarded after parsing and chunking. Only chunks are stored in ChromaDB. If the volume is lost, documents must be re-uploaded, they cannot be recovered from the system. This is a known Phase 1 limitation. |
| Log volume growth | Docker log rotation configured in docker-compose.yml: `max-size: 10m, max-file: 3`. Prevents unbounded log growth on long-running instances. |
| Query PII in logs | Raw user queries are never logged. Every log line records `query_hash` (SHA-256 first 8 hex chars) and `query_length` instead. Hash is deterministic, same query always produces same hash for correlation. Cannot be reversed to recover original query. |
| `.env` file on GitHub | `.env` is listed in `.gitignore`. Only `.env.example` (non-secret defaults, no key) is committed. The API key is never in `.env`, it comes from the host shell/OS env (D-11). Enforced by project convention, documented in README. |

---

## 4. Capacity and Limits

All limits are configurable via environment variables unless marked as fixed.

| Limit | Default | Env var | Notes |
|---|---|---|---|
| Max file size per upload | 10 MB | `UPLOAD_SIZE_LIMIT_MB` | Enforced before parsing |
| Max documents in index | 20 | `MAX_DOCUMENTS` | Enforced at upload time |
| Max question length | 2000 chars | Fixed (guardrail) | Hardcoded in safety check |
| Min question length | 3 chars | Fixed (guardrail) | Hardcoded in safety check |
| Min chunk content length | 20 chars | `MIN_CHUNK_CHARS` | Chunks below this are discarded |
| Chunk size | 200 words | `CHUNK_SIZE` | Chosen to stay within all-MiniLM-L6-v2's 256 token limit (~192 words for typical English prose) |
| Chunk overlap | 25 words | `CHUNK_OVERLAP` | 12.5% of chunk size, preserves boundary context without excessive repetition |
| Chunks fetched before re-ranking | 10 | `TOP_K_RETRIEVAL` | Candidates for the ReRanker |
| Chunks passed to Reasoner after re-ranking | 5 | `TOP_K_RERANK` | Must be ≤ TOP_K_RETRIEVAL |
| Similarity threshold | 0.4 | `SIMILARITY_THRESHOLD` | Below this = short-circuit. Range in practice: 0.0–1.0 (underlying cosine math is -1 to +1 but text embeddings never produce negative values in normal use). Calibrated for general English prose with all-MiniLM-L6-v2. Domain-specific deployments should re-tune. |
| LLM call timeout | 20 seconds (first attempt) | `LLM_TIMEOUT_SECONDS` | Retry at 10s (half). Total max 30s per LLM call. |
| Query hash algorithm | sha256 | `QUERY_HASH_ALGO` | First 8 hex chars used in logs. Raw query never logged, zero PII risk. |
| Reasoner max output tokens | 800 | Fixed in code | Keeps answers concise |
| Docker log max size | 10 MB | docker-compose.yml | Per log file |
| Docker log max files | 3 | docker-compose.yml | Rotating |
| Recommended total corpus | ~5000 chunks | n/a | Beyond this, embedded ChromaDB slows noticeably (~20 × 10-page PDFs) |

---

## 5. Embedding Model Constraints and Chunking Rationale

### Model Used
ChromaDB's default embedding function uses **all-MiniLM-L6-v2** (sentence-transformers).

| Property | Value |
|---|---|
| Model | all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Max input tokens | 256 tokens |
| Approximate word equivalent | ~192 words (typical English prose) |
| Technical/medical text equivalent | ~150–160 words (longer words tokenize to more tokens) |
| Model size | ~80MB (pre-downloaded in Docker image) |
| Cost | Free, runs locally on CPU, no API key required |

### Why Chunk Size is 200 Words

The embedding model silently truncates any input exceeding 256 tokens. For a chunk of 400 words (our original design), the model only fully processes the first ~190 words, nearly half the chunk is invisible to the embedding. This means the vector stored in ChromaDB does not accurately represent the full chunk content, leading to imprecise retrieval.

**Decision:** Set CHUNK_SIZE=200 words to stay within the model's token limit for typical English prose. This ensures the embedding vector accurately represents the entire chunk.

**Tradeoff accepted:** Smaller chunks mean less context per chunk sent to the Reasoner. This is acceptable because:
- The Reasoner receives TOP_K_RERANK=5 chunks after re-ranking
- 5 × 200 words = ~1000 words of context, sufficient for accurate answer generation
- Retrieval precision (finding the right chunks) is more valuable than larger but imprecisely retrieved chunks

**Overlap set to 25 words:** 12.5% of chunk size, consistent with the original ratio. Preserves boundary context between adjacent chunks without excessive repetition.

### CSV Chunking Limitation

Our chunking implementation converts CSV rows to text strings and applies the same word-based chunker:

```
Row text: "Name: John Smith | Age: 45 | Diagnosis: Hypertension | Admitted: 2024-01-15"
```

**Known limitation:** The 200-word chunker may group multiple rows into one chunk or split a row across chunks if row text is long. This is acceptable for small CSVs with short values. For large CSVs or rows with many columns, retrieval quality degrades.

**Production-correct approach (Phase 2):** Row-aware chunking, group N complete rows per chunk where N keeps the chunk under the token limit. Never split mid-row. This requires a separate chunking strategy for tabular data.

**Documented assumption:** CSV and Excel documents with more than ~50 columns or very long cell values may produce lower quality retrieval results. Users should be aware of this limitation when uploading tabular data.

### Token-to-Word Conversion Reference

| Token count | Approximate words (prose) | Approximate words (technical) |
|---|---|---|
| 256 (model limit) | ~192 | ~150 |
| 200 | ~150 | ~120 |
| 100 | ~75 | ~60 |

Rule of thumb: 1 token ≈ 0.75 words for common English. Technical documents, medical terms, numbers, and structured data (CSV pipes, colons) tokenize less efficiently, budget 0.6 words per token for safety.

---

## 6. Out of Scope for Phase 1

The following were considered and explicitly deferred to Phase 2 or later:

| Feature | Reason deferred |
|---|---|
| OCR for scanned PDFs | Heavy dependency (Tesseract), significant complexity, not required for capstone |
| Password-protected PDF unlocking | Niche use case, security complexity |
| Multi-tenant document isolation | Requires authentication, session management, Phase 2 |
| Authentication and access control | Phase 2 |
| Streaming LLM responses | **Done in Phase 1 (D-10)**, `custom` mode streams the answer over chunked HTTP via `POST /query/stream` (no WebSocket). Phase 2 may add SSE + `llama_index`/`compare` streaming. |
| Multi-turn conversation history | Phase 2 (where `prompt`-level caching, D-12, pays off) |
| Document versioning and update history | Phase 2 |
| Concurrent write safety at scale | Embedded ChromaDB limitation, Phase 2 swaps to managed DB |
| Language detection and multilingual support | Embedding model is primarily English, Phase 2 |
| Document update without full re-upload | Requires chunk-level diffing, Phase 2 |
| Grafana / external observability | Phase 2 |
| AWS deployment | Phase 2 |

---

## 7. Glossary

| Term | Definition in this system |
|---|---|
| Session | The lifetime of a running Docker Compose stack. Documents persist across browser tabs but reset if the ChromaDB volume is deleted. |
| Chunk | A fixed-size overlapping segment of document text, produced by the chunking step and stored as a single vector in ChromaDB. |
| Similarity score | Cosine similarity between the query embedding and a chunk embedding. Range 0.0–1.0. Higher = more similar. The ChromaDB collection is created with `hnsw:space=cosine`; ChromaDB returns a distance, so `vector_store.py` converts every result with `similarity = 1 - distance`. |
| Similarity threshold | The minimum acceptable similarity score for the top retrieved chunk. Below this, the pipeline short-circuits. |
| Short-circuit | Early pipeline termination without calling any LLM. Triggered by the similarity threshold guardrail or an empty document store. |
| Re-ranking | The step after retrieval where chunks are reordered by how well they actually answer the question, as opposed to how topically similar they are. |
| Agent trace | A structured log of every agent's execution: what it received, what it decided, and its status. Returned in the API response when `include_trace=true`. |
| Content hash | SHA-256 hash of raw file bytes (first 32 hex chars stored as `content_hash`). Used at upload time to detect duplicate content across different filenames. |
| Replace | The document update operation: delete all existing chunks for a filename, then index the new version. |
