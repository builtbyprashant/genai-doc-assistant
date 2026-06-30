# Agentic RAG Knowledge System: Decision Log

Design-first development was a core principle: the architecture, API contract, requirements, and
edge cases were documented and reviewed before any code was written, and every consequential change
was logged here so the evolution of the design stays visible.

Each decision has a stable ID: `C-x` for a conflict resolved during design review, `D-x` for a
design decision, `P-x` for a process/workflow decision.

---

## Index

| ID | Decision |
|---|---|
| C-A | 3 LLM agents + 4 non-LLM steps (the reranker is a local encoder step) |
| C-B | Drop the `reranker_mode` field; the reranker is a fixed cross-encoder |
| C-C | Content-dedup hash is SHA-256 (first 32 hex chars) |
| C-D | `agent_mode` is overridable per request (the env var is only the default) |
| C-E | `compare` mode returns a side-by-side response envelope |
| C-F | An empty document store returns HTTP 200 short-circuit, not 409 |
| C-G | Typical custom-mode response time is 2 to 4 seconds |
| C-H | Canonical trace step names and a fixed `hallucination_risk` enum |
| D-1 | ChromaDB uses cosine space; `similarity = 1 - distance` |
| D-2 | Validate raw bytes first, then parse with direct per-format libraries |
| D-3 | Added env vars: `MAX_DOCUMENTS`, `MIN_CHUNK_CHARS`, model knobs, `GROUNDING_THRESHOLD` |
| D-4 | Docker is the primary path; a Windows venv is secondary |
| D-5 | Validator independence enforced by its input contract |
| D-6 | Extensibility seams baked in at near-zero cost |
| D-7 | The LlamaIndex library is not used; `llama_index` mode is a from-scratch ReAct loop |
| D-8 | Docker image optimization: CPU-only torch + split requirements |
| D-9 | Default model switched to Haiku for latency (still configurable) |
| D-10 | Streamed answer for custom mode (`POST /query/stream`) |
| D-11 | Config is environment-specific; `.env` is local-dev only |
| D-12 | Prompt caching toggle with usage logged for observability |
| D-13 | Similarity threshold default lowered 0.40 to 0.30 |
| D-14 | Per-query token usage and estimated cost shown in the UI |
| D-15 | The ReAct loop stops searching on diminishing returns |
| D-16 | Cache the retrieved CONTEXT block, not the short system prompt |
| P-1 | Commit strategy: one commit per task, conventional format |
| P-2 | Test-driven development with pytest |
| P-3 | CI: lint and test everywhere; image build only when it can break |

---

## Conflicts resolved during design review

**C-A. Agent count: 3 LLM agents + 4 non-LLM steps.** The LLM agents are PlannerAgent,
ReasonerAgent, and ValidatorAgent. The non-LLM steps are SafetyGuard, Retriever,
SimilarityThreshold, and ReRanker. ReRanker uses the local cross-encoder
`ms-marco-MiniLM-L-6-v2` and makes no API call. Custom mode is therefore 3 LLM calls plus 1 local
encoder call per query.

**C-B. Drop the `reranker_mode` field.** The reranker is fixed to the cross-encoder; there is no
LLM-rerank mode, so a mode field was misleading. The trace now emits the reranker model name and the
UI shows a static "Reranker: cross-encoder" label.

**C-C. Content-dedup hash.** Document deduplication uses the SHA-256 of the raw file bytes (first 32
hex chars). This is distinct from the query-log hash (also SHA-256, first 8 hex chars), which exists
only to correlate logs without recording the raw query.

**C-D. `agent_mode` overridable per request.** The `AGENT_MODE` env var supplies the default; an
optional `agent_mode` field on `POST /query` overrides it for a single query. This is required
because `compare` mode is inherently a per-request choice.

**C-E. `compare` response envelope.** In `compare` mode both pipelines run and the response is a
side-by-side envelope: each side carries its own `answer`, `confidence`, `sources_used`,
`llm_calls`, `duration_ms`, `chunks`, and `trace`, with top-level `mode`, `request_id`, `timestamp`,
`question`, `processing_time_ms`, `short_circuit`, and `validation`. Validation normally runs on the
custom answer; if the custom pipeline short-circuits, the `llama_index` pipeline still runs and
validation targets its answer instead. The `llama_index` answer self-rates its confidence.

**C-F. Empty store returns HTTP 200.** An empty document store is a deliberate pipeline outcome, not
a bad request, so it returns `200` with `short_circuit: true` and a reason. (409 is retained for
duplicate content and the max-documents limit.)

**C-G. Typical response time is 2 to 4 seconds** (custom mode), standardized across the docs. This
target is met via the Haiku default (D-9) and streamed answers (D-10); streaming also makes the
perceived time the time-to-first-token rather than the full generation time.

**C-H. Canonical trace names and enums.** Trace step names are fixed to the agent names above, and
`hallucination_risk` is one of `low | medium | high | unknown | n/a`, with `n/a` used only on
short-circuit responses.

---

## Design decisions

**D-1. ChromaDB cosine space.** The collection is created explicitly with `hnsw:space=cosine`.
ChromaDB returns a distance, so `vector_store.py` converts every result with `similarity = 1 -
distance`. The default `SIMILARITY_THRESHOLD` is only meaningful in cosine space. Without this,
ChromaDB's default L2 space would make every documented score and threshold wrong.

**D-2. Validate raw bytes, then parse with direct libraries.** Validation (empty/size checks, MIME
detection, SHA-256 dedup, scanned-PDF and password-PDF rejection) runs on the raw bytes before any
parser touches the file. Parsing then uses each format's own library (`pypdf`, `python-docx`,
`pandas` with `openpyxl`/`xlrd`, `pyyaml`, `json`, `chardet`) for full control over the documented
per-format edge cases and for deterministic tests.

**D-3. Added env vars.** `MAX_DOCUMENTS=20` (enforced at upload), `MIN_CHUNK_CHARS=20` (smaller
chunks discarded), `EMBEDDING_MODEL=all-MiniLM-L6-v2` (bi-encoder, config-driven, must be 384-dim or
the index is rebuilt), `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` (outputs a score not a
vector, so it is drop-in swappable), and `GROUNDING_THRESHOLD=0.6` (present but unused for now).

**D-4. Docker primary, Windows venv secondary.** `requirements.txt` pins `python-magic` for
Linux/Docker; Windows venv users install `python-magic-bin` instead (documented in `setup_venv.sh`).
Docker is the supported path.

**D-5. Validator independence by input contract.** The Validator receives only the original
question, the final answer text, and the top retrieved chunks. It never receives the Reasoner's
reasoning, the Planner output, or any other pipeline state, so its hallucination check is genuinely
independent.

**D-6. Extensibility seams baked in.** Added at near-zero cost without changing current behaviour:
the vector store is the only module that imports ChromaDB and is reached through a narrow interface;
all file persistence goes through one `save_upload()` function; chunking dispatches by file
extension behind a stable call site; and `agent_mode` is passed as a parameter rather than read
directly inside the pipeline.

**D-7. No LlamaIndex library.** The LlamaIndex library is not used anywhere and
is absent from `requirements.txt`. The `llama_index` mode is a genuine ReAct reasoning loop built
using python, interacting directly with the Anthropic API: it seeds context from the vector store, then loops up to four turns
asking the model for exactly one action per turn (`SEARCH: <query>` to fetch more context, or
`FINAL: <answer>` to finish), making 1 to 4 LLM calls depending on complexity. This keeps the system
self-contained and the image small.

**D-8. Docker image optimization.** The backend installs CPU-only torch from the PyTorch CPU index
before the rest, so `sentence-transformers` reuses it instead of pulling ~2.5GB of unused CUDA
wheels. Requirements are split into backend, frontend, and dev sets, so the frontend image carries
no ML stack and is a fraction of the size.

**D-9. Default model switched to Haiku.** A latency review traced multi-second responses to answer
generation. The default model is now `claude-haiku-4-5` (roughly 3x faster generation, cheaper, and
sufficient for grounded RAG where answers are constrained to retrieved context). It stays fully
configurable via `ANTHROPIC_MODEL`; set it to a larger model to trade latency for answer quality.

**D-10. Streamed answer for custom mode.** `POST /query/stream` streams the answer token-by-token,
then a single record-separator byte, then a JSON metadata frame (confidence, sources, validation,
trace, timing). The structured scaffolding is hidden mid-stream and only the answer text is shown.
Safety and filter checks run before any byte is sent, so errors still surface as clean responses.
`llama_index` and `compare` remain batch for now.

**D-11. Config is environment-specific.** `config.py` reads `os.environ`, so the app does not care
where env vars come from, only that they are present. `.env` is a local-dev convenience, never a
production config or secret surface. The API key comes from the host shell environment, never from
`.env` and never baked into the image.

**D-12. Prompt-caching toggle.** `ANTHROPIC_CACHE_MODE` selects `block` (default), `prompt`, or
`off`. Every LLM call logs an `llm_usage` event with cache-read and cache-creation token counts, so
cache effectiveness is observable from the logs. (See D-16: block-level caching of the short system
prompt turned out to have no real effect, which D-16 corrected.)

**D-13. Similarity threshold lowered 0.40 to 0.30.** A genuinely relevant query scored 0.39 on the
bi-encoder and was wrongly rejected by the 0.40 gate, while `llama_index` mode (no gate) answered it
honestly. 0.40 is too high for `all-MiniLM-L6-v2`, where relevant short or colloquial queries land in
the 0.3 to 0.5 band. The default is now 0.30 and is live-tunable; the gate still refuses truly
irrelevant queries, which score far lower. A more accurate gate on the cross-encoder score was
considered and deferred as a larger change.

**D-14. Per-query tokens and cost in the UI.** Tokens consumed and an estimated USD cost are shown
next to confidence, call count, and time (per side in `compare`), using model-aware pricing with
cache discounts, so the speed, quality, and cost tradeoff between modes is visible rather than buried
in the logs.

**D-15. Diminishing-returns stop for the ReAct loop.** The loop could spend its whole search budget
re-asking near-duplicate queries. Now, after a `SEARCH`, if it adds zero new chunks or overlaps the
already-seen chunks by 70% or more, the loop synthesizes immediately instead of searching again. The
call range stays 1 to 4; the guard only ever ends the loop earlier, avoiding roughly 4x token waste
from redundant retrievals.

**D-16. Cache the retrieved CONTEXT block, not the system prompt.** The only thing originally marked
for caching was the system prompt, which at roughly 100 tokens is below the model's minimum cacheable
length, so caching was silently ignored and never fired. The fix moves the cache breakpoint to the
retrieved context (the large, reused content). Within a custom-mode query the Reasoner and Validator
share one grounding system and restate the identical context block, so the Validator reads the
context the Reasoner cached. Independence (D-5) still holds: the Validator sees only question, answer,
and context. Cache effectiveness is verifiable from the `cache_read_input_tokens` field in the logs.

---

## Process and workflow

**P-1. Commit and git strategy.** One commit per task (not per file), in Conventional Commits format.
Solo workflow: commit directly to `main` and push, no PRs or feature branches. Stage only the named
files for each task so secrets never enter history; a final commit verifies no secrets were
committed.

**P-2. Test-driven development.** Tests are the spec, written before or alongside each service and
shipped in the same task commit. Each service has required coverage (cosine and threshold behaviour;
chunk sizing and routing; all eight document formats including scanned/password-PDF rejection and
dedup; all API endpoints including multi-status upload, the compare envelope, and short-circuits;
safety, empty-store, and threshold short-circuits).

**P-3. CI strategy.** Lint and test run on every push and PR for fast feedback. The Docker image
build is path-gated: it runs only on a push to `main` that changed an image-affecting file, since the
image can only break when dependencies or Dockerfiles change. A scheduled job keeps the dependency
and model-download caches warm so a low-activity repo does not go cold, and a build-layer cache keeps
warm builds short.

<!-- refinement -->
