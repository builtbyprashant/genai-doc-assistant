# Agentic RAG Knowledge System

[![CI](https://github.com/builtbyprashant/genai-doc-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/builtbyprashant/genai-doc-assistant/actions/workflows/ci.yml)

> Upload your documents. Ask questions in plain English. Get grounded, cited answers powered by a multi-agent AI pipeline.

---

## What It Is

An AI-powered document intelligence system that allows users to upload enterprise or personal documents in multiple formats and ask natural language questions. The system retrieves relevant content using semantic search and generates grounded and cited answers with full cycle observability.

**The problem it solves:** Knowledge locked in documents, PDFs, spreadsheets, reports, markdown files etc is hard to query. This system makes any document collection conversational. Upload your documents, ask questions in plain English, get answers with source citations.

**What makes it agentic:** Rather than a single LLM call, in **custom** mode this system uses three specialised LLM agents (Planner, Reasoner, Validator) that collaborate in sequence, each with a focused role. Four non-LLM pipeline steps handle safety, retrieval, threshold checking, and cross-encoder re-ranking without wasting API calls.
While in **llama_index** mode (inspired by LlamaIndex) it uses a ReAct loop agent, built from scratch using python.

---

## What Makes This Different

Most RAG systems give you one pipeline and one answer. This system gives you **two architecturally distinct pipelines** and lets you **compare them side by side**.

- **`custom` mode** is a fixed 7-step sequential pipeline: deterministic, fully traced, optimised for predictability. Every step is visible, including which chunks were retrieved, how they were reranked, and what confidence the Validator assigned.
- **`llama_index` mode** implements the ReAct reasoning pattern internally as llama_agent, the same Think → Search → Observe loop that LlamaIndex's ReActAgent uses. Instead of a fixed retrieval pass, llama_agent decides dynamically when to search again and when it has enough to answer.
- **`compare` mode** runs both pipelines on the same query and renders the results side by side: answers, confidence scores, retrieved chunks with similarity scores, LLM call counts, per-step timings, and full agent traces for both. This turns the system into a **live experimentation and fine-tuning workbench**, where you can see exactly where the two approaches diverge, which chunks each pipeline found, and where one outperforms the other.

Built for engineers who want to understand what is happening inside their RAG pipeline, not just get an answer out of it.

---

**Built from scratch.** We implemented three pipeline modes using python and ChromaDB. For LLM interaction Anthropic API is used, which makes it hardbound with Anthropic LLMs of choice as of now. The `custom` mode is a deterministic sequential pipeline optimised for predictability and makes **3 LLM calls** per request (Planner, Reasoner, Validator; fewer when the pipeline short-circuits on a safety block or low similarity). The `llama_index` mode implements the ReAct reasoning pattern without the framework dependency; it makes **1–4 LLM calls** and stops early when a search stops surfacing new content (a **diminishing-returns guard**) rather than burning its search budget on redundant retrievals. `compare` mode runs both on the same query for evaluation.

---

## Architecture

Two architecturally distinct pipelines run behind the same FastAPI backend and share one retrieval stack (ChromaDB vectors plus the cross-encoder reranker). `compare` mode runs both on a single query.

### `custom` mode: fixed 7-step pipeline

```text
Browser ──▶ Streamlit UI ──HTTP──▶ FastAPI backend
                                        │
                  ┌─────────────────────┴──────────────────────┐
                  │              7-step query pipeline           │
                  │  SafetyGuard → Planner → Retriever →         │
                  │  SimilarityThreshold → ReRanker → Reasoner → │
                  │  Validator                                   │
                  └──────────┬───────────────────────┬──────────┘
                             ▼                        ▼
                   ChromaDB (vectors,         Anthropic API
                   cosine; persisted)         (Planner / Reasoner / Validator)
```

In `custom` mode the Reasoner's answer is **streamed** back to the UI token-by-token (`POST /query/stream`). The path is deterministic: the same query runs the same steps every time.

### `llama_index` mode: ReAct loop

```text
Browser ──▶ Streamlit UI ──HTTP──▶ FastAPI backend ──▶ ReAct loop (1–4 turns)
                                                              │
        ┌───────────────────────────────────────────────────┘
        ▼
   Think ──▶ SEARCH <query> ──▶ Observe (fetch + append context) ──┐
        ▲                                                          │
        └──────────────────────── loop ◀──────────────────────────┘
        │  (enough context?)
        ▼
   FINAL ──▶ grounded answer + confidence
        │
        ├──▶ ChromaDB (vectors, cosine)
        └──▶ Anthropic API  (one llama_agent; context grows with each turn and is cached)
```

The loop runs on a single system prompt with a cached context block that grows as it searches, so each turn (including the final synthesis) reads the previous turn's context from cache. A **diminishing-returns guard** stops searching once a SEARCH stops surfacing new chunks, holding the loop to 1–4 LLM calls instead of burning the whole search budget.

### Modes and pipeline steps

**Three modes via the `AGENT_MODE` flag:**

| Mode | Description | LLM calls | Best for |
|---|---|---|---|
| `custom` (default) | Deterministic 7-step pipeline | 3 | Speed, predictability, full trace |
| `llama_index` | ReAct search→answer loop | 1–4 variable | Complex multi-step reasoning |
| `compare` | Runs both, side-by-side results | 4–7 combined | Pipeline evaluation and tuning |

| Step | Type | Model | Purpose |
|---|---|---|---|
| SafetyGuard | Pipeline function | None | Prompt injection detection |
| PlannerAgent | LLM Agent | claude-haiku-4-5 | Query intent + rewrite |
| Retriever | Pipeline function | all-MiniLM-L6-v2 | Semantic search |
| SimilarityThreshold | Pipeline function | None | Relevance gate |
| ReRanker | Pipeline function | ms-marco-MiniLM-L-6-v2 | Answer relevance scoring |
| ReasonerAgent | LLM Agent | claude-haiku-4-5 | Grounded answer generation (streamed in `custom` mode) |
| ValidatorAgent | LLM Agent | claude-haiku-4-5 | Hallucination check |

> **Model & speed.** The three LLM agents default to **`claude-haiku-4-5`**, we switched from
> `claude-sonnet-4-6` to **improve response time** (Haiku generates ~3× faster). The model stays
> **configurable** via the `ANTHROPIC_MODEL` env var; set it back to `claude-sonnet-4-6` for maximum
> answer quality. In `custom` mode the answer is also **streamed** to the UI token-by-token, so the
> first words appear in ~2–4 s instead of after the whole answer is written.

> **Observability.** Every answer shows its **confidence**, **latency**, **tokens consumed**, and an
> **estimated cost** (per side in `compare` mode), so the speed/quality/cost tradeoff between modes,
> and the savings from prompt caching, are visible at a glance.


---

## Quick Start

**Prerequisites:** Docker and Docker Compose installed. Anthropic API key ([get one free at console.anthropic.com](https://console.anthropic.com)).

```bash
# 1. Clone the repository
git clone https://github.com/builtbyprashant/genai-doc-assistant
cd genai-doc-assistant

# 2. Create your local .env from the committed template. It holds the non-secret
#    config knobs (model, thresholds, chunking…). .env is gitignored; .env.example
#    is checked in so the defaults are always tracked.
#    PowerShell:  Copy-Item .env.example .env
#    bash:        cp .env.example .env

# 3. Provide your API key via your shell environment: it is NOT stored in .env.
#    PowerShell:  setx ANTHROPIC_API_KEY "sk-ant-..."   (new shells): or for this shell:
#                 $env:ANTHROPIC_API_KEY = "sk-ant-..."
#    bash:        export ANTHROPIC_API_KEY=sk-ant-...

# 4. Start the system (API key from your shell, config from .env)
docker compose up --build

# 5. Open in browser
# Streamlit UI:  http://localhost:8501
# API docs:      http://localhost:8000/docs
```

> **`.env` vs the system env:** the Anthropic API key lives **only in your shell/OS
> environment** (Docker Compose reads it from there), never in `.env`. Everything else
> lives in `.env`, edit a value and re-run `docker compose up` to see it take effect.
> A fresh clone with no `.env` still runs on the built-in defaults.

**First use:**
1. Upload one or more documents using the left sidebar
2. Wait for indexing confirmation
3. Type a question in the centre panel
4. See the answer, confidence score, sources, and agent trace

**Stop the system:**
```bash
docker compose down          # stop containers, keep data
docker compose down -v       # stop containers, DELETE all indexed data
```

> ⚠️ `docker compose down -v` permanently deletes the ChromaDB volume. All indexed documents must be re-uploaded.

---

## Supported Document Formats

Each format is parsed by a dedicated library (pypdf, python-docx, pandas, pyyaml, …); validation and rejection (scanned/password PDFs, duplicates, oversized files) happen on the raw bytes *before* parsing:

| Format | Extension | Notes |
|---|---|---|
| PDF | .pdf | Text-layer only. Scanned/image PDFs rejected with clear error. |
| Text | .txt | UTF-8 and common encodings auto-detected. |
| Markdown | .md | Parsed as plain text. Formatting stripped. |
| CSV | .csv | Rows converted to text. Headers auto-generated if missing. |
| Excel | .xlsx, .xls | All sheets indexed. Charts and images ignored. |
| JSON | .json | Keys and values converted to readable text. |
| YAML | .yaml, .yml | Parsed to key-value text. Config files supported. |
| Word | .docx | Text extracted from paragraphs and tables. |

---

## How It Works

### Document Ingestion
1. File uploaded via Streamlit UI or API
2. Raw bytes validated (MIME type, size, format) before any parsing
3. Duplicate detection via SHA-256 content hash, same content under a different filename is rejected
4. Text extracted and split into 200-word overlapping chunks (25-word overlap)
5. Each chunk embedded using `all-MiniLM-L6-v2` (384-dimensional vectors, cosine space)
6. Vectors and metadata stored in ChromaDB (persisted to disk via Docker volume)

### Question Answering

Answer generation runs in one of **two modes** (chosen per request via the `AGENT_MODE` flag), plus a `compare` mode that runs both side by side.

#### `custom` mode: deterministic 7-step pipeline (3 LLM calls)
1. **SafetyGuard**, checks for prompt injection patterns (rule-based, no LLM call)
2. **PlannerAgent**, analyses query intent, rewrites for semantic density (LLM call 1)
3. **Retriever**, cosine similarity search in ChromaDB, top-10 chunks (no LLM call)
4. **SimilarityThreshold**, rejects if best match scores below 0.3 (cosine), short-circuits pipeline
5. **ReRanker**, cross-encoder reranks chunks by answer relevance, selects top-5 (no LLM call)
6. **ReasonerAgent**, generates grounded answer using only retrieved context (LLM call 2); the answer is **streamed** to the UI token-by-token via `POST /query/stream`
7. **ValidatorAgent**, independent hallucination risk check with fresh context (LLM call 3)

Every step is fixed and fully traced, so the same query always runs the same path.

#### `llama_index` mode: agentic ReAct loop (1–4 LLM calls)
The `llama_agent` runs a **Think → Search → Observe** loop on a single system prompt, built from scratch in python on the Anthropic API (no LlamaIndex library). Each turn it emits one action: `SEARCH: <query>` to fetch more context (appended to a growing, cached context block), or `FINAL: <answer>` to finish. It decides dynamically when it has enough context and stops early when a search stops surfacing new chunks (a diminishing-returns guard), so it makes **1–4 LLM calls** instead of a fixed number. There is no separate Validator on this side; the loop self-rates its own confidence. The answer is returned in one batch (it is not token-streamed in this phase).

#### `compare` mode: both pipelines, side by side (4–7 LLM calls)
Runs the `custom` and `llama_index` pipelines on the same query and renders the results side by side: answers, confidence scores, retrieved chunks with similarity scores, LLM call counts, per-step timings, and full agent traces for both. Validation runs on the custom answer (or on the llama_index answer if the custom pipeline short-circuits). This turns the system into an **evaluation workbench**: you can see exactly where the two approaches diverge and compare them on cost, latency, and answer quality.

---

## Design Documentation

This project was designed before any code was written. The core design documentation is in `/docs`:

| Document | Contents |
|---|---|
| [Project Scope](docs/PROJECT_SCOPE.md) | Architecture, tech stack, pipeline design, env vars |
| [Requirements and Assumptions](docs/REQUIREMENTS_AND_ASSUMPTIONS.md) | 40+ edge cases with acceptable behaviours, design decisions |
| [API Contract](docs/API_CONTRACT.md) | All endpoints, request/response shapes, error types |
| [UI Specification](docs/UI_SPECIFICATION.md) | Every screen state, component behaviour, session state |
| [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) | Build order, per-task file scope, test mapping, all mapped to decisions |
| [Decision Log](docs/DecisionLogs.md) | Key design decisions, conflicts resolved, and rationale |

> Design-first development was a core principle of this project: the architecture, API contracts, requirements, and edge cases were all documented and reviewed before any code was written. The decision log captures the key decisions and their rationale.

---

## Configuration

`.env` (copied from the committed `.env.example`) holds all the **non-secret** config.
Edit a value and re-run `docker compose up` to apply it. The `ANTHROPIC_API_KEY` is **not**
in this file, it is read from your shell/OS environment (see Quick Start).

```bash
# ANTHROPIC_API_KEY: set in your shell/OS env, NOT in .env:
#   setx ANTHROPIC_API_KEY "sk-ant-..."   (key from https://console.anthropic.com)

# Agent mode (default: custom)
# custom      → 7-step pipeline, 3 LLM calls, predictable
# llama_index → ReAct search→answer loop, variable LLM calls
# compare     → both modes, side-by-side results for evaluation
AGENT_MODE=custom

# Key defaults (all configurable)
# Default is Haiku for faster responses; set to claude-sonnet-4-6 for max answer quality.
ANTHROPIC_MODEL=claude-haiku-4-5
ANTHROPIC_CACHE_MODE=block    # prompt caching: block (default) | prompt | off, see "Prompt caching"
EMBEDDING_MODEL=all-MiniLM-L6-v2                     # 384-dim, see "Swappable models"
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 # any cross-encoder
SIMILARITY_THRESHOLD=0.3      # below this (cosine) = no answer returned
TOP_K_RETRIEVAL=10            # chunks fetched from ChromaDB
TOP_K_RERANK=5                # chunks passed to Reasoner after ranking
CHUNK_SIZE=200                # words per chunk
MAX_DOCUMENTS=20              # max documents in index
LLM_TIMEOUT_SECONDS=20       # per LLM call, retries at half timeout (10s)
```

See `.env.example` for all options with descriptions.

### Prompt caching

`ANTHROPIC_CACHE_MODE` (`block` default | `prompt` | `off`) toggles Anthropic prompt caching. The
thing actually cached is the **retrieved CONTEXT block**, not the system prompt: within one query
the **Reasoner and Validator process the same context**, so the Reasoner writes it to cache and the
Validator reads it instead of re-processing it. The context is placed in its own `cache_control`
block at the head of the user message, with the agent-specific task trailing it uncached; both
agents share a grounding system so the cached prefix matches. In `llama_index` mode the whole
ReAct loop runs on one system and the cached `question + context` block **grows with each turn**, so
each turn (including the final synthesis) reads the previous turn's context from cache.

> **Why not cache the system prompts?** They're ~100 tokens each, below the model's minimum
> cacheable length (~2048 for Haiku, ~1024 for Sonnet/Opus), so `cache_control` on them is silently
> ignored. Caching only pays off on the large CONTEXT, and only when it clears that floor (bigger
> documents / more chunks, or a Sonnet/Opus model). Verify with the logs below, `off` disables it.

Cache effectiveness is **observable in the logs**: every LLM call emits an `llm_usage` event with `cache_read_input_tokens` (served from cache) and `cache_creation_input_tokens` (written to cache):

```bash
docker compose logs backend | grep llm_usage
# {"event":"llm_usage","agent":"ReasonerAgent","cache_mode":"block",
#  "input_tokens":120,"output_tokens":30,"cache_read_input_tokens":95,"cache_creation_input_tokens":0}
```

### Swappable models

Both local models are chosen via env vars (`EMBEDDING_MODEL`, `RERANKER_MODEL`), no code change needed. They are independent of each other, but the embedding model is constrained by the vector index.

**Embedding model (bi-encoder)**, produces the vectors stored in ChromaDB.
> ⚠️ Must output **384-dimensional** vectors to match the existing index, and changing it means **deleting the index and re-uploading** all documents (old and new vectors must come from the same model). Mind the token limit relative to `CHUNK_SIZE` too.

| Model | Dim | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` (default) | 384 | Fast, general English |
| `BAAI/bge-small-en-v1.5` | 384 | Stronger on retrieval benchmarks |
| `intfloat/e5-small-v2` | 384 | Good multilingual-ish |
| `thenlper/gte-small` | 384 | Competitive small model |

**Reranker (cross-encoder)**, scores `(query, chunk)` pairs after retrieval.
> No dimension constraint, it outputs a relevance score, not a vector. **Drop-in swappable**, no re-indexing.

| Model | Notes |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (default) | Fast, good quality |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Larger, more accurate, slower |
| `BAAI/bge-reranker-base` | Strong alternative |

The embedding model is *plugged into* ChromaDB as its embedding function, it is not ChromaDB-specific. A different vector backend (a future version) would use the same model. The only coupling is the 384-dim/re-index rule above, which is true of any vector store.

---

## API

The FastAPI backend exposes a REST API with auto-generated documentation at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status and stats |
| POST | `/documents/upload` | Upload one or more documents (207 multi-status) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Remove a document |
| POST | `/query` | Ask a question (batch, full JSON response) |
| POST | `/query/stream` | Ask a question, **streamed** (`custom` mode streams the answer token-by-token, then a JSON metadata frame) |

**Example query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the ICU transfer criteria?",
    "agent_mode": "custom",
    "include_chunks": true,
    "include_trace": true
  }'
```

**Example compare mode:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarise all transfer requirements",
    "agent_mode": "compare"
  }'
```

**Example streamed query (`custom` mode):**
```bash
# Streams the answer text, then a 0x1E byte, then a JSON metadata frame
# (confidence, sources, validation, trace). Use -N to disable curl buffering.
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the ICU transfer criteria?", "agent_mode": "custom"}'
```

---

## Running Tests

```bash
# Inside Docker
docker compose exec backend pytest tests/ -v

# Local venv
pytest tests/ -v

# Specific service
pytest tests/test_vector_store.py -v
pytest tests/test_api.py -v
```

Tests cover all services, all API endpoints, edge cases from requirements, and pipeline short-circuit behaviour.

---

## Industry-Standard Best Practices

The project follows production-grade engineering practices end to end.

### Continuous integration and delivery (CI/CD)
- **GitHub Actions** runs on every push and every pull request (status is shown by the build badge at the top of this README).
- **Lint and test on every change.** Linting (`ruff`) and the full test suite run on all pushes and PRs, for fast feedback.
- **Coverage on every run.** Tests run under coverage (`pytest --cov=app`), and a coverage report is posted to the workflow run summary so coverage is visible per run.
- **Images built only when they can break.** The Docker image build is path-gated: it runs only on a push to `main` that changed an image-affecting file (a requirements file or a Dockerfile), so everyday commits run lint and test only. Build-layer caching keeps warm builds short.
- **Cache keep-warm.** A scheduled job refreshes the dependency and model-download caches so a low-activity repository does not go cold.

### Test-driven development
- **Tests are the spec.** Each service's tests were written before or alongside its implementation and ship in the same commit as the code.
- **105 tests across five suites:** vector store, chunking, document loading, the agent pipeline, and the API.
- **Coverage targets the real risks:** cosine similarity and threshold behaviour, chunk sizing and routing, all eight document formats (including scanned and password-protected PDF rejection and duplicate detection), every API endpoint (including multi-status upload and the compare envelope), and pipeline short-circuits.

### Code quality and coverage
- **Linting.** `ruff` enforces a consistent style (Python 3.11 target, 110-character lines).
- **Coverage measurement.** `pytest-cov` measures coverage of the `app/` package on every CI run, reported with missing-line detail and an XML artifact.
- **Small, reviewable changes.** One commit per task in Conventional Commits format, so history maps cleanly to the design.

### Security practices
- **No secrets in the repository.** The Anthropic API key is read only from the host/OS environment; it is never written to `.env` and never committed. `.env` is gitignored, and only `.env.example` (non-secret defaults) is tracked.
- **No secrets in git history.** A final verification step checks the full git history for any `.env` or key material before release.
- **Configuration, not hardcoding.** All configuration comes from environment variables; there are no hardcoded secrets, URLs, or paths.
- **Prompt-injection guard.** A rule-based SafetyGuard screens input before any model call.
- **PII-safe logging.** Raw user queries are never logged; each log line carries only a short SHA-256 hash for correlation.
- **Grounded answers.** Generation is constrained to the retrieved context, with an independent Validator hallucination check, which reduces fabrication.
- **Validated input and structured errors.** Uploaded files are validated on their raw bytes before any parser runs, and errors are returned as structured RFC-7807 problem details.

---

## Limitations

- Single-turn Q&A only (no conversation history), planned for a future version
- Maximum 20 documents per index (configurable via `MAX_DOCUMENTS`)
- PDF must have a text layer, scanned/image PDFs not supported
- Original uploaded files are not persisted, only chunks stored in ChromaDB
- Embedding model optimised for English, other languages produce lower quality
- No authentication or access control, planned for a future version
- Single user assumed, concurrent writes not guaranteed safe at scale
- Query response time ~2–4 seconds typical in `custom` mode with the Haiku default; the answer streams so the first words appear in ~2–4 s. Switching `ANTHROPIC_MODEL` to `claude-sonnet-4-6` raises quality but roughly triples generation time

See Requirements and Assumptions in `/docs` for the full list with design rationale.

---

## Roadmap (Future Version)

- AWS deployment via Terraform (ECS, S3, ElastiCache)
- Grafana observability via CloudWatch + Loki
- Retryable vs non-retryable error classification
- PII redaction in logs (AWS Comprehend / Presidio)
- Multi-turn conversation (DynamoDB session history)
- Vector store abstraction (swap ChromaDB → Pinecone via env var)
- Row-aware chunking for CSV/Excel and support for advanced pdfs
- Authentication (API key middleware)
- Maximum 2 LLM calls per query (deterministic grounding check replaces ValidatorAgent) in both the modes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| Backend API | FastAPI (Python) |
| Vector database | ChromaDB (embedded, cosine space, persisted) |
| Embedding model | all-MiniLM-L6-v2 (sentence-transformers, 384 dimensions) |
| Re-ranking model | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) |
| LLM | Claude `claude-haiku-4-5` via Anthropic API (default, for speed; configurable via `ANTHROPIC_MODEL`, e.g. `claude-sonnet-4-6` for max quality) |
| Document parsing | Per-format libraries, pypdf, python-docx, pandas, openpyxl, pyyaml, chardet (8 formats) |
| llama_index mode | Lightweight ReAct loop built using python |
| Testing | pytest + httpx |
| Deployment | Docker Compose (primary) + Python venv (local dev) |

---

## Project Structure

```
genai-doc-assistant/
├── app/
│   ├── api/main.py              ← FastAPI routes
│   ├── agents/                  ← pipeline.py, safety, planner, ranker,
│   │                               reasoner, validator, llama_agent
│   ├── services/                ← storage, document_loader, chunk_service,
│   │                               vector_store
│   ├── core/config.py           ← environment variable loading
│   └── utils/logging.py         ← structured JSON logging + query hashing
├── frontend/app.py              ← Streamlit UI
├── tests/                       ← pytest test suite
├── data/                        ← uploaded documents (gitignored)
├── docs/                        ← design documentation (published after cleanup)
├── requirements.txt             ← backend runtime
├── requirements-frontend.txt    ← Streamlit UI deps (no ML stack)
├── requirements-dev.txt         ← backend + test toolchain
├── docker-compose.yml
├── Dockerfile.backend           ← CPU-only torch; pre-downloads encoders
├── Dockerfile.frontend          ← streamlit + httpx only
├── .env.example
├── .gitignore
├── setup_venv.sh
└── README.md
```

---

## Acknowledgements

This project stands on excellent open tools and services:

- **[Anthropic Claude](https://www.anthropic.com/claude)**: the LLM behind the Planner, Reasoner, Validator agents in custom mode, and llama_agent in llama_index mode. Interaction with Anthropic LLMs was built using Anthropic API.
- **[LlamaIndex](https://www.llamaindex.ai/)**: inspiration for the `llama_index` mode's ReAct loop. The library itself is not a dependency; the pattern is reimplemented from scratch using python.
- **[Hugging Face](https://huggingface.co/) sentence-transformers**: the `all-MiniLM-L6-v2` embeddings and the `ms-marco-MiniLM-L-6-v2` cross-encoder reranker, run locally on CPU.
- **[ChromaDB](https://www.trychroma.com/)**: the persistent vector store (cosine similarity).
- **[FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/)**: the backend API and the UI.
- **[Docker](https://www.docker.com/) and [GitHub Actions](https://github.com/features/actions)**: containerized deployment and CI (lint, tests, image build).

---

## Author

It's [@builtbyprashant](https://github.com/builtbyprashant) as a learning project exploring production-grade Generative AI engineering, RAG pipelines, agentic systems, vector databases, CI/CD, and Docker deployment.

Design-first methodology was adopted, wherein system architecture, API contracts, requirements, edge cases, CI pipeline, and deployment strategy were fully specified prior to implementation. Code was developed collaboratively with Claude Code under continuous and rigorous author oversight, encompassing design decisions, technical details, output validation, and test review.

---

## License

Released under the [MIT License](LICENSE).
