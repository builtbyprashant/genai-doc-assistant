# Agentic RAG Knowledge System

> Upload your documents. Ask questions in plain English. Get grounded, cited answers powered by a multi-agent AI pipeline.

---

## What It Is

An AI-powered document intelligence system that allows users to upload enterprise documents in multiple formats and ask natural language questions. The system retrieves relevant content using semantic search and generates grounded, cited answers using a 7-step agent pipeline built on Claude.

**The problem it solves:** Knowledge locked in documents — PDFs, spreadsheets, reports, markdown files — is hard to query. This system makes any document collection conversational. Upload your documents, ask questions in plain English, get answers with source citations.

**What makes it agentic:** Rather than a single LLM call, the system uses three specialised LLM agents (Planner, Reasoner, Validator) that collaborate in sequence, each with a focused role. Four non-LLM pipeline steps handle safety, retrieval, threshold checking, and cross-encoder re-ranking without wasting API calls.

---

## Architecture

<!-- Architecture diagram will be added after build is complete -->

**7-step query pipeline — three modes via AGENT_MODE flag:**

| Mode | Description | LLM calls | Best for |
|---|---|---|---|
| `custom` (default) | Sequential 7-step pipeline | 3 fixed | Speed, predictability, full trace |
| `llama_index` | ReAct search→answer loop (Anthropic SDK) | 1–4 variable | Complex multi-step reasoning |
| `compare` | Runs both, side-by-side results | 4–7 combined | Pipeline evaluation and tuning |

| Step | Type | Model | Purpose |
|---|---|---|---|
| SafetyGuard | Pipeline function | None | Prompt injection detection |
| PlannerAgent | LLM Agent | claude-sonnet-4-6 | Query intent + rewrite |
| RetrieverAgent | Pipeline function | all-MiniLM-L6-v2 | Semantic search |
| SimilarityThreshold | Pipeline function | None | Relevance gate |
| RankerAgent | Pipeline function | ms-marco-MiniLM-L-6-v2 | Answer relevance scoring |
| ReasonerAgent | LLM Agent | claude-sonnet-4-6 | Grounded answer generation |
| ValidatorAgent | LLM Agent | claude-sonnet-4-6 | Hallucination check |

---

## Quick Start

**Prerequisites:** Docker and Docker Compose installed. Anthropic API key ([get one free at console.anthropic.com](https://console.anthropic.com)).

```bash
# 1. Clone the repository
git clone https://github.com/builtbyprashant/genai-doc-assistant
cd genai-doc-assistant

# 2. Configure environment
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY

# 3. Start the system
docker compose up --build

# 4. Open in browser
# Streamlit UI:  http://localhost:8501
# API docs:      http://localhost:8000/docs
```

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
3. Duplicate detection via SHA-256 content hash — same content under a different filename is rejected
4. Text extracted and split into 200-word overlapping chunks (25-word overlap)
5. Each chunk embedded using `all-MiniLM-L6-v2` (384-dimensional vectors, cosine space)
6. Vectors and metadata stored in ChromaDB (persisted to disk via Docker volume)

### Question Answering
1. **SafetyGuard** — checks for prompt injection patterns (rule-based, no LLM call)
2. **PlannerAgent** — analyses query intent, rewrites for semantic density (LLM call 1)
3. **RetrieverAgent** — cosine similarity search in ChromaDB, top-10 chunks (no LLM call)
4. **SimilarityThreshold** — rejects if best match scores below 0.4, short-circuits pipeline
5. **RankerAgent** — cross-encoder reranks chunks by answer relevance, selects top-5 (no LLM call)
6. **ReasonerAgent** — generates grounded answer using only retrieved context (LLM call 2)
7. **ValidatorAgent** — independent hallucination risk check with fresh context (LLM call 3)

---

## Design Documentation

This project was designed before any code was written. Full documentation is available in `/docs`:

| Document | Contents |
|---|---|
| Phase 1 Project Scope | Architecture, tech stack, pipeline design, env vars |
| Requirements and Assumptions | 40+ edge cases with acceptable behaviours, design decisions |
| API Contract | All endpoints, request/response shapes, error types |
| UI Specification | Every screen state, component behaviour, session state |
| Phase 2 Scope | AWS deployment, Grafana observability, production optimisations |
| Decision Log | Single source of truth — every design decision, rationale, and status |
| Implementation Plan | Build order, per-task file scope, test mapping — all mapped to decisions |

> Note: The `/docs` folder will be published after cleanup. Design-first development was a core principle of this project — all architecture, API contracts, and edge cases were documented and reviewed before any code was written.

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required — get your key at https://console.anthropic.com
ANTHROPIC_API_KEY=sk-ant-...

# Agent mode (default: custom)
# custom      → 7-step pipeline, 3 LLM calls, predictable
# llama_index → LlamaIndex ReActAgent, variable LLM calls
# compare     → both modes, side-by-side results for evaluation
AGENT_MODE=custom

# Key defaults (all configurable)
ANTHROPIC_MODEL=claude-sonnet-4-6
SIMILARITY_THRESHOLD=0.4      # below this = no answer returned
TOP_K_RETRIEVAL=10            # chunks fetched from ChromaDB
TOP_K_RERANK=5                # chunks passed to Reasoner after ranking
CHUNK_SIZE=200                # words per chunk
MAX_DOCUMENTS=20              # max documents in index
LLM_TIMEOUT_SECONDS=20       # per LLM call, retries at half timeout (10s)
```

See `.env.example` for all options with descriptions.

---

## API

The FastAPI backend exposes a REST API with auto-generated documentation at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status and stats |
| POST | `/documents/upload` | Upload one or more documents (207 multi-status) |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Remove a document |
| POST | `/query` | Ask a question |

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

## Limitations

- Single-turn Q&A only (no conversation history) — Phase 2
- Maximum 20 documents per index (configurable via `MAX_DOCUMENTS`)
- PDF must have a text layer — scanned/image PDFs not supported
- Original uploaded files are not persisted — only chunks stored in ChromaDB
- Embedding model optimised for English — other languages produce lower quality
- No authentication or access control — Phase 2
- Single user assumed — concurrent writes not guaranteed safe at scale
- Query response time 2–4 seconds typical (3 LLM calls + 1 encoder call)

See Requirements and Assumptions in `/docs` for the full list with design rationale.

---

## Phase 2 Roadmap

- AWS deployment via Terraform (ECS, S3, ElastiCache)
- Grafana observability via CloudWatch + Loki
- Retryable vs non-retryable error classification
- PII redaction in logs (AWS Comprehend / Presidio)
- Multi-turn conversation (DynamoDB session history)
- Vector store abstraction (swap ChromaDB → Pinecone via env var)
- Row-aware chunking for CSV/Excel
- Authentication (API key middleware)
- Maximum 2 LLM calls per query (deterministic grounding check replaces ValidatorAgent)

See Phase 2 Scope in `/docs` for the full roadmap with implementation details.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| Backend API | FastAPI (Python) |
| Vector database | ChromaDB (embedded, cosine space, persisted) |
| Embedding model | all-MiniLM-L6-v2 (sentence-transformers, 384 dimensions) |
| Re-ranking model | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers) |
| LLM | Claude claude-sonnet-4-6 via Anthropic API |
| Document parsing | Per-format libraries — pypdf, python-docx, pandas, openpyxl, pyyaml, chardet (8 formats) |
| llama_index mode | Lightweight ReAct loop on the Anthropic SDK (full LlamaIndex deferred to Phase 2) |
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
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example
├── .gitignore
├── setup_venv.sh
└── README.md
```

---

## Author

Built by [@builtbyprashant](https://github.com/builtbyprashant) as a learning project exploring production-grade Generative AI engineering — RAG pipelines, agentic systems, vector databases, LlamaIndex, and Docker deployment.

Design-first approach: architecture, API contracts, requirements, and edge cases fully documented before any code was written.

---

## License

Released under the [MIT License](LICENSE).
