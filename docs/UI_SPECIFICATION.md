# Agentic RAG Knowledge System: UI Specification (Phase 1)

## Overview

Single-page Streamlit application. No navigation, no tabs, no page transitions.
The entire user journey, document management, question asking, answer reading, pipeline
inspection, happens on one screen.

Layout: native Streamlit sidebar (collapsible) + two-column main area.

---

## Layout Structure

```
◀ ┌─────────────────────┐  ┌──────────────────────────┬──────────────────────┐
   │   SIDEBAR           │  │   CENTRE  (ratio: 2)     │   RIGHT  (ratio: 1)  │
   │   Document Manager  │  │   Q&A                    │   Pipeline Detail    │
   │   (native, auto-    │  │                          │                      │
   │    collapsible)     │  │                          │                      │
   └─────────────────────┘  └──────────────────────────┴──────────────────────┘
```

Streamlit column ratio: `st.columns([2, 1])` inside the main area.
Sidebar uses `st.sidebar`, collapse arrow provided natively by Streamlit, no custom code.

---

## Sidebar: Document Manager

Always accessible. Collapses to a thin sliver with a `▶` arrow to restore.

### Section 1: System Status

Displayed at the very top of the sidebar. Calls `GET /health` on page load and refreshes
every 30 seconds via `st.rerun()` with a session state timer.

```
─────────────────────────
🟢 System online
─────────────────────────
Documents   5 / 20
Chunks      342
Reranker    cross-encoder
Threshold   0.40
─────────────────────────
```

| Element | Detail |
|---|---|
| Status indicator | 🟢 green dot if `status: ok`, 🔴 red if `status: degraded` |
| Documents | `documents_indexed / max_documents`, shows capacity at a glance |
| Chunks | `total_chunks` from health response |
| Reranker | Static text: `cross-encoder` (always ms-marco-MiniLM-L-6-v2, no mode toggle) |
| Threshold | Active `similarity_threshold` value |

When documents reach max capacity, the counter turns red: `20 / 20 ⚠️`

---

### Section 2: Upload Documents

```
─────────────────────────
Upload documents
─────────────────────────
[ Drag and drop files here ]
[ or Browse files          ]
  Supported: PDF, TXT, MD, CSV, XLSX, JSON, YAML, DOCX

[ Upload ]
─────────────────────────
```

| Element | Behaviour |
|---|---|
| `st.file_uploader` | `accept_multiple_files=True`. Accepts .pdf, .txt, .md, .csv, .xlsx, .xls, .json, .yaml, .yml, .docx |
| Upload button | Disabled if no files selected. Disabled if at MAX_DOCUMENTS capacity. |
| On click | Calls `POST /documents/upload` with all selected files. Shows spinner while in progress. |
| On success | Shows per-file result inline below the uploader. Green ✓ for success, red ✗ for failure. Refreshes document list automatically. Clears the file uploader. |
| Warnings | Shown in amber below the success indicator (e.g. "No headers detected") |
| On failure | Shows the RFC 7807 `detail` field in a red `st.error()` box. Never shows raw JSON. |
| At capacity | Upload button replaced with: `⚠️ Document limit reached (20/20). Delete a document to upload more.` |

**Upload result display (inline, below uploader):**
```
✓ report.pdf, 87 chunks indexed
✓ data.csv, 34 chunks indexed  ⚠ No headers detected
✗ scan.pdf, Scanned PDF. Text could not be extracted.
```

---

### Section 3: Indexed Documents

Fixed header, scrollable list below. Calls `GET /documents` on page load and after
every upload or delete action.

```
─────────────────────────
Indexed documents  (5)
─────────────────────────
┌────────────────────────┐  ← st.container(height=300)
│ 📄 report.pdf      🗑  │  ↑
│    87 chunks           │  │
│ 📄 policy.txt      🗑  │  │  scrollable
│    34 chunks           │  │
│ 📄 data.csv        🗑  │  │
│    22 chunks           │  │
│ 📄 q3-results.xlsx 🗑  │  │
│    61 chunks           │  ↓
└────────────────────────┘
```

| Element | Behaviour |
|---|---|
| Container | `st.container(height=300, border=True)`, fixed height, scrolls when list exceeds it |
| Each document row | Filename + chunk count + delete button (🗑) on the same row |
| Delete button | On click: confirmation prompt inline (`"Delete report.pdf? This cannot be undone."` with Yes / Cancel). On confirm: calls `DELETE /documents/{filename}`. Refreshes list on success. |
| Empty state | `No documents indexed yet. Upload a document above to get started.` |
| Document count | Shown in the section header: `Indexed documents (5)` |

---

## Main Area: Centre Column (Q&A)

The primary user interaction surface. Top to bottom: input → filters → submit → answer.

### Question Input

```
Ask a question
──────────────────────────────────────────────────────
[ Type your question here...                          ]
──────────────────────────────────────────────────────
Search in:   [ All documents ▾ ]  (multiselect)
Agent mode:  ● Custom  ○ llama_index  ○ Compare
                                          [ Ask ]
```

| Element | Behaviour |
|---|---|
| `st.text_area` | Label: "Ask a question". Placeholder: "Type your question here...". Height: 100px. Max chars: 2000. |
| Character counter | Shown below text area: `142 / 2000 characters`. Turns red above 1800. |
| Doc filter | `st.multiselect` populated from `GET /documents` filenames. Label: "Search in". Default: empty (searches all). |
| Agent mode selector | `st.radio` with three options: Custom (default), llama_index, Compare. Custom calls `POST /query/stream`; llama_index/Compare call `POST /query`. |
| Ask button | Primary button. Disabled when: question is empty, under 3 chars, no documents indexed, query in progress. |
| Answer streaming (custom) | In custom mode the answer is rendered live with `st.write_stream` as it arrives from `POST /query/stream`, first words appear in ~2–4 s. Once complete, a rerun shows the final answer + confidence + trace (single render). |
| Spinner | `st.spinner("Running agent pipeline...")` shown while a llama_index/Compare (batch) query is in progress. Custom mode streams instead of showing the spinner. |
| Client-side validation | Empty question shows inline warning before API call. |

**Compare mode layout:** When AGENT_MODE=compare, the centre column splits into two sub-columns:

```
┌──────────────────────────────────┬──────────────────────────────────┐
│  Custom pipeline                 │  llama_index loop                │
│  ─────────────────               │  ─────────────────────           │
│  Confidence 🟢 High · 3 LLM      │  Confidence 🟢 High · 4 LLM      │
│  calls · 2.4s · 2,847 tokens ·   │  calls · 6.1s · 21,686 tokens ·  │
│  ~$0.0041                        │  ~$0.0228                        │
│  Answer: ...                     │  Answer: ...                     │
│  Sources: [list]                 │  Sources: [list]                 │
└──────────────────────────────────┴──────────────────────────────────┘
```

- Each side shows tokens + estimated cost (D-14); a short-circuited side shows its **reason** inline (e.g. "scored 0.39, below 0.30") instead of a bald "(no answer)".
- Below the answers, a single **`Validation (<pipeline>): …`** line names whose answer the Validator ran on, it is a custom-pipeline agent (D-5), so it validates the custom answer (or the llama answer if custom short-circuited); llama otherwise **self-rates its own confidence**.
- Then **both pipelines' trace + retrieved chunks** render **side by side** (not just custom's): custom's fixed 7 named steps vs the llama ReAct loop's action-labelled steps (`LlamaReAct · Search` / `· Synthesize`), so you can compare how each reached its answer.

---

### Answer Display

Shown below the input section after a successful query. Hidden on page load.

**Successful answer:**
```
──────────────────────────────────────────────────────
Answer                              ⏱ 5.2s
──────────────────────────────────────────────────────
According to the ICU guidelines document, patients may
be transferred from ICU when they meet the following
criteria...

Confidence: 🟢 High · ⏱ 5.2s · 🔢 2,847 tokens · ~$0.0041 · Sources: icu-guidelines.txt, policy.pdf
──────────────────────────────────────────────────────
```

| Element | Behaviour |
|---|---|
| Answer text | Custom mode: streamed live via `st.write_stream`, then re-rendered with `st.markdown()` on completion. llama_index/Compare: `st.markdown()` once the batch response returns. Supports bullet points and formatting from the Reasoner. |
| Processing time | Shown in the caption: `⏱ 5.2s` from `processing_time_ms` |
| Confidence | 🟢 High / 🟡 Medium / 🔴 Low with the `confidence_reason` shown as a tooltip on hover |
| Tokens + cost | `🔢 2,847 tokens · ~$0.0041` from `tokens` / `cost_usd` (D-14); shown per side in compare. Prompt-cache hits (D-12) visibly lower the cost. |
| Sources | Comma-separated list of `sources_used` filenames |
| Validation warning | If `hallucination_risk` is `"high"` or `suggested_action` is `"review"`: amber warning box below answer: `⚠️ This answer may require verification. The validator flagged potential accuracy concerns.` |

**Short-circuit response (threshold not met):**
```
──────────────────────────────────────────────────────
ℹ️  No relevant content found
──────────────────────────────────────────────────────
The question could not be answered from the indexed
documents. The most similar content scored 0.18,
below the minimum threshold of 0.40.

Try: rephrasing your question, or uploading documents
     that contain relevant information.
──────────────────────────────────────────────────────
```

**Short-circuit response (no documents):**
```
──────────────────────────────────────────────────────
ℹ️  No documents indexed
──────────────────────────────────────────────────────
Please upload at least one document using the panel
on the left before asking questions.
──────────────────────────────────────────────────────
```

**API error response (503, 500):**
```
──────────────────────────────────────────────────────
⚠️  Something went wrong
──────────────────────────────────────────────────────
The service encountered an error while processing
your question. Please try again in a moment.

If the problem persists, check the agent trace on
the right for details.
──────────────────────────────────────────────────────
```

Raw HTTP errors, stack traces, and JSON bodies are never shown in the centre column.
The right column trace section surfaces detailed error info for technical users.

---

## Main Area: Right Column (Pipeline Detail)

Always visible alongside the centre column. Shows pipeline detail after a query runs.
Empty state before first query.

### Empty State (before first query)

```
Pipeline detail
──────────────────────────
Run a query to see the
agent pipeline trace and
retrieved document chunks.
```

### After Query: Agent Trace

```
Pipeline detail
──────────────────────────
▼ Agent trace       5.2s

  ✓ SafetyGuard          2ms
  ✓ PlannerAgent       890ms
  ✓ Retriever     120ms
  ✓ SimilarityThreshold  1ms
  ✓ ReRanker       1340ms
  ✓ ReasonerAgent     2100ms
  ✓ ValidatorAgent     787ms
```

Each agent row is expandable, click to see the `details` object for that agent:

```
▼ PlannerAgent             890ms
  Intent:    Find ICU transfer policy
  Strategy:  targeted
  Rewritten: ICU transfer criteria patient
             discharge hemodynamic stability
```

| Element | Behaviour |
|---|---|
| Section header | "Agent trace" with total pipeline time on the right |
| Agent rows | One row per agent. Icon: ✓ green (passed/completed), ✗ red (failed), ⊘ gray (skipped) |
| Expandable detail | `st.expander` per agent showing the `details` field formatted as key-value pairs |
| Short-circuit indicator | If `short_circuit: true`, pipeline stops visually at the agent that triggered it. Remaining agents shown grayed out as "Not reached". |
| LLM failure | Failed agent shown in red with the `detail` message from the RFC 7807 error. |

---

### After Query: Retrieved Chunks

Shown below the agent trace in the same right column.
Only populated when `include_chunks: true` (always set to true by the Streamlit UI).

```
──────────────────────────
▼ Retrieved chunks  (5)

  1. icu-guidelines.txt
     Similarity  0.87  Rerank  0.94
     ▶ Patients may be transferred
       from ICU when hemodynamic...

  2. policy.pdf
     Similarity  0.81  Rerank  0.88
     ▶ ICU transfer criteria include:
       attending physician sign-off...

  3. policy.pdf
     Similarity  0.74  Rerank  0.76
     ▶ The transfer checklist must be
       completed and signed by...
```

| Element | Behaviour |
|---|---|
| Section header | "Retrieved chunks (N)" where N is the count |
| Each chunk | Filename, similarity score, rerank score, first 100 chars of text |
| Expand | `st.expander` per chunk to show full chunk text |
| Score colours | Similarity/rerank score: green ≥ 0.75, amber 0.5–0.74, red < 0.5 |
| Empty state (short-circuit) | "No chunks retrieved, pipeline stopped before retrieval." or "No chunks cleared the similarity threshold." depending on `short_circuit_reason` |

---

## Page-Level Settings

Set once via `st.set_page_config()` at the top of `app.py`:

| Setting | Value |
|---|---|
| Page title | `"RAG Knowledge System"` |
| Page icon | `"🔍"` |
| Layout | `"wide"`, uses full browser width |
| Initial sidebar state | `"expanded"`, sidebar open on first load |
| Menu items | Hide default Streamlit menu (clean demo appearance) |

---

## Session State Variables

**Why session state exists:**
Streamlit reruns the entire Python script from top to bottom on every user interaction, every button click, every file upload, every text input change. Without session state, all variables reset on every rerun. The document list would disappear when the user types a question. The last query result would vanish when the user clicks delete. Session state persists specific values across these reruns for the lifetime of the browser tab.

```python
# Initialise once on first load
if "documents" not in st.session_state:
    st.session_state.documents = []

# Update after upload or delete
st.session_state.documents = fetch_documents()

# Read anywhere in the script: survives reruns
for doc in st.session_state.documents:
    st.write(doc["filename"])
```

Session state is per browser tab, each tab has its own isolated state. The backend document index is shared across all tabs.

Streamlit session state keys used across the app:

| Key | Type | Purpose |
|---|---|---|
| `documents` | list | Cached result of GET /documents. Refreshed on upload/delete. |
| `last_query_result` | dict or None | Last response from `POST /query` (batch) or the metadata frame from `POST /query/stream`. Drives answer + trace display. |
| `pending_stream` | dict or None | Set when a custom-mode question is submitted; holds the request so the answer is streamed via `st.write_stream` on the next render, then cleared. |
| `query_in_progress` | bool | True while a batch query is running. Disables Ask button. |
| `GET /health` polling | (not session state) | Throttled with `@st.cache_data(ttl=30)` on `fetch_health()`, at most one `/health` call per 30s regardless of reruns, matching the Docker healthcheck interval. `fetch_health.clear()` is called after upload/delete so the doc/chunk stats refresh immediately. |

---

## Error Display Rules

These rules apply everywhere in the UI without exception:

1. Never show raw JSON to the user.
2. Never show Python exception tracebacks.
3. Never show HTTP status codes in the main answer area.
4. Always show the RFC 7807 `detail` field, it is written to be human-readable.
5. Technical detail (agent name, HTTP status, request_id) goes in the right column trace only.
6. Every error message ends with what the user can do next.
7. The `request_id` is shown at the bottom of the trace section for support reference.

---

## Responsive Behaviour

Streamlit `layout="wide"` fills the browser window. The column ratio `[2, 1]` means:
- Centre column: approximately 65% of main area width
- Right column: approximately 35% of main area width

On narrow screens (laptop at 1280px), the sidebar auto-collapses to give maximum space
to the main content. Users on narrow screens should collapse the sidebar for best experience.
No mobile optimisation in Phase 1, document as a known limitation.

---

## UI Limitations (Phase 1)

| Limitation | Notes |
|---|---|
| Streaming is custom-mode only | Custom mode streams the answer live (first words ~2–4s). llama_index and Compare still appear all at once after full generation, with the spinner shown during the wait. Full llama/compare streaming is Phase 2. |
| No conversation history | Each question is independent. No chat history displayed. Phase 2. |
| No mobile optimisation | Designed for desktop browser at ≥ 1280px width. |
| No dark mode toggle | Follows system/browser preference via Streamlit default. |
| Sidebar collapse on mobile | Streamlit auto-collapses sidebar on narrow viewports. |
| No drag-and-drop reordering | Document list is ordered by index time, not user-defined. |
