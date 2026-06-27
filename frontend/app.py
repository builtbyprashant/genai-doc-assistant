"""Streamlit UI for the Agentic RAG system.

Single page: a sidebar Document Manager + a main Q&A area, with the pipeline trace
and retrieved chunks tucked into collapsed expanders below the answer. The frontend
runs in its own container and only talks to the backend over HTTP — it never imports
the app package, and never shows raw JSON or tracebacks to the user.
"""

from __future__ import annotations

import os
import re

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 90  # a query can take a few seconds end to end

st.set_page_config(page_title="RAG Knowledge System", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

# Hide Streamlit's own chrome — the top header (Deploy button + "running" animation),
# the hamburger menu, and the footer — for a clean app look.
st.markdown(
    """
    <style>
      /* Strip Streamlit's header chrome — but KEEP the header element, because the
         collapsed-sidebar reopen button (stExpandSidebarButton, the ») lives inside
         its toolbar. Hiding the whole header (display:none) left a collapsed sidebar
         with no way to reopen. So: hide the Deploy button, the ⋮ menu, the status
         widget and the rainbow bar; make the header transparent + click-through; and
         keep the reopen button clickable. (Verified against the live DOM.) */
      header[data-testid="stHeader"] {background: transparent !important; box-shadow: none !important; pointer-events: none !important;}
      [data-testid="stToolbar"] [data-testid="stBaseButton-header"] {display: none !important;}  /* Deploy */
      [data-testid="stMainMenuButton"] {display: none !important;}                                /* hamburger menu */
      [data-testid="stStatusWidget"] {display: none !important;}                                  /* "Running..." indicator */
      [data-testid="stDecoration"] {display: none !important;}                                    /* rainbow top bar */
      [data-testid="stExpandSidebarButton"] {pointer-events: auto !important;}                    /* keep the reopen button clickable */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      /* Compact main content + use more horizontal width. */
      .block-container {padding: 0.6rem 2.5rem 1rem 2.5rem !important;}
      [data-testid="stVerticalBlock"] {gap: 0.5rem !important;}   /* gap between stacked elements */
      hr {margin: 0.4rem 0 !important;}                            /* dividers */
      h1 {margin: 0 0 0.2rem 0 !important; font-size: 2rem !important;}
      h2, h3 {margin: 0.4rem 0 0.2rem 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── backend helpers ───────────────────────────────────────────────────────────

def api_get(path: str):
    return httpx.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT)


def api_post(path: str, **kwargs):
    return httpx.post(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


def fetch_documents() -> list[dict]:
    try:
        r = api_get("/documents")
        return r.json().get("documents", []) if r.status_code == 200 else []
    except httpx.HTTPError:
        return []


def detail_of(response: httpx.Response) -> str:
    """The human-readable RFC-7807 detail, never raw JSON."""
    try:
        return response.json().get("detail", "Something went wrong.")
    except ValueError:
        return "The service returned an unexpected response."


# ── session state ─────────────────────────────────────────────────────────────

if "documents" not in st.session_state:
    st.session_state.documents = fetch_documents()
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── sidebar: document manager ─────────────────────────────────────────────────

with st.sidebar:
    st.subheader("System status")
    try:
        health = api_get("/health").json()
        stats = health["stats"]
        dot = "🟢" if health["status"] == "ok" else "🔴"
        st.markdown(f"{dot} **System {health['status']}**")
        st.text(f"Documents   {stats['documents_indexed']} / {stats['max_documents']}")
        st.text(f"Chunks      {stats['total_chunks']}")
        st.text("Reranker    cross-encoder")
        st.text(f"Threshold   {stats['similarity_threshold']:.2f}")
        at_capacity = stats["documents_remaining"] <= 0
    except httpx.HTTPError:
        st.error("Cannot reach the backend.")
        at_capacity = False

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader(
        "1. Select files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "csv", "xlsx", "xls", "json", "yaml", "yml", "docx"],
    )
    st.caption("Supported: PDF, TXT, MD, CSV, XLSX, JSON, YAML, DOCX")

    if at_capacity:
        st.warning("⚠️ Document limit reached. Delete a document to upload more.")
    elif st.button(f"2. Index {len(uploads)} file(s)" if uploads else "2. Index files",
                   disabled=not uploads, help="Chunks and embeds the selected files into the knowledge base."):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
        with st.spinner("Indexing..."):
            resp = api_post("/documents/upload", files=files)
        if resp.status_code == 207:
            for res in resp.json()["results"]:
                if res["status"] == "success":
                    note = f"  ⚠ {res['warnings'][0]}" if res["warnings"] else ""
                    st.success(f"✓ {res['filename']} — {res['chunks_indexed']} chunks{note}")
                else:
                    st.error(f"✗ {res['filename']} — {res['error']['detail']}")
            st.session_state.documents = fetch_documents()
        else:
            st.error(detail_of(resp))

    st.divider()
    docs = st.session_state.documents
    st.subheader(f"Indexed documents ({len(docs)})")
    if not docs:
        st.caption("No documents indexed yet. Upload one above to get started.")
    else:
        with st.container(height=240, border=True):
            for doc in docs:
                left, right = st.columns([4, 1])
                left.markdown(f"📄 **{doc['filename']}**  \n{doc['chunks']} chunks")
                if right.button("🗑", key=f"del_{doc['filename']}"):
                    httpx.delete(f"{BACKEND_URL}/documents/{doc['filename']}", timeout=REQUEST_TIMEOUT)
                    st.session_state.documents = fetch_documents()
                    st.rerun()


# ── header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center;margin:0;'>🔍 Agentic RAG Knowledge System</h1>"
    "<p style='text-align:center;color:#888;margin:.2rem 0 0;'>"
    "Upload documents, ask questions in plain English, get grounded answers with a multi-agent pipeline.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── ask ───────────────────────────────────────────────────────────────────────

documents = st.session_state.documents
question = st.text_area("Ask a question", placeholder="Type your question here...", height=100)
_chars = len(question)
st.caption(f":red[{_chars} / 2000 characters]" if _chars > 1800 else f"{_chars} / 2000 characters")

filenames = [d["filename"] for d in documents]
selected = st.multiselect(
    "Search in", filenames,
    placeholder="All documents",
    help="Leave empty to search across all indexed documents.",
)
st.caption(f"Searching: **{', '.join(selected) if selected else 'all documents'}**")

can_ask = bool(question.strip()) and len(question.strip()) >= 3 and documents
mode_col, ask_col = st.columns([5, 1], vertical_alignment="bottom")
with mode_col:
    mode = st.radio("Agent mode", ["custom", "llama_index", "compare"], horizontal=True,
                    help="custom = fixed 3-call pipeline · llama_index = ReAct loop · compare = both side by side")
with ask_col:
    ask_clicked = st.button("Ask", type="primary", disabled=not can_ask, use_container_width=True)

if ask_clicked:
    with st.spinner("Running agent pipeline..."):
        resp = api_post("/query", json={
            "question": question,
            "filter_filenames": selected,
            "agent_mode": mode,
            "include_chunks": True,
            "include_trace": True,
        })
    st.session_state.last_result = resp.json() if resp.status_code == 200 else {"_error": detail_of(resp)}


# ── answer ────────────────────────────────────────────────────────────────────

result = st.session_state.last_result


def _tame_markdown(md: str) -> str:
    """Render any markdown headings in an answer as bold text instead.

    Models (especially the llama_index ReAct answer) sometimes format their reply as
    a document with `#`/`##` headings, which render huge — particularly in the narrow
    compare columns. This keeps the structure as bold labels at normal size.
    """
    return re.sub(r"(?m)^#{1,6}\s+(.*)$", r"**\1**", md or "")


def _confidence_badge(conf: str) -> str:
    return {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low",
            "n/a": "— not rated"}.get(conf, conf)


def _fmt_ms(ms: int) -> str:
    """Human-friendly duration: milliseconds under a second, seconds above."""
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


if result and "_error" in result:
    st.error(f"⚠️ {result['_error']}")

elif result and result.get("mode") == "compare":
    st.subheader("Comparison")
    st.caption(f"Total time {_fmt_ms(result['processing_time_ms'])} · both pipelines ran on the same question.")
    col_a, col_b = st.columns(2)
    for col, key, title in ((col_a, "custom", "Custom pipeline"), (col_b, "llama_index", "LlamaIndex ReAct")):
        side = result[key]
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"Confidence {_confidence_badge(side['confidence'])} · {side['llm_calls']} LLM calls · {_fmt_ms(side['duration_ms'])}")
                st.markdown(_tame_markdown(side["answer"]) or "_(no answer)_")
                if side["sources_used"]:
                    st.caption(f"Sources: {', '.join(side['sources_used'])}")
    v = result["validation"]
    st.caption(f"Validation: {'✓ valid' if v['is_valid'] else '⚠ flagged'} · hallucination risk: {v['hallucination_risk']}")

elif result:
    if result["short_circuit"]:
        st.info(f"ℹ️ {result['validation']['issues'][0]}")
    else:
        st.subheader("Answer")
        st.markdown(_tame_markdown(result["answer"]))
        st.caption(
            f"Confidence: {_confidence_badge(result['confidence'])}"
            f" · ⏱ {_fmt_ms(result['processing_time_ms'])}"
            f" · Sources: {', '.join(result['sources_used']) or '—'}"
        )
        risk = result["validation"]["hallucination_risk"]
        if risk == "high" or result["validation"]["suggested_action"] == "review":
            st.warning("⚠️ This answer may require verification — the validator flagged potential accuracy concerns.")


# ── diagnostics (collapsed, below the answer) ─────────────────────────────────

def _trace_and_chunks(result: dict):
    """Return (trace, chunks, label) for the side we display diagnostics for."""
    mode = result.get("mode", "custom")
    if mode == "compare":
        return result["custom"].get("trace", []), result["custom"].get("chunks", []), "custom pipeline"
    return result.get("trace", []), result.get("chunks", []), mode


if result and "_error" not in result:
    trace, chunks, label = _trace_and_chunks(result)
    icons = {"passed": "✓", "completed": "✓", "skipped": "⊘", "failed": "✗"}

    with st.expander(f"🔧 Pipeline trace — {label}", expanded=False):
        if result.get("mode") == "compare":
            st.caption("Trace shown for the custom pipeline. The LlamaIndex side runs its own ReAct loop.")
        for step in trace:
            st.markdown(f"{icons.get(step['status'], '•')} **{step['agent']}** · `{_fmt_ms(step['duration_ms'])}`")
            details = step["details"]
            if isinstance(details, dict):
                st.caption(" · ".join(f"{k}: {val}" for k, val in details.items()))
            else:
                st.caption(str(details))

    with st.expander(f"📄 Retrieved chunks — {label} ({len(chunks)})", expanded=False):
        if not chunks:
            st.caption("No chunks (pipeline short-circuited before retrieval).")
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**{i}. {c['filename']}** · similarity {c['similarity_score']:.2f} · rerank {c['rerank_score']:.2f}"
            )
            text = c["text"]
            st.caption(text[:300] + ("…" if len(text) > 300 else ""))
