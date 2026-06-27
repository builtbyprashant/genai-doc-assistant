"""Streamlit UI for the Agentic RAG system.

Single page: a sidebar Document Manager + a two-column main area (Q&A on the left,
pipeline detail on the right). The frontend runs in its own container and only
talks to the backend over HTTP — it never imports the app package, and it never
shows raw JSON or tracebacks to the user (UI spec, error display rules).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 90  # query can take a few seconds end to end

st.set_page_config(page_title="RAG Knowledge System", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")


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
    """Pull the human-readable RFC-7807 detail, never raw JSON."""
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
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "Drag and drop files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "csv", "xlsx", "xls", "json", "yaml", "yml", "docx"],
    )
    st.caption("Supported: PDF, TXT, MD, CSV, XLSX, JSON, YAML, DOCX")

    if at_capacity:
        st.warning("⚠️ Document limit reached. Delete a document to upload more.")
    elif st.button("Upload", disabled=not uploads):
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
        with st.container(height=300, border=True):
            for doc in docs:
                left, right = st.columns([4, 1])
                left.markdown(f"📄 **{doc['filename']}**  \n{doc['chunks']} chunks")
                if right.button("🗑", key=f"del_{doc['filename']}"):
                    httpx.delete(f"{BACKEND_URL}/documents/{doc['filename']}", timeout=REQUEST_TIMEOUT)
                    st.session_state.documents = fetch_documents()
                    st.rerun()


# ── main area ─────────────────────────────────────────────────────────────────

centre, right = st.columns([2, 1])

with centre:
    st.subheader("Ask a question")
    question = st.text_area("Question", placeholder="Type your question here...",
                            height=100, label_visibility="collapsed")
    st.caption(f"{len(question)} / 2000 characters")

    selected = st.multiselect("Search in", [d["filename"] for d in st.session_state.documents])
    mode = st.radio("Agent mode", ["custom", "llama_index", "compare"], horizontal=True)

    can_ask = bool(question.strip()) and len(question.strip()) >= 3 and st.session_state.documents
    if st.button("Ask", type="primary", disabled=not can_ask):
        with st.spinner("Running agent pipeline..."):
            resp = api_post("/query", json={
                "question": question,
                "filter_filenames": selected,
                "agent_mode": mode,
                "include_chunks": True,
                "include_trace": True,
            })
        st.session_state.last_result = resp.json() if resp.status_code == 200 else {"_error": detail_of(resp)}

    result = st.session_state.last_result
    if result and "_error" in result:
        st.error(f"⚠️ {result['_error']}")
    elif result and result.get("mode") == "compare":
        col_a, col_b = st.columns(2)
        for col, key, title in ((col_a, "custom", "Custom Pipeline"), (col_b, "llama_index", "LlamaIndex ReAct")):
            side = result[key]
            with col:
                st.markdown(f"**{title}**")
                st.write(side["answer"] or "_(no answer)_")
                st.caption(f"Confidence: {side['confidence']} · LLM calls: {side['llm_calls']} · {side['duration_ms']} ms")
    elif result:
        if result["short_circuit"]:
            st.info(f"ℹ️ {result['validation']['issues'][0]}")
        else:
            st.markdown("**Answer**")
            st.markdown(result["answer"])
            st.caption(f"Confidence: {result['confidence']} · Sources: {', '.join(result['sources_used'])}")
            if result["validation"]["hallucination_risk"] in ("high",) or result["validation"]["suggested_action"] == "review":
                st.warning("⚠️ This answer may require verification.")

with right:
    st.subheader("Pipeline detail")
    result = st.session_state.last_result
    if not result or "_error" in result:
        st.caption("Run a query to see the agent trace and retrieved chunks.")
    else:
        trace = result.get("trace") or result.get("custom", {}).get("trace", [])
        for step in trace:
            icon = {"passed": "✓", "completed": "✓", "skipped": "⊘", "failed": "✗"}.get(step["status"], "•")
            with st.expander(f"{icon} {step['agent']}  ·  {step['duration_ms']} ms"):
                st.write(step["details"])

        chunks = result.get("chunks") or result.get("custom", {}).get("chunks", [])
        if chunks:
            st.subheader(f"Retrieved chunks ({len(chunks)})")
            for i, c in enumerate(chunks, 1):
                with st.expander(f"{i}. {c['filename']} · sim {c['similarity_score']:.2f} · rerank {c['rerank_score']:.2f}"):
                    st.write(c["text"])
