"""Streamlit UI for the Agentic RAG system.

Single page: a sidebar Document Manager + a main Q&A area, with the pipeline trace
and retrieved chunks tucked into collapsed expanders below the answer. The frontend
runs in its own container and only talks to the backend over HTTP — it never imports
the app package, and never shows raw JSON or tracebacks to the user.
"""

from __future__ import annotations

import json
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

      /* Shrink the in-sidebar collapse bar so content sits near the top. The collapse
         and reopen controls keep working (verified); the spacer is an empty logo
         placeholder we don't use, so its height goes to 0. */
      [data-testid="stSidebar"] [data-testid="stLogoSpacer"] {height: 0 !important;}
      [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {min-height: 0 !important; height: auto !important; padding-top: 0.25rem !important; padding-bottom: 0 !important;}
      [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top: 0.25rem !important;}

      /* Keep each indexed-document row on one line. In the narrow sidebar Streamlit
         gives every column min-width:100% and wraps them, so the 🗑 button drops
         below the filename. Force side-by-side instead. */
      [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: 0.25rem !important;}
      [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width: 0 !important;}
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


HEALTH_POLL_SECONDS = 30  # never poll /health more often than this


@st.cache_data(ttl=HEALTH_POLL_SECONDS, show_spinner=False)
def fetch_health() -> dict:
    """Cached /health for the sidebar. Streamlit reruns on every interaction, so an
    uncached call would hit the backend constantly; the TTL caps it at one call per
    HEALTH_POLL_SECONDS (30s). Call `fetch_health.clear()` after an upload/delete to
    refresh the stats immediately rather than waiting out the TTL."""
    return api_get("/health").json()


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
if "query_in_progress" not in st.session_state:
    st.session_state.query_in_progress = False
if "pending_request" not in st.session_state:
    st.session_state.pending_request = None


# ── sidebar: document manager ─────────────────────────────────────────────────

with st.sidebar:
    st.subheader("System status")
    try:
        health = fetch_health()
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
            fetch_health.clear()  # doc count/chunks changed → refresh stats now
        else:
            st.error(detail_of(resp))

    st.divider()
    docs = st.session_state.documents
    st.subheader(f"Indexed documents ({len(docs)})")
    if not docs:
        st.caption("No documents indexed yet. Upload one above to get started.")
    else:
        # Let the box hug its content for a handful of docs — no clipping and no
        # wasted space below it — and only cap + scroll once the list gets long.
        box_height = "content" if len(docs) <= 4 else 360
        with st.container(height=box_height, border=True):
            for doc in docs:
                left, right = st.columns([4, 1])
                left.markdown(f"📄 **{doc['filename']}**  \n{doc['chunks']} chunks")
                if right.button("🗑", key=f"del_{doc['filename']}"):
                    httpx.delete(f"{BACKEND_URL}/documents/{doc['filename']}", timeout=REQUEST_TIMEOUT)
                    st.session_state.documents = fetch_documents()
                    fetch_health.clear()  # doc count/chunks changed → refresh stats now
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
    # Disabled while a request is in flight (and until it's processed/errored/lost),
    # so rapid or queued double-clicks can't fire a second request.
    ask_clicked = st.button("Ask", type="primary",
                            disabled=not can_ask or st.session_state.query_in_progress,
                            use_container_width=True)

# On click: flag the request in-flight, stash it, and rerun so the actual work happens
# on a pass where the Ask button renders disabled — a queued second click then lands on
# a disabled button and is ignored.
if ask_clicked and not st.session_state.query_in_progress:
    st.session_state.query_in_progress = True
    st.session_state.pending_request = {
        "question": question, "filter_filenames": selected, "agent_mode": mode,
        "include_chunks": True, "include_trace": True,
    }
    st.session_state.last_result = None
    st.rerun()


def _stream_answer(req: dict, holder: dict):
    """Yield the answer text as it streams; stash the trailing metadata JSON in `holder`.

    The backend sends the answer text, then a 0x1E separator, then a JSON blob with
    confidence/sources/validation/trace. We split on that separator so only answer
    text reaches st.write_stream.
    """
    sep = "\x1e"
    meta_started = False
    try:
        with httpx.stream("POST", f"{BACKEND_URL}/query/stream", json=req,
                          timeout=REQUEST_TIMEOUT) as r:
            if r.status_code != 200:
                r.read()
                holder["error"] = detail_of(r)
                return
            for chunk in r.iter_text():
                if meta_started:
                    holder["meta_raw"] += chunk
                elif sep in chunk:
                    answer_part, meta_part = chunk.split(sep, 1)
                    if answer_part:
                        yield answer_part
                    holder["meta_raw"] += meta_part
                    meta_started = True
                elif chunk:
                    yield chunk
    except httpx.HTTPError:
        holder["error"] = "Could not reach the backend."


# Process the in-flight request. This pass renders the Ask button disabled; it is
# re-enabled in `finally` whether the request is processed, errored, or lost (the
# frontend REQUEST_TIMEOUT covers the backend's full retry window). Custom mode streams
# the answer; the others run batch. Then rerun so the final result (caption + diagnostics)
# renders through the same path for every mode.
_req = st.session_state.pending_request
if _req:
    st.session_state.pending_request = None
    try:
        if _req["agent_mode"] == "custom":
            st.subheader("Answer")
            _holder = {"meta_raw": "", "error": None}
            st.write_stream(_stream_answer(_req, _holder))
            if _holder["error"]:
                st.session_state.last_result = {"_error": _holder["error"]}
            elif _holder["meta_raw"]:
                st.session_state.last_result = json.loads(_holder["meta_raw"])
            else:
                st.session_state.last_result = {"_error": "The service returned an empty response."}
        else:
            try:
                with st.spinner("Running agent pipeline..."):
                    resp = api_post("/query", json=_req)
                st.session_state.last_result = (
                    resp.json() if resp.status_code == 200 else {"_error": detail_of(resp)}
                )
            except httpx.HTTPError:
                st.session_state.last_result = {
                    "_error": "Could not reach the backend — the request timed out or was lost."
                }
    finally:
        st.session_state.query_in_progress = False  # re-enable: processed / errored / lost
    st.rerun()


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


def _fmt_usage(d: dict) -> str:
    """Tokens consumed + an estimated cost, e.g. '2,847 tokens · ~$0.0041'."""
    return f"{d.get('tokens', 0):,} tokens · ~${d.get('cost_usd', 0.0):.4f}"


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
                st.caption(f"Confidence {_confidence_badge(side['confidence'])} · {side['llm_calls']} LLM calls · {_fmt_ms(side['duration_ms'])} · {_fmt_usage(side)}")
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
            f" · 🔢 {_fmt_usage(result)}"
            f" · Sources: {', '.join(result['sources_used']) or '—'}"
        )
        risk = result["validation"]["hallucination_risk"]
        if risk == "high" or result["validation"]["suggested_action"] == "review":
            st.warning("⚠️ This answer may require verification — the validator flagged potential accuracy concerns.")


# ── diagnostics (collapsed, below the answer) ─────────────────────────────────

_STEP_ICONS = {"passed": "✓", "completed": "✓", "skipped": "⊘", "failed": "✗"}


def _render_trace(trace: list):
    if not trace:
        st.caption("No trace.")
    for step in trace:
        st.markdown(f"{_STEP_ICONS.get(step['status'], '•')} **{step['agent']}** · `{_fmt_ms(step['duration_ms'])}`")
        details = step["details"]
        st.caption(" · ".join(f"{k}: {val}" for k, val in details.items())
                   if isinstance(details, dict) else str(details))


def _render_chunks(chunks: list):
    if not chunks:
        st.caption("No chunks (pipeline short-circuited before retrieval).")
    for i, c in enumerate(chunks, 1):
        st.markdown(f"**{i}. {c['filename']}** · similarity {c['similarity_score']:.2f} · rerank {c['rerank_score']:.2f}")
        text = c["text"]
        st.caption(text[:300] + ("…" if len(text) > 300 else ""))


def _diagnostics_panel(trace: list, chunks: list, label: str):
    """One pipeline's collapsed trace + retrieved-chunks expanders."""
    with st.expander(f"🔧 Pipeline trace — {label}", expanded=False):
        _render_trace(trace)
    with st.expander(f"📄 Retrieved chunks — {label} ({len(chunks)})", expanded=False):
        _render_chunks(chunks)


if result and "_error" not in result:
    if result.get("mode") == "compare":
        # Both pipelines' internals, side by side under their respective answers —
        # custom's fixed 7-step trace vs the LlamaIndex ReAct loop, and what each retrieved.
        st.caption("Pipeline internals — compare how each side reached its answer:")
        diag_custom, diag_llama = st.columns(2)
        with diag_custom:
            _diagnostics_panel(result["custom"].get("trace", []),
                               result["custom"].get("chunks", []), "Custom pipeline")
        with diag_llama:
            _diagnostics_panel(result["llama_index"].get("trace", []),
                               result["llama_index"].get("chunks", []), "LlamaIndex ReAct")
    else:
        _diagnostics_panel(result.get("trace", []), result.get("chunks", []),
                           result.get("mode", "custom"))
