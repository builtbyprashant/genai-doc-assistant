"""Tests for app/services/vector_store.py (Task 5, extended in Task 6).

Covers DECISIONS D-1 (cosine space, similarity = 1 - distance) and C-F
(empty-store signal), plus the index / retrieve / delete / list interface.

The first test run downloads the embedding model (all-MiniLM-L6-v2) once.
"""

from __future__ import annotations

import pytest

from app.services.vector_store import VectorStore


def _chunks(*texts):
    """Build chunk dicts the way chunk_service will (text + chunk_index)."""
    return [{"text": t, "chunk_index": i} for i, t in enumerate(texts)]


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_path=str(tmp_path / "chroma"))


def test_collection_uses_cosine_space(store):
    # The whole similarity scale depends on this — guard it explicitly. (D-1)
    assert store.collection_space() == "cosine"


def test_index_returns_count_and_updates_total(store):
    added = store.index_chunks(
        doc_id="d1", filename="a.txt",
        chunks=_chunks("alpha one", "alpha two", "alpha three"),
        content_hash="hash-a", size_bytes=42,
    )
    assert added == 3
    assert store.chunk_count() == 3


def test_empty_store_retrieve_returns_empty_list(store):
    # No documents indexed → no ChromaDB query, just an empty result. (C-F)
    assert store.retrieve("anything", top_k=5) == []


def test_similarity_in_range_and_is_one_minus_distance(store):
    store.index_chunks(
        doc_id="d1", filename="a.txt",
        chunks=_chunks(
            "Cats are small domesticated animals kept as pets.",
            "The capital of France is Paris.",
        ),
        content_hash="h", size_bytes=1,
    )
    results = store.retrieve("Tell me about pet cats", top_k=2)

    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["similarity_score"] <= 1.0
    # Most relevant chunk ranks first and clears a sensible bar.
    assert "cats" in results[0]["text"].lower()
    assert results[0]["similarity_score"] > results[1]["similarity_score"]


def test_identical_text_scores_near_one(store):
    text = "Patients may be transferred from ICU after four hours of stability."
    store.index_chunks(
        doc_id="d1", filename="a.txt", chunks=_chunks(text),
        content_hash="h", size_bytes=1,
    )
    top = store.retrieve(text, top_k=1)[0]
    # Querying with the exact chunk text → distance ~0 → similarity ~1.
    assert top["similarity_score"] > 0.9


def test_filter_by_filename(store):
    store.index_chunks("d1", "a.txt", _chunks("alpha content"), "h1", 1)
    store.index_chunks("d2", "b.txt", _chunks("beta content"), "h2", 1)

    results = store.retrieve("content", top_k=10, filter_filenames=["a.txt"])
    assert results
    assert {r["filename"] for r in results} == {"a.txt"}


def test_top_k_capped_at_chunk_count(store):
    store.index_chunks("d1", "a.txt", _chunks("only one chunk"), "h", 1)
    # Asking for more than exist must not error — return what we have.
    assert len(store.retrieve("chunk", top_k=10)) == 1


def test_delete_document_removes_its_chunks(store):
    store.index_chunks("d1", "a.txt", _chunks("x", "y"), "h1", 1)
    store.index_chunks("d2", "b.txt", _chunks("z"), "h2", 1)

    removed = store.delete_document("a.txt")
    assert removed == 2
    assert store.chunk_count() == 1
    assert {d["filename"] for d in store.list_documents()} == {"b.txt"}


def test_delete_unknown_filename_removes_nothing(store):
    store.index_chunks("d1", "a.txt", _chunks("x"), "h", 1)
    assert store.delete_document("missing.txt") == 0
    assert store.chunk_count() == 1


def test_existing_filenames(store):
    store.index_chunks("d1", "a.txt", _chunks("x"), "h1", 1)
    store.index_chunks("d2", "b.txt", _chunks("y"), "h2", 1)
    assert store.existing_filenames() == {"a.txt", "b.txt"}


def test_unknown_filenames_flags_missing(store):
    # Supports the pipeline's filter-not-found check (Layer 4 input validation).
    store.index_chunks("d1", "a.txt", _chunks("x"), "h", 1)
    assert store.unknown_filenames(["a.txt", "ghost.txt"]) == ["ghost.txt"]
    assert store.unknown_filenames(["a.txt"]) == []


def test_list_documents_aggregates_by_file(store):
    store.index_chunks("d1", "a.txt", _chunks("one", "two"), "hash-a", 100)
    store.index_chunks("d2", "b.txt", _chunks("three"), "hash-b", 200)

    docs = {d["filename"]: d for d in store.list_documents()}
    assert set(docs) == {"a.txt", "b.txt"}
    assert docs["a.txt"]["chunks"] == 2
    assert docs["a.txt"]["content_hash"] == "hash-a"
    assert docs["a.txt"]["doc_id"] == "d1"
    assert docs["b.txt"]["chunks"] == 1
