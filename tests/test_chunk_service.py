"""Tests for app/services/chunk_service.py (Task 4).

Covers DECISIONS D-3 (chunk size / overlap / MIN_CHUNK_CHARS) and D-6c
(dispatch by file type, all routed to the word chunker for now).
"""

from __future__ import annotations

from app.services.chunk_service import chunk_document, word_chunker


def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_respects_chunk_size_and_overlap():
    chunks = word_chunker(_words(500), chunk_size=200, overlap=25)

    assert chunks[0]["word_count"] == 200
    # Consecutive chunks share `overlap` words at the boundary.
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-25:] == second[:25]
    # Indexes are sequential from 0.
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_uses_config_defaults(sample_text):
    # No explicit size/overlap → falls back to CHUNK_SIZE=200 / CHUNK_OVERLAP=25.
    chunks = word_chunker(_words(500))
    assert chunks[0]["word_count"] == 200


def test_short_document_makes_one_chunk():
    text = "This is a short policy document describing discharge rules."
    chunks = word_chunker(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_non_empty_document_never_yields_zero_chunks():
    chunks = word_chunker("Meaningful sentence well above the minimum length.")
    assert len(chunks) >= 1


def test_whitespace_only_is_discarded():
    assert word_chunker("   \n\t   ") == []


def test_punctuation_only_is_discarded():
    # Long enough to clear MIN_CHUNK_CHARS, but no actual content.
    assert word_chunker("." * 40) == []


def test_below_min_chunk_chars_is_discarded():
    # "hi" is 2 chars, under the default MIN_CHUNK_CHARS of 20.
    assert word_chunker("hi") == []


def test_min_chars_param_is_honoured():
    text = "a fairly normal sentence here"
    assert word_chunker(text, min_chars=5) != []
    assert word_chunker(text, min_chars=1000) == []


def test_chunk_metadata_fields_present():
    chunk = word_chunker("A normal sentence used to check metadata fields.")[0]
    assert set(chunk) >= {"text", "chunk_index", "word_count", "char_count"}
    assert chunk["char_count"] == len(chunk["text"])


def test_chunk_document_routes_by_extension():
    text = _words(300)
    # Every extension currently routes to the same word chunker (D-6c).
    assert chunk_document(text, "report.txt") == word_chunker(text)
    assert chunk_document(text, "data.csv") == word_chunker(text)
    assert chunk_document(text, "notes.pdf") == word_chunker(text)
