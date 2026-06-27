"""Splitting documents into overlapping, embeddable chunks.

`chunk_document` is a router that dispatches by file extension. In Phase 1 every
type goes through the same word-based chunker — the router exists so Phase 2 can
add a row chunker (CSV/Excel) and a page chunker (PDF) without touching any
caller. (DECISIONS: D-6c)

Chunk size is kept around 200 words so each chunk stays within the embedding
model's 256-token limit. (DECISIONS: D-3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.config import get_settings


def chunk_document(text: str, filename: str, raw_content: Optional[bytes] = None) -> list[dict]:
    """Chunk a document, choosing a strategy from its file extension.

    `raw_content` is unused in Phase 1 but is part of the signature so the Phase 2
    tabular/page chunkers can read the original bytes without an interface change.
    """
    _ext = Path(filename).suffix.lower()
    # Phase 1: one strategy for everything. Phase 2 branches here on _ext.
    return word_chunker(text)


def word_chunker(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chars: Optional[int] = None,
) -> list[dict]:
    """Split text into overlapping word windows.

    Defaults come from config (CHUNK_SIZE / CHUNK_OVERLAP / MIN_CHUNK_CHARS) but
    can be overridden per call, which keeps the function easy to test.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    min_chars = settings.min_chunk_chars if min_chars is None else min_chars

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap  # config guarantees overlap < chunk_size, so step >= 1

    chunks: list[dict] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        chunk_text = " ".join(window)
        if _is_meaningful(chunk_text, min_chars):
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "word_count": len(window),
                    "char_count": len(chunk_text),
                }
            )
        if start + chunk_size >= len(words):
            break  # last window reached — avoid empty trailing iterations

    return chunks


def _is_meaningful(text: str, min_chars: int) -> bool:
    """Drop chunks that are too short or carry no real content (whitespace/punctuation)."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    return any(ch.isalnum() for ch in stripped)
