"""ChromaDB vector store.

This is the *only* module that imports ChromaDB. Everything else talks to the
store through the small interface below (index / retrieve / list / delete /
count). Keeping the dependency in one place is what lets Phase 2 swap in a
different backend behind an abstract interface without touching callers.
(DECISIONS: D-6a)

Two things matter for correctness here (DECISIONS: D-1):
  - the collection is created with cosine space (`hnsw:space=cosine`)
  - ChromaDB returns a *distance*, so we convert it to a similarity with
    `similarity = 1 - distance` on every result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.core.config import get_settings

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, matches the model baked into the image


class VectorStore:
    def __init__(self, persist_path: Optional[str] = None):
        path = persist_path or get_settings().chroma_persist_path
        self._client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Same embedder for indexing and querying so vectors live in one space.
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )

    # ── write ────────────────────────────────────────────────────────────────

    def index_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[dict],
        content_hash: str = "",
        size_bytes: int = 0,
    ) -> int:
        """Embed and store chunks for one document. Returns how many were added."""
        if not chunks:
            return 0

        indexed_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{doc_id}:{c['chunk_index']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": c["chunk_index"],
                "content_hash": content_hash,
                "size_bytes": size_bytes,
                "indexed_at": indexed_at,
            }
            for c in chunks
        ]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def delete_document(self, filename: str) -> int:
        """Remove every chunk belonging to a filename. Returns the count removed."""
        existing = self._collection.get(where={"filename": {"$eq": filename}})
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ── read ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int,
        filter_filenames: Optional[list[str]] = None,
    ) -> list[dict]:
        """Top-k chunks by cosine similarity, optionally restricted to filenames.

        Returns an empty list for an empty store — the pipeline turns that into a
        200 short-circuit rather than an error. (C-F)
        """
        count = self.chunk_count()
        if count == 0:
            return []

        where = None
        if filter_filenames:
            where = {"filename": {"$in": list(filter_filenames)}}

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),  # asking for more than we have is fine
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        ids = result["ids"][0]
        for i in range(len(ids)):
            meta = result["metadatas"][0][i]
            # Cosine distance is in [0, 2], so 1 - distance is in [-1, 1]. The API
            # contract reports similarity as 0.0-1.0, so clamp: a (near-)negative
            # cosine just means "not relevant" → 0. (D-1)
            similarity = max(0.0, min(1.0, 1.0 - result["distances"][0][i]))
            chunks.append(
                {
                    "id": ids[i],
                    "text": result["documents"][0][i],
                    "filename": meta["filename"],
                    "chunk_index": meta["chunk_index"],
                    "similarity_score": similarity,
                }
            )
        return chunks

    def list_documents(self) -> list[dict]:
        """One row per indexed document, aggregated from its chunk metadata."""
        data = self._collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in data["metadatas"]:
            filename = meta["filename"]
            doc = docs.get(filename)
            if doc is None:
                docs[filename] = {
                    "doc_id": meta["doc_id"],
                    "filename": filename,
                    "chunks": 1,
                    "content_hash": meta.get("content_hash", ""),
                    "size_bytes": meta.get("size_bytes", 0),
                    "indexed_at": meta.get("indexed_at", ""),
                }
            else:
                doc["chunks"] += 1
        return list(docs.values())

    def chunk_count(self) -> int:
        return self._collection.count()

    def collection_space(self) -> Optional[str]:
        """The configured distance space — guards the cosine assumption. (D-1)"""
        return (self._collection.metadata or {}).get("hnsw:space")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """App-wide singleton. Phase 2 replaces this with a backend factory. (D-6a)"""
    return VectorStore()
