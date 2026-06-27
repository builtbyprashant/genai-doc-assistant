"""Tests for app/services/vector_store.py — added in Task 5, extended in Task 6.

Will cover (DECISIONS D-1, C-F):
  - collection created with hnsw:space=cosine
  - similarity = 1 - distance, scores in [0, 1]
  - index / retrieve / delete round-trip
  - document filter + threshold / short-circuit behaviour
"""
