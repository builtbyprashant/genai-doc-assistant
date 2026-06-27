"""Tests for app/services/chunk_service.py — added in Task 4.

Will cover (DECISIONS D-3, D-6c):
  - 200-word chunks with 25-word overlap
  - chunks below MIN_CHUNK_CHARS are discarded
  - a non-empty document never produces zero chunks
  - dispatch by file extension (all routes to the word chunker for now)
"""
