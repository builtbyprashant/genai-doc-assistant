"""Tests for app/api/main.py via FastAPI TestClient — added in Task 2.

Will cover (DECISIONS C-E, C-F):
  - every endpoint: /, /health, /documents (upload/list/delete), /query
  - upload returns 207 multi-status
  - compare-mode response shape
  - short-circuit responses (empty store, below threshold) return HTTP 200
"""
