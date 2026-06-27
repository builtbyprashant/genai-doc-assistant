"""Tests for app/services/document_loader.py — added in Task 3.

Will cover (DECISIONS D-2, C-C):
  - all 8 supported formats load correctly
  - scanned PDF and password-protected PDF are rejected with clear errors
  - duplicate content detected via SHA-256 hash (different filename rejected)
  - oversized / empty files rejected before parsing
"""
