"""Tests for the agent pipeline — added across Tasks 7-9.

Will cover (DECISIONS C-A, C-E, C-F, C-H, D-5):
  - Reasoner answers only from retrieved context
  - Planner / Ranker / Validator behave correctly in isolation
  - Validator receives only question + answer + chunks (independence)
  - compare-mode envelope shape (processing_time_ms, short_circuit, both sides)
  - SafetyGuard blocks injection; empty store and low similarity short-circuit (HTTP 200)
"""
