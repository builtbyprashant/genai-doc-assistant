"""Tests for the agent pipeline — built across Tasks 7-9.

LLM agents are tested with a mocked `llm.complete` so no real API calls happen.

This file starts with the Reasoner (Task 7). Planner / Ranker / Validator /
SafetyGuard and the orchestrator are appended in Tasks 8-9. Covers DECISIONS
C-A, C-E, C-F, C-H, D-5.
"""

from __future__ import annotations

import pytest

from app.agents import llm, reasoner


@pytest.fixture
def chunks():
    return [
        {"filename": "icu.txt", "text": "Transfer is allowed after four hours of stability.", "chunk_index": 0},
        {"filename": "policy.pdf", "text": "The attending physician must sign off.", "chunk_index": 1},
    ]


def test_reasoner_parses_structured_response(monkeypatch, chunks):
    monkeypatch.setattr(
        llm, "complete",
        lambda system, user, **kw: (
            "ANSWER: Transfer after four hours.\n"
            "CONFIDENCE: HIGH\n"
            "REASON: Stated directly in the guidelines.\n"
            "SOURCES: icu.txt"
        ),
    )
    out = reasoner.reasoner_agent("When can ICU patients transfer?", chunks)
    assert out["answer"] == "Transfer after four hours."
    assert out["confidence"] == "high"
    assert out["confidence_reason"] == "Stated directly in the guidelines."
    assert out["sources_used"] == ["icu.txt"]


def test_reasoner_falls_back_when_format_slips(monkeypatch, chunks):
    monkeypatch.setattr(llm, "complete", lambda system, user, **kw: "Just a plain answer.")
    out = reasoner.reasoner_agent("q", chunks)
    assert out["answer"] == "Just a plain answer."
    assert out["confidence"] == "medium"          # conservative default
    assert set(out["sources_used"]) == {"icu.txt", "policy.pdf"}  # fall back to provided


def test_reasoner_only_keeps_real_sources(monkeypatch, chunks):
    # The model lists a file we never gave it — it must be dropped.
    monkeypatch.setattr(
        llm, "complete",
        lambda system, user, **kw: "ANSWER: x\nCONFIDENCE: LOW\nSOURCES: icu.txt, ghost.txt",
    )
    out = reasoner.reasoner_agent("q", chunks)
    assert out["sources_used"] == ["icu.txt"]
    assert out["confidence"] == "low"


def test_reasoner_sends_context_and_only_rule_to_llm(monkeypatch, chunks):
    captured = {}

    def fake_complete(system, user, **kw):
        captured["system"] = system
        captured["user"] = user
        return "ANSWER: ok\nCONFIDENCE: HIGH\nSOURCES: icu.txt"

    monkeypatch.setattr(llm, "complete", fake_complete)
    reasoner.reasoner_agent("When can patients transfer?", chunks)

    assert "ONLY" in captured["system"]                      # the grounding rule
    assert "four hours of stability" in captured["user"]     # the chunk text
    assert "When can patients transfer?" in captured["user"] # the question
