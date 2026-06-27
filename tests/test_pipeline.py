"""Tests for the agent pipeline — built across Tasks 7-9.

LLM agents are tested with a mocked `llm.complete` so no real API calls happen.

This file starts with the Reasoner (Task 7). Planner / Ranker / Validator /
SafetyGuard and the orchestrator are appended in Tasks 8-9. Covers DECISIONS
C-A, C-E, C-F, C-H, D-5.
"""

from __future__ import annotations

import pytest

from app.agents import llm, planner, ranker, reasoner, validator


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


# ── PlannerAgent (Task 8) ─────────────────────────────────────────────────────

def test_planner_parses_json_plan(monkeypatch):
    monkeypatch.setattr(
        llm, "complete",
        lambda system, user, **kw: '{"intent": "find transfer policy", '
        '"retrieval_strategy": "semantic", "rewritten_query": "ICU transfer criteria"}',
    )
    plan = planner.planner_agent("when can patients leave ICU?")
    assert plan["intent"] == "find transfer policy"
    assert plan["rewritten_query"] == "ICU transfer criteria"
    assert plan["top_k"] == 10  # fixed from config, not model-chosen


def test_planner_falls_back_on_malformed_json(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda system, user, **kw: "sorry, no JSON here")
    plan = planner.planner_agent("original question")
    # Safe default keeps the pipeline going with the original query.
    assert plan["rewritten_query"] == "original question"
    assert plan["retrieval_strategy"] == "semantic"


# ── RankerAgent (Task 8, cross-encoder — no LLM) ──────────────────────────────

def test_ranker_orders_by_answer_relevance():
    chunks = [
        {"filename": "a.txt", "text": "The capital of France is Paris.", "chunk_index": 0},
        {"filename": "b.txt", "text": "Cats are small domesticated animals kept as pets.", "chunk_index": 1},
    ]
    ranked = ranker.ranker_agent("Tell me about pet cats", chunks, top_k=2)
    assert ranked[0]["filename"] == "b.txt"
    assert all(0.0 <= c["rerank_score"] <= 1.0 for c in ranked)


def test_ranker_respects_top_k():
    chunks = [{"filename": f"{i}.txt", "text": f"text {i}", "chunk_index": i} for i in range(4)]
    assert len(ranker.ranker_agent("query", chunks, top_k=2)) == 2


def test_ranker_empty_input():
    assert ranker.ranker_agent("query", [], top_k=3) == []


# ── ValidatorAgent (Task 8) ───────────────────────────────────────────────────

def test_validator_parses_json(monkeypatch, chunks):
    monkeypatch.setattr(
        llm, "complete",
        lambda system, user, **kw: '{"is_valid": true, "issues": [], '
        '"hallucination_risk": "low", "suggested_action": "none"}',
    )
    out = validator.validator_agent("q", "Transfer after four hours.", chunks)
    assert out["is_valid"] is True
    assert out["hallucination_risk"] == "low"


def test_validator_falls_back_on_malformed_json(monkeypatch, chunks):
    monkeypatch.setattr(llm, "complete", lambda system, user, **kw: "not json")
    out = validator.validator_agent("q", "a", chunks)
    assert out == {"is_valid": True, "issues": [], "hallucination_risk": "unknown", "suggested_action": "none"}


def test_validator_sees_only_question_answer_context(monkeypatch, chunks):
    captured = {}

    def fake(system, user, **kw):
        captured["user"] = user
        return '{"is_valid": true, "hallucination_risk": "low"}'

    monkeypatch.setattr(llm, "complete", fake)
    validator.validator_agent("When can patients transfer?", "After four hours.", chunks)
    # Independence (D-5): the answer, question and context are present...
    assert "After four hours." in captured["user"]
    assert "When can patients transfer?" in captured["user"]
    assert "attending physician" in captured["user"]
    # ...and there is no place to leak the reasoner's chain — the function simply
    # has no parameter for it (enforced by signature).
