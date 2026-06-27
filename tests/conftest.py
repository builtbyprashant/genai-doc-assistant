"""Shared pytest fixtures.

The real test bodies arrive with each service (TDD — see IMPLEMENTATION_PLAN.md).
This file gives every test a clean, predictable environment to build on.
"""

from __future__ import annotations

import os

import pytest

from app.core import config


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Give each test a fresh, valid config backed by a temp ChromaDB path.

    Autouse so no test accidentally inherits real env vars or a stale cache.
    Individual tests can still override any value via monkeypatch.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # keep test output quiet

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def sample_text() -> str:
    """A short paragraph for chunking / embedding tests."""
    return (
        "ICU patients may be transferred when hemodynamic stability is maintained "
        "for at least four hours. The attending physician must sign off and a bed "
        "must be available on the receiving ward."
    )


@pytest.fixture
def sample_files(tmp_path):
    """Write a couple of tiny documents to disk and return their paths.

    Loader/ingestion tests extend this with the remaining supported formats.
    """
    txt = tmp_path / "policy.txt"
    txt.write_text("Discharge requires a completed checklist.", encoding="utf-8")

    csv = tmp_path / "data.csv"
    csv.write_text("name,age\nJohn,45\nMary,62\n", encoding="utf-8")

    return {"txt": txt, "csv": csv}
