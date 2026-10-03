#!/usr/bin/env python
"""Step-by-step demo of llama_index (ReAct) mode, showing prompt-cache reuse across turns.

Runs the REAL pipeline in llama_index mode against a seeded in-memory store (so it is
reproducible and the context clears the cacheable floor), captures each LLM turn's cache
usage, and prints a per-turn breakdown. Expected after the multi-block fix: turn 1 WRITES
the accumulated context, and later turns READ it back (cache_read > 0) while writing only
the newly-retrieved chunks.

Needs ANTHROPIC_API_KEY in the environment. Costs a few cents.
Run from the repo root:  python scripts/demo_llama.py
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import llm, pipeline  # noqa: E402
from app.core.config import get_settings  # noqa: E402

ASPECTS = [
    "root cause", "timeline", "customer impact",                 # the initial retrieval (3 of 6 asked)
    "affected services", "monitoring gaps", "severity classification",
    "rollback procedure", "escalation path", "communication plan",
    "data integrity checks", "dependency map", "SLA breach details", "post-incident actions",
    "remediation steps", "on-call schedule", "follow-up owners",  # asked for, but found only by searching
]
FIRST_RETRIEVAL = 3   # small but large-chunked, so it clears the floor yet is clearly incomplete
SEARCH_BATCH = 3      # new chunks returned per follow-up SEARCH


def _pool() -> list[dict]:
    pool = []
    for i, aspect in enumerate(ASPECTS):
        filler = " ".join(f"detail{j}" for j in range(330))  # ~340 words/chunk, ~3 chunks clear the floor
        text = (f"This section of the incident runbook documents the {aspect}. "
                f"Operators consult it when handling a Sev-2 event. {filler}")
        pool.append({"id": f"c{i}", "filename": f"runbook_{i:02d}.md", "text": text,
                     "similarity_score": 0.6, "chunk_index": i, "rerank_score": 0.6})
    return pool


class DemoStore:
    """Returns the next unseen batch of chunks on each retrieve() (query ignored), so every
    SEARCH grows the accumulated context append-only until the pool is exhausted."""

    def __init__(self):
        self._pool = _pool()
        self._served = 0

    def chunk_count(self) -> int:
        return len(self._pool)

    def unknown_filenames(self, names):
        return []

    def retrieve(self, query, top_k=None, filter_filenames=None):
        n = FIRST_RETRIEVAL if self._served == 0 else SEARCH_BATCH
        out = self._pool[self._served:self._served + n]
        self._served += len(out)
        return out


def main() -> None:
    s = get_settings()
    print(f"=== genai-doc-assistant: llama_index mode demo ===")
    print(f"model={s.anthropic_model}  cache_mode={s.cache_mode}\n")

    logging.getLogger("rag").setLevel(logging.WARNING)  # hush the per-call JSON log lines

    turns = []
    orig_log = llm._log_usage

    def capture(agent, usage, cache_mode):
        if usage is not None:
            turns.append({
                "agent": agent,
                "input": getattr(usage, "input_tokens", 0) or 0,
                "read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            })
    llm._log_usage = capture

    question = ("I need six specific facts from the runbook: (1) root cause, (2) timeline, "
                "(3) customer impact, (4) remediation steps, (5) on-call schedule, (6) follow-up "
                "owners. If any are missing from the current context, use SEARCH to fetch more "
                "before answering. Do not give a FINAL answer until you have looked for all six.")
    print(f"question: {question}\n")

    try:
        result = pipeline.run_pipeline(question, agent_mode="llama_index",
                                       include_trace=True, store=DemoStore())
    except Exception as e:
        llm._log_usage = orig_log
        print(f"run failed: {type(e).__name__}: {e}")
        return
    llm._log_usage = orig_log

    trace = result.get("trace", [])
    print("per-turn cache usage (each line is one LLM call in the ReAct loop):")
    print(f"  {'turn':<5}{'action':<12}{'input':>8}{'cache_read':>12}{'cache_write':>13}   note")
    for i, t in enumerate(turns):
        action = ""
        if i < len(trace):
            action = trace[i]["agent"].split("·")[-1].strip()
        note = "writes the context" if t["read"] == 0 else "READS prior context, writes only new chunks"
        print(f"  {i+1:<5}{action:<12}{t['input']:>8}{t['read']:>12}{t['write']:>13}   {note}")

    print("\nloop trace:")
    for st in trace:
        d = st.get("details", {})
        extra = {k: v for k, v in d.items() if k != "step"}
        print(f"  step {d.get('step','?')}: {st['agent']}  {extra}")

    ans = (result.get("answer") or "").replace("\n", " ")
    print(f"\nfinal answer: {ans[:160]}{'...' if len(ans) > 160 else ''}")
    print(f"llm_calls={result.get('llm_calls')}  tokens={result.get('tokens')}  "
          f"cost_usd={result.get('cost_usd')}  cache_read_tokens={result.get('cache_read_tokens')}")

    reads = [t for t in turns if t["read"] > 0]
    if reads:
        print(f"\n=> {len(reads)} of {len(turns)} turns reused the accumulated context from cache "
              f"(cache_read > 0). The growing prefix is stable.")
    elif len(turns) >= 2:
        print("\n=> No turn read from cache despite multiple turns: investigate (floor or drift).")
    else:
        print("\n=> Only one turn ran (model answered immediately); re-run for a multi-turn loop.")


if __name__ == "__main__":
    main()
