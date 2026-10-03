#!/usr/bin/env python
"""Live probe: does the accumulated ReAct context actually read from cache across turns?

This issues two back-to-back calls that mimic consecutive ReAct turns, turn 2's context is
turn 1's context plus new chunks (append-only), and prints per-call cache usage for two
context structures:

  multi-block : one cache block per chunk (the current code, paper Prop. 2)
  single-block: one growing string block (the old code, paper Prop. 4)

Expected on a model whose context clears the minimum cacheable length:
  multi-block  turn 2 -> cache_read_input_tokens > 0   (prior chunks served from cache)
  single-block turn 2 -> cache_read_input_tokens ~ 0   (breakpoint drift, nothing matches)

It fabricates its own chunks, so it needs no indexed documents. It does need a real API key
(ANTHROPIC_API_KEY) and costs roughly 5 to 15 cents (four short calls over a large context).

Run from anywhere:
    python scripts/probe_cache.py
Force a model/mode for the run:
    ANTHROPIC_MODEL=claude-haiku-4-5 ANTHROPIC_CACHE_MODE=block python scripts/probe_cache.py
"""
from __future__ import annotations

import os
import sys

# Make `app` importable no matter where this is run from (repo root is this file's grandparent).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import llm  # noqa: E402
from app.agents.llama_agent import LLAMA_SYSTEM, NEXT_ACTION, context_blocks  # noqa: E402
from app.core.config import get_settings  # noqa: E402

# 20 chunks of ~200 words keeps the turn-1 context well above every model's cacheable floor
# (~5k tokens > Haiku's ~4096 and Sonnet's ~1024), so a zero read means drift, not the floor.
N_TURN1 = 20
N_NEW = 4
WORDS_PER_CHUNK = 200
MAX_TOKENS = 16  # we only measure input caching; keep the completion tiny


def _make_chunks(n: int, start: int = 0) -> list[dict]:
    body = " ".join(f"token{j}" for j in range(WORDS_PER_CHUNK))
    return [
        {"id": f"c{i}", "filename": f"doc{i}.txt", "text": f"chunk {i}: {body}"}
        for i in range(start, start + n)
    ]


def _single_block(question: str, chunks: list[dict]) -> str:
    """The OLD structure: one growing string block (what the code used before the fix)."""
    ctx = "\n\n".join(f"[{c['filename']}] {c['text']}" for c in chunks)
    return f"QUESTION: {question}\n\nCONTEXT:\n{ctx}"


def _one_call(cache_context) -> dict:
    sink = llm.new_usage()
    llm.complete(LLAMA_SYSTEM, NEXT_ACTION, max_tokens=MAX_TOKENS,
                 agent="probe", usage=sink, cache_context=cache_context)
    return sink


def _run(label: str, build) -> int:
    question = "Summarize the documents."
    turn1 = _make_chunks(N_TURN1)
    turn2 = turn1 + _make_chunks(N_NEW, start=N_TURN1)  # append-only: same chunks + new ones
    c1 = _one_call(build(question, turn1))
    c2 = _one_call(build(question, turn2))
    print(f"\n[{label}]")
    for i, s in enumerate((c1, c2), 1):
        print(f"  turn {i}: input={s['input_tokens']:>6}  "
              f"cache_read={s['cache_read_input_tokens']:>6}  "
              f"cache_write={s['cache_creation_input_tokens']:>6}")
    read2 = c2["cache_read_input_tokens"]
    print(f"  -> turn 2 {'READ fired (prefix reused)' if read2 > 0 else 'NO read (prefix not reused)'}")
    return read2


def main() -> None:
    s = get_settings()
    print(f"model={s.anthropic_model}  cache_mode={s.cache_mode}")
    if s.cache_mode == "off":
        print("cache_mode is off, so nothing caches. Set ANTHROPIC_CACHE_MODE=block and re-run.")
        return
    try:
        multi = _run("multi-block (current code, Prop. 2)", context_blocks)
        single = _run("single-block (old code, Prop. 4)", _single_block)
    except RuntimeError as e:  # require_api_key() raises this when the key is missing
        print(f"\n{e}")
        return

    print("\nSummary (turn-2 cache_read_input_tokens):")
    print(f"  multi-block  = {multi}")
    print(f"  single-block = {single}")
    if multi > 0 and single == 0:
        print("  => Fix confirmed: the multi-block context reads the prior chunks from cache; "
              "the single growing block does not (breakpoint drift).")
    elif multi > 0 and single > 0:
        print("  => Both read from cache on this model/SDK. The fix still helps by writing only "
              "the new chunks on turn 2 (compare the cache_write columns).")
    elif multi == 0:
        print("  => No read even for multi-block. The cached prefix is likely below this model's "
              "floor, raise the context size or use a lower-floor model (e.g. Sonnet).")


if __name__ == "__main__":
    main()
