#!/usr/bin/env python3
"""
Query the Timebox memory bank with a natural-language request.

Pipeline: raw query -> Lightning cleans it -> embed -> FAISS search ->
Lightning decides match vs. ambiguous vs. no match.

Supports two modes:
  - Single query (original behavior): one request, no memory of prior turns.
  - Interactive (--chat): keeps a running conversation so follow-ups like
    "no, the other one" can resolve against what was actually shown, not
    just the previous cleaned query string. This matters because Lightning
    can only resolve "the other one" if the conversation history actually
    names what "one" and "the other" refer to — passing back only the
    cleaned query text (the original approach) gives it nothing concrete
    to resolve against.

Usage:
    python3 scripts/query_memory.py "I want to revisit my 18th birthday"
    python3 scripts/query_memory.py --chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_pipeline import gpu_swap
from memory_pipeline.embedder import Embedder
from memory_pipeline.memory_store import MemoryStore
from memory_pipeline.query_engine import QueryEngine


def _summarize_for_history(result, results) -> str:
    """
    Build the text that goes into the conversation history as Lightning's
    own prior turn — this is what makes "no, the other one" resolvable
    later. Naming the match AND the runner-up alternatives gives the next
    turn something concrete to refer back to, rather than just the cleaned
    search string (which carries no information about what was found).
    """
    if result.status == "match":
        alt_names = [m.note or m.caption[:40] for m, _ in results[1:3]]
        alt_note = f" (other candidates: {', '.join(alt_names)})" if alt_names else ""
        return f"Loading memory: {result.memory.note or result.memory.caption[:60]}{alt_note}"
    elif result.status == "ambiguous":
        names = [m.note or m.caption[:40] for m in result.candidates]
        return f"Multiple matches found: {', '.join(names)}. {result.message}"
    else:
        return result.message


def run_query(raw_query: str, engine: QueryEngine, embedder: Embedder, store: MemoryStore, history: list[dict] | None = None):
    """One query turn. Returns (result, results, cleaned_query) so callers
    (interactive mode) can build history for the next turn."""
    print(f"[1/3] Cleaning query with Lightning: '{raw_query}'")
    clean = engine.clean_query(raw_query, conversation_history=history)
    print(f"      -> cleaned: '{clean}'")

    print("[2/3] Embedding query and searching memory bank ...")
    gpu_swap.ensure_embed_active()
    query_vec = embedder.embed_query(clean)
    results = store.search(query_vec)
    for memory, score in results:
        print(f"      -> {score:.3f}  {memory.note or memory.caption[:60]}")

    print("[3/3] Resolving with Lightning ...")
    result = engine.resolve_results(raw_query, results)

    print(f"\nStatus: {result.status}")
    print(f"Response: {result.message}")

    if result.status == "match":
        print(f"\n-> Load scene: {result.memory.scene_path or '(no 3D scene attached yet)'}")
    elif result.status == "ambiguous":
        print("\nCandidates:")
        for m in result.candidates:
            print(f"  - {m.id}: {m.note or m.caption[:60]}")

    return result, results, clean


def query(raw_query: str) -> None:
    """Single-shot query, no conversation history — original behavior,
    kept for simple one-off use."""
    engine = QueryEngine()
    embedder = Embedder()
    store = MemoryStore()

    if len(store) == 0:
        print("Memory bank is empty — ingest a memory first with ingest_memory.py")
        return

    run_query(raw_query, engine, embedder, store)


def chat() -> None:
    """Interactive multi-turn mode. Maintains real history — including what
    was actually retrieved — so follow-ups like 'no, the other one' have
    something concrete to resolve against."""
    engine = QueryEngine()
    embedder = Embedder()
    store = MemoryStore()

    if len(store) == 0:
        print("Memory bank is empty — ingest a memory first with ingest_memory.py")
        return

    print("Interactive mode. Type a query, or 'quit' to exit.\n")
    history: list[dict] = []

    while True:
        raw_query = input("> ").strip()
        if not raw_query or raw_query.lower() in ("quit", "exit"):
            break

        result, results, clean = run_query(raw_query, engine, embedder, store, history=history)

        # Extend history with THIS turn: the raw query, and a summary of
        # what was actually found — not just the cleaned query text.
        history.append({"role": "user", "content": raw_query})
        history.append({"role": "assistant", "content": _summarize_for_history(result, results)})
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Timebox memory bank.")
    parser.add_argument(
        "query", nargs="?", default=None, help="Natural-language request, e.g. 'revisit my 18th birthday'"
    )
    parser.add_argument("--chat", action="store_true", help="Interactive multi-turn mode with follow-up support.")
    args = parser.parse_args()

    if args.chat:
        chat()
    elif args.query:
        query(args.query)
    else:
        parser.error("Provide a query, or use --chat for interactive mode.")


if __name__ == "__main__":
    main()