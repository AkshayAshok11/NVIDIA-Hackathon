#!/usr/bin/env python3
"""
Query the Timebox memory bank with a natural-language request.

Pipeline: raw query -> Lightning cleans it -> embed -> FAISS search ->
Lightning decides match vs. ambiguous vs. no match.

Usage:
    python3 scripts/query_memory.py "I want to revisit my 18th birthday"
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


def query(raw_query: str) -> None:
    engine = QueryEngine()
    embedder = Embedder()
    store = MemoryStore()

    if len(store) == 0:
        print("Memory bank is empty — ingest a memory first with ingest_memory.py")
        return

    print(f"[1/3] Cleaning query with Lightning: '{raw_query}'")
    clean = engine.clean_query(raw_query)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Timebox memory bank.")
    parser.add_argument("query", help="Natural-language request, e.g. 'revisit my 18th birthday'")
    args = parser.parse_args()

    query(args.query)


if __name__ == "__main__":
    main()