#!/usr/bin/env python3
"""
Ingest a video file into the Timebox memory bank.

Pipeline: video -> sample frames -> VLM caption -> Lightning enrichment ->
embed (note + caption + enrichment) -> store in FAISS + metadata.

The enrichment step asks Lightning to generate a few alternative phrasings
of what the memory is likely about, beyond the literal note/caption text.
This widens what gets embedded, so a thin note ("bday") plus a caption that
never says the word "birthday" can still be found later by a query that
does say "birthday" — the enrichment step is what bridges that gap.

Usage:
    python3 scripts/ingest_memory.py path/to/video.mp4 --note "18th birthday party"
    python3 scripts/ingest_memory.py path/to/video.mp4 --note "bday" --scene path/to/trained_splat_scene
    python3 scripts/ingest_memory.py path/to/video.mp4 --note "bday" --no-enrich  # skip enrichment step
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
from memory_pipeline.video_utils import get_frames
from memory_pipeline.vlm_captioner import VLMCaptioner


def ingest(video_path: str, note: str, scene_path: str | None = None, enrich: bool = True) -> None:
    print(f"[1/5] Extracting frames from {video_path} ...")
    frames = get_frames(video_path)
    print(f"      -> {len(frames)} frames sampled")

    print("[2/5] Captioning with local VLM ...")
    gpu_swap.ensure_vlm_active()
    captioner = VLMCaptioner()
    caption = captioner.caption_frames(frames, user_note=note)
    print(f"      -> caption: {caption[:120]}{'...' if len(caption) > 120 else ''}")

    enrichment = ""
    if enrich:
        print("[3/5] Generating alternative phrasings with Lightning ...")
        engine = QueryEngine()
        try:
            enrichment = engine.enrich_memory_text(note, caption)
            if enrichment:
                print(f"      -> {enrichment[:120]}{'...' if len(enrichment) > 120 else ''}")
            else:
                print("      -> (nothing additional inferred, skipping)")
        except Exception as e:
            # Enrichment is a nice-to-have, not core to the pipeline — never
            # let a failure here block ingestion of the actual memory.
            print(f"      -> enrichment failed ({e}), continuing without it")
    else:
        print("[3/5] Skipping enrichment (--no-enrich)")

    print("[4/5] Embedding note + caption + enrichment with local Embed model ...")
    gpu_swap.ensure_embed_active()
    embedder = Embedder()
    searchable_text = " | ".join(p for p in [note, caption, enrichment] if p)
    embedding = embedder.embed_document(searchable_text)
    print(f"      -> vector dim: {embedding.shape[0]}")

    print("[5/5] Storing in memory bank ...")
    store = MemoryStore()
    memory = store.add_memory(
        note=note,
        caption=caption,
        video_path=video_path,
        embedding=embedding,
        scene_path=scene_path,
        enrichment=enrichment,
    )
    print(f"      -> stored as memory {memory.id}")
    print(f"\nDone. Memory bank now holds {len(store)} memories.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a video into the Timebox memory bank.")
    parser.add_argument("video_path", help="Path to the video file to ingest.")
    parser.add_argument("--note", default="", help="User's note for this memory (can be sparse).")
    parser.add_argument(
        "--scene", default=None, help="Path to the trained 3D Gaussian splat scene for this memory, if available yet."
    )
    parser.add_argument(
        "--no-enrich", action="store_true", help="Skip the Lightning enrichment step (faster, less searchable)."
    )
    args = parser.parse_args()

    ingest(args.video_path, args.note, args.scene, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()