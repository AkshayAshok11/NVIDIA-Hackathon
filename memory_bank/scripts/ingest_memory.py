#!/usr/bin/env python3
"""
Ingest a video file into the Timebox memory bank.

Pipeline: video -> sample frames -> VLM caption -> embed (note + caption) ->
store in FAISS + metadata.

Usage:
    python3 scripts/ingest_memory.py path/to/video.mp4 --note "18th birthday party"
    python3 scripts/ingest_memory.py path/to/video.mp4 --note "bday" --scene path/to/trained_splat_scene
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_pipeline.embedder import Embedder
from memory_pipeline.memory_store import MemoryStore
from memory_pipeline.video_utils import extract_frames
from memory_pipeline.vlm_captioner import VLMCaptioner


def ingest(video_path: str, note: str, scene_path: str | None = None) -> None:
    print(f"[1/4] Extracting frames from {video_path} ...")
    frames = extract_frames(video_path)
    print(f"      -> {len(frames)} frames sampled")

    print("[2/4] Captioning with local VLM ...")
    captioner = VLMCaptioner()
    caption = captioner.caption_frames(frames, user_note=note)
    print(f"      -> caption: {caption[:120]}{'...' if len(caption) > 120 else ''}")

    print("[3/4] Embedding note + caption with local Embed model ...")
    embedder = Embedder()
    searchable_text = " | ".join(p for p in [note, caption] if p)
    embedding = embedder.embed_document(searchable_text)
    print(f"      -> vector dim: {embedding.shape[0]}")

    print("[4/4] Storing in memory bank ...")
    store = MemoryStore()
    memory = store.add_memory(
        note=note,
        caption=caption,
        video_path=video_path,
        embedding=embedding,
        scene_path=scene_path,
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
    args = parser.parse_args()

    ingest(args.video_path, args.note, args.scene)


if __name__ == "__main__":
    main()
