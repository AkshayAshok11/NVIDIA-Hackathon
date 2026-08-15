"""
Frame extraction from a video file, for feeding into the VLM.

We don't send every frame to the VLM — that's wasteful and the VLM's context
handles a handful of representative frames just fine for a short memory clip.
Sample at a fixed interval, capped at MAX_FRAMES_PER_VIDEO.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2

from . import config


def extract_frames(video_path: str | Path, out_dir: Path | None = None) -> list[Path]:
    """
    Sample frames from a video at config.FRAME_SAMPLE_INTERVAL_SEC, capped at
    config.MAX_FRAMES_PER_VIDEO. Saves them as JPEGs and returns their paths.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = out_dir or (config.FRAMES_DIR / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * config.FRAME_SAMPLE_INTERVAL_SEC))

    saved_paths: list[Path] = []
    frame_idx = 0

    try:
        while len(saved_paths) < config.MAX_FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = out_dir / f"frame_{len(saved_paths):03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(frame_path)

            frame_idx += 1
    finally:
        cap.release()

    if not saved_paths:
        raise RuntimeError(
            f"No frames extracted from {video_path} — check the file is a valid video."
        )

    return saved_paths


def image_to_base64(image_path: str | Path) -> str:
    """Read an image file and return a base64-encoded string for VLM payloads."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
