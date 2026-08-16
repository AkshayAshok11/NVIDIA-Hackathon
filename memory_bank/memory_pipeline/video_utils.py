"""
Frame extraction for feeding into the VLM — supports both video files and
folders of individual images (photo-based captures, NeRF-style datasets
like Mip-NeRF 360, or any multi-photo scan of a room/scene).

We don't send every frame to the VLM — that's wasteful and the VLM's context
handles a handful of representative frames just fine for a short memory clip.
Sample at a fixed interval (video) or evenly across the set (images), capped
at MAX_FRAMES_PER_VIDEO either way.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2

from . import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_frames(source_path: str | Path, out_dir: Path | None = None) -> list[Path]:
    """
    Unified entry point: figures out whether source_path is a video file or
    a folder of images, and dispatches to the right extraction method. This
    is what ingest_memory.py should call — callers don't need to know or
    care which kind of source they were given.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source_path.is_dir():
        return extract_frames_from_folder(source_path, out_dir)
    return extract_frames(source_path, out_dir)


def extract_frames_from_folder(folder_path: str | Path, out_dir: Path | None = None) -> list[Path]:
    """
    Sample frames from a folder of still images (e.g. a NeRF-style
    multi-angle photo capture, or any set of photos of the same room/scene).
    Picks up to MAX_FRAMES_PER_VIDEO images, evenly spaced through the
    sorted file list — so a folder of 200 photos taken all the way around
    a room gives the VLM a representative spread of angles, not just the
    first few alphabetically.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    image_files = sorted(
        p for p in folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise RuntimeError(
            f"No image files found in {folder_path} "
            f"(looked for: {', '.join(sorted(IMAGE_EXTENSIONS))})"
        )

    # Evenly sample up to MAX_FRAMES_PER_VIDEO images across the full set,
    # rather than just taking the first N — for a folder of many angles,
    # this gives the VLM a spread across the whole scene instead of a
    # cluster of near-duplicate adjacent shots.
    n = len(image_files)
    cap = config.MAX_FRAMES_PER_VIDEO
    if n <= cap:
        selected = image_files
    else:
        step = n / cap
        selected = [image_files[int(i * step)] for i in range(cap)]

    out_dir = out_dir or (config.FRAMES_DIR / folder_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for i, src in enumerate(selected):
        # Re-encode as JPEG into FRAMES_DIR for consistency with the video
        # path (same downstream handling either way — vlm_captioner.py
        # doesn't need to know the source was a folder, not a video), and
        # to keep FRAMES_DIR as the single place all sampled frames live,
        # regardless of source type.
        img = cv2.imread(str(src))
        if img is None:
            continue  # skip unreadable files rather than failing the whole ingest
        dest = out_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(dest), img)
        saved_paths.append(dest)

    if not saved_paths:
        raise RuntimeError(
            f"Found {n} image file(s) in {folder_path} but none were readable."
        )

    return saved_paths


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