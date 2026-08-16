#!/usr/bin/env python3
"""
Decimate a .splat file (antimatter15 binary format) down to a target splat
count, for cases where a scene has too many splats to render smoothly in
real time (e.g. choppy framerate in Unity/VR).

Each splat is a fixed 32-byte record:
  position (3x float32) + scale (3x float32) + color (4x uint8 RGBA) +
  rotation quaternion (4x uint8)

This picks splats via EVEN sampling across the file (every Nth record),
not random or front-biased sampling — this preserves the overall spatial
distribution of the scene, so a decimated garden still reads as "a garden
viewed from many angles" rather than losing whole regions.

Usage:
    python3 decimate_splat.py garden.splat --target 400000
    python3 decimate_splat.py garden.splat --target 800000 -o garden_v2.splat

Try a few different --target values and look at each output side by side
before picking one — there's no universally "right" number, it depends on
how choppy the original is and how much quality loss is acceptable for
your scene.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

RECORD_BYTES = 32  # fixed size per splat in the .splat format


def decimate_splat(input_path: Path, target_count: int, output_path: Path) -> None:
    data = input_path.read_bytes()
    original_count = len(data) // RECORD_BYTES

    if original_count == 0:
        raise ValueError(f"{input_path} contains no splat records (or isn't a valid .splat file).")

    print(f"Input: {input_path.name} — {original_count:,} splats ({len(data) / 1024 / 1024:.1f} MB)")

    if target_count >= original_count:
        print(f"Target ({target_count:,}) >= original count ({original_count:,}) — copying unchanged.")
        output_path.write_bytes(data)
        print(f"Wrote {output_path.name} — {original_count:,} splats (no reduction applied)")
        return

    # Even sampling: pick every Nth record, where N = original / target.
    # This preserves spatial distribution across the whole scene rather
    # than biasing toward whatever was captured/written first.
    stride = original_count / target_count

    out_records = []
    for i in range(target_count):
        src_index = int(i * stride)
        start = src_index * RECORD_BYTES
        out_records.append(data[start:start + RECORD_BYTES])

    output_path.write_bytes(b"".join(out_records))

    actual_count = len(out_records)
    reduction = original_count / actual_count
    out_size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"Wrote {output_path.name} — {actual_count:,} splats ({out_size_mb:.1f} MB)")
    print(f"Reduction: {reduction:.1f}x smaller ({original_count:,} -> {actual_count:,})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decimate a .splat file to a target splat count.")
    parser.add_argument("input", type=Path, help="Path to the source .splat file")
    parser.add_argument(
        "--target", type=int, required=True,
        help="Target number of splats in the output (e.g. 400000)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: <input>_<target>.splat next to the input file)"
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output or args.input.with_stem(f"{args.input.stem}_{args.target}")
    if output_path.suffix != ".splat":
        output_path = output_path.with_suffix(".splat")

    decimate_splat(args.input, args.target, output_path)


if __name__ == "__main__":
    main()