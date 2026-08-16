import sys
import torch
import numpy as np

gaussians_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/timebox_smoketest/gsplat_full_render/trained_gaussians.pt"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/timebox_smoketest/splat_export.bin"

g = torch.load(gaussians_path, map_location="cpu")
means = g["means"].numpy().astype(np.float32)      # [N, 3]
scales = g["scales"].numpy().astype(np.float32)    # [N, 3]
opacities = g["opacities"].numpy().astype(np.float32)  # [N]
colors = g["colors"].numpy().astype(np.float32)    # [N, 3]

N = means.shape[0]
print(f"loaded {N} gaussians")

center = means.mean(axis=0)
extent = np.percentile(np.linalg.norm(means - center, axis=1), 99.0)
positions = (means - center) / extent  # roughly fit in a unit-radius ball

radius = scales.mean(axis=1) / extent  # average scale, normalized to same space

colors_u8 = np.clip(colors, 0.0, 1.0)
colors_u8 = (colors_u8 * 255.0).astype(np.uint8)  # [N, 3]

opacities_u8 = np.clip(opacities, 0.0, 1.0)
opacities_u8 = (opacities_u8 * 255.0).astype(np.uint8)  # [N]

print(f"position range after normalization: min={positions.min(axis=0)} max={positions.max(axis=0)}")
print(f"radius range: min={radius.min():.5f} max={radius.max():.5f} mean={radius.mean():.5f}")

# binary layout per point (20 bytes): pos.xyz f32 (12) + radius f32 (4) + rgb u8 (3) + opacity u8 (1)
dtype = np.dtype([("pos", "<f4", 3), ("radius", "<f4"), ("rgb", "u1", 3), ("opacity", "u1")])
records = np.zeros(N, dtype=dtype)
records["pos"] = positions.astype(np.float32)
records["radius"] = radius.astype(np.float32)
records["rgb"] = colors_u8
records["opacity"] = opacities_u8

with open(out_path, "wb") as f:
    f.write(np.uint32(N).tobytes())
    f.write(records.tobytes())

import os
size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"wrote {out_path} ({size_mb:.2f} MB, {N} points, 20 bytes/point + 4 byte header)")
