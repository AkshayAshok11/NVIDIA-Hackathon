"""Preview a VGGT .pt before gsplat: colored PLY, orbit HTML, camera reprojections."""
import glob
import os
import sys
import base64
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/vggt")
from vggt.utils.load_fn import load_and_preprocess_images

vggt_out_path = sys.argv[1]
frames_dir = sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else "/workspace/gsplat_generation/pipeline/garden/preview"
max_images = int(sys.argv[4]) if len(sys.argv) > 4 else 0
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 4

os.makedirs(out_dir, exist_ok=True)

d = torch.load(vggt_out_path, map_location="cpu", weights_only=False)
extrinsic = d["extrinsic"][0].numpy().astype(np.float32)  # [C, 3, 4]
intrinsic = d["intrinsic"][0].numpy().astype(np.float32)  # [C, 3, 3]
point_map = d["point_map"][0].numpy().astype(np.float32)  # [C, H, W, 3]
point_conf = d["point_conf"][0].numpy().astype(np.float32)  # [C, H, W]
C, H, W, _ = point_map.shape
print(f"VGGT: C={C} H={H} W={W}")

image_names = sorted(
    glob.glob(f"{frames_dir}/*.png")
    + glob.glob(f"{frames_dir}/*.jpg")
    + glob.glob(f"{frames_dir}/*.jpeg")
    + glob.glob(f"{frames_dir}/*.JPG")
    + glob.glob(f"{frames_dir}/*.JPEG")
)
if max_images > 0 and len(image_names) > max_images:
    idx = [round(i * (len(image_names) - 1) / (max_images - 1)) for i in range(max_images)]
    image_names = [image_names[i] for i in idx]
if len(image_names) != C:
    raise SystemExit(f"image count {len(image_names)} != VGGT cameras {C}. Pass the same max_images used at inference.")

images = load_and_preprocess_images(image_names).cpu().numpy()  # [C, 3, H, W]
if images.shape[-2:] != (H, W):
    raise SystemExit(f"preprocessed images {images.shape[-2:]} != point map {(H, W)}")
rgb = np.transpose(images, (0, 2, 3, 1))  # [C, H, W, 3]

pts = point_map[:, ::stride, ::stride].reshape(-1, 3)
conf = point_conf[:, ::stride, ::stride].reshape(-1)
cols = rgb[:, ::stride, ::stride].reshape(-1, 3)

# Garden conf is not floored like the aerial clip; keep the upper 60%.
conf_thresh = float(np.quantile(conf, 0.4))
keep = conf > conf_thresh
pts, cols, conf = pts[keep], cols[keep], conf[keep]
print(f"kept {len(pts)} points (stride={stride}, conf>{conf_thresh:.3f})")

# --- PLY (MeshLab / CloudCompare / Blender) ---
ply_path = os.path.join(out_dir, "pointcloud.ply")
cols_u8 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
header = (
    "ply\nformat binary_little_endian 1.0\n"
    f"element vertex {len(pts)}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    "end_header\n"
)
vert = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
vert["x"], vert["y"], vert["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
vert["r"], vert["g"], vert["b"] = cols_u8[:, 0], cols_u8[:, 1], cols_u8[:, 2]
with open(ply_path, "wb") as f:
    f.write(header.encode("ascii"))
    f.write(vert.tobytes())
print(f"wrote {ply_path}")


def project(pts_xyz, extr, K):
    R, t = extr[:, :3], extr[:, 3]
    xyz = pts_xyz @ R.T + t
    z = xyz[:, 2]
    uvw = xyz @ K.T
    u = uvw[:, 0] / np.clip(z, 1e-6, None)
    v = uvw[:, 1] / np.clip(z, 1e-6, None)
    return u, v, z


def render_view(cam_idx, max_points=400_000):
    """Z-buffer the fused point cloud into camera cam_idx."""
    sel = np.arange(len(pts))
    if len(sel) > max_points:
        sel = np.random.default_rng(0).choice(sel, max_points, replace=False)
    u, v, z = project(pts[sel], extrinsic[cam_idx], intrinsic[cam_idx])
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    valid = (z > 1e-4) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z, c = ui[valid], vi[valid], z[valid], cols[sel][valid]
    depth = np.full((H, W), np.inf, np.float32)
    out = np.zeros((H, W, 3), np.float32)
    # nearer points overwrite (process far-to-near)
    order = np.argsort(z)[::-1]
    depth[vi[order], ui[order]] = z[order]
    out[vi[order], ui[order]] = c[order]
    hit = np.isfinite(depth)
    return out, hit


for idx in [0, C // 2, C - 1]:
    pred, hit = render_view(idx)
    gt = np.clip(rgb[idx], 0, 1)
    pred_img = pred.copy()
    pred_img[~hit] = 0.08
    side = np.concatenate([gt, pred_img], axis=1)
    side_u8 = (np.clip(side, 0, 1) * 255).astype(np.uint8)
    path = os.path.join(out_dir, f"reproject_{idx:03d}.png")
    Image.fromarray(side_u8).save(path)
    print(f"wrote {path}  coverage={hit.mean():.1%}  (left=photo, right=VGGT points)")

# --- orbit HTML (same binary layout as the splat viewer, tiny radii) ---
center = pts.mean(axis=0)
extent = np.percentile(np.linalg.norm(pts - center, axis=1), 99.0)
positions = (pts - center) / max(extent, 1e-6)
radius = np.full((len(pts),), 0.004, np.float32)
dtype = np.dtype([("pos", "<f4", 3), ("radius", "<f4"), ("rgb", "u1", 3), ("opacity", "u1")])
records = np.zeros(len(pts), dtype=dtype)
records["pos"] = positions.astype(np.float32)
records["radius"] = radius
records["rgb"] = cols_u8
records["opacity"] = np.full((len(pts),), 255, np.uint8)
blob = np.uint32(len(pts)).tobytes() + records.tobytes()
bin_path = os.path.join(out_dir, "pointcloud.bin")
Path(bin_path).write_bytes(blob)

tmpl = Path("/workspace/gsplat_generation/scripts/splat_viewer_template.html").read_text()
b64 = base64.b64encode(blob).decode("ascii")
html = (
    tmpl.replace("__SPLAT_DATA_BASE64__", b64)
    .replace("Great Wall, Jinshanling", "Garden — VGGT point cloud")
    .replace("24 frames, VGGT+gsplat", f"{C} views, VGGT points only")
    .replace("<dt>Gaussians</dt>", "<dt>Points</dt>")
)
html_path = os.path.join(out_dir, "pointcloud_viewer.html")
Path(html_path).write_text(html)
print(f"wrote {html_path}")
print("\nOpen the HTML in a browser (drag to orbit, scroll to zoom).")
print("Or open the PLY in MeshLab / CloudCompare.")
