import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vggt_out_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/timebox_smoketest/vggt_output_great_wall.pt"
out_png = sys.argv[2] if len(sys.argv) > 2 else "/workspace/timebox_smoketest/vggt_pointcloud.png"

d = torch.load(vggt_out_path, map_location="cpu")
extrinsic = d["extrinsic"][0].numpy()   # [C, 3, 4]
point_map = d["point_map"][0].numpy()   # [C, H, W, 3]
point_conf = d["point_conf"][0].numpy() # [C, H, W]
C, H, W, _ = point_map.shape

stride = 6
pts = point_map[:, ::stride, ::stride, :].reshape(-1, 3)
conf = point_conf[:, ::stride, ::stride].reshape(-1)

conf_thresh = np.quantile(conf, 0.5)
keep = conf > conf_thresh
pts = pts[keep]

cam_positions = np.stack([
    -extrinsic[c, :, :3].T @ extrinsic[c, :, 3] for c in range(C)
])

# color points by height (z) for readability since we don't have per-point RGB here
colors = pts[:, 1]
colors = (colors - colors.min()) / (np.ptp(colors) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(pts[:, 0], pts[:, 2], c=colors, cmap="viridis", s=1, alpha=0.5)
ax.plot(cam_positions[:, 0], cam_positions[:, 2], "r.-", markersize=8, linewidth=1.5, label="camera path")
ax.scatter(cam_positions[0, 0], cam_positions[0, 2], c="lime", s=80, marker="^", zorder=5, label="start")
ax.scatter(cam_positions[-1, 0], cam_positions[-1, 2], c="red", s=80, marker="v", zorder=5, label="end")
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_title("Top-down view (X-Z)")
ax.legend(fontsize=8)
ax.set_aspect("equal")

ax = axes[1]
ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap="viridis", s=1, alpha=0.5)
ax.plot(cam_positions[:, 0], cam_positions[:, 1], "r.-", markersize=8, linewidth=1.5, label="camera path")
ax.scatter(cam_positions[0, 0], cam_positions[0, 1], c="lime", s=80, marker="^", zorder=5, label="start")
ax.scatter(cam_positions[-1, 0], cam_positions[-1, 1], c="red", s=80, marker="v", zorder=5, label="end")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Side view (X-Y)")
ax.legend(fontsize=8)
ax.invert_yaxis()
ax.set_aspect("equal")

fig.suptitle(f"VGGT reconstruction: {C} cameras, {pts.shape[0]} points (conf-filtered, stride={stride})")
plt.tight_layout()
plt.savefig(out_png, dpi=130)
print(f"saved {out_png}")
print(f"num points plotted: {pts.shape[0]}")
print(f"camera positions range: x[{cam_positions[:,0].min():.3f},{cam_positions[:,0].max():.3f}] "
      f"y[{cam_positions[:,1].min():.3f},{cam_positions[:,1].max():.3f}] "
      f"z[{cam_positions[:,2].min():.3f},{cam_positions[:,2].max():.3f}]")
