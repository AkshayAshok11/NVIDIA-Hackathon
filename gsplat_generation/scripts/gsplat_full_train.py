import sys, glob, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, "/workspace/vggt")
import gsplat
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda"
print("gsplat version:", gsplat.__version__)

vggt_out_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/timebox_smoketest/vggt_output_great_wall.pt"
frames_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/timebox_smoketest/frames_great_wall"
point_stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
num_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
out_dir = sys.argv[5] if len(sys.argv) > 5 else "/workspace/timebox_smoketest/gsplat_full_render"
CONF_THRESH = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
GROW_GRAD2D = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0002

os.makedirs(out_dir, exist_ok=True)

d = torch.load(vggt_out_path, map_location=device)
extrinsic = d["extrinsic"][0].to(device)   # [C, 3, 4]
intrinsic = d["intrinsic"][0].to(device)   # [C, 3, 3]
point_map = d["point_map"][0].to(device)   # [C, H, W, 3]
point_conf = d["point_conf"][0].to(device) # [C, H, W]
C, H, W, _ = point_map.shape
print(f"loaded VGGT output: C={C} H={H} W={W}")

image_names = sorted(
    glob.glob(f"{frames_dir}/*.png")
    + glob.glob(f"{frames_dir}/*.jpg")
    + glob.glob(f"{frames_dir}/*.jpeg")
    + glob.glob(f"{frames_dir}/*.JPG")
    + glob.glob(f"{frames_dir}/*.JPEG")
)
if len(image_names) > C:
    idx = [round(i * (len(image_names) - 1) / (C - 1)) for i in range(C)]
    image_names = [image_names[i] for i in idx]
    print(f"subsampled {len(idx)} images to match VGGT camera count C={C}")
elif len(image_names) != C:
    raise SystemExit(f"found {len(image_names)} images but VGGT has C={C} cameras")
images = load_and_preprocess_images(image_names).to(device)  # [C, 3, H, W]
assert images.shape[-2:] == (H, W)

# use every frame as a training view
viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
viewmats[:, :3, :4] = extrinsic
Ks = intrinsic
gt_images = images.permute(0, 2, 3, 1).contiguous()  # [C, H, W, 3]

pts = point_map[:, ::point_stride, ::point_stride, :].reshape(-1, 3)
conf = point_conf[:, ::point_stride, ::point_stride].reshape(-1)
cols = images[:, :, ::point_stride, ::point_stride].permute(0, 2, 3, 1).reshape(-1, 3)

# NOTE: point_conf is floored at exactly 1.0 for ~60% of raw points (mostly sky/
# horizon/textureless regions in this aerial footage) -- a *quantile* threshold at
# low percentiles collides with that floor plateau and barely filters anything.
# Use an absolute cutoff instead, clear of the floor.
keep = conf > CONF_THRESH
pts, cols = pts[keep], cols[keep]
N = pts.shape[0]
print(f"initialized {N} Gaussians from point cloud (stride={point_stride}, conf_thresh={CONF_THRESH} absolute)")

bbox_diag = (pts.max(0).values - pts.min(0).values).norm().item()
init_scale = max(bbox_diag / (N ** (1 / 3)) * 0.5, 1e-4)
print(f"bbox_diag={bbox_diag:.4f} init_scale={init_scale:.5f}")

quats_init = torch.zeros(N, 4, device=device)
quats_init[:, 0] = 1.0

# params dict + one optimizer per key, exactly as gsplat.strategy.DefaultStrategy
# requires (see check_sanity: params/optimizers must share keys, and each optimizer
# must have exactly one param_group). scales/opacities/colors are stored raw
# (pre-activation) and only exponentiated/sigmoided at call sites -- DefaultStrategy's
# internal prune/grow heuristics also expect raw scales/opacities in this form.
params = {
    "means": nn.Parameter(pts.clone()),
    "quats": nn.Parameter(quats_init),
    "scales": nn.Parameter(torch.full((N, 3), init_scale, device=device).log()),
    "opacities": nn.Parameter(torch.full((N,), 0.5, device=device).logit()),
    "colors": nn.Parameter(cols.clone().clamp(1e-4, 1 - 1e-4).logit()),
}
lrs = {
    "means": 1.6e-3 * bbox_diag,
    "quats": 1e-2,
    "scales": 5e-3,
    "opacities": 5e-2,
    "colors": 2.5e-2,
}
optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}
means_lr_init = lrs["means"]
means_lr_final = means_lr_init * 0.01  # standard 3DGS practice: decay position LR ~100x

# refine cadence scaled down for an 800-1500 step budget (DefaultStrategy defaults
# assume a 15k-30k step run). scene_scale grounds the scale-based grow/prune
# thresholds (grow_scale3d/prune_scale3d are fractions of this).
strategy = DefaultStrategy(
    prune_opa=0.005,
    grow_grad2d=GROW_GRAD2D,
    grow_scale3d=0.01,
    prune_scale3d=0.1,
    refine_start_iter=max(10, num_steps // 16),
    refine_stop_iter=int(num_steps * 0.9),
    refine_every=max(5, num_steps // 32),
    reset_every=num_steps * 10,  # effectively disabled -- too few remaining steps to recover from a full opacity reset this late
    verbose=True,
)
strategy.check_sanity(params, optimizers)
strategy_state = strategy.initialize_state(scene_scale=bbox_diag)

SAVE_EVERY = 100  # write a checkpoint every N steps so Ctrl+C still leaves an exportable file

def save_gaussians(path, step=None):
    torch.save({
        "means": params["means"].detach().cpu(),
        "quats": params["quats"].detach().cpu(),
        "scales": torch.exp(params["scales"]).detach().cpu(),
        "opacities": torch.sigmoid(params["opacities"]).detach().cpu(),
        "colors": torch.sigmoid(params["colors"]).detach().cpu(),
        "step": step,
    }, path)
    extra = f" (step {step})" if step is not None else ""
    print(f"saved trained Gaussians{extra} to {path}")

print(f"\nrunning {num_steps} training steps on all {C} views (starting at {N} Gaussians, DefaultStrategy will prune/grow)...")
print(f"checkpoints every {SAVE_EVERY} steps -> {out_dir}/trained_gaussians.pt  (Ctrl+C also saves)")
t0 = time.time()
losses = []
interrupted = False
try:
    for step in range(num_steps):
        t = step / max(num_steps - 1, 1)
        optimizers["means"].param_groups[0]["lr"] = means_lr_init * (means_lr_final / means_lr_init) ** t

        render_colors, render_alphas, meta = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.sigmoid(params["colors"]),
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=1e-3,
            far_plane=1e3,
            render_mode="RGB",
            packed=False,
        )
        strategy.step_pre_backward(params, optimizers, strategy_state, step, meta)

        loss = F.l1_loss(render_colors, gt_images)

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()

        strategy.step_post_backward(params, optimizers, strategy_state, step, meta, packed=False)

        for opt in optimizers.values():
            opt.step()

        with torch.no_grad():
            params["quats"].data = F.normalize(params["quats"].data, dim=-1)

        losses.append(loss.item())
        if step % 100 == 0 or step == num_steps - 1:
            elapsed = time.time() - t0
            print(f"  step {step:4d}  loss={loss.item():.5f}  N={params['means'].shape[0]}  ({elapsed:.1f}s elapsed)")
        if step % SAVE_EVERY == 0 or step == num_steps - 1:
            save_gaussians(f"{out_dir}/trained_gaussians.pt", step=step)
            save_gaussians(f"{out_dir}/ckpt_step{step:04d}.pt", step=step)
except KeyboardInterrupt:
    interrupted = True
    last = len(losses) - 1
    print(f"\ninterrupted at step {last}")
    save_gaussians(f"{out_dir}/trained_gaussians.pt", step=last)

elapsed = time.time() - t0
N_final = params["means"].shape[0]
done_label = "interrupted" if interrupted else "done"
print(f"\ntraining {done_label} in {elapsed:.1f}s ({elapsed/max(len(losses),1)*1000:.1f}ms/step)")
print(f"Gaussian count: {N} -> {N_final}")
if losses:
    print(f"loss: first={losses[0]:.5f} -> last={losses[-1]:.5f}  (ratio={losses[-1]/losses[0]:.3f})")

with torch.no_grad():
    final_render, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=torch.sigmoid(params["colors"]),
        viewmats=viewmats, Ks=Ks, width=W, height=H,
        near_plane=1e-3, far_plane=1e3, render_mode="RGB", packed=False,
    )
    mse = F.mse_loss(final_render, gt_images).item()
    psnr = -10 * np.log10(mse + 1e-10)
    print(f"final PSNR across all {C} training views: {psnr:.2f} dB")

    for idx in [0, C // 2, C - 1]:
        pred = (final_render[idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gt = (gt_images[idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        side_by_side = np.concatenate([gt, pred], axis=1)
        Image.fromarray(side_by_side).save(f"{out_dir}/compare_frame{idx:03d}.png")
        print(f"saved {out_dir}/compare_frame{idx:03d}.png (left=ground truth, right=gsplat render)")

save_gaussians(f"{out_dir}/trained_gaussians.pt", step=len(losses) - 1 if losses else None)

if interrupted:
    print("\nSTOPPED EARLY — export trained_gaussians.pt whenever you want.")
else:
    assert torch.isfinite(torch.tensor(losses[-1]))
    assert losses[-1] < losses[0] * 0.5, "loss did not drop enough for a real training run"
    print("\nFULL TRAINING RUN PASSED.")
