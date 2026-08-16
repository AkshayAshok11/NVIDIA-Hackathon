"""Higher-quality VGGT to gsplat train: SSIM, SH, random views, scale regularizer."""
import argparse, glob, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/workspace/vggt")
import gsplat
from gsplat import rasterization
from gsplat.losses import scale_reg_loss, ssim_loss
from gsplat.strategy import DefaultStrategy
from vggt.utils.load_fn import load_and_preprocess_images

SH_C0 = 0.28209479177387814

def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0

def list_images(frames_dir):
    return sorted(
        glob.glob(f"{frames_dir}/*.png")
        + glob.glob(f"{frames_dir}/*.jpg")
        + glob.glob(f"{frames_dir}/*.jpeg")
        + glob.glob(f"{frames_dir}/*.JPG")
        + glob.glob(f"{frames_dir}/*.JPEG")
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vggt", default="/workspace/gsplat_generation/pipeline/garden/vggt_output.pt")
    p.add_argument("--frames", default="/workspace/Downloads/360_v2/garden/images_8")
    p.add_argument("--out", default="/workspace/gsplat_generation/pipeline/garden/gsplat_quality")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--conf", type=float, default=2.2)
    p.add_argument("--grow", type=float, default=0.00015)
    p.add_argument("--views-per-step", type=int, default=4)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--ssim-lambda", type=float, default=0.2)
    p.add_argument("--scale-reg", type=float, default=0.01)
    p.add_argument("--absgrad", action="store_true", default=True)
    p.add_argument("--no-absgrad", action="store_false", dest="absgrad")
    p.add_argument("--save-every", type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    print("gsplat version:", gsplat.__version__)
    print("args:", vars(args))

    d = torch.load(args.vggt, map_location=device, weights_only=False)
    extrinsic = d["extrinsic"][0].to(device)
    intrinsic = d["intrinsic"][0].to(device)
    point_map = d["point_map"][0].to(device)
    point_conf = d["point_conf"][0].to(device)
    C, H, W, _ = point_map.shape
    print(f"loaded VGGT: C={C} H={H} W={W}")

    image_names = list_images(args.frames)
    if len(image_names) > C:
        idx = [round(i * (len(image_names) - 1) / (C - 1)) for i in range(C)]
        image_names = [image_names[i] for i in idx]
        print(f"subsampled images to C={C}")
    elif len(image_names) != C:
        raise SystemExit(f"found {len(image_names)} images but VGGT has C={C}")
    images = load_and_preprocess_images(image_names).to(device)
    assert images.shape[-2:] == (H, W)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, :3, :4] = extrinsic
    Ks = intrinsic
    gt_images = images.permute(0, 2, 3, 1).contiguous()

    pts = point_map[:, ::args.stride, ::args.stride].reshape(-1, 3)
    conf = point_conf[:, ::args.stride, ::args.stride].reshape(-1)
    cols = images[:, :, ::args.stride, ::args.stride].permute(0, 2, 3, 1).reshape(-1, 3)
    keep = conf > args.conf
    pts, cols = pts[keep], cols[keep]
    N = pts.shape[0]
    print(f"init {N} Gaussians (stride={args.stride}, conf>{args.conf})")

    bbox_diag = (pts.max(0).values - pts.min(0).values).norm().item()
    init_scale = max(bbox_diag / (N ** (1 / 3)) * 0.5, 1e-4)
    print(f"bbox_diag={bbox_diag:.4f} init_scale={init_scale:.5f}")

    sh_dim = (args.sh_degree + 1) ** 2
    sh = torch.zeros(N, sh_dim, 3, device=device)
    sh[:, 0, :] = rgb_to_sh(cols.clamp(0, 1))
    quats_init = torch.zeros(N, 4, device=device)
    quats_init[:, 0] = 1.0
    params = {
        "means": nn.Parameter(pts.clone()),
        "quats": nn.Parameter(quats_init),
        "scales": nn.Parameter(torch.full((N, 3), init_scale, device=device).log()),
        "opacities": nn.Parameter(torch.full((N,), 0.5, device=device).logit()),
        "sh0": nn.Parameter(sh[:, :1].clone()),
        "shN": nn.Parameter(sh[:, 1:].clone()),
    }
    lrs = {
        "means": 1.6e-3 * bbox_diag,
        "quats": 1e-2,
        "scales": 5e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k]) for k in params}
    means_lr_init = lrs["means"]
    means_lr_final = means_lr_init * 0.01
    num_steps = args.steps
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=args.grow,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        refine_start_iter=max(200, num_steps // 16),
        refine_stop_iter=int(num_steps * 0.85),
        refine_every=max(50, num_steps // 80),
        reset_every=3000 if num_steps >= 6000 else num_steps * 10,
        absgrad=args.absgrad,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=bbox_diag)

    def colors_for_render():
        return torch.cat([params["sh0"], params["shN"]], dim=1)

    def save_gaussians(path, step=None):
        sh_all = colors_for_render().detach()
        rgb = (sh_all[:, 0, :] * SH_C0 + 0.5).clamp(0, 1).cpu()
        torch.save({
            "means": params["means"].detach().cpu(),
            "quats": F.normalize(params["quats"].detach(), dim=-1).cpu(),
            "scales": torch.exp(params["scales"]).detach().cpu(),
            "opacities": torch.sigmoid(params["opacities"]).detach().cpu(),
            "colors": rgb,
            "sh0": params["sh0"].detach().cpu(),
            "shN": params["shN"].detach().cpu(),
            "step": step,
        }, path)
        extra = f" (step {step})" if step is not None else ""
        print(f"saved{extra} -> {path}  N={params['means'].shape[0]}")

    print(f"\n{num_steps} steps, {args.views_per_step} views/step, SH={args.sh_degree}, ssim={args.ssim_lambda}")
    t0 = time.time()
    losses = []
    interrupted = False
    try:
        for step in range(num_steps):
            t = step / max(num_steps - 1, 1)
            optimizers["means"].param_groups[0]["lr"] = means_lr_init * (means_lr_final / means_lr_init) ** t
            cam_idx = torch.randint(0, C, (args.views_per_step,), device=device)
            sh_degree_to_use = min(step // max(500, num_steps // 8), args.sh_degree)
            render_colors, render_alphas, meta = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors_for_render(),
                viewmats=viewmats[cam_idx],
                Ks=Ks[cam_idx],
                width=W, height=H,
                near_plane=1e-3, far_plane=1e3,
                render_mode="RGB", packed=False,
                absgrad=args.absgrad, sh_degree=sh_degree_to_use,
            )
            strategy.step_pre_backward(params, optimizers, strategy_state, step, meta)
            gt = gt_images[cam_idx]
            l1 = F.l1_loss(render_colors, gt)
            ssim = ssim_loss(render_colors.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
            loss = torch.lerp(l1, ssim, args.ssim_lambda)
            if args.scale_reg > 0:
                loss = loss + args.scale_reg * scale_reg_loss(params["scales"])
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
                print(f"  step {step:5d}  loss={loss.item():.5f}  l1={l1.item():.5f}  ssim={ssim.item():.5f}  SH={sh_degree_to_use}  N={params['means'].shape[0]}  ({elapsed:.1f}s)")
            if step % args.save_every == 0 or step == num_steps - 1:
                save_gaussians(f"{args.out}/trained_gaussians.pt", step=step)
                save_gaussians(f"{args.out}/ckpt_step{step:05d}.pt", step=step)
    except KeyboardInterrupt:
        interrupted = True
        last = len(losses) - 1
        print(f"\ninterrupted at step {last}")
        save_gaussians(f"{args.out}/trained_gaussians.pt", step=last)

    elapsed = time.time() - t0
    print(f"\ntraining {'interrupted' if interrupted else 'done'} in {elapsed:.1f}s")
    print(f"Gaussian count: {N} -> {params['means'].shape[0]}")
    if losses:
        print(f"loss first={losses[0]:.5f} last={losses[-1]:.5f}")

    with torch.no_grad():
        final_render, _, _ = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors_for_render(),
            viewmats=viewmats, Ks=Ks, width=W, height=H,
            near_plane=1e-3, far_plane=1e3, render_mode="RGB", packed=False,
            sh_degree=args.sh_degree,
        )
        mse = F.mse_loss(final_render, gt_images).item()
        psnr = -10 * np.log10(mse + 1e-10)
        print(f"final PSNR all {C} views: {psnr:.2f} dB")
        for idx in [0, C // 2, C - 1]:
            pred = (final_render[idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            gt = (gt_images[idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(np.concatenate([gt, pred], axis=1)).save(f"{args.out}/compare_frame{idx:03d}.png")
            print(f"saved {args.out}/compare_frame{idx:03d}.png")
    save_gaussians(f"{args.out}/trained_gaussians.pt", step=len(losses) - 1 if losses else None)

if __name__ == "__main__":
    main()
