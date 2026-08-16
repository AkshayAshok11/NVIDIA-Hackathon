"""Crop/prune a post-gsplat trained_gaussians.pt, then you re-export .ply/.splat."""
import argparse
from pathlib import Path

import numpy as np
import torch


def vggt_scene_box(vggt_path: Path, conf: float, pad: float):
    g = torch.load(vggt_path, map_location="cpu", weights_only=False)
    pts = g["point_map"].reshape(-1, 3)
    c = g["point_conf"].reshape(-1)
    sel = pts[c >= conf]
    if sel.shape[0] < 1000:
        sel = pts[c >= torch.quantile(c, 0.7)]
    lo = torch.quantile(sel, 0.02, dim=0)
    hi = torch.quantile(sel, 0.98, dim=0)
    span = (hi - lo).clamp(min=1e-6)
    lo = lo - pad * span
    hi = hi + pad * span
    print(f"VGGT box from {int(sel.shape[0])} pts conf>={conf}: lo={lo.tolist()} hi={hi.tolist()}")
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--keep", type=int, default=150000)
    p.add_argument("--vggt", required=True)
    p.add_argument("--conf", type=float, default=2.2)
    p.add_argument("--pad", type=float, default=0.08)
    p.add_argument("--aniso", type=float, default=20.0)
    p.add_argument("--max-scale", type=float, default=0.08)
    p.add_argument("--min-opacity", type=float, default=0.02)
    args = p.parse_args()

    g = torch.load(args.src, map_location="cpu", weights_only=False)
    means = g["means"].float()
    scales = g["scales"].float().clamp(min=1e-8)
    opacities = g["opacities"].float()
    n0 = means.shape[0]
    aniso = scales.max(-1).values / scales.min(-1).values.clamp(min=1e-8)
    max_sc = scales.max(-1).values

    lo, hi = vggt_scene_box(Path(args.vggt), args.conf, args.pad)
    inside = ((means >= lo) & (means <= hi)).all(dim=1)
    needles = aniso > args.aniso
    huge = max_sc > args.max_scale
    faint = opacities < args.min_opacity
    print(f"N={n0}  outside={int((~inside).sum())}  needles={int(needles.sum())}  huge={int(huge.sum())}  faint={int(faint.sum())}")
    keep = inside & ~needles & ~huge & ~faint
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    print(f"after crop/needles/opacity: {int(idx.numel())}")
    if idx.numel() > args.keep:
        score = opacities[idx] / max_sc[idx].clamp(min=1e-6)
        take = torch.topk(score, args.keep).indices
        idx = idx[take]
        print(f"kept top {args.keep} by opacity/scale")

    out = {}
    for k, v in g.items():
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n0:
            out[k] = v[idx]
        else:
            out[k] = v
    out["trimmed_from"] = args.src
    torch.save(out, args.dst)
    print(f"wrote {args.dst}  N={int(idx.numel())}")


if __name__ == "__main__":
    main()
