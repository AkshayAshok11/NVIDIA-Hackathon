"""Convert trained_gaussians.pt (activated tensors) to 3DGS .ply and antimatter15 .splat."""
import sys
from pathlib import Path

import torch
from gsplat import export_splats

SH_C0 = 0.28209479177387814

pt_path = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) > 2 else str(Path(pt_path).with_suffix(""))

g = torch.load(pt_path, map_location="cpu", weights_only=False)
means = g["means"].float()
scales = g["scales"].float().clamp(min=1e-8)          # activated
opacities = g["opacities"].float().clamp(1e-6, 1 - 1e-6)
colors = g["colors"].float().clamp(0.0, 1.0)
quats = torch.nn.functional.normalize(g["quats"].float(), dim=-1)

# exporters expect log-scale, logit-opacity, SH DC
log_scales = scales.log()
logit_opa = torch.logit(opacities)
sh0 = ((colors - 0.5) / SH_C0).unsqueeze(1)  # [N, 1, 3]
shN = torch.zeros(means.shape[0], 0, 3)

N = means.shape[0]
print(f"loaded {N} gaussians from {pt_path}")

ply_path = out_prefix + ".ply"
splat_path = out_prefix + ".splat"
export_splats(means, log_scales, quats, logit_opa, sh0, shN, format="ply", save_to=ply_path)
export_splats(means, log_scales, quats, logit_opa, sh0, shN, format="splat", save_to=splat_path)
print(f"wrote {ply_path} ({Path(ply_path).stat().st_size / 1e6:.2f} MB)")
print(f"wrote {splat_path} ({Path(splat_path).stat().st_size / 1e6:.2f} MB)")
