"""Remove needle/floater Gaussians that show up as 'fractals' on scene edges."""
import sys
from pathlib import Path

import torch

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else str(Path(src).with_name("trained_gaussians_cleaned.pt"))
aniso_edge = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
aniso_hard = float(sys.argv[4]) if len(sys.argv) > 4 else 80.0

g = torch.load(src, map_location="cpu", weights_only=False)
means, scales, opacities = g["means"], g["scales"], g["opacities"]
aniso = scales.max(-1).values / scales.min(-1).values.clamp(min=1e-8)
radius = (means - means.median(0).values).norm(dim=-1)
max_scale = scales.max(-1).values
r90 = radius.quantile(0.90)

drop = (
    ((aniso > aniso_edge) & (radius > r90))
    | (aniso > aniso_hard)
    | (max_scale > 0.12)
    | (opacities < 0.015)
)
keep = ~drop
print(f"N {int(means.shape[0])} -> {int(keep.sum())}  dropped {int(drop.sum())} ({float(drop.float().mean()):.1%})")
print(f"  needles on the rim (aniso>{aniso_edge} and far): {int(((aniso>aniso_edge)&(radius>r90)).sum())}")
print(f"  extreme needles (aniso>{aniso_hard}): {int((aniso>aniso_hard).sum())}")
print(f"  huge scale: {int((max_scale>0.12).sum())}")
print(f"  near-transparent: {int((opacities<0.015).sum())}")

out = {k: (v[keep] if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == means.shape[0] else v) for k, v in g.items()}
out["step"] = g.get("step")
out["cleaned_from"] = src
torch.save(out, dst)
print(f"wrote {dst}")
