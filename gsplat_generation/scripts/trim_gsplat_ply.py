"""Trim a gsplat/3DGS .ply without scrambling vertex records.

Reads properties by name (any source layout), subsets Gaussians, then writes a
Unity-compatible INRIA 3DGS binary PLY: x/y/z, dummy nx/ny/nz, f_dc, f_rest
(zeros if the source has no SH rest), opacity, scale, rot. Missing f_rest/nx
is what makes UnityGaussianSplatting look 'corrupted' (those fields are not
zero-filled on import).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


# Official 3DGS / UnityGaussianSplatting vertex (62 floats).
INRIA_PROPS: list[str] = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)

TYPE_TO_DTYPE = {
    b"float": np.float32,
    b"float32": np.float32,
    b"double": np.float64,
    b"float64": np.float64,
    b"uchar": np.uint8,
    b"uint8": np.uint8,
    b"int": np.int32,
    b"int32": np.int32,
}


def parse_ply(path: Path) -> tuple[int, list[tuple[str, np.dtype]], int]:
    """Return (n_vertices, [(name, dtype), ...], header_byte_len)."""
    props: list[tuple[str, np.dtype]] = []
    n = None
    fmt = None
    with path.open("rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file (starts with {first!r})")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing end_header")
            s = line.strip()
            if s == b"end_header":
                break
            parts = s.split()
            if len(parts) >= 3 and parts[0] == b"format":
                fmt = parts[1]
            if len(parts) >= 3 and parts[0] == b"element" and parts[1] == b"vertex":
                n = int(parts[2])
            if len(parts) >= 3 and parts[0] == b"property":
                typ, name = parts[1], parts[2].decode("ascii")
                if typ not in TYPE_TO_DTYPE:
                    raise ValueError(f"{path}: unsupported property type {typ!r}")
                props.append((name, np.dtype(TYPE_TO_DTYPE[typ])))
        header_len = f.tell()
    if n is None:
        raise ValueError(f"{path}: no element vertex count")
    if fmt != b"binary_little_endian":
        raise ValueError(f"{path}: need binary_little_endian, got {fmt!r}")
    if not props:
        raise ValueError(f"{path}: no vertex properties")
    return n, props, header_len


def load_ply_columns(path: Path) -> dict[str, np.ndarray]:
    n, props, header_len = parse_ply(path)
    dtype = np.dtype({"names": [p[0] for p in props], "formats": [p[1] for p in props]})
    expected = n * dtype.itemsize
    size = path.stat().st_size
    payload = size - header_len
    if payload != expected:
        raise ValueError(
            f"{path}: payload {payload} bytes != N({n}) * stride({dtype.itemsize}) = {expected}"
        )
    rec = np.fromfile(path, dtype=dtype, offset=header_len, count=n)
    if rec.shape[0] != n:
        raise ValueError(f"{path}: read {rec.shape[0]} vertices, header says {n}")
    out = {name: np.ascontiguousarray(rec[name], dtype=np.float32) for name, _ in props}
    return out


def write_inria_ply(path: Path, cols: dict[str, np.ndarray]) -> None:
    n = next(iter(cols.values())).shape[0]
    for v in cols.values():
        if v.shape[0] != n:
            raise ValueError("column length mismatch")
    table = np.zeros((n, len(INRIA_PROPS)), dtype=np.float32)
    for i, name in enumerate(INRIA_PROPS):
        if name in cols:
            table[:, i] = cols[name]
    # Identity quaternion (3DGS stores w,x,y,z) if rot is missing/zero.
    rot = table[:, -4:]
    norms = np.linalg.norm(rot, axis=1)
    bad = ~np.isfinite(norms) | (norms < 1e-8)
    if bad.any():
        rot[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norms = np.linalg.norm(rot, axis=1)
    table[:, -4:] = rot / norms[:, None]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in INRIA_PROPS)
        + "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(table, dtype=np.float32).tobytes())


def vggt_scene_box(vggt_path: Path, conf: float, pad: float):
    g = torch.load(vggt_path, map_location="cpu", weights_only=False)
    pts = g["point_map"].reshape(-1, 3).numpy()
    c = g["point_conf"].reshape(-1).numpy()
    sel = pts[c >= conf]
    if sel.shape[0] < 1000:
        sel = pts[c >= np.quantile(c, 0.7)]
    lo = np.percentile(sel, 2, axis=0)
    hi = np.percentile(sel, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - pad * span
    hi = hi + pad * span
    print(f"VGGT box from {sel.shape[0]} pts conf>={conf}: lo={lo} hi={hi}")
    return lo, hi


def require(cols: dict[str, np.ndarray], names: list[str]) -> None:
    missing = [n for n in names if n not in cols]
    if missing:
        raise ValueError(f"PLY missing required properties: {missing}")


def validate_inria_file(path: Path) -> None:
    n, props, header_len = parse_ply(path)
    names = [p[0] for p in props]
    if names != INRIA_PROPS:
        raise ValueError(
            f"{path}: property list is not INRIA 3DGS layout\n"
            f"  got {len(names)} props starting {names[:8]}..."
        )
    stride = sum(p[1].itemsize for p in props)
    expected = n * stride
    payload = path.stat().st_size - header_len
    if payload != expected:
        raise ValueError(f"{path}: size mismatch payload={payload} expected={expected}")
    rec = np.fromfile(path, dtype=np.float32, offset=header_len, count=n * len(INRIA_PROPS))
    rec = rec.reshape(n, len(INRIA_PROPS))
    if not np.isfinite(rec).all():
        raise ValueError(f"{path}: contains NaN/Inf")
    rot_n = np.linalg.norm(rec[:, -4:], axis=1)
    if (rot_n < 0.5).any() or (rot_n > 1.5).any():
        raise ValueError(f"{path}: quaternion norms out of range")
    print(
        f"OK {path}  N={n}  bytes={path.stat().st_size}  "
        f"xyz=[{rec[:, :3].min(0)} .. {rec[:, :3].max(0)}]"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--keep", type=int, default=150000)
    p.add_argument("--vggt", default="")
    p.add_argument("--conf", type=float, default=2.2)
    p.add_argument("--pad", type=float, default=0.08)
    p.add_argument("--aniso", type=float, default=20.0)
    p.add_argument("--max-scale", type=float, default=0.08)
    p.add_argument("--min-opacity", type=float, default=0.02)
    args = p.parse_args()

    cols = load_ply_columns(Path(args.src))
    require(cols, ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    n0 = cols["x"].shape[0]
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    opa_logit = cols["opacity"]
    opa = 1.0 / (1.0 + np.exp(-np.clip(opa_logit, -20, 20)))
    sc = np.exp(
        np.clip(np.stack([cols["scale_0"], cols["scale_1"], cols["scale_2"]], axis=1), -20, 20)
    )
    aniso = sc.max(1) / np.clip(sc.min(1), 1e-8, None)
    max_sc = sc.max(1)

    keep = np.isfinite(xyz).all(1) & np.isfinite(opa_logit) & np.isfinite(sc).all(1)
    if args.vggt:
        lo, hi = vggt_scene_box(Path(args.vggt), args.conf, args.pad)
        inside = np.all(xyz >= lo, axis=1) & np.all(xyz <= hi, axis=1)
        print(f"outside box: {int((~inside).sum())}")
        keep &= inside
    else:
        lo = np.percentile(xyz, 5, axis=0)
        hi = np.percentile(xyz, 95, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo, hi = lo - 0.05 * span, hi + 0.05 * span
        inside = np.all(xyz >= lo, axis=1) & np.all(xyz <= hi, axis=1)
        print(f"outside p05-p95 box: {int((~inside).sum())}")
        keep &= inside

    needles = aniso > args.aniso
    huge = max_sc > args.max_scale
    faint = opa < args.min_opacity
    print(f"needles aniso>{args.aniso}: {int(needles.sum())}")
    print(f"huge scale>{args.max_scale}: {int(huge.sum())}")
    print(f"opacity<{args.min_opacity}: {int(faint.sum())}")
    keep &= ~needles & ~huge & ~faint
    print(f"after crop/needles/opacity: {int(keep.sum())} / {n0}")

    idx = np.flatnonzero(keep)
    if idx.size > args.keep:
        score = opa[idx] / np.clip(max_sc[idx], 1e-6, None)
        take = np.argpartition(score, -args.keep)[-args.keep :]
        idx = idx[take]
        print(f"kept top {args.keep} by opacity/scale")

    out = {k: v[idx] for k, v in cols.items()}
    write_inria_ply(Path(args.dst), out)
    validate_inria_file(Path(args.dst))
    print(f"wrote {args.dst}  N={idx.size}  ({Path(args.dst).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
