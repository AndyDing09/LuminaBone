"""
Turn one location's all-on relief into a watertight SOLID 3-D model, scaled to
real-world units.
================================================================================

Front = reconstructed bone surface, closed with a flat back + side walls into a
watertight solid. Coordinates are written in a real physical unit so the model
imports at true size.

Scale is set from ONE measurement: TARGET_WIDTH = real width of the area a photo
covers (left edge to right edge). Everything (x, y and depth) is multiplied by
the same inches-per-pixel factor, so proportions are TRUE (no vertical
exaggeration).

Caveat on depth: single-image shading is not depth-calibrated, so the relief
height is the reconstruction's native scale (uniformly scaled with x/y so it
isn't stretched). If you ever measure the real peak-to-valley height, set
NATIVE_DEPTH_SCALE to match and it becomes metric in z too.

Run:  python make_solid_from_allon.py
"""

import os
import struct
import numpy as np
from PIL import Image

import endoscope_depth as ed

LOC = 1
SRC = f"./bone_picture/p{LOC}-1.jpg"
OUT_PLY = f"./depth_outputs/loc{LOC:02d}_solid_in.ply"    # colored, for viewing
OUT_STL = f"./depth_outputs/loc{LOC:02d}_solid_in.stl"    # for CAD / 3-D print

UNIT_NAME = "inch"
TARGET_WIDTH = 1.0          # real width of the photographed area, in UNIT_NAME
LONG_EDGE = 480             # sampling resolution of the model grid
NATIVE_DEPTH_SCALE = 30.0   # reconstruction's native relief (pixels of z per depth unit)
BACK_MARGIN = 0.05          # base thickness behind the deepest point, in UNIT_NAME
BACK_RGB = (165, 165, 165)


def load_rgb(path, long_edge):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = long_edge / max(w, h)
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_depth(rgb):
    spec = ed.detect_specular_mask(rgb)
    cleaned = rgb.copy()
    for c in range(3):
        cleaned[..., c] = ed.inpaint_mask(rgb[..., c], spec)
    lum = ed.to_luminance(cleaned)
    return ed.bilateral_smooth(ed.brightness_to_depth(lum), lum)


def perimeter_ring(H, W):
    idx = np.arange(H * W).reshape(H, W)
    return np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])


def write_ply(path, V, F):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(V)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(F)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    fr = np.empty(len(F), np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")]))
    fr["n"] = 3; fr["a"] = F[:, 0]; fr["b"] = F[:, 1]; fr["c"] = F[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii")); V.tofile(fh); fr.tofile(fh)


def write_stl(path, xyz, F):
    p0, p1, p2 = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 0, n / ln, 0.0)
    rec = np.zeros(len(F), dtype=[("v", "<f4", 12), ("a", "<u2")])
    rec["v"] = np.concatenate([n, p0, p1, p2], axis=1).astype("<f4")
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(F)))
        rec.tofile(fh)


def main():
    rgb = load_rgb(SRC, LONG_EDGE)
    depth = compute_depth(rgb)
    H, W = depth.shape
    N = H * W
    ipp = TARGET_WIDTH / W                      # units per pixel (same for x,y,z)

    j, i = np.meshgrid(np.arange(W), np.arange(H))
    X = ((j - W / 2.0) * ipp).astype(np.float32)
    Y = (-(i - H / 2.0) * ipp).astype(np.float32)
    Zf = (-depth * NATIVE_DEPTH_SCALE * ipp).astype(np.float32)   # true-proportion depth
    zback = float(Zf.min() - BACK_MARGIN)

    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    V = np.empty(2 * N, vdt)
    V["x"][:N] = X.ravel(); V["y"][:N] = Y.ravel(); V["z"][:N] = Zf.ravel()
    cols = (np.clip(rgb, 0, 1).reshape(-1, 3) * 255).astype(np.uint8)
    V["r"][:N] = cols[:, 0]; V["g"][:N] = cols[:, 1]; V["b"][:N] = cols[:, 2]
    V["x"][N:] = X.ravel(); V["y"][N:] = Y.ravel(); V["z"][N:] = zback
    V["r"][N:], V["g"][N:], V["b"][N:] = BACK_RGB

    idx = np.arange(N).reshape(H, W)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel();  d = idx[1:, 1:].ravel()
    front = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)], 0)
    back = np.concatenate([np.stack([a + N, d + N, b + N], 1),
                           np.stack([a + N, c + N, d + N], 1)], 0)
    ring = perimeter_ring(H, W); r2 = np.roll(ring, -1)
    walls = np.concatenate([np.stack([ring, r2, r2 + N], 1),
                            np.stack([ring, r2 + N, ring + N], 1)], 0)
    F = np.concatenate([front, back, walls], 0)

    write_ply(OUT_PLY, V, F)
    xyz = np.stack([V["x"], V["y"], V["z"]], axis=1).astype(np.float32)
    write_stl(OUT_STL, xyz, F)

    dims = (W * ipp, H * ipp, Zf.max() - Zf.min())
    print(f"solid ({UNIT_NAME}) -> {OUT_PLY}  and  {OUT_STL}")
    print(f"  {len(V):,} verts, {len(F):,} faces")
    print(f"  size: {dims[0]:.3f} x {dims[1]:.3f} {UNIT_NAME} (W x H), "
          f"relief {dims[2]:.3f} {UNIT_NAME}, base +{BACK_MARGIN} -> "
          f"total thickness {dims[2] + BACK_MARGIN:.3f} {UNIT_NAME}")


if __name__ == "__main__":
    main()
