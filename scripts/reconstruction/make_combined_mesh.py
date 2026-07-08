"""
Stitch all 17 per-location meshes into ONE colored mesh PLY (grid layout).
================================================================================

The 17 locations are separate specimens, and each mesh is centered on the
origin -- so simply merging them would stack all 17 on top of each other. To
make a single model you can view/import as a whole, we lay them out on a grid:
each location keeps its own shape + color and is just translated into its own
cell (row-major by location number). The result is one binary PLY with faces,
so it imports straight into 3D Slicer as a single Model.

Depth = the same single-image (all-on) shading used for the individual meshes,
recomputed here at a lighter resolution so the combined file stays manageable.

Run:  python make_combined_mesh.py
"""

import os
import numpy as np
from PIL import Image

import endoscope_depth as ed
import bone_depth_batch as bd

OUT_PATH = bd.project_path("depth_outputs", "all17_combined_mesh.ply")
ALL_ON_INDEX = 1
TILE_LONG_EDGE = 320        # per-tile resolution (smaller => lighter combined file)
DEPTH_SCALE = 30.0
NCOLS = 5                    # grid columns
GAP_X = 1.15                # horizontal spacing as a fraction of tile width
GAP_Y = 1.35                # vertical spacing as a fraction of tile height


def load_rgb(path, long_edge):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = long_edge / max(w, h)
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def tile_depth(rgb):
    spec = ed.detect_specular_mask(rgb)
    cleaned = rgb.copy()
    for c in range(3):
        cleaned[..., c] = ed.inpaint_mask(rgb[..., c], spec)
    lum = ed.to_luminance(cleaned)
    return ed.bilateral_smooth(ed.brightness_to_depth(lum), lum)


def grid_faces(H, W, base):
    """Two triangles per cell, indices offset by `base` (this tile's 1st vertex)."""
    idx = base + np.arange(H * W).reshape(H, W)
    v00 = idx[:-1, :-1].ravel(); v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel();  v11 = idx[1:, 1:].ravel()
    return np.concatenate([np.stack([v00, v01, v11], 1),
                           np.stack([v00, v11, v10], 1)], 0)


def main():
    groups = bd.discover(bd.INPUT_FOLDER)
    locs = [l for l in sorted(groups) if ALL_ON_INDEX in groups[l]]
    n = len(locs)
    nrows = (n + NCOLS - 1) // NCOLS

    # figure out tile size from the first location (all share it)
    r0 = load_rgb(groups[locs[0]][ALL_ON_INDEX], TILE_LONG_EDGE)
    H, W = r0.shape[:2]
    dx = W * GAP_X
    dy = H * GAP_Y
    # offsets that center the whole grid on the origin
    x0 = -(NCOLS - 1) * dx / 2.0
    y0 = (nrows - 1) * dy / 2.0

    all_v, all_f = [], []
    base = 0
    for k, loc in enumerate(locs):
        col, row = k % NCOLS, k // NCOLS
        rgb = load_rgb(groups[loc][ALL_ON_INDEX], TILE_LONG_EDGE)
        depth = tile_depth(rgb)
        H, W = depth.shape
        j, i = np.meshgrid(np.arange(W), np.arange(H))
        X = (j - W / 2.0) + (x0 + col * dx)
        Y = -(i - H / 2.0) + (y0 - row * dy)
        Z = -depth * DEPTH_SCALE

        vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        v = np.empty(H * W, vdt)
        v["x"] = X.ravel(); v["y"] = Y.ravel(); v["z"] = Z.ravel()
        cols = (np.clip(rgb, 0, 1).reshape(-1, 3) * 255).astype(np.uint8)
        v["r"] = cols[:, 0]; v["g"] = cols[:, 1]; v["b"] = cols[:, 2]
        all_v.append(v)
        all_f.append(grid_faces(H, W, base))
        base += H * W
        print(f"  placed loc {loc:2d} at grid (row {row}, col {col})")

    V = np.concatenate(all_v)
    F = np.concatenate(all_f)
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    frec = np.empty(len(F), fdt)
    frec["n"] = 3
    frec["a"] = F[:, 0]; frec["b"] = F[:, 1]; frec["c"] = F[:, 2]

    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(V)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(F)}\n"
              "property list uchar int vertex_indices\n"
              "end_header\n")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        fh.write(header.encode("ascii"))
        V.tofile(fh); frec.tofile(fh)

    mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\nCombined {n} locations ({nrows}x{NCOLS} grid) -> {OUT_PATH}")
    print(f"  {len(V):,} vertices, {len(F):,} faces, {mb:.1f} MB")


if __name__ == "__main__":
    main()
