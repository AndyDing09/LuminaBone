"""
Build a colored SURFACE MESH (PLY with faces) per location from the all-on photo.
================================================================================

Same depth as make_ply_from_allon.py (single-image inverse-square shading of the
p{loc}-1 all-LEDs-on photo), but exported as a triangulated surface mesh instead
of a bare point cloud. The pixel grid is turned into two triangles per cell, so
the PLY has real FACES -- which is what 3D Slicer (and most viewers) need in
order to render a visible surface. Vertices carry the photo's RGB color.

Output: binary PLY (compact, Slicer-friendly) in depth_outputs/ply_allon_mesh/.
Run:  python make_mesh_from_allon.py
"""

import os
import numpy as np
from PIL import Image

import endoscope_depth as ed
import bone_depth_batch as bd

OUT_DIR = "./depth_outputs/ply_allon_mesh"
ALL_ON_INDEX = 1
WORK_LONG_EDGE = 640
DEPTH_SCALE = 30.0          # vertical exaggeration (same as the point clouds)


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


def write_mesh_ply(path, X, Y, Z, rgb):
    """Write a colored triangle-mesh PLY (binary_little_endian) over the grid."""
    H, W = Z.shape
    n_v = H * W

    # vertices (+ colors)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.empty(n_v, vdt)
    v["x"] = X.ravel(); v["y"] = Y.ravel(); v["z"] = Z.ravel()
    cols = (np.clip(rgb, 0, 1).reshape(-1, 3) * 255).astype(np.uint8)
    v["r"] = cols[:, 0]; v["g"] = cols[:, 1]; v["b"] = cols[:, 2]

    # two triangles per grid cell
    idx = np.arange(n_v).reshape(H, W)
    v00 = idx[:-1, :-1].ravel(); v01 = idx[:-1, 1:].ravel()
    v10 = idx[1:, :-1].ravel();  v11 = idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([v00, v01, v11], 1),
                           np.stack([v00, v11, v10], 1)], 0)
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    f = np.empty(len(tris), fdt)
    f["n"] = 3
    f["a"] = tris[:, 0]; f["b"] = tris[:, 1]; f["c"] = tris[:, 2]

    header = ("ply\n"
              "format binary_little_endian 1.0\n"
              f"element vertex {n_v}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(tris)}\n"
              "property list uchar int vertex_indices\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        v.tofile(fh)
        f.tofile(fh)
    return n_v, len(tris)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    n = 0
    for loc in sorted(groups):
        files = groups[loc]
        if ALL_ON_INDEX not in files:
            print(f"  [loc {loc}] SKIP - no p{loc}-{ALL_ON_INDEX}")
            continue
        rgb = load_rgb(files[ALL_ON_INDEX], WORK_LONG_EDGE)
        depth = compute_depth(rgb)
        H, W = depth.shape
        j, i = np.meshgrid(np.arange(W), np.arange(H))
        X = (j - W / 2.0).astype(np.float32)
        Y = -(i - H / 2.0).astype(np.float32)
        Z = (-depth * DEPTH_SCALE).astype(np.float32)
        out = os.path.join(OUT_DIR, f"loc{loc:02d}_allon_mesh.ply")
        nv, nf = write_mesh_ply(out, X, Y, Z, rgb)
        n += 1
        print(f"  loc {loc:2d}: {nv:>7} verts, {nf:>7} faces -> {os.path.basename(out)}")
    print(f"\n{n} mesh PLYs -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
