"""
Stitch the 17 overlapping all-on frames into ONE continuous colored surface.
================================================================================

The 17 locations are NOT separate specimens -- they are a slow camera pan across
one continuous spine region (p1..p17 sweep left->right; the same foramina, facet
joints and ink marks march across consecutive frames). So they can be stitched.

Pipeline:
  1. OpenCV Stitcher (PANORAMA): feature match + global bundle adjustment +
     multi-band blend across all 17 frames -> one sharp fused mosaic. (A hand-rolled
     consecutive-homography chain drifts into a keystoned fan and ghosts the
     overlaps; the bundle-adjusted stitcher does not.)
  2. Clean the mosaic's coverage mask (largest region, fill holes, erode the blend
     border) so the surface has no ragged black edge.
  3. Reconstruct relief from the fused mosaic with the verified single-image shading
     method (endoscope_depth) -> one continuous depth surface.
  4. Export a colored triangle MESH + a point cloud PLY in the SAME pixel-unit
     convention as depth_outputs/ply_allon_mesh (X=col-W/2, Y=-(row-H/2),
     Z=-depth*30), so the registration/ ICP + landmark tooling consumes it directly.

CAVEAT (honest): the panorama is a projective/cylindrical mosaic, so it does NOT
recover the spine's true metric 3D curvature -- relief comes only from per-pixel
shading. It gives ICP a much wider, better-constrained target than any single patch,
but treat it as a wide "orthophoto with relief", not a metric reconstruction. For
true 3D curvature you need SfM/photogrammetry (recover camera poses + triangulate).

Run:  python stitch_bone_surface.py
"""

import os
import re
import glob
import numpy as np
import cv2
from scipy import ndimage

import endoscope_depth as ed

SRC_GLOB = "bone_picture/p*-1.jpg"
OUT_DIR = "depth_outputs/stitched"
MATCH_LONG_EDGE = 1280      # resolution frames are fed to the stitcher at
SURF_LONG_EDGE = 1300       # output surface sampling (keeps PLY manageable)
DEPTH_SCALE = 30.0          # relief exaggeration (matches the per-patch meshes)
BORDER_ERODE = 6            # px eroded off the coverage edge (drop blend seam)


def load_frames():
    fs = sorted(glob.glob(SRC_GLOB),
                key=lambda p: int(re.search(r"p(\d+)-", p).group(1)))
    frames = []
    for f in fs:
        bgr = cv2.imread(f)
        h, w = bgr.shape[:2]
        s = MATCH_LONG_EDGE / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)),
                             interpolation=cv2.INTER_AREA)
        frames.append(bgr)
    return frames, [os.path.basename(f) for f in fs]


def stitch_panorama(frames):
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = st.stitch(frames)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"stitching failed (status {status})")
    return pano


def coverage_mask(pano):
    """Clean binary mask of the real (non-black) mosaic region."""
    gray = pano.mean(2)
    m = gray > 6
    m = ndimage.binary_closing(m, iterations=3)
    lbl, n = ndimage.label(m)
    if n > 1:                                   # keep the largest blob only
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_erosion(m, iterations=BORDER_ERODE)
    return m


def reconstruct_depth(rgb01):
    spec = ed.detect_specular_mask(rgb01)
    cleaned = rgb01.copy()
    for c in range(3):
        cleaned[..., c] = ed.inpaint_mask(rgb01[..., c], spec)
    lum = ed.to_luminance(cleaned)
    return ed.bilateral_smooth(ed.brightness_to_depth(lum), lum)


def write_mesh_ply(path, X, Y, Z, rgb01, valid):
    """Colored triangle mesh over the valid region (cells with 4 valid corners)."""
    H, W = valid.shape
    vid = -np.ones((H, W), np.int64)
    ys, xs = np.where(valid)
    vid[ys, xs] = np.arange(len(ys))
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.empty(len(ys), vdt)
    v["x"] = X[ys, xs]; v["y"] = Y[ys, xs]; v["z"] = Z[ys, xs]
    cols = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    v["r"] = cols[ys, xs, 0]; v["g"] = cols[ys, xs, 1]; v["b"] = cols[ys, xs, 2]

    c = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a00 = vid[:-1, :-1][c]; a01 = vid[:-1, 1:][c]
    a10 = vid[1:, :-1][c];  a11 = vid[1:, 1:][c]
    tris = np.concatenate([np.stack([a00, a01, a11], 1),
                           np.stack([a00, a11, a10], 1)], 0)
    fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    f = np.empty(len(tris), fdt)
    f["n"] = 3; f["a"] = tris[:, 0]; f["b"] = tris[:, 1]; f["c"] = tris[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(tris)}\n"
              "property list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii")); v.tofile(fh); f.tofile(fh)
    return len(v), len(tris)


def write_cloud_ply(path, X, Y, Z, rgb01, valid):
    ys, xs = np.where(valid)
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.empty(len(ys), vdt)
    v["x"] = X[ys, xs]; v["y"] = Y[ys, xs]; v["z"] = Z[ys, xs]
    cols = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    v["r"] = cols[ys, xs, 0]; v["g"] = cols[ys, xs, 1]; v["b"] = cols[ys, xs, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii")); v.tofile(fh)
    return len(v)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames, names = load_frames()
    print(f"loaded {len(frames)} frames: {names[0]} .. {names[-1]}")

    print("stitching panorama (bundle-adjusted, multi-band blend)...")
    pano = stitch_panorama(frames)
    print(f"  raw mosaic: {pano.shape[1]} x {pano.shape[0]}")

    # downsample to surface sampling resolution
    Hc, Wc = pano.shape[:2]
    s = SURF_LONG_EDGE / max(Hc, Wc)
    if s < 1.0:
        pano = cv2.resize(pano, (round(Wc * s), round(Hc * s)),
                          interpolation=cv2.INTER_AREA)

    valid = coverage_mask(pano)
    rgb01 = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    print("reconstructing relief from fused mosaic...")
    depth = reconstruct_depth(rgb01)

    H, W = depth.shape
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    X = (j - W / 2.0).astype(np.float32)
    Y = -(i - H / 2.0).astype(np.float32)
    Z = (-np.where(valid, depth, 0.0) * DEPTH_SCALE).astype(np.float32)

    mesh_path = os.path.join(OUT_DIR, "spine_stitched_mesh.ply")
    cloud_path = os.path.join(OUT_DIR, "spine_stitched_cloud.ply")
    nv, nf = write_mesh_ply(mesh_path, X, Y, Z, rgb01, valid)
    ncp = write_cloud_ply(cloud_path, X, Y, Z, rgb01, valid)

    # inspection artifacts
    cv2.imwrite(os.path.join(OUT_DIR, "fused_mosaic.png"), pano)
    dv = np.where(valid, depth, np.nan)
    lo, hi = np.nanpercentile(dv, 5), np.nanpercentile(dv, 95)
    dn = np.nan_to_num(np.clip((dv - lo) / (hi - lo + 1e-9), 0, 1))
    heat = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    cv2.imwrite(os.path.join(OUT_DIR, "fused_depth.png"), heat)

    mb = os.path.getsize(mesh_path) / 1024 / 1024
    print("\n==== DONE ====")
    print(f"valid surface pixels: {int(valid.sum()):,} of {valid.size:,}")
    print(f"mesh  -> {mesh_path}  ({nv:,} verts, {nf:,} faces, {mb:.1f} MB)")
    print(f"cloud -> {cloud_path}  ({ncp:,} points)")
    print(f"mosaic-> {OUT_DIR}/fused_mosaic.png   depth -> {OUT_DIR}/fused_depth.png")


if __name__ == "__main__":
    main()
