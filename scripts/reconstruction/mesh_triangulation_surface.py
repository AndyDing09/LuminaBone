"""
Mesh the 17 triangulation depth patches into 3-D surfaces.
================================================================================

Two products, both from the SAME three-LED photometric-triangulation depth used
by make_triangulation_slides.py (measured light geometry: symmetric equal LEDs
at 12.08 mm offset, 75 deg elevation, depth scaled to the 37 mm field):

  1. PER-LOCATION colored triangle meshes  -> depth_outputs/meshes/locNN_mesh.ply
     Each is the 3-D surface reconstruction of one patch, vertices in mm
     (X,Y in the image plane, Z = height toward the camera).

  2. ONE combined surface  -> depth_outputs/meshes/combined_surface.ply (mesh)
                              depth_outputs/meshes/combined_surface_points.ply
     The 17 locations are a slow left->right pan across one spine region (~92%
     overlap frame-to-frame), so they stitch. We estimate each frame's offset
     from the previous one (Hanning-windowed phase correlation on the all-on
     frames, low-confidence shifts replaced by the robust median), lay the
     patches on one canvas, align each patch's relative-depth DC level to its
     neighbours in the overlap, and feather-blend depth + colour into one
     continuous relief surface, then mesh it.

HONEST CAVEAT: the stitch is a TRANSLATIONAL relief mosaic (the pan is treated
as in-plane translation). It recovers fine surface relief across a wide area but
NOT the spine's true global 3-D curvature -- for that use the CT registration in
registration/. Treat the combined surface as a "wide orthophoto with relief".

Run:  python mesh_triangulation_surface.py
"""

import os
import numpy as np
import cv2

import bone_depth_batch as bd
# Reuse the corrected, measured light geometry from the slide generator.
from make_triangulation_slides import (LED_AZIMUTH_DEG, LIGHT_ELEVATION_DEG,
                                       ORDER)

OUT_DIR = bd.project_path("depth_outputs", "meshes")
STITCH_LONG_EDGE = 480          # working resolution for depth + stitch canvas
MM_PER_IN = 25.4
MIN_PHASE_RESPONSE = 0.10       # below this, a shift estimate is untrusted
MIN_OVERLAP_PX = 400            # need this many overlap pixels to align a patch


# ---------------------------------------------------------------------------
# Per-location triangulation depth (mm), colour and bone mask
# ---------------------------------------------------------------------------

def location_depth_mm(files):
    """Triangulation depth in mm + all-on colour + bone mask for one location."""
    rgbs = [bd.load_rgb(files[i], STITCH_LONG_EDGE) for i in ORDER]
    lums = bd.balance_exposure([bd.solve_luminance(r) for r in rgbs])
    Ls = np.array([bd.light_vector(LED_AZIMUTH_DEG[i], LIGHT_ELEVATION_DEG)
                   for i in ORDER])
    normals, albedo = bd.photometric_stereo(lums, Ls)
    z = bd.normals_to_depth(normals)                 # relative, working-px units

    H, W = z.shape
    mm_per_px = (bd.FIELD_WIDTH_IN * MM_PER_IN) / W  # same scale as the slides
    z_mm = z * mm_per_px
    mask = bd.bone_mask(albedo)
    z_mm = z_mm - (np.median(z_mm[mask]) if mask.any() else np.median(z_mm))

    # Colour: all-on frame if present, else the mean of the three LED frames.
    if 1 in files:
        color = bd.load_rgb(files[1], STITCH_LONG_EDGE)
    else:
        color = np.mean(rgbs, axis=0)
    return z_mm, color, mask, mm_per_px


# ---------------------------------------------------------------------------
# PLY writers
# ---------------------------------------------------------------------------

def _grid_faces(valid):
    """Two triangles per fully-covered cell of a boolean grid.

    Returns (idx_map, faces) where idx_map[r,c] is the vertex index of valid
    pixels (-1 elsewhere) and faces is an (F,3) int array.
    """
    H, W = valid.shape
    idx = -np.ones((H, W), np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[full], b[full], c[full], d[full]
    faces = np.concatenate([np.stack([a, c, b], 1),
                            np.stack([b, c, d], 1)], 0)
    return idx, faces


def write_mesh_ply(path, z_mm, color, valid, mm_per_px, sign=-1.0):
    """Colored triangle mesh from a depth grid. Z = sign*depth (sign<0 -> up =
    toward camera). X,Y in mm."""
    H, W = z_mm.shape
    idx, faces = _grid_faces(valid)
    r, c = np.where(valid)
    X = (c * mm_per_px).astype(np.float32)
    Y = (-(r * mm_per_px)).astype(np.float32)
    Z = (sign * z_mm[valid]).astype(np.float32)
    C = (np.clip(color[valid], 0, 1) * 255).astype(np.uint8)
    verts = np.empty(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts["x"], verts["y"], verts["z"] = X, Y, Z
    verts["r"], verts["g"], verts["b"] = C[:, 0], C[:, 1], C[:, 2]
    frec = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"),
                                       ("b", "<i4"), ("c", "<i4")])
    frec["n"] = 3
    frec["a"], frec["b"], frec["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(verts)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(frec)}\n"
           "property list uchar int vertex_indices\n"
           "end_header\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(hdr)
        verts.tofile(f)
        frec.tofile(f)
        f.flush(); os.fsync(f.fileno())
    return len(verts), len(frec)


def write_points_ply(path, z_mm, color, valid, mm_per_px, sign=-1.0):
    """Colored point cloud (no faces)."""
    r, c = np.where(valid)
    X = (c * mm_per_px).astype(np.float32)
    Y = (-(r * mm_per_px)).astype(np.float32)
    Z = (sign * z_mm[valid]).astype(np.float32)
    C = (np.clip(color[valid], 0, 1) * 255).astype(np.uint8)
    rec = np.empty(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = X, Y, Z
    rec["r"], rec["g"], rec["b"] = C[:, 0], C[:, 1], C[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(rec)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(hdr)
        rec.tofile(f)
        f.flush(); os.fsync(f.fileno())
    return len(rec)


# ---------------------------------------------------------------------------
# Consecutive-frame offset estimation (robust)
# ---------------------------------------------------------------------------

def estimate_offsets(colors):
    """Global (x, y) pixel offset of each frame from Hanning-windowed phase
    correlation on consecutive all-on frames. Low-confidence shifts are
    replaced by the median of the trusted ones so a bad pair can't derail the
    chain. Returns integer offsets with the min at (0, 0)."""
    grays = [cv2.cvtColor((np.clip(c, 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_RGB2GRAY).astype(np.float32) for c in colors]
    H, W = grays[0].shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)

    raw, resp = [], []
    for a, b in zip(grays[:-1], grays[1:]):
        (dx, dy), r = cv2.phaseCorrelate(a, b, win)
        raw.append((dx, dy)); resp.append(r)

    good = [d for d, r in zip(raw, resp) if r >= MIN_PHASE_RESPONSE]
    med = np.median(np.array(good), axis=0) if good else np.array([0.0, 0.0])
    shifts = []
    for (dx, dy), r in zip(raw, resp):
        if r < MIN_PHASE_RESPONSE or abs(dx - med[0]) > 3 * (abs(med[0]) + 5):
            dx, dy = med
        shifts.append((dx, dy))

    # Cumulative global positions: P_{i+1} = P_i - shift_i (see module notes).
    pos = [np.array([0.0, 0.0])]
    for dx, dy in shifts:
        pos.append(pos[-1] - np.array([dx, dy]))
    pos = np.array(pos)
    pos -= pos.min(axis=0)
    return np.round(pos).astype(int), resp


# ---------------------------------------------------------------------------
# Stitch into one canvas
# ---------------------------------------------------------------------------

def stitch(depths, colors, masks, positions, mm_per_px):
    """Feather-blend the patches into one continuous depth+colour canvas,
    aligning each patch's relative-depth DC level to the overlap as it is laid
    down. Returns (z_canvas, color_canvas, valid_canvas)."""
    H, W = depths[0].shape
    span = positions.max(axis=0)
    CW, CH = int(span[0] + W), int(span[1] + H)

    acc_z = np.zeros((CH, CW), np.float64)
    acc_c = np.zeros((CH, CW, 3), np.float64)
    acc_w = np.zeros((CH, CW), np.float64)

    order = np.argsort(positions[:, 0])          # lay down left -> right
    for k in order:
        z, col, m = depths[k], colors[k], masks[k]
        x0, y0 = int(positions[k, 0]), int(positions[k, 1])
        # feather weight: distance to the nearest invalid pixel, so seams fade
        w = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        w = w / (w.max() + 1e-9)

        sub_w = acc_w[y0:y0 + H, x0:x0 + W]
        sub_z = acc_z[y0:y0 + H, x0:x0 + W]
        overlap = (sub_w > 0) & m
        z_use = z.copy()
        if overlap.sum() >= MIN_OVERLAP_PX:
            canvas_here = np.divide(sub_z, sub_w, where=sub_w > 0)
            dc = np.median(canvas_here[overlap] - z[overlap])
            z_use = z + dc

        acc_z[y0:y0 + H, x0:x0 + W] += w * z_use
        acc_c[y0:y0 + H, x0:x0 + W] += w[..., None] * np.clip(col, 0, 1)
        acc_w[y0:y0 + H, x0:x0 + W] += w

    valid = acc_w > 1e-6
    z_canvas = np.divide(acc_z, acc_w, where=valid)
    color_canvas = np.divide(acc_c, acc_w[..., None], where=valid[..., None])
    z_canvas[valid] -= np.median(z_canvas[valid])
    return z_canvas, color_canvas, valid


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    locs = sorted(groups)

    print("Computing per-location triangulation depth + meshes...")
    depths, colors, masks = [], [], []
    mm_per_px = None
    for loc in locs:
        files = groups[loc]
        if not all(i in files for i in ORDER):
            print(f"  [loc {loc}] SKIP - missing single-LED photo(s)")
            continue
        z_mm, color, mask, mm_per_px = location_depth_mm(files)
        depths.append(z_mm); colors.append(color); masks.append(mask)

        v, fcnt = write_mesh_ply(os.path.join(OUT_DIR, f"loc{loc:02d}_mesh.ply"),
                                 z_mm, color, mask, mm_per_px)
        print(f"  [loc {loc:2d}] mesh: {v:6d} verts, {fcnt:6d} faces")

    print("\nEstimating pan offsets between consecutive frames...")
    positions, resp = estimate_offsets(colors)
    H, W = depths[0].shape
    cw, ch = positions[:, 0].max() + W, positions[:, 1].max() + H
    print(f"  canvas: {cw}x{ch} px from {len(depths)} patches "
          f"({STITCH_LONG_EDGE}px frames)")

    print("Stitching + feather-blending into one surface...")
    z_canvas, color_canvas, valid = stitch(depths, colors, masks, positions,
                                           mm_per_px)

    mesh_path = os.path.join(OUT_DIR, "combined_surface.ply")
    v, fcnt = write_mesh_ply(mesh_path, z_canvas, color_canvas, valid, mm_per_px)
    pts_path = os.path.join(OUT_DIR, "combined_surface_points.ply")
    p = write_points_ply(pts_path, z_canvas, color_canvas, valid, mm_per_px)

    relief = (np.percentile(z_canvas[valid], 98) -
              np.percentile(z_canvas[valid], 2))
    print(f"\nCombined surface: {v:,} verts, {fcnt:,} faces, relief "
          f"{relief:.1f} mm across {valid.sum() * mm_per_px**2 / 100:.0f} cm^2")
    print(f"  mesh   -> {mesh_path}")
    print(f"  points -> {pts_path}")
    print(f"  per-location meshes -> {OUT_DIR}/locNN_mesh.ply")
    _save_preview(z_canvas, color_canvas, valid)


def _save_preview(z_canvas, color_canvas, valid):
    """A quick colour + relief preview PNG of the stitched surface."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    lo, hi = np.percentile(z_canvas[valid], [2, 98])
    disp_c = np.where(valid[..., None], np.clip(color_canvas, 0, 1), 1.0)
    disp_z = np.where(valid, z_canvas, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), dpi=110)
    fig.patch.set_facecolor("#f5f5f0")
    axes[0].imshow(disp_c); axes[0].set_title(
        "Stitched colour mosaic (17 frames)", fontweight="bold"); axes[0].axis("off")
    im = axes[1].imshow(disp_z, cmap="turbo", vmin=lo, vmax=hi)
    axes[1].set_title("Stitched relief (mm)", fontweight="bold"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.8, label="relative depth (mm)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "combined_surface_preview.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)
    print(f"  preview -> {out}")


if __name__ == "__main__":
    main()
