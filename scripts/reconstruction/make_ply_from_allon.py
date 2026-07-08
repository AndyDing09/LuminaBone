"""
Build one colored PLY point cloud per location from the all-LEDs-on photo.
================================================================================

For each location it takes the single uniform reference image  p{loc}-1.jpg
(all LEDs on) and recovers depth with the single-image inverse-square shading
method from endoscope_depth.py:

        I = albedo * light / r^2     ->     r = sqrt(albedo * light / I)

i.e. brighter pixels are nearer, darker pixels are farther. Specular glints are
masked + inpainted first, and the depth is edge-aware smoothed. The result is
exported as an ASCII PLY point cloud (x, y, z, r, g, b) -- one point per pixel,
colored with the photo -- exactly like paper_endoscope_c.ply.

Reuses the functions in endoscope_depth.py.  Run:  python make_ply_from_allon.py
Open the .ply files in MeshLab / CloudCompare / Blender.
"""

import os
import numpy as np
from PIL import Image

import endoscope_depth as ed        # reuse the verified single-image method
import bone_depth_batch as bd       # reuse discover() + config

OUT_DIR = bd.project_path("depth_outputs", "ply_allon")
ALL_ON_INDEX = 1                    # p{loc}-1 = all LEDs on
WORK_LONG_EDGE = 640               # downscale long edge (keeps point count sane)


def load_rgb(path, long_edge):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def process(loc, path, out_path):
    rgb = load_rgb(path, WORK_LONG_EDGE)

    # Whole bone fills the frame; keep every non-black pixel.
    fov_mask = ed.to_luminance(rgb) > 0.05

    # Remove specular highlights (they'd read as false "near" spikes).
    spec = ed.detect_specular_mask(rgb)
    cleaned = rgb.copy()
    for c in range(3):
        cleaned[..., c] = ed.inpaint_mask(rgb[..., c], spec)

    # Inverse-square depth + edge-aware smoothing.
    lum = ed.to_luminance(cleaned)
    depth = ed.bilateral_smooth(ed.brightness_to_depth(lum), lum)

    # Depth map -> colored point cloud -> PLY.
    pts, cols = ed.depth_to_pointcloud(depth, rgb, fov_mask)
    ed.write_ply(out_path, pts, cols)
    return len(pts), int(spec.sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    n = 0
    for loc in sorted(groups):
        files = groups[loc]
        if ALL_ON_INDEX not in files:
            print(f"  [loc {loc}] SKIP - no all-on photo p{loc}-{ALL_ON_INDEX}")
            continue
        out = os.path.join(OUT_DIR, f"loc{loc:02d}_allon.ply")
        npts, nspec = process(loc, files[ALL_ON_INDEX], out)
        n += 1
        print(f"  loc {loc:2d}: {npts:>7} points  ({nspec} specular px removed)  -> {os.path.basename(out)}")
    print(f"\n{n} PLY files -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
