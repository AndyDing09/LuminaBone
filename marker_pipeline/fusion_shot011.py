"""
shot_011 depth by FUSING the two near-field cues:
  * inverse-square attenuation  -> robust low-frequency / absolute depth,
  * directional photometric stereo (4 images) -> high-frequency, albedo-free detail.

Naive near-field PS (joint solve) lets the ill-conditioned lateral part inject a
low-frequency ramp. Instead we frequency-split: low-freq from attenuation,
high-freq from the PS normals. z_fused = LP(z_att) + [z_ps - LP(z_ps)].

Real calibration for undistortion + intrinsics, corrected 6.55 mm offset, markers
inpainted. Run: python fusion_shot011.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
import nearfield_lambertian as nf
from nearfield_shot_depth import provisional_calibration
from nearfield_shot011_calibrated import undistorted_intrinsics

SHOT = "shot_011"
WD = 80.0            # CT-derived working distance
LP_SIGMA = 30        # px; low-pass cutoff separating "shape" from "detail"


def main():
    d = bd.project_path("Data_collection", "shots", SHOT)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums, ref = [], None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))   # markers inpainted
    H, W = lums[0].shape
    fx, fy, cx, cy = undistorted_intrinsics()

    # --- cue 1: inverse-square attenuation depth (low-freq / absolute) ---
    I = np.mean(lums, axis=0)
    mask = bd.bone_mask(I)
    r = 1.0 / np.sqrt(np.maximum(I, 1e-4))
    z_att = r * (WD / (np.median(r[mask]) if mask.any() else np.median(r)))
    z_att = z_att - np.median(z_att[mask])          # relative, mm

    # --- cue 2: directional near-field PS depth (for its high-freq) ---
    # luminance frames, NOT exposure-balanced (keep the attenuation the model uses)
    z_ps, normals, _ = nf.nearfield_stereo(lums, fx, fy, cx, cy, WD,
                                           provisional_calibration())
    z_ps = z_ps - np.median(z_ps[mask])

    # --- fuse: low-freq from attenuation, high-freq from PS ---
    z_fused = gaussian_filter(z_att, LP_SIGMA) + \
        (z_ps - gaussian_filter(z_ps, LP_SIGMA))

    def relief(z):
        v = z[mask] if mask.any() else z.ravel()
        return float(np.percentile(v, 98) - np.percentile(v, 2))

    # --- figure: three depth maps ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")
    for ax, z, title in zip(
            axes, [z_att, z_ps, z_fused],
            [f"inverse-square (attenuation)\nrelief {relief(z_att):.1f} mm",
             f"directional PS (4 images)\nrelief {relief(z_ps):.1f} mm",
             f"FUSED  (LP attenuation + HP PS)\nrelief {relief(z_fused):.1f} mm"]):
        v = z[mask] if mask.any() else z
        lo, hi = np.percentile(v, [2, 98])
        im = ax.imshow(z, cmap="turbo", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10, fontweight="bold"); ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle(f"{SHOT} — inverse-square + photometric stereo, fused",
                 fontsize=13, fontweight="bold", y=1.02)
    out = bd.project_path("depth_outputs", "nearfield", f"{SHOT}_fusion.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"relief mm: attenuation {relief(z_att):.1f} | PS {relief(z_ps):.1f} | "
          f"fused {relief(z_fused):.1f}")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
