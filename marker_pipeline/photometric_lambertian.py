"""
Plain Lambertian photometric stereo (Woodham) on the calib_charuco bone shots.

Just the classic solve, nothing fancy: assume a Lambertian surface, I_k = rho (n . L_k),
4 directional lights -> per-pixel normals (least squares) -> Frankot-Chellappa depth.
Real calibration is used only for undistortion. Retroreflective markers are
inpainted (they are specular, not Lambertian). Run: python photometric_lambertian.py
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd

SHOT_DIR = bd.project_path("Data_collection", "calib_charuco")
SHOTS = [f"shot_{n:03d}" for n in range(4, 9)]
AZ = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}         # right / top / left / bottom
A_MM, WD_MM = 6.05, 40.0                            # LED radius, working distance
ELEV = math.degrees(math.atan2(WD_MM, A_MM))       # ~81 deg (near-coaxial)


def photometric_depth(shot):
    d = os.path.join(SHOT_DIR, shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums, Ls, ref = [], [], None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        corr = np.clip(rgb - dark, 0, 1)           # dark subtraction
        lums.append(bd.solve_luminance(corr))      # linear luminance + glint inpaint
        Ls.append(bd.light_vector(AZ[i], ELEV))
    lums = bd.balance_exposure(lums)               # normalise LED flux imbalance
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))   # Woodham
    z = bd.normals_to_depth(normals)               # Frankot-Chellappa
    mask = bd.bone_mask(albedo)
    z = z - (np.median(z[mask]) if mask.any() else np.median(z))
    return z, ref, mask


def main():
    have = [s for s in SHOTS if os.path.isdir(os.path.join(SHOT_DIR, s))]
    if not have:
        print(f"no shots found in {SHOT_DIR}"); return
    n = len(have)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.6), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")
    if n == 1:
        axes = axes.reshape(2, 1)
    for c, shot in enumerate(have):
        z, ref, mask = photometric_depth(shot)
        v = z[mask] if mask.any() else z
        lo, hi = np.percentile(v, [2, 98])
        axes[0, c].imshow(np.clip(ref, 0, 1)); axes[0, c].axis("off")
        axes[0, c].set_title(shot, fontsize=9, fontweight="bold")
        im = axes[1, c].imshow(gaussian_filter(z, 1.0), cmap="turbo",
                               vmin=lo, vmax=hi)
        axes[1, c].axis("off")
        axes[1, c].set_title(f"relief {hi-lo:.2f}", fontsize=8)
        print(f"  {shot}: relief(p2-p98) = {hi-lo:.3f}  (relative units)")
    fig.suptitle("Lambertian photometric stereo (Woodham, 4 lights) — "
                 "calib_charuco shots 4-8", fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.5, 0.005, f"assume Lambertian, dark-subtracted, flux-balanced, "
             f"markers inpainted; lights az 0/90/180/270 at {ELEV:.0f} deg "
             "elevation (a=6.05mm, wd=40mm). blue=near, red=far.",
             ha="center", fontsize=8, style="italic", color="#555")
    out = bd.project_path("depth_outputs", "nearfield", "calib_charuco_lambertian.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"\nslide -> {out}")


if __name__ == "__main__":
    main()
