"""
Inverse-square depth on the CT shots, calibrated to lens with CT (z scale+offset).

r = 1/sqrt(I_combined): near-field brightness -> distance directly (no far-field
bowl). Combine the 4 LEDs (dark-subtracted), markers inpainted, then affine-anchor
to the CT marker depths (PnP with the real calibration). Prints marker RMS and
bone relief per shot so it can be compared to far-field / near-field photometric.

Run: python inverse_square_ct.py
"""

import os
import sys
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
OUT = bd.project_path("depth_outputs", "calib_charuco_slides")


def ct_depths(corr):
    ids = sorted(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    img = np.array([corr[i][0] for i in ids])
    _, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rv)
    return (R @ obj.T + tv).T[:, 2]


def sample(z, u, v, r=14):
    H, W = z.shape
    yy, xx = np.ogrid[:H, :W]
    m = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    return float(np.median(z[m]))


def main():
    print("inverse-square depth, CT-calibrated (z scale + offset):")
    for shot, corr in C.CT_CORR.items():
        ids = sorted(corr)
        d = bd.project_path("Data_collection", "calib_charuco", shot)
        dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
        lums, ref = [], None
        for i in (1, 2, 3, 4):
            rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
            if i == 1:
                ref = rgb
            lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
        I = np.mean(lums, axis=0)
        mask = bd.bone_mask(I)
        r = 1.0 / np.sqrt(np.maximum(I, 1e-4))         # inverse-square range
        ct = ct_depths(corr)
        rr = np.array([sample(r, *corr[i][0]) for i in ids])
        a, b = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]), ct,
                               rcond=None)[0]
        z = a * r + b
        rms = np.sqrt(np.mean((a * rr + b - ct) ** 2))
        me = binary_erosion(mask, iterations=8)
        relief = (np.percentile(z[me], 90) - np.percentile(z[me], 10)) if me.any() else 0
        print(f"  {shot}: marker RMS {rms:5.2f} mm   bone relief {relief:5.1f} mm "
              f"(true CT range {ct.max()-ct.min():.1f} mm)")

        # slide
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.4), dpi=120)
        fig.patch.set_facecolor("#f5f5f0")
        ax[0].imshow(np.clip(ref, 0, 1)); ax[0].axis("off")
        ax[0].set_title(f"{shot} led1", fontsize=10, fontweight="bold")
        lo, hi = np.percentile(z[mask] if mask.any() else z, [2, 98])
        im = ax[1].imshow(gaussian_filter(z, 1.0), cmap="turbo_r", vmin=lo, vmax=hi)
        for i in ids:
            u, v = corr[i][0]
            ax[1].add_patch(plt.Circle((u, v), 14, fill=False, ec="k", lw=1.5))
        ax[1].set_title(f"inverse-square depth from lens (mm)\nmarker RMS "
                        f"{rms:.1f} mm", fontsize=10, fontweight="bold")
        ax[1].axis("off")
        plt.colorbar(im, ax=ax[1], shrink=0.8).set_label("mm from lens", fontsize=8)
        fig.savefig(os.path.join(OUT, f"{shot}_inverse_square.png"),
                    bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"\nslides -> {OUT}")


if __name__ == "__main__":
    main()
