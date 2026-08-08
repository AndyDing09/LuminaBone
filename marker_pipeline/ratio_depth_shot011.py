"""
shot_011 within-bone relief from CALIBRATED (provisional) opposing-pair ratios.

  * Psi-normalize each LED frame by its bone median (provisional equal-intensity
    calibration straight from the data),
  * bone mask (drop the background silhouette step that broke the fusion),
  * albedo-free log-ratios R_h=ln(I1/I3), R_v=ln(I2/I4),
  * remove the flat-surface baseline (a fitted plane over the bone) -> the
    residual ratio is relief-induced and ~proportional to the surface gradient,
  * integrate (Frankot-Chellappa) -> within-bone relief.

Provisional (Psi from data, not a real mirror-ball/white-card calibration), so
the ABSOLUTE scale is arbitrary; this tests whether masking + albedo-free ratios
recover a bone-like SHAPE. Run: python ratio_depth_shot011.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd

SHOT = "shot_011"


def plane_detrend(R, mask, u, v):
    """Subtract the least-squares plane fit over the bone (flat-surface baseline)."""
    A = np.column_stack([u[mask], v[mask], np.ones(mask.sum())])
    c, *_ = np.linalg.lstsq(A, R[mask], rcond=None)
    return R - (c[0] * u + c[1] * v + c[2])


def main():
    d = bd.project_path("Data_collection", "shots", SHOT)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    L, ref = {}, None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        L[i] = bd.solve_luminance(np.clip(rgb - dark, 0, 1))   # markers inpainted
    H, W = L[1].shape
    mask = bd.bone_mask(np.mean([L[i] for i in L], axis=0))
    mask = binary_erosion(mask, iterations=4)                  # stay off the rim

    # provisional Psi calibration: equalise per-LED bone-median brightness
    for i in L:
        m = np.median(L[i][mask]) if mask.any() else np.median(L[i])
        L[i] = L[i] / max(m, 1e-6)

    eps = 1e-3
    Rh = np.log((L[1] + eps) / (L[3] + eps))       # right / left
    Rv = np.log((L[2] + eps) / (L[4] + eps))       # top / bottom
    v, u = np.mgrid[0:H, 0:W].astype(float)
    un, vn = (u - W / 2) / W, (v - H / 2) / H

    # relief-induced ratio = residual after removing the flat-surface plane
    dRh = plane_detrend(Rh, mask, un, vn)
    dRv = plane_detrend(Rv, mask, un, vn)
    dRh = np.where(mask, dRh, 0.0)
    dRv = np.where(mask, dRv, 0.0)
    dRh = gaussian_filter(dRh, 2.0)                # gentle denoise of the ratio
    dRv = gaussian_filter(dRv, 2.0)

    # residual ratio ~ surface gradient -> integrate
    z = bd.integrate_frankot_chellappa(dRh, dRv)
    z = z - (np.median(z[mask]) if mask.any() else np.median(z))
    zin = z[mask]
    lo, hi = np.percentile(zin, [2, 98])

    fig = plt.figure(figsize=(13, 5.6), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(ref, 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (reference)", fontsize=10, fontweight="bold")

    zmask = np.where(mask, z, np.nan)
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(zmask, cmap="turbo", vmin=lo, vmax=hi)
    ax1.set_title("within-bone relief from albedo-free ratios\n(arbitrary units)",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsurf = np.where(mask, gaussian_filter(np.where(mask, z, 0), 2), np.nan)
    step = max(1, W // 140); yg, xg = np.mgrid[0:H:step, 0:W:step]
    ax2.plot_surface(xg, yg, np.clip(zsurf[::step, ::step], lo, hi),
                     cmap="turbo", vmin=lo, vmax=hi, linewidth=0, antialiased=True)
    ax2.set_title("3-D relief", fontsize=10, fontweight="bold")
    ax2.set_xlabel("u"); ax2.set_ylabel("v"); ax2.view_init(elev=60, azim=-60)

    fig.suptitle(f"{SHOT} — within-bone relief from calibrated opposing-pair ratios",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Provisional Psi (data-derived), bone-masked, albedo-free "
             "ratios, plane-detrended, integrated. Shape test only — absolute scale "
             "uncalibrated.", ha="center", fontsize=8, style="italic", color="#555")

    out = bd.project_path("depth_outputs", "nearfield", f"{SHOT}_ratio_relief.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"within-bone relief p2-p98 = {hi-lo:.3f} (arbitrary units)")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
