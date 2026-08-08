"""
Inverse-square (near-field single-light) depth for shot_011.

Different physics from photometric stereo: a point light's brightness falls off
as 1/r^2, so brightness -> distance directly (r = sqrt(albedo*power / I)). This
SUITS the near-coaxial rig (the opposite of what photometric stereo needs).

  * combine the 4 LEDs (dark-subtracted) -> ~uniform illumination, fills shadows,
  * linear luminance with the retroreflective markers inpainted (they'd spike),
  * r = 1/sqrt(I); scale the bone median to the CT working distance (~80 mm).

Weakness (be aware): it conflates albedo and surface tilt with distance -- a
darker patch reads as "farther" whether it's really farther, a slope, or just
darker bone. Real calibration used for undistortion. Run: python inverse_square_shot011.py
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
import detect_trackers as dt

SHOT = "shot_011"
WD_CT_MM = 80.0          # CT-derived working distance (median), for absolute scale


def main():
    d = bd.project_path("Data_collection", "shots", SHOT)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums, ref = [], None
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        if i == 1:
            ref = rgb
        # solve_luminance: sRGB->linear luminance + specular (marker) inpaint
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
    I = np.mean(lums, axis=0)                         # ~all-on, marker-free
    H, W = I.shape

    mask = bd.bone_mask(I)
    r = 1.0 / np.sqrt(np.maximum(I, 1e-4))            # relative distance (1/sqrt I)
    scale = WD_CT_MM / (np.median(r[mask]) if mask.any() else np.median(r))
    dist = r * scale                                 # mm distance from lens

    # markers for overlay (detect on the real-calib undistorted led1)
    bgr = cv2.cvtColor((np.clip(ref, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    markers = [(a[2], a[3]) for a in dt.aggregate_by_shot(
        [["s", "led1", 0, m["x"], m["y"], m["r"], 0, 0, 0, 0]
         for m in dt.filter_candidates(dt.detect(bgr), 130, 80, 25, 45, 130, 30, 145)],
        merge_dist=35)]
    yy, xx = np.ogrid[:H, :W]
    mdist = []
    for (u, v) in markers:
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= 14 ** 2
        mdist.append(float(np.median(dist[disk])))

    lo, hi = np.percentile(dist[mask] if mask.any() else dist, [2, 98])

    fig = plt.figure(figsize=(13, 5.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(ref, 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (reference)", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(dist, cmap="turbo", vmin=lo, vmax=hi)
    for (u, v), dd in zip(markers, mdist):
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="white", lw=1.6))
        ax1.annotate(f"{dd:.0f}", (u, v), (u + 17, v), color="white", fontsize=8,
                     fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55,
                               ec="none"))
    ax1.set_title("inverse-square distance from lens (mm)\nblue = near, red = far",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label(
        "distance from lens (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    dsm = gaussian_filter(dist, 1.5)
    ddisp = np.clip(-dsm, -hi, -lo)                  # near = up
    step = max(1, W // 140); yg, xg = np.mgrid[0:H:step, 0:W:step]
    mmpp = 60.0 / W
    ax2.plot_surface(xg * mmpp, yg * mmpp, ddisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("mm")
    ax2.view_init(elev=55, azim=-60)

    fig.suptitle(f"{SHOT} — INVERSE-SQUARE depth (single-light brightness model)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "4 LEDs combined, markers inpainted, real calibration; "
             "bone median anchored to CT ~80 mm. Conflates albedo/tilt with "
             "distance -- darker = reads farther.", ha="center", fontsize=8,
             style="italic", color="#555")

    out = bd.project_path("depth_outputs", "nearfield", f"{SHOT}_inverse_square.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"peak-to-valley {hi-lo:.1f} mm   marker distances (mm): "
          f"{[round(x) for x in mdist]}")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
