"""
shot_011 near-field depth using the REAL endoscope calibration.

Uses the checkerboard calibration (calibration/calibration.txt: fx~860, real
distortion) instead of the old assumed intrinsics, plus the corrected 6.57 mm LED
offset, anchored to the CT-derived ~80 mm working distance. Frames are undistorted
by that calibration (via bd.load_rgb), so the near-field solve runs on real
geometry. This is to eyeball whether the depth SHAPE looks right.

Run:  python nearfield_shot011_calibrated.py
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
import detect_trackers as dt
from nearfield_shot_depth import provisional_calibration, load_bodies

SHOT = "shot_011"
WD_CT_MM = 80.0          # working distance from CT+PnP with the real calibration


def undistorted_intrinsics():
    """Intrinsics of the frames AFTER bd.undistort_bgr (getOptimalNewCameraMatrix,
    alpha=0) -- these match the undistorted images the solve runs on."""
    calib = bd.load_camera_calibration()
    K = np.asarray(calib["camera_matrix"], float)
    dist = np.asarray(calib["dist_coeffs"], float)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (640, 480), 0, (640, 480))
    return newK[0, 0], newK[1, 1], newK[0, 2], newK[1, 2]


def detect_on_undistorted(rgbs):
    """Markers on the undistorted frames (aggregated across the 4 LEDs)."""
    rows = []
    for i, rgb in enumerate(rgbs, 1):
        bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        for m in dt.filter_candidates(dt.detect(bgr), 130, 80, 25, 45, 130, 30, 145):
            rows.append(["s", f"led{i}", 0, m["x"], m["y"], m["r"], 0, 0, 0, 0])
    return [(a[2], a[3]) for a in dt.aggregate_by_shot(rows, merge_dist=35)]


def main():
    body, rgbs = load_bodies(bd.project_path("Data_collection", "shots", SHOT))
    H, W = body[0].shape
    fx, fy, cx, cy = undistorted_intrinsics()
    print(f"using real calibration: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")

    z, normals, albedo = nf.nearfield_stereo(body, fx, fy, cx, cy, WD_CT_MM,
                                             provisional_calibration())
    mask = bd.bone_mask(albedo)
    markers = detect_on_undistorted(rgbs)

    # depth relative to bone median (shape), and absolute distance-from-lens
    zc = z - (np.median(z[mask]) if mask.any() else np.median(z))
    lo, hi = np.percentile(zc[mask] if mask.any() else zc, [2, 98])
    relief = float(hi - lo)
    yy, xx = np.ogrid[:H, :W]
    mdepth = []
    for (u, v) in markers:
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= 14 ** 2
        mdepth.append(float(np.median(zc[disk])))

    fig = plt.figure(figsize=(13, 5.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(rgbs[0], 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (real-calib undistorted)", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(zc, cmap="turbo", vmin=lo, vmax=hi)
    for k, ((u, v), d) in enumerate(zip(markers, mdepth), 1):
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="white", lw=1.6))
        ax1.annotate(f"{d:+.1f}", (u, v), (u + 17, v), color="white", fontsize=8,
                     fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55,
                               ec="none"))
    ax1.set_title(f"near-field depth (mm, rel. to bone median)\nrelief {relief:.1f} mm",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label(
        "depth toward/away camera (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(zc, 1.5); zdisp = np.clip(-zsm, -hi, -lo)
    step = max(1, W // 140); yg, xg = np.mgrid[0:H:step, 0:W:step]
    mmpp = 60.0 / W          # ~60 mm field at the CT working distance
    ax2.plot_surface(xg * mmpp, yg * mmpp, zdisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10,
                  fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("mm")
    ax2.view_init(elev=55, azim=-60)

    fig.suptitle(f"{SHOT} — near-field depth with REAL calibration (fx=860)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Real checkerboard intrinsics + distortion, corrected 6.57 mm "
             "LED offset, anchored to CT working distance ~80 mm. Near-coaxial rig -> "
             "photometric stereo still weakly conditioned.", ha="center",
             fontsize=8, style="italic", color="#555")

    out = bd.project_path("depth_outputs", "nearfield", f"{SHOT}_realcalib.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"relief {relief:.1f} mm   markers depth(mm): "
          f"{[round(d,1) for d in mdepth]}")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
