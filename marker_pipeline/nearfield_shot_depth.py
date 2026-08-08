"""
PROVISIONAL near-field depth for one 4-LED shot (e.g. shot_013).

IMPORTANT — this is NOT the solved calibration. We built led_calibration.py and
validated it, but never captured the mirror-ball / white-card targets, so the
real led_calibration.json does not exist. This script instead feeds the
near-field solver a PROVISIONAL calibration from the measured rig numbers we do
have (LED offset 12.08 mm, working distance 30 mm, measured intrinsics) with the
4 LEDs ASSUMED symmetric (right/top/left/bottom, axis +Z, equal brightness,
Lambertian mu=1). Per-LED brightness IS taken from the data (exposure balance),
since the shots show real pair imbalance.

So this shows the near-field MODEL (inverse-square + per-pixel light directions
-- the actual cure for the far-field bowl), NOT a properly calibrated result.
The LED positions remain assumed until the calibration captures are taken.

Run:  python nearfield_shot_depth.py --shot shot_013
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
import nearfield_lambertian as nf

ORDER = [1, 2, 3, 4]
LED_AZ = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}   # right / top / left / bottom
MARKER_CSV = bd.project_path("depth_outputs", "marker_depth_slides",
                             "marker_depths.csv")

# MEASURED (user, 2026-08): opposing LEDs sit 13.14 mm apart on an 18 mm endoscope
# tip, so each LED is 13.14/2 = 6.57 mm off the lens axis. This SUPERSEDES the old
# nf.LED_OFFSET_MM = 12.08 mm, which was physically impossible here (12.08 mm off
# axis => 24 mm across, wider than the 18 mm endoscope). Per-LED tilt ignored on
# purpose (user: not a big deal) -> axis kept at +Z.
LED_ACROSS_MM = 13.14
LED_OFFSET_MM = LED_ACROSS_MM / 2.0              # 6.57 mm from the lens axis


def provisional_calibration(offset_mm=LED_OFFSET_MM):
    """ASSUMED symmetric 4-LED layout in the lens plane, axis +Z, mu=1, Psi=1.
    (Y points down, so image-up = -Y.)"""
    leds = []
    for i in ORDER:
        a = np.radians(LED_AZ[i])
        leds.append(dict(x_s=np.array([offset_mm * np.cos(a),
                                       -offset_mm * np.sin(a), 0.0]),
                         n_s=np.array([0.0, 0.0, 1.0]), mu=1.0, Psi=1.0))
    return leds


def load_bodies(shot_dir):
    """dark-subtracted, specular-free (R-B), glint-inpainted, exposure-balanced
    body frames for led1..4, plus a reference RGB."""
    dark = bd.load_rgb(os.path.join(shot_dir, "dark.png"), bd.WORK_LONG_EDGE)
    rgbs, body = [], []
    for i in ORDER:
        rgb = bd.load_rgb(os.path.join(shot_dir, f"led{i}.png"), bd.WORK_LONG_EDGE)
        corr = np.clip(rgb - dark, 0, 1)
        sf = nf.specular_free_channel(corr)
        sf = bd.inpaint_specular(sf, bd.detect_specular_mask(corr))
        rgbs.append(rgb)
        body.append(sf)
    body = nf.balance_body_frames(rgbs, body)   # LEDs are NOT equal -> balance
    return body, rgbs


def load_markers(shot):
    m = {}
    if not os.path.exists(MARKER_CSV):
        return m
    with open(MARKER_CSV) as f:
        for r in csv.DictReader(f):
            if r["shot"] == shot:
                m[int(r["id"])] = (float(r["x_px"]), float(r["y_px"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="shot_013")
    a = ap.parse_args()

    shot_dir = bd.project_path("Data_collection", "shots", a.shot)
    body, rgbs = load_bodies(shot_dir)
    H, W = body[0].shape
    fx, fy, cx, cy = nf.measured_intrinsics(W, H)
    z_work = nf.working_distance_mm()
    leds = provisional_calibration()

    z, normals, albedo = nf.nearfield_stereo(body, fx, fy, cx, cy, z_work, leds)
    mask = bd.bone_mask(albedo)
    zc = z - (np.median(z[mask]) if mask.any() else np.median(z))
    lo, hi = np.percentile(zc[mask] if mask.any() else zc, [2, 98])
    relief = float(hi - lo)

    markers = load_markers(a.shot)
    mdepth = {}
    yy, xx = np.ogrid[:H, :W]
    for i, (u, v) in markers.items():
        disk = (xx - u) ** 2 + (yy - v) ** 2 <= 14 ** 2
        mdepth[i] = float(np.median(zc[disk]))

    # ---- figure ----
    fig = plt.figure(figsize=(13, 5.6), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(rgbs[0], 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (reference)", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(zc, cmap="turbo", vmin=lo, vmax=hi)
    for i, (u, v) in markers.items():
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="white", lw=1.6))
        ax1.annotate(f"{i}:{mdepth[i]:+.1f}", (u, v), (u + 18, v), color="white",
                     fontsize=8, fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55,
                               ec="none"))
    ax1.set_title(f"near-field depth (mm) — relief {relief:.1f} mm",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label("depth (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(zc, 1.5); zdisp = np.clip(-zsm, -hi, -lo)
    step = max(1, W // 140); yg, xg = np.mgrid[0:H:step, 0:W:step]
    mmpp = nf.FIELD_WIDTH_MM / W
    ax2.plot_surface(xg * mmpp, yg * mmpp, zdisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("mm")
    ax2.view_init(elev=55, azim=-60)

    fig.suptitle(f"{a.shot} — PROVISIONAL near-field depth (assumed symmetric "
                 "LED geometry; NOT the solved calibration)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "near-field MODEL (inverse-square + per-pixel light) with "
             "measured offset 12.08 mm / working dist 30 mm; LED positions ASSUMED "
             "symmetric. No CT for this shot -> ordering not validated.",
             ha="center", fontsize=8, style="italic", color="#555")

    out = bd.project_path("depth_outputs", "nearfield",
                          f"{a.shot}_nearfield_provisional.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)

    print(f"{a.shot}: near-field relief {relief:.1f} mm   (PROVISIONAL, assumed geometry)")
    for i in sorted(mdepth):
        print(f"  marker {i}: near-field depth {mdepth[i]:+.2f} mm")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
