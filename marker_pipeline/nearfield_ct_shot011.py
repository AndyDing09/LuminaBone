"""
CT-calibrated NEAR-FIELD depth for shot_011.

Combines the two things that are actually right for this rig:
  * the near-field photometric model (inverse-square + per-pixel light dirs), and
  * the shot_011 CT ground control (the only shot with CT).

Steps:
  1. near-field solve on shot_011 (provisional symmetric LED geometry -- the real
     led_calibration.json still needs captures; positions are ASSUMED),
  2. sample the near-field depth at the markers,
  3. PnP the CT markers into the camera frame (pose disambiguated by the CONFIRMED
     fact that marker 4 is closest), take each marker's axial depth,
  4. affine-anchor the near-field depth to CT (z_cal = a*z_nf + b) -- and REPORT
     the slope sign: positive => near-field agrees with CT (unlike far-field,
     which came out inverted).

Run:  python nearfield_ct_shot011.py
"""

import os
import sys
import csv
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
from nearfield_shot_depth import provisional_calibration, load_bodies

SHOT = "shot_011"
MARKER_CSV = bd.project_path("depth_outputs", "marker_depth_slides",
                             "marker_depths.csv")
OUT = bd.project_path("depth_outputs", "nearfield",
                      f"{SHOT}_nearfield_ct.png")
CT = {1: (9.224, -1.632, 11.818),
      3: (3.729, -13.209, 23.259),
      4: (-17.665, 0.732, 17.412)}          # CT frame, mm
CLOSEST = 4                                   # confirmed physical fact


def markers_px():
    m = {}
    with open(MARKER_CSV) as f:
        for r in csv.DictReader(f):
            if r["shot"] == SHOT:
                m[int(r["id"])] = (float(r["x_px"]), float(r["y_px"]))
    return m


def ct_axial_depth(mpx, K):
    """PnP the CT markers, pick the pose with marker CLOSEST nearest, return
    {id: axial camera-frame depth Z (mm)}."""
    ids = sorted(CT)
    img = np.array([[mpx[i][0], mpx[i][1]] for i in ids])
    obj = np.array([CT[i] for i in ids])
    _, rvecs, tvecs = cv2.solveP3P(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_AP3P)
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        cam = (R @ obj.T + tv).T
        if (cam[:, 2] > 0).all():
            dist = {i: float(np.linalg.norm(c)) for i, c in zip(ids, cam)}
            if min(dist, key=dist.get) == CLOSEST:
                return {i: float(c[2]) for i, c in zip(ids, cam)}
    raise RuntimeError("no pose with marker 4 closest")


def sample(zmap, u, v, r=14):
    H, W = zmap.shape
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    return float(np.median(zmap[disk]))


def main():
    shot_dir = bd.project_path("Data_collection", "shots", SHOT)
    body, rgbs = load_bodies(shot_dir)
    H, W = body[0].shape
    fx, fy, cx, cy = nf.measured_intrinsics(W, H)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    z, normals, albedo = nf.nearfield_stereo(body, fx, fy, cx, cy,
                                             nf.working_distance_mm(),
                                             provisional_calibration())
    mpx = markers_px()
    z_nf = {i: sample(z, u, v) for i, (u, v) in mpx.items()}
    ct_z = ct_axial_depth(mpx, K)

    # affine-anchor near-field depth to CT axial depth over the CT markers
    ids = sorted(CT)
    xe = np.array([z_nf[i] for i in ids])
    yt = np.array([ct_z[i] for i in ids])
    a, b = np.linalg.lstsq(np.column_stack([xe, np.ones_like(xe)]), yt,
                           rcond=None)[0]
    resid = (a * xe + b) - yt
    corr = float(np.corrcoef(xe, yt)[0, 1])
    z_cal = a * z + b
    m_cal = {i: a * z_nf[i] + b for i in mpx}

    # ---- figure (same 3-panel style as shot_013) ----
    fig = plt.figure(figsize=(13, 5.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(rgbs[0], 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (reference)", fontsize=10, fontweight="bold")

    mask = bd.bone_mask(albedo)
    lo, hi = np.percentile(z_cal[mask] if mask.any() else z_cal, [2, 98])
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(z_cal, cmap="turbo", vmin=lo, vmax=hi)
    for i, (u, v) in mpx.items():
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="white", lw=1.6))
        tag = f"{i}:{m_cal[i]:.0f}"
        if i in ct_z:
            tag += f"|CT{ct_z[i]:.0f}"
        ax1.annotate(tag, (u, v), (u + 18, v), color="white", fontsize=8,
                     fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6,
                               ec="none"))
    ax1.set_title("CT-anchored near-field depth (mm)\nlabel = est | CT",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label(
        "distance from lens (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(z_cal, 1.5); zdisp = np.clip(-zsm, -hi, -lo)
    step = max(1, W // 140); yg, xg = np.mgrid[0:H:step, 0:W:step]
    mmpp = nf.FIELD_WIDTH_MM / W
    ax2.plot_surface(xg * mmpp, yg * mmpp, zdisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10,
                  fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("mm")
    ax2.view_init(elev=55, azim=-60)

    sign = "AGREES with CT (+)" if a > 0 else "INVERTED vs CT (-)"
    fig.suptitle(f"{SHOT} — CT-anchored NEAR-FIELD depth   "
                 f"(near-field {sign})", fontsize=12, fontweight="bold", y=0.99)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    fig.text(0.5, 0.01, f"z_cal = {a:.2f}*z_nf + {b:.1f}   corr(near-field, CT) = "
             f"{corr:+.2f}   control-RMS {rms:.1f} mm.   Provisional symmetric LED "
             "geometry (positions not yet solved from captures).",
             ha="center", fontsize=8, style="italic", color="#555")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)

    print(f"near-field vs CT:  corr = {corr:+.2f}   slope = {a:+.2f}  "
          f"({'agrees' if a > 0 else 'INVERTED'})   control-RMS {rms:.1f} mm")
    for i in sorted(mpx):
        ctv = f"  CT {ct_z[i]:.1f}" if i in ct_z else "  (no CT)"
        print(f"  marker {i}: near-field {z_nf[i]:6.1f} -> CT-anchored "
              f"{m_cal[i]:6.1f} mm{ctv}")
    print(f"slide -> {OUT}")


if __name__ == "__main__":
    main()
