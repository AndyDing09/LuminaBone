"""
Transfer the shot_011 CT calibration (near-field offset + scale) to another shot.

Idea: shot_011 is the only shot with CT. Anchor its NEAR-FIELD depth to CT once
to get an affine map  z_cal = a * z_nf + b  (a = scale, b = offset). Because both
shots' near-field depths are produced the same way (same solver, same working-
distance anchor, same provisional LED geometry), that a/b is treated as a rig-
level calibration and applied to a target shot (default shot_013) which has no CT.

Caveats: the LED geometry is still ASSUMED symmetric (real calibration captures
not taken), and a/b is transferred from ONE shot, so the target result is an
estimate, not a validated measurement. shot_011's own control-RMS (~6 mm) is a
floor on the accuracy.

Run:  python nearfield_ct_transfer.py --shot shot_013
"""

import os
import sys
import csv
import argparse
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

MARKER_CSV = bd.project_path("depth_outputs", "marker_depth_slides",
                             "marker_depths.csv")
CT_SHOT011 = {1: (9.224, -1.632, 11.818),
              3: (3.729, -13.209, 23.259),
              4: (-17.665, 0.732, 17.412)}
CLOSEST = 4


def markers_px(shot):
    m = {}
    with open(MARKER_CSV) as f:
        for r in csv.DictReader(f):
            if r["shot"] == shot:
                m[int(r["id"])] = (float(r["x_px"]), float(r["y_px"]))
    return m


def sample(zmap, u, v, r=14):
    H, W = zmap.shape
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    return float(np.median(zmap[disk]))


def nearfield_depth(shot):
    """Near-field depth map (mm) + reference rgb + bone mask for one shot."""
    shot_dir = bd.project_path("Data_collection", "shots", shot)
    body, rgbs = load_bodies(shot_dir)
    H, W = body[0].shape
    fx, fy, cx, cy = nf.measured_intrinsics(W, H)
    z, _, albedo = nf.nearfield_stereo(body, fx, fy, cx, cy,
                                       nf.working_distance_mm(),
                                       provisional_calibration())
    return z, rgbs[0], bd.bone_mask(albedo), (fx, fy, cx, cy)


def derive_shot011_calibration():
    """Fit z_cal = a*z_nf + b anchoring shot_011 near-field depth to CT axial
    depth (pose picked so marker 4 is closest). Returns (a, b, corr, rms)."""
    z, _, _, (fx, fy, cx, cy) = nearfield_depth("shot_011")
    mpx = markers_px("shot_011")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    ids = sorted(CT_SHOT011)
    obj = np.array([CT_SHOT011[i] for i in ids])
    img = np.array([[mpx[i][0], mpx[i][1]] for i in ids])
    _, rvecs, tvecs = cv2.solveP3P(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_AP3P)
    ct_z = None
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        cam = (R @ obj.T + tv).T
        if (cam[:, 2] > 0).all():
            dist = {i: float(np.linalg.norm(c)) for i, c in zip(ids, cam)}
            if min(dist, key=dist.get) == CLOSEST:
                ct_z = {i: float(c[2]) for i, c in zip(ids, cam)}
    xe = np.array([sample(z, *mpx[i]) for i in ids])
    yt = np.array([ct_z[i] for i in ids])
    a, b = np.linalg.lstsq(np.column_stack([xe, np.ones_like(xe)]), yt,
                           rcond=None)[0]
    rms = float(np.sqrt(np.mean((a * xe + b - yt) ** 2)))
    return float(a), float(b), float(np.corrcoef(xe, yt)[0, 1]), rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="shot_013")
    a_arg = ap.parse_args()

    a, b, corr, rms = derive_shot011_calibration()
    print(f"shot_011 calibration:  z_cal = {a:.3f}*z_nf + {b:.3f}   "
          f"(corr {corr:+.2f}, RMS {rms:.1f} mm)  -> applying to {a_arg.shot}")

    z, ref, mask, _ = nearfield_depth(a_arg.shot)
    z_cal = a * z + b
    mpx = markers_px(a_arg.shot)
    m_cal = {i: a * sample(z, u, v) + b for i, (u, v) in mpx.items()}

    fig = plt.figure(figsize=(13, 5.8), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(ref, 0, 1)); ax0.axis("off")
    ax0.set_title("led1 (reference)", fontsize=10, fontweight="bold")

    lo, hi = np.percentile(z_cal[mask] if mask.any() else z_cal, [2, 98])
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(z_cal, cmap="turbo", vmin=lo, vmax=hi)
    for i, (u, v) in mpx.items():
        ax1.add_patch(plt.Circle((u, v), 15, fill=False, ec="white", lw=1.6))
        ax1.annotate(f"{i}:{m_cal[i]:.0f}mm", (u, v), (u + 18, v), color="white",
                     fontsize=8, fontweight="bold", va="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6,
                               ec="none"))
    ax1.set_title("depth via shot_011 calibration (mm)\nlabel = distance from lens",
                  fontsize=10, fontweight="bold"); ax1.axis("off")
    plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02).set_label(
        "distance from lens (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(z_cal, 1.5); zdisp = np.clip(-zsm, -hi, -lo)
    step = max(1, z_cal.shape[1] // 140)
    yg, xg = np.mgrid[0:z_cal.shape[0]:step, 0:z_cal.shape[1]:step]
    mmpp = nf.FIELD_WIDTH_MM / z_cal.shape[1]
    ax2.plot_surface(xg * mmpp, yg * mmpp, zdisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10,
                  fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("mm")
    ax2.view_init(elev=55, azim=-60)

    fig.suptitle(f"{a_arg.shot} — near-field depth, calibrated with shot_011 "
                 "offset+scale", fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, f"z_cal = {a:.2f}*z_nf {b:+.1f}  (transferred from shot_011 "
             f"CT, corr {corr:+.2f}, RMS {rms:.1f} mm). Provisional symmetric LED "
             "geometry; no CT on this shot -> estimate, not validated.",
             ha="center", fontsize=8, style="italic", color="#555")

    out = bd.project_path("depth_outputs", "nearfield",
                          f"{a_arg.shot}_nearfield_ct_transfer.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)

    print(f"{a_arg.shot} calibrated marker depths (distance from lens):")
    for i in sorted(m_cal):
        print(f"  marker {i}: {m_cal[i]:.1f} mm")
    print(f"slide -> {out}")


if __name__ == "__main__":
    main()
