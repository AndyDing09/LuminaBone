"""
Register the inverse-square depth map INTO the CT coordinate frame.

Pipeline per shot:
  1. inverse-square depth  r = 1/sqrt(I),  CT-calibrated per shot  z = a*r + b  (mm from lens)
  2. back-project every bone pixel to 3-D camera coords with the real intrinsics
     (undistortPoints handles k1..k3):   X_cam = z * [x_n, y_n, 1]
  3. camera -> CT frame using the marker PnP pose (x_cam = R x_ct + t):
        X_ct = R^T (X_cam - t)
  4. write the surface as a coloured PLY in CT coordinates + a 3-D overlay against
     the CT marker points, and report the full 3-D marker residual (mm).

Output PLYs live in depth_outputs/ct_registered/ and load directly alongside the
CT model (same coordinate frame, mm). Run: python register_inverse_square_ct.py
"""

import os
import sys
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
OUT = bd.project_path("depth_outputs", "ct_registered")
os.makedirs(OUT, exist_ok=True)


def pose(corr):
    """CT->camera pose (x_cam = R x_ct + t) from the markers."""
    ids = sorted(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    img = np.array([corr[i][0] for i in ids])
    _, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(3)


def backproject(u, v, z):
    """pixels + depth-from-lens -> camera-frame 3-D (mm), distortion-corrected."""
    pts = np.stack([u, v], axis=-1).astype(np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, K, DIST).reshape(-1, 2)   # [x_n, y_n]
    rays = np.column_stack([norm, np.ones(len(norm))])        # z=1 plane
    return rays * z.reshape(-1, 1)                             # scale by depth


def write_ply(path, xyz, rgb):
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, zz), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.3f} {y:.3f} {zz:.3f} {r} {g} {b}\n")


def depth_map(shot, corr):
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
    r = 1.0 / np.sqrt(np.maximum(I, 1e-4))
    # per-shot affine calibration to CT marker depths
    Rp, tp = pose(corr)
    obj = np.array([corr[i][1] for i in ids], float)
    ct_depth = (Rp @ obj.T + tp.reshape(3, 1)).T[:, 2]
    rr = np.array([_disk(r, *corr[i][0]) for i in ids])
    a, b = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]),
                           ct_depth, rcond=None)[0]
    return a * r + b, ref, bd.bone_mask(I), Rp, tp, a, b


def _disk(z, u, v, rad=14):
    H, W = z.shape
    yy, xx = np.ogrid[:H, :W]
    return float(np.median(z[(xx - u) ** 2 + (yy - v) ** 2 <= rad * rad]))


def main():
    print("register inverse-square surface into CT frame (mm):")
    for shot, corr in C.CT_CORR.items():
        ids = sorted(corr)
        z, ref, mask, R, t, a, b = depth_map(shot, corr)
        # clip the surface to the CT marker depth band (+/- 15 mm): the raw
        # inverse-square depth fans out to 100s of mm on dark (low-albedo) pixels.
        obj = np.array([corr[i][1] for i in ids], float)
        ct_d = (R @ obj.T + t.reshape(3, 1)).T[:, 2]
        zlo, zhi = ct_d.min() - 15, ct_d.max() + 15
        keep = mask & (z >= zlo) & (z <= zhi)
        vv, uu = np.where(keep)
        zc = z[vv, uu]
        cam = backproject(uu.astype(float), vv.astype(float), zc)
        ct = (R.T @ (cam - t).T).T                       # camera -> CT frame
        col = ref[vv, uu]
        ply = os.path.join(OUT, f"{shot}_surface_ct.ply")
        write_ply(ply, ct, col)

        # 3-D marker residual: reconstruct each marker in CT frame, compare to truth
        mk_uv = np.array([corr[i][0] for i in ids], float)
        mk_z = np.array([_disk(z, *corr[i][0]) for i in ids])
        mk_cam = backproject(mk_uv[:, 0], mk_uv[:, 1], mk_z)
        mk_ct = (R.T @ (mk_cam - t).T).T
        truth = np.array([corr[i][1] for i in ids], float)
        err = np.linalg.norm(mk_ct - truth, axis=1)
        print(f"  {shot}: {len(ct):6d} pts   3-D marker RMS {np.sqrt(np.mean(err**2)):5.2f} mm"
              f"   (per-marker {np.round(err,1)})  -> {os.path.basename(ply)}")

        # overlay figure in CT frame
        fig = plt.figure(figsize=(7, 6), dpi=120)
        fig.patch.set_facecolor("#f5f5f0")
        ax = fig.add_subplot(111, projection="3d")
        s = max(1, len(ct) // 6000)
        ax.scatter(ct[::s, 0], ct[::s, 1], ct[::s, 2], c=col[::s], s=2, alpha=0.5)
        ax.scatter(truth[:, 0], truth[:, 1], truth[:, 2], c="k", s=90,
                   marker="X", label="CT markers (truth)", depthshade=False)
        ax.scatter(mk_ct[:, 0], mk_ct[:, 1], mk_ct[:, 2], facecolors="none",
                   edgecolors="r", s=140, linewidths=2,
                   label="reconstructed markers")
        pad = 20
        ax.set_xlim(truth[:, 0].min() - pad, truth[:, 0].max() + pad)
        ax.set_ylim(truth[:, 1].min() - pad, truth[:, 1].max() + pad)
        ax.set_zlim(truth[:, 2].min() - pad, truth[:, 2].max() + pad)
        ax.set_xlabel("CT x (mm)"); ax.set_ylabel("CT y (mm)"); ax.set_zlabel("CT z (mm)")
        ax.set_title(f"{shot} — inverse-square surface registered to CT\n"
                     f"3-D marker RMS {np.sqrt(np.mean(err**2)):.1f} mm",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        fig.savefig(os.path.join(OUT, f"{shot}_ct_overlay.png"),
                    bbox_inches="tight", facecolor="#f5f5f0"); plt.close(fig)
    print(f"\nPLYs + overlays -> {OUT}")
    print("Load *_surface_ct.ply next to your CT model (same mm frame).")


if __name__ == "__main__":
    main()
