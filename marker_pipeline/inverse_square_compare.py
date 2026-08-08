"""
Inverse-square depth: NO CT calibration vs CT-calibrated.

"No CT"  = ONE global scale a,b fit across ALL shots' markers pooled (a single
           pre-calibration constant, r=1/sqrt(I) -> mm, applied to every shot).
           This is what you get with no per-shot ground control.
"CT cal" = affine a,b fit per shot to that shot's own CT markers.

The gap between the two columns is exactly what per-shot CT anchoring buys.
Run: python inverse_square_compare.py
"""

import os
import sys
import numpy as np
import cv2
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])


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


def inv_sq_at_markers(shot, corr):
    ids = sorted(corr)
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    lums = []
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
    I = np.mean(lums, axis=0)
    r = 1.0 / np.sqrt(np.maximum(I, 1e-4))
    rr = np.array([sample(r, *corr[i][0]) for i in ids])
    return rr, ct_depths(corr)


def main():
    data = {s: inv_sq_at_markers(s, c) for s, c in C.CT_CORR.items()}

    # global fit across ALL markers (no per-shot CT)
    allr = np.concatenate([d[0] for d in data.values()])
    allc = np.concatenate([d[1] for d in data.values()])
    ag, bg = np.linalg.lstsq(np.column_stack([allr, np.ones(len(allr))]),
                             allc, rcond=None)[0]

    print("inverse-square marker error (mm RMS):")
    print(f"{'shot':10} {'no-CT (global)':>15} {'CT-calibrated':>15}")
    ng, nc = [], []
    for s, (rr, ct) in data.items():
        rms_g = np.sqrt(np.mean((ag * rr + bg - ct) ** 2))
        ap, bp = np.linalg.lstsq(np.column_stack([rr, np.ones(len(rr))]),
                                 ct, rcond=None)[0]
        rms_c = np.sqrt(np.mean((ap * rr + bp - ct) ** 2))
        ng.append(rms_g); nc.append(rms_c)
        print(f"{s:10} {rms_g:15.2f} {rms_c:15.2f}")
    print(f"{'mean':10} {np.mean(ng):15.2f} {np.mean(nc):15.2f}")
    print(f"\nglobal scale: z_lens = {ag:.2f}/sqrt(I) {bg:+.1f} mm  "
          f"(one constant, all shots)")
    print("CT-calibrated = per-shot affine anchor (best case for inverse-square).")


if __name__ == "__main__":
    main()
