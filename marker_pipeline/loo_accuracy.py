"""
Leave-one-marker-out accuracy: how good is the depth AWAY from the fiducials
that the alignment used?

FRE (the number in the paper) is measured at the very points the rigid
transform was fitted to, so it is optimistically biased -- with n fiducials and
6 absorbed DOF, E[FRE^2] = (1 - 2/n) FLE^2 (Fitzpatrick et al. 1998), i.e. ~29%
low at n=4. It says nothing about accuracy elsewhere on the surface.

This script instead fits the rigid 6-DOF transform on n-1 markers and measures
the error at the HELD-OUT marker, which the fit never saw. That is a true
predictive (target-registration-style) error, and it is the honest answer to
"is the depth right away from the fiducials?".

Reported per method (all CT-free in the reconstruction; CT only evaluates):
    FRE   : all-marker fit, error at the fitted markers   (optimistic)
    LOO   : (n-1)-marker fit, error at the held-out marker (predictive)

Run: python loo_accuracy.py [shot_004 ...]
"""

import os
import sys
import math
import json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R
import nearfield_ctfree as CF
import marker_depth_slides as S

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
MU = 8.0


def depth_maps(shot):
    """the three candidate depth maps for one shot, all CT-free."""
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    rgbs = [bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
            for i in (1, 2, 3, 4)]
    lums = bd.balance_exposure(
        [bd.solve_luminance(np.clip(r - dark, 0, 1)) for r in rgbs])
    bright = np.stack(lums, 0).mean(0)
    Ls = np.array([bd.light_vector(S.LED_AZIMUTH_DEG[i], S.LIGHT_ELEVATION_DEG)
                   for i in (1, 2, 3, 4)])
    n, _ = bd.photometric_stereo(lums, Ls)
    z_ff = bd.normals_to_depth(n)
    z_is = 1.0 / np.sqrt(np.maximum(bright, 1e-6))
    D, _, _, _, _, _ = CF.reconstruct(shot, mu_fixed=MU, verbose=False)
    return {"inverse-square": z_is, "far-field": z_ff, "near-field": D}


def marker_points(shot, z, metric):
    """reconstructed marker positions in camera mm.

    metric=True  : z is already depth-from-lens in mm (near-field CT-free).
    metric=False : z is a relative map; it needs a 2-parameter (scale, offset)
                   fit. There is no CT-free way to set those for far-field or
                   inverse-square, so we grant them the CT-anchored affine --
                   generous to the baselines, and it cannot help the LOO test
                   discriminate in their disfavour."""
    corr = C.CT_CORR[shot]; ids = sorted(corr)
    uv = np.array([corr[i][0] for i in ids], float)
    obj = np.array([corr[i][1] for i in ids], float)
    _, rv, tv = cv2.solvePnP(obj, uv, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    Rp, _ = cv2.Rodrigues(rv)
    ct_d = (Rp @ obj.T + tv.reshape(3, 1)).T[:, 2]
    zp = np.array([R._disk(z, *corr[i][0]) for i in ids])
    if not metric:
        a, b = np.linalg.lstsq(np.column_stack([zp, np.ones(len(zp))]),
                               ct_d, rcond=None)[0]
        zp = a * zp + b
    return R.backproject(uv[:, 0], uv[:, 1], zp), obj


def loo_errors(mk, obj):
    """error at each held-out marker, transform fitted on the rest."""
    errs = []
    for k in range(len(obj)):
        keep = [j for j in range(len(obj)) if j != k]
        Rk, tk = CF.kabsch(mk[keep], obj[keep])
        pred = Rk @ mk[k] + tk
        errs.append(float(np.linalg.norm(pred - obj[k])))
    return np.array(errs)


def fre_errors(mk, obj):
    Rk, tk = CF.kabsch(mk, obj)
    return np.linalg.norm((Rk @ mk.T).T + tk - obj, axis=1)


def main():
    shots = [a for a in sys.argv[1:] if a.startswith("shot_")] or \
            ["shot_004", "shot_005", "shot_006"]
    print("Leave-one-marker-out accuracy (rigid 6-DOF, no scale correction)\n")
    print(f"{'shot':10}{'method':17}{'n':>3}{'FRE rms':>10}{'LOO rms':>10}"
          f"{'LOO max':>10}   per-held-out-marker error (mm)")
    agg = {}
    for shot in shots:
        maps = depth_maps(shot)
        for name in ("inverse-square", "far-field", "near-field"):
            mk, obj = marker_points(shot, maps[name],
                                    metric=(name == "near-field"))
            fre = fre_errors(mk, obj)
            loo = loo_errors(mk, obj)
            agg.setdefault(name, []).append(loo)
            print(f"{shot:10}{name:17}{len(obj):3d}"
                  f"{np.sqrt((fre**2).mean()):10.2f}"
                  f"{np.sqrt((loo**2).mean()):10.2f}{loo.max():10.2f}   "
                  f"{np.array2string(loo, precision=2)}")
        print()
    print("pooled over all shots:")
    print(f"{'method':17}{'LOO rms':>10}{'LOO median':>12}{'LOO max':>10}")
    for name, rows in agg.items():
        v = np.concatenate(rows)
        print(f"{name:17}{np.sqrt((v**2).mean()):10.2f}{np.median(v):12.2f}"
              f"{v.max():10.2f}")
    print("\nNote: far-field and inverse-square are granted a CT-anchored "
          "scale/offset\n(they have no CT-free way to set one); near-field is "
          "fully CT-free.")


if __name__ == "__main__":
    main()
