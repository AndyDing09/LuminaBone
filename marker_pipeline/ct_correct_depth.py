"""
Correct the (relative, bowl-distorted) photometric-stereo marker depths against
CT ground-control points.

The photometric-stereo z is RELATIVE and carries the far-field "bowl" bias, so it
is wrong by an unknown global scale + offset and by a slowly-varying spatial
tilt/bowl across the field. CT gives the TRUE 3-D position of a few markers. We
use those as control points to fit a correction and apply it to every marker
(and, optionally, the whole depth map).

Two correction models, from weakest data need to strongest:

  affine-z   z_true ~= a * z_est + b
             2 unknowns -> needs >=2 markers. Removes the global scale + offset
             the relative depth is missing. Does NOT remove the spatial bowl:
             each marker's leftover error still depends on where it sits.

  tilt-plane z_true ~= z_est + (alpha*x + beta*y + gamma)
             3 unknowns -> needs >=3 NON-COLLINEAR markers. Absorbs the first-
             order (linear) part of the bowl across the field as well as the
             offset -- usually the dominant bowl term over a small patch.

Both are least-squares fits; with exactly the minimum number of points they
interpolate (zero residual there), so trust the model only as far as the control
points spread. Run --selftest to validate the math on a synthetic bowl.
"""

import argparse
import numpy as np


def fit_affine_z(z_est, z_true):
    """z_true ~= a*z_est + b. Returns (a, b, residuals, rms)."""
    z_est = np.asarray(z_est, float)
    z_true = np.asarray(z_true, float)
    A = np.column_stack([z_est, np.ones_like(z_est)])
    (a, b), *_ = np.linalg.lstsq(A, z_true, rcond=None)
    resid = (a * z_est + b) - z_true
    return float(a), float(b), resid, float(np.sqrt(np.mean(resid ** 2)))


def apply_affine_z(z, a, b):
    return a * np.asarray(z, float) + b


def fit_tilt_plane(x, y, z_est, z_true):
    """z_true ~= z_est + (alpha*x + beta*y + gamma). Fits the residual as a
    plane in (x, y). Returns (alpha, beta, gamma, residuals, rms)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    r = np.asarray(z_true, float) - np.asarray(z_est, float)   # target residual
    A = np.column_stack([x, y, np.ones_like(x)])
    (alpha, beta, gamma), *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = (alpha * x + beta * y + gamma) - r
    return (float(alpha), float(beta), float(gamma),
            resid, float(np.sqrt(np.mean(resid ** 2))))


def apply_tilt_plane(z, x, y, alpha, beta, gamma):
    return np.asarray(z, float) + alpha * np.asarray(x, float) \
        + beta * np.asarray(y, float) + gamma


def report(model, params, ids, resid):
    print(f"  model: {model}")
    print(f"  params: {params}")
    for i, r in zip(ids, resid):
        print(f"    marker {i}: residual {r:+.3f} mm")
    print(f"  control-point RMS: {np.sqrt(np.mean(resid ** 2)):.3f} mm")


def selftest():
    """Validate each corrector on the exact distortion it is designed to remove:
    affine-z on a pure scale+offset error, tilt-plane on an offset+tilt error.
    Fit on the minimum control points, then check held-out markers recover."""
    rng = np.random.default_rng(3)
    x = np.array([9.2, 20.0, 3.7, 30.0, 15.0])         # marker positions (mm)
    y = np.array([1.6, 12.0, 13.2, 25.0, 8.0])
    z_true = np.array([11.8, 15.0, 23.3, 18.0, 13.5])

    # (1) pure scale+offset -> affine-z should recover it from 2 control points.
    a_g, b_g = 0.35, -4.0
    z_aff = (z_true - b_g) / a_g + 0.01 * rng.standard_normal(z_true.size)
    a, b, _, _ = fit_affine_z(z_aff[[0, 2]], z_true[[0, 2]])
    err_aff = np.abs(apply_affine_z(z_aff, a, b) - z_true)[[1, 3, 4]]

    # (2) offset+tilt (unit scale) -> tilt-plane should recover it from 3.
    plane = -0.22 * x + 0.13 * y + 3.0
    z_tlt = z_true - plane + 0.01 * rng.standard_normal(z_true.size)
    al, be, ga, _, _ = fit_tilt_plane(x[[0, 2, 1]], y[[0, 2, 1]],
                                      z_tlt[[0, 2, 1]], z_true[[0, 2, 1]])
    err_tlt = np.abs(apply_tilt_plane(z_tlt, x, y, al, be, ga) - z_true)[[3, 4]]

    print("CT-CORRECTION SELF-TEST")
    print(f"  affine-z: held-out error {np.round(err_aff, 3)} mm  "
          "(scale+offset distortion, 2 control pts)")
    print(f"  tilt-plane: held-out error {np.round(err_tlt, 3)} mm  "
          "(offset+tilt distortion, 3 control pts)")
    ok = err_aff.max() < 0.1 and err_tlt.max() < 0.1
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    print("Provide control points (z_est from marker_depths.csv paired with CT "
          "z_true) and call fit_affine_z / fit_tilt_plane. Run --selftest to "
          "validate.")


if __name__ == "__main__":
    main()
