"""
calib_lights.py — per-LED photometric calibration from the white board half.

Implements the Batlle et al. (IROS 2022, eq. 7) near-field forward model and fits
it to the blank-white-half samples, whose 3-D pose comes from the coplanar ChArUco
half (Charuco_Calibration.py). One capture set -> intrinsics AND photometry.

Forward model, corrected gray level of a board point x with normal n:

    I_i(x) = ( Phi_i * cos^{k_i}(psi_i)/r_i^2 * f_r(theta_i) * {cos theta_i}_+
                     * V(alpha) * g )^(1/gamma)

  psi_i   angle of the surface point off LED i's own axis  (LED falloff)
  theta_i incidence angle: board normal vs surface->LED    (Lambert + BRDF)
  f_r     separable BRDF, 8 knots over theta in [0, pi/2]   (paper is specular
          near normal; fit on the board, swap bone's in later)
  V       camera vignetting cos^{k'}(alpha), alpha = pixel off-axis angle
  gamma   camera response (~2.2). NONLINEAR -- fit it, don't skip it.
  g       gain, fixed (we lock exposure).

Free parameters (per the protocol's "28 light params + gamma + vignetting + BRDF"):
  per LED: position(3) + direction(2) + falloff k(1) + flux Phi(1) = 7  -> 28
  + gamma(1) + vignetting k'(1) + BRDF knots 1..7 (knot0 pinned to 1) = 37 total.

Scale degeneracy (Phi * f_r * g) is broken by g=1 and f_r(0)=1.

Run:  python calib_lights.py --selftest    # plant a 2 mm axial offset, recover it
"""

import argparse
import numpy as np
from scipy.optimize import least_squares

A_MM = 6.05                                   # measured LED emitter radius (CLAUDE.md)
AZ_DEG = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}
N_LED = 4
N_BRDF = 8
BRDF_THETA = np.linspace(0.0, np.pi / 2, N_BRDF)   # knot angles


# ---------------------------------------------------------------------------
# parameter packing:  positions(12) dir_xy(8) k(4) flux(4) gamma vign brdf1..7
# ---------------------------------------------------------------------------

def pack(P, dxy, k, flux, gamma, vign, brdf):
    return np.concatenate([P.ravel(), dxy.ravel(), k, flux,
                           [gamma, vign], brdf[1:]])


def unpack(v):
    i = 0
    P = v[i:i + 12].reshape(4, 3); i += 12
    dxy = v[i:i + 8].reshape(4, 2); i += 8
    k = v[i:i + 4]; i += 4
    flux = v[i:i + 4]; i += 4
    gamma = v[i]; vign = v[i + 1]; i += 2
    brdf = np.concatenate([[1.0], v[i:i + (N_BRDF - 1)]])
    dirs = np.column_stack([dxy, np.ones(4)])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return P, dirs, k, flux, gamma, vign, brdf


def brdf_eval(theta, brdf):
    return np.interp(theta, BRDF_THETA, brdf)


# ---------------------------------------------------------------------------
# forward model
# ---------------------------------------------------------------------------

def predict(v, X, N, alpha):
    """Predicted per-LED gray levels for samples X (M,3), normals N (M,3),
    pixel off-axis angle alpha (M,). Returns (M, 4)."""
    P, dirs, k, flux, gamma, vign, brdf = unpack(v)
    V = np.cos(np.clip(alpha, 0, np.pi / 2)) ** max(vign, 0.0)     # (M,)
    out = np.empty((X.shape[0], N_LED))
    for i in range(N_LED):
        d = X - P[i][None, :]                     # LED -> surface
        r = np.linalg.norm(d, axis=1)
        omega = d / r[:, None]
        cos_psi = np.clip(omega @ dirs[i], 0.0, 1.0)          # off LED axis
        cos_theta = np.clip(-(omega * N).sum(1), 0.0, 1.0)    # incidence
        theta = np.arccos(cos_theta)
        E = (flux[i] * cos_psi ** np.clip(k[i], 0, None) / r ** 2
             * brdf_eval(theta, brdf) * cos_theta * V)
        out[:, i] = np.clip(E, 1e-12, None) ** (1.0 / max(gamma, 0.3))
    return out


def residuals(v, X, N, alpha, meas):
    return (predict(v, X, N, alpha) - meas).ravel()


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def initial_guess():
    P = np.array([[A_MM * np.cos(np.radians(AZ_DEG[i])),
                   A_MM * np.sin(np.radians(AZ_DEG[i])), 0.0]
                  for i in range(4)])
    dxy = np.zeros((4, 2))
    k = np.full(4, 1.0)
    flux = np.full(4, 1.0)
    return pack(P, dxy, k, flux, 2.2, 2.0, np.ones(N_BRDF))


def fit(X, N, alpha, meas, v0=None, verbose=0):
    v0 = initial_guess() if v0 is None else v0
    res = least_squares(residuals, v0, args=(X, N, alpha, meas),
                        method="trf", max_nfev=4000, verbose=verbose)
    return res.x, res


# ---------------------------------------------------------------------------
# self-test: plant a config with a 2 mm axial offset, render, fit, recover
# ---------------------------------------------------------------------------

def _synth_poses(rng, n=28):
    """Diverse board planes: distances 15-45 mm, tilts to 40 deg. Pose diversity
    is what separates LED spread from vignetting (CLAUDE.md)."""
    poses = []
    for _ in range(n):
        z0 = rng.uniform(15, 45)
        tilt = np.radians(rng.uniform(0, 40))
        az = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(az), np.sin(az), 0.0])
        Rm = _rot(axis, tilt)
        t = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), z0])
        poses.append((Rm, t))
    return poses


def _rot(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _samples(poses, fx, cx, cy, grid=7, half=12.0):
    """Board points on the WHITE half of each plane -> (X, N, alpha)."""
    gx, gy = np.meshgrid(np.linspace(-half, half, grid),
                         np.linspace(-half, half, grid))
    local = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    X, N, AL = [], [], []
    for Rm, t in poses:
        x = local @ Rm.T + t
        n = np.tile(Rm @ np.array([0, 0, -1.0]), (x.shape[0], 1))  # faces camera
        front = x[:, 2] > 1.0
        x, n = x[front], n[front]
        u = fx * x[:, 0] / x[:, 2] + cx
        vv = fx * x[:, 1] / x[:, 2] + cy
        xn = (u - cx) / fx; yn = (vv - cy) / fx
        alpha = np.arctan(np.sqrt(xn ** 2 + yn ** 2))
        X.append(x); N.append(n); AL.append(alpha)
    return np.vstack(X), np.vstack(N), np.concatenate(AL)


def selftest(seed=0, axial_offset=2.0, noise=0.0):
    rng = np.random.default_rng(seed)
    fx, cx, cy = 450.0, 320.0, 240.0

    # ground truth: symmetric LEDs at radius a, but with a planted axial offset
    P = np.array([[A_MM * np.cos(np.radians(AZ_DEG[i])),
                   A_MM * np.sin(np.radians(AZ_DEG[i])), 0.0] for i in range(4)])
    P[:, 2] = axial_offset * np.array([0.0, 1.0, 0.5, -0.5])   # planted spread
    dxy = np.array([[0.05, 0.0], [-0.03, 0.04], [0.0, -0.05], [0.02, 0.02]])
    k = np.array([1.1, 0.9, 1.0, 1.2])
    flux = np.array([1.0, 0.85, 1.1, 0.95])
    gamma, vign = 2.2, 2.5
    brdf = np.array([1.0, 0.95, 0.9, 0.86, 0.83, 0.82, 0.85, 0.95])  # specular-ish
    v_true = pack(P, dxy, k, flux, gamma, vign, brdf)

    poses = _synth_poses(rng)
    X, N, alpha = _samples(poses, fx, cx, cy)
    meas = predict(v_true, X, N, alpha)
    if noise:
        meas = meas * (1 + noise * rng.standard_normal(meas.shape))

    v_est, res = fit(X, N, alpha, meas)
    Pe, dire, ke, fluxe, ge, vge, brdfe = unpack(v_est)

    axial_err = float(np.max(np.abs(Pe[:, 2] - P[:, 2])))
    pos_err = float(np.max(np.linalg.norm(Pe - P, axis=1)))
    print("CALIB_LIGHTS SELF-TEST (planted 2 mm axial spread, 28+ params)")
    print(f"  poses {len(poses)}  samples {X.shape[0]}  final cost {res.cost:.2e}")
    print(f"  planted axial z : {np.round(P[:,2],3)}")
    print(f"  recovered axial : {np.round(Pe[:,2],3)}")
    print(f"  max axial error : {axial_err*1000:.3f} um   max pos error "
          f"{pos_err*1000:.3f} um")
    print(f"  gamma {ge:.3f} (true {gamma})   vign {vge:.3f} (true {vign})")
    # noise-free must be ~exact; with noise, allow graceful degradation (the fit
    # is noise-limited, not wrong -- this is the sensitivity curve for the paper).
    tol = max(0.05, 40.0 * noise)               # mm
    ok = axial_err < tol and pos_err < 2 * tol
    print(f"  -> {'PASS' if ok else 'FAIL'}  (tolerance {tol*1000:.0f} um "
          f"at noise {noise*100:.1f}%)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="relative sensor noise for the self-test")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest(noise=a.noise) else 1)
    print("Capture the ChArUco + white-half set, calibrate intrinsics/poses with "
          "Charuco_Calibration.py, then feed the white-half samples here. "
          "Run --selftest to validate the fitter.")


if __name__ == "__main__":
    main()
