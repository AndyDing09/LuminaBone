"""
CT-FREE near-field photometric stereo: metric depth with NO ground truth in the
reconstruction. (The CT markers are used for EVALUATION ONLY, after the fact.)

Where nearfield_ct.py leaked CT into the solve (bootstrap plane at the CT
working distance, per-iteration affine anchor to CT marker depths, CT depth
band in the trim), this script recovers the same two numbers photometrically:

    D(u,v) = b + s * (b / f) * ztilde(u,v)

  b  working distance (mm)      -- the affine offset
  s  relief gain (dimensionless) -- the affine scale, expressed relative to the
                                    perspective-consistent value b/f so s ~ 1
                                    when the pinhole geometry is self-consistent

Both are found by minimising the 4-LED PHOTOMETRIC RESIDUAL: with 4 lights and
3 unknowns per pixel (g = rho*n) the per-pixel solve is overdetermined, and the
leftover residual measures whether one consistent surface explains all four
images under the assumed geometry. The LED ring radius (6.05 mm, a hardware
measurement) fixes the absolute length scale inside the model, so a wrong b or
s makes the four images mutually inconsistent -> the residual has a minimum at
the true geometry. This is the near-field property far-field PS lacks.

Evaluation (registration is allowed to be rigid ONLY -- 6 DOF, no scale):
  * pairwise marker-distance ratio vs CT  (registration-free scale check)
  * rigid Kabsch onto the CT markers      -> 3-D FRE
  * similarity Umeyama                    -> recovered scale (should be ~1)

Run: python nearfield_ctfree.py            # shot_004, verbose + landscape plot
     python nearfield_ctfree.py all        # every CT shot
"""

import os
import sys
import math
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_erosion, binary_opening, label,
                           median_filter, gaussian_filter)
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import register_inverse_square_ct as R          # backproject / _disk helpers
from nearfield_ct import (A_MM, AZ, P, load_lums, grid_faces, write_ply_mesh)

cal = json.load(open(bd.project_path("calibration", "calibration.txt")))
K = np.array(cal["camera_matrix"]); DIST = np.array(cal["dist_coeffs"])
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
OUT = bd.project_path("depth_outputs", "ct_registered")
os.makedirs(OUT, exist_ok=True)

DS = 4                                   # downsample factor for the (b,s) search
AIM_DX = None                            # LED beam-crossing distance (mm); None =
                                         # parallel aim. Rig-level, like mu.
TRIM = 0.0                               # fraction of worst pixels dropped from
                                         # the objective (robustness to glints)
UNDISTORT_GRID = False                   # True: shading solve uses the same
                                         # undistorted rays as backprojection
# The working-distance prior is the fixed-focus endoscope's depth-of-field
# band: a bench-measurable LENS property (images are only sharp in this range,
# which is why every capture sits near ~55 mm). It is NOT taken from CT. The
# photometric objective is shallow and multi-modal in (b, mu) on this
# near-coaxial rig; the focus band excludes the false close-and-flat basin.
B_RANGE = (45.0, 70.0)
S_RANGE = (0.10, 5.0)                    # relief gain bounds (s~1 expected)
MU_RANGE = (0.0, 14.0)                   # LED emission exponent cos^mu bounds
                                         # (mu ~ 11 for a 20-deg half-angle LED)


# ---------------------------------------------------------------------------
# forward model pieces (parameterised intrinsics so they work downsampled)
# ---------------------------------------------------------------------------

_GRIDS = {}


def norm_grid(fx, fy, cx, cy, H, W):
    """undistorted normalised ray coords (xn, yn) for every pixel, cached.

    The backprojection path undistorts (cv2.undistortPoints); the shading
    solve must place surface points with the SAME rays or edge pixels get
    misplaced light vectors. Scaled K with the same coefficients is exact:
    distortion acts on normalised coords, and (u/s - cx/s)/(fx/s) is the
    same normalised coordinate at any image scale s.
    """
    key = (round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6), H, W,
           UNDISTORT_GRID)
    if key not in _GRIDS:
        U, V = np.meshgrid(np.arange(W, dtype=np.float64),
                           np.arange(H, dtype=np.float64))
        if UNDISTORT_GRID:
            Kl = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            pts = np.stack([U.ravel(), V.ravel()], -1).reshape(-1, 1, 2)
            n = cv2.undistortPoints(pts, Kl, DIST).reshape(H, W, 2)
            xn = np.clip(n[..., 0], -2.0, 2.0)  # guard iterative-inversion blowups
            yn = np.clip(n[..., 1], -2.0, 2.0)
        else:
            # legacy pinhole grid: reproduces the published table. The 2026-08-11
            # sweep found the undistorted grid SHUFFLES errors rather than
            # reducing them (suspect: the extreme k3 in calibration.txt), so the
            # consistent-rays variant stays opt-in until recalibration.
            xn, yn = (U - cx) / fx, (V - cy) / fy
        _GRIDS[key] = (xn, yn)
    return _GRIDS[key]


def led_axes(aim_dx):
    """unit emission axis per LED: parallel to the optical axis, or aimed at
    the axis-crossing point (0,0,-aim_dx) -- the convergent aim the beam-map
    study observed (led_wiring_ctfree.py, 5/5 shots)."""
    axes = {}
    for s in (1, 2, 3, 4):
        a = (np.array([0.0, 0.0, -1.0]) if not aim_dx else
             np.array([-P[s][0], -P[s][1], -aim_dx]))
        axes[s] = a / np.linalg.norm(a)
    return axes


def geometry(D, fx, fy, cx, cy, mu=0.0, aim_dx=None):
    """per-pixel unit light vectors (H,W,4,3) + attenuation (H,W,4).

    attenuation = cos^mu(theta_s) / r^2 : inverse-square times a Lambertian-
    emitter anisotropy with exponent mu. theta_s is measured from the LED's
    emission axis: parallel to the optical axis by default, or aimed at the
    crossing point (0,0,-aim_dx) when aim_dx is set (convergent-aim model).
    mu = 0 is the isotropic point source.
    """
    H, W = D.shape
    xn, yn = norm_grid(fx, fy, cx, cy, H, W)
    Xw = np.stack([xn * D, -yn * D, -D], axis=-1)
    axes = led_axes(aim_dx)
    Lhat = np.empty((H, W, 4, 3)); att = np.empty((H, W, 4))
    for k, s in enumerate((1, 2, 3, 4)):
        w = P[s][None, None, :] - Xw                     # surface -> LED
        d2 = np.sum(w * w, axis=-1)
        r = np.sqrt(d2)
        att[..., k] = 1.0 / d2
        if mu:
            cos = np.einsum("hws,s->hw", -w, axes[s]) / r   # LED -> surface vs axis
            att[..., k] *= np.clip(cos, 0.0, None) ** mu
        Lhat[..., k, :] = w / r[..., None]
    return Lhat, att


def solve_g(M, Lhat, att):
    """per-pixel least squares for g = rho*n from the 4 lit images."""
    m = M / att                                          # rho (n . l)
    A = np.einsum("hwks,hwkt->hwst", Lhat, Lhat)
    rhs = np.einsum("hwks,hwk->hws", Lhat, m)
    g = np.linalg.solve(A + 1e-6 * np.eye(3), rhs[..., None])[..., 0]
    return g


def photo_residual(M, g, Lhat, att, mask):
    """mean relative 4-image inconsistency over the mask (free-normal solve)."""
    Ipred = np.einsum("hws,hwks->hwk", g, Lhat) * att
    r = np.abs(M - Ipred).sum(-1) / (M.sum(-1) + 1e-6)
    return float(r[mask].mean())


def normals_from_depth(D, fx, fy, cx, cy):
    """exact perspective normals of the surface Xw(u,v) implied by depth D.

    This is the load-bearing difference from the free-normal solve: here the
    normal is TIED to the candidate surface, so a wrong (b, s) produces wrong
    slopes and a shading misfit the per-pixel solve cannot absorb. (A freely
    solved normal tilts to compensate, which is why that objective is flat.)
    """
    H, W = D.shape
    xn, yn = norm_grid(fx, fy, cx, cy, H, W)
    Xw = np.stack([xn * D, -yn * D, -D], axis=-1)
    du = np.gradient(Xw, axis=1)                          # tangent along u
    dv = np.gradient(Xw, axis=0)                          # tangent along v
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    n[n[..., 2] < 0] *= -1.0                              # orient toward camera
    return n


# ---------------------------------------------------------------------------
# the CT-free (b, s) recovery
# ---------------------------------------------------------------------------

def _shrink(img, ds):
    return cv2.resize(img, (img.shape[1] // ds, img.shape[0] // ds),
                      interpolation=cv2.INTER_AREA)


class Objective:
    """E(b, s) evaluated on downsampled images (fast enough for a dense grid)."""

    def __init__(self, lums, mask):
        self.Ms = np.stack([_shrink(l, DS) for l in lums], axis=-1)  # (h,w,4)
        m = _shrink(mask.astype(np.float32), DS) > 0.5
        self.mask = binary_erosion(m, iterations=2)
        self.fx, self.fy = FX / DS, FY / DS
        self.cx, self.cy = CX / DS, CY / DS
        self.zt = np.zeros(self.Ms.shape[:2])            # ztilde, downsampled

    def set_relief(self, ztilde, zt_fx):
        """store relief NORMALISED by the focal length of the grid whose
        pixels its gradients were integrated over (zt_fx). This makes the
        depth family resolution-invariant: relief_mm = s * b * (ztilde/zt_fx)
        means the same surface whether the search runs at DS or full res.
        (An earlier version used self.fx here -- the search then optimised a
        surface with DSx the relief of the one reconstruct() built.)"""
        h, w = self.Ms.shape[:2]
        zt = cv2.resize(ztilde.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_AREA).astype(np.float64)
        zt -= np.median(zt[self.mask])
        # clip runaway border ramps so they cannot drive the search
        lo, hi = np.percentile(zt[self.mask], [1, 99])
        self.zt = np.clip(zt, lo, hi) / zt_fx

    def depth(self, b, s):
        return b * (1.0 + s * self.zt)

    def __call__(self, b, s, mu=0.0):
        """tied-normal shading misfit of the candidate surface D(b, s).

        Normals come from the candidate depth itself (not a free per-pixel
        solve); the only per-pixel freedom is albedo, plus one gain per LED
        and one global emission exponent mu. Albedo absorbs texture + lens
        vignetting (common mode across LEDs); the gains absorb LED-power
        differences; mu absorbs the LEDs' angular falloff; what remains is
        geometry.
        """
        if not (B_RANGE[0] <= b <= B_RANGE[1] and S_RANGE[0] <= s <= S_RANGE[1]
                and MU_RANGE[0] <= mu <= MU_RANGE[1]):
            return 1e9
        D = self.depth(b, s)
        if D.min() <= 2.0:                               # surface through the lens
            return 1e9
        n = normals_from_depth(D, self.fx, self.fy, self.cx, self.cy)
        Lhat, att = geometry(D, self.fx, self.fy, self.cx, self.cy, mu=mu,
                             aim_dx=AIM_DX)
        m = np.einsum("hws,hwks->hwk", n, Lhat) * att    # model shading (H,W,4)
        m = np.maximum(m, 0.0)
        I = self.Ms
        w = (m > 1e-12).astype(np.float64)               # skip attached shadows
        gam = np.ones(4)
        for _ in range(2):                               # bilinear rho/gain fit
            mg = m * gam
            rho = (w * I * mg).sum(-1) / np.maximum((w * mg * mg).sum(-1), 1e-12)
            pred = rho[..., None] * m
            msk = self.mask
            gam = ((w * I * pred)[msk].sum(0) /
                   np.maximum((w * pred * pred)[msk].sum(0), 1e-12))
        pred = rho[..., None] * m * gam
        r = (w * np.abs(I - pred)).sum(-1) / np.maximum((w * I).sum(-1), 1e-9)
        rv = r[self.mask]
        if TRIM > 0 and rv.size > 100:                   # drop glint/pit outliers
            rv = np.sort(rv)[: int(rv.size * (1.0 - TRIM))]
        out = float(rv.mean())
        return out if np.isfinite(out) else 1e9          # NaN poisons min()/keep-best


def search_bs(E, b0=None, mu0=None, mu_fixed=None):
    """coarse grid + Nelder-Mead refine of (b, s[, mu]). CT-free.

    mu_fixed pins the LED emission exponent (it is a property of the rig, not
    the scene -- letting each shot pick its own mu opens a (b, mu) ridge that
    drops trajectories into a false close-and-flat basin)."""
    bs = np.arange(B_RANGE[0], B_RANGE[1] + 1e-9, 5.0) if b0 is None \
        else np.arange(max(B_RANGE[0], b0 - 20), min(B_RANGE[1], b0 + 20) + 1e-9, 5.0)
    ss = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    if mu_fixed is not None:
        grid = [(E(b, s, mu_fixed), b, s) for b in bs for s in ss]
        _, b_best, s_best = min(grid)
        res = minimize(lambda x: E(x[0], math.exp(x[1]), mu_fixed),
                       x0=[b_best, math.log(s_best)], method="Nelder-Mead",
                       options=dict(xatol=0.05, fatol=1e-6, maxiter=200))
        return (float(res.x[0]), float(math.exp(res.x[1])), float(mu_fixed),
                float(res.fun))
    mus = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]) if mu0 is None \
        else np.clip(np.array([mu0 - 1.0, mu0, mu0 + 1.0]), *MU_RANGE)
    grid = [(E(b, s, mu), b, s, mu) for b in bs for s in ss for mu in mus]
    _, b_best, s_best, mu_best = min(grid)
    res = minimize(lambda x: E(x[0], math.exp(x[1]), x[2]),
                   x0=[b_best, math.log(s_best), mu_best], method="Nelder-Mead",
                   options=dict(xatol=0.05, fatol=1e-6, maxiter=300))
    return (float(res.x[0]), float(math.exp(res.x[1])), float(res.x[2]),
            float(res.fun))


def flat_scan(E):
    """stage 0: working-distance x emission-exponent scan with a flat plane."""
    bs = np.arange(B_RANGE[0], B_RANGE[1] + 1e-9, 2.0)
    mus = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    Es = np.array([[E(b, 1.0, mu) for mu in mus] for b in bs])  # zt=0
    return bs, mus, Es


def _trajectory(E, M, base_mask, fx, fy, cx, cy, b0, iters, mu_fixed=None,
                mu0=None, verbose=False, tag=""):
    """one fixed-point trajectory from seed b0; returns (best, hist).
    best = (residual, b, s, mu, ztilde) at the photometrically best iterate --
    the map is not monotone (ztilde moves under the search), so keep-best is
    the honest CT-free selection. Works at any resolution: pass the matching
    (M, base_mask, fx, fy, cx, cy); s is resolution-invariant because the
    depth family scales ztilde by b/fx of the SAME grid."""
    H, W = M.shape[:2]
    b, mu = b0, (mu_fixed if mu_fixed is not None else (mu0 or 0.0))
    D = np.full((H, W), b)
    hist, best = [], None
    for it in range(iters):
        Lhat, att = geometry(D, fx, fy, cx, cy, mu=mu, aim_dx=AIM_DX)
        g = solve_g(M, Lhat, att)
        alb = np.linalg.norm(g, axis=-1)
        n = g / np.where(alb > 1e-9, alb, 1.0)[..., None]
        n[n[..., 2] < 0] *= -1.0
        ztilde = bd.normals_to_depth(n)
        ztilde = ztilde - np.median(ztilde[base_mask])
        E.set_relief(ztilde, fx)
        if mu_fixed is not None:
            b, s, mu, e = search_bs(E, b0=b, mu_fixed=mu_fixed)
        else:
            b, s, mu, e = search_bs(E, b0=b, mu0=(mu if it else None))
        D = np.maximum(b + s * (b / fx) * ztilde, 2.0)   # border ramps in ztilde
        hist.append((b, s, mu, e))                       # must not go behind lens
        if best is None or e < best[0]:
            best = (e, b, s, mu, ztilde.copy())
        if verbose:
            print(f"  {tag}iter {it+1}: b = {b:6.2f} mm  s = {s:5.3f}  "
                  f"mu = {mu:4.2f}  residual {e:.4f}")
    return best, hist


B_STARTS = (48.0, 57.0, 66.0)            # multi-start seeds inside the focus band
MU_FILE = os.path.join(HERE, "ctfree_mu.json")


def calibrate_mu(shots, mu_grid=(0, 2, 4, 6, 8, 10, 12), iters=3,
                 b_starts=B_STARTS):
    """rig-level LED emission exponent, jointly across shots. CT-free.

    mu is hardware (one LED type on one ring), so every shot must share it:
    for each candidate mu, each shot reports its best achievable residual
    (over seeds); votes are per-shot normalised so no shot dominates.
    Runs entirely at the DS resolution -- plenty for a model-selection vote."""
    print(f"calibrating LED emission exponent over {list(shots)} ...")
    R_tab = {}
    for shot in shots:
        lums, _ = load_lums(shot)
        base_mask = bd.bone_mask(np.stack(lums, 0).mean(0))
        E = Objective(lums, base_mask)
        row = []
        for mu in mu_grid:
            e_best = min(_trajectory(E, E.Ms, E.mask, E.fx, E.fy, E.cx, E.cy,
                                     b0, iters, mu_fixed=float(mu))[0][0]
                         for b0 in b_starts)
            row.append(e_best)
        R_tab[shot] = np.array(row)
        print(f"  {shot}: " + "  ".join(f"mu{m}:{e:.4f}"
                                        for m, e in zip(mu_grid, row)))
    votes = sum(R_tab[s] / R_tab[s].min() for s in shots)
    mu_star = float(mu_grid[int(np.argmin(votes))])
    print(f"  -> shared mu* = {mu_star:.1f} "
          f"(votes: {np.array2string(votes, precision=3)})")
    json.dump({"mu": mu_star,
               "votes": {m: float(v) for m, v in zip(mu_grid, votes)},
               "per_shot": {s: [float(x) for x in R_tab[s]] for s in R_tab}},
              open(MU_FILE, "w"), indent=1)
    print(f"  saved -> {MU_FILE}")
    return mu_star


def reconstruct(shot, iters=6, mu_fixed=None, multi_start=True, verbose=True):
    """CT-free near-field solve. Returns full-res D (mm), plus diagnostics.
    NOTHING derived from CT enters this function."""
    lums, ref = load_lums(shot)
    I = np.stack(lums, axis=0)
    H, W = I.shape[1:]
    M = np.moveaxis(I, 0, -1)
    base_mask = bd.bone_mask(I.mean(0))
    E = Objective(lums, base_mask)

    if multi_start:
        bs, mus, Es = None, None, None
        # full-res trajectories from each seed in the focus band (the DS
        # objective votes for the wrong basin, so triage must be full-res)
        best, hist = None, []
        for b0 in B_STARTS:
            bi, hi = _trajectory(E, M, base_mask, FX, FY, CX, CY, b0, iters,
                                 mu_fixed=mu_fixed)
            hist += hi
            if verbose:
                print(f"  start b0={b0:5.1f}: best b = {bi[1]:6.2f} mm  "
                      f"s = {bi[2]:5.3f}  mu = {bi[3]:5.2f}  "
                      f"residual {bi[0]:.4f}")
            if best is None or bi[0] < best[0]:
                best = bi
    else:
        bs, mus, Es = flat_scan(E)                        # seed: photometric scan
        ib, imu = np.unravel_index(np.argmin(Es), Es.shape)
        b0, mu0 = float(bs[ib]), float(mus[imu])
        if verbose:
            print(f"  flat-plane scan: b0 = {b0:.1f} mm, mu0 = {mu0:.0f} "
                  f"(residual {Es.min():.4f}, worst {Es.max():.4f})")
        best, hist = _trajectory(E, M, base_mask, FX, FY, CX, CY, b0, iters,
                                 mu_fixed=mu_fixed, mu0=mu0, verbose=verbose)

    e, b, s, mu, ztilde = best
    D = np.maximum(b + s * (b / FX) * ztilde, 2.0)
    hist.append((b, s, mu, e))
    if verbose:
        print(f"  best kept: b = {b:.2f} mm  s = {s:.3f}  mu = {mu:.2f}"
              f"  residual {e:.4f}")

    # final full-res solve for the confidence gate
    Lhat, att = geometry(D, FX, FY, CX, CY, mu=mu, aim_dx=AIM_DX)
    g = solve_g(M, Lhat, att)
    Ipred = np.einsum("hws,hwks->hwk", g, Lhat) * att
    resid = np.abs(M - Ipred).sum(-1) / (M.sum(-1) + 1e-6)

    # trim: residual confidence + spike + border + components. NO CT band --
    # the depth band is self-referential (robust spread of D itself).
    conf = resid < 0.22
    med = np.median(D[base_mask])
    band = np.abs(D - med) < 25.0
    spike = np.abs(D - median_filter(D, size=5)) < 8.0
    keep = base_mask & conf & band & spike
    bm = int(0.09 * min(H, W))
    border = np.zeros_like(keep); border[bm:-bm, bm:-bm] = True
    keep &= border
    keep = binary_opening(keep, iterations=2)
    lb, num = label(keep)
    if num:
        sizes = np.array([(lb == i).sum() for i in range(1, num + 1)])
        big = 1 + np.where(sizes >= max(300, 0.15 * sizes.max()))[0]
        keep = np.isin(lb, big)
    keep = binary_erosion(keep, iterations=1)

    w = keep.astype(float)
    Dsm = gaussian_filter(D * w, 2.0) / np.maximum(gaussian_filter(w, 2.0), 1e-6)
    Dsm = np.where(keep, Dsm, D)
    scan = (bs, mus, Es) if Es is not None else None
    return D, Dsm, ref, keep, scan, hist


# ---------------------------------------------------------------------------
# evaluation (CT enters here ONLY, and only through rigid/at-a-distance checks)
# ---------------------------------------------------------------------------

def kabsch(Pp, Q):
    """rigid 6-DOF P->Q (no scale)."""
    cP, cQ = Pp.mean(0), Q.mean(0)
    Hm = (Pp - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return Rk, cQ - Rk @ cP


def umeyama_scale(Pp, Q):
    """similarity 7-DOF P->Q; returns the recovered scale (1.0 = metric-true)."""
    cP, cQ = Pp.mean(0), Q.mean(0)
    Sig = (Q - cQ).T @ (Pp - cP) / len(Pp)
    U, Dg, Vt = np.linalg.svd(Sig)
    Ssel = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Ssel[2, 2] = -1.0
    varP = ((Pp - cP) ** 2).sum() / len(Pp)
    return float(np.trace(np.diag(Dg) @ Ssel) / varP)


def evaluate(shot, D, keep):
    corr = C.CT_CORR[shot]; ids = sorted(corr)
    obj = np.array([corr[i][1] for i in ids], float)     # CT truth (evaluation)
    uv = np.array([corr[i][0] for i in ids], float)
    mz = np.array([R._disk(D, u, v) for u, v in uv])
    mk = R.backproject(uv[:, 0], uv[:, 1], mz)           # our markers, camera mm

    # registration-free scale: pairwise distance ratios
    iu, ju = np.triu_indices(len(ids), k=1)
    d_rec = np.linalg.norm(mk[iu] - mk[ju], axis=1)
    d_ct = np.linalg.norm(obj[iu] - obj[ju], axis=1)
    ratio = d_rec / d_ct

    Rk, tk = kabsch(mk, obj)                             # rigid ONLY
    fre = np.linalg.norm((Rk @ mk.T).T + tk - obj, axis=1)
    return dict(mk=mk, obj=obj, Rk=Rk, tk=tk, ids=ids,
                ratio=ratio, d_rec=d_rec, d_ct=d_ct,
                fre_rms=float(np.sqrt((fre ** 2).mean())), fre=fre,
                scale=umeyama_scale(mk, obj))


def run(shot, mu_fixed=None, multi_start=True, make_plots=True):
    print(f"{shot}:")
    D, Dsm, ref, keep, scan, hist = reconstruct(shot, mu_fixed=mu_fixed,
                                                multi_start=multi_start)
    if keep.sum() < 2000:
        print(f"  UNSTABLE ({keep.sum()} px survive) -- skipping mesh")
        return None
    ev = evaluate(shot, D, keep)

    # meshes: camera frame (the deliverable) + CT frame via rigid Kabsch (Slicer)
    idx = -np.ones(keep.shape, int)
    vv, uu = np.where(keep)
    idx[vv, uu] = np.arange(len(vv))
    cam = R.backproject(uu.astype(float), vv.astype(float), Dsm[vv, uu])
    col = ref[vv, uu]
    faces = grid_faces(idx, Dsm)
    write_ply_mesh(os.path.join(OUT, f"{shot}_mesh_ctfree_cam.ply"), cam, col, faces)
    ct = (ev["Rk"] @ cam.T).T + ev["tk"]
    write_ply_mesh(os.path.join(OUT, f"{shot}_mesh_ctfree_ct.ply"), ct, col, faces)

    b, s, mu, e = hist[-1]
    relief = np.percentile(D[keep], 90) - np.percentile(D[keep], 10)
    print(f"  recovered b = {b:.2f} mm, s = {s:.3f}, mu = {mu:.2f}, "
          f"residual {e:.4f}, relief {relief:.1f} mm")
    print(f"  pairwise-distance scale: {np.median(ev['ratio']):.3f} "
          f"(each: {np.array2string(ev['ratio'], precision=3)})")
    print(f"  similarity scale (Umeyama): {ev['scale']:.3f}   "
          f"[1.0 = perfect metric]")
    print(f"  3-D FRE, RIGID-only align: {ev['fre_rms']:.2f} mm "
          f"(each: {np.array2string(ev['fre'], precision=2)})")
    print(f"  -> {shot}_mesh_ctfree_cam.ply / _ctfree_ct.ply "
          f"({len(cam)}v/{len(faces)}f)")

    if make_plots and scan is not None:
        bs, mus, Es = scan
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for j, m_ in enumerate(mus):
            axs[0].plot(bs, Es[:, j], "-", lw=1.2,
                        label=f"$\\mu$ = {m_:.0f}")
        axs[0].axvline(b, color="#b23b3b", ls="--", lw=1,
                       label=f"final b = {b:.1f} mm")
        axs[0].set_xlabel("working distance b (mm), flat plane")
        axs[0].set_ylabel("photometric residual")
        axs[0].set_title(f"{shot}: flat-plane residual landscape")
        axs[0].legend(fontsize=7)
        it = np.arange(1, len(hist) + 1)
        axs[1].plot(it, [h[0] for h in hist], "-o", label="b (mm)")
        ax2 = axs[1].twinx()
        ax2.plot(it, [h[1] for h in hist], "-s", color="#b28b3b", label="s")
        ax2.plot(it, [h[2] for h in hist], "-^", color="#3b8b5a", label="mu")
        axs[1].set_xlabel("iteration"); axs[1].set_ylabel("b (mm)")
        ax2.set_ylabel("relief gain s / emission mu")
        ax2.legend(fontsize=7, loc="center right")
        axs[1].set_title("convergence")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{shot}_ctfree_landscape.png"), dpi=120)
        plt.close(fig)
    return ev


if __name__ == "__main__":
    # lightweight flags: --mu X --aim-dx Y --trim F  (rig-level model knobs)
    argv = sys.argv[1:]
    for flag, setter in (("--mu", lambda v: globals().__setitem__("_MU_CLI", v)),
                         ("--aim-dx", lambda v: globals().__setitem__("AIM_DX", v)),
                         ("--trim", lambda v: globals().__setitem__("TRIM", v))):
        if flag in argv:
            i = argv.index(flag)
            setter(float(argv[i + 1]))
            del argv[i:i + 2]
    if "--undistort-grid" in argv:
        UNDISTORT_GRID = True
        argv.remove("--undistort-grid")
    if "--wiring" in argv:                   # e.g. --wiring 270,90,180,0
        i = argv.index("--wiring")
        az = [float(x) for x in argv[i + 1].split(",")]
        P = {s: np.array([A_MM * math.cos(math.radians(az[s - 1])),
                          A_MM * math.sin(math.radians(az[s - 1])), 0.0])
             for s in (1, 2, 3, 4)}
        _GRIDS.clear()
        print(f"wiring override: " +
              " ".join(f"{s}:{az[s-1]:.0f}" for s in (1, 2, 3, 4)))
        del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    print("CT-FREE near-field photometric stereo (CT used for evaluation only):")
    print(f"model: AIM_DX={AIM_DX}  TRIM={TRIM}")
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "calibrate":
        calibrate_mu(["shot_004", "shot_005", "shot_006", "shot_007"])
        raise SystemExit(0)
    # rig-level mu: CLI --mu wins, else ctfree_mu.json, else free per shot
    mu_rig = globals().get("_MU_CLI")
    if mu_rig is None and os.path.exists(MU_FILE):
        mu_rig = json.load(open(MU_FILE))["mu"]
    if mu_rig is not None:
        print(f"rig-level mu = {mu_rig} ({os.path.basename(MU_FILE)})")
    if mode == "all":
        mu_star, shots = mu_rig, list(C.CT_CORR)
    elif mode.startswith("shot_"):
        mu_star, shots = mu_rig, sys.argv[1:]
    else:
        mu_star, shots = mu_rig, ["shot_004"]             # diagnostic single shot
    rows = []
    for sh in shots:
        ev = run(sh, mu_fixed=mu_star)
        if ev:
            rows.append((sh, np.median(ev["ratio"]), ev["scale"], ev["fre_rms"]))
    if len(rows) > 1:
        print(f"\nsummary (all CT-free, shared mu = {mu_star}; "
              f"rigid-only alignment):")
        print(f"{'shot':10}{'dist-scale':>12}{'sim-scale':>11}{'FRE rms':>9}")
        for sh, r, sc, f in rows:
            print(f"{sh:10}{r:12.3f}{sc:11.3f}{f:9.2f}")
