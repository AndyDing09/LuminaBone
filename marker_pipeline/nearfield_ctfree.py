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
B_RANGE = (20.0, 110.0)                  # plausible working distances (rig design)
S_RANGE = (0.10, 5.0)                    # relief gain bounds (s~1 expected)
MU_RANGE = (0.0, 14.0)                   # LED emission exponent cos^mu bounds
                                         # (mu ~ 11 for a 20-deg half-angle LED)


# ---------------------------------------------------------------------------
# forward model pieces (parameterised intrinsics so they work downsampled)
# ---------------------------------------------------------------------------

def geometry(D, fx, fy, cx, cy, mu=0.0):
    """per-pixel unit light vectors (H,W,4,3) + attenuation (H,W,4).

    attenuation = cos^mu(theta_s) / r^2 : inverse-square times a Lambertian-
    emitter anisotropy with exponent mu (LEDs aimed along the optical axis, so
    cos(theta_s) = D / r_s). mu = 0 is the isotropic point source.
    """
    H, W = D.shape
    U, V = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    Xw = np.stack([(U - cx) / fx * D, -(V - cy) / fy * D, -D], axis=-1)
    Lhat = np.empty((H, W, 4, 3)); att = np.empty((H, W, 4))
    for k, s in enumerate((1, 2, 3, 4)):
        w = P[s][None, None, :] - Xw
        d2 = np.sum(w * w, axis=-1)
        att[..., k] = 1.0 / d2
        if mu:
            att[..., k] *= (D / np.sqrt(d2)) ** mu
        Lhat[..., k, :] = w / np.sqrt(d2)[..., None]
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
    U, V = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    Xw = np.stack([(U - cx) / fx * D, -(V - cy) / fy * D, -D], axis=-1)
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

    def set_relief(self, ztilde):
        zt = _shrink(ztilde.astype(np.float32), DS).astype(np.float64)
        zt -= np.median(zt[self.mask])
        # clip runaway border ramps so they cannot drive the search
        lo, hi = np.percentile(zt[self.mask], [1, 99])
        self.zt = np.clip(zt, lo, hi)

    def depth(self, b, s):
        return b + s * (b / self.fx) * self.zt

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
        Lhat, att = geometry(D, self.fx, self.fy, self.cx, self.cy, mu=mu)
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
        return float(r[self.mask].mean())


def search_bs(E, b0=None, mu0=None):
    """coarse grid + Nelder-Mead refine of (b, s, mu). CT-free."""
    bs = np.arange(B_RANGE[0], B_RANGE[1] + 1e-9, 5.0) if b0 is None \
        else np.arange(max(B_RANGE[0], b0 - 20), min(B_RANGE[1], b0 + 20) + 1e-9, 5.0)
    ss = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
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


def reconstruct(shot, iters=6, verbose=True):
    """CT-free near-field solve. Returns full-res D (mm), plus diagnostics.
    NOTHING derived from CT enters this function."""
    lums, ref = load_lums(shot)
    I = np.stack(lums, axis=0)
    H, W = I.shape[1:]
    M = np.moveaxis(I, 0, -1)
    base_mask = bd.bone_mask(I.mean(0))

    E = Objective(lums, base_mask)
    bs, mus, Es = flat_scan(E)                            # seed: photometric WD scan
    ib, imu = np.unravel_index(np.argmin(Es), Es.shape)
    b, mu = float(bs[ib]), float(mus[imu])
    s = 1.0
    if verbose:
        print(f"  flat-plane scan: b0 = {b:.1f} mm, mu0 = {mu:.0f} "
              f"(residual {Es.min():.4f}, worst {Es.max():.4f})")

    D = np.full((H, W), b)
    hist, best = [], None
    for it in range(iters):
        Lhat, att = geometry(D, FX, FY, CX, CY, mu=mu)
        g = solve_g(M, Lhat, att)
        alb = np.linalg.norm(g, axis=-1)
        n = g / np.where(alb > 1e-9, alb, 1.0)[..., None]
        n[n[..., 2] < 0] *= -1.0
        ztilde = bd.normals_to_depth(n)
        ztilde = ztilde - np.median(ztilde[base_mask])
        E.set_relief(ztilde)
        b, s, mu, e = search_bs(E, b0=b, mu0=(mu if it else None))
        D = b + s * (b / FX) * ztilde
        hist.append((b, s, mu, e))
        # the fixed-point map is not monotone (ztilde moves under the search),
        # so keep the photometrically best iterate -- still CT-free selection
        if best is None or e < best[0]:
            best = (e, b, s, mu, ztilde.copy())
        if verbose:
            print(f"  iter {it+1}: b = {b:6.2f} mm  s = {s:5.3f}  mu = {mu:4.2f}"
                  f"  residual {e:.4f}")

    e, b, s, mu, ztilde = best
    D = b + s * (b / FX) * ztilde
    hist.append((b, s, mu, e))
    if verbose:
        print(f"  best iterate kept: b = {b:.2f} mm  s = {s:.3f}  mu = {mu:.2f}"
              f"  residual {e:.4f}")

    # final full-res solve for the confidence gate
    Lhat, att = geometry(D, FX, FY, CX, CY, mu=mu)
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
    return D, Dsm, ref, keep, (bs, mus, Es), hist


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


def run(shot, make_plots=True):
    print(f"{shot}:")
    D, Dsm, ref, keep, scan, hist = reconstruct(shot)
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

    if make_plots:
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
    shots = list(C.CT_CORR) if (len(sys.argv) > 1 and sys.argv[1] == "all") \
        else ["shot_004"]
    print("CT-FREE near-field photometric stereo (CT used for evaluation only):")
    rows = []
    for sh in shots:
        ev = run(sh)
        if ev:
            rows.append((sh, np.median(ev["ratio"]), ev["scale"], ev["fre_rms"]))
    if len(rows) > 1:
        print("\nsummary (all CT-free; rigid-only alignment):")
        print(f"{'shot':10}{'dist-scale':>12}{'sim-scale':>11}{'FRE rms':>9}")
        for sh, r, sc, f in rows:
            print(f"{sh:10}{r:12.3f}{sc:11.3f}{f:9.2f}")
