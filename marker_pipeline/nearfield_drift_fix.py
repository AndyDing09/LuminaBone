"""
Fix the drift: pin the LOW frequencies with per-pixel attenuation depth.

THE PROBLEM (measured, dense_ct_accuracy.py + drift_confound.py):
surface error grows 2.1 -> 4.4 -> 5.3 mm with distance from the fiducials.
Ruled out as an edge artifact (partial r for fiducial distance 0.34-0.51 vs
0.04-0.06 for edge distance, and the two predictors are independent).

THE CAUSE:
the current pipeline gets its absolute depth from just TWO global numbers.
Photometric stereo yields SLOPES; Frankot-Chellappa integrates them into a
relative surface; then a single (b, s) pair scales that whole surface. So the
entire 30 mm patch is pinned by 2 degrees of freedom. Any integration drift or
scale error is unconstrained locally and grows with distance from wherever the
evaluation happens to be anchored.

THE FIX:
the near-field model already contains ABSOLUTE depth at EVERY pixel, in the
attenuation term -- we were only using it to choose (b, s). Given the albedo
and normal from the photometric solve, each pixel's brightness independently
determines its distance:

    I_s = rho (n . l_s) * cos^mu(theta_s) / r_s^2 ,   r_s = ||P_s - X(D)||

Solving that per pixel (over all four LEDs jointly) gives z_att: absolute,
drift-free, but noisy and albedo-sensitive. The photometric surface z_ps is the
opposite: clean and detailed, but only relative. So fuse them by frequency --

    z_fused = LP_sigma(z_att) + [ z_ps - LP_sigma(z_ps) ]

low frequencies (absolute position, no drift) from attenuation; high
frequencies (shape detail, albedo-free) from photometric stereo. This is the
frequency split prototyped in fusion_shot011.py, carried into the CT-free
pipeline. Nothing here uses CT.

Run: python nearfield_drift_fix.py [shot_004 ...] [--sigma 40]
"""

import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import nearfield_ctfree as CF
import register_inverse_square_ct as R
from nearfield_ct import load_lums, grid_faces, write_ply_mesh

MU = 8.0
D_LO, D_HI = 15.0, 120.0


def attenuation_depth(M, g, mu, xn, yn, iters=44):
    """per-pixel absolute depth from the 1/r^2 + cos^mu attenuation.

    g = rho*n is fixed from the photometric solve, so the only unknown per
    pixel is D. Minimising sum_s (I_s - g.l_s(D) att_s(D))^2 over D is done by
    bisection on the total predicted brightness, which decreases with D."""
    lo = np.full(M.shape[:2], D_LO)
    hi = np.full(M.shape[:2], D_HI)
    Iobs = M.sum(-1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Lhat, att = CF.geometry(mid, CF.FX, CF.FY, CF.CX, CF.CY, mu=mu)
        pred = (np.einsum("hws,hwks->hwk", g, Lhat) * att).sum(-1)
        too_bright = pred > Iobs                 # predicted brighter => farther
        lo = np.where(too_bright, mid, lo)
        hi = np.where(too_bright, hi, mid)
    return 0.5 * (lo + hi)


def fuse(shot, sigma=40.0, verbose=True):
    """CT-free reconstruction with the low frequencies pinned by attenuation."""
    lums, ref = load_lums(shot)
    I = np.stack(lums, 0)
    M = np.moveaxis(I, 0, -1)
    base_mask = bd.bone_mask(I.mean(0))

    # 1. the existing CT-free solve -> relative-but-clean surface + its (b, s)
    D_ps, Dsm, ref2, keep, scan, hist = CF.reconstruct(shot, mu_fixed=MU,
                                                       verbose=False)
    b, s, mu, resid = hist[-1]
    if verbose:
        print(f"  photometric solve: b = {b:.2f} mm, s = {s:.3f}")

    # 2. albedo+normal at that surface, then per-pixel attenuation depth
    Lhat, att = CF.geometry(D_ps, CF.FX, CF.FY, CF.CX, CF.CY, mu=mu)
    g = CF.solve_g(M, Lhat, att)
    xn, yn = CF.norm_grid(CF.FX, CF.FY, CF.CX, CF.CY, *M.shape[:2])
    D_att = attenuation_depth(M, g, mu, xn, yn)
    if verbose:
        v = D_att[base_mask]
        print(f"  attenuation depth: median {np.median(v):.2f} mm, "
              f"IQR {np.percentile(v,75)-np.percentile(v,25):.2f} mm")

    # 3. frequency fusion, computed only where the mask is valid
    w = base_mask.astype(float)

    def masked_lp(x):
        num = gaussian_filter(np.where(base_mask, x, 0.0), sigma)
        den = np.maximum(gaussian_filter(w, sigma), 1e-6)
        return num / den

    D_fused = masked_lp(D_att) + (D_ps - masked_lp(D_ps))
    return D_ps, D_att, D_fused, keep, ref, b, s


def evaluate(shot, D, label):
    ev = CF.evaluate(shot, D, None)
    print(f"    {label:24} FRE {ev['fre_rms']:5.2f} mm   "
          f"scale {ev['scale']:5.3f}   dist-scale {np.median(ev['ratio']):5.3f}")
    return ev


def main():
    args = sys.argv[1:]
    sigma = 40.0
    if "--sigma" in args:
        i = args.index("--sigma"); sigma = float(args[i + 1]); del args[i:i + 2]
    shots = [a for a in args if a.startswith("shot_")] or \
            ["shot_004", "shot_005", "shot_006"]
    print(f"Drift fix by frequency fusion (sigma = {sigma:.0f} px), CT-free\n")
    OUT = bd.project_path("depth_outputs", "ct_registered")
    for sh in shots:
        print(f"{sh}:")
        D_ps, D_att, D_fused, keep, ref, b, s = fuse(sh, sigma)
        evaluate(sh, D_ps, "photometric only (now)")
        evaluate(sh, D_att, "attenuation only")
        evaluate(sh, D_fused, "FUSED")
        # write the fused mesh so dense_ct_accuracy.py can score it
        idx = -np.ones(keep.shape, int)
        vv, uu = np.where(keep)
        idx[vv, uu] = np.arange(len(vv))
        cam = R.backproject(uu.astype(float), vv.astype(float), D_fused[vv, uu])
        ev = CF.evaluate(sh, D_fused, keep)
        ct = (ev["Rk"] @ cam.T).T + ev["tk"]
        faces = grid_faces(idx, D_fused)
        p = os.path.join(OUT, f"{sh}_mesh_ctfree_fused_ct.ply")
        write_ply_mesh(p, ct, ref[vv, uu], faces)
        print(f"    -> {os.path.basename(p)}  ({len(ct)}v/{len(faces)}f)")
        print()


if __name__ == "__main__":
    main()
