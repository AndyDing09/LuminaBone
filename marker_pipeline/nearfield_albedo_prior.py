"""
Break the depth/albedo degeneracy with a LOW-FREQUENCY ALBEDO PRIOR.

Established by nearfield_drift_fix.py and the degeneracy demo: with albedo free
at every pixel, per-pixel depth is unidentifiable -- a 25 mm depth error is
reproduced to within 5.9% by simply rescaling albedo. That is why the pipeline
pins a whole 30 mm patch with only two numbers (b, s), and why the error grows
2.1 -> 5.3 mm with distance from the fiducials: nothing constrains the surface
locally, so integration drift is free to accumulate.

The degeneracy is exact only because albedo is unconstrained. It breaks under
one physically motivated assumption:

    BONE ALBEDO HAS LITTLE LOW-FREQUENCY VARIATION.
    Cortical bone is roughly uniform in reflectance over centimetre scales;
    its texture (trabecular pattern, staining, blood) is high-frequency.

Under that assumption, slow spatial variation in the recovered albedo is not
real albedo -- it is mis-estimated depth. Since I ~ rho/r^2 at fixed shading,
rho ~ D^2, so a low-frequency log-albedo excess delta implies the depth is too
large by exp(delta/2):

    D_corrected = D * exp(-LP_sigma(log rho) - mean) / 2)

This is a genuine assumption, not a derivation, so it is judged empirically:
does it reduce the CT surface error and flatten the error-vs-fiducial-distance
curve? If not, bone albedo is not low-frequency-flat and the idea is wrong.
Nothing here uses CT; CT only scores the result afterwards.

Run: python nearfield_albedo_prior.py [shot_004 ...] [--sigma 60] [--iters 3]
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


def masked_lp(x, mask, sigma):
    w = mask.astype(float)
    num = gaussian_filter(np.where(mask, x, 0.0), sigma)
    den = np.maximum(gaussian_filter(w, sigma), 1e-6)
    return num / den


def apply_prior(D, M, mask, sigma, iters, mu=MU, verbose=True):
    """iteratively flatten the low-frequency albedo by correcting depth."""
    D = D.copy()
    for it in range(iters):
        Lhat, att = CF.geometry(D, CF.FX, CF.FY, CF.CX, CF.CY, mu=mu)
        g = CF.solve_g(M, Lhat, att)
        alb = np.linalg.norm(g, axis=-1)
        la = np.log(np.maximum(alb, 1e-9))
        lp = masked_lp(la, mask, sigma)
        delta = lp - np.median(lp[mask])
        # rho ~ D^2  =>  correct depth by exp(-delta/2); damp for stability
        D = np.clip(D * np.exp(-0.5 * delta * 0.7), 12.0, 130.0)
        if verbose:
            print(f"    iter {it+1}: low-freq log-albedo spread "
                  f"{np.percentile(delta[mask],95)-np.percentile(delta[mask],5):.3f}"
                  f"   median depth {np.median(D[mask]):.2f} mm")
    return D


def main():
    args = sys.argv[1:]
    sigma, iters = 60.0, 3
    if "--sigma" in args:
        i = args.index("--sigma"); sigma = float(args[i + 1]); del args[i:i + 2]
    if "--iters" in args:
        i = args.index("--iters"); iters = int(args[i + 1]); del args[i:i + 2]
    shots = [a for a in args if a.startswith("shot_")] or \
            ["shot_004", "shot_005", "shot_006"]
    OUT = bd.project_path("depth_outputs", "ct_registered")
    print(f"Low-frequency albedo prior (sigma={sigma:.0f} px, {iters} iters), "
          f"CT-free\n")
    for sh in shots:
        print(f"{sh}:")
        lums, ref = load_lums(sh)
        M = np.moveaxis(np.stack(lums, 0), 0, -1)
        mask = bd.bone_mask(np.stack(lums, 0).mean(0))
        D0, _, _, keep, _, hist = CF.reconstruct(sh, mu_fixed=MU, verbose=False)
        b, s, mu, e = hist[-1]
        print(f"    baseline: b = {b:.2f} mm, s = {s:.3f}")
        ev0 = CF.evaluate(sh, D0, keep)
        print(f"    {'baseline':22} FRE {ev0['fre_rms']:5.2f} mm   "
              f"scale {ev0['scale']:5.3f}")
        D1 = apply_prior(D0, M, mask, sigma, iters)
        ev1 = CF.evaluate(sh, D1, keep)
        print(f"    {'+ albedo prior':22} FRE {ev1['fre_rms']:5.2f} mm   "
              f"scale {ev1['scale']:5.3f}")
        idx = -np.ones(keep.shape, int)
        vv, uu = np.where(keep)
        idx[vv, uu] = np.arange(len(vv))
        cam = R.backproject(uu.astype(float), vv.astype(float), D1[vv, uu])
        ct = (ev1["Rk"] @ cam.T).T + ev1["tk"]
        faces = grid_faces(idx, D1)
        p = os.path.join(OUT, f"{sh}_mesh_ctfree_albprior_ct.ply")
        write_ply_mesh(p, ct, ref[vv, uu], faces)
        print(f"    -> {os.path.basename(p)}  ({len(ct)}v/{len(faces)}f)\n")


if __name__ == "__main__":
    main()
