"""
FAR-FIELD ablation of the CT-free pipeline.

Far-field photometric stereo (I = rho (n.L), constant light directions) has NO
depth term: scaling the whole scene leaves every predicted image unchanged, so
far-field alone cannot produce a metric reconstruction -- there is nothing to
recover. Every far-field number in this project therefore came from a CT-
anchored affine fit.

To compare light models honestly WITHOUT ground truth, this script keeps the
CT-free metric machinery of nearfield_ctfree.py (the tied-normal photometric
residual searched over working distance b and relief gain s) and swaps ONLY the
source of the relief map:

    near-field : normals from the per-pixel point-source solve, re-estimated
                 each iteration as the depth improves
    far-field  : normals from Woodham's constant-direction solve, computed ONCE
                 (they cannot change with depth -- the model has no depth)

Everything downstream is identical: same objective, same search, same trim,
same rigid-only 6-DOF evaluation. The difference in FRE is attributable to the
light model alone.

Run: python farfield_ctfree.py                 # shots 4 and 5
     python farfield_ctfree.py shot_006 ...    # any subset
"""

import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
import nearfield_ctfree as CF
from nearfield_ct import load_lums, AZ, A_MM

ELEV = math.degrees(math.atan2(40.0, A_MM))      # far-field elevation, as used
                                                 # throughout the far-field code
MU = 8.0                                         # rig-level emission exponent


def farfield_relief(lums, base_mask):
    """Woodham far-field normals -> relative depth. Depth-independent, so this
    is computed once rather than iterated."""
    Ls = np.array([bd.light_vector(AZ[i], ELEV) for i in (1, 2, 3, 4)])
    n, _ = bd.photometric_stereo(lums, Ls)
    z = bd.normals_to_depth(n)
    return z - np.median(z[base_mask])


def run(shot, verbose=True):
    lums, ref = load_lums(shot)
    I = np.stack(lums, axis=0)
    base_mask = bd.bone_mask(I.mean(0))
    E = CF.Objective(lums, base_mask)

    ztilde = farfield_relief(lums, base_mask)
    E.set_relief(ztilde, CF.FX)

    # identical CT-free metric recovery, multi-start over the focus band
    best = None
    for b0 in CF.B_STARTS:
        b, s, mu, e = CF.search_bs(E, b0=b0, mu_fixed=MU)
        if best is None or e < best[3]:
            best = (b, s, mu, e)
    b, s, mu, e = best
    D = np.maximum(b + s * (b / CF.FX) * ztilde, 2.0)

    ev = CF.evaluate(shot, D, base_mask)
    if verbose:
        print(f"{shot}:")
        print(f"  far-field relief + CT-free metric recovery: "
              f"b = {b:.2f} mm, s = {s:.3f}, residual {e:.4f}")
        print(f"  pairwise-distance scale: {np.median(ev['ratio']):.3f}")
        print(f"  similarity scale (Umeyama): {ev['scale']:.3f}")
        print(f"  3-D FRE, RIGID-only align: {ev['fre_rms']:.2f} mm "
              f"(each: {np.array2string(ev['fre'], precision=2)})")
    return b, s, ev


if __name__ == "__main__":
    shots = [a for a in sys.argv[1:] if a.startswith("shot_")] or \
            ["shot_004", "shot_005"]
    print("FAR-FIELD relief + CT-free metric recovery "
          "(no ground truth in the reconstruction):")
    rows = []
    for sh in shots:
        b, s, ev = run(sh)
        rows.append((sh, b, s, float(np.median(ev["ratio"])), ev["scale"],
                     ev["fre_rms"]))
    print(f"\n{'shot':10}{'b (mm)':>9}{'s':>7}{'dist-scale':>12}"
          f"{'sim-scale':>11}{'FRE rms':>9}")
    for sh, b, s, r, sc, f in rows:
        print(f"{sh:10}{b:9.2f}{s:7.3f}{r:12.3f}{sc:11.3f}{f:9.2f}")
