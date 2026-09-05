"""
Recover the rigid transform between the CT_CORR marker frame and ct_spine.stl.

The marker coordinates in calibrate_photometric_ct.CT_CORR were hand-read in
Slicer into a recentred frame; ct_spine.stl is in raw Slicer RAS. No transform
between them was ever saved, so a dense surface-accuracy comparison is
impossible until it is recovered.

The 19 beads are glued to the bone, so the true transform is the one that lands
every marker on (or just above) the CT bone surface. This searches for it by
multi-restart point-set-to-surface ICP.

CRITICAL VALIDATION: registration/ACCURACY_REPORT.md documents that ICP of a
small patch onto this CT "drapes" -- it finds A low-residual pose, not the
CORRECT one, and reports a deceptively good ~1.9 mm. A 19-point set spanning
141 mm over five levels is far better constrained than a flat patch, but the
same failure mode must be excluded, so this script reports:
  * the spread of independent restarts that reach a low residual, and
  * whether they agree on ONE pose (trustworthy) or many (a drape).
Only a tight, repeatable optimum justifies using the result.

Run: python ct_frame_recover.py
"""

import os
import sys
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
import calibrate_photometric_ct as C

STL = bd.project_path("registration", "ct_spine.stl")
N_RESTART = 240
MAX_ITER = 60


def load_stl(path, stride=3):
    raw = np.fromfile(path, dtype=np.uint8)
    ntri = int(np.frombuffer(raw[80:84].tobytes(), dtype="<u4")[0])
    rec = raw[84:84 + ntri * 50].reshape(ntri, 50)
    f = np.frombuffer(np.ascontiguousarray(rec[:, :48]).tobytes(),
                      dtype="<f4").reshape(ntri, 12)
    V = f[:, 3:12].reshape(-1, 3).astype(np.float64)
    return np.unique(V[::stride], axis=0)


def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cQ - R @ cP


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def icp(markers, tree, surf, R0, t0):
    R, t = R0, t0
    prev = np.inf
    for _ in range(MAX_ITER):
        P = (R @ markers.T).T + t
        d, idx = tree.query(P)
        R, t = kabsch(markers, surf[idx])
        rms = float(np.sqrt((d ** 2).mean()))
        if abs(prev - rms) < 1e-6:
            break
        prev = rms
    P = (R @ markers.T).T + t
    d, _ = tree.query(P)
    return R, t, float(np.sqrt((d ** 2).mean())), d


def main():
    print("recovering CT_CORR -> ct_spine.stl transform ...")
    surf = load_stl(STL)
    tree = cKDTree(surf)
    markers = np.array([xyz for sh in C.CT_CORR
                        for uv, xyz in C.CT_CORR[sh].values()])
    print(f"  CT surface: {len(surf):,} points   markers: {len(markers)}")

    rng = np.random.default_rng(0)
    cS, cM = surf.mean(0), markers.mean(0)
    sols = []
    for i in range(N_RESTART):
        R0 = np.eye(3) if i == 0 else rand_rot(rng)
        # centroid init, jittered along the spine so restarts probe all levels
        jitter = np.zeros(3) if i < 2 else rng.normal(scale=25.0, size=3)
        t0 = cS - R0 @ cM + jitter
        R, t, rms, d = icp(markers, tree, surf, R0, t0)
        sols.append((rms, R, t, d))
    sols.sort(key=lambda s: s[0])

    best_rms = sols[0][0]
    print(f"\n  best residual: {best_rms:.2f} mm")
    print(f"  {'rank':6}{'residual (mm)':>15}{'centroid of markers in RAS':>34}")
    for r, (rms, R, t, d) in enumerate(sols[:8], 1):
        c = (R @ markers.T).T.mean(0) + t
        print(f"  {r:<6}{rms:15.3f}   "
              f"({c[0]:8.2f},{c[1]:8.2f},{c[2]:9.2f})")

    # how many DISTINCT poses reach within 20% of the best residual?
    good = [s for s in sols if s[0] < best_rms * 1.2]
    cents = np.array([((s[1] @ markers.T).T.mean(0) + s[2]) for s in good])
    spread = float(np.linalg.norm(cents - cents.mean(0), axis=1).max()) \
        if len(cents) > 1 else 0.0
    print(f"\n  {len(good)}/{N_RESTART} restarts land within 20% of the best "
          f"residual")
    print(f"  their marker-centroid spread: {spread:.1f} mm")
    if spread < 3.0 and best_rms < 3.0:
        verdict = ("UNIQUE optimum -> transform is trustworthy, dense "
                   "comparison is valid")
    elif spread < 3.0:
        verdict = (f"consistent pose but residual {best_rms:.1f} mm is high -- "
                   "markers may not sit on the segmented surface")
    else:
        verdict = ("MANY distinct poses fit equally well -> this is the drape "
                   "failure of ACCURACY_REPORT.md; do NOT use for accuracy")
    print(f"  VERDICT: {verdict}")

    rms, R, t, d = sols[0]
    print(f"\n  per-marker distance to CT surface at the best pose (mm):")
    print("   " + np.array2string(d, precision=2, max_line_width=100))
    np.savez(os.path.join(HERE, "ct_frame_transform.npz"), R=R, t=t,
             rms=rms, spread=spread, per_marker=d)
    print(f"\n  saved -> ct_frame_transform.npz")


if __name__ == "__main__":
    main()
