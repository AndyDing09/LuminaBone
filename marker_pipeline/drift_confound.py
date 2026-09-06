"""
Is the error really growing "away from the fiducials", or is that a confound?

dense_ct_accuracy.py reported median error rising 2.1 -> 7.0 mm with distance
from the nearest fiducial, and drift_diagnose.py then showed the error is
68-92% HIGH-FREQUENCY residual, not a smooth bowl. That combination is
suspicious: the beads sit in the middle of each bone patch, so
"far from a fiducial" is also "near the edge of the reconstruction", where
grazing view, the FFT border crop, and mask errors all bite.

The fix differs completely depending on which it is:
  * genuinely distance-from-fiducial -> the reconstruction drifts and needs a
    better anchor / integrator.
  * really distance-from-edge        -> the interior is fine and the edges are
    junk; trim them and the accuracy claim improves for free.

This regresses |error| on BOTH predictors together (and on distance from the
image centre) so their unique contributions can be separated.

Run: python drift_confound.py
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
from ct_frame_recover import load_stl
from clean_mesh_edges import read_ply, tri_max_edge

OUT = bd.project_path("depth_outputs", "ct_registered")


def boundary_distance(V, F):
    """distance from every vertex to the nearest mesh BOUNDARY vertex."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    uniq, inv, cnt = np.unique(E, axis=0, return_inverse=True,
                              return_counts=True)
    bedge = uniq[cnt == 1]
    bverts = np.unique(bedge)
    if len(bverts) == 0:
        return np.full(len(V), np.inf)
    return cKDTree(V[bverts, :3]).query(V[:, :3])[0]


def main():
    surf_ras = load_stl(bd.project_path("registration", "ct_spine.stl"))
    z = np.load(os.path.join(HERE, "ct_frame_transform.npz"))
    R, t = z["R"], z["t"]
    surf = (R.T @ (surf_ras - t).T).T
    tree = cKDTree(surf)

    print("Unique contribution of each predictor to |surface error|\n")
    print(f"{'shot':10}{'r(fiducial)':>13}{'r(edge)':>10}"
          f"{'partial fid':>13}{'partial edge':>14}   winner")
    pooled = []
    for sh in ["shot_004", "shot_005", "shot_006"]:
        p = os.path.join(OUT, f"{sh}_mesh_ctfree_ct.ply")
        if not os.path.exists(p):
            continue
        V, F = read_ply(p)
        P = V[:, :3]
        d, _ = tree.query(P)
        mk = np.array([xyz for uv, xyz in C.CT_CORR[sh].values()])
        d_fid = cKDTree(mk).query(P)[0]
        d_edge = boundary_distance(V, F)
        keep = np.isfinite(d_edge) & (d < 25)
        e, a, b = d[keep], d_fid[keep], d_edge[keep]
        pooled.append((e, a, b))

        def z_(v):
            return (v - v.mean()) / max(v.std(), 1e-9)
        ez, az, bz = z_(e), z_(a), z_(b)
        r_fid = float((ez * az).mean())
        r_edge = float((ez * bz).mean())
        # partial correlations: unique contribution with the other held fixed
        r_ab = float((az * bz).mean())
        den = np.sqrt(max((1 - r_ab ** 2), 1e-12))
        p_fid = (r_fid - r_edge * r_ab) / den / np.sqrt(
            max(1 - r_edge ** 2, 1e-12))
        p_edge = (r_edge - r_fid * r_ab) / den / np.sqrt(
            max(1 - r_fid ** 2, 1e-12))
        win = "EDGE" if abs(p_edge) > abs(p_fid) else "fiducial"
        print(f"{sh:10}{r_fid:13.3f}{r_edge:10.3f}"
              f"{p_fid:13.3f}{p_edge:14.3f}   {win}")

    e = np.concatenate([x[0] for x in pooled])
    a = np.concatenate([x[1] for x in pooled])
    b = np.concatenate([x[2] for x in pooled])
    print(f"\nhow the two predictors are themselves related: "
          f"r(fiducial-dist, edge-dist) = "
          f"{np.corrcoef(a, b)[0,1]:+.3f}")
    print("\nmedian |error| by BOTH, pooled (mm):")
    print(f"{'':22}" + "".join(f"{f'edge>{t}mm':>12}" for t in (0, 3, 6, 10)))
    for flo, fhi, lab in [(0, 10, "fiducial <10mm"),
                          (10, 20, "fiducial 10-20mm"),
                          (20, 999, "fiducial >20mm")]:
        row = f"  {lab:20}"
        for t in (0, 3, 6, 10):
            m = (a >= flo) & (a < fhi) & (b > t)
            row += f"{np.median(e[m]):12.2f}" if m.sum() > 100 else f"{'-':>12}"
        print(row)
    print("\nRead across a row: if error falls as you demand more edge "
          "clearance,\nthe edges are the problem. Read down a column: if it "
          "rises with\nfiducial distance at FIXED edge clearance, the drift is "
          "real.")


if __name__ == "__main__":
    main()
