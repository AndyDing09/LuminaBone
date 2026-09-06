"""
What SHAPE is the error that grows away from the fiducials?

dense_ct_accuracy.py showed the median surface error rising from 2.1 mm near a
fiducial to 7.0 mm beyond 25 mm. The fix depends entirely on what that error
looks like:

  * mostly CONSTANT      -> a depth offset; b is wrong.
  * mostly LINEAR (tilt) -> the recovered scale/relief gain s is wrong, or the
                            rigid alignment is levering.
  * mostly QUADRATIC     -> a bowl/saddle: classic Frankot-Chellappa
                            integration drift, the signature of integrating a
                            slope field over a large region with no anchor.
  * mostly RESIDUAL      -> high-frequency noise, not drift; needs denoising,
                            not regularisation.

This fits each model to the signed error field of the CT-free meshes and
reports the variance explained, per shot. Whatever dominates is what a fix has
to remove.

Run: python drift_diagnose.py
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
from clean_mesh_edges import read_ply

OUT = bd.project_path("depth_outputs", "ct_registered")


def signed_error(P, surf, tree):
    """signed distance from each vertex to the CT surface, along the local
    surface normal (estimated from the neighbourhood of the nearest point)."""
    d, idx = tree.query(P)
    # local normal from PCA of the 24 nearest surface points
    _, nbr = tree.query(surf[idx], k=24)
    Q = surf[nbr]                                    # (N,24,3)
    Qc = Q - Q.mean(axis=1, keepdims=True)
    # smallest-eigenvector via SVD on each neighbourhood
    n = np.empty_like(P)
    step = 20000
    for a in range(0, len(P), step):
        _, _, Vt = np.linalg.svd(Qc[a:a + step], full_matrices=False)
        n[a:a + step] = Vt[:, 2, :]
    v = P - surf[idx]
    s = np.einsum("ij,ij->i", v, n)
    return s, d


def fit_models(P, e):
    """variance of e explained by constant / linear / quadratic in (x,y,z)."""
    X = P - P.mean(0)
    X = X / np.maximum(np.abs(X).max(0), 1e-9)
    x, y, z = X.T
    designs = {
        "constant": np.ones((len(e), 1)),
        "linear": np.column_stack([np.ones_like(x), x, y, z]),
        "quadratic": np.column_stack([np.ones_like(x), x, y, z,
                                        x * x, y * y, z * z,
                                        x * y, x * z, y * z]),
    }
    var = e.var()
    out = {}
    for name, A in designs.items():
        coef, *_ = np.linalg.lstsq(A, e, rcond=None)
        resid = e - A @ coef
        out[name] = (1.0 - resid.var() / var) * 100.0
    return out


def main():
    surf_ras = load_stl(bd.project_path("registration", "ct_spine.stl"))
    z = np.load(os.path.join(HERE, "ct_frame_transform.npz"))
    R, t = z["R"], z["t"]
    surf = (R.T @ (surf_ras - t).T).T
    tree = cKDTree(surf)

    print("Shape of the CT-free reconstruction error (variance explained, %)\n")
    print(f"{'shot':10}{'signed err mean':>17}{'rms':>8}"
          f"{'constant':>10}{'+linear':>9}{'+quadratic':>12}{'residual':>10}")
    for sh in ["shot_004", "shot_005", "shot_006"]:
        p = os.path.join(OUT, f"{sh}_mesh_ctfree_ct.ply")
        if not os.path.exists(p):
            continue
        V, F = read_ply(p)
        P = V[:, :3]
        if len(P) > 60000:                       # subsample for the SVD pass
            P = P[np.linspace(0, len(P) - 1, 60000).astype(int)]
        e, dist = signed_error(P, surf, tree)
        m = np.abs(e) < 25.0                     # drop gross outliers
        e, P2 = e[m], P[m]
        r = fit_models(P2, e)
        print(f"{sh:10}{e.mean():17.2f}{np.sqrt((e**2).mean()):8.2f}"
              f"{r['constant']:10.1f}{r['linear']:9.1f}"
              f"{r['quadratic']:12.1f}{100-r['quadratic']:10.1f}")
    print("\nreading: 'constant' is what a depth offset explains; '+linear' adds"
          "\ntilt/scale; '+quadratic' adds bowl/saddle. Whatever the quadratic"
          "\nmodel captures is removable low-frequency drift; the residual is"
          "\nhigh-frequency error that regularisation cannot fix.")


if __name__ == "__main__":
    main()
