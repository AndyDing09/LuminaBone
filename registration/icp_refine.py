#!/usr/bin/env python3
"""
icp_refine.py  --  ICP refinement after a landmark initialization
==================================================================

Refines an already-coarsely-aligned source cloud onto a target surface
(the CT spine model) using point-to-plane ICP with trimmed correspondences.

Pipeline this belongs to:
    1) landmark_register.py  -> coarse similarity transform (scale + pose)
    2) icp_refine.py         -> tighten the fit against the full surface

Point-to-plane ICP converges faster and to a better minimum than point-to-point
for surface data (Chen & Medioni 1992). Trimming (keep the best `trim` fraction of
correspondences each iter) makes it robust to the endoscope patch covering only a
PART of the vertebra -- non-overlapping points are rejected as outliers.

IMPORTANT (honest limitation): ICP only *refines*. It cannot rescue a wrong
initialization. On a near-planar brightness-relief patch, the sliding/rotational
degrees of freedom in the tangent plane are weakly constrained, so ICP can drift
along the surface. Always sanity-check the reported final RMSE and the overlay,
and keep the landmark FRE as the primary trust metric.

Targets it reads: .stl (binary/ascii) or .ply (via landmark_register.read_ply).

Usage (standalone):
    python icp_refine.py --source aligned.ply --target ct_spine.stl \
        --outdir reg_out [--with-scale] [--trim 0.8] [--max-iter 60]
"""

import argparse, os, struct, sys
import numpy as np
from scipy.spatial import cKDTree

# reuse PLY IO from the landmark tool
import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("lr", os.path.join(_here, "landmark_register.py"))
lr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lr)


# --------------------------------------------------------------------------- #
# STL reader (binary + ascii) -> vertices, triangles
# --------------------------------------------------------------------------- #
def read_stl(path):
    with open(path, "rb") as f:
        head = f.read(5)
        f.seek(0)
        if head.lower() == b"solid":
            txt = f.read().decode("ascii", "ignore")
            if "facet normal" in txt:      # genuine ascii STL
                verts = []
                for line in txt.splitlines():
                    s = line.strip().split()
                    if len(s) == 4 and s[0] == "vertex":
                        verts.append([float(s[1]), float(s[2]), float(s[3])])
                V = np.asarray(verts, float)
                tris = np.arange(len(V)).reshape(-1, 3)
                return V, tris
        # binary STL
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        floats = data[:, :48].copy().view("<f4").reshape(n, 12)
        tri_v = floats[:, 3:12].reshape(n, 3, 3)
        V = tri_v.reshape(-1, 3)
        tris = np.arange(len(V)).reshape(-1, 3)
        return V.astype(float), tris


def load_surface(path):
    """Return (points[N,3], normals[N,3]) for STL or PLY targets."""
    if path.lower().endswith(".stl"):
        V, F = read_stl(path)
    else:
        V, _, F = lr.read_ply(path)
        if F is None:
            # point cloud target: estimate normals via local PCA
            return V, estimate_normals(V)
    # per-vertex normals area-weighted from faces
    N = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    ln[ln == 0] = 1
    return V, N / ln


def estimate_normals(P, k=12):
    tree = cKDTree(P)
    _, idx = tree.query(P, k=k)
    N = np.empty_like(P)
    for i in range(len(P)):
        nb = P[idx[i]] - P[idx[i]].mean(0)
        _, _, vt = np.linalg.svd(nb, full_matrices=False)
        N[i] = vt[-1]
    return N


# --------------------------------------------------------------------------- #
# Point-to-plane ICP (optionally with uniform scale), trimmed
# --------------------------------------------------------------------------- #
def icp_point_to_plane(src, tgt_pts, tgt_nrm, max_iter=60, trim=0.8,
                       with_scale=False, tol=1e-7):
    """
    Aligns src onto target surface. Returns 4x4 M (src->target), rmse, history.
    Linearized point-to-plane (small-angle) solve each iteration.
    """
    tree = cKDTree(tgt_pts)
    M = np.eye(4)
    P = src.copy()
    prev = None
    hist = []
    for it in range(max_iter):
        d, j = tree.query(P)
        # trim: keep closest `trim` fraction as inliers
        if 0 < trim < 1:
            thr = np.quantile(d, trim)
            keep = d <= thr
        else:
            keep = np.ones(len(P), bool)
        q = tgt_pts[j[keep]]           # matched target points
        nn = tgt_nrm[j[keep]]          # target normals
        p = P[keep]
        # residual along normal
        r = np.einsum("ij,ij->i", (q - p), nn)
        # Jacobian for [alpha,beta,gamma (rot), tx,ty,tz, (s)]
        c = np.cross(p, nn)
        if with_scale:
            sp = np.einsum("ij,ij->i", p, nn)          # d(s*p)·n
            A = np.concatenate([c, nn, sp[:, None]], axis=1)
        else:
            A = np.concatenate([c, nn], axis=1)
        # solve A x = r  (least squares)
        x, *_ = np.linalg.lstsq(A, r, rcond=None)
        a, b, g = x[0], x[1], x[2]
        t = x[3:6]
        R = np.array([[1, -g, b], [g, 1, -a], [-b, a, 1]], float)
        U, _, Vt = np.linalg.svd(R); R = U @ Vt          # re-orthonormalize
        s = (1.0 + x[6]) if with_scale else 1.0
        step = np.eye(4); step[:3, :3] = s * R; step[:3, 3] = t
        P = (step[:3, :3] @ P.T).T + step[:3, 3]
        M = step @ M
        rmse = float(np.sqrt((r ** 2).mean()))
        hist.append(rmse)
        if prev is not None and abs(prev - rmse) < tol:
            break
        prev = rmse
    # final rmse on inliers
    d, _ = tree.query(P);
    final_rmse = float(np.sqrt((np.sort(d)[:int(len(d)*trim)] ** 2).mean()))
    return M, final_rmse, hist


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="already landmark-aligned PLY")
    ap.add_argument("--target", required=True, help="CT model .stl or .ply")
    ap.add_argument("--outdir", default="./reg_out")
    ap.add_argument("--with-scale", action="store_true")
    ap.add_argument("--trim", type=float, default=0.8)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--sample", type=int, default=40000,
                    help="max source points used for ICP (random subsample)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    xyz, rgb, faces = lr.read_ply(args.source)
    tgt_pts, tgt_nrm = load_surface(args.target)
    print(f"source {len(xyz)} pts   target {len(tgt_pts)} pts")

    sub = xyz
    if len(xyz) > args.sample:
        idx = np.random.default_rng(0).choice(len(xyz), args.sample, replace=False)
        sub = xyz[idx]

    M, rmse, hist = icp_point_to_plane(sub, tgt_pts, tgt_nrm,
                                       max_iter=args.max_iter, trim=args.trim,
                                       with_scale=args.with_scale)
    print(f"ICP final RMSE (inliers) = {rmse:.4f}   iters={len(hist)}")
    print(f"RMSE history: {['%.3f'%h for h in hist[:8]]}{' ...' if len(hist)>8 else ''}")

    # apply refinement to the FULL cloud and save
    xyz_ref = (M[:3, :3] @ xyz.T).T + M[:3, 3]
    base = os.path.splitext(os.path.basename(args.source))[0]
    out_ply = os.path.join(args.outdir, f"{base}_icp.ply")
    lr.write_ply(out_ply, xyz_ref, rgb, faces)
    np.save(os.path.join(args.outdir, "icp_refine_delta.npy"), M)
    with open(os.path.join(args.outdir, "icp_refine_delta.txt"), "w") as o:
        o.write(f"# ICP refinement delta (applied AFTER landmark transform)\n")
        o.write(f"# final inlier RMSE = {rmse:.6f}\n")
        for row in M:
            o.write("  ".join(f"{v: .8f}" for v in row) + "\n")
    print(f"wrote {out_ply}")


if __name__ == "__main__":
    main()
