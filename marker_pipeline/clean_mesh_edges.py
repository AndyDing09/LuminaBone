"""
Delete mesh that runs off the edge (the 'fan'/'tail' artifacts where the surface
stretches across an occlusion/silhouette boundary or off the frame). Works on any
coloured PLY produced by this pipeline.

Two artifacts are removed:
  * stretched 'fans' that bridge an occlusion/silhouette (long 3-D edges), and
  * 'flip-up' walls where the surface curls up off the bone at the silhouette and
    stands nearly perpendicular to the flat cap (short edges, so the edge test
    misses them -- they are caught by a normal-tilt test instead).

Pipeline:
  1. remove triangles whose longest 3-D edge exceeds K x the median edge (adaptive),
  2. remove triangles that stand more than --max-tilt deg from the dominant
     orientation of their own connected component (the flip-up walls),
  3. erode --peel boundary rings,
  4. drop orphaned vertices and keep only sizeable connected components,
  5. write the cleaned PLY.

Usage:
  python clean_mesh_edges.py IN.ply OUT.ply [--k 4.0] [--max-edge MM]
                             [--min-comp 200]
  python clean_mesh_edges.py --all        # clean every *_ct.ply in ct_registered/
"""
import os, sys, argparse, math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
OUTDIR = bd.project_path("depth_outputs", "ct_registered")


def read_ply(path):
    L = open(path).read().splitlines()
    nv = [int(l.split()[2]) for l in L if l.startswith("element vertex")][0]
    nfl = [int(l.split()[2]) for l in L if l.startswith("element face")]
    nf = nfl[0] if nfl else 0                            # point clouds have no faces
    h = L.index("end_header") + 1
    V = np.array([[float(x) for x in L[h + i].split()] for i in range(nv)])  # x y z r g b
    F = np.array([[int(x) for x in L[h + nv + i].split()[1:4]] for i in range(nf)], int)
    return V, F


def write_ply(path, V, F):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for r in V:
            f.write(f"{r[0]:.3f} {r[1]:.3f} {r[2]:.3f} "
                    f"{int(r[3])} {int(r[4])} {int(r[5])}\n")
        for a, b, c in F:
            f.write(f"3 {a} {b} {c}\n")


def tri_max_edge(P, F):
    """longest 3-D edge length per triangle."""
    v0, v1, v2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    e = np.stack([np.linalg.norm(v1 - v0, axis=1),
                  np.linalg.norm(v2 - v1, axis=1),
                  np.linalg.norm(v0 - v2, axis=1)], axis=1)
    return e.max(axis=1)


def tri_normals(P, F):
    """unit normal and area per triangle."""
    v0, v1, v2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    c = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(c, axis=1)
    ln_safe = np.where(ln == 0, 1.0, ln)
    return c / ln_safe[:, None], 0.5 * ln                # unit normal, area


def dominant_normal(n, area):
    """area-weighted mean normal of a patch (winding-agnostic via hemisphere flip)."""
    ref = n[np.argmax(area)]
    s = np.sign(n @ ref); s[s == 0] = 1.0
    m = (n * (s * area)[:, None]).sum(0)
    ln = np.linalg.norm(m)
    return m / (ln if ln else 1.0)


def _labels(V, F):
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V)))
    _, lab = connected_components(A, directed=False)
    return lab


def remove_folds(V, F, max_tilt, iters=3):
    """Delete the 'flip-up' walls: triangles whose normal tilts more than
    `max_tilt` deg from the dominant orientation of their own connected
    component (so a multi-vertebra mesh keeps each surface's own tilt).
    Iterated: peeling a wall re-flattens the reference, exposing the rest."""
    if not max_tilt or not len(F):
        return F
    P = V[:, :3]
    cmin = math.cos(math.radians(max_tilt))
    for _ in range(iters):
        lab = _labels(V, F)
        n, area = tri_normals(P, F)
        flab = lab[F[:, 0]]
        keep = np.ones(len(F), bool)
        for c in np.unique(flab):
            m = flab == c
            d = dominant_normal(n[m], area[m])
            keep[m] = np.abs(n[m] @ d) >= cmin           # |cos| small => stands up
        if keep.all():
            break
        F = F[keep]
    return F


def peel_boundary(F, rings):
    """remove the outermost `rings` triangle rings (edges shared by one triangle)."""
    for _ in range(rings):
        if not len(F):
            break
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                    axis=1)                              # (3F, 2) sorted edges
        _, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        on_boundary = (cnt[inv].reshape(3, -1).T == 1).any(axis=1)  # (F,)
        F = F[~on_boundary]
    return F


def clean(V, F, k=4.0, max_edge=None, min_comp=200, peel=1, max_tilt=70.0):
    P = V[:, :3]
    emax = tri_max_edge(P, F)
    thr = max_edge if max_edge is not None else k * np.median(emax)
    F = F[emax <= thr]                                   # (1) drop stretched triangles
    F = remove_folds(V, F, max_tilt)                     # (2) cut the flip-up walls
    if peel:
        F = peel_boundary(F, peel)                       # (3) erode boundary rings
    n0 = len(F)

    if min_comp and len(F):                              # (4) keep big components
        rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V), len(V)))
        ncomp, lab = connected_components(A, directed=False)
        size = np.bincount(lab, minlength=ncomp)         # size per COMPONENT label
        F = F[size[lab[F[:, 0]]] >= min_comp]             # component of the triangle

    used = np.unique(F)                                  # (5) drop orphaned vertices
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F], thr, n0


def run(inp, outp, **kw):
    V, F = read_ply(inp)
    if not len(F):                                       # point cloud, nothing to clean
        print(f"{os.path.basename(inp)}: no faces (point cloud) -- skipped")
        return
    V2, F2, thr, n_after_edge = clean(V, F, **kw)
    write_ply(outp, V2, F2)
    print(f"{os.path.basename(inp)}: {len(F)}->{len(F2)} faces "
          f"({len(F)-len(F2)} removed; edge thr {thr:.2f} mm) -> {os.path.basename(outp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?"); ap.add_argument("outp", nargs="?")
    ap.add_argument("--k", type=float, default=4.0,
                    help="edge threshold = k x median edge (adaptive)")
    ap.add_argument("--max-edge", type=float, default=None,
                    help="absolute longest-edge cap in mm (overrides --k)")
    ap.add_argument("--min-comp", type=int, default=200,
                    help="drop connected components smaller than this many vertices")
    ap.add_argument("--peel", type=int, default=1,
                    help="erode this many boundary triangle rings (the frame/silhouette edge)")
    ap.add_argument("--max-tilt", type=float, default=70.0,
                    help="cut triangles standing more than this many deg from the "
                         "local surface (the flip-up walls); 0 disables")
    ap.add_argument("--all", action="store_true",
                    help="clean every mesh PLY in ct_registered/ -> *_clean.ply")
    a = ap.parse_args()
    kw = dict(k=a.k, max_edge=a.max_edge, min_comp=a.min_comp, peel=a.peel,
              max_tilt=a.max_tilt)
    if a.all:
        for fn in sorted(os.listdir(OUTDIR)):
            if fn.endswith("_ct.ply") and "_clean" not in fn:
                run(os.path.join(OUTDIR, fn),
                    os.path.join(OUTDIR, fn.replace(".ply", "_clean.ply")), **kw)
    else:
        run(a.inp, a.outp or a.inp.replace(".ply", "_clean.ply"), **kw)


if __name__ == "__main__":
    main()
