#!/usr/bin/env python3
"""
batch_register_all.py -- register all 17 endoscope patches to the CT spine and
quantify how well the endoscope reconstruction matches the true bone surface.

For each locNN:
  * automatic similarity ICP (rotation+translation+uniform scale), with
    centroid initialization, a small realistic scale prior, and several
    rotation restarts (keep the best trimmed fit) -- this is the best we can
    do WITHOUT hand-placed correspondences.
  * record recovered scale, surface deviation to the CT (median/p90 mm),
    and a RELIABILITY flag (plausible scale + the patch conforms to the
    surface rather than slicing through it).

Outputs:
  reg_out/locNN_reg.ply             each aligned patch
  reg_out/combined_CT_plus_17.ply   CT (grey) + all 17 patches (colored)
  reg_out/accuracy_summary.csv      per-patch metrics + reliability
  reg_out/accuracy_overlay.png      rendered overlay
"""
import os, glob, importlib.util, numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
lr  = _load("lr", "landmark_register.py")
icp = _load("icp", "icp_refine.py")

OUT = os.path.join(HERE, "reg_out"); os.makedirs(OUT, exist_ok=True)
CT_STL = os.path.join(HERE, "ct_spine.stl")
ENDO_DIR = os.path.join(HERE, "..", "depth_outputs", "ply_allon_mesh")

SCALE_PRIOR = 0.06          # mm per pixel, realistic endoscopic field starting guess
N_SRC = 2000               # source subsample
N_TGT = 20000              # target subsample


def rotmat(axis, ang):
    axis = axis / np.linalg.norm(axis); x, y, z = axis; c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def load_ct_ras():
    c = np.load(os.path.join(HERE, "ct_ras_cache.npz"))
    return c["V"].astype(float), c["Nn"].astype(float)


def register_one(src_pts, tgt_pts, tgt_nrm, tgt_tree):
    """Similarity ICP with centroid init + rotation restarts. Returns M, med_dev, p90, scale."""
    tc = tgt_pts.mean(0); sc = src_pts.mean(0)
    best = None
    for ax in [(1,0,0),(0,0,1)]:
        for ang in np.linspace(0, 2*np.pi, 3, endpoint=False):
            R0 = rotmat(np.array(ax, float), ang)
            P = (SCALE_PRIOR * R0 @ (src_pts - sc).T).T + tc
            M0 = np.eye(4); M0[:3, :3] = SCALE_PRIOR*R0; M0[:3, 3] = tc - (SCALE_PRIOR*R0 @ sc)
            M, rmse, _ = icp.icp_point_to_plane(P, tgt_pts, tgt_nrm,
                                                max_iter=25, trim=0.6, with_scale=True)
            Mtot = M @ M0
            if best is None or rmse < best[1]:
                best = (Mtot, rmse)
    M = best[0]
    Pf = (M[:3, :3] @ src_pts.T).T + M[:3, 3]
    d, _ = tgt_tree.query(Pf)
    scale = float(np.linalg.norm(M[:3, :3], axis=0).mean())
    # conformity: for a good fit the patch hugs one side (points near surface);
    # slicing-through shows as a bimodal / large spread of signed room -> use frac within 3mm
    frac3 = float(np.mean(d < 3.0))
    return M, float(np.median(d)), float(np.percentile(d, 90)), scale, frac3


def main():
    Vt, Nt = load_ct_ras()
    rng = np.random.default_rng(0)
    ti = rng.choice(len(Vt), min(N_TGT, len(Vt)), replace=False)
    Vt_s, Nt_s = Vt[ti], Nt[ti]
    tree = cKDTree(Vt_s)

    files = sorted(glob.glob(os.path.join(ENDO_DIR, "loc*_allon_mesh.ply")))
    palette = (rng.random((len(files), 3))*0.7 + 0.3)
    combined_xyz = [Vt]; combined_rgb = [np.tile([180, 180, 180], (len(Vt), 1))]
    rows = []
    for i, f in enumerate(files):
        loc = os.path.basename(f).split("_")[0]
        xyz, rgb, faces = lr.read_ply(f)
        si = rng.choice(len(xyz), min(N_SRC, len(xyz)), replace=False)
        M, med, p90, scale, frac3 = register_one(xyz[si], Vt_s, Nt_s, tree)
        full = (M[:3, :3] @ xyz.T).T + M[:3, 3]
        lr.write_ply(os.path.join(OUT, f"{loc}_reg.ply"), full, rgb, faces)
        # plausibility: scale in a sane endoscopic range and patch conforms
        plausible = (0.02 < scale < 0.12) and (frac3 > 0.6)
        rows.append((loc, scale, med, p90, frac3, "OK" if plausible else "SUSPECT"))
        col = (palette[i]*255).astype(np.uint8)
        combined_xyz.append(full[::3]); combined_rgb.append(np.tile(col, (len(full[::3]), 1)))
        print(f"{loc}: scale={scale:.4f}  median_dev={med:.2f}mm  p90={p90:.2f}mm  "
              f"within3mm={100*frac3:.0f}%  -> {rows[-1][5]}")

    # combined model
    CX = np.vstack(combined_xyz); CR = np.vstack(combined_rgb).astype(np.uint8)
    lr.write_ply(os.path.join(OUT, "combined_CT_plus_17.ply"), CX, CR)

    # summary csv
    with open(os.path.join(OUT, "accuracy_summary.csv"), "w") as o:
        o.write("location,recovered_scale_mm_per_px,median_dev_mm,p90_dev_mm,frac_within_3mm,reliability\n")
        for r in rows:
            o.write(f"{r[0]},{r[1]:.5f},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]}\n")

    ok = [r for r in rows if r[5] == "OK"]
    print("\n==== SUMMARY ====")
    print(f"patches registered plausibly (OK): {len(ok)}/{len(rows)}")
    if ok:
        meds = np.array([r[2] for r in ok])
        print(f"median surface deviation over OK patches: {np.median(meds):.2f} mm "
              f"(range {meds.min():.2f}-{meds.max():.2f})")
    print("scales (should cluster if working distance is consistent):",
          [round(r[1], 3) for r in rows])


if __name__ == "__main__":
    main()
