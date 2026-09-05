"""
Dense surface accuracy against CT -- the honest answer to "is the depth right
AWAY from the fiducial markers?"

FRE samples the error at 4 points that the alignment was fitted to. This
measures every reconstructed vertex against the CT bone surface, and -- the
part that actually answers the question -- plots error as a function of
distance from the nearest fiducial.

Uses ct_frame_transform.npz (from ct_frame_recover.py) to bring the CT surface
into the CT_CORR marker frame that our meshes live in.

Validation guards, because registration/ACCURACY_REPORT.md documents that
patch-to-CT ICP "drapes" and reports a deceptively good ~1.9 mm:
  * the CT->marker transform was fitted to the BEADS only, never to our
    reconstruction, so it cannot flatter us;
  * each shot's markers must land on a DISTINCT vertebra, spaced ~25-35 mm
    along the spine -- checked and printed;
  * our meshes are placed by rigid Kabsch on the markers, NOT by ICP to the
    surface, so no draping freedom exists.

Run: python dense_ct_accuracy.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
import calibrate_photometric_ct as C
from ct_frame_recover import load_stl
from clean_mesh_edges import read_ply

OUT = bd.project_path("depth_outputs", "ct_registered")
TF = os.path.join(HERE, "ct_frame_transform.npz")


def main():
    if not os.path.exists(TF):
        raise SystemExit("run ct_frame_recover.py first")
    z = np.load(TF)
    R, t = z["R"], z["t"]
    print(f"CT->marker transform: bead-to-surface rms {float(z['rms']):.2f} mm")

    surf_ras = load_stl(bd.project_path("registration", "ct_spine.stl"))
    # markers_ras = R @ marker + t   =>   surface into marker frame:
    surf = (R.T @ (surf_ras - t).T).T
    tree = cKDTree(surf)

    # --- guard: do the five shots land on five distinct, spaced vertebrae? ---
    print("\nvalidation - marker groups along the spine:")
    cents = {}
    for sh in sorted(C.CT_CORR):
        m = np.array([xyz for uv, xyz in C.CT_CORR[sh].values()])
        d, _ = tree.query(m)
        cents[sh] = m.mean(0)
        print(f"  {sh}: centroid ({m.mean(0)[0]:6.1f},{m.mean(0)[1]:6.1f},"
              f"{m.mean(0)[2]:7.1f})  bead-to-bone {d.mean():.2f} mm")
    ks = sorted(cents)
    gaps = [float(np.linalg.norm(cents[b] - cents[a]))
            for a, b in zip(ks, ks[1:])]
    print(f"  level-to-level spacing: "
          f"{', '.join(f'{g:.1f}' for g in gaps)} mm "
          f"({'plausible vertebral pitch' if all(18 < g < 45 for g in gaps) else 'IMPLAUSIBLE - transform suspect'})")

    # --- dense error per shot ---
    print(f"\n{'shot':10}{'verts':>9}{'median':>9}{'rms':>8}{'p90':>8}"
          f"{'near markers':>14}{'far from markers':>18}")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    rows = []
    for ax, sh in zip(axs, ["shot_004", "shot_005", "shot_006"]):
        p = os.path.join(OUT, f"{sh}_mesh_ctfree_ct.ply")
        if not os.path.exists(p):
            print(f"  {sh}: mesh missing ({os.path.basename(p)}) - skipped")
            continue
        V, F = read_ply(p)
        P = V[:, :3]
        d, _ = tree.query(P)
        mk = np.array([xyz for uv, xyz in C.CT_CORR[sh].values()])
        dm = cKDTree(mk).query(P)[0]          # distance to nearest fiducial
        near = d[dm < 10]
        far = d[dm > 25]
        rows.append((sh, d, dm))
        print(f"{sh:10}{len(P):9,}{np.median(d):9.2f}"
              f"{np.sqrt((d**2).mean()):8.2f}{np.percentile(d,90):8.2f}"
              f"{np.median(near):14.2f}{np.median(far):18.2f}")
        # error vs distance-from-fiducial
        bins = np.arange(0, min(60, dm.max()), 5.0)
        med = [np.median(d[(dm >= a) & (dm < a + 5)])
               if ((dm >= a) & (dm < a + 5)).sum() > 50 else np.nan
               for a in bins]
        ax.plot(bins + 2.5, med, "-o", ms=4, color="#0f6e8c")
        ax.set_title(f"{sh}", fontsize=11, fontweight="bold")
        ax.set_xlabel("distance from nearest fiducial (mm)")
        ax.set_ylabel("median surface error (mm)")
        ax.grid(alpha=0.3)
    fig.suptitle("Does the reconstruction degrade away from the fiducials?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    o = os.path.join(OUT, "dense_error_vs_fiducial_distance.png")
    fig.savefig(o, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"\nplot -> {o}")

    if rows:
        alld = np.concatenate([r[1] for r in rows])
        alldm = np.concatenate([r[2] for r in rows])
        print(f"\npooled: median {np.median(alld):.2f} mm, "
              f"rms {np.sqrt((alld**2).mean()):.2f} mm, "
              f"p90 {np.percentile(alld,90):.2f} mm")
        print(f"  within 10 mm of a fiducial: median "
              f"{np.median(alld[alldm<10]):.2f} mm")
        print(f"  beyond  25 mm of a fiducial: median "
              f"{np.median(alld[alldm>25]):.2f} mm")


if __name__ == "__main__":
    main()
