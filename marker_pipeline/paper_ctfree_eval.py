"""Paper driver: rerun the CT-free solve on the CT shots, persist every number
to JSON, and render the evaluation overlay figure (CT-free surface in CT frame,
rigid-only alignment, reconstructed vs CT markers)."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nearfield_ctfree as NF

OUT = NF.OUT
SHOTS = ["shot_004", "shot_005", "shot_006", "shot_007"]
res = {}
plots = {}
for shot in SHOTS:
    D, Dsm, ref, keep, scan, hist = NF.reconstruct(shot, mu_fixed=8.0,
                                                   multi_start=True, verbose=True)
    b, s, mu, e = hist[-1]
    if keep.sum() < 2000:
        res[shot] = dict(unstable=True, keep=int(keep.sum()))
        continue
    ev = NF.evaluate(shot, D, keep)
    relief = float(np.percentile(D[keep], 90) - np.percentile(D[keep], 10))
    res[shot] = dict(
        b=round(b, 2), s=round(s, 3), mu=mu, residual=round(e, 4),
        relief=round(relief, 1), keep_px=int(keep.sum()),
        dist_scale_median=round(float(np.median(ev["ratio"])), 3),
        dist_scale_each=[round(float(x), 3) for x in ev["ratio"]],
        sim_scale=round(ev["scale"], 3),
        fre_rms=round(ev["fre_rms"], 2),
        fre_each=[round(float(x), 2) for x in ev["fre"]],
        n_markers=len(ev["ids"]))
    # store for figure: mesh points in CT frame after rigid Kabsch
    vv, uu = np.where(keep)
    step = max(1, len(vv) // 8000)
    cam = NF.R.backproject(uu[::step].astype(float), vv[::step].astype(float),
                           Dsm[vv, uu][::step])
    ct = (ev["Rk"] @ cam.T).T + ev["tk"]
    col = ref[vv, uu][::step]
    mk_ct = (ev["Rk"] @ ev["mk"].T).T + ev["tk"]
    plots[shot] = (ct, col, ev["obj"], mk_ct, ev["fre_rms"])

json.dump(res, open(os.path.join(OUT, "ctfree_paper_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

fig = plt.figure(figsize=(10.5, 3.6), dpi=200)
for i, shot in enumerate(["shot_004", "shot_005", "shot_006"]):
    ct, col, obj, mk_ct, fre = plots[shot]
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.scatter(ct[:, 0], ct[:, 1], ct[:, 2], c=np.clip(col, 0, 1), s=1.5, alpha=0.5)
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], c="k", s=70, marker="X",
               label="CT fiducials", depthshade=False)
    ax.scatter(mk_ct[:, 0], mk_ct[:, 1], mk_ct[:, 2], facecolors="none",
               edgecolors="r", s=110, linewidths=1.8, label="reconstructed",
               depthshade=False)
    pad = 14
    ax.set_xlim(obj[:, 0].min() - pad, obj[:, 0].max() + pad)
    ax.set_ylim(obj[:, 1].min() - pad, obj[:, 1].max() + pad)
    ax.set_zlim(obj[:, 2].min() - pad, obj[:, 2].max() + pad)
    lvl = int(shot.split("_")[1])
    ax.set_title(f"level {lvl}: FRE {fre:.2f} mm", fontsize=9)
    ax.tick_params(labelsize=5, pad=-2)
    ax.view_init(elev=22, azim=-60)
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")
fig.suptitle("CT-free surfaces in CT coordinates (rigid 6-DOF alignment only)",
             fontsize=10, y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_ctfree_overlay.png"), bbox_inches="tight",
            facecolor="white")
print("wrote fig_ctfree_overlay.png")
