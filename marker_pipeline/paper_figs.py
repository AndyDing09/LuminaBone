"""Paper figures from saved data + a fresh landscape scan with CURRENT code.
(a) fig_landscape.png : flat-plane E(b,1,mu) over the focus band (shot_004)
                        + the three mu*=8 multi-start trajectories.
(b) fig_overlay.png : brightened 3-panel CT-frame overlay from npz.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure font for every label in this file (titles, axes, ticks, legends).
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica",
                                          "Liberation Sans", "DejaVu Sans"]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"  # keeps $\mu$ consistent
matplotlib.rcParams["axes.unicode_minus"] = False       # Arial lacks U+2212

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nearfield_ctfree as NF

OUT = NF.OUT
FIGS = os.path.join(os.path.dirname(HERE), "paper", "figures")

# ---------- (a) landscape + trajectories ----------
shot = "shot_004"
lums, ref = NF.load_lums(shot)
base_mask = NF.bd.bone_mask(np.stack(lums, 0).mean(0))
E = NF.Objective(lums, base_mask)
bs, mus, Es = NF.flat_scan(E)

M = np.moveaxis(np.stack(lums, 0), 0, -1)
trajs = []
for b0 in NF.B_STARTS:
    best, hist = NF._trajectory(E, M, base_mask, NF.FX, NF.FY, NF.CX, NF.CY,
                                b0, 6, mu_fixed=8.0)
    trajs.append((b0, best, hist))
kept = min(t[1] for t in trajs)  # (e, b, s, mu, zt)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.95), dpi=250)

# (a) warm-to-cool ramp across mu: red -> orange -> yellow -> green. Ordered so
# the reader can read the (b, mu) trade-off off the colour alone -- the red end
# (low mu) rises with distance, the green end (high mu) falls.
MU_COLORS = ["#c81e1e", "#e8590c", "#f08c00", "#e6b800",
             "#b5c400", "#5aa32f", "#1a7a3a"]
for j, m_ in enumerate(mus):
    axs[0].plot(bs, Es[:, j], "-", lw=1.4, color=MU_COLORS[j % len(MU_COLORS)],
                label=rf"$\mu={m_:.0f}$")
axs[0].set_xlabel("working distance $b$ (mm), flat plane", fontsize=8)
axs[0].set_ylabel("photometric residual $E$", fontsize=8)
axs[0].tick_params(labelsize=7)
# legend stretched horizontally across the top, above the axes
axs[0].legend(fontsize=6, ncol=len(mus), frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.7, handlelength=1.1, handletextpad=0.4)
axs[0].set_title("(a) flat-plane residual landscape", fontsize=8.5, pad=20)

# (b) red / green / blue, one per seed; the kept value is neutral so it does
# not read as a fourth trajectory
SEED_COLORS = ["#c81e1e", "#1a7a3a", "#1f5fa8"]
marks = ("o", "s", "^")
for (b0, best, hist), mk, cc in zip(trajs, marks, SEED_COLORS):
    it = np.arange(1, len(hist) + 1)
    axs[1].plot(it, [h[0] for h in hist], "-" + mk, ms=3.5, lw=1.1, color=cc,
                label=rf"seed $b_0={b0:.0f}$")
axs[1].axhline(kept[1], color="#444444", ls="--", lw=1,
               label=rf"kept: $b={kept[1]:.2f}$, $s={kept[2]:.3f}$")
axs[1].set_xlabel("iteration", fontsize=8)
axs[1].set_ylabel("$b$ (mm)", fontsize=8)
axs[1].tick_params(labelsize=7)
axs[1].legend(fontsize=6, frameon=False)
axs[1].set_title(r"(b) multi-start trajectories ($\mu^\ast{=}8$)", fontsize=8.5,
                 pad=20)                       # match (a), whose legend sits above
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_landscape.png"), bbox_inches="tight",
            facecolor="white")
print("wrote fig_landscape.png; kept:", kept[:4])

# ---------- (b) overlay, brightened, on the CT bone surface ----------
# The CT surface lives in raw Slicer RAS while the meshes live in the CT_CORR
# marker frame; ct_frame_recover.py recovered the rigid map between them by
# fitting the 19 beads onto the bone (1.22 mm residual). Load it if present so
# each panel shows the reconstruction sitting on the actual CT anatomy.
CT_SURF = None
_tf = os.path.join(HERE, "ct_frame_transform.npz")
if os.path.exists(_tf):
    from ct_frame_recover import load_stl
    _z = np.load(_tf)
    _ras = load_stl(NF.bd.project_path("registration", "ct_spine.stl"))
    CT_SURF = (_z["R"].T @ (_ras - _z["t"]).T).T
    print(f"CT surface loaded for overlay: {len(CT_SURF):,} points")
else:
    print("no ct_frame_transform.npz -- run ct_frame_recover.py to show the CT")

fig = plt.figure(figsize=(10.5, 3.4), dpi=220)
for i, shot in enumerate(["shot_004", "shot_005", "shot_006"]):
    z = np.load(os.path.join(OUT, f"{shot}_plotdata_mu8.npz"))
    ct, col, obj, mk, fre = z["ct"], z["col"], z["obj"], z["mk"], float(z["fre"])
    disp = np.clip((np.clip(col, 0, 1) ** (1 / 2.2)) * 1.35, 0, 1)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    if CT_SURF is not None:                    # CT bone first, so it sits behind
        c = obj.mean(0)
        near = CT_SURF[np.linalg.norm(CT_SURF - c, axis=1) < 42.0]
        if len(near) > 16000:
            near = near[np.linspace(0, len(near) - 1, 16000).astype(int)]
        # Cropped WIDER than the reconstructed patch: the reconstruction sits
        # on the bone and would otherwise occlude it entirely, so the CT is only
        # legible where it extends past the patch. Cool blue keeps the two
        # distinguishable; the photometric surface is drawn opaque on top.
        ax.scatter(near[:, 0], near[:, 1], near[:, 2], c="#5b8db8", s=1.1,
                   alpha=0.34, linewidths=0, depthshade=False, zorder=0,
                   label="CT bone surface" if i == 0 else None)
    ax.scatter(ct[:, 0], ct[:, 1], ct[:, 2], c=disp, s=2.0, alpha=0.95,
               linewidths=0, depthshade=False, zorder=3)
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], c="k", s=85, marker="X",
               label="CT fiducials", depthshade=False, zorder=5)
    ax.scatter(mk[:, 0], mk[:, 1], mk[:, 2], facecolors="none",
               edgecolors="#c62828", s=130, linewidths=2.0,
               label="reconstructed", depthshade=False, zorder=6)
    pad = 16
    ax.set_xlim(obj[:, 0].min() - pad, obj[:, 0].max() + pad)
    ax.set_ylim(obj[:, 1].min() - pad, obj[:, 1].max() + pad)
    ax.set_zlim(obj[:, 2].min() - pad, obj[:, 2].max() + pad)
    lvl = int(shot.split("_")[1])
    ax.set_title(f"level {lvl}: rigid-only FRE {fre:.2f} mm", fontsize=10)
    ax.tick_params(labelsize=5, pad=-2)
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((1, 1, 0.85))
    if i == 0:
        ax.legend(fontsize=7.5, loc="upper left")
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.04)
fig.savefig(os.path.join(FIGS, "fig_overlay.png"), bbox_inches="tight",
            facecolor="white")
print("wrote fig_overlay.png")


# ---------- (c) far-field vs near-field bowl, from current verified meshes ----------
def load_ply(path):
    verts = []
    # explicit encoding: Windows defaults to cp1252, which chokes on stray
    # bytes in these ASCII PLYs (0x81 is undefined in cp1252)
    with open(path, encoding="utf-8", errors="replace") as f:
        n = 0
        for line in f:
            if line.startswith("element vertex"): n = int(line.split()[-1])
            if line.strip() == "end_header": break
        for i, line in enumerate(f):
            if i >= n: break
            p = line.split()
            verts.append([float(p[0]), float(p[1]), float(p[2]),
                          int(p[3]), int(p[4]), int(p[5])])
    return np.array(verts)

fig = plt.figure(figsize=(7.2, 3.0), dpi=220)
for i, (name, title) in enumerate((
        ("shot_004_surface_photo_ct", "far-field: bowl"),
        ("shot_004_mesh_nearfield_ct", "near-field: flattened"))):
    V = load_ply(os.path.join(OUT, name + ".ply"))
    step = max(1, len(V) // 12000)
    V = V[::step]
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    zr = V[:, 2].max() - V[:, 2].min()
    sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=V[:, 2], cmap="viridis",
                    s=1.5, alpha=0.7, linewidths=0)
    ax.set_title(f"{title}   ($z$-extent {zr:.0f} mm)", fontsize=9)
    ax.tick_params(labelsize=5, pad=-2)
    ax.view_init(elev=12, azim=-70)
    ax.set_box_aspect((1, 1, 0.7))
fig.suptitle("level 4, identical registration -- only the light model differs",
             fontsize=9.5, y=0.99)
fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.02, wspace=0.02)
fig.savefig(os.path.join(FIGS, "fig_bowl_new.png"), bbox_inches="tight",
            facecolor="white")
print("wrote fig_bowl_new.png")
