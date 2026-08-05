#!/usr/bin/env python3
"""Figure 2 — the shot_011 depth-inversion result.

CT distances are RECOMPUTED here from the CT control points and the detected
fiducial pixels at the recess-corrected intrinsics (f_x = W*d_work/field_0 with
d_work = 31 mm), rather than read from the committed CSV, which was written with
the uncorrected 30 mm working distance.

(a) far-field photometric relative depth vs CT-derived lens-to-marker distance,
(b) the near-to-far ordering of the three CT-known markers under each method.

Photometric depths come from the committed pipeline output; CT distances are
solved here. Nothing is synthetic.
Source: depth_outputs/marker_depth_slides/marker_depths.csv (pixels, z_est)
        + the CT control points below.
"""
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SLIDES = os.path.join(ROOT, "depth_outputs", "marker_depth_slides")
OUT = os.path.join(HERE, "fig2_inversion.png")

# categorical slots 1-3, validated all-pairs (CVD dE 9.2, normal 24.0) on white
COLOR = {1: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
SHAPE = {1: "o", 3: "s", 4: "^"}          # secondary encoding: survives grayscale

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


# rig geometry, recess-corrected (see manuscript Sec. III)
TIP_MM, RECESS_MM, FIELD0_MM, W, H = 30.0, 1.0, 37.0, 640, 480
D_WORK_MM = TIP_MM + RECESS_MM
FX = W * D_WORK_MM / FIELD0_MM                          # 536.2 px

CT = {1: (9.224, -1.632, 11.818),
      3: (3.729, -13.209, 23.259),
      4: (-17.665, 0.732, 17.412)}
CLOSEST = 4                                             # confirmed physical fact


def load():
    """z_est from the committed depth CSV; CT distances recomputed by P3P."""
    z_est, px = {}, {}
    with open(os.path.join(SLIDES, "marker_depths.csv")) as f:
        for r in csv.DictReader(f):
            if r["shot"] != "shot_011":
                continue
            i = int(r["id"])
            z_est[i] = float(r["z_mm"])
            px[i] = (float(r["x_px"]), float(r["y_px"]))

    ids = sorted(CT)
    obj = np.array([CT[i] for i in ids])
    img = np.array([px[i] for i in ids])
    K = np.array([[FX, 0, W / 2.0], [0, FX, H / 2.0], [0, 0, 1.0]])
    _, rvecs, tvecs = cv2.solveP3P(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_AP3P)
    dist, ct_rel = None, None
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        cam = (R @ obj.T + tv).T
        if not (cam[:, 2] > 0).all():
            continue
        d = {i: float(np.linalg.norm(c)) for i, c in zip(ids, cam)}
        if min(d, key=d.get) == CLOSEST:
            dist = d
            med = float(np.median(cam[:, 2]))
            ct_rel = {i: float(cam[k, 2] - med) for k, i in enumerate(ids)}
    if dist is None:
        raise RuntimeError("no pose with the confirmed closest fiducial")
    return ids, z_est, ct_rel, dist


def main():
    ids, z_est, ct_rel, dist = load()
    x = np.array([dist[i] for i in ids])                # CT lens distance (mm)
    y = np.array([z_est[i] for i in ids])               # photometric depth (mm)
    r = np.corrcoef(x, y)[0, 1]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.1, 3.05), dpi=300,
        gridspec_kw=dict(width_ratios=[1.32, 1.0], wspace=0.34))

    # ---------------- (a) the anti-correlation ----------------
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 3, x.max() + 3, 50)
    axa.plot(xs, m * xs + b, ls="--", lw=1.0, color=MUTED, zorder=1)

    # per-marker label offsets, chosen so no label touches the fitted line
    LABEL_OFF = {4: (34, -2), 1: (0, 13), 3: (0, 13)}
    LABEL_HA = {4: "center", 1: "center", 3: "center"}
    for i in ids:
        axa.plot(dist[i], z_est[i], SHAPE[i], ms=9, color=COLOR[i],
                 mec="white", mew=1.2, zorder=3, clip_on=False)
        axa.annotate(f"marker {i}", (dist[i], z_est[i]),
                     textcoords="offset points", xytext=LABEL_OFF[i],
                     ha=LABEL_HA[i], va="center", fontsize=8, color=INK)

    axa.set_xlabel("CT lens-to-marker distance (mm)")
    axa.set_ylabel("photometric relative depth $z_{est}$ (mm)")
    axa.set_title("(a)  far-field depth vs. CT ground truth",
                  fontsize=9, loc="left", pad=9)
    axa.annotate(f"$r = {r:+.3f}$   ($n=3$)", xy=(0.97, 0.93),
                 xycoords="axes fraction", ha="right", va="top", fontsize=9)
    axa.annotate("farther from lens $\\rightarrow$ shallower depth reported",
                 xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                 fontsize=7.5, style="italic", color=MUTED)
    axa.set_ylim(-4.2, 4.6)
    axa.set_xlim(31, 59)

    # ---------------- (b) the ordering reversal ----------------
    ph = sorted(ids, key=lambda i: z_est[i])            # ascending z: near -> far
    ct = sorted(ids, key=lambda i: dist[i])             # ascending distance
    pos_ph = {mk: k for k, mk in enumerate(ph)}
    pos_ct = {mk: k for k, mk in enumerate(ct)}

    for i in ids:
        axb.plot([0, 1], [pos_ph[i], pos_ct[i]], "-", lw=1.6,
                 color=COLOR[i], alpha=0.85, zorder=2)
        for xcol, pos in ((0, pos_ph[i]), (1, pos_ct[i])):
            axb.plot(xcol, pos, SHAPE[i], ms=9, color=COLOR[i],
                     mec="white", mew=1.2, zorder=3)
        axb.annotate(str(i), (0, pos_ph[i]), textcoords="offset points",
                     xytext=(-16, 0), ha="center", va="center", fontsize=8)
        axb.annotate(str(i), (1, pos_ct[i]), textcoords="offset points",
                     xytext=(16, 0), ha="center", va="center", fontsize=8)

    axb.set_xlim(-0.42, 1.42)
    axb.set_ylim(2.55, -0.55)                           # rank 1 (nearest) on top
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["photometric", "CT"], fontsize=8.5, color=INK)
    axb.set_yticks([0, 1, 2])
    axb.set_yticklabels(["nearest", "middle", "farthest"], fontsize=8)
    axb.set_title("(b)  near-to-far ordering", fontsize=9, loc="left", pad=9)
    # sits in the empty band below the 'farthest' row, clear of every mark
    axb.annotate("ordering exactly reversed", xy=(0.5, 0.045),
                 xycoords="axes fraction", ha="center", fontsize=7.5,
                 style="italic", color=MUTED)
    axb.tick_params(length=0)

    for ax in (axa, axb):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axa.grid(True, color=GRID, lw=0.6)
    axb.grid(True, axis="y", color=GRID, lw=0.6)
    axb.spines["bottom"].set_visible(False)

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"r = {r:+.4f}   slope = {m:+.4f} mm/mm")
    print(f"photometric near->far: {ph}")
    print(f"CT          near->far: {ct}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
