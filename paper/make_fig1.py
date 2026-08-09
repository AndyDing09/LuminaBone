"""Generate Fig. 1: (a) near-coaxial rig geometry, (b) pipeline block diagram.
Run: python make_fig1.py  ->  figures/fig1_system.png
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
TEAL, INK, MUT, RED = "#0f6e8c", "#19222b", "#5a6673", "#b23b3b"

fig = plt.figure(figsize=(7.2, 3.1), dpi=200)
fig.patch.set_facecolor("white")

# ---- (a) geometry (side view) ----
ax = fig.add_axes([0.04, 0.08, 0.44, 0.86]); ax.axis("off")
ax.set_xlim(-0.5, 6.2); ax.set_ylim(-2.2, 2.2)
D = 5.0; ar = 1.5
# optical axis
ax.annotate("", (D + 0.7, 0), (-0.2, 0),
            arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
ax.text(D + 0.75, -0.02, "optical axis", color=MUT, fontsize=6, va="center")
# scope body
ax.add_patch(FancyBboxPatch((-0.35, -0.55), 0.7, 1.1, boxstyle="round,pad=0.02",
             fc="#e8eef1", ec=INK, lw=1))
ax.text(0, 0, "lens", ha="center", va="center", fontsize=6, color=INK)
# LEDs
for sy in (ar, -ar):
    ax.plot(0.05, sy, "o", ms=6, mfc=TEAL, mec=INK, mew=0.6, zorder=5)
    ax.plot([0.05, D], [sy, 0.12 * np.sign(sy)], color=TEAL, lw=0.9, alpha=0.7)
ax.annotate("", (0.05, ar), (0.05, 0), arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9))
ax.text(0.18, ar / 2, r"$a_r{=}6.05$ mm", color=RED, fontsize=6, va="center")
ax.text(0.15, ar + 0.16, "LED", color=TEAL, fontsize=6)
# surface
ys = np.linspace(-1.7, 1.7, 100)
ax.plot(D + 0.25 * np.sin(ys * 1.6), ys, color=INK, lw=1.6)
ax.text(D + 0.35, 1.75, "bone surface", color=INK, fontsize=6)
# working distance
ax.annotate("", (D, -1.9), (0, -1.9), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
ax.text(D / 2, -2.12, r"working distance $D\approx55$ mm", ha="center", fontsize=6.3, color=INK)
# half-angle alpha
ax.add_patch(Arc((0.05, ar), 2.2, 2.2, angle=0, theta1=-72, theta2=-58, color=RED, lw=1))
ax.text(1.5, ar - 0.5, r"$\alpha\approx6^\circ$", color=RED, fontsize=6.5)
ax.text(0.05, -ar - 0.55, r"near-coaxial: $\sin\alpha\approx0.11$",
        ha="left", fontsize=6, color=MUT)
ax.set_title("(a) rig geometry", fontsize=8, fontweight="bold", loc="left")

# ---- (b) pipeline ----
ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.86]); ax2.axis("off")
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
boxes = [
    (5, 9.0, "4 single-LED frames + dark", "#e8eef1"),
    (5, 7.1, "Near-field PS solve\n(per-pixel, 1/$r^2$)", "#d7e7ec"),
    (5, 5.2, "Frankot--Chellappa\nrelative depth $Z$", "#d7e7ec"),
    (5, 3.3, "PnP pose + affine\n(CT fiducials)", "#f3e2c8"),
    (5, 1.3, "CT-registered mesh\n(3-D FRE)", "#cfe6d8"),
]
for (x, y, t, c) in boxes:
    ax2.add_patch(FancyBboxPatch((x - 3.0, y - 0.62), 6.0, 1.24,
                  boxstyle="round,pad=0.04", fc=c, ec=INK, lw=1))
    ax2.text(x, y, t, ha="center", va="center", fontsize=6.6, color=INK)
for i in range(len(boxes) - 1):
    ax2.annotate("", (5, boxes[i + 1][1] + 0.66), (5, boxes[i][1] - 0.66),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax2.annotate("iterate ×4", (8.2, 6.15), (8.2, 6.15), fontsize=6, color=TEAL, ha="center")
ax2.annotate("", (7.9, 7.1), (7.9, 5.2),
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9,
                             connectionstyle="arc3,rad=0.5"))
ax2.set_title("(b) pipeline", fontsize=8, fontweight="bold", loc="left")

out = os.path.join(HERE, "figures", "fig1_system.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
