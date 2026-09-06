"""Fig. 1 for the ICRA paper: (a) two raw single-LED frames (system in
action), (b) near-coaxial rig geometry with unknown working distance, (c)
reconstruction pipeline with the evaluation stage dashed.
Run: python make_fig1_pipeline.py -> figures/fig1_pipeline.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEAL, INK, MUT, RED = "#0f6e8c", "#19222b", "#5a6673", "#b23b3b"

fig = plt.figure(figsize=(3.6, 4.35), dpi=300)
fig.patch.set_facecolor("white")

# ---- (a) two raw single-LED frames ----
d = os.path.join(ROOT, "Data_collection", "calib_charuco", "shot_006")
for i, (led, x0) in enumerate((("led1", 0.015), ("led3", 0.505))):
    ax = fig.add_axes([x0, 0.665, 0.48, 0.30])
    img = mpimg.imread(os.path.join(d, f"{led}.png"))
    g = 1.9  # display stretch only
    ax.imshow(np.clip(img.astype(float) ** (1 / g) * 1.25, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(0.8)
    ax.text(0.03, 0.05, f"LED {led[-1]}", transform=ax.transAxes, fontsize=6,
            color="white", fontweight="bold",
            bbox=dict(fc="black", alpha=0.55, pad=1.5, ec="none"))
fig.text(0.015, 0.972, "(a) single-LED frames of the exposed lumbar surface"
         " (2 of 4)", fontsize=6.8, fontweight="bold", color=INK)

# ---- (b) geometry ----
ax = fig.add_axes([0.02, 0.40, 0.96, 0.235]); ax.axis("off")
ax.set_xlim(-0.5, 8.6); ax.set_ylim(-2.35, 2.3)
D, ar = 5.0, 1.5
ax.annotate("", (D + 0.7, 0), (-0.2, 0),
            arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
ax.text(D + 0.78, 0, "optical axis", color=MUT, fontsize=5.6, va="center")
ax.add_patch(FancyBboxPatch((-0.35, -0.55), 0.7, 1.1, boxstyle="round,pad=0.02",
             fc="#e8eef1", ec=INK, lw=0.9))
ax.text(0, 0, "lens", ha="center", va="center", fontsize=5.6, color=INK)
for sy in (ar, -ar):
    ax.plot(0.05, sy, "o", ms=5, mfc=TEAL, mec=INK, mew=0.5, zorder=5)
    ax.plot([0.05, D], [sy, 0.12 * np.sign(sy)], color=TEAL, lw=0.8, alpha=0.7)
ax.annotate("", (0.05, ar), (0.05, 0),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=0.8))
ax.text(0.2, ar / 2, r"$a_r{=}6.05$ mm", color=RED, fontsize=5.6, va="center")
ax.text(0.18, ar + 0.28, "LED ring (4)", color=TEAL, fontsize=5.6)
ys = np.linspace(-1.7, 1.7, 100)
ax.plot(D + 0.25 * np.sin(ys * 1.6), ys, color=INK, lw=1.4)
ax.text(D + 0.32, 1.85, "bone", color=INK, fontsize=5.6)
ax.annotate("", (D, -2.0), (0, -2.0),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(D / 2, -2.32, r"$b$ unknown (recovered); focus band 45--70 mm",
        ha="center", fontsize=5.8, color=INK)
ax.add_patch(Arc((0.05, ar), 2.2, 2.2, angle=0, theta1=-72, theta2=-58,
             color=RED, lw=0.9))
ax.text(1.55, ar - 0.52, r"$\alpha\approx6^\circ$", color=RED, fontsize=5.8)
ax.text(6.05, -1.2, "near-coaxial:\n" r"$\sin\alpha\approx0.11$",
        fontsize=5.6, color=MUT)
fig.text(0.015, 0.632, "(b) near-coaxial ring: tilt signal scales with"
         r" $\sin\alpha$", fontsize=6.8, fontweight="bold", color=INK)

# ---- (c) pipeline ----
ax2 = fig.add_axes([0.02, 0.008, 0.96, 0.365]); ax2.axis("off")
ax2.set_xlim(0, 10); ax2.set_ylim(-0.4, 10.4)
boxes = [
    (5.0, 9.5, "4 single-LED frames + dark", "#e8eef1", "solid"),
    (5.0, 7.6, "near-field PS (free normals)\n"
               r"$\rightarrow$ relief $\tilde z$", "#d7e7ec", "solid"),
    (5.0, 5.5, "tied-normal residual search\n"
               r"$\min_{b,s}\,E(b,s;\mu^\ast)$", "#d7e7ec", "solid"),
    (5.0, 3.4, r"metric depth $D=b\,(1+s\,\tilde z/f)$" "\n+ mesh",
     "#cfe6d8", "solid"),
    (5.0, 1.1, "evaluation: rigid 6-DOF FRE /\nscale vs CT fiducials", "#f2f2f2", "dashed"),
]
for (x, y, t, c, ls) in boxes:
    ax2.add_patch(FancyBboxPatch((x - 3.6, y - 0.78), 7.2, 1.56,
                  boxstyle="round,pad=0.04", fc=c, ec=INK, lw=0.9,
                  linestyle=ls))
    ax2.text(x, y, t, ha="center", va="center", fontsize=5.9, color=INK)
for i in range(len(boxes) - 1):
    ls = "dashed" if i == len(boxes) - 2 else "solid"
    ax2.annotate("", (5, boxes[i + 1][1] + 0.82), (5, boxes[i][1] - 0.82),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                 linestyle=ls))
ax2.annotate("", (8.85, 7.6), (8.85, 5.5),
             arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8,
                             connectionstyle="arc3,rad=0.6"))
ax2.text(9.35, 6.55, "iterate,\nkeep best", fontsize=5.3, color=TEAL,
         ha="center", va="center")
ax2.text(0.28, 5.5, r"scale from" "\n" r"$a_r$ + focus" "\nband",
         fontsize=5.3, color=RED, ha="center", va="center")
fig.text(0.015, 0.372, "(c) metric reconstruction pipeline",
         fontsize=6.8, fontweight="bold", color=INK)

out = os.path.join(HERE, "figures", "fig1_pipeline.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
