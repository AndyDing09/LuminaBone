#!/usr/bin/env python3
"""Figure 1 — system geometry.

(a) face view of the LED ring: four emitters 90 deg apart at the measured
    12.08 mm radius about the lens;
(b) side view: working distance, field width, and the resulting illumination
    elevation, with the near-field ratio annotated.

Every dimension is the measured value from nearfield_lambertian.py; nothing
here is decorative invention.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig1_rig.png")

LED_OFFSET_MM = 12.08
FIELD_WIDTH_MM = 37.0
WORKING_DISTANCE_MM = 30.0
AZ = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c9c4"
LED_C, LENS_C, SURF_C = "#eb6834", "#2a78d6", "#8a8a84"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def panel_a(ax):
    """Face view of the ring."""
    ax.add_patch(plt.Circle((0, 0), LED_OFFSET_MM, fill=False, ls="--",
                            lw=0.9, ec=GRID))
    ax.add_patch(plt.Circle((0, 0), 3.1, fc="white", ec=LENS_C, lw=1.6, zorder=3))
    ax.annotate("lens", (0, 0), ha="center", va="center", fontsize=7.5,
                color=LENS_C, zorder=4)

    for i, az in AZ.items():
        a = math.radians(az)
        x, y = LED_OFFSET_MM * math.cos(a), LED_OFFSET_MM * math.sin(a)
        ax.add_patch(plt.Circle((x, y), 2.3, fc=LED_C, ec="white",
                                lw=1.1, zorder=3))
        ax.annotate(f"L{i}", (x, y), ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=4)
        ax.annotate(f"{az:.0f}$^\\circ$", (x * 1.42, y * 1.42), ha="center",
                    va="center", fontsize=7, color=MUTED)

    # radius call-out
    a = math.radians(45)
    ax.annotate("", xy=(LED_OFFSET_MM * math.cos(a), LED_OFFSET_MM * math.sin(a)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.annotate("$r_{\\mathrm{LED}} = 12.08$ mm", (4.6, 5.6), fontsize=7.5,
                rotation=45, ha="center", va="bottom", color=INK)

    # 90 deg separation arc, drawn in the lower-left quadrant so it does not
    # collide with the radius call-out
    ax.add_patch(Arc((0, 0), 2 * 7.4, 2 * 7.4, theta1=180, theta2=270,
                     ec=MUTED, lw=0.8))
    ax.annotate("$90^\\circ$", (7.9 * math.cos(math.radians(225)),
                                7.9 * math.sin(math.radians(225))),
                fontsize=7, color=MUTED, ha="right", va="top")

    ax.set_xlim(-18, 18); ax.set_ylim(-18, 18)
    ax.set_aspect("equal"); ax.axis("off")


def panel_b(ax):
    """Side view: offset, working distance, field, elevation."""
    d, half = WORKING_DISTANCE_MM, FIELD_WIDTH_MM / 2.0

    # surface
    ax.plot([-half - 4, half + 4], [d, d], color=SURF_C, lw=2.4, zorder=2)
    ax.annotate("specimen surface", (half + 3.6, d + 1.6), fontsize=7.5,
                color=MUTED, ha="right")

    # lens + LEDs in the lens plane
    ax.add_patch(plt.Circle((0, 0), 1.5, fc="white", ec=LENS_C, lw=1.5, zorder=4))
    for s, lab in ((+1, "L1"), (-1, "L3")):
        ax.add_patch(plt.Circle((s * LED_OFFSET_MM, 0), 1.5, fc=LED_C,
                                ec="white", lw=1.0, zorder=4))
        ax.annotate(lab, (s * LED_OFFSET_MM, -3.0), fontsize=7,
                    ha="center", color=LED_C)
    ax.annotate("lens", (0, -3.0), fontsize=7, ha="center", color=LENS_C)
    ax.plot([-LED_OFFSET_MM - 4, LED_OFFSET_MM + 4], [0, 0],
            color=GRID, lw=1.0, zorder=1)

    # field-of-view cone
    ax.plot([0, -half], [0, d], color=LENS_C, lw=0.9, ls=":", zorder=2)
    ax.plot([0, half], [0, d], color=LENS_C, lw=0.9, ls=":", zorder=2)

    # ray from LED1 to the on-axis surface point -> elevation angle
    ax.plot([LED_OFFSET_MM, 0], [0, d], color=LED_C, lw=1.3, zorder=3)
    elev = math.degrees(math.atan2(d, LED_OFFSET_MM))
    ax.add_patch(Arc((0, d), 13, 13, theta1=180,
                     theta2=180 + (90 - (90 - elev)), ec=MUTED, lw=0.8))
    ax.annotate(f"$\\theta_{{\\mathrm{{elev}}}} = {elev:.1f}^\\circ$",
                (-1.5, d - 8.4), fontsize=7.5, color=INK, ha="right")

    # working distance
    ax.annotate("", xy=(-half - 2.4, 0), xytext=(-half - 2.4, d),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.annotate("$d_{\\mathrm{work}}$\n$=30$ mm", (-half - 3.4, d / 2),
                fontsize=7.5, ha="right", va="center")

    # field width. NOTE: the y-axis is inverted, so a LARGER y sits LOWER on
    # the page — the label y must exceed the arrow y to clear it.
    ax.annotate("", xy=(-half, d + 4.2), xytext=(half, d + 4.2),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.annotate("field width $= 37$ mm", (0, d + 7.0), fontsize=7.5,
                ha="center", va="center")

    # offset call-out
    ax.annotate("", xy=(0, -6.2), xytext=(LED_OFFSET_MM, -6.2),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.annotate("$r_{\\mathrm{LED}}$", (LED_OFFSET_MM / 2, -7.4), fontsize=7.5,
                ha="center", va="top")

    ax.annotate("near-field ratio  $r_{\\mathrm{LED}}/d_{\\mathrm{work}} = 0.40$",
                (0, d + 11.5), fontsize=8, ha="center", style="italic",
                color=INK)

    ax.set_xlim(-30, 26); ax.set_ylim(-11, 48)
    ax.set_aspect("equal"); ax.axis("off")
    ax.invert_yaxis()


def main():
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.1, 3.5), dpi=300,
        gridspec_kw=dict(width_ratios=[1.0, 1.45], wspace=0.02))
    panel_a(axa)
    panel_b(axb)
    # titles placed in figure coords so both panels label at the same height
    fig.text(0.055, 0.965, "(a)  LED ring, face view", fontsize=9, ha="left")
    fig.text(0.455, 0.965, "(b)  side view (Z into the scene)", fontsize=9,
             ha="left")
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"elevation = {math.degrees(math.atan2(WORKING_DISTANCE_MM, LED_OFFSET_MM)):.2f} deg")
    print(f"near-field ratio = {LED_OFFSET_MM / WORKING_DISTANCE_MM:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
