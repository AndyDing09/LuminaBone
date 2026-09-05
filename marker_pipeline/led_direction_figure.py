"""
Visual check: which side of the image does each LED actually brighten?

Each panel is led_i / mean(all four). Dividing by the 4-LED mean cancels the
bone's albedo and shape, leaving only that LED's illumination signature: red =
this LED lights it more than average, blue = less. The arrow is the
brightness-weighted centroid of the ratio (where the beam LANDS).

Run: python led_direction_figure.py [shot_004 ...]
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
from nearfield_ct import load_lums, AZ

NAMES = {0.0: "RIGHT", 90.0: "TOP", 180.0: "LEFT", 270.0: "BOTTOM"}


def compass(deg):
    """nearest cardinal name for an azimuth (0=right, 90=top)."""
    d = deg % 360.0
    best = min(NAMES, key=lambda a: min(abs(d - a), 360 - abs(d - a)))
    off = min(abs(d - best), 360 - abs(d - best))
    return NAMES[best], off


def analyse(shot):
    lums, _ = load_lums(shot)
    I = np.stack(lums, 0)
    mask = bd.bone_mask(I.mean(0))
    mean4 = I.mean(0) + 1e-9
    H, W = mean4.shape
    yy, xx = np.mgrid[:H, :W]
    cx, cy = xx[mask].mean(), yy[mask].mean()
    out = []
    for k in range(4):
        ratio = I[k] / mean4
        w = np.where(mask, np.maximum(ratio - 1.0, 0.0), 0.0)  # only the LIT excess
        if w.sum() < 1e-6:
            w = np.where(mask, ratio, 0.0)
        dx = (w * xx).sum() / w.sum() - cx
        dy = (w * yy).sum() / w.sum() - cy
        az = math.degrees(math.atan2(-dy, dx)) % 360.0       # +y up convention
        out.append((ratio, mask, az, dx, dy, cx, cy))
    return out


def main():
    shots = [a for a in sys.argv[1:] if a.startswith("shot_")] or \
            ["shot_004", "shot_005", "shot_006"]
    for shot in shots:
        res = analyse(shot)
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
        fig.patch.set_facecolor("white")
        print(f"\n{shot}:")
        for k, (ratio, mask, az, dx, dy, cx, cy) in enumerate(res):
            led = k + 1
            ax = axs[k]
            disp = np.where(mask, ratio, np.nan)
            lo, hi = np.nanpercentile(disp, [3, 97])
            m = max(abs(lo - 1), abs(hi - 1))
            im = ax.imshow(disp, cmap="RdBu_r", vmin=1 - m, vmax=1 + m)
            ax.arrow(cx, cy, dx * 2.2, dy * 2.2, width=4, color="#111",
                     length_includes_head=True, head_width=18, zorder=5)
            ax.plot(cx, cy, "o", ms=6, mfc="w", mec="#111", zorder=6)
            name, off = compass(az)
            claim = NAMES[AZ[led]]
            agree = "agrees" if name == claim else "DIFFERS"
            ax.set_title(f"led{led}.png   beam lands: {name}  ({az:.0f}°)\n"
                         f"code says: {claim}   → {agree}",
                         fontsize=10,
                         color=("#1a7a3a" if name == claim else "#b23b3b"),
                         fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            print(f"  led{led}: beam lands {name:6} ({az:5.1f} deg)   "
                  f"code claims {claim:6}   {agree}")
        fig.suptitle(f"{shot} — each LED divided by the 4-LED mean "
                     f"(red = this LED lights it more)", fontsize=12,
                     fontweight="bold")
        fig.tight_layout()
        out = bd.project_path("depth_outputs", "ct_registered",
                              f"{shot}_led_directions.png")
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
