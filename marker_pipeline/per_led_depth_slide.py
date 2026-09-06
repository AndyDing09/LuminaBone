"""
Per-LED depth: an inverse-square depth map from EACH single LED frame.

Photometric stereo needs >= 3 lights, so a single frame supports only the
intensity cue: brighter = nearer, r ~ 1/sqrt(I). Computing it once per LED
shows how much a single-image depth estimate depends on WHICH light was on --
each map tilts toward its own LED, which is exactly the directional
information photometric stereo exploits and single-image depth cannot separate
from albedo.

Honest caveat rendered on the figure: measured against CT at the fiducials,
inverse-square depth correlates only r = +0.20 (near-field PS: +0.99), and its
leave-one-out error is 12.6 mm (near-field: 5.1 mm). The structure is real
shading; the depth values are not reliable.

All four panels share one colour scale so they can be compared directly, and
all are masked to bone (the background carries no valid depth).

Run: python per_led_depth_slide.py [shot_004]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import nearfield_lambertian as nf
import calibrate_photometric_ct as C
import marker_depth_slides as S

# direction each LED is MEASURED to light (led_direction_figure.py, 3 shots):
# the code's LED_NAME disagrees on the vertical pair, so label what we measured.
MEASURED = {1: "lights BOTTOM", 2: "lights TOP",
            3: "lights LEFT", 4: "lights RIGHT"}


def main():
    shot = next((a for a in sys.argv[1:] if a.startswith("shot_")), "shot_004")
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    rgbs, lums = [], []
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        rgbs.append(rgb)
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
    lums = bd.balance_exposure(lums)
    mask = bd.bone_mask(np.stack(lums, 0).mean(0))
    mm_per_px = nf.FIELD_WIDTH_MM / lums[0].shape[1]

    # Inverse-square depth per LED. Two corrections are needed before the four
    # maps can share a colour scale:
    #  (1) floor the intensity at 5% of the bone median -- 1/sqrt(I) diverges in
    #      the darkest pixels (raw led4 ran to +58 mm while led1 spanned 0.06);
    #  (2) each LED has its own power/exposure, so r = k_s/sqrt(I_s) carries an
    #      unknown per-LED scale k_s. Normalising each map by its own robust
    #      spread (IQR over bone) puts them in comparable units, which is the
    #      most an uncalibrated single-image cue can honestly claim.
    zs = []
    for L in lums:
        floor = 0.05 * np.median(L[mask])
        z = 1.0 / np.sqrt(np.maximum(L, floor))
        z = z - np.median(z[mask])
        q1, q3 = np.percentile(z[mask], [25, 75])
        z = z / max(q3 - q1, 1e-9)
        zs.append(np.where(mask, z, np.nan))
    stack = np.concatenate([z[mask] for z in zs])
    lo, hi = np.percentile(stack, [3, 97])

    markers = S.drop_other_segment(
        shot, S.find_markers({i: S.undistorted_bgr(os.path.join(d, f"led{i}.png"))
                              for i in S.ORDER}))

    fig, axs = plt.subplots(2, 4, figsize=(18, 8.4),
                            gridspec_kw=dict(height_ratios=[1.0, 1.35]))
    fig.patch.set_facecolor("#f5f5f0")
    for c in range(4):
        axs[0, c].imshow(rgbs[c])
        axs[0, c].set_title(f"led{c+1}.png   ({MEASURED[c+1]})",
                            fontsize=11, fontweight="bold")
        axs[0, c].axis("off")

        im = axs[1, c].imshow(zs[c], cmap="turbo", vmin=lo, vmax=hi)
        for k, (x, y, r, _n) in enumerate(markers, 1):
            axs[1, c].add_patch(plt.Circle((x, y), max(r, 8) + 3, fill=False,
                                           edgecolor="white", linewidth=1.6))
            axs[1, c].plot(x, y, "+", color="white", ms=6, mew=1.3)
            axs[1, c].annotate(str(k), (x, y), (x + max(r, 8) + 5, y),
                               color="white", fontsize=9, fontweight="bold",
                               ha="left", va="center",
                               bbox=dict(boxstyle="circle,pad=0.14", fc="black",
                                         alpha=0.6, ec="white", lw=0.7))
        vals = [float(np.nanmedian(zs[c][max(0, int(y) - 6):int(y) + 7,
                                         max(0, int(x) - 6):int(x) + 7]))
                for (x, y, r, _n) in markers]
        axs[1, c].set_title("depth from this LED alone\n" +
                            "  ".join(f"{k}:{v:+.1f}"
                                      for k, v in enumerate(vals, 1)),
                            fontsize=10)
        axs[1, c].axis("off")

    cb = fig.colorbar(im, ax=axs[1, :].tolist(), shrink=0.85, pad=0.012)
    cb.set_label("relative depth (normalised per LED), blue = near",
                 fontsize=10)
    fig.suptitle(f"{shot} - single-image (inverse-square) depth from each LED "
                 f"separately", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "Each map is r ~ 1/sqrt(I) from ONE frame, masked to bone, shared "
             "colour scale. Note how the estimate tilts toward whichever LED "
             "was lit - a single image cannot separate distance from surface "
             "tilt or albedo. Against CT at the fiducials these score r = "
             "+0.20 (near-field photometric stereo: +0.99); leave-one-out "
             "error 12.6 mm vs 5.1 mm. Structure is real shading; the depth "
             "values are not.",
             ha="center", fontsize=8.5, style="italic", color="#555555",
             wrap=True)
    out = bd.project_path("depth_outputs", "calib_charuco_slides",
                          f"{shot}_per_led_depth.png")
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="#f5f5f0")
    print("wrote", out)
    for c in range(4):
        v = zs[c][mask]
        print(f"  led{c+1}: normalised depth span over bone "
              f"{np.nanpercentile(v,3):+.2f} .. {np.nanpercentile(v,97):+.2f}")


if __name__ == "__main__":
    main()
