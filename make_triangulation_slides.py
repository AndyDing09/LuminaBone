"""
One full "triangulation slide" per location, combining the three figures from
photometric_depth_test.py into a single slide:

    ps_01  -> the THREE single-LED input photos with their light directions
                  p{loc}-2  RIGHT LED
                  p{loc}-3  TOP   LED
                  p{loc}-4  LEFT  LED
    ps_03  -> the depth heatmap those three images calculate
    ps_05  -> the 3-D surface reconstruction of that depth

Uses Woodham photometric stereo (the verified method in bone_depth_batch.py).
Run:  python make_triangulation_slides.py   (or:  py make_triangulation_slides.py)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

import bone_depth_batch as bd

OUT_DIR = "./depth_outputs/triangulation_slides"

LED_NAME = {2: "RIGHT", 3: "UPPER", 4: "LEFT"}
ORDER = [2, 3, 4]                       # px-2, px-3, px-4 left-to-right on the slide


def build_slide(loc, files):
    # --- load the 3 inputs + their light directions ---
    rgbs, lums, Ls = [], [], []
    for i in ORDER:
        rgb = bd.load_rgb(files[i], bd.WORK_LONG_EDGE)   # sRGB, for display
        Ls.append(bd.light_vector(bd.SINGLE_LED_AZIMUTH_DEG[i], bd.LIGHT_ELEVATION_DEG))
        rgbs.append(rgb)
        lums.append(bd.solve_luminance(rgb))             # linear + de-glinted
    lums = bd.balance_exposure(lums)                     # equalise exposure/gain

    # --- triangulate: 3 images -> normals -> depth (away from camera) ---
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))
    z = bd.normals_to_depth(normals)

    # Measure/scale on the bone only (drop dark background + FFT borders).
    mask = bd.bone_mask(albedo)
    zref = z[mask] if mask.any() else z.ravel()
    z = z - np.median(zref)
    lo, hi = np.percentile(z[mask] if mask.any() else z, [2, 98])
    depth_px = float(hi - lo)

    # --- draw the slide ---
    fig = plt.figure(figsize=(14, 9.2), dpi=115)
    fig.patch.set_facecolor("#f5f5f0")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.35],
                          hspace=0.25, wspace=0.45)

    # Top row (ps_01): the three input photos, each labelled with its L vector.
    spans = [(0, 2), (2, 4), (4, 6)]
    for (c0, c1), i, L in zip(spans, ORDER, Ls):
        ax = fig.add_subplot(gs[0, c0:c1])
        ax.imshow(rgbs[ORDER.index(i)])
        ax.set_title(f"p{loc}-{i}.jpg   {LED_NAME[i]} LED\n"
                     f"L = ({L[0]:+.2f}, {L[1]:+.2f}, {L[2]:+.2f})",
                     fontsize=10, fontweight="bold")
        ax.axis("off")

    # Bottom-left (ps_03): depth heatmap.
    axh = fig.add_subplot(gs[1, 0:3])
    im = axh.imshow(z, cmap="turbo", vmin=lo, vmax=hi)
    axh.set_title(f"Depth heatmap  —  peak-to-valley = {depth_px:.1f} px\n"
                  f"(blue = near, red = far)", fontsize=11, fontweight="bold")
    axh.axis("off")
    cb = plt.colorbar(im, ax=axh, shrink=0.82, pad=0.02)
    cb.set_label("relative depth (px)", fontsize=9)

    # Bottom-right (ps_05): 3-D surface, -depth so toward-camera = up.
    from scipy.ndimage import gaussian_filter
    ax3 = fig.add_subplot(gs[1, 3:6], projection="3d")
    zsm = gaussian_filter(z, 1.5)               # light smooth for a clean surface
    zdisp = np.where(mask, -zsm, np.nan)        # draw the bone only (no background)
    H, W = z.shape
    step = max(1, W // 90)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    # turbo_r so the colour matches the heatmap (blue = near) while height still
    # puts near = up: -lo (nearest, top) -> blue, -hi (farthest, bottom) -> red.
    ax3.plot_surface(xx, yy, zdisp[::step, ::step], cmap="turbo_r",
                     vmin=-hi, vmax=-lo, linewidth=0, antialiased=True)
    ax3.set_title("3-D surface reconstruction  (up = toward camera)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("x (px)"); ax3.set_ylabel("y (px)")
    ax3.set_zlabel("height toward camera (px)")
    ax3.view_init(elev=55, azim=-60)

    fig.suptitle(f"Location {loc}  —  three-image photometric triangulation "
                 f"(RIGHT + TOP + LEFT LEDs)",
                 fontsize=14, fontweight="bold", y=0.98)
    out = os.path.join(OUT_DIR, f"loc{loc:02d}_triangulation.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)
    return depth_px


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    n = 0
    for loc in sorted(groups):
        files = groups[loc]
        if not all(i in files for i in ORDER):
            print(f"  [loc {loc}] SKIP - missing one of p{loc}-2/3/4")
            continue
        d = build_slide(loc, files)
        n += 1
        print(f"  loc {loc:2d}: slide saved  (depth = {d:6.2f} px)")
    print(f"\n{n} slides -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
