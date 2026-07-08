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

LIGHT GEOMETRY (measured rig, not the old empirical shading fit):
The three LEDs are identical (equal brightness) and mounted symmetrically at
the same measured LED_OFFSET_MM = 12.08 mm from the lens center, so their
in-plane azimuths are RIGHT=0, TOP=90, LEFT=180 deg. Since every LED sits the
same distance off-axis and the bone is at the working distance Z implied by the
calibrated fx and the 37 mm field, all three share one elevation

    elevation = atan2(Z, LED_OFFSET_MM)   (~75 deg here)

measured directly from the rig rather than the previous 40 deg guess. This is
the light-direction input the triangulation solve depends on.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

import bone_depth_batch as bd
import nearfield_lambertian as nf        # measured LED_OFFSET_MM, working distance

OUT_DIR = bd.project_path("depth_outputs", "triangulation_slides")

LED_NAME = {2: "RIGHT", 3: "TOP", 4: "LEFT"}
ORDER = [2, 3, 4]                       # px-2, px-3, px-4 left-to-right on the slide

# Corrected, symmetric, equal-LED geometry (right / top / left).
LED_AZIMUTH_DEG = {2: 0.0, 3: 90.0, 4: 180.0}
# One elevation for all three: they are equidistant from the lens, so the
# light ray's lift above the surface plane is set by working distance / offset.
WORKING_DISTANCE_MM = nf.working_distance_mm()
LIGHT_ELEVATION_DEG = math.degrees(math.atan2(WORKING_DISTANCE_MM,
                                              nf.LED_OFFSET_MM))


def build_slide(loc, files):
    # --- load the 3 inputs + their light directions ---
    rgbs, lums, Ls = [], [], []
    for i in ORDER:
        rgb = bd.load_rgb(files[i], bd.WORK_LONG_EDGE)   # sRGB, for display
        Ls.append(bd.light_vector(LED_AZIMUTH_DEG[i], LIGHT_ELEVATION_DEG))
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

    # Physical units: FIELD_WIDTH_IN is the real width of the imaged area, so
    # one working pixel = FIELD_WIDTH_IN / W inches (same conversion as the
    # relief_in_p2_p98 column in depth_summary.csv). Falls back to px if unset.
    if bd.FIELD_WIDTH_IN:
        unit_scale = bd.FIELD_WIDTH_IN / z.shape[1]
        unit = "in"
        fmt = "{:.3f}"
    else:
        unit_scale = 1.0
        unit = "px"
        fmt = "{:.1f}"
    z = z * unit_scale
    lo, hi = lo * unit_scale, hi * unit_scale
    depth_val = float(hi - lo)

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
    axh.set_title(f"Depth heatmap  —  peak-to-valley = {fmt.format(depth_val)} {unit}\n"
                  f"(blue = near, red = far)", fontsize=11, fontweight="bold")
    axh.axis("off")
    cb = plt.colorbar(im, ax=axh, shrink=0.82, pad=0.02)
    cb.set_label(f"relative depth ({unit})", fontsize=9)

    # Bottom-right (ps_05): 3-D surface, -depth so toward-camera = up.
    from scipy.ndimage import gaussian_filter
    ax3 = fig.add_subplot(gs[1, 3:6], projection="3d")
    zsm = gaussian_filter(z, 1.5)               # light smooth for a clean surface
    # Continuous, hole-free surface: render the FULL depth field (no mask
    # cut-outs) and clamp heights to the bone's [lo, hi] range so border and
    # background integration artifacts plateau instead of spiking the plot.
    zdisp = np.clip(-zsm, -hi, -lo)
    H, W = z.shape
    step = max(1, W // 160)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    zgrid = zdisp[::step, ::step]
    # turbo_r so the colour matches the heatmap (blue = near) while height still
    # puts near = up: -lo (nearest, top) -> blue, -hi (farthest, bottom) -> red.
    # x/y are scaled to the same physical unit as depth so axes are comparable.
    # rcount/ccount keep the full grid (matplotlib decimates to 50x50 otherwise).
    ax3.plot_surface(xx * unit_scale, yy * unit_scale, zgrid,
                     cmap="turbo_r", vmin=-hi, vmax=-lo,
                     rcount=zgrid.shape[0], ccount=zgrid.shape[1],
                     linewidth=0, antialiased=True)
    ax3.set_title("3-D surface reconstruction  (up = toward camera)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel(f"x ({unit})"); ax3.set_ylabel(f"y ({unit})")
    ax3.set_zlabel(f"height toward camera ({unit})")
    ax3.view_init(elev=55, azim=-60)

    fig.suptitle(f"Location {loc}  —  three-image photometric triangulation "
                 f"(RIGHT + TOP + LEFT LEDs)",
                 fontsize=14, fontweight="bold", y=0.98)
    if unit == "in":
        fig.text(0.5, 0.005,
                 f"Measured geometry: {bd.FIELD_WIDTH_IN * 25.4:.0f} mm field of "
                 f"view, LEDs {nf.LED_OFFSET_MM:.2f} mm off-axis at "
                 f"{WORKING_DISTANCE_MM:.0f} mm -> {LIGHT_ELEVATION_DEG:.0f}° "
                 "elevation; relative depth, scaled to the field width.",
                 ha="center", fontsize=8, style="italic", color="#555555")
    out = os.path.join(OUT_DIR, f"loc{loc:02d}_triangulation.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)
    return depth_val, unit


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    n = 0
    for loc in sorted(groups):
        files = groups[loc]
        if not all(i in files for i in ORDER):
            print(f"  [loc {loc}] SKIP - missing one of p{loc}-2/3/4")
            continue
        d, unit = build_slide(loc, files)
        n += 1
        print(f"  loc {loc:2d}: slide saved  (depth = {d:8.4f} {unit})")
    print(f"\n{n} slides -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
