"""
One "marker-depth triangulation slide" per shot in Data_collection/shots.

Mirrors make_triangulation_slides.py, but for the FOUR-LED shot rig
(dark + led1..led4) and it locates the retroreflective trackers and reads off
each one's surface depth. Layout:

    top row     -> the four single-LED input photos with their light directions
    bottom-left -> the photometric-stereo depth heatmap, with every tracker
                   circled and labelled with its relative surface depth
    bottom-right-> the 3-D surface, with a stem at each tracker

METHOD (same Woodham photometric stereo as bone_depth_batch.py, 4 lights):
  1. dark-subtract each LED frame (removes ambient the rig's dark.png measures),
  2. linear luminance + glint removal + per-frame exposure balance,
  3. 4x3 least-squares photometric stereo -> normals -> Frankot-Chellappa depth,
  4. sample the depth at each tracker (found with scripts/tracking).

The four LEDs are the two opposing pairs grab4.py drives (1-3 and 2-4), taken
to sit at right/top/left/bottom around the lens; elevation is the measured
near-field rig angle atan2(working distance, LED offset), same as the 3-LED
triangulation slides. Depth is RELATIVE photometric-stereo shape (uncalibrated
absolute scale), so the meaningful read is which tracker sits nearer or farther
and by how much, not the raw millimetre value.
"""

import os
import sys
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                       # sibling modules
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))

import bone_depth_batch as bd            # shared infra (scripts/reconstruction)
import nearfield_lambertian as nf        # shared infra (scripts/reconstruction)
import detect_trackers as dt             # sibling in marker_pipeline

SHOTS_DIR = bd.project_path("Data_collection", "shots")
OUT_DIR = bd.project_path("depth_outputs", "marker_depth_slides")

# grab4.py drives four LEDs as two opposing pairs (1-3, 2-4). Placed
# symmetrically around the lens: right / top / left / bottom.
LED_NAME = {1: "RIGHT", 2: "TOP", 3: "LEFT", 4: "BOTTOM"}
LED_AZIMUTH_DEG = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}
ORDER = [1, 2, 3, 4]

WORKING_DISTANCE_MM = nf.working_distance_mm()
LIGHT_ELEVATION_DEG = math.degrees(math.atan2(WORKING_DISTANCE_MM,
                                              nf.LED_OFFSET_MM))
MM_PER_IN = 25.4


def undistorted_bgr(path, long_edge=bd.WORK_LONG_EDGE):
    """Same undistort+resize bd.load_rgb applies, but kept as uint8 BGR so the
    tracker detector sees exactly the pixels the depth solve does."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    bgr = bd.undistort_bgr(bgr)
    h, w = bgr.shape[:2]
    if long_edge:
        scale = long_edge / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
    return bgr


def find_markers(bgr_by_led):
    """Detect trackers on each undistorted LED frame and consolidate across the
    four frames (camera + markers are fixed; only the light moves)."""
    rows = []
    for i in ORDER:
        cands = dt.detect(bgr_by_led[i])
        markers = dt.filter_candidates(cands, 130.0, 80.0, 25.0, 45.0,
                                       130.0, 30.0, 145.0)
        for m in markers:
            rows.append(["s", f"led{i}", 0, m["x"], m["y"], m["r"], 0, 0, 0, 0])
    agg = dt.aggregate_by_shot(rows, merge_dist=35)   # [shot,id,x,y,r,n_frames]
    return [(a[2], a[3], a[4], a[5]) for a in agg]     # (x, y, r, n_frames)


def marker_depth(z, x, y, r):
    """Relative surface depth at a tracker: median z over its disk. The disk is
    specular so solve_luminance has inpainted it from the surrounding bone, so
    this reads the local seat depth rather than the (non-Lambertian) bead."""
    H, W = z.shape
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= (max(r, 6)) ** 2
    return float(np.median(z[disk])) if disk.any() else float("nan")


def build_slide(shot, files):
    dark = bd.load_rgb(files["dark"], bd.WORK_LONG_EDGE) if "dark" in files else None

    rgbs, lums, Ls = [], [], []
    bgr_by_led = {}
    for i in ORDER:
        rgb = bd.load_rgb(files[i], bd.WORK_LONG_EDGE)          # for display
        bgr_by_led[i] = undistorted_bgr(files[i])              # for detection
        corr = np.clip(rgb - dark, 0, 1) if dark is not None else rgb
        Ls.append(bd.light_vector(LED_AZIMUTH_DEG[i], LIGHT_ELEVATION_DEG))
        rgbs.append(rgb)
        lums.append(bd.solve_luminance(corr))
    lums = bd.balance_exposure(lums)

    # 4 images -> normals -> relative depth (away from camera).
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))
    z = bd.normals_to_depth(normals)

    mask = bd.bone_mask(albedo)
    zref = z[mask] if mask.any() else z.ravel()
    z = z - np.median(zref)
    lo, hi = np.percentile(z[mask] if mask.any() else z, [2, 98])

    # Physical scale: relative mm across the measured field width.
    W = z.shape[1]
    mm_per_px = nf.FIELD_WIDTH_MM / W
    z_mm = z * mm_per_px
    lo_mm, hi_mm = lo * mm_per_px, hi * mm_per_px

    markers = find_markers(bgr_by_led)
    mdepth = [marker_depth(z_mm, x, y, r) for (x, y, r, _n) in markers]
    # Full (x, y, z) per tracker, in mm across the measured field. x runs right,
    # y runs down (image frame, matching the 3-D axes); z is relative depth,
    # + = farther from the camera.
    coords = [(x * mm_per_px, y * mm_per_px, d)
              for (x, y, r, _n), d in zip(markers, mdepth)]

    # ---------------- draw ----------------
    fig = plt.figure(figsize=(14, 9.9), dpi=115)
    fig.patch.set_facecolor("#f5f5f0")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.32, 1.4],
                          hspace=0.30, wspace=0.30)

    for c, i, L in zip(range(4), ORDER, Ls):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(rgbs[c])
        ax.set_title(f"led{i}.png   {LED_NAME[i]}\n"
                     f"L=({L[0]:+.2f},{L[1]:+.2f},{L[2]:+.2f})",
                     fontsize=9, fontweight="bold")
        ax.axis("off")

    # coordinate table (id, x, y, z in mm) spanning the full width
    axt = fig.add_subplot(gs[1, 0:4])
    axt.axis("off")
    if coords:
        cell_text = [[str(k), f"{cx:.1f}", f"{cy:.1f}", f"{cz:+.2f}"]
                     for k, (cx, cy, cz) in enumerate(coords, 1)]
        tbl = axt.table(
            cellText=cell_text,
            colLabels=["tracker", "x (mm)", "y (mm)", "z (mm)"],
            colColours=["#d8d8d0"] * 4, cellLoc="center", loc="center",
            bbox=[0.18, 0.0, 0.64, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (rr, _cc), cell in tbl.get_celld().items():
            cell.set_edgecolor("#b0b0a8")
            if rr == 0:
                cell.set_text_props(fontweight="bold")
    axt.set_title("Tracker coordinates  (x = right, y = down, "
                  "z = relative depth, + = farther)",
                  fontsize=10, fontweight="bold", pad=2)

    # depth heatmap + trackers
    axh = fig.add_subplot(gs[2, 0:2])
    im = axh.imshow(z_mm, cmap="turbo", vmin=lo_mm, vmax=hi_mm)
    for k, (x, y, r, _n) in enumerate(markers, 1):
        axh.add_patch(plt.Circle((x, y), max(r, 8) + 3, fill=False,
                                 edgecolor="white", linewidth=1.8))
        axh.plot(x, y, "+", color="white", ms=6, mew=1.4)
        axh.annotate(str(k), (x, y), (x + max(r, 8) + 5, y),
                     color="white", fontsize=10, fontweight="bold",
                     ha="left", va="center",
                     bbox=dict(boxstyle="circle,pad=0.15", fc="black", alpha=0.6,
                               ec="white", lw=0.8))
    axh.set_title("Depth heatmap with trackers  (blue = near, red = far)",
                  fontsize=11, fontweight="bold")
    axh.axis("off")
    cb = plt.colorbar(im, ax=axh, shrink=0.82, pad=0.02)
    cb.set_label("relative depth z (mm)", fontsize=9)

    # 3-D surface + tracker stems
    ax3 = fig.add_subplot(gs[2, 2:4], projection="3d")
    zsm = gaussian_filter(z_mm, 1.5)
    zdisp = np.clip(-zsm, -hi_mm, -lo_mm)
    H, Wd = z_mm.shape
    step = max(1, Wd // 160)
    yy, xx = np.mgrid[0:H:step, 0:Wd:step]
    ax3.plot_surface(xx * mm_per_px, yy * mm_per_px, zdisp[::step, ::step],
                     cmap="turbo_r", vmin=-hi_mm, vmax=-lo_mm,
                     rcount=zdisp[::step, ::step].shape[0],
                     ccount=zdisp[::step, ::step].shape[1],
                     linewidth=0, antialiased=True, alpha=0.9)
    for k, ((x, y, r, _n), d) in enumerate(zip(markers, mdepth), 1):
        hx, hy = x * mm_per_px, y * mm_per_px
        top = np.clip(-d, -hi_mm, -lo_mm)
        ax3.plot([hx, hx], [hy, hy], [-hi_mm, top], color="k", lw=1.0)
        ax3.scatter([hx], [hy], [top], color="magenta", s=28,
                    edgecolor="k", depthshade=False)
        ax3.text(hx, hy, top, f" {k}", fontsize=8, fontweight="bold")
    ax3.set_title("3-D surface  (up = toward camera)  +  trackers",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("x (mm)"); ax3.set_ylabel("y (mm)")
    ax3.set_zlabel("height toward camera (mm)")
    ax3.view_init(elev=55, azim=-60)

    fig.suptitle(f"{shot}  —  four-LED photometric triangulation: "
                 f"tracker depth  ({len(markers)} markers)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             f"Measured geometry: {nf.FIELD_WIDTH_MM:.0f} mm field of view, LEDs "
             f"{nf.LED_OFFSET_MM:.2f} mm off-axis at {WORKING_DISTANCE_MM:.0f} mm "
             f"-> {LIGHT_ELEVATION_DEG:.0f}° elevation. Relative photometric-stereo "
             "depth (uncalibrated absolute scale): compare trackers, not raw mm.",
             ha="center", fontsize=8, style="italic", color="#555555")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{shot}_marker_depth.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)
    return markers, mdepth, coords, out


def discover_shots(folder):
    """{shot_name: {1..4: ledpath, 'dark': darkpath}} for shots with all 4 LEDs."""
    groups = {}
    for name in sorted(os.listdir(folder)):
        d = os.path.join(folder, name)
        if not os.path.isdir(d):
            continue
        files = {}
        for i in ORDER:
            p = os.path.join(d, f"led{i}.png")
            if os.path.exists(p):
                files[i] = p
        dark = os.path.join(d, "dark.png")
        if os.path.exists(dark):
            files["dark"] = dark
        if all(i in files for i in ORDER):
            groups[name] = files
    return groups


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="*", default=None,
                    help="specific shot names, e.g. shot_009 (default: all)")
    a = ap.parse_args()

    groups = discover_shots(SHOTS_DIR)
    if a.shots:
        groups = {k: v for k, v in groups.items() if k in a.shots}
    if not groups:
        print("no shots with led1..led4 found")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    import csv
    rows = []
    for shot in sorted(groups):
        markers, mdepth, coords, out = build_slide(shot, groups[shot])
        depths = ", ".join(f"{k}:{d:+.2f}" for k, d in enumerate(mdepth, 1))
        print(f"  {shot}: {len(markers)} markers  depth(mm) [{depths}]  -> "
              f"{os.path.basename(out)}")
        for k, ((x, y, r, n), (cx, cy, cz)) in enumerate(
                zip(markers, coords), 1):
            rows.append([shot, k, x, y, round(cx, 2), round(cy, 2),
                         round(cz, 3), r, n])
    csv_path = os.path.join(OUT_DIR, "marker_depths.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["shot", "id", "x_px", "y_px", "x_mm", "y_mm", "z_mm",
                     "r_px", "n_frames"])
        wr.writerows(rows)
    print(f"\nslides + {os.path.basename(csv_path)} -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
