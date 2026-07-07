"""
Per-location DEPTH ESTIMATE from the three single-LED photos.
================================================================================

For each location it uses exactly three photos -- LEFT LED (pN-4), TOP LED
(pN-3) and RIGHT LED (pN-2) -- and "triangulates" the surface: the three known
light directions fix the surface orientation (normal) at every pixel (Woodham
photometric stereo), and those normals are integrated into a depth map. The
single number reported per location is the surface's peak-to-valley DEPTH.

It reuses the exact, already-verified method in bone_depth_batch.py.

UNITS
--------------------------------------------------------------------------------
The depth map comes out in PIXEL units (depth per pixel-step), which is an
exact, real number for the recovered shape. To express it in inches we multiply
by how many inches one pixel covers on the bone:

        inch_per_pixel = FIELD_WIDTH_IN / image_width_px

So set FIELD_WIDTH_IN to the real width (in inches) of the area each photo
shows and the inch column becomes true physical depth. The pixel-unit column
does NOT depend on that and is always valid for comparing locations.

(Two things set the overall vertical scale: this inch-per-pixel factor and the
assumed LED elevation in bone_depth_batch.py. Relative comparisons between the
17 locations are reliable regardless; only the absolute size needs those.)

RUN
--------------------------------------------------------------------------------
    python bone_depth_values.py
"""

import os
import csv
import numpy as np

import bone_depth_batch as bd     # reuse the verified pipeline


# --- config ---------------------------------------------------------------
INPUT_FOLDER = bd.INPUT_FOLDER          # "./bone_picture"
OUT_CSV = "./depth_outputs/depth_estimates.csv"

# Width (inches) of the area shown in each photo. SET THIS to your real value to
# get true inches. Default preserves the old 30 mm placeholder in inch units;
# the pixel column is exact either way.
FIELD_WIDTH_IN = 30.0 / 25.4


def depth_estimate(files):
    """Run the 3-LED triangulation for one location, return a stats dict."""
    idxs = sorted(bd.SINGLE_LED_AZIMUTH_DEG)        # [2, 3, 4] = right, upper, left
    # Linear luminance + glint removal (solve_luminance), then exposure balance.
    lums = [bd.solve_luminance(bd.load_rgb(files[i], bd.WORK_LONG_EDGE))
            for i in idxs]
    lums = bd.balance_exposure(lums)

    H, W = lums[0].shape
    azimuths = [bd.SINGLE_LED_AZIMUTH_DEG[i] for i in idxs]
    L = np.array([bd.light_vector(a, bd.LIGHT_ELEVATION_DEG) for a in azimuths])

    normals, albedo = bd.photometric_stereo(lums, L)
    z = bd.normals_to_depth(normals)               # depth away from camera

    # Measure relief on the BONE only (exclude dark background + FFT borders).
    mask = bd.bone_mask(albedo)
    zin = z[mask] if mask.any() else z.ravel()
    zin = zin - np.median(zin)                       # center for a clean ptp

    p2, p98 = np.percentile(zin, [2, 98])
    inch = FIELD_WIDTH_IN / W                        # inches per pixel
    return dict(W=W, H=H,
                depth_px=float(p98 - p2),            # headline: robust peak-to-valley
                depth_in=float((p98 - p2) * inch),
                max_protrusion_px=float(zin.max() - zin.min()),
                roughness_px=float(zin.std()))


def main():
    groups = bd.discover(INPUT_FOLDER)
    rows = []
    print(f"\nDepth estimate per location  (LEFT+TOP+RIGHT LEDs, "
          f"FIELD_WIDTH_IN={FIELD_WIDTH_IN:.6f})\n")
    print(f"{'loc':>3} | {'depth (px)':>10} | {'depth (in)':>10} | "
          f"{'max relief (px)':>15} | {'roughness (px)':>14}")
    print("-" * 66)
    for loc in sorted(groups):
        files = groups[loc]
        need = sorted(bd.SINGLE_LED_AZIMUTH_DEG)
        if not all(i in files for i in need):
            print(f"{loc:>3} | SKIP - missing one of the L/T/R photos {need}")
            continue
        s = depth_estimate(files)
        rows.append(dict(location=loc, **s))
        print(f"{loc:>3} | {s['depth_px']:>10.2f} | {s['depth_in']:>10.4f} | "
              f"{s['max_protrusion_px']:>15.2f} | {s['roughness_px']:>14.2f}")

    if rows:
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        fields = ["location", "W", "H", "depth_px", "depth_in",
                  "max_protrusion_px", "roughness_px"]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")
        d = [r["depth_px"] for r in rows]
        print(f"deepest: loc {rows[int(np.argmax(d))]['location']} "
              f"({max(d):.2f} px) | flattest: loc "
              f"{rows[int(np.argmin(d))]['location']} ({min(d):.2f} px)")


if __name__ == "__main__":
    main()
