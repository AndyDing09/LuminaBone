#!/usr/bin/env python3
"""
depth_recompute.py -- CORRECTED depth export for the bone endoscope data.
=========================================================================
Fixes the two problems found in review:

 1. The registerable point clouds (ply_allon/*.ply) were built by the weak
    single-image brightness->depth method (endoscope_depth.py), which assumes
    constant albedo and produces a near-flat surface. THIS script instead
    exports the *photometric-stereo* depth (bone_depth_batch.py) -- the same
    method the triangulation slides use, which has real 3D relief.

 2. The reported physical depth was built on guessed constants
    (LIGHT_ELEVATION_DEG=40 and a placeholder field width). Photometric stereo
    is relative-only, so this script reports RELATIVE relief honestly and shows
    how the relief depends on the assumed LED elevation (a sensitivity sweep),
    rather than a single false-precision physical value.

Outputs -> depth_outputs/recomputed/
    locNN_ps.ply            colored point cloud with REAL relief (register this)
    corrected_depth_summary.csv   relief per location + elevation sensitivity

Absolute inches: leave as relative, OR set --field-width-in if you actually
measured the field of view, OR (recommended) recover true scale from the CT
after a similarity registration.

Run:  python depth_recompute.py
"""
import os, csv, argparse, importlib.util, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bd", os.path.join(HERE, "bone_depth_batch.py"))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

OUT = os.path.join(HERE, "depth_outputs", "recomputed")


def ps_depth(files, elevation_deg):
    """Photometric-stereo relative depth for one location at a given LED elevation."""
    idxs = sorted(bd.SINGLE_LED_AZIMUTH_DEG)              # [2,3,4]
    lums = [bd.solve_luminance(bd.load_rgb(files[i], bd.WORK_LONG_EDGE)) for i in idxs]
    lums = bd.balance_exposure(lums)
    az = [bd.SINGLE_LED_AZIMUTH_DEG[i] for i in idxs]
    L = np.array([bd.light_vector(a, elevation_deg) for a in az])
    normals, albedo = bd.photometric_stereo(lums, L)
    z = bd.normals_to_depth(normals)                     # relative, px-height units
    return z, albedo


def relief(z, mask):
    zin = z[mask] if mask.any() else z.ravel()
    lo, hi = np.percentile(zin, [2, 98])
    return float(hi - lo)


def export_ply(path, z, albedo, ref_rgb):
    """Colored point cloud: x,y in px (centered), z = height toward camera (px)."""
    mask = bd.bone_mask(albedo)
    zc = z - (np.median(z[mask]) if mask.any() else np.median(z))
    H, W = z.shape
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    X = (j - W / 2.0).astype(np.float32)
    Y = -(i - H / 2.0).astype(np.float32)
    Z = (-zc).astype(np.float32)                         # toward-camera = up
    keep = mask.ravel()
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)[keep]
    C = (np.clip(ref_rgb, 0, 1).reshape(-1, 3) * 255).astype(np.uint8)[keep]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(P)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, zz), (r, g, b) in zip(P, C):
            f.write(f"{x:.3f} {y:.3f} {zz:.3f} {int(r)} {int(g)} {int(b)}\n")
    return len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-width-in", type=float, default=None,
                    help="real FOV width in inches; if set, adds an honest inch column")
    ap.add_argument("--field-width-mm", type=float, default=None,
                    help="deprecated: real FOV width in mm; converted to inches")
    args = ap.parse_args()
    field_width_in = args.field_width_in
    if field_width_in is None and args.field_width_mm:
        field_width_in = args.field_width_mm / 25.4

    os.makedirs(OUT, exist_ok=True)
    groups = bd.discover(bd.INPUT_FOLDER)
    elevations = [30.0, 40.0, 50.0, 60.0]                # sensitivity sweep
    rows = []
    for loc in sorted(groups):
        files = groups[loc]
        if not all(i in files for i in sorted(bd.SINGLE_LED_AZIMUTH_DEG)):
            print(f"  loc {loc}: SKIP (missing single-LED photo)"); continue
        # main export at the default 40 deg
        z40, alb = ps_depth(files, 40.0)
        mask = bd.bone_mask(alb)
        ref = bd.load_rgb(files.get(bd.ALL_ON_INDEX, sorted(files)[0]), bd.WORK_LONG_EDGE)
        npts = export_ply(os.path.join(OUT, f"loc{loc:02d}_ps.ply"), z40, alb, ref)
        # elevation sensitivity: relief at each assumed elevation
        rel = {e: relief(ps_depth(files, e)[0], mask) for e in elevations}
        W = z40.shape[1]
        row = dict(location=loc, points=npts, W=W,
                   relief_px_elev40=round(rel[40.0], 2),
                   relief_px_elev30=round(rel[30.0], 2),
                   relief_px_elev50=round(rel[50.0], 2),
                   relief_px_elev60=round(rel[60.0], 2))
        inch_suffix = ""
        if field_width_in:
            relief_in = rel[40.0] * field_width_in / W
            row["relief_in_elev40"] = round(relief_in, 4)
            inch_suffix = f" ({relief_in:.4f} in)"
        rows.append(row)
        print(f"  loc {loc:2d}: {npts:>7} pts  relief@40deg={rel[40.0]:6.1f}px  "
              f"{inch_suffix} (range over elev 30-60: {rel[30.0]:.0f}-{rel[60.0]:.0f}px)")
    if rows:
        fields = list(rows[0].keys())
        with open(os.path.join(OUT, "corrected_depth_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} clouds + corrected_depth_summary.csv -> {OUT}")
        print("NOTE: relief is RELATIVE (px). The spread across elevations shows how "
              "much the assumed LED angle inflates/deflates it. Get true inches from "
              "a measured field width, or recover physical scale from CT registration.")


if __name__ == "__main__":
    main()
