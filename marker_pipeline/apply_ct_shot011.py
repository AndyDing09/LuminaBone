"""
Apply the CT-anchored depth correction to shot_011 and emit a corrected slide.

Pipeline:
  1. recompute shot_011's far-field marker-depth map (same as marker_depth_slides),
  2. read the detected markers + relative depths (marker_depths.csv),
  3. PnP-align the CT markers into the CAMERA frame (marker pixels + intrinsics),
  4. fit the affine z-correction z_corr = a*z_est + b against the camera-frame CT
     depths, apply it to every marker,
  5. render a slide: corrected depth heatmap + a marker table (mine / corrected /
     CT / residual), with an honest caveat.

RESULT (honest): with the CONFIRMED pose (marker 4 closest), the far-field
photometric depth turns out ANTI-correlated with CT -- the affine slope is
NEGATIVE and even the sign-flipping fit only reaches ~4 mm RMS on an 11 mm depth
span. So the far-field marker depth here is essentially inverted/unusable; this
script documents that rather than producing a trustworthy correction. The real
fix is the calibrated near-field solve (nearfield_lambertian.py), which needs the
LED calibration captures that do not yet exist.
"""

import os
import sys
import math
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd            # shared infra (scripts/reconstruction)
import nearfield_lambertian as nf        # shared infra (scripts/reconstruction)

SHOT = "shot_011"
SHOT_DIR = bd.project_path("Data_collection", "shots", SHOT)
CSV = bd.project_path("depth_outputs", "marker_depth_slides", "marker_depths.csv")
OUT = bd.project_path("depth_outputs", "marker_depth_slides",
                      f"{SHOT}_ct_corrected.png")

LED_AZ = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}
ORDER = [1, 2, 3, 4]
ELEV = math.degrees(math.atan2(nf.working_distance_mm(), nf.LED_OFFSET_MM))

# CT ground truth (CT frame, mm) for shot_011 markers 1, 3, 4.
CT = {1: (9.224, -1.632, 11.818),
      3: (3.729, -13.209, 23.259),
      4: (-17.665, 0.732, 17.412)}


def far_field_depth_map():
    """Recompute shot_011's relative depth map (mm), same as marker_depth_slides."""
    dark = bd.load_rgb(os.path.join(SHOT_DIR, "dark.png"), bd.WORK_LONG_EDGE)
    lums, Ls, rgbs = [], [], []
    for i in ORDER:
        rgb = bd.load_rgb(os.path.join(SHOT_DIR, f"led{i}.png"), bd.WORK_LONG_EDGE)
        corr = np.clip(rgb - dark, 0, 1)
        lums.append(bd.solve_luminance(corr))
        Ls.append(bd.light_vector(LED_AZ[i], ELEV))
        rgbs.append(rgb)
    lums = bd.balance_exposure(lums)
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))
    z = bd.normals_to_depth(normals)
    mask = bd.bone_mask(albedo)
    z = z - np.median(z[mask] if mask.any() else z)
    mm_per_px = nf.FIELD_WIDTH_MM / z.shape[1]
    return z * mm_per_px, rgbs[0], mask


def load_markers():
    """{id: dict(x_px, y_px, z_est)} for shot_011 from marker_depths.csv."""
    m = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["shot"] == SHOT:
                m[int(r["id"])] = dict(x_px=float(r["x_px"]), y_px=float(r["y_px"]),
                                       z_est=float(r["z_mm"]))
    return m


def ct_camera_depths(markers):
    """PnP the CT markers into the camera frame; return {id: Z_cam} and the
    median working distance. Marker pixels come from the detections."""
    ids = sorted(CT)
    img = np.array([[markers[i]["x_px"], markers[i]["y_px"]] for i in ids])
    obj = np.array([CT[i] for i in ids])
    fx, fy, cx, cy = nf.measured_intrinsics(640, 480)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    _, rvecs, tvecs = cv2.solveP3P(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_AP3P)
    # P3P is ambiguous (up to 4 poses) with 3 points. Disambiguate with the
    # CONFIRMED physical fact that marker 4 is the closest to the lens.
    Zc = None
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        cam = (R @ obj.T + tv).T
        if (cam[:, 2] > 0).all():
            dist = {i: float(np.linalg.norm(c)) for i, c in zip(ids, cam)}
            if min(dist, key=dist.get) == 4:
                Zc = cam[:, 2]
    if Zc is None:
        raise RuntimeError("no pose with marker 4 closest; check inputs")
    return {i: float(z) for i, z in zip(ids, Zc)}, float(np.median(Zc))


def main():
    z_mm, ref_rgb, mask = far_field_depth_map()
    markers = load_markers()
    ct_cam, wd = ct_camera_depths(markers)

    # camera-frame CT depth relative to its median (matches z_est convention)
    ct_med = np.median(list(ct_cam.values()))
    ct_rel = {i: ct_cam[i] - ct_med for i in ct_cam}

    # affine fit z_true(cam,rel) = a*z_est + b over the CT markers
    ids_ct = sorted(CT)
    ze = np.array([markers[i]["z_est"] for i in ids_ct])
    zt = np.array([ct_rel[i] for i in ids_ct])
    A = np.column_stack([ze, np.ones_like(ze)])
    (a, b), *_ = np.linalg.lstsq(A, zt, rcond=None)

    # apply to every marker + the whole map (relative)
    zc_map = a * z_mm + b
    for i in markers:
        markers[i]["z_corr"] = a * markers[i]["z_est"] + b

    # ---------- slide ----------
    fig = plt.figure(figsize=(13, 6.6), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.15)

    lo, hi = np.percentile(zc_map[mask] if mask.any() else zc_map, [2, 98])
    axh = fig.add_subplot(gs[0, 0])
    im = axh.imshow(zc_map, cmap="turbo", vmin=lo, vmax=hi)
    for i in sorted(markers):
        m = markers[i]
        axh.add_patch(plt.Circle((m["x_px"], m["y_px"]), 16, fill=False,
                                 edgecolor="white", lw=1.8))
        axh.annotate(str(i), (m["x_px"], m["y_px"]), (m["x_px"] + 20, m["y_px"]),
                     color="white", fontsize=10, fontweight="bold", va="center",
                     bbox=dict(boxstyle="circle,pad=0.15", fc="black", alpha=0.6,
                               ec="white", lw=0.8))
    axh.set_title(f"{SHOT}: CT-anchored depth (markers), turbo = corrected mm",
                  fontsize=11, fontweight="bold")
    axh.axis("off")
    plt.colorbar(im, ax=axh, shrink=0.8, pad=0.02).set_label(
        "corrected relative depth (mm)", fontsize=9)

    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    header = ["id", "z_est", "z_corr", "CT (cam)", "resid"]
    cell = []
    for i in sorted(markers):
        m = markers[i]
        if i in ct_rel:
            cell.append([str(i), f"{m['z_est']:+.2f}", f"{m['z_corr']:+.2f}",
                         f"{ct_rel[i]:+.2f}", f"{m['z_corr'] - ct_rel[i]:+.2f}"])
        else:
            cell.append([str(i), f"{m['z_est']:+.2f}", f"{m['z_corr']:+.2f}",
                         "-- (no CT)", "--"])
    tbl = axt.table(cellText=cell, colLabels=header, cellLoc="center",
                    colColours=["#d8d8d0"] * 5, bbox=[0.0, 0.55, 1.0, 0.40])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (rr, _c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#b0b0a8")
        if rr == 0:
            cellobj.set_text_props(fontweight="bold")
    rms = np.sqrt(np.mean([(markers[i]["z_corr"] - ct_rel[i]) ** 2
                           for i in ids_ct]))
    axt.text(0.0, 0.42, f"correction:  z_corr = {a:.3f}·z_est {b:+.3f}   (mm, "
             f"camera frame)\ncontrol-point RMS: {rms:.2f} mm   ·   working "
             f"dist (PnP): {wd:.0f} mm", fontsize=9, va="top", family="monospace")
    axt.text(0.0, 0.20,
             "Anchored to CT at markers 1 & 3 (marker 4 checks it). All depths\n"
             "are RELATIVE to the marker-set median. The heatmap is the affine-\n"
             "rescaled far-field map: marker numbers are CT-anchored (~2 mm), but\n"
             "the surface still carries the far-field bowl (amplified ~3x by the\n"
             "rescale). A true surface fix needs the near-field solve + LED calib.",
             fontsize=8.5, va="top", style="italic", color="#555555")

    fig.suptitle(f"{SHOT} — CT-anchored depth correction", fontsize=14,
                 fontweight="bold", y=0.99)
    fig.savefig(OUT, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)

    # corrected CSV
    out_csv = OUT.replace(".png", ".csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_px", "y_px", "z_est_mm", "z_corr_mm",
                    "ct_cam_rel_mm", "residual_mm"])
        for i in sorted(markers):
            m = markers[i]
            ctv = f"{ct_rel[i]:.3f}" if i in ct_rel else ""
            resid = f"{m['z_corr'] - ct_rel[i]:.3f}" if i in ct_rel else ""
            w.writerow([i, round(m["x_px"], 1), round(m["y_px"], 1),
                        round(m["z_est"], 3), round(m["z_corr"], 3), ctv, resid])

    print(f"correction  z_corr = {a:.3f}*z_est {b:+.3f}   RMS {rms:.2f} mm")
    for i in sorted(markers):
        m = markers[i]
        tag = f"  CT {ct_rel[i]:+.2f}" if i in ct_rel else "  (no CT)"
        print(f"  marker {i}: z_est {m['z_est']:+.2f} -> z_corr "
              f"{m['z_corr']:+.2f} mm{tag}")
    print(f"\nslide -> {OUT}\ncsv   -> {out_csv}")


if __name__ == "__main__":
    main()
