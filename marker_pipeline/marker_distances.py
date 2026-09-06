"""
Distance-from-lens for the shot markers (shots 10-15).

Two very different reliabilities, and this script keeps them clearly separate:

  CT-validated (shot_011 only): PnP-align the CT markers into the camera frame
      (marker pixels + intrinsics), pick the pose matching the KNOWN physical
      order (marker 4 closest), and read the Euclidean lens-to-marker distance.

  Photometric estimate (all other shots): there is NO CT, so distance can only
      come from the far-field marker depth. BUT at shot_011 -- the one shot we
      can check -- that depth came out ANTI-correlated with CT (corr -0.92, order
      inverted). So these numbers are UNRELIABLE and flagged as such; they are
      emitted only so the raw values are on record, not because they are trusted.

Absolute scale is additionally uncertain by ~1.5x because the focal length is
unresolved (measured_intrinsics fx=519 vs checkerboard fx=794 give a 1.5x range).

Bottom line: only shot_011 has a real distance. Every other shot needs its own CT
markers, or the calibrated near-field solve, before its distances mean anything.

Run:  python marker_distances.py
"""

import os
import sys
import csv
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd            # shared infra (scripts/reconstruction)
import nearfield_lambertian as nf        # shared infra (scripts/reconstruction)

CSV_IN = bd.project_path("depth_outputs", "marker_depth_slides", "marker_depths.csv")
CSV_OUT = bd.project_path("depth_outputs", "marker_depth_slides",
                          "marker_distances.csv")

# CT ground control, shot_011 only (CT frame, mm). Marker 2 has no CT point.
CT_SHOT011 = {1: (9.224, -1.632, 11.818),
              3: (3.729, -13.209, 23.259),
              4: (-17.665, 0.732, 17.412)}
# Physical order you confirmed: marker 4 is the closest to the lens.
CLOSEST_MARKER = 4

# Anchor for the (unreliable) photometric estimate: the shot_011 CT marker-set
# median distance (~46 mm with measured intrinsics). Better than the 30 mm bench
# number, but it is one data point from one shot -- do not read too much into it.
WD_ANCHOR_MM = 46.0


def load_markers():
    shots = {}
    with open(CSV_IN) as f:
        for r in csv.DictReader(f):
            shots.setdefault(r["shot"], {})[int(r["id"])] = dict(
                u=float(r["x_px"]), v=float(r["y_px"]), z=float(r["z_mm"]))
    return shots


def ct_distances(markers, K):
    """PnP the shot_011 CT markers, choosing the pose whose closest marker is
    CLOSEST_MARKER. Returns {id: euclidean_distance_mm}."""
    ids = sorted(CT_SHOT011)
    img = np.array([[markers[i]["u"], markers[i]["v"]] for i in ids])
    obj = np.array([CT_SHOT011[i] for i in ids])
    _, rvecs, tvecs = cv2.solveP3P(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                   K, None, flags=cv2.SOLVEPNP_AP3P)
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        cam = (R @ obj.T + tv).T
        if (cam[:, 2] > 0).all():
            dist = {i: float(np.linalg.norm(c)) for i, c in zip(ids, cam)}
            if min(dist, key=dist.get) == CLOSEST_MARKER:
                return dist
    return None


def photometric_distance(u, v, z_rel, fx, fy, cx, cy):
    """UNRELIABLE estimate: anchor the bone median at WD_ANCHOR, add the (far-
    field, possibly sign-inverted) relative depth, backproject to a range."""
    Z = WD_ANCHOR_MM + z_rel
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    return Z * np.linalg.norm(ray)


def main():
    shots = load_markers()
    fx, fy, cx, cy = nf.measured_intrinsics(640, 480)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    ct = ct_distances(shots["shot_011"], K) if "shot_011" in shots else None
    rows = []
    for shot in [f"shot_{n:03d}" for n in range(10, 16)]:
        if shot not in shots:
            continue
        for i in sorted(shots[shot]):
            m = shots[shot][i]
            if shot == "shot_011" and ct and i in ct:
                rows.append([shot, i, round(m["u"], 1), round(m["v"], 1),
                             round(m["z"], 2), "CT (validated)",
                             round(ct[i], 1), ""])
            else:
                d = photometric_distance(m["u"], m["v"], m["z"], fx, fy, cx, cy)
                rows.append([shot, i, round(m["u"], 1), round(m["v"], 1),
                             round(m["z"], 2),
                             "photometric (UNRELIABLE: inverted vs CT@011)",
                             round(d, 1), "far-field depth; scale/sign untrusted"])

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot", "id", "u_px", "v_px", "z_rel_mm", "method",
                    "distance_from_lens_mm", "note"])
        w.writerows(rows)

    # console summary
    print(f"intrinsics: measured (fx={fx:.0f}); absolute scale uncertain ~1.5x "
          "(fx unresolved)\n")
    cur = None
    for r in rows:
        if r[0] != cur:
            cur = r[0]
            tag = "  [CT-VALIDATED]" if cur == "shot_011" else \
                  "  [UNRELIABLE - photometric only, inverted at shot_011]"
            print(f"{cur}{tag}")
        print(f"    marker {r[1]}: {r[6]:6.1f} mm from lens   ({r[5]})")
    print(f"\nwrote {len(rows)} rows -> {CSV_OUT}")
    print("\nONLY shot_011 is real. The rest need per-shot CT or the calibrated "
          "near-field solve before the distances mean anything.")


if __name__ == "__main__":
    main()
