"""
Verify each LED's true ILLUMINATION-INCIDENCE direction from the images alone.

Why this and not the beam-centroid map (led_wiring_ctfree.py): the centroid of
a single-LED image tells you where that LED's beam LANDS, which confounds LED
position with beam aim. Shading, however, depends on where the light ARRIVES
FROM, i.e. the LED's position. The cleanest position-only cue in these frames
is the shadow each raised retroreflective bead casts on the surrounding bone:
a shadow falls directly AWAY from the source, whatever direction the beam is
aimed.

Method, per marker and per LED:
  1. sample an annulus of bone just outside the bead (the bead itself is
     specular and is excluded),
  2. divide that LED's annulus by the 4-LED mean -- this cancels the bone's
     own albedo and shape, leaving only that LED's illumination signature,
  3. the angle of MINIMUM ratio is the cast shadow -> the light lies at the
     opposite angle; the angle of MAXIMUM ratio is the lit side.
  Both estimates are reported; they should agree to within a few tens of
  degrees and are averaged circularly across markers and shots.

Angles use the project's azimuth convention: 0 = right, 90 = up
(image rows run downward, so azimuth = atan2(-dy_image, dx_image)).

Run: python led_incidence_check.py
"""

import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import calibrate_photometric_ct as C
from nearfield_ct import load_lums, AZ

CLAIMED = AZ                       # {1:90, 2:270, 3:180, 4:0} -- what the code assumes
NAMES = {0.0: "RIGHT", 90.0: "TOP", 180.0: "LEFT", 270.0: "BOTTOM"}
R_IN, R_OUT = 1.35, 2.6            # annulus radii, in bead radii
NBINS = 36                         # 10-degree angular bins


def annulus_profile(img, u, v, bead_r):
    """mean intensity per angular bin in the bone annulus around a bead."""
    H, W = img.shape
    yy, xx = np.mgrid[:H, :W]
    dx = xx - u
    dy = yy - v
    rr = np.hypot(dx, dy)
    sel = (rr >= R_IN * bead_r) & (rr <= R_OUT * bead_r)
    if sel.sum() < 200:
        return None
    ang = (np.degrees(np.arctan2(-dy[sel], dx[sel]))) % 360.0
    val = img[sel]
    prof = np.full(NBINS, np.nan)
    idx = (ang / (360.0 / NBINS)).astype(int) % NBINS
    for b in range(NBINS):
        m = idx == b
        if m.sum() >= 4:
            prof[b] = np.median(val[m])
    return prof


def circ_mean(angles_deg, weights=None):
    a = np.radians(np.asarray(angles_deg, float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    s = np.sum(w * np.sin(a)); c = np.sum(w * np.cos(a))
    return math.degrees(math.atan2(s, c)) % 360.0, math.hypot(s, c) / max(w.sum(), 1e-9)


def bead_radius(shot, u, v, lums):
    """crude bead radius: the specular core is much brighter than the bone."""
    img = np.stack(lums, 0).mean(0)
    H, W = img.shape
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(xx - u, yy - v)
    core = img[rr <= 6]
    thr = 0.5 * (np.median(core) + np.median(img[(rr > 12) & (rr < 25)]))
    for r in range(4, 20):
        ring = img[(rr >= r) & (rr < r + 1)]
        if ring.size and np.median(ring) < thr:
            return float(r)
    return 10.0


def main():
    print("LED incidence direction from cast shadows (position cue, aim-immune)")
    print(f"convention: 0=RIGHT, 90=TOP, 180=LEFT, 270=BOTTOM\n")
    per_led = {s: [] for s in (1, 2, 3, 4)}
    for shot in ["shot_004", "shot_005", "shot_006"]:
        lums, _ = load_lums(shot)
        mean4 = np.mean(np.stack(lums, 0), axis=0) + 1e-9
        corr = C.CT_CORR[shot]
        for mid, (uv, _xyz) in sorted(corr.items()):
            u, v = uv
            br = bead_radius(shot, u, v, lums)
            for k, s in enumerate((1, 2, 3, 4)):
                prof = annulus_profile(lums[k] / mean4, u, v, br)
                if prof is None or np.all(np.isnan(prof)):
                    continue
                bins = (np.arange(NBINS) + 0.5) * (360.0 / NBINS)
                good = ~np.isnan(prof)
                lit = bins[good][np.nanargmax(prof[good])]
                shadow = bins[good][np.nanargmin(prof[good])]
                contrast = (np.nanmax(prof) - np.nanmin(prof)) / np.nanmean(prof)
                per_led[s].append((lit, (shadow + 180.0) % 360.0, contrast))
    print(f"{'LED':5}{'code claims':>14}{'lit-side':>12}{'anti-shadow':>14}"
          f"{'combined':>11}{'consistency':>13}   verdict")
    for s in (1, 2, 3, 4):
        rows = per_led[s]
        if not rows:
            print(f"{s:<5}{'(no data)':>14}")
            continue
        lit = [r[0] for r in rows]; anti = [r[1] for r in rows]
        w = [r[2] for r in rows]
        m_lit, _ = circ_mean(lit, w)
        m_anti, _ = circ_mean(anti, w)
        m_all, R = circ_mean(lit + anti, w + w)
        claim = CLAIMED[s]
        d = min(abs(m_all - claim), 360 - abs(m_all - claim))
        verdict = ("MATCHES" if d < 45 else
                   "OPPOSITE" if d > 135 else "90 deg OFF")
        print(f"{s:<5}{claim:9.0f} {NAMES[claim]:<5}{m_lit:12.0f}{m_anti:14.0f}"
              f"{m_all:11.0f}{R:13.2f}   {verdict} (off by {d:.0f} deg)")
    print("\nconsistency = circular concentration over 12 marker/shot samples "
          "(1.0 = perfect agreement, <0.3 = no reliable direction)")


if __name__ == "__main__":
    main()
