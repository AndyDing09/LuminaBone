#!/usr/bin/env python3
"""Detect retroreflective trackers in the LED shot images.

A retroreflective marker in this rig looks like a roughly circular disk with a
high-frequency "sparkle" texture (the glass-bead coating), usually rimmed by a
dark ring against the smoother bone surface. We find circular candidates and
keep the ones that (a) sparkle inside and (b) sit inside a darker annulus.

Outputs, per image:
  <img>_trackers.png   annotated overlay
and a single trackers.csv with one row per detected marker.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def local_texture(gray):
    """High-frequency energy map: markers sparkle, bone is smooth."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # local RMS of the Laplacian in a small window
    energy = cv2.GaussianBlur(lap * lap, (0, 0), 3.0)
    return np.sqrt(energy)


def _score(gray, raw, tex, x, y, r):
    """Sparkle inside and brightness contrast vs the surrounding ring."""
    h, w = gray.shape
    inner = np.zeros(gray.shape, np.uint8)
    cv2.circle(inner, (x, y), max(int(r * 0.8), 3), 255, -1)
    outer = np.zeros(gray.shape, np.uint8)
    cv2.circle(outer, (x, y), int(r * 1.7), 255, -1)
    cv2.circle(outer, (x, y), int(r * 1.1), 0, -1)
    inner_tex = tex[inner > 0].mean()
    inner_bri = gray[inner > 0].mean()
    ring_bri = gray[outer > 0].mean() if (outer > 0).any() else 255.0
    raw_ring = float(raw[outer > 0].mean()) if (outer > 0).any() else 255.0
    return dict(x=int(x), y=int(y), r=int(r),
                sparkle=float(inner_tex),
                contrast=float(inner_bri - ring_bri),
                inner_bri=float(inner_bri),
                raw_bri=float(raw[inner > 0].mean()),
                raw_ring=raw_ring)


def _blob_candidates(gray, raw, tex, min_r, max_r):
    """Primary detector: threshold the sparkle map and keep round blobs."""
    t = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t = cv2.GaussianBlur(t, (0, 0), 2.0)
    thr = max(int(t.mean() + 1.2 * t.std()), 40)
    _, mask = cv2.threshold(t, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    amin, amax = np.pi * (min_r * 0.6) ** 2, np.pi * (max_r * 1.4) ** 2
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (amin <= area <= amax):
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if min(bw, bh) / max(bw, bh) < 0.45:      # too elongated -> not a disk
            continue
        fill = area / (bw * bh)
        if fill < 0.45:                            # too sparse -> not a solid disk
            continue
        x, y = np.round(cent[i]).astype(int)
        r = int(np.clip(np.sqrt(area / np.pi), min_r, max_r))
        out.append(_score(gray, raw, tex, x, y, r))
    return out


def _hough_candidates(gray, raw, tex, min_r, max_r, dp=1.2):
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_r * 2,
        param1=110, param2=22, minRadius=min_r, maxRadius=max_r)
    out = []
    if circles is None:
        return out
    h, w = gray.shape
    for x, y, r in np.round(circles[0]).astype(int):
        if 0 <= x < w and 0 <= y < h:
            out.append(_score(gray, raw, tex, x, y, int(r)))
    return out


def _dedupe(cands, min_sep):
    """Keep the sparkliest of any cluster of near-coincident candidates."""
    kept = []
    for c in sorted(cands, key=lambda d: -d["sparkle"]):
        if all((c["x"] - k["x"]) ** 2 + (c["y"] - k["y"]) ** 2 > min_sep ** 2
               for k in kept):
            kept.append(c)
    return kept


def detect(img, min_r=12, max_r=60):
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    tex = local_texture(gray)
    cands = _blob_candidates(gray, raw, tex, min_r, max_r) + \
        _hough_candidates(gray, raw, tex, min_r, max_r)
    return _dedupe(cands, min_sep=48)


def filter_candidates(cands, sparkle_hi, sparkle_lo, contrast_min, raw_floor,
                      bright_raw, bright_contrast, bright_ring_max):
    """Accept a candidate on any of three marker signatures:

      * strong  -- an obvious sparkler (mid-lit glass-bead texture);
      * ringed  -- a dimmer sparkler sitting inside a clear dark ring;
      * bright  -- a bright, saturated disk (sparkle washes out to flat white
                   when strongly lit) still ringed by a strong dark border.

    A raw-brightness floor first rejects noise blobs from near-black frames: a
    real retroreflective marker returns lots of light and is genuinely bright,
    even when the overall frame is dark.
    """
    out = []
    for c in cands:
        if c["raw_bri"] < raw_floor:
            continue
        strong = c["sparkle"] >= sparkle_hi
        ringed = c["sparkle"] >= sparkle_lo and c["contrast"] >= contrast_min
        # a bright disk counts only if it sits in a genuinely dark surround --
        # a specular glint on bright bone has a bright ring and is rejected.
        bright = (c["raw_bri"] >= bright_raw
                  and c["contrast"] >= bright_contrast
                  and c["raw_ring"] < bright_ring_max)
        if strong or ringed or bright:
            out.append(c)
    return out


def annotate(img, markers):
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, m in enumerate(markers, 1):
        cv2.circle(vis, (m["x"], m["y"]), m["r"], (0, 0, 255), 2)
        cv2.circle(vis, (m["x"], m["y"]), 2, (0, 255, 255), -1)
        cv2.putText(vis, str(i), (m["x"] + m["r"], m["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="image files to process")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--sparkle-hi", type=float, default=130.0,
                    help="sparkle above this is accepted on its own")
    ap.add_argument("--sparkle-lo", type=float, default=80.0,
                    help="dimmer sparklers need the dark-ring contrast too")
    ap.add_argument("--contrast-min", type=float, default=25.0)
    ap.add_argument("--raw-floor", type=float, default=45.0,
                    help="reject blobs whose raw (un-normalized) brightness is "
                         "below this -- kills noise from near-black frames")
    ap.add_argument("--bright-raw", type=float, default=130.0,
                    help="a saturated bright disk this bright, with a strong "
                         "dark ring, is accepted even with low sparkle")
    ap.add_argument("--bright-contrast", type=float, default=30.0)
    ap.add_argument("--bright-ring-max", type=float, default=145.0,
                    help="the bright-disk path requires the surrounding ring's "
                         "raw brightness below this (a dark, genuine border)")
    ap.add_argument("--merge-dist", type=float, default=25.0,
                    help="cluster radius (px) when consolidating a shot's four "
                         "LED frames into one marker list")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for path in a.images:
        path = Path(path)
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip (unreadable): {path}")
            continue
        cands = detect(img)
        markers = filter_candidates(cands, a.sparkle_hi, a.sparkle_lo,
                                    a.contrast_min, a.raw_floor,
                                    a.bright_raw, a.bright_contrast,
                                    a.bright_ring_max)
        markers.sort(key=lambda m: (m["y"], m["x"]))
        outdir = Path(a.outdir) if a.outdir else path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        vis = annotate(img, markers)
        prefix = f"{path.parent.name}_" if a.outdir else ""
        op = outdir / f"{prefix}{path.stem}_trackers.png"
        cv2.imwrite(str(op), vis)
        print(f"{path.parent.name}/{path.name}: {len(markers)} markers -> {op.name}")
        for i, m in enumerate(markers, 1):
            rows.append([path.parent.name, path.name, i, m["x"], m["y"], m["r"],
                         round(m["sparkle"], 2), round(m["contrast"], 2),
                         round(m["raw_bri"], 1), round(m["raw_ring"], 1)])

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["shot", "image", "id", "x", "y", "r",
                         "sparkle", "contrast", "raw_bri", "raw_ring"])
            wr.writerows(rows)
        print(f"\nwrote {len(rows)} per-image rows -> {a.csv}")
        agg = aggregate_by_shot(rows, merge_dist=a.merge_dist)
        by_csv = str(Path(a.csv).with_name(Path(a.csv).stem + "_by_shot.csv"))
        with open(by_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["shot", "id", "x", "y", "r", "n_frames"])
            wr.writerows(agg)
        print(f"wrote {len(agg)} consolidated markers -> {by_csv}")


def aggregate_by_shot(rows, merge_dist=25):
    """Collapse the per-LED detections of a shot into one marker list.

    Within a shot the camera and markers are fixed across led1..led4, so the
    same physical marker appears at ~the same pixel in every frame that caught
    it. Cluster nearby detections, average their position, and report how many
    of the (up to four) LED frames saw each -- a marker seen in several frames
    is a confident detection; one seen once is worth eyeballing.
    """
    from collections import defaultdict
    by_shot = defaultdict(list)
    for shot, image, _id, x, y, r, *_ in rows:
        by_shot[shot].append((int(x), int(y), int(r)))

    out = []
    for shot in sorted(by_shot):
        clusters = []  # each: [sumx, sumy, sumr, n]
        for x, y, r in by_shot[shot]:
            for c in clusters:
                cx, cy = c[0] / c[3], c[1] / c[3]
                if (x - cx) ** 2 + (y - cy) ** 2 <= merge_dist ** 2:
                    c[0] += x; c[1] += y; c[2] += r; c[3] += 1
                    break
            else:
                clusters.append([x, y, r, 1])
        clusters.sort(key=lambda c: (c[1] / c[3], c[0] / c[3]))
        for i, c in enumerate(clusters, 1):
            out.append([shot, i, round(c[0] / c[3], 1), round(c[1] / c[3], 1),
                        round(c[2] / c[3], 1), c[3]])
    return out


if __name__ == "__main__":
    main()
