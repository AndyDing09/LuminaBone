# marker_pipeline

Everything from the tracker → marker-depth → CT-distance work, grouped in one
place. Read the **status** column first: it says what you can trust.

## TL;DR — what is actually reliable

| Result | Trust it? |
|---|---|
| Marker **pixel coordinates** (all shots) | ✅ Yes — direct detection |
| Marker **distance from lens, shot_011** | ✅ Yes — CT ground truth + PnP (4≈33, 1≈51, 3≈53 mm*) |
| Marker **relative depth** (`marker_depth_slides`) | ❌ No — far-field model, came out **inverted** vs CT at shot_011 |
| Marker **distance, shots 10/12/13/14/15** | ❌ No — photometric only, no CT to validate (inherits the inversion) |

\* absolute values carry a ~1.5× uncertainty because the focal length is still unresolved (fx 519 vs 794).

**Only shot_011 has CT, and there the far-field depth is inverted.** So the depth
numbers are for locating/plotting markers, not for trustworthy depth. The real
depth fix (near-field solve) is built but **blocked on calibration captures**.

## Files — used vs. not

| File | Does | Status |
|---|---|---|
| `detect_trackers.py` | Finds the retroreflective markers, per image + consolidated per shot | ✅ **USED, reliable** |
| `marker_depth_slides.py` | Per-shot far-field photometric depth + the triangulation-style slides with (x,y,z) | ⚠️ **USED for x,y + slides; depth is far-field & inverted** |
| `marker_distances.py` | Lens-to-marker distance: CT+PnP for shot_011, photometric (flagged) for the rest | ✅ **USED** (shot_011 reliable; rest flagged UNRELIABLE) |
| `apply_ct_shot011.py` | Anchors shot_011 depth to CT; **documents** that the far-field depth is inverted | ⚠️ **USED as a diagnostic**, not a trustworthy correction |
| `ct_correct_depth.py` | Affine-z / tilt-plane correction helpers (self-test passes) | ✅ tool reliable; its *input* (far-field depth) is the weak link |
| `led_calibration.py` | Quéau et al. 2017 LED calibration (position/direction/µ/intensity) | 🚧 **BUILT, NOT USED** — needs mirror-ball + white-card captures that don't exist yet |

### Shared infrastructure (NOT moved — lives in `../scripts/reconstruction/`)
| File | Role | Status |
|---|---|---|
| `bone_depth_batch.py` | Far-field photometric-stereo core + image/calibration helpers | dependency (has its own pre-existing uncommitted edits) |
| `nearfield_lambertian.py` | Near-field solver; **upgraded this session to require a solved `led_calibration.json`** (no assumed geometry) | 🚧 **the intended real depth path**, blocked until calibration exists |

## How to reproduce (run order)

```bash
# from marker_pipeline/
python detect_trackers.py ../Data_collection/shots/shot_*/led*.png \
       --outdir ../tracker_outputs --csv ../tracker_outputs/trackers.csv
python marker_depth_slides.py                 # per-shot depth slides + marker_depths.csv
python marker_distances.py                    # lens-to-marker distances (shots 10-15)
python apply_ct_shot011.py                    # shot_011 CT anchor (diagnostic)

# validation (no data needed):
python ct_correct_depth.py --selftest
python led_calibration.py  --selftest
```

Outputs land in `../depth_outputs/marker_depth_slides/` and `../tracker_outputs/`.

## The one path to trustworthy depth (currently blocked)
1. Capture calibration targets: a **mirror ball** (~10 poses/LED) and a **matte
   white card** (~3–5 tilts/LED).
2. `led_calibration.py` → writes `calibration/led_calibration.json`.
3. `nearfield_lambertian.py` re-solves depth with the real light model (no bowl,
   no inversion).
4. Anchor the result to CT with `ct_correct_depth.py`.

Also worth doing: resolve the focal length (photograph a known-size object at a
known distance) to kill the 1.5× absolute-scale ambiguity.

## Key caveats discovered
- **CT is in the CT scanner's frame, not the camera frame** — you must PnP-align
  before comparing depth. Raw-axis comparison is meaningless.
- **P3P with 3 points is ambiguous** (up to 4 poses); we disambiguate with the
  confirmed physical fact that *marker 4 is closest*.
- With that pose, the far-field photometric depth is **anti-correlated with CT
  (corr −0.92)** at shot_011 — i.e. inverted. That is the headline finding.
