# Endoscope-vs-CT accuracy: what the registration actually shows

## What I did
Registered all 17 endoscope reconstructions (`loc01–17`) to the CT spine
(`ct_spine.stl`, a lumbar+sacrum segment, mm coordinates) using automatic
similarity ICP (rotation + translation + uniform scale), with centroid
initialization, a realistic scale prior, and rotation restarts. Outputs are in
`reg_out/`: each aligned patch, a combined model (`combined_CT_plus_17.ply`),
per-patch metrics (`accuracy_summary.csv`), and the overlay (`accuracy_overlay.png`).

## The honest result: automatic registration did NOT place the patches correctly
- **12 of 17** patches got a plausible scale (~0.07 mm/px) and ~1.9 mm median
  surface distance — which *looks* like a good accuracy number.
- **But the overlay shows they all piled onto the same region of the bone**,
  stacked as flat cards, instead of spreading to the 17 different spots where the
  photos were actually taken. So the low distance is a **draping artifact**: a
  bumpy patch laid flat against a bumpy surface always sits ~2 mm off, *wherever*
  you put it. It is not a measure of reconstruction accuracy.
- **5 of 17** (loc07, 09, 10, 11, 14) collapsed to scale ≈ 0 — the scale-ICP
  shrank the patch to a point to trivially minimise distance. Flagged SUSPECT.

The uniformity is the tell: median deviation across the 12 "OK" patches is
1.86 mm with a standard deviation of only 0.17 mm. Genuinely-registered patches
at 17 different anatomical sites would not all score the same.

## Why this happens (not fixable by better ICP)
Each endoscope view is a small, nearly-flat brightness-relief patch; the CT is a
whole 232 mm multi-vertebra segment. With no point correspondences and no known
probe pose, a flat patch can rest on many places with equally low residual, and
its scale is unconstrained. ICP therefore cannot recover the *true* location — it
finds *a* low-distance drape, not the *correct* one.

## The one real quantitative finding
The recovered scales for the 12 non-degenerate patches cluster at
**0.070 ± 0.012 mm/px**. Consistent scale across independent views is expected if
the endoscope working distance was roughly constant — a believable internal
cross-check on the reconstruction pipeline. (Note: the pipeline's own
`depth_estimates.csv` implies ~0.047 mm/px for the depth axis; the ~0.07 in-plane
value is the same order, differing because depth and in-plane were scaled
separately.)

## What a valid accuracy study needs
To actually measure "how accurate is the endoscope reconstruction vs CT," each
patch must first be correctly located on the bone. That requires one of:
1. **Matched landmarks per patch** — 3–4 corresponding points (e.g. the through-
   holes/foramina visible in both the photo and the CT). The tools here
   (`landmark_register.py` + `icp_refine.py`) compute this in closed form and
   report a real error, but placing the points needs someone who knows where each
   shot was taken.
2. **CT-visible fiducial markers** (metal beads) placed on the specimen before
   scanning, appearing in both modalities.
3. **Recorded probe poses** at capture time.

Once any patch is correctly anchored (the loc03 foramina are the best candidate),
the residual surface distance at that site becomes a *true* reconstruction-accuracy
number, and the same can be repeated for the rest.

## Files
- `reg_out/accuracy_overlay.png` — shows the piling-up problem
- `reg_out/accuracy_summary.csv` — per-patch scale, deviation, reliability
- `reg_out/combined_CT_plus_17.ply` — CT + all patches, viewable in Slicer
- `reg_out/locNN_reg.ply` — individual aligned patches
