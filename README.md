# LuminaBone

Endoscope-based bone reconstruction and validation against CT ground truth,
using retroreflective marker tracking for depth/positioning.

## Status (marker-pipeline branch)

This branch is exploratory/diagnostic — several results are flagged
unreliable and documented as such rather than hidden. Read this before
trusting any numbers:

- **Reliable:** marker pixel-coordinate detection (all shots); marker
  distance from lens for shot_011 (CT + PnP ground truth).
- **Not reliable:** far-field photometric depth (`marker_depth_slides.py`)
  — inverted vs. CT at shot_011 (corr −0.92); everything downstream of it
  for shots 10/12/13/14/15 inherits that inversion.
- **Built but blocked:** near-field Lambertian depth solver
  (`nearfield_lambertian.py`) and LED calibration (`led_calibration.py`) —
  the physically-correct depth path, waiting on mirror-ball/white-card
  calibration captures that don't exist yet.
- **Registration (endoscope patches vs. CT spine):** automatic
  similarity-ICP registration of 17 reconstructed patches does **not**
  correctly localize them — patches drape onto the same region rather than
  spreading to their true anatomical sites, so the ~1.9 mm "accuracy"
  figure is an artifact, not a real accuracy measurement. A valid study
  needs landmark correspondences or fiducial markers.

## Repo layout

- `marker_pipeline/` — tracker detection → marker depth → CT-distance
  scripts, with a detailed status table in its own README.
- `registration/` — endoscope-to-CT ICP registration, accuracy report,
  and Slicer/photogrammetry helpers.
- `scripts/` — shared `calibration/`, `experiments/`, and `reconstruction/`
  code (e.g. `bone_depth_batch.py`, the far-field core dependency).
- `presentations/` — triangulation slide deck (PDF/PPTX).
- `viewer_checks/` — sanity-check renders of reconstructed models.
- `tools/` — misc utilities (e.g. `rclone.exe`).

## Open next steps

1. Capture LED calibration targets (mirror ball + matte white card) to
   unblock the near-field depth solver.
2. Resolve focal length ambiguity (fx 519 vs. 794) to remove the ~1.5×
   depth-scale uncertainty.
3. Add landmark or fiducial-based anchoring so endoscope-vs-CT accuracy
   can be measured honestly instead of via drape-fitting ICP.
