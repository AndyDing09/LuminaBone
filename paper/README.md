# paper/

Draft journal manuscript documenting the `marker-pipeline` measurement stack.

- **`manuscript.md`** — the draft (v0.1), Markdown + LaTeX math.
  Convert with e.g. `pandoc manuscript.md -o manuscript.pdf`.

## Scope

Covers sessions **`shot_011`–`shot_015`** and the code behind their
marker-depth slides:

| Stage | Module |
|---|---|
| Marker detection | `marker_pipeline/detect_trackers.py` |
| Depth + slide generation | `marker_pipeline/marker_depth_slides.py` |
| Photometric-stereo core | `scripts/reconstruction/bone_depth_batch.py` |
| Lens-to-marker distance | `marker_pipeline/marker_distances.py` |
| CT anchoring (diagnostic) | `marker_pipeline/apply_ct_shot011.py` |
| Correction models | `marker_pipeline/ct_correct_depth.py` |
| Near-field solver (blocked) | `scripts/reconstruction/nearfield_lambertian.py` |
| LED calibration (blocked) | `marker_pipeline/led_calibration.py` |

Deliberately **out of scope**: the older `loc01`–`loc17` triangulation slides,
the meshing/PLY family, and the correspondence-free surface-ICP registration.
The ICP episode appears only as a one-paragraph cautionary note in §2.10.

## Status of the numbers

Everything outside a marked placeholder block is reproduced from committed
outputs and verified. Headline figures:

- 18 markers over 5 sessions; 78% confirmed in all four LED frames
- photometric depth vs CT lens distance at `shot_011`: **r = −0.923**
- depth ordering of the 3 CT markers: **exactly reversed**
- affine fit: `z_corr = −4.218·z_est − 5.791`, **4.36 mm RMS** over a 20.7 mm span
- both self-tests (`ct_correct_depth`, `led_calibration`) **pass**

## Placeholders — go back and fill

Marked inline with `⚠️ PLACEHOLDER`. The registration sections are scaffolding
only; bracketed values like `[F.FF]` are invented, not measured.

- **§2.10** Registration method — the paper's principal section, unwritten
- **§3.6** Registration accuracy — no result exists
- **§4.4** Implications for registration — partial
- **§5** item 7, **§6** final conclusion
- Front matter, reference list

Full checklist with dependencies: **Appendix C** (items R1–R6, C1–C5, M1–M6).
