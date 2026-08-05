# paper/

Draft journal manuscript documenting the `marker-pipeline` measurement stack.

| File | What it is |
|---|---|
| `manuscript.md` | the draft (v0.2), Markdown + LaTeX math |
| `manuscript.pdf` | built output — **read this** |
| `build_pdf.sh` | `manuscript.md` → PDF (pandoc → MathML → headless Chromium; no LaTeX needed) |
| `style.css` | journal-style page/typography rules used by the build |
| `make_fig_inversion.py` | generates Figure 2 from the committed CSVs |
| `fig2_inversion.png` | Figure 2 output |

Rebuild: `python3 make_fig_inversion.py && ./build_pdf.sh`

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
The ICP episode survives only as a one-paragraph cautionary note in §2.11.4.

## Status of the numbers

Everything outside a marked placeholder block is reproduced from committed
outputs and independently verified. Headline figures:

- 18 markers over 5 sessions; 78% confirmed in all four LED frames
- photometric depth vs CT lens distance at `shot_011`: **r = −0.923**
- depth ordering of the 3 CT markers: **reversed**, (3,1,4) → (4,1,3)
- affine fit: `z_corr = −4.218·z_est − 5.791`, **4.36 mm RMS** over a 20.7 mm span
- both self-tests (`ct_correct_depth`, `led_calibration`) **pass**

## Placeholders — go back and fill

Marked inline with `⚠️ PLACEHOLDER` (red blocks in the PDF). Registration
sections are scaffolding only; bracketed values like `[T.TT]` are invented.

- **§2.11** Registration method — the paper's principal section, unwritten
- **§3.7** Registration accuracy — no result exists
- **§2.2.4** Capture conditions — not documented
- **§2.3** Calibration residuals — not recorded
- **§4.4, §5** item 10, **§6** final conclusion
- **Statements and Declarations** — all headings
- Front matter and reference list

Full checklist with dependencies: **Appendix C** (R1–R8, C1–C5, M1–M6).

## v0.2 changes

Revised after surveying how comparable papers report data collection and
registration validation. Two of the changes are **corrections to v0.1**:

1. **Quéau et al. is 2018**, *J. Math. Imaging Vis.* 60(3):313–340 — not 2017.
   (`led_calibration.py`'s docstring cites the 2017 preprint year; worth
   aligning.)
2. **The P3P citation was wrong.** The code calls `SOLVEPNP_AP3P`, which OpenCV
   implements from Ke & Roumeliotis (CVPR 2017) — not Gao et al. 2003. The
   neighbouring `SOLVEPNP_P3P` flag maps to Gao on OpenCV 4.5.5 but to a
   different algorithm on current 4.x, so **the version must be pinned or the
   citation is unfalsifiable**.

Structural additions: acquisition-invariants table, per-session file manifest,
rig-geometry table with a provenance column, an assumption ledger for the
photometric-stereo model, a software-provenance table binding each stage to its
solver flag and paper, FLE/FRE/TRE nomenclature with **TRE as the accuracy
endpoint and FRE explicitly demoted to a diagnostic**, a fiducial-configuration
subsection, ground-truth-uncertainty flags, and a full declarations block.

## Reference verification — read before submitting

Publisher sites (IEEE, Springer, SPIE, doi.org, PubMed, arXiv) were unreachable
from the build environment, so most references could **not** be confirmed against
a publisher record. Appendix D marks each entry `verified` or `unverified`, and
every unverified entry is bracketed in the reference list. Two entries are
genuinely verified — Quéau (via the author's own released code) and the
AP3P/OpenCV linkage (via the OpenCV source). **Everything else needs checking.**
Citation drift in third-party BibTeX is rampant; do not populate the list from a
reference manager without checking each entry.
