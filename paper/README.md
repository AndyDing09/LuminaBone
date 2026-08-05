# paper/

Draft journal manuscript: **near-light photometric stereo** for the four-LED
ring endoscope, its calibration, and fiducial-anchored registration to CT.

| File | What it is |
|---|---|
| `manuscript.tex` | the draft (v0.3), IEEEtran **two-column journal format** |
| `manuscript.pdf` | built output — **read this** (15 pp.) |
| `make_fig_rig.py` → `fig1_rig.png` | Fig. 1, system geometry (LED ring + side view) |
| `make_fig_inversion.py` → `fig2_inversion.png` | Fig. 2, the CT inversion result |
| `manuscript.md` | superseded v0.2 single-column draft, kept for reference |
| `build_pdf.sh`, `style.css` | v0.2 markdown→PDF path, superseded by LaTeX |

Rebuild:

```bash
python3 make_fig_rig.py && python3 make_fig_inversion.py
pdflatex manuscript && pdflatex manuscript    # twice, for refs
```

## Structure

Reorganised so near-light photometric stereo is the method, not an appendix:

1. **System design** — optical head, four LEDs 90° apart on a ring, Arduino Pro
   sequencing, coordinate conventions, capture protocol
2. **Geometric calibration** — the ring **offset** (12.08 mm) and the **scale**
   (37 mm field → mm/px, focal length), and the focal-length conflict
3. **Near-light photometric stereo** — why far-field is inadmissible at
   η = 0.40, the point-source irradiance model, dichromatic specular separation,
   per-pixel normal solve, **perspective log-depth derivation**, iteration
4. **LED calibration** — anisotropy, mirror-sphere triangulation, direction and
   intensity by white-plane least squares
5. **Fiducial localization** — sparkle cue, acceptance, cross-frame confirmation
6. **Registration to CT** — error nomenclature, P3P transfer *(rest placeholder)*
7. **Experimental validation** — yield, the inversion, distances, scale
   conflict, self-tests

## Math verification

Every derivation was independently re-derived and numerically cross-checked
before this draft. **Verified correct:** the perspective log-depth relation
(symbolically and against finite differences), the `r³` power in the irradiance
model, the normal-equation solve, the LED ring geometry including the Y-sign,
`fx = W·d/F` and all four focal-length figures, the elevation expression, the
`s = d/fx` identity, the P3P transfer and range formula, and every LED-calibration
solver.

**Problems found and now reported in the paper** — several would have been
caught by a reviewer:

- **The iteration-count justification was false.** The code claimed higher
  iteration counts degrade the normals; a convergence sweep shows a stable fixed
  point from sweep 4 through sweep 30. The paper now reports the real convergence
  behaviour (Table IV).
- **The `fx/fy = 1.33` claim was backwards.** At the 640×480 call actually used
  the formula gives fx/fy = 1.0000; 1.33 appears only for 16:9 input.
- **The 46.0 mm anchor is mislabelled** as the "median" P3P distance — the median
  is 50.92 mm; 46.0 matches the *mean*. It is also a Euclidean statistic used as
  an axial depth, double-counting obliquity by up to 15% off-axis.
- **A scale conflict no focal length resolves** (new §VIII-E). The pose solution
  implies a 59–63 mm field width for *any* fx, versus the 37 mm bench figure.
  Either that measurement is wrong by ~1.7×, or the CT-to-detection fiducial
  correspondence is wrong — the detector numbers fiducials by image position,
  which bears no necessary relation to CT numbering, and index equality is
  assumed but nowhere verified. This does **not** affect the inversion result
  (correlation and rank are scale-invariant).
- **The calibration self-tests are not sound validation.** Mutation testing shows
  the emitter-position test passes with a deliberately wrong tangent-cone formula
  (sin→tan), and its 2.0 mm tolerance passes a broken solver on 10 of 12 seeds
  against a true error of 2.5e−12 mm. The cos⁴ and 1/r³ laws are tautological
  (identical in generator and solver), and μ = 1 is hard-coded so the
  linearisation is never exercised.
- Principal point set to the image centre rather than its calibrated value
  (14.5 px in cy, 0.1–0.7 mm on reported distances).
- Cheirality filtering in P3P is a **no-op**; the ordinal constraint is a
  one-bit test that fails in 2.9% of random configurations — though it is
  unambiguous for this dataset across 40,000 perturbation trials.
- Dichromatic separation needs equal *linear R and B camera response*, not merely
  a "white" LED: a 5% imbalance leaks ~18% of the specular signal.

## Placeholders — go back and fill

Red boxes in the PDF. Registration sections are scaffolding; bracketed values
like `[T.TT]` are invented.

- **§VII** Registration method — the principal section, unwritten
- **§VIII-G** Registration accuracy — no result exists
- **§II-B** Arduino Pro variant, LED part number, drive current, θ½
- **§III-C** calibration residuals; **§VIII-A** specimen description

Full checklist: **Appendix A** (R1–R8, C1–C9, M1–M6).

## Reference verification

Publisher sites were unreachable from the build environment. **Appendix B** marks
each entry verified or unverified; only Quéau *et al.* (2018) and the
AP3P/OpenCV linkage are verified. Note the code calls `SOLVEPNP_AP3P` (Ke &
Roumeliotis), *not* Gao *et al.* — and the neighbouring `SOLVEPNP_P3P` flag maps
to different papers across OpenCV versions, so pin the version.
