# Verified ground truth for the LuminaBone ICRA paper
Every number below was reproduced by fresh runs in this container on the real
capture data (Aug 2026). Sources: nearfield_ctfree.py, nearfield_ct.py,
register_photometric_ct.py, register_inverse_square_ct.py,
led_permutation_study.py, led_wiring_ctfree.py, calibration/calibration.txt,
depth_outputs/ct_registered/ctfree_paper_results.json, ctfree_mu_compare.json.

## Rig
- LED ring radius a_r = 6.05 mm (measured; CAD nominal 5.091 mm)
- 4 white LEDs at azimuths 0/90/180/270; wiring LED1->90, LED2->270, LED3->180, LED4->0
- Native 640x480; fx = 860.175 px, fy = 857.914 px (ChArUco, 121 images,
  reprojection 0.107 px). The 0.42 px figure = 4-point PnP validation residual.
- Probe OD 18 mm (tubular-retractor scale).
- alpha = arctan(a_r/b): at b=55 mm, alpha = 6.28 deg; sin(alpha) ~= 0.109.
- Far-field 4-light matrix: L^T L = diag(2sin^2a, 2sin^2a, 4cos^2a) exactly,
  so condition number = sqrt(2)*cot(alpha) (~12.9 at 55 mm). Verified analytically.

## CT-free solver (nearfield_ctfree.py) — parameters as coded
- Depth family D = b*(1 + s*ztilde/f); ztilde median-centred, clipped 1-99 pct.
- Objective: tied normals from candidate depth; per-pixel albedo + per-LED gain
  (2 alternations) + global mu; relative L1 residual, attached shadows masked.
- Search: DS=4 downsample; B_RANGE=(45,70) mm (lens depth-of-field/focus band);
  S_RANGE=(0.10,5.0); MU_RANGE=(0,14); coarse grid (b step 5 mm;
  s in {0.25,0.5,0.75,1,1.5,2,3}) + Nelder-Mead in (b, log s).
- Multi-start seeds b0 in {48, 57, 66}; 6 iterations; keep-best (lowest E) over
  all iterations and seeds. Full-res trajectories; DS objective only inside E.
- Trim: free-normal residual < 0.22; self-referential depth band |D-med|<25 mm;
  spike |D - median5(D)| < 8 mm; border crop 9% of min(H,W); opening x2;
  components >= max(300 px, 15% of largest); erosion x1. NO CT anywhere.
- Evaluation only: pairwise fiducial-distance ratios (rec/CT); rigid 6-DOF
  Kabsch -> FRE RMS; Umeyama similarity scale (CT ~ c*R*rec + t).

## CT-free results (mu*=8, rigid-only alignment) — ctfree_paper_results.json
| level | b (mm) | s | residual | dist-scale | sim-scale | FRE rms | n mark |
| 4 | 45.05 | 0.735 | 0.1704 | 1.132 | 0.850 | 2.68 | 4 |
| 5 | 45.55 | 1.000 | 0.2194 | 0.793 | 1.271 | 3.07 | 4 |
| 6 | 52.32 | 1.000 | 0.2606 | 0.897 | 1.054 | 2.78 | 4 |
| 7 | 69.89 | 1.857 | 0.2527 | 1.249 | 0.766 | 6.02 | 3 |
| 8 | unstable (0 px survive the trim) |
- Per-marker FRE: lvl4 [1.46 3.38 1.19 3.72]; lvl5 [1.25 3.82 1.15 4.5];
  lvl6 [2.33 4.56 1.85 1.15]; lvl7 [8.19 4.19 4.92].
- Scale-error range on 4-marker levels: |dist-scale-1| = 13%, 21%, 10%.
  (Paper says 10-21%.)

## CT-derived working distances (evaluation only; PnP + CT markers)
lvl4 60.0, lvl5 55.4, lvl6 58.9, lvl7 58.6, lvl8 51.5 mm (all 51-60 mm).
CT fiducial depth ranges: 17.2 / 22.4 / 26.0 / 25.7 / 24.4 mm.

## CT-anchored oracle (nearfield_ct.py, mu=0 point source, 4 iters, CT affine)
FRE: lvl4 1.72, lvl5 1.32, lvl6 1.74, lvl7 9.72, lvl8 unstable (0 px).

## Far-field baseline (register_photometric_ct.py; CT-anchored)
FRE: lvl4 1.54, lvl5 1.22, lvl6 6.57, lvl7 11.14, lvl8 5.66.

## Inverse-square baseline (register_inverse_square_ct.py; CT-anchored)
FRE: lvl4 6.39, lvl5 9.44, lvl6 9.55, lvl7 7.91, lvl8 6.40 (range 6.4-9.6).

## Emission exponent mu
- ctfree_mu.json in repo: mu=8, note: validated by truth-test on shots 4/5
  (CT-validated surfaces both prefer mu~8); TODO bench white-target calibration.
- CT-free cross-capture vote (calibrate_mu, DS resolution): votes
  mu0 4.167, mu2 4.051 (min), mu4 4.147, mu6 4.125, mu8 4.184, mu10 4.278,
  mu12 4.657 -> spread < 6% over mu in [0,10]; weak minimum at mu=2.
- End-to-end at mu=2 (ctfree_mu_compare.json): lvl4 2.70, lvl5 5.52, lvl6 8.35,
  lvl7 8.05. => accuracy is strongly mu-sensitive while the objective is not.
- Paper position: mu*=8 pinned; provenance (CT truth-test) disclosed as the one
  impure rig-level constant; bench calibration is the fix.

## Wiring / aim (led_permutation_study.py, led_wiring_ctfree.py)
- Permutation study (24 assignments, far-field depth vs CT marker depth):
  true wiring r = +0.97 (lvl4) / +0.99 (lvl5), mean +0.98; diametric flip
  r = -0.97/-0.99. Mislabelling INVERTS the surface.
- Beam-centroid aim map: 5/5 shots agree; vertical pair (LED1/LED2) beams land
  OPPOSITE their ring positions -> convergent aim; horizontal pair ~parallel.
  Aim redistributes intensity but cannot flip incidence side; wiring rests on
  incidence. One-time bench check makes wiring provenance fully CT-free.

## Flat-plane landscape (current code, shot_004)
Residual variation over b in [45,70] at any mu: < 2.5% total (0.3162-0.3232).
Multi-start trajectories at mu*=8 all reach b ~= 45; kept b=45.05, s=0.735.

## Misc
- Relief (P90-P10 of kept depth): lvl4 26.6, lvl5 26.6, lvl6 21.9, lvl7 25.4 mm.
- Keep pixels: 121834 / 87939 / 48086 / 39401.
- Runtime: minutes per capture, unoptimised Python (single-core numpy).
- Phantom: Sawbones 1324-56 lumbar L1-sacrum, rigid foam; 4 retroreflective
  fiducials per vertebra (lvl7: 3 usable); CT-scanned with markers in place.
- ICRA 2027: 8-page limit INCLUDING references; double-anonymous; deadline
  Sept 15, 2026 (11:59 PST). Current build: 8 pages (at the cap; re-check after restoring the real author block).

## Bowl figure (regenerated from current verified meshes, level 4)
- Far-field surface plane-fit RMS: 6.90 mm; near-field: 4.37 mm (fresh SVD
  plane fit on the reran CT-frame meshes; old paper said 6.90/4.98 -- the
  near-field mesh changed with the newer trim).

## Literature numbers used in the rewrite (web-verified by extraction agents, Aug 2026)
- Batlle IROS 2022: 7% / 2.8 mm mean depth error, SIMULATED colon only; in-vivo
  frame qualitative with ARBITRARY scale (auto-gain unknown); photometric
  calibration ~3 gray levels. Scale needs known albedo + auto-gain.
- LightNeuS MICCAI 2023: 2.80 mm mean MAE on C3VD (18 high-parallax seqs),
  20 views + GT-registered metric poses, per-scene 300k iters; 8.23 mm when
  camera motion < 1 cm.
- LightDepth ICCV 2023: 3.70-4.37 mm MAE on C3VD AFTER median GT scale
  alignment; authors state predictions up-to-scale (gain/albedo ambiguity).
- EndoMetric MICCAI 2025: ~3 mm baseline; ~1% scale error at <=8 mm, 5% by
  20 mm (simulation); real polyps deviate 1.0 mm (13%) from endoscopist.
- Parot JBO 2013: qualitative by design (high-pass); no accuracy figure; use
  up to ~40 mm WD. Durr SPIE 2014 + DDW 2014: 14 mm clinical scope, in vivo
  n=8 rectums, no accuracy metric.
- Collins & Bartoli MICCAI 2012: error maps 1-6 mm ex vivo (~34.5 mm range);
  no summary statistic; in vivo sparse-marker histogram bulk within ±3 mm.
- Hao Sensors 2020: 4 LEDs at 5.5 mm offset; 0.51 mm / 2.57% RMSE at 21.37 mm
  WD on one smooth convex target; seed from specular highlight; ±0.94 mm seed
  error -> 1.65-1.78 mm RMSE.
- Tamura Eur Spine J 2005 (PMID 15526221): 0.96 ± 0.24 mm RMS with tracked
  stylus, 20 lamina points, dry-bone phantom.
- Jakubovic Sci Rep 2018: OTI structured light; registration 5.07 ± 1.83 s,
  >250k pts; clinical screw error ~1.21/1.13 mm median (95th pct 3.4-4.3 mm).
- Guha Spine J 2017: clinical 3-D navigation mean abs translational error
  1.20-1.75 mm, angular 3.1-3.6 deg.
- Guha Global Spine J 2019: navigation error >=4 mm at 5 levels from DRF;
  manipulation displaces vertebrae 1.55 ± 1.13 mm; respiration 1.96 ± 1.32 mm.
- LUCES 2021: best calibrated near-field PS 13.3 deg / 3.17 mm avg (52 LEDs,
  10-30 cm); naive far-field 37.5 deg; Queau2018 17.9 deg / 7.27 mm;
  GT floors 3.3 deg / 1.67 mm.

## Additional code/hardware constants (verified in code or CLAUDE.md)
- search_bs b-grid: +/-20 mm about current iterate, 5 mm steps (code L215-216);
  coarse s samples {0.25,0.5,0.75,1,1.5,2,3} within S_RANGE [0.1,5].
- Depth floor 2 mm (reject + clamp); quad emitted only if 4 corners span <3 mm
  (grid_faces tol=3.0); fiducial depth = median over 14-px-radius disk (_disk).
- Hardware from Claude_Context/CLAUDE.md: camera bore ⌀9.72 mm on axis (CAD);
  FOV <=90 deg diagonal (measured); LED axial spread ~2 mm (calipers).
- Rampersaud Spine 2001: required accuracy for pedicle screws, ~1 mm
  translational at tightest levels (citation-sweep verified).

- Guha 2019 respiration detail: 1.96 ± 1.32 mm mean, greatest in the lower
  thoracic spine (p<.001) — from the extraction agent's verified full-text read.
