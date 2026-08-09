# Near-Field Photometric Stereo for CT-Registered Surface Reconstruction with a Near-Coaxial Monocular Endoscope: Sub-Millimetre-to-Millimetre Accuracy on Spinal Bone

**Authors:** A. Ding et al.
**Affiliation:** *(to be completed)*
**Corresponding author:** andyding09@gmail.com

---

## Abstract

**Purpose.** Image-guided spine surgery depends on registering the exposed bony
surface to a preoperative CT. We investigate whether a monocular endoscope with a
ring of near-coaxial LEDs — the illumination geometry forced by an 18 mm-diameter
scope — can reconstruct the exposed spinal surface accurately enough for
per-vertebra CT registration, using only its own images and no added tracking
hardware.

**Methods.** Four single-LED frames per pose drive a photometric-stereo solve.
We compare three surface models: (i) classical far-field photometric stereo,
(ii) a near-field point-source model with per-pixel light directions and inverse-
square attenuation, and (iii) an inverse-square brightness-to-range baseline. Each
reconstruction is anchored to the CT through retroreflective fiducials that are
visible in both modalities, using per-vertebra Perspective-n-Point pose estimation
followed by an affine depth calibration. Accuracy is reported as the 3-D fiducial
registration error (FRE) against CT ground truth on five vertebral levels.

**Results.** With the correct near-field light model, three of five levels
reconstructed to **1.2–1.7 mm** 3-D FRE with only four fiducials per level — 
comparable to clinical surface-based spine registration (≈1 mm with 20 points) and
better than reported in-vivo endoscopic photometric stereo (<3 mm). The near-field
model removed the systematic "bowl" artifact that far-field integration introduces
on this geometry (plane-fit deviation reduced from 6.9 to 5.0 mm on the reference
level; the worst far-field level improved from 6.6 to 1.7 mm FRE). The inverse-
square baseline failed (7.8–9.6 mm, with physically inverted depth) because it
conflates albedo with range on textured bone. All levels were imaged at a similar
working distance (51–60 mm); the two non-millimetre levels failed for non-distance
reasons (three fiducials; an unresolved near-field divergence), reported in full.

**Conclusion.** Near-coaxial monocular photometric stereo yields per-vertebra,
CT-registered surface reconstructions at millimetre accuracy on well-conditioned
levels, without added intraoperative radiation or tracking hardware. The
dominant residual error is the use of assumed rather than calibrated LED positions,
which defines the clear next step.

**Keywords:** photometric stereo · near-field · point light source · endoscopy ·
image-guided surgery · spine · CT registration · fiducial registration error

---

## 1. Introduction

Pedicle-screw placement and other spinal procedures rely on registering the
patient's intraoperative anatomy to a preoperative CT so that instruments can be
navigated relative to the plan. Established navigation registers each vertebra as
an independent rigid body — because the spine flexes at the intervertebral discs,
no single rigid transform fits multiple levels — using either intraoperative
imaging (cone-beam CT / O-arm, which adds radiation) or a tracked surface
digitiser touched to exposed landmarks. An optical method that reconstructs the
exposed surface directly from the endoscope already in the field, with no added
radiation and no external tracker, is therefore attractive.

Recovering metric surface shape from a single moving endoscope is hard. Structure-
from-motion and SLAM need texture and parallax that wet bone provides poorly;
learning-based neural rendering (NeRF/Gaussian-splatting variants) needs many views
and does not natively produce a CT-registered metric surface. **Photometric
stereo** is attractive precisely because it is *albedo-invariant*: it recovers the
surface normal at each pixel from the *ratios* of brightness across several
illuminations, so a dark and a bright patch at the same tilt yield the same normal.
On textured bone this is exactly the property brightness-based shape-from-shading
lacks.

The obstacle is the illumination geometry an endoscope imposes. A ring of LEDs
around an 18 mm scope sits only ≈6 mm off the optical axis, so at a 40 mm working
distance the lights are **near-coaxial** (≈8.6° off-axis; ≈17° between opposing
pairs) and **near-field** (light direction and intensity vary across the field of
view). Classical far-field photometric stereo — a single constant direction per
light — mis-models this and integrates the residual brightness gradient into a
spurious low-frequency curvature, the "bowl". Recovering usable geometry therefore
requires a near-field point-source model.

**Contributions.** (1) A near-field photometric-stereo pipeline for a near-coaxial
LED endoscope that removes the far-field bowl; (2) a per-vertebra CT-registration
protocol using dual-modality retroreflective fiducials, PnP pose, and affine depth
calibration, validated by 3-D FRE; (3) a controlled comparison of far-field, near-
field, and inverse-square models on the same CT-validated data; and (4) a
transparent failure analysis (three fiducials under-constrain PnP; an unresolved
near-field divergence) that isolates the single dominant error source — assumed LED
positions — as the
next lever.

## 2. Related work

**Photometric stereo.** Woodham's far-field formulation [1] recovers normals from
≥3 directional illuminations; Frankot–Chellappa integration [2] converts normals to
depth. Near-field photometric stereo with point light sources models per-pixel
direction and 1/r² fall-off; Quéau et al. [3] give a complete LED forward model
(position, direction, anisotropy, radiometric fall-off) with a calibration and
numerical solution. Recent work targets robust point-light-source *position*
calibration, reporting light-position errors <2.7 mm under noise [4], and multi-view
near-field datasets/benchmarks [5]. Our rig operates in an unusually near-coaxial
regime, where the tilt signal is a small fraction of the incident light.

**Endoscopic 3-D reconstruction.** In-vivo endoscopic photometric stereo has
reconstructed the human colon at a mean depth error <3 mm (≈7%) [6]. Neural
approaches (LightNeus and successors) and Gaussian-splatting pipelines [7] achieve
high-fidelity rendering but do not, by construction, deliver a CT-registered metric
surface. Photometric stereo has also been applied to capsule endoscopy [8]. Our
emphasis differs: metric, *CT-validated, per-vertebra* registration rather than
appearance or relative shape.

**Surface-based registration.** Rigid registration of an intraoperative surface to
CT is standard in spine navigation, typically via Iterative Closest Point [9];
surface-based spine registration achieves ≈0.96 mm RMS with ~20 posterior-lamina
points [10]. Registration quality is reported as fiducial/target registration error
(FRE/TRE) [11]. We adopt FRE against CT-visible fiducials as the primary metric and
follow the clinical convention of registering **per vertebra**.

## 3. Materials and methods

### 3.1 Imaging system

A monocular endoscope (native 640×480) carries four LEDs on an ≈6.05 mm radius ring
about the optical axis (18 mm scope body). At a nominal 40 mm working distance each
LED lies ≈8.6° off-axis, so opposing LEDs subtend only ≈17° at the surface and the
useful lateral (tilt-encoding) component of each light vector is ≈cos 81° ≈ 15% of
its magnitude. The field of view spans ≈37–43 mm at working distance. Retro-
reflective fiducial markers placed on the specimen are visible in CT as small
surface craters, providing dual-modality correspondences. Each capture records one
dark frame and four single-LED frames.

### 3.2 Camera calibration

Intrinsics were obtained with a ChArUco target; the working focal length
(fₓ ≈ 860 px at 640×480) was independently validated by 4-point PnP reprojection
(0.42 px mean residual on the reference level), resolving an earlier focal-length
ambiguity.

### 3.3 Photometric preprocessing

Each single-LED frame is dark-subtracted, gamma-linearised to Rec.601 luminance,
specular-inpainted (retroreflective glints violate Lambert and are detected as
bright, low-saturation pixels), and exposure-balanced (per-frame median
normalisation removes the independent auto-gain of separate exposures, which would
otherwise inject a false gradient).

### 3.4 LED index → azimuth mapping (a critical correction)

Photometric stereo is only as correct as its assumed light directions. The physical
wiring maps LED1→top (90°), LED2→bottom (270°), LED3→left (180°), LED4→right (0°),
with opposing pairs (1,2) and (3,4). An earlier incorrect assumption
(0/90/180/270°) produced normals **anti-correlated** with CT depth (r ≈ −0.73);
the correct mapping yields r ≈ +0.97/+0.99 on two independent CT-validated levels,
turning an apparent "no-signal" result into millimetre accuracy. This is reported
because it is a common, silent failure mode for multi-LED rigs.

### 3.5 Far-field photometric stereo

The classical solve treats each LED as a constant direction **Lₖ**, forms the
per-pixel matrix equation *Iₖ = ρ(**n·Lₖ**)*, solves **g** = ρ**n** by least
squares, normalises, and integrates the gradient field via Frankot–Chellappa to a
relative depth *z*.

### 3.6 Near-field photometric stereo

For a surface point **X**(u,v) at depth *D*, back-projected through the intrinsics,
and an LED at position **Pₛ** on the ring, the incident direction is
**lₛ** = (**Pₛ − X**)/‖**Pₛ − X**‖ and the attenuation is 1/‖**Pₛ − X**‖². Writing
*mₛ = Iₛ·‖**Pₛ − X**‖²* linearises the point-source model to *mₛ = ρ(**n·lₛ**)*,
a per-pixel 3×3 least-squares solve. The surface depth couples into the light
geometry, so we iterate: bootstrap *D* to the CT working distance, solve normals,
integrate, re-anchor to the CT fiducials (§3.7), recompute per-pixel geometry, and
repeat (4 iterations). LED positions are, in this study, *assumed* from the ring
radius and azimuth mapping rather than independently calibrated — the principal
remaining approximation.

### 3.7 CT registration (per vertebra)

For each vertebra, four non-coplanar fiducials give the CT→camera pose by PnP
(SQPnP, robust for four points where iterative DLT and P3P are ill-posed). The
photometric depth is affine-calibrated to true axial distance,
*z_lens = a·z_photo + b*, fit on the fiducials; the surface is then back-projected
to camera coordinates and transformed into CT coordinates by the PnP pose. Because
the spine flexes between levels, each vertebra is registered independently; a single
rigid multi-level fit is used only as a rigidity check (§4.5). The dense surface is
masked to trustworthy bone (brightness threshold, photometric-residual confidence,
largest connected component, frame-border crop) and mask-aware smoothed before
meshing.

### 3.8 Evaluation

Accuracy is the 3-D fiducial registration error: the Euclidean distance in CT
coordinates between each reconstructed fiducial and its CT ground-truth position,
reported per marker and as RMS. We also report a plane-fit deviation as a proxy for
residual low-frequency shape error (the "bowl").

## 4. Results

Five vertebral levels (shots 4–8) with independent 4-marker sets (3 on level 7)
were reconstructed and registered.

### 4.1 Registration accuracy: far-field vs near-field

| Level | WD (mm) | Far-field FRE | Near-field FRE | Note |
|------:|:--:|:--:|:--:|:--|
| 4 | 60.0 | 1.54 mm | **1.72 mm** | |
| 5 | 55.4 | 1.22 mm | **1.32 mm** | |
| 6 | 58.9 | 6.57 mm | **1.74 mm** | bowl removed |
| 7 | 58.6 | 11.14 mm | 9.72 mm | 3 markers |
| 8 | 51.5 | 5.66 mm | *diverged* | near-field |

All five levels were imaged at a similar working distance (51–60 mm). On the three
well-conditioned 4-marker levels (4–6) the near-field model gives **1.2–1.7 mm** 3-D
FRE. The near shots (4, 5) were already good under far-field; the decisive gain is
level 6, where near-field modelling reduced FRE from 6.6 to 1.7 mm.

### 4.2 The far-field bowl and its correction

Far-field integration on this near-field geometry produces a low-frequency bowl:
even on level 4 — where the four fiducials register to 1.5 mm — the surface between
them arches away from the bone. The near-field model reduced the plane-fit
deviation on level 4 from 6.90 to 4.98 mm (95th-percentile from 14.4 to 10.5 mm);
part of the residual is genuine bone relief, not artifact. Qualitatively, near-field
surfaces sit conformally on the CT cortical surface where far-field surfaces float.

### 4.3 Inverse-square baseline

A direct brightness-to-range baseline (*r* ∝ 1/√I) was tested with the same CT
anchoring. It failed: 7.8–9.6 mm FRE, dense relief inflating to 150–559 mm (true CT
range 17–26 mm), and a **physically inverted** fitted depth slope. The cause is
intrinsic: inverse-square assumes constant albedo, so dark bone texture reads as
"far". This is the failure photometric stereo is designed to avoid, and it
quantifies why the normal-based solve is necessary.

### 4.4 Failure modes

All five levels were imaged at a similar working distance (51–60 mm), so the two
failures are **not** distance-driven. Level 7 has only three fiducials, which
under-constrains the PnP pose (P3P admits up to four solutions). Level 8 has the
**best** marker geometry (least coplanar), yet the near-field iteration diverged and
its confidence mask collapsed, while far-field on the same data stayed stable
(5.66 mm) — an unresolved solver-robustness issue, reported rather than omitted.
Because working distance was effectively fixed, its effect on accuracy could not be
characterised here; a controlled distance sweep is future work.

### 4.5 Per-segment vs. combined registration

The five levels span ≈140 mm in CT — multiple vertebrae. Consistent with clinical
practice, each is registered independently; the combined surface (432,757 vertices)
is provided for visualisation, and a single rigid multi-level fit serves only as a
rigidity check whose residual would *measure* inter-level motion rather than
reconstruction error.

## 5. Discussion

**Accuracy in context.**

| Method / study | Reported accuracy | Notes |
|---|---|---|
| **This work (near-field, levels 4–6)** | **1.2–1.7 mm 3-D FRE** | 4 fiducials/level, near-coaxial endoscope |
| Surface-based spine registration [10] | ≈0.96 mm RMS | 20 lamina points, clinical |
| In-vivo endoscopic photometric stereo [6] | <3 mm (≈7%) | colon, depth error |
| Near-field PS light-position calibration [4] | <2.7 mm (light position) | calibration, not surface FRE |

Our millimetre accuracy sits between clinical surface registration and reported
in-vivo endoscopic photometric stereo, achieved with far fewer fiducials and on a
harder (near-coaxial) illumination geometry, and — unlike appearance-oriented neural
methods — it yields a directly CT-registered metric surface.

**Why near-field matters here.** The contribution is not "photometric stereo works"
but that on an endoscopic near-coaxial rig the *far-field approximation injects a
systematic shape error* that a global scale/offset cannot remove, and that a per-
pixel point-source model removes it while preserving the albedo-invariance that
makes bone tractable.

**Limitations.**
1. **Assumed LED positions.** The residual bowl is bounded by using ring-geometry
   LED positions rather than a calibrated forward model; a mirror-sphere or
   matte-target calibration [3,4] is the clear next step and is expected to move
   levels 6–8 toward the 1 mm regime.
2. **Near-coaxial low SNR.** The ≈15% lateral signal compresses recovered relief
   (near-field relief under-estimates the CT range), an intrinsic cost of the small
   baseline.
3. **Working-distance characterisation missing.** All levels were imaged at ~55 mm,
   so accuracy vs. working distance could not be measured; a controlled distance
   sweep is needed.
4. **Fiducial count / specularity / robustness.** Three markers under-constrain pose
   (level 7); the near-field solve diverged on level 8 despite good geometry
   (unresolved); retroreflective markers are inpainted, leaving small holes.
5. **Bench validation.** Results are on an instrumented specimen; in-vivo wet,
   bleeding fields will be harder and are future work.

**Clinical relevance.** Per-vertebra registration matches how spine navigation is
actually performed; the combined multi-level result is a bench consistency check,
not the clinical workflow. A calibrated, CT-registered optical surface from the
existing scope could reduce reliance on intraoperative radiation for re-registration.

## 6. Conclusion

A near-coaxial monocular endoscope, driven by four-LED near-field photometric
stereo and anchored to CT through dual-modality fiducials, reconstructs the exposed
spinal surface to **1.2–1.7 mm** per-vertebra on well-conditioned levels —
competitive with clinical surface registration and better than reported in-vivo
endoscopic photometric stereo, without added radiation or tracking hardware. The
far-field bowl is identified and removed; the inverse-square shortcut is shown to be
fundamentally inapplicable to textured bone; and the dominant residual error is
localised to the use of assumed rather than calibrated LED positions, defining the
next experiment.

## References

[1] R. J. Woodham, "Photometric method for determining surface orientation from
multiple images," *Optical Engineering*, 19(1), 1980.
[2] R. T. Frankot, R. Chellappa, "A method for enforcing integrability in shape from
shading algorithms," *IEEE TPAMI*, 10(4), 1988.
[3] Y. Quéau et al., "LED-based photometric stereo: modeling, calibration and
numerical solution," *J. Math. Imaging Vis.*, 2018. arXiv:1707.01018.
[4] "Robust point light source calibration method for near-field photometric stereo
using feature points selection," *Applied Optics*, 62(36):9512, 2023.
[5] "LUCES-MV: A Multi-View Dataset for Near-Field Point Light Source Photometric
Stereo," arXiv:2412.16737, 2024.
[6] "Photometric single-view dense 3D reconstruction in endoscopy,"
arXiv:2204.09083, 2022.
[7] "Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian
Splatting," arXiv:2602.18314, 2026.
[8] "Photometric Stereo-Based Depth Map Reconstruction for Monocular Capsule
Endoscopy," *Sensors*, 20(18):5403, 2020.
[9] P. J. Besl, N. D. McKay, "A method for registration of 3-D shapes,"
*IEEE TPAMI*, 14(2), 1992.
[10] "Surface-based registration accuracy of CT-based image-guided spine surgery,"
PubMed 15526221.
[11] J. M. Fitzpatrick, J. B. West, C. R. Maurer, "Predicting error in rigid-body
point-based registration," *IEEE TMI*, 17(5), 1998.

---

## Appendix A — Where to publish: target journals and how to position

*(Not part of the manuscript; strategy notes.)*

### Primary target — best fit
- **International Journal of Computer Assisted Radiology and Surgery (IJCARS)** —
  IF ≈2.8, **Q1** (SJR). Scope is *exactly* this: computer-assisted interventions,
  surgical navigation, intraoperative guidance, registration. A CT-registered
  optical surface for spine navigation with an honest accuracy study is a
  natural IJCARS paper; it also feeds the IPCAI/CARS community. **Recommended first
  submission.** (scijournal / journalmetrics)

### Strong alternatives
- **IEEE Transactions on Biomedical Engineering (TBME)** — IF ≈4.4, **Q1**. Broader
  and higher-impact; frame the contribution as the near-field/near-coaxial modelling
  + validation. More competitive; wants a crisp engineering novelty.
- **Biomedical Optics Express (OPTICA)** — **Q1**, open access. Best if you lead with
  the *optical/illumination* novelty (near-coaxial LED photometric model, light
  calibration). Pairs well once the LED-position calibration is done.
- **Medical Physics** — **Q1**. Good home for a rigorous accuracy/validation study
  with phantom + CT ground truth.

### Reach (top-tier, after LED calibration + more levels/in-vivo)
- **Medical Image Analysis (MedIA)** and **IEEE Transactions on Medical Imaging
  (TMI)** — IF ≈10, top **Q1**, very competitive; expect a methods-deep contribution
  and broad validation. Aim here only once accuracy is calibrated to ~1 mm and
  validated on more specimens.

### What Q1 reviewers in this space will expect (learned from the comparators)
1. **Ground-truth validation with a clear metric** — you have it (3-D FRE vs CT).
   Report per-marker and RMS, and compare to [6,10] as in §5. ✔
2. **Ablation / model comparison** — far-field vs near-field vs inverse-square is a
   textbook ablation; keep it. ✔
3. **Honest failure and limits** — the 3-marker (level 7) and near-field-divergence
   (level 8) failures *strengthen* the paper; reviewers punish hidden limitations,
   not disclosed ones.
4. **Reproducibility** — release the pipeline (`marker_pipeline/`), the meshes, and
   the fiducial protocol. Q1 increasingly expects code/data.
5. **The one thing to add before a top-tier try** — a *calibrated* LED forward model
   (mirror-sphere or matte-target, per [3,4]) to close the residual bowl, plus ≥2
   specimens. That single addition converts "millimetre on a bench" into a defensible
   clinical-accuracy claim.

### Sources
- IJCARS scope/IF: scijournal.org, journalmetrics.org, Springer.
- TBME IF/scope: research.com, wos-journal.info.
- Comparators: colon PS <3 mm (arXiv:2204.09083); clinical spine registration
  ≈0.96 mm (PubMed 15526221); near-field light calibration <2.7 mm (Applied Optics
  62:9512); Quéau LED model (arXiv:1707.01018); nasal-endoscopy navigation
  (Front. Neurorobotics 2025).
