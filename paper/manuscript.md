# Retroreflective-Marker-Anchored Endoscopic Surface Measurement for Registration to CT: Pipeline Description and a CT-Validated Failure of the Far-Field Photometric Model

**Authors.** *[AUTHOR LIST — TO BE COMPLETED]*
**Affiliations.** *[AFFILIATIONS — TO BE COMPLETED]*
**Corresponding author.** *[EMAIL — TO BE COMPLETED]*

**Manuscript status.** Draft v0.2. Sections dealing with endoscope-to-CT
registration (§2.11, §3.7, and parts of §4) are **placeholders** and are marked
as such throughout. Every number reported outside those marked blocks is
reproduced from the committed pipeline outputs on the `marker-pipeline` branch
and is traceable to a named file and column (Appendix B). Reference entries carry
an explicit verification status (Appendix D); several could not be confirmed
against a publisher record and must be checked before submission.

---

## Abstract

*[Structured to five labelled blocks — Background / Purpose / Methods / Results /
Conclusions — so that sections can be merged or relabelled to whichever target
journal's abstract format applies. See Appendix C, item M6.]*

**Background.** Image-guided procedures on bone require a spatial correspondence
between what an intra-operative endoscope sees and a pre-operative CT volume.
Establishing that correspondence — registration — requires the endoscopic view to
yield metric three-dimensional surface information, which a single monocular
endoscope does not natively provide.

**Purpose.** To document a complete four-LED endoscopic measurement pipeline for
recovering marker coordinates and lens-to-marker distances suitable as
registration inputs, and to evaluate its depth channel against independent CT
ground truth.

**Methods.** The pipeline recovers per-pixel surface relief by photometric
stereo, localizes retroreflective fiducial markers in the same image frame, and
reports each marker as a three-dimensional coordinate together with a
lens-to-marker distance. Five capture sessions (`shot_011`–`shot_015`) of an
ex-vivo vertebral specimen were processed. One session (`shot_011`) carries
independent CT ground truth for three of its markers. Marker geometry is
transferred into the camera frame by a P3P solution disambiguated using a
confirmed physical ordering constraint.

**Results.** Marker detection recovered 18 markers across the five sessions, 14
of which (78%) were independently confirmed in all four single-LED frames.
Recovered relative depth spanned 3.69–5.68 mm within a session (pooled standard
deviation 2.25 mm). At the one session with CT ground truth, the photometrically
recovered depth was **anti-correlated** with the CT-derived lens-to-marker
distance (Pearson *r* = −0.923; *r* = −0.891 against camera-frame CT depth), and
the near-to-far ordering of the three CT markers was exactly reversed. A
least-squares affine rescaling onto the CT control points returned a **negative**
slope (*z*<sub>corr</sub> = −4.218·*z*<sub>est</sub> − 5.791) and still left a
4.36 mm RMS residual across a 20.7 mm true depth span. We attribute this
inversion to the use of a far-field illumination model on a rig whose LEDs sit
12.08 mm off-axis at a 30 mm working distance — squarely in the near field.

**Conclusions.** Marker pixel localization and the CT/P3P lens-to-marker
distances are reliable and usable as registration inputs. The photometric depth
channel, in its current far-field formulation, is not: it is sign-inverted
against the only available ground truth and must not be propagated into a
registration result. We report the near-field forward model and LED calibration
procedure that are implemented and numerically validated but currently blocked on
uncollected calibration captures. *[REGISTRATION ACCURACY RESULT — PLACEHOLDER,
see §3.7.]*

**Keywords.** endoscopy; photometric stereo; near-field illumination;
retroreflective fiducials; CT registration; surgical navigation

---

## 1. Introduction

Registering an endoscopic view to a pre-operative CT volume is the enabling step
for image-guided intervention on bone. The registration problem is well posed
only when the endoscopic side of the correspondence carries three-dimensional,
metrically-scaled information. A monocular endoscope does not supply this
directly, so it must be recovered — either from motion, from structured
illumination, or, as here, from shading.

Photometric stereo recovers surface orientation from the way brightness changes
as the illumination direction changes, and it is attractive for endoscopy because
the light sources are already integrated into the instrument tip and no
additional hardware enters the surgical field. The classical formulation, however,
assumes *far-field* illumination: that each light source is distant enough that
its direction is constant across the field of view and its intensity does not
fall off within the scene. Endoscopic rigs violate both assumptions
comprehensively. On the instrument described here the LEDs sit 12.08 mm off the
optical axis at a working distance of 30 mm, so the illumination direction
sweeps across the field and the inverse-square falloff is a first-order effect,
not a correction term.

This paper documents a complete pipeline built around that instrument and, as its
principal empirical contribution, reports a **CT-validated failure** of the
far-field model on real data. Rather than presenting a depth result and
qualifying it, we present the diagnostic that establishes the depth result is
sign-inverted, quantify the inversion, and separate the pipeline's trustworthy
outputs from its untrustworthy ones. We regard this as the more useful
contribution: the far-field shortcut is widely taken in endoscopic shading work,
and the magnitude and *sign* of the resulting error appear not to have been
measured against ground truth in this geometry.

The paper is organized as a description of each pipeline stage (§2), the results
each stage produced on the five-session dataset (§3), and a discussion of what
this implies for the registration problem the pipeline exists to serve (§4). The
registration stage itself is under active development; its sections are marked as
placeholders and enumerated in Appendix C.

### 1.1 Contributions

1. A documented, reproducible four-LED endoscopic photometric-stereo pipeline
   with retroreflective fiducial localization, reporting per-marker (*x*, *y*,
   *z*) coordinates and lens-to-marker distances (§2.4–§2.8).
2. A P3P-based transfer of CT marker geometry into the camera frame, with an
   explicit and physically-grounded resolution of the P3P pose ambiguity
   (§2.8.1).
3. A quantified, CT-validated demonstration that far-field photometric stereo
   inverts depth ordering in near-field endoscopic geometry (§3.4).
4. An implemented and numerically-validated near-field forward model and LED
   calibration procedure, together with an explicit statement of the calibration
   captures required to deploy it (§2.10).
5. *[REGISTRATION CONTRIBUTION — PLACEHOLDER, see §2.11.]*

---

## 2. Materials and Methods

### 2.1 Optical rig and measured geometry

The instrument is a monocular endoscope with four LEDs mounted symmetrically
around the lens and driven as two opposing pairs (LEDs 1–3 and 2–4). All
quantities in this paper are expressed in a single right-handed camera frame:
**X to the right, Y downward, Z from the lens into the scene, origin at the
optical centre, millimetres.** The four LEDs are treated as lying at azimuths of
0°, 90°, 180° and 270°, designated RIGHT, TOP, LEFT and BOTTOM respectively.

Table 1 gives every geometric parameter the pipeline consumes, together with
**how each was obtained** — a distinction that matters here because the
measured and calibration-derived values disagree (§2.3).

**Table 1.** Rig and imaging parameters. Frame: X right, Y down, Z into the
scene, origin at the optical centre; all lengths in millimetres unless stated.
Source: `scripts/reconstruction/nearfield_lambertian.py`,
`scripts/reconstruction/bone_depth_batch.py`.

| Symbol / constant | Value | Provenance | Notes |
|---|---|---|---|
| `LED_OFFSET_MM` | 12.08 mm | **measured** (bench, 2026-07-08) | lateral LED-to-lens distance |
| `FIELD_WIDTH_MM` | 37.0 mm | **measured** (bench, 2026-07-07) | width of imaged field |
| `WORKING_DISTANCE_MM` | 30.0 mm | **measured** (bench) | lens-to-surface distance |
| *θ*<sub>elev</sub> | 68.1° | **derived** from the above | illumination elevation |
| *s* (mm/px) | 0.05781 | **derived** | at 640 px working width |
| *f<sub>x</sub>* | 519 px | **derived** from measurements | `measured_intrinsics()`, 640 px width |
| *f<sub>x</sub>* (checkerboard) | 794 px | **calibrated** | 1588 px native; **conflicts**, see §2.3 |
| `WORK_LONG_EDGE` | 640 px | **configuration** | working resolution |
| *θ*<sub>1/2</sub> | *[NOT YET RECORDED]* | **manufacturer datasheet** | LED half-intensity angle; required by §2.10 |

The illumination elevation is derived rather than assumed:

$$\theta_{\text{elev}} = \arctan\!\left(\frac{d_{\text{work}}}{r_{\text{LED}}}\right)
 = \arctan\!\left(\frac{30.0}{12.08}\right) = 68.1^{\circ},$$

and the in-plane spatial scale at the working resolution is

$$s = \frac{\texttt{FIELD\_WIDTH\_MM}}{W} = \frac{37.0}{640} = 0.05781\ \text{mm}\,\text{px}^{-1}.$$

### 2.2 Data collection protocol

#### 2.2.1 Acquisition invariants

Photometric stereo rests on a specific acquisition contract: the viewpoint is
held fixed while the illumination is varied, so that inter-frame pixel
correspondence is known *a priori* rather than estimated. We state that contract
explicitly (Table 2), because every downstream step depends on it — in
particular the cross-frame marker consolidation of §2.4.3, which is valid only
because the camera and specimen do not move within a session.

**Table 2.** Acquisition invariants within one capture session.

| Quantity | Status within a session | Consequence |
|---|---|---|
| Camera pose | **held constant** | frames are pixel-registered by construction |
| Specimen pose | **held constant** | a marker occupies the same pixel in every frame |
| Marker placement | **held constant** | cross-frame consolidation is a confidence check |
| Illumination direction | **varied** (4 LEDs, one per frame) | the photometric-stereo signal |
| Ambient light | **measured once** (`dark.png`) | subtracted; see §2.5(i) |
| Exposure | nominally constant, **balanced in software** | see §2.5(iii) |

#### 2.2.2 Per-session file manifest

Each session is a directory containing a fixed five-file manifest. We list which
files the depth solve consumes and, equally important, which it does **not**:

**Table 3.** Per-session capture manifest.

| File | Content | Used by the depth solve? |
|---|---|---|
| `led1.png` | RIGHT LED only | **yes** — photometric stereo |
| `led2.png` | TOP LED only | **yes** — photometric stereo |
| `led3.png` | LEFT LED only | **yes** — photometric stereo |
| `led4.png` | BOTTOM LED only | **yes** — photometric stereo |
| `dark.png` | all LEDs off, ambient only | **yes** — subtracted from each LED frame |

A session is admitted to processing only if all four `led*.png` frames are
present; `discover_shots()` enforces this and silently skips incomplete
directories. All five files are additionally used for marker detection, which
runs independently on each of the four LED frames (§2.4.3).

#### 2.2.3 Acquisition hierarchy

The dataset analysed here is **5 sessions × 5 exposures = 25 images**, yielding
18 consolidated markers. Ground truth is available for **1 session and 3 of its
4 markers**. This is a small *n*, and it is bounded by a stated constraint rather
than by convenience: CT marker coordinates exist only for `shot_011`, so the
depth channel can be checked at exactly three points. Expanding that check
requires additional CT acquisition (Appendix C, item R2), not additional
endoscopy.

#### 2.2.4 Capture conditions

> ⚠️ **PLACEHOLDER — CAPTURE CONDITIONS NOT YET DOCUMENTED.**
> The following must be recorded from the capture sessions and stated here, since
> a photometric method is not reproducible without them: whether the LEDs were the
> only light source present; whether camera auto-exposure and auto-white-balance
> were disabled; the exposure/aperture settings and whether any frame contained
> saturated pixels; the ambient level measured by `dark.png`; and the specimen
> preparation and marker attachment method. See Appendix C, item M3.

### 2.3 Camera model and the intrinsic-scale ambiguity

Images are undistorted using the stored calibration and resized so that the long
edge is `WORK_LONG_EDGE` = 640 px. A pinhole model is used thereafter, with

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{bmatrix},$$

the principal point placed at the image centre.

The pipeline does **not** use the checkerboard-calibrated focal length. The
function `measured_intrinsics()` instead derives intrinsics from the two bench
measurements of Table 1:

$$f_x = \frac{W \cdot d_{\text{work}}}{\texttt{FIELD\_WIDTH\_MM}}
      = \frac{640 \times 30.0}{37.0} = 519\ \text{px},
\qquad
f_y = \frac{H \cdot d_{\text{work}}}{\tfrac{3}{4}\texttt{FIELD\_WIDTH\_MM}} .$$

The 3:4 field-height convention reproduces the calibration's own
*f<sub>x</sub>*/*f<sub>y</sub>* = 1.33 anisotropy on the native 4:3 sensor.

This is a substantive and unresolved discrepancy and must be reported as such.
The checkerboard calibration returns *f<sub>x</sub>* = 1588 px at native
resolution, i.e. **794 px** at the 640-px working width — a factor of **1.53×**
above the measurement-derived 519 px. Equivalently, the calibrated focal length
implies a 45.9 mm working distance where 30 mm was measured, and a ~44° field of
view where 37 mm at 30 mm implies ~63°. Every absolute distance reported in this
work therefore carries an unresolved ~1.5× scale uncertainty; relative
comparisons *within* a session are unaffected.

We resolve the conflict in favour of the bench measurements on the grounds that
the two measurements are mutually consistent while the checkerboard result
contradicts both. This is an arbitration, not a resolution.

> ⚠️ **PLACEHOLDER — CALIBRATION RESIDUALS NOT REPORTED.**
> The checkerboard calibration's reprojection-error statistic, the number and
> spread of calibration poses, and the distortion coefficients are not currently
> recorded in the repository. A calibration reported without its residuals cannot
> be assessed, and the residual may itself explain the 1.53× discrepancy. See
> Appendix C, item C2.

### 2.4 Retroreflective marker detection

Markers are retroreflective glass-bead disks affixed to the specimen. Detection
is implemented in `detect_trackers.py` and exploits a signature specific to the
bead coating rather than simple brightness thresholding, which would confuse
markers with specular glints on wet bone.

#### 2.4.1 Sparkle (high-frequency energy) map

The discriminating cue is texture: the glass-bead coating produces
high-frequency "sparkle" where bone is smooth. `local_texture()` computes a
local RMS of the Laplacian,

$$T(\mathbf{p}) = \sqrt{G_{\sigma=3}\!\left[\left(\nabla^2 I\right)^2\right](\mathbf{p})},$$

with a 3×3 Laplacian kernel and Gaussian smoothing of σ = 3 px.

#### 2.4.2 Candidate generation and scoring

Two detectors run in parallel and their outputs are pooled:

- **Blob detector** (`_blob_candidates`, primary). The sparkle map is normalized,
  blurred (σ = 2), and thresholded at max(*μ* + 1.2*σ*, 40). Morphological
  closing (9×9 ellipse) then opening (5×5 ellipse) cleans the mask. Connected
  components are retained only if their area falls within
  [π(0.6·*r*<sub>min</sub>)², π(1.4·*r*<sub>max</sub>)²], their bounding-box
  aspect ratio exceeds 0.45 (rejecting elongated highlights), and their fill
  ratio exceeds 0.45 (requiring a solid disk).
- **Hough circle detector** (`_hough_candidates`, secondary), with `dp` = 1.2,
  `param1` = 110, `param2` = 22, over radii *r* ∈ [12, 60] px.

Each candidate is scored by `_score()` over an inner disk of radius 0.8*r* and a
surrounding annulus spanning 1.1*r* to 1.7*r*, yielding four features: interior
sparkle, interior-minus-ring brightness contrast, raw interior brightness, and
raw ring brightness. Near-coincident candidates are deduplicated by `_dedupe()`,
which retains the sparkliest of any cluster within 48 px.

Acceptance (`filter_candidates`) requires passing a raw-brightness floor
(default 45) — a genuine retroreflector returns a great deal of light even in a
dim frame — and then satisfying **any one** of three signatures:

**Table 4.** Marker acceptance signatures. Defaults as called from
`marker_depth_slides.py`.

| Signature | Condition | Rationale |
|---|---|---|
| `strong` | sparkle ≥ 130 | unambiguous mid-lit bead texture |
| `ringed` | sparkle ≥ 80 **and** contrast ≥ 25 | dimmer bead, confirmed by its dark rim |
| `bright` | raw ≥ 130 **and** contrast ≥ 30 **and** ring raw < 145 | saturated bead whose sparkle has washed out |

The ring-brightness ceiling in the `bright` path is what separates a marker from
a specular glint: a genuine marker sits in a dark surround, whereas a glint on
bright bone is surrounded by bright bone and is rejected.

#### 2.4.3 Cross-frame consolidation

Because the camera and markers are static within a session while only the
illumination moves (Table 2), a physical marker appears at essentially the same
pixel in every LED frame that captured it. `aggregate_by_shot()` clusters
detections across the four frames within a merge radius (35 px as called from the
slide generator), averages position and radius, and records **`n_frames`**: the
number of the four single-LED frames in which that marker was independently
detected.

`n_frames` is the pipeline's per-marker confidence statistic and is carried
through to the output CSV. A marker at `n_frames` = 4 is a four-fold independent
confirmation under four different illumination directions; `n_frames` = 1 is a
single detection warranting visual review.

### 2.5 Four-light photometric stereo depth

Depth is recovered by the photometric-stereo core in `bone_depth_batch.py`,
invoked with four lights. The stages are as follows.

**(i) Ambient subtraction.** Each LED frame has the dark frame subtracted,
*I*′ = clip(*I* − *I*<sub>dark</sub>, 0, 1). This is the mechanism by which the
model's "no ambient term" assumption is made to hold, and it is why `dark.png`
is part of the required manifest rather than an optional extra.

**(ii) Linear luminance.** `solve_luminance()` inverts the sRGB transfer
function, converts to a single brightness channel, and inpaints specular
highlights from surrounding pixels. Specular removal matters here for a specific
reason: the retroreflective markers are themselves strongly non-Lambertian, so
they are inpainted from the surrounding bone. The depth subsequently sampled at a
marker is therefore the depth of the *bone seat beneath it*, not of the bead —
the intended behaviour.

**(iii) Exposure balance.** `balance_exposure()` divides each frame by its median
brightness, so that brightness differences between frames reflect surface
orientation rather than camera auto-exposure.

**(iv) Normal recovery.** With unit light vectors **l**<sub>*i*</sub> assembled
from the azimuths of §2.1 and the elevation of 68.1°, the Lambertian image
formation model for each pixel is

$$\mathbf{L}\,\mathbf{g} = \mathbf{I}, \qquad
\mathbf{L}\in\mathbb{R}^{4\times3}, \quad
\mathbf{I} = \big(I_1, I_2, I_3, I_4\big)^{\!\top},$$

solved in the least-squares sense (four lights, three unknowns — an overdetermined
system, in contrast to the exactly-determined three-light case). Albedo and unit
normal follow as *ρ* = ‖**g**‖ and **n** = **g**/*ρ*.

**(v) Gradient field.** `normals_to_depth()` converts normals to depth gradients
under the repository convention that larger *z* means farther from the camera:

$$\frac{\partial z}{\partial u} = +\frac{n_x}{n_z}, \qquad
  \frac{\partial z}{\partial v} = -\frac{n_y}{n_z}.$$

**(vi) Integration.** `integrate_frankot_chellappa()` recovers the depth map whose
gradients best match that field, by an FFT Poisson solve.

The result is a **relative** depth map in working-resolution pixel units. It is
converted to millimetres by the scale *s* of Table 1 and re-centred on the median
depth over a bone mask derived from the albedo (`bone_mask(albedo)`), so that
*z* = 0 denotes the median surface depth of the session and positive *z* means
farther from the lens.

#### 2.5.1 Assumption ledger

Because this stage is the one that fails (§3.4), we state each assumption it
makes and its status explicitly rather than leaving them implicit.

**Table 5.** Status of the classical photometric-stereo assumptions on this rig.

| Assumption | Status | How handled / why it matters |
|---|---|---|
| Lambertian reflectance | **relaxed** | specular highlights detected and inpainted, §2.5(ii) |
| No ambient term | **handled** | `dark.png` subtracted per frame, §2.5(i) |
| Known light directions | **assumed** | derived from bench geometry, not calibrated, §2.1 |
| Equal source intensity | **assumed** | not calibrated; exposure balance only partially compensates |
| Constant light direction over field (far-field) | **VIOLATED** | LED offset/working distance = 0.40; see §3.4, §4.2 |
| No intensity falloff within scene (far-field) | **VIOLATED** | 1/*r*² is first-order at 30 mm; see §4.2 |
| Orthographic projection | **assumed** | perspective ignored in this path; addressed in §2.10 |
| Integrable normal field | **enforced** | by the Frankot–Chellappa step, §2.5(vi) |

The two violated rows are the subject of the paper's principal result. No
radiometric correction (e.g. vignetting compensation) is currently applied; if
one is added it must be documented here with its formula, since reported
intensities are otherwise not reproducible.

### 2.6 Composite slide generation

`marker_depth_slides.py` is the top-level driver: it processes each session and
emits a single composite figure plus a row per marker in `marker_depths.csv`.

An important implementation detail concerns pixel correspondence between the
depth solve and the detector. `undistorted_bgr()` applies exactly the same
undistortion and resize as the depth path but retains 8-bit BGR, so the detector
operates on precisely the pixel grid the depth map is defined on. Marker
coordinates therefore index the depth map directly, with no resampling between
the two.

Each generated figure (Fig. 1) is laid out on a 3×4 grid: the four single-LED
frames with their light vectors across the top; a table of per-marker
(*x*, *y*, *z*) in millimetres; the depth heat map (turbo colormap, clipped to
the 2nd–98th percentile over the bone mask) with each marker circled and
numbered; and the depth field as a 3-D surface with a vertical stem at each
marker, plotted as −*z* so that protrusions point toward the viewer. A footer
records the measured rig geometry and states that the depth is relative and
uncalibrated in absolute scale.

<figure>
<img src="../depth_outputs/marker_depth_slides/shot_011_marker_depth.png" alt="Composite session figure for shot_011">
<figcaption><strong>Figure 1.</strong> Composite session figure, <code>shot_011</code>, as emitted by <code>marker_depth_slides.py</code>. Top row: the four single-LED exposures, each captioned with its LED designation and unit light vector. Middle: the recovered marker coordinate table. Bottom left: the relative depth field with all four detected markers circled and numbered. Bottom right: the same field as a 3-D surface with a vertical stem at each marker. The footer records the measured rig geometry. Note that the depth field shown here is the far-field result subsequently shown to be inverted (§3.4). <em>File:</em> <code>depth_outputs/marker_depth_slides/shot_011_marker_depth.png</code></figcaption>
</figure>

### 2.7 Marker coordinate extraction

Depth at a marker is read by `marker_depth()` as the **median** depth over the
marker's disk, with the radius floored at 6 px. The median is chosen over the
mean for robustness to any residual specular contamination inside the disk.

The reported coordinate triple for marker *k* is

$$(x_k, y_k, z_k) = \big(u_k\,s,\; v_k\,s,\; \tilde{z}(\mathcal{D}_k)\big),$$

where *u*, *v* are pixel coordinates, *s* is the mm/px scale, and
*z̃*(𝒟<sub>*k*</sub>) is the median depth over the marker disk. Here *x* and *y*
are true in-plane millimetre positions across the measured field, whereas *z* is
a *relative* depth about the session median. This asymmetry is important when
interpreting the output: the coordinate triple mixes two different kinds of
quantity, and only the first two are metric in the ordinary sense.

### 2.8 Lens-to-marker distance

`marker_distances.py` reports a distance from the lens to each marker, and
deliberately keeps two methods of sharply different reliability separate and
explicitly labelled in the output.

#### 2.8.1 CT-validated distance by P3P (shot_011 only)

For `shot_011`, three markers have known positions in the CT frame:

**Table 6.** CT ground-control points, `shot_011` (CT frame, mm). Marker 2 has no
CT correspondence.

| Marker | *X* | *Y* | *Z* |
|---|---|---|---|
| 1 | 9.224 | −1.632 | 11.818 |
| 3 | 3.729 | −13.209 | 23.259 |
| 4 | −17.665 | 0.732 | 17.412 |

Because CT coordinates live in the scanner frame and not the camera frame, they
cannot be compared to camera depths directly; a pose must be solved first. The
pipeline solves P3P on the three (CT point ↔ detected pixel) correspondences
using the measured intrinsics of §2.3, then transforms the CT points into the
camera frame, **x**<sub>cam</sub> = **R x**<sub>CT</sub> + **t**, and reports
‖**x**<sub>cam</sub>‖ as the lens-to-marker distance.

**Resolving the P3P ambiguity.** Three correspondences admit up to four
geometrically valid poses, and the pipeline enumerates them rather than assuming
the solver returns the right one. Two filters are applied in sequence:

1. **Cheirality** — all three points must lie in front of the camera (*Z* > 0).
2. **Physical ordering** — the recovered pose must place **marker 4 nearest the
   lens**, independently confirmed as a fact about the capture and encoded as
   `CLOSEST_MARKER = 4`.

If no pose satisfies both, the pipeline raises rather than returning a
best-effort answer. This is a legitimate and, in our view, underused
disambiguation strategy: a single confirmed ordinal fact about the scene selects
among the P3P branches and requires no extra instrumentation.

*[TO REPORT: the number of candidate poses actually returned for `shot_011`, and
how many survived each filter. The code enumerates them but does not currently
log the count. See Appendix C, item R7.]*

#### 2.8.2 Photometric distance (all other markers)

Where no CT is available, distance can only be inferred from the photometric
depth. The marker-set median depth is anchored at `WD_ANCHOR_MM` = 46.0 mm — the
median distance recovered by the `shot_011` P3P solution — and the relative depth
added, then back-projected along the pixel ray:

$$Z_k = \texttt{WD\_ANCHOR\_MM} + z_k, \qquad
d_k = Z_k \left\lVert \left(\tfrac{u_k-c_x}{f_x},\ \tfrac{v_k-c_y}{f_y},\ 1\right) \right\rVert .$$

These values are emitted **only so that the raw numbers are on record**. Every
such row is tagged in the `method` column as
`photometric (UNRELIABLE: inverted vs CT@011)` and carries the note
`far-field depth; scale/sign untrusted`. Section 3.4 gives the justification.
They must not be used as registration inputs.

### 2.9 CT-anchored depth correction

`ct_correct_depth.py` implements two correction models mapping relative
photometric depth onto CT control points; `apply_ct_shot011.py` applies the first
to `shot_011` as a diagnostic.

**Affine-*z*.** Two unknowns, requiring ≥ 2 control markers:
*z*<sub>true</sub> ≈ *a z*<sub>est</sub> + *b*. This removes the global scale and
offset that relative depth is missing. It does **not** remove any
spatially-varying bias.

**Tilt-plane.** Three unknowns, requiring ≥ 3 non-collinear control markers:
*z*<sub>true</sub> ≈ *z*<sub>est</sub> + (*αx* + *βy* + *γ*). This additionally
absorbs the first-order component of the spatial "bowl" the far-field model
introduces.

Both are ordinary least-squares fits. With exactly the minimum number of control
points each interpolates — residuals are then zero by construction and carry no
information — so the models may be trusted only as far as the control points
spread.

### 2.10 Near-field forward model and LED calibration (implemented; not yet deployed)

The far-field assumption is the identified root cause of the depth failure
(§3.4), and the two modules that replace it are implemented and numerically
validated but cannot be run on real data because the required calibration
captures do not exist. We describe them because they define the corrective path.

#### 2.10.1 Near-field solver

`nearfield_lambertian.py` replaces the orthographic/far-field treatment with a
perspective, inverse-square formulation. Under perspective projection
**P** = *Z*·((*u*−*c<sub>x</sub>*)/*f<sub>x</sub>*,
(*v*−*c<sub>y</sub>*)/*f<sub>y</sub>*, 1), requiring the surface normal to be
orthogonal to the two surface tangents yields gradients in **log** depth:

$$\frac{\partial \ln Z}{\partial u} = -\frac{n_1}{f_x D}, \qquad
\frac{\partial \ln Z}{\partial v} = -\frac{n_2}{f_y D}, \qquad
D = n_1\frac{u-c_x}{f_x} + n_2\frac{v-c_y}{f_y} + n_3 .$$

Integrating in log space and exponentiating about a depth anchor gives a metric
depth map rather than a relative one. The solver iterates: estimate normals,
back-project to a point cloud, re-evaluate the per-pixel light directions and
inverse-square attenuation at those points, and re-solve.

Critically, the module was upgraded to **require** a solved
`led_calibration.json` rather than assume rig geometry. It will not silently fall
back to assumed symmetric LED placement, equal brightness, or an on-axis
principal direction — a deliberate choice, since an assumed-geometry fallback is
precisely how the current far-field result came to look credible.

#### 2.10.2 LED calibration

`led_calibration.py` implements the calibration of Quéau et al. [1], §2.2, whose
per-LED image formation model is

$$I^i(\mathbf{p}) = \Psi_i\,\rho(\mathbf{p})
\left[\frac{\mathbf{n}_s^i\cdot(\mathbf{x}-\mathbf{x}_s^i)}{\lVert\mathbf{x}-\mathbf{x}_s^i\rVert}\right]^{\mu_i}
\frac{\big\{(\mathbf{x}_s^i-\mathbf{x})\cdot\mathbf{n}(\mathbf{p})\big\}_+}{\lVert\mathbf{x}_s^i-\mathbf{x}\rVert^{3}} ,$$

recovering for each LED its position **x**<sub>*s*</sub>, principal direction
**n**<sub>*s*</sub>, anisotropy exponent *μ*, and intensity Ψ. The bracketed term
is the cos<sup>*μ*</sup> angular emission profile; the final factor combines
Lambert's cosine law with inverse-square attenuation.

**Table 7.** Per-LED parameters recovered by calibration, and their status.

| Parameter | Dimension | Units / frame | How obtained | Status |
|---|---|---|---|---|
| **x**<sub>*s*</sub> | 3×1 per LED | mm, camera frame | mirror-ball triangulation | **blocked** — captures missing |
| **n**<sub>*s*</sub> | 3×1 per LED, unit | camera frame | white-plane least squares | **blocked** — captures missing |
| Ψ | scalar per LED | relative | white-plane least squares | **blocked** — up to one common factor |
| *μ* | scalar per LED | dimensionless | **manufacturer** half-angle *θ*<sub>1/2</sub> | value not yet recorded |

Calibration proceeds in three independent steps: *μ* from the datasheet
half-angle; **x**<sub>*s*</sub> by triangulating rays reflected off a spherical
mirror across many poses; and **n**<sub>*s*</sub>, Ψ by linear least squares on a
matte white Lambertian plane with **x**<sub>*s*</sub> and *μ* known, using the
substitution **m**<sub>*s*</sub> = Ψ<sup>1/*μ*</sup>**n**<sub>*s*</sub> to
linearize the cos<sup>*μ*</sup> nonlinearity.

**Blocking dependency.** Deployment requires capturing a mirror ball at roughly
10 poses per LED and a matte white card at 3–5 tilts per LED. These captures do
not exist and are the single blocking item for a physically-correct depth path.

### 2.11 Registration of endoscopic measurements to CT

> ### ⚠️ PLACEHOLDER — THIS IS THE PAPER'S PRINCIPAL SECTION AND IS NOT YET WRITTEN
>
> **This entire subsection is scaffolding.** No registration method has been
> finalized and no registration result exists. The prose below states the
> *intended* structure so the surrounding manuscript reads continuously; every
> bracketed value is a placeholder, not a measurement. Do not circulate this
> section. See Appendix C, items R1–R8.

#### 2.11.1 Error nomenclature

The registration literature distinguishes three error quantities that are
routinely conflated, and the distinction determines what this paper may claim.
Each must be defined at first use with the exact expression used to compute it,
and every reported value must state whether it is a Euclidean magnitude or
per-axis, and whether it is a mean, RMS, or median.

**Table 8.** Error quantities. *[Definitions to be finalized with the method.]*

| Symbol | Name | Definition | Role here |
|---|---|---|---|
| FLE | fiducial localization error | error in locating a fiducial in each modality | *[to be estimated — §2.11.3]* |
| FRE | fiducial registration error | residual at the points used to fit the transform | **diagnostic only — not an accuracy claim** |
| TRE | target registration error | error at points *not* used to fit the transform | **the accuracy endpoint** |

**This paper will report TRE as its accuracy metric.** FRE will be reported for
completeness as a within-registration residual, accompanied by the explicit
statement that FRE is uncorrelated with TRE and therefore carries no information
about registration accuracy for a given case [2,3]. Leave-one-out FRE-based
surrogates are likewise not offered as a substitute, having also been reported as
poor indicators of true TRE [4]. Evaluation targets must be points not used to
compute the transform, and if any evaluation target doubles as a registration
fiducial that must be disclosed, since the resulting TRE is optimistically
biased.

#### 2.11.2 Transform model

The registration problem is to recover the similarity transform
𝒯 = (*s*, **R**, **t**) mapping endoscopic measurements into the CT frame, where
the uniform scale *s* is required because the endoscopic depth channel is
relative rather than metric.

*[METHOD — TO BE WRITTEN. The intended approach is marker-based: the
retroreflective markers localized in §2.4 are also identifiable in the CT volume,
giving direct point correspondences and therefore a closed-form solution rather
than an iterative surface fit. With *n* ≥ 3 non-collinear correspondences a
closed-form similarity solution applies, minimizing]*

$$\mathcal{T}^\star = \arg\min_{s,\mathbf{R},\mathbf{t}}
\sum_{k=1}^{n} \big\lVert\, s\mathbf{R}\mathbf{x}_k^{\text{endo}} + \mathbf{t} - \mathbf{x}_k^{\text{CT}} \,\big\rVert^2 .$$

*[The estimator must be named precisely (closed-form point-set vs iterative;
weighted vs unweighted, and the weighting scheme if weighted), and the guard
against the reflection failure mode reported: state that det(**R**) = +1 was
verified on every registration and report any degenerate branch hits. Error
should be decomposed by parameter — rotation, translation, and scale reported
separately with their own figures — rather than collapsed into one lumped RMS.]*

*[The relationship between this and the P3P pose of §2.8.1 must be made explicit
— the P3P solution is already a rigid registration of three markers and is
arguably the paper's existing registration result; whether the final method
generalizes that or replaces it is UNDECIDED.]*

#### 2.11.3 Fiducial configuration

*[TO BE WRITTEN — REQUIRED, NOT OPTIONAL. Configuration shape governs TRE:
widely spread markers with the configuration centroid near the target minimize
it, while near-collinear configurations far from the target maximize it [5]. This
subsection must report, per session: the number of fiducials, their coordinates
(or full geometry in supplementary material), their spatial spread, the position
of the configuration centroid relative to each target, and any near-collinearity
or near-coplanarity. The configuration must be justified, not merely counted.
Existing tooling already emits a `near_planar_warning` flag for exactly this
condition and it must be reported.]*

*[The FLE model must also be stated rather than assumed silently: whether FLE is
modelled as zero-mean isotropic Gaussian or as anisotropic/inhomogeneous, and how
it was estimated. Every TRE prediction is a function of FLE. Note that our two
modalities plausibly have different and possibly anisotropic FLE — CT voxel
anisotropy on one side, and the sub-pixel centroid precision of §2.4 on the
other.]*

#### 2.11.4 Validation protocol

*[TO BE WRITTEN. A five-part skeleton is intended: (1) design of the evaluation
dataset; (2) ground-truth definition **and its own accuracy**; (3) evaluation
criteria; (4) evaluation metrics; (5) evaluation protocol. Accuracy, precision,
robustness and stability are to be reported as four separate quantities rather
than merged.]*

*[Leave-one-marker-out cross-validation across the five sessions is the obvious
protocol given n = 18 markers, but with only 3 CT-known markers in a single
session it is not currently executable. This is a data-collection dependency, not
an analysis dependency. The number of independent registrations per condition and
how repeats were generated (re-localization, re-acquisition, or Monte-Carlo
perturbation of FLE) must be stated — a TRE from a single registration is an
anecdote, not a measurement.]*

> ⚠️ **GROUND-TRUTH UNCERTAINTY NOT YET QUANTIFIED.** A validation cannot resolve
> error below the uncertainty of its own reference standard. The CT's voxel size,
> reconstruction kernel, and the localization uncertainty of the marker centroids
> *within* the CT volume are not currently recorded, so no accuracy claim finer
> than that unstated floor can be defended. See Appendix C, item R8.

**Known methodological hazard, to be addressed explicitly.** An earlier
exploratory attempt in this project registered dense reconstructed surface
patches to the CT bone surface by automatic similarity ICP without
correspondences. It produced a superficially attractive ~1.9 mm median surface
deviation that was, on inspection, an artifact: the patches draped onto a single
region of the bone rather than localizing to their true anatomical sites, and a
near-planar patch resting on a curved surface yields a low residual *wherever* it
is placed. The uniformity of the result across patches (standard deviation of only
0.17 mm across 12 patches at supposedly 12 different sites) was the diagnostic
tell. That approach is not used here and no surface-ICP accuracy figure is
reported. The episode is worth reporting in the final manuscript as a cautionary
result, because correspondence-free ICP of small near-planar patches onto large
curved surfaces is under-constrained in a way that does not announce itself in
the residual.

### 2.12 Software provenance

Every algorithmic step is bound to the implementation that produced the numbers,
so that a reviewer can check the flag-to-algorithm claim directly.

**Table 9.** Implementation provenance.

| Pipeline stage | Module / function | Library + flag | Algorithm reference |
|---|---|---|---|
| Undistort, resize | `bone_depth_batch.undistort_bgr` | OpenCV | — |
| Sparkle map | `detect_trackers.local_texture` | OpenCV `Laplacian`, `GaussianBlur` | — |
| Circle candidates | `detect_trackers._hough_candidates` | OpenCV `HoughCircles`, `HOUGH_GRADIENT` | — |
| Normal recovery | `bone_depth_batch.photometric_stereo` | NumPy least squares | [6] |
| Depth integration | `bone_depth_batch.integrate_frankot_chellappa` | NumPy FFT | [7] |
| Pose from 3 points | `marker_distances.ct_distances` | OpenCV `solveP3P`, **`SOLVEPNP_AP3P`** | **[8]** |
| Depth correction | `ct_correct_depth.fit_affine_z` | NumPy `lstsq` | — |
| LED calibration | `led_calibration` | NumPy | [1] |
| Registration | *[TO BE COMPLETED]* | *[TO BE COMPLETED]* | *[e.g. 9]* |

> ⚠️ **VERSION PINNING REQUIRED — AND THE SOLVER CITATION DEPENDS ON IT.**
> The exact OpenCV, NumPy, SciPy and Matplotlib versions used to produce the
> reported numbers are not currently recorded and **must** be, because the
> algorithm behind an OpenCV solver flag has changed across releases. Our code
> calls `SOLVEPNP_AP3P`, which OpenCV documents as implementing Ke &
> Roumeliotis [8]; the neighbouring `SOLVEPNP_P3P` flag maps to Gao et al. [9] in
> OpenCV 4.5.5 but to a different, more recent algorithm on current 4.x. Citing
> the wrong paper for the solver actually executed is a real and easy error here.
> See Appendix C, item M5.

---

## 3. Results

### 3.1 Dataset

Five capture sessions, `shot_011` through `shot_015`, of an ex-vivo vertebral
specimen were processed (25 images; §2.2.3). Each comprises four single-LED
exposures and one dark frame. Only `shot_011` carries CT ground truth, for three
of its four markers.

### 3.2 Marker detection yield

Detection recovered **18 markers** across the five sessions (Table 10). The
cross-frame confirmation statistic `n_frames` is the primary confidence
indicator: **14 of 18 markers (78%) were independently detected in all four
single-LED frames**, i.e. under four different illumination directions. Two
markers were seen in three frames, and two — `shot_014` marker 1 and `shot_015`
marker 4 — in only one, and should be treated as provisional.

Detected marker radii averaged 24.6 px (range 12.0–31.8 px) at the 640-px
working width, corresponding to roughly 0.7–1.8 mm at the measured field scale.
The two single-frame detections are the smallest (12.0 px) and a mid-sized
(18.0 px) candidate respectively, consistent with detection confidence degrading
at the small-radius end of the range.

**Table 10.** Marker detection and relative depth by session, `shot_011`–`shot_015`.
Source: `depth_outputs/marker_depth_slides/marker_depths.csv`.

| Session | Markers | *z* range (mm) | *z* span (mm) | Radius range (px) | `n_frames` per marker |
|---|---|---|---|---|---|
| `shot_011` | 4 | −2.302 … +3.011 | 5.31 | 17.7–26.8 | 4, 4, 4, 3 |
| `shot_012` | 3 | −3.389 … +2.295 | 5.68 | 25.8–26.5 | 4, 4, 4 |
| `shot_013` | 3 | −2.600 … +1.673 | 4.27 | 22.0–26.8 | 4, 4, 4 |
| `shot_014` | 4 | −5.087 … +0.197 | 5.28 | 12.0–29.3 | 1, 4, 4, 3 |
| `shot_015` | 4 | −1.168 … +2.522 | 3.69 | 18.0–31.8 | 4, 4, 4, 1 |
| **Pooled** | **18** | −5.087 … +3.011 | — | 12.0–31.8 | 78% at 4 |

### 3.3 Recovered relative depth

Pooled across the five sessions the relative depth has mean −0.047 mm and
standard deviation 2.25 mm, with per-session spans of 3.69–5.68 mm. These
magnitudes are physically plausible for vertebral surface relief over a 37 mm
field, which is precisely why the failure documented next is not self-announcing.

### 3.4 CT validation at `shot_011`: the depth channel is inverted

This is the paper's principal empirical result.

For the three `shot_011` markers with CT ground truth, the P3P solution of
§2.8.1 — selected under the confirmed constraint that marker 4 is nearest —
places the markers at lens distances of 33.5 mm (marker 4), 50.9 mm (marker 1)
and 53.2 mm (marker 3), spanning 19.7 mm. Converted to camera-frame depth
relative to the marker-set median, the CT values span 20.7 mm.

The photometrically recovered relative depths for the same three markers are
+0.087 mm, −2.302 mm and +2.673 mm.

**Table 11.** Photometric depth versus CT ground truth, `shot_011`. All values in
millimetres. Sources: `shot_011_ct_corrected.csv`, `marker_distances.csv`.

| Marker | *z*<sub>est</sub> | CT camera-frame *z*<sub>rel</sub> | CT lens distance | Affine-corrected *z*<sub>corr</sub> | Residual |
|---|---|---|---|---|---|
| 1 | +0.087 | 0.000 | 50.9 | −6.157 | −6.157 |
| 3 | −2.302 | +0.719 | 53.2 | +3.920 | +3.201 |
| 4 | +2.673 | −20.023 | 33.5 | −17.066 | +2.957 |
| 2 | +3.011 | *(no CT)* | *(56.5, photometric)* | −18.492 | — |

<figure>
<img src="fig2_inversion.png" alt="Photometric depth versus CT ground truth, and the reversed depth ordering">
<figcaption><strong>Figure 2.</strong> The depth inversion at <code>shot_011</code>. <strong>(a)</strong> Far-field photometric relative depth against CT-derived lens-to-marker distance for the three CT-known markers, with the least-squares trend (dashed). The relationship is negative: markers that CT places <em>farther</em> from the lens are assigned <em>shallower</em> depth. <strong>(b)</strong> Near-to-far ordering of the same three markers under each method. Marker 1 holds the middle position while markers 3 and 4 exchange the extremes, reversing the sequence (3, 1, 4) to (4, 1, 3). Marker identity is encoded by both colour and symbol so the figure survives greyscale reproduction. Generated by <code>paper/make_fig_inversion.py</code> directly from the committed CSV outputs.</figcaption>
</figure>

Three findings follow (Fig. 2).

**(i) The correlation is negative.** Pearson correlation between the photometric
relative depth and the CT-derived lens-to-marker distance is ***r*** **= −0.923**;
against camera-frame CT depth it is *r* = −0.891. The photometric depth channel
varies inversely with true distance.

**(ii) The depth ordering is exactly reversed.** Ranked nearest to farthest,
photometry gives (3, 1, 4) while CT gives (4, 1, 3) — a reversal of the sequence.
The failure is not noise degrading a correct ordering; it is a systematic sign
inversion.

**(iii) Affine correction cannot rescue it.** The least-squares affine fit onto
the CT control points returns

$$z_{\text{corr}} = -4.218\,z_{\text{est}} - 5.791\ \ \text{mm},$$

whose **negative slope** is the inversion stated algebraically. Even permitting
this sign flip, the fit leaves a **4.36 mm RMS residual** across a 20.7 mm true
depth span — roughly 21% of the measured range. Because the affine model has two
free parameters and there are three control points, this residual reflects a real
lack of fit rather than an interpolation artifact.

The corrected map is emitted as `shot_011_ct_corrected.png` (Fig. 3) for the
record, but it is a **diagnostic, not a usable correction**: the affine rescale
with slope −4.218 amplifies the underlying far-field spatial bowl roughly
threefold while merely re-seating the marker values.

<figure>
<img src="../depth_outputs/marker_depth_slides/shot_011_ct_corrected.png" alt="CT-anchored depth correction diagnostic for shot_011">
<figcaption><strong>Figure 3.</strong> CT-anchored depth correction for <code>shot_011</code>, emitted by <code>apply_ct_shot011.py</code>. Left: the affine-rescaled depth field with the four markers. Right: per-marker comparison of the photometric estimate, the corrected value, the CT camera-frame depth, and the residual. This figure documents the failure rather than correcting it — the fitted slope is negative and the control-point RMS is 4.36 mm across a 20.7 mm true span. <em>File:</em> <code>depth_outputs/marker_depth_slides/shot_011_ct_corrected.png</code></figcaption>
</figure>

**Interpretation.** The rig places its LEDs 12.08 mm off-axis at a 30 mm working
distance — a light-offset-to-working-distance ratio of 0.40, unambiguously
near-field. Under near-field illumination the direction to each LED varies
substantially across a 37 mm field and intensity falls off as 1/*r*², neither of
which the far-field model represents (Table 5). The resulting systematic
brightness bias across the field is integrated by the Frankot–Chellappa step into
a large-scale spurious surface curvature that dominates the true relief, and
whose sign relationship to true depth is inverted in this geometry.

### 3.5 Lens-to-marker distances

Eighteen distances were computed for sessions `shot_011`–`shot_015`, of which
**3 are CT-validated and 15 are photometric and flagged unreliable**.

- **CT-validated** (`shot_011`, markers 1, 3, 4): 50.9, 53.2 and 33.5 mm; median
  50.9 mm. These are trustworthy up to the ~1.5× intrinsic-scale ambiguity of
  §2.3, which affects all three multiplicatively and therefore does not affect
  their ordering or ratios.
- **Photometric** (15 markers): mean 48.6 mm, standard deviation 4.4 mm, range
  41.9–56.5 mm.

The apparent tightness of the photometric distribution is **not** evidence of
accuracy and must not be read as such. Those values are constructed by adding a
small relative depth (standard deviation 2.25 mm) to a *constant* 46.0 mm anchor
and back-projecting; their spread is therefore largely inherited from the anchor
by construction, and the relative-depth component that does vary is the very
quantity shown in §3.4 to be sign-inverted.

### 3.6 Numerical validation of the correction and calibration modules

Both auxiliary modules ship self-tests on synthetic ground truth, executable
without captured data. Both were run for this manuscript and **both pass**.

**Table 12.** Self-test results (synthetic ground truth).

| Module | Stage | Result |
|---|---|---|
| `ct_correct_depth` | affine-*z*, scale+offset distortion, 2 control points | held-out error 0.015, 0.006, 0.008 mm |
| `ct_correct_depth` | tilt-plane, offset+tilt distortion, 3 control points | held-out error 0.030, 0.046 mm |
| `led_calibration` | anisotropy, *θ*<sub>1/2</sub> = 60° → *μ* | *μ* = 1.0000 (expected 1.000) — PASS |
| `led_calibration` | LED position, mirror-ball triangulation | error 0.00 mm — PASS |
| `led_calibration` | direction and intensity, plane fit | **n**<sub>*s*</sub> error 0.090°; intensity error 0.01% — PASS |

These establish that the correction algebra and the calibration solvers are
correctly implemented. They say nothing about the input depth, and in particular
the passing affine-*z* self-test does **not** license the affine correction of
§3.4: the self-test confirms the estimator recovers a scale-and-offset
distortion, whereas the real distortion is a spatially-varying near-field bowl,
which that model is not designed to remove.

### 3.7 Registration accuracy

> ### ⚠️ PLACEHOLDER — NO REGISTRATION RESULT EXISTS
>
> **Every number in this subsection is invented scaffolding.** It exists only so
> the results section has the correct shape. Nothing here may be cited,
> circulated, or carried into a submission. See Appendix C, items R1–R8.

*[Registration of endoscopic marker coordinates to CT was evaluated over `[N]`
correspondences drawn from `[M]` sessions. The closed-form similarity solution
returned a scale of `[S.SSS]` mm per cloud unit. Rotation, translation and scale
residuals are reported separately in Table `[N]`.]*

*[**TRE (the accuracy endpoint)** at independent target markers: mean `[T.TT]`
mm, median `[T.TT]` mm, SD `[T.TT]` mm, maximum `[T.TT]` mm, 95th percentile
`[T.TT]` mm — reported per target rather than pooled, since TRE varies spatially
with target position relative to the fiducial configuration. **FRE (diagnostic
only)**: `[F.FF]` mm RMS. FRE is reported for completeness and is not an accuracy
claim; it is uncorrelated with TRE [2,3].]*

*[Sensitivity of the registration result to the §2.3 focal-length ambiguity must
be reported as a paired analysis at f_x = 519 px and f_x = 794 px, since a 1.5×
intrinsic error propagates directly into recovered scale and translation.]*

---

## 4. Discussion

### 4.1 What this pipeline currently supports, and what it does not

The clearest outcome of this work is a sharp separation between the pipeline's
reliable and unreliable outputs, established by measurement rather than by
assumption.

**Reliable.** Marker pixel localization is robust: 78% of markers were confirmed
in all four independent illumination conditions, and the detector's
three-signature acceptance logic with its dark-surround requirement demonstrably
separates retroreflective markers from specular glints on wet bone. The CT/P3P
lens-to-marker distances for `shot_011` rest on independent ground truth and a
physically disambiguated pose. Both are suitable registration inputs.

**Unreliable.** The photometric depth channel is sign-inverted against the only
ground truth available, and every quantity derived from it — the *z* column of
`marker_depths.csv` for all sessions, and the 15 photometric lens distances —
inherits that inversion. These outputs are retained in the repository, and
labelled in the data files themselves, so that the failure is documented rather
than hidden; they are not results.

We emphasize that this separation was only possible because one session carried
CT ground truth. With no ground truth at all, the depth maps of Figure 1 are
entirely plausible: they are smooth, their relief magnitudes of a few millimetres
are anatomically sensible (§3.3), and the markers sit at sensible-looking depths.
Nothing in the output signals that the ordering is reversed. This is a general
hazard of shading-based reconstruction, and it argues for including ground-truth
control points in at least one session of any such study as a matter of routine
design.

### 4.2 Why the far-field model fails here specifically

The failure is a predictable consequence of geometry rather than of
implementation. Far-field photometric stereo requires that light direction be
constant over the field and that intensity falloff be negligible within the
scene. With an LED offset of 12.08 mm, a working distance of 30 mm, and a 37 mm
field width, the angle subtended by the field from an LED is large: the direction
to a given LED differs by tens of degrees between opposite edges of the image, and
the LED-to-surface distance varies by well over 10% across the same span.

The consequence is a smooth, low-spatial-frequency brightness bias that the
normal-recovery step interprets as surface tilt. Frankot–Chellappa integration
then accumulates that spurious tilt across the field into a large-scale
curvature. Because this artifact is smooth and large in amplitude, it dominates
the genuine relief — and, in this configuration, does so with a sign that
reverses the true depth ordering.

This also explains why post-hoc correction fails. The affine-*z* model removes a
global scale and offset, but the near-field artifact is *spatially varying*; the
tilt-plane model would absorb its first-order component but requires three
non-collinear control markers, which only `shot_011` provides — and even there,
with exactly three control points the tilt-plane fit interpolates and yields no
residual information. Correction after the fact is the wrong layer at which to
address the problem. The forward model must be fixed instead.

### 4.3 The corrective path and its cost

The near-field solver and LED calibration of §2.10 constitute a complete,
implemented, and unit-validated path to physically-correct depth. The remaining
cost is not development but *data acquisition*: a mirror ball at ~10 poses per
LED and a matte white card at 3–5 tilts per LED, one session's work.

Separately, the 1.53× focal-length discrepancy of §2.3 must be resolved before
any absolute metric claim. It is currently arbitrated by trusting the bench
measurements over the checkerboard calibration, which is defensible — the two
bench measurements are mutually consistent while the checkerboard result
contradicts both — but it is an arbitration, not a resolution, and it can be
settled definitively by imaging an object of known size at a known distance. The
absence of a recorded reprojection residual for that calibration (§2.3) is itself
part of the problem.

### 4.4 Implications for registration

*[PLACEHOLDER — expand once §2.11 and §3.7 are written.]*

The finding of §3.4 constrains the registration design directly and, we would
argue, favourably. Because the depth channel is unusable, registration cannot
depend on it; but the marker pixel coordinates and the CT/P3P geometry are sound,
and correspondence-based registration on markers uses exactly those. A
marker-correspondence formulation is therefore not merely a convenient choice but
the only one the current data supports — and it is the more rigorous choice
regardless, since it yields a closed-form solution and a directly measurable
target error rather than an iterative fit whose residual, as the surface-ICP
episode in §2.11.4 illustrates, can be low for reasons unrelated to correctness.

*[The relationship between the P3P pose of §2.8.1 and the registration of §2.11
must be resolved: they may be the same computation described twice. If so, the
paper's registration result may already exist in the form of the `shot_011` pose,
and the contribution should be reframed accordingly.]*

---

## 5. Limitations

1. **Photometric depth is sign-inverted** (*r* = −0.923 against CT distance).
   This invalidates the *z* column of `marker_depths.csv` and the 15 photometric
   distances in `marker_distances.csv` (§3.4).
2. **Ground truth exists for one session and three markers.** All validation of
   the depth channel rests on `shot_011` markers 1, 3 and 4 — *n* = 3. The
   correlation and ordering findings are unambiguous at this *n*, but the affine
   residual (4.36 mm RMS from three points onto a two-parameter model) is a weak
   estimate. The binding constraint is CT availability, not endoscopy (§2.2.3).
3. **The ground truth's own uncertainty is unquantified.** CT voxel size,
   reconstruction kernel, and in-volume marker localization uncertainty are not
   recorded, so no error claim finer than that unstated floor is defensible.
4. **Absolute scale is uncertain by ~1.53×** owing to the unresolved
   *f<sub>x</sub>* = 519 px versus 794 px discrepancy (§2.3), and the calibration
   reprojection residual is not recorded.
5. **The near-field path is unexecuted.** It is implemented and passes synthetic
   self-tests but has never been run on real data pending calibration captures
   (§2.10.2).
6. **Capture conditions are undocumented** (ambient light control, auto-exposure
   and white-balance settings, saturation) — §2.2.4.
7. **Two markers rest on single-frame detections** (`shot_014` marker 1,
   `shot_015` marker 4) and lack cross-frame confirmation.
8. **Raw capture data is not committed.** `Data_collection/shots/` and
   `tracker_outputs/` are absent from the repository, so the pipeline cannot be
   re-executed end-to-end from the repository alone (Appendix A).
9. **Software versions are not pinned**, which additionally makes one solver
   citation version-dependent (§2.12).
10. ***[Registration limitations — TO BE WRITTEN once §3.7 exists.]***

---

## 6. Conclusions

We have documented a four-LED endoscopic measurement pipeline spanning marker
detection, photometric depth recovery, composite figure generation, and
CT-anchored distance estimation, and have evaluated it against CT ground truth.

The marker localization stage is reliable, with 78% of 18 markers confirmed under
all four illumination directions, and the CT/P3P lens-to-marker distances are
sound. The photometric depth stage is not: against the single available ground
truth it is anti-correlated (*r* = −0.923) and reverses the depth ordering of the
three CT-known markers, and no affine correction recovers it. We attribute this
to the application of a far-field illumination model in a geometry (12.08 mm LED
offset at 30 mm working distance) that is decisively near-field, and we identify
the implemented, self-test-validated near-field solver and LED calibration as the
corrective path, blocked only on one session of calibration captures.

For the registration problem this pipeline exists to serve, the practical
conclusion is that registration must be built on marker correspondences rather
than on recovered surface geometry — which the reliable half of the pipeline
already supports. *[FINAL REGISTRATION CONCLUSION — PLACEHOLDER.]*

---

## Statements and Declarations

> ⚠️ **PLACEHOLDER — ALL DECLARATIONS REQUIRED BEFORE SUBMISSION.** Several
> target journals treat a missing declaration as an incompleteness ground for
> return without review. Emit every heading and write "Not applicable" rather
> than deleting a heading. Heading names and placement differ by journal —
> confirm against the chosen journal's current author guidelines (Appendix C,
> item M6).

**Funding.** *[TO BE COMPLETED]*

**Competing interests.** *[TO BE COMPLETED]*

**Ethics approval and specimen provenance.** *[TO BE COMPLETED — must NAME the
institutional oversight or licensing body in full, not assert approval
generically, and state how the ex-vivo specimen was obtained.]*

**Consent for publication.** *[TO BE COMPLETED — state affirmatively or state
that the requirement was waived by the named body; silence is not acceptable.]*

**Data availability.** *[TO BE COMPLETED — must resolve Limitation 8. Cite a
persistent identifier (repository DOI or accession), not a bare URL, and archive
before submission rather than promising a link at proof stage.]*

**Code availability.** *[TO BE COMPLETED — the pipeline is version-controlled;
a tagged, licensed release with a reproduction instruction document should be
archived under a DOI and cited here.]*

**Author contributions.** *[TO BE COMPLETED]*

**Use of AI tools.** *[TO BE COMPLETED — disclosure requirements differ sharply
by journal, and at least one requires the prompts themselves to be reported.
Keep a running log during manuscript preparation; prompts cannot be reconstructed
after the fact.]*

---

## Appendix A. Reproduction

From `marker_pipeline/`:

```bash
# 1. marker detection (per-image overlays + consolidated CSV)
python detect_trackers.py ../Data_collection/shots/shot_*/led*.png \
       --outdir ../tracker_outputs --csv ../tracker_outputs/trackers.csv

# 2. per-session composite figures + marker_depths.csv
python marker_depth_slides.py                 # or --shots shot_011 shot_012 ...

# 3. lens-to-marker distances (CT for shot_011, photometric elsewhere)
python marker_distances.py

# 4. CT-anchored diagnostic for shot_011
python apply_ct_shot011.py

# 5. validation (no captured data required)
python ct_correct_depth.py --selftest
python led_calibration.py  --selftest
```

To rebuild this manuscript and its generated figure, from `paper/`:

```bash
python make_fig_inversion.py     # Figure 2, from the committed CSVs
./build_pdf.sh                   # manuscript.md -> manuscript.pdf
```

Outputs are written to `depth_outputs/marker_depth_slides/` and
`tracker_outputs/`.

**Note.** Steps 1–4 require `Data_collection/shots/`, which is **not committed to
the repository**. Only step 5 and the manuscript build are runnable from a fresh
clone. The committed CSV and PNG outputs in `depth_outputs/marker_depth_slides/`
are the record of a previous execution, and all measured values in §3 are read
from them.

**Environment.** Python 3 with NumPy, OpenCV, SciPy and Matplotlib. Self-tests in
step 5 and the figure build require only NumPy and Matplotlib. *[Exact versions
must be pinned — see §2.12 and Appendix C item M5.]*

## Appendix B. Data dictionary

**`marker_depths.csv`** — one row per detected marker per session.

| Column | Meaning | Reliability |
|---|---|---|
| `shot` | session identifier | — |
| `id` | marker index within the session (top-to-bottom, then left-to-right) | — |
| `x_px`, `y_px` | marker centre, working-resolution pixels | reliable |
| `x_mm`, `y_mm` | in-plane position, mm across the measured 37 mm field | reliable |
| `z_mm` | **relative** depth about the session median; + = farther | **UNRELIABLE — inverted (§3.4)** |
| `r_px` | detected marker radius, px | reliable |
| `n_frames` | number of the four LED frames in which the marker was detected | reliable |

**`marker_distances.csv`** — one row per marker, sessions 010–015.

| Column | Meaning |
|---|---|
| `u_px`, `v_px` | marker pixel coordinates |
| `z_rel_mm` | relative photometric depth (as above) |
| `method` | `CT (validated)` or `photometric (UNRELIABLE: inverted vs CT@011)` |
| `distance_from_lens_mm` | Euclidean lens-to-marker distance |
| `note` | free-text caveat |

**`shot_011_ct_corrected.csv`** — CT-anchoring diagnostic for `shot_011`.

| Column | Meaning |
|---|---|
| `z_est_mm` | far-field photometric relative depth |
| `z_corr_mm` | affine-corrected depth, *z*<sub>corr</sub> = −4.218·*z*<sub>est</sub> − 5.791 |
| `ct_cam_rel_mm` | CT depth in the camera frame, relative to the marker-set median (blank for marker 2) |
| `residual_mm` | *z*<sub>corr</sub> − *z*<sub>CT</sub> |

## Appendix C. Open items — what must be completed before submission

### Registration (the paper's principal claim)

| ID | Item | Blocking on |
|---|---|---|
| **R1** | Decide and document the registration formulation (§2.11). Resolve whether it generalizes or replaces the existing `shot_011` P3P pose. | authors' decision |
| **R2** | Acquire per-session CT marker coordinates for `shot_012`–`shot_015`. Currently only `shot_011` has any, and only for 3 of 4 markers. | CT acquisition |
| **R3** | Produce the registration result: recovered scale, rotation/translation/scale residuals reported separately, TRE per target, FRE as diagnostic (§3.7). | R1, R2 |
| **R4** | Define and execute a validation protocol with independent evaluation targets. Not currently executable at *n* = 3 in one session. State repetition count and how repeats were generated. | R2 |
| **R5** | Write §2.11.3: fiducial configuration table (N, coordinates, spread, centroid-to-target position, near-collinearity/coplanarity) and justify the configuration. Report the existing `near_planar_warning` flag. | R3 |
| **R6** | Report registration sensitivity to the focal-length ambiguity as a paired analysis at *f<sub>x</sub>* = 519 and 794 px. | R3, C2 |
| **R7** | Log and report the P3P candidate-pose count and how many survived each disambiguation filter (§2.8.1). | code change only |
| **R8** | Quantify the CT ground truth's own uncertainty (voxel size, kernel, in-volume marker localization error) and state it as the validation noise floor. | CT metadata |

### Measurement and calibration

| ID | Item | Blocking on |
|---|---|---|
| **C1** | Capture LED calibration targets: mirror ball (~10 poses/LED) and matte white card (3–5 tilts/LED). Unblocks the entire near-field path. Record *θ*<sub>1/2</sub> from the LED datasheet at the same time. | one capture session |
| **C2** | Resolve the *f<sub>x</sub>* = 519 vs 794 px ambiguity by imaging a known-size object at a known distance. Record the checkerboard reprojection residual, pose count and distortion coefficients. | one bench measurement |
| **C3** | Run `nearfield_lambertian.py` on real data and re-validate depth against CT at `shot_011`. Confirms or refutes the near-field diagnosis of §4.2. | C1 |
| **C4** | Re-derive lens-to-marker distances from the near-field depth and replace the 15 flagged photometric values. | C3 |
| **C5** | Obtain CT ground truth for `shot_011` marker 2. | CT acquisition |

### Manuscript

| ID | Item |
|---|---|
| **M1** | Author list, affiliations, corresponding author; complete every heading in *Statements and Declarations*. |
| **M2** | **Verify every reference against the publisher record** — see Appendix D. Several entries are currently unverified. |
| **M3** | Complete §2.2.4: specimen species/anatomy/preparation, marker attachment method and physical marker diameter (which would independently validate the field-width scale against the 12.0–31.8 px radii), and the capture-condition checklist. |
| **M4** | Figure preparation at journal resolution; decide which session figures appear in the main text versus supplementary. |
| **M5** | Pin exact software versions (§2.12) — required for correctness of the solver citation, not only for reproducibility. Archive raw captures under a DOI (Limitation 8). |
| **M6** | Select target journal and reformat: abstract structure, declaration heading names and placement, and word/page budget all differ by journal and several budgets are inclusive of references, tables and captions. |

## Appendix D. Reference verification status

Reference metadata is a common source of published error, and entries drawn from
third-party bibliography files are frequently corrupt. Each reference below
therefore carries an explicit verification status. **No entry marked *unverified*
may be carried into a submission without being checked against the publisher
record** (Appendix C, item M2).

| # | Reference | Status |
|---|---|---|
| 1 | Quéau et al., LED-Based Photometric Stereo, *J. Math. Imaging Vis.* | **verified** against the corresponding author's released code repository; publisher page not reachable at time of writing |
| 2–5 | Registration-error literature (FRE/TRE) | **unverified** — bibliographic details from search results only |
| 6–7 | Photometric stereo; integrability | **unverified** — publisher pages not reachable |
| 8 | Ke & Roumeliotis, AP3P | **verified** as the algorithm behind OpenCV `SOLVEPNP_AP3P`, by direct inspection of the OpenCV source and its bibliography; proceedings pagination unverified |
| 9 | Gao et al., P3P | **unverified**; note this is *not* the flag our code calls (§2.12) |

---

## References

> ⚠️ **REFERENCE LIST INCOMPLETE AND PARTLY UNVERIFIED.** See Appendix D.
> Entries below record what could be established at time of writing; fields that
> could not be confirmed are marked. Do not submit without checking each against
> the publisher record.

1. Quéau, Y., Durix, B., Wu, T., Cremers, D., Lauze, F., & Durou, J.-D. (2018).
   LED-Based Photometric Stereo: Modeling, Calibration and Numerical Solution.
   *Journal of Mathematical Imaging and Vision*, 60(3), 313–340.
   DOI: 10.1007/s10851-017-0761-1.
   *(Note: `led_calibration.py` cites this as "Quéau et al. 2017", the preprint
   year; the journal version is 2018. Align the citation year in code comments
   and manuscript.)*

2. *[Fitzpatrick, J. M., West, J. B., & Maurer, C. R. Jr. (1998). Predicting
   error in rigid-body point-based registration. IEEE Transactions on Medical
   Imaging, 17, 694–702. — **UNVERIFIED**; issue number and DOI unconfirmed.
   Cited for the FRE/TRE distinction.]*

3. *[Fitzpatrick, J. M. (2009). Fiducial registration error and target
   registration error are uncorrelated. Proc. SPIE 7261, 726102. — **UNVERIFIED**.
   Cited for the explicit FRE-is-not-accuracy disclaimer in §2.11.1.]*

4. *[Shamir, R. R., Joskowicz, L., Spektor, S., & Shoshan, Y. (2009).
   Localization and registration accuracy in image guided neurosurgery: a
   clinical study. Int. J. Computer Assisted Radiology and Surgery, 4, 45–52. —
   **UNVERIFIED**. Cited for leave-one-out F-TRE also being a poor TRE
   indicator.]*

5. *[West, J. B., Fitzpatrick, J. M., Toms, S. A., Maurer, C. R. Jr., & Maciunas,
   R. J. (2001). Fiducial point placement and the accuracy of point-based, rigid
   body registration. Neurosurgery, 48(4), 810–816. — **UNVERIFIED**. Cited for
   fiducial configuration governing TRE.]*

6. *[Woodham, R. J. (1980). Photometric method for determining surface
   orientation from multiple images. Optical Engineering, 19(1), 139–144. —
   **UNVERIFIED**; DOI 10.1117/12.7972479 from search results only.]*

7. *[Frankot, R. T., & Chellappa, R. (1988). A method for enforcing integrability
   in shape from shading algorithms. IEEE Trans. Pattern Analysis and Machine
   Intelligence, 10(4), 439–451. — **UNVERIFIED**; DOI 10.1109/34.3909 from
   search results only.]*

8. Ke, T., & Roumeliotis, S. (2017). An Efficient Algebraic Solution to the
   Perspective-Three-Point Problem. *IEEE Conference on Computer Vision and
   Pattern Recognition (CVPR)*. arXiv:1701.08237.
   *(Confirmed as the algorithm implemented by OpenCV's `SOLVEPNP_AP3P`, which
   is the flag this pipeline calls. Proceedings page range **unverified**.)*

9. *[Gao, X.-S., Hou, X.-R., Tang, J., & Cheng, H.-F. (2003). Complete solution
   classification for the perspective-three-point problem. IEEE Trans. Pattern
   Analysis and Machine Intelligence, 25(8), 930–943. — **UNVERIFIED**. Cite only
   if the `SOLVEPNP_P3P` flag is used on an OpenCV version that maps to it; our
   code calls `SOLVEPNP_AP3P` instead (§2.12).]*

10. *[Closed-form similarity registration (Horn 1987 / Umeyama 1991) — to be
    cited once §2.11.2 selects an estimator. Both **UNVERIFIED**.]*
