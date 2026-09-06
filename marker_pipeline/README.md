# marker_pipeline — complete walkthrough, capture to validated mesh

How five photographs of a vertebra become a metric 3-D mesh in millimetres,
with **no CT information anywhere in the reconstruction**. CT is used only at
the end, to score the result.

Read top to bottom: each section is one stage of the pipeline, in the order it
runs, with the command, what it computes, and what was actually measured.

**Current headline result** (levels 4/5/6, rigid 6-DOF alignment, nothing fitted
to CT):

| metric | value |
|---|---|
| FRE at the fiducials | 2.68 / 3.07 / 2.78 mm |
| Leave-one-out (predictive) | 5.11 mm rms |
| Dense error vs the CT bone surface | 2.79 mm median, 3.96 mm rms |
| — within 10 mm of a fiducial | 2.09 mm |
| — beyond 25 mm of a fiducial | 6.99 mm ← open problem, §12 |

---

# Stage 0 — the rig

| property | value | source |
|---|---|---|
| Probe outer diameter | 18.0 mm | CAD |
| LED ring radius `a_r` | **6.05 mm** off the optical axis | measured — *this is the metric anchor* |
| LEDs | 4 white, Arduino Mega pins 12/11/10/9 | `arduino_led_stage.ino` |
| Camera | 640×480, fixed focus | `calibration/calibration.txt` |
| Focus band | 45–70 mm | bench property of the fixed-focus optics |
| Working distance at capture | ~55–60 mm | measured from CT afterwards |

The LED ring radius is the single most important number in the project. It is a
physical length in millimetres, and it is what lets the reconstruction recover
absolute scale without ground truth (§6).

**Near-coaxial geometry.** Each LED subtends only `α ≈ atan(6.05/55) ≈ 6°` from
the optical axis. This is the defining constraint of the whole system: it makes
the rig fit an 18 mm MIS tube, but it also means the four light directions are
nearly parallel, the photometric solve is ill-conditioned (`cond(L) ≈ cot α ≈ 9`),
and **cast shadows are almost non-existent** — a 2 mm bead displaces its shadow
by only `h·a_r/D ≈ 0.2 mm ≈ 3 px`. That last fact kills several standard
calibration techniques (§10).

---

# Stage 1 — capture

```powershell
python Data_collection/grab4.py --port COM3
```

Drives the Arduino over serial at 115200 baud (`L1`…`L4` = one LED on, `L0` =
all off), grabbing one frame per state. Per shot it writes:

```
Data_collection/calib_charuco/shot_004/
├── led1.png    one LED on
├── led2.png
├── led3.png
├── led4.png
├── dark.png    all LEDs off — ambient + sensor baseline
└── notes.txt   timestamp, per-frame means, pair-imbalance check
```

**The camera and specimen must not move between the five frames.** Only the
light changes. That is the entire measurement; everything downstream is
computed from these five images.

Retroreflective bead markers are glued to the bone before capture. They serve
two purposes later: PnP pose and accuracy scoring. They are **not** used to
build the depth.

> **Surviving data: shots 004, 005, 006 only.** Shots 007 and 008 were captured
> and appear in the paper, but their raw frames were lost. Their hand-picked
> marker correspondences are preserved as a comment block in
> `calibrate_photometric_ct.py`. Those two levels are no longer reproducible.

---

# Stage 2 — camera calibration (one-time)

`calibration/calibration.txt` holds the intrinsics from a checkerboard
calibration (51 images, reprojection error 0.107 px):

```
fx = 860.18, fy = 857.91, cx = 286.51, cy = 224.34      (at 640×480)
dist = [0.128, 1.355, -0.0203, -0.0240, -11.97]
```

Note `k3 = −11.97` — distortion is severe at the frame edges. This is why the
pipeline works on **raw, un-undistorted frames** (`bd.UNDISTORT_INPUTS = False`)
and undistorts only the individual ray directions at back-projection time
(§8). Undistorting whole images with `alpha=0` silently crops edge markers.

---

# Stage 3 — preprocessing

Handled inside `bone_depth_batch.py`; you never call it directly. Three steps,
each of which matters:

**3a. Dark subtraction** — `led_i − dark`, clipped at 0. Removes ambient light
and sensor bias so each frame contains only the light that LED contributed.

**3b. Linear luminance** (`solve_luminance`) — undoes the sRGB gamma, converts
RGB to a single Rec.601 brightness channel, and inpaints specular glints on the
marker beads (which violate Lambert badly).

**3c. Exposure balance** (`balance_exposure`) — divides each frame by its own
median.

> **3c is load-bearing, not cosmetic.** The four frames are separate JPEGs with
> independent auto-exposure, and the LEDs differ in output by up to 2×
> (shot_004 frame means: 40.7 / 37.3 / 61.2 / 30.4). Tested by removing it: the
> reconstruction comes out **inverted** — correlation with CT goes from +0.97
> to *negative*. An uncorrected gain difference injects a fake surface gradient
> that the solve reads as shape.

An alternative, normalizing over the bone mask only, was also tested: it helps
shot_004 slightly (1.54 → 1.38 mm) but destroys shots 005 and 006 (9.06,
10.07 mm). Whole-frame median stays.

---

# Stage 4 — surface normals from shading

The near-field image formation model, per pixel and per LED `s`:

```
I_s = ρ · (n · l_s) · cos^μ(θ_s) / r_s²

  ρ      albedo (unknown, per pixel)
  n      surface normal (unknown, per pixel)
  l_s    unit vector from the surface point to LED s
  r_s    distance from the surface point to LED s
  cos^μ  LED emission profile, μ = beam-width exponent
```

Because each LED is a **point source at a known position**, `l_s` and `r_s`
differ at every pixel — unlike far-field photometric stereo, where one constant
direction is assumed for the whole image.

With 4 lights and 3 unknowns (`ρ` × the 2 free components of `n`) the system is
overdetermined, and is solved per pixel by least squares for `g = ρn`, then
normalized. This is `solve_g()` in `nearfield_ctfree.py`.

**Why the model needs depth to run:** `l_s` and `r_s` depend on where the
surface *is*. So the solve is iterated — bootstrap a depth, solve normals,
integrate, re-derive the geometry, repeat (§7).

---

# Stage 5 — integrate normals into a surface

`bd.normals_to_depth()` converts normals to gradients and integrates them with
**Frankot–Chellappa** (frequency-domain least squares):

```
p = nx/nz          (∂z/∂column)
q = −ny/nz         (∂z/∂row; image rows run down, +y is up)
z̃ = FC_integrate(p, q)
```

The output `z̃` is correct in **shape** but arbitrary in **scale and offset** —
integration of gradients discards both. Recovering them without CT is the next
stage, and it is the core contribution.

Two consequences worth knowing: FC assumes periodic boundaries, which produces
wrap-around ramps at the frame edges (cropped later), and it is a global
low-pass operator, so fine detail is attenuated. That smoothing is why the
near-field depth map looks *less* structured than a raw brightness map — it is
correctly removing albedo texture, not losing information.

---

# Stage 6 — recover the metric scale (the CT-free core)

Depth is modelled as a **two-parameter family**:

```
D(u,v) = b + s · (b/f) · z̃(u,v)

  b   working distance (mm)
  s   relief gain (dimensionless)
```

`b` and `s` are found by minimising the **four-image photometric residual**:

1. Build the candidate surface from `(b, s)`.
2. Compute normals **from that surface itself** (`normals_from_depth`) — not a
   free per-pixel solve.
3. Predict all four images through the forward model.
4. Allow one free albedo per pixel, one gain per LED, one global `μ`.
5. Compare to the real photos → scalar residual `E(b, s)`.

**Why this recovers real millimetres.** The LED positions are a measured
hardware length (6.05 mm). Scaling the scene does *not* leave the imaging
geometry unchanged, because `1/r²` and the per-pixel light directions both
change. So `E` has a minimum at the true geometry.

> **Tying the normals to the surface is essential.** With freely-solved
> per-pixel normals, the residual is *flat* in `b` — the normals simply tilt to
> absorb any geometric error. Measured: the landscape varied by <1% across the
> entire 20–110 mm range. Tying them is what makes scale observable.

**Two constants pin the search:**

- `b ∈ [45, 70] mm` — the lens focus band. A bench property (the fixed-focus
  optics are only sharp there), **not** taken from CT.
- `μ = 8.0` — read from `ctfree_mu.json`. It is a **rig constant** and must not
  float per shot: `b` and `μ` form a near-perfect degenerate ridge (measured
  cos between their gradients ≈ −1.0), so letting both float lets the optimizer
  slide along the valley.

Search: multi-start from `b₀ ∈ {48, 57, 66}`, coarse grid then Nelder–Mead,
6 fixed-point iterations, keeping the photometrically best iterate.

```powershell
python nearfield_ctfree.py shot_004        # one shot
python nearfield_ctfree.py all             # all shots in CT_CORR
```

Optional flags: `--mu 8`, `--wiring 270,90,180,0`, `--aim-dx 30`, `--trim 0.10`,
`--undistort-grid`. All default off; see §13 for why.

---

# Stage 7 — the iteration loop

```
bootstrap: flat plane at b₀
  ↓
compute per-pixel l_s, 1/r², cos^μ from the current depth
  ↓
solve g = ρn per pixel  (Stage 4)
  ↓
integrate to z̃          (Stage 5)
  ↓
search (b, s) by photometric residual   (Stage 6)
  ↓
D = b + s·(b/f)·z̃  →  feed back        ×6
```

The map is **not monotone** — `z̃` moves as the search moves — so the code keeps
the lowest-residual iterate rather than the last one. That is still a CT-free
choice: the selection criterion is the photometric residual, not accuracy.

---

# Stage 8 — back-project pixels to 3-D

Each pixel becomes a 3-D point in **camera coordinates**:

```python
norm  = cv2.undistortPoints((u,v), K, DIST)     # pixel -> undistorted ray
ray   = [xn, yn, 1]
X_cam = D(u,v) · ray                            # scale the ray by depth
```

A pixel alone is only a *direction*; the depth from Stage 6 is what supplies the
*distance*. Distortion is corrected here, per ray, rather than by warping the
whole image (§2).

Frame convention (`+x` right, `+y` **up**, `+z` **toward** the camera) — OpenCV's
frame with y and z negated. The bone therefore sits at *negative* z, and the
LEDs are at `z = 0`, the lens plane.

---

# Stage 9 — mesh and clean

**Trim** (inside `reconstruct`): drop pixels with photometric residual > 0.22,
depth spikes >8 mm from a 5×5 median, a 9% frame border (FC wrap-around), and
small connected components.

**Mesh**: adjacent pixels form quads → two triangles each, skipping any quad
spanning more than 3 mm of depth (that would bridge an occlusion).

**Clean**:
```powershell
python clean_mesh_edges.py --all                      # every *_ct.ply -> *_clean.ply
python clean_mesh_edges.py in.ply out.ply --peel 2 --max-tilt 60
```
Removes in order: stretched triangles (edge > k × median), **flip-up walls**
(triangles standing more than `--max-tilt` from their patch's dominant
orientation — these have short edges so the stretch test misses them), boundary
rings (`--peel`), and small components.

Outputs, in `../depth_outputs/ct_registered/`:

| file | frame |
|---|---|
| `shot_004_mesh_ctfree_cam.ply` | camera — **the actual product** |
| `shot_004_mesh_ctfree_ct.ply` | CT — rigidly placed, for Slicer |

---

# Stage 10 — the LED wiring question (unresolved)

The forward model needs to know which software LED index sits where on the ring.
Get it wrong and the surface inverts.

**How it was originally decided (and why that's a problem):** by scoring all 24
index→azimuth permutations against CT marker depth
(`led_permutation_study.py`). That means ground truth influenced the
reconstruction — exactly what the CT-free claim forbids.

**Three attempts to settle it CT-free:**

| method | script | result |
|---|---|---|
| Photometric residual vote over all 24 permutations | `led_wiring_vote.py` | wiring in use ranks **#1 of 24** (3.082 vs 3.182) — but #1 and #2 differ by exactly the vertical swap, only 3% apart |
| Direct illumination measurement (`led_i` ÷ 4-LED mean, brightness centroid) | `led_direction_figure.py` | led1 lights **BOTTOM**, led2 lights **TOP** — the *opposite* of the code, consistently across all three shots (led1 at 226–237°, led2 at 116–129°). led3/led4 agree (left/right) |
| Cast-shadow incidence | `led_incidence_check.py` | **inconclusive by construction** — near-coaxial geometry displaces a bead shadow by only ~3 px; the lit-side and anti-shadow estimates disagree by up to 165° |

**Status: unresolved.** The residual vote's 3% margin cannot separate the two
candidates, and the direct measurement confounds LED *position* with beam
*aim* (a convergently-aimed LED lands its beam on the far side). The bench
capture in §14 settles it.

---

# Stage 11 — evaluation (CT finally enters)

Reconstruction is finished before this point. CT is used only to score it.

**11a. Fiducial registration error.** `evaluate()` in `nearfield_ctfree.py`
samples the depth at each marker pixel, back-projects, fits a **rigid 6-DOF**
transform (Kabsch, no scale) onto the CT markers, and reports the residual.

**11b. Scale checks.** Two, deliberately independent:
- *Pairwise distance ratio* — registration-free; compares marker-to-marker
  distances directly, so no alignment can hide a scale error.
- *Umeyama similarity scale* `c`, solving `CT ≈ c·R·rec + t`. Verified on
  synthetic data: `umeyama_scale(A,B)` returns the factor scaling A up to B.

**11c. Leave-one-out** — the honest accuracy number:
```powershell
python loo_accuracy.py
```
Fits the transform on `n−1` markers and measures at the **held-out** one. FRE is
biased low by `√(1−2/n)` ≈ 29% at n=4 because it is measured at the same points
the transform was fitted to; LOO is not.

**11d. Dense surface error** — every vertex against the real CT bone:
```powershell
python ct_frame_recover.py      # once — recovers the missing CT frame transform
python dense_ct_accuracy.py
```

`ct_frame_recover.py` exists because the marker coordinates in `CT_CORR` were
hand-read in Slicer into a recentred frame, while `ct_spine.stl` is in raw RAS,
and **no transform between them was ever saved**. It recovers one by fitting the
19 beads onto the CT bone surface: 1.22 mm residual, beads sitting 0.2–2.9 mm
off the surface (exactly right for glued beads), and the five marker groups
landing 29–34 mm apart along the spine — a correct vertebral pitch.

> **Guard against the draping artifact.** `registration/ACCURACY_REPORT.md`
> documents that ICP of a small patch onto this CT finds *a* low-residual pose,
> not the *correct* one, reporting a deceptively good ~1.9 mm. Two things
> prevent that here: the CT transform is fitted to the **beads only**, never to
> our mesh; and our meshes are placed by marker Kabsch, not ICP, so no draping
> freedom exists.

---

# Stage 12 — results and what they mean

## Recovered parameters vs truth

| shot | level | `b` recovered | `b` true | mean rec depth at markers | mean CT depth |
|---|---|---|---|---|---|
| 004 | 4 | 45.05 | 60.01 | 51.73 | 60.01 |
| 005 | 5 | 45.55 | 55.43 | 43.02 | 55.43 |
| 006 | 6 | 52.32 | 58.86 | 51.59 | 58.86 |

`b` is a *family parameter*, not the depth at the markers — the relief term
displaces it (shot_004: `b` = 45.05 but the markers sit at 51.73 mm).

## Scale decomposition (rec ÷ CT, camera frame)

| shot | lateral | axial | combined | Umeyama `c` |
|---|---|---|---|---|
| 004 | 0.888 | **1.447** | 1.155 | 0.850 |
| 005 | 0.765 | 0.780 | 0.775 | 1.271 |
| 006 | 0.870 | 0.940 | 0.922 | 1.054 |

Two things fall out:

- **Lateral scale is systematically too small (0.765–0.888)** on every level —
  working distance is under-estimated by 11–24%, consistent with `b` sitting at
  the band floor.
- **Axial ratios have no consistent sign** (1.447, 0.780, 0.940). Relief gain is
  uncontrolled per shot. This is also why shot_004 looks anomalous in a results
  table: it is the one level where the relief error *opposes and outweighs* the
  depth error, flipping `c` below 1 even though `b_rec < b_true`.

## Accuracy

| method | FRE (fitted) | LOO (predictive) | worst |
|---|---|---|---|
| **near-field (CT-free)** | 2.68 / 3.07 / 2.78 | **5.11** | 8.18 |
| far-field (CT-anchored) | 1.34 / 1.39 / 6.14 | 9.50 | 23.81 |
| inverse-square (CT-anchored) | 3.23 / 5.90 / 7.20 | 12.64 | 26.96 |

Far-field's flattering FRE **collapses under leave-one-out** — its apparent
accuracy does not generalize. Near-field stays consistent. Note both baselines
are *given* a CT-anchored scale they cannot compute themselves.

## Open problem: error grows away from the fiducials

| distance from nearest bead | median surface error |
|---|---|
| < 10 mm | 2.09 mm |
| 10–20 mm | 4.41 mm |
| > 25 mm | 6.99 mm |

Confirmed **not** an edge artifact (`drift_confound.py`): partial correlation
0.34–0.51 for fiducial distance vs 0.04–0.06 for edge distance, and the two
predictors are independent (r = −0.11).

**Cause:** the entire 30 mm patch is pinned by only two numbers. Photometric
stereo gives slopes, integration gives a relative surface, and two parameters
scale it. Nothing constrains the surface locally, so drift accumulates.

---

# Stage 13 — approaches tried and rejected

Documented so they are not repeated.

| attempt | script | outcome |
|---|---|---|
| Fuse per-pixel attenuation depth into the low frequencies | `nearfield_drift_fix.py` | **No effect** (2.68→2.70). With albedo free per pixel, depth is *unidentifiable*: a 25 mm depth error is reproduced to within 5.9% by rescaling albedo alone. No per-pixel attenuation prior can work. |
| Assume bone albedo is low-frequency-flat | `nearfield_albedo_prior.py` | **Worse** (2.68→4.87, 3.07→7.17, 2.78→8.72). Bone albedo *does* vary slowly — and so does lens vignetting, which is indistinguishable from it. |
| Convergent-aim LED model (`--aim-dx`) | sweep, 6 configs | Worse on all levels; pins `b` at the band floor |
| Shadow rejection (Coleman–Jain, drop the shadowed light) | tested inline | Much worse. With 4 near-coaxial lights the 4th provides *conditioning*, not redundancy — three lights 6 mm off-axis are nearly coplanar |
| Undistorted-ray shading grid (`--undistort-grid`) | sweep | Shuffles error rather than reducing it; suspect the extreme `k3` |
| Robust trimmed objective (`--trim`) | sweep | Helps shot_004 only; fitting per-shot noise |

The albedo-prior failure is the informative one: it points at **lens vignetting
flat-field calibration** as the real fix, since vignetting is a slow
multiplicative field indistinguishable from slow albedo variation.

---

# Stage 14 — bench measurements still needed

These block better numbers. One session, ~15 minutes.

**White-target captures.** Photograph a flat white sheet head-on at 3–4
ruler-measured distances (~10 mm, 40, 55, 70), one frame per LED plus dark. At
~10 mm the beam lands on its own LED's side regardless of aim, which
disambiguates position from aim. This single experiment yields:

1. **μ** — beam-width exponent (currently the provisional 8.0)
2. **beam aim** — parallel or convergent
3. **per-LED power** — replaces the whole-frame median heuristic
4. **vignetting flat-field** — the fix §13 points at
5. **LED wiring in image coordinates** — settles §10

**Focus band.** Confirm 45–70 mm is really where the fixed-focus lens is sharp.
`b` sits at the band floor on levels 4 and 5, so this prior is currently
load-bearing for the metric scale.

Until then `μ = 8` is provisional — chosen because the CT-validated surfaces
prefer it, a mild circularity the bench measurement removes.

---

# Quick reference

```powershell
cd marker_pipeline

# reconstruct
python nearfield_ctfree.py all            # photos -> metric meshes (CT-free)
python clean_mesh_edges.py --all          # trim artifacts

# evaluate
python ct_frame_recover.py                # once: recover the CT frame
python dense_ct_accuracy.py               # dense error vs the CT bone surface
python loo_accuracy.py                    # predictive (leave-one-out) error

# figures and data
python paper_figs.py                      # Fig. 2 overlay + Fig. 3 landscape
python export_chart_data.py               # CSVs for Sheets/Excel
python marker_depth_slides.py             # marker slides (defaults now correct)
python led_direction_figure.py            # which side each LED lights
python per_led_depth_slide.py shot_004    # single-image depth per LED
python farfield_ctfree.py                 # far-field ablation
```

## Viewing in Slicer

Load `shot_004_mesh_ctfree_ct.ply`. Two gotchas:
- **PLY imports as LPS**, Slicer works in RAS internally, so X and Y flip.
  `export_fiducials.py` writes both conventions.
- Point clouds (`*_surface_*.ply`) have no faces and render as nothing — use
  the `_mesh_` files.

## Adding a new shot

1. Capture with `grab4.py` into `Data_collection/calib_charuco/shot_NNN/`.
2. Click the marker pixel coordinates and read the matching CT mm coordinates in
   Slicer.
3. Add the entry to `CT_CORR` in `calibrate_photometric_ct.py`.
4. `python nearfield_ctfree.py shot_NNN`.

## Shared infrastructure

Lives in `../scripts/reconstruction/`:
- `bone_depth_batch.py` — photometric core, image loading, calibration,
  `bone_mask`, Frankot–Chellappa
- `nearfield_lambertian.py` — standalone near-field solver + synthetic self-test

## Baselines and diagnostics

| script | purpose |
|---|---|
| `farfield_ctfree.py` | far-field relief + the same CT-free scale recovery. Far-field has no depth term, so this is generous to it — and it still gives 10.76/1.45/6.32 mm: occasionally accurate, never trustworthy |
| `inverse_square_ct.py` | single-image `1/√I` depth. Correlates +0.20 with CT (near-field: +0.99), LOO 12.6 mm. The structure is real shading; the depth values are not |
| `nearfield_ct.py` | **CT-anchored** near-field — the "oracle" path, fits an affine to CT marker depths each iteration. Reference ceiling only (1.3–1.7 mm), *not* a CT-free result |
| `drift_diagnose.py` | shape of the error — 68–92% high-frequency residual, not a smooth bowl |
| `drift_confound.py` | separates fiducial-distance from edge-distance effects |
| `led_wiring_vote.py` | 24-permutation CT-free wiring vote |
