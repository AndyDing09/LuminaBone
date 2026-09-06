# marker_pipeline — from raw LED photos to a CT-registered mesh

This is the working pipeline for the CT-free near-field photometric-stereo
reconstruction. It turns **five photographs** of a vertebra into a **metric 3-D
mesh in millimetres**, using no CT information anywhere in the reconstruction.
CT is used only afterwards, to score the result.

Current accuracy (levels 4/5/6, rigid 6-DOF alignment, nothing fitted to CT):

| metric | value | what it means |
|---|---|---|
| FRE at the fiducials | 2.68 / 3.07 / 2.78 mm | error where the alignment was fitted (optimistic) |
| Leave-one-out | 5.11 mm rms | error at a fiducial the alignment never saw (predictive) |
| Dense vs CT surface | 2.79 mm median | every vertex against the real bone |
| — within 10 mm of a fiducial | 2.09 mm | |
| — beyond 25 mm of a fiducial | 6.99 mm | **known open problem, see §6** |

---

## 0. What the raw data is

```
Data_collection/calib_charuco/shot_004/
├── led1.png   one LED on   (measured: lights BOTTOM)
├── led2.png   one LED on   (measured: lights TOP)
├── led3.png   one LED on   (measured: lights LEFT)
├── led4.png   one LED on   (measured: lights RIGHT)
├── dark.png   all LEDs off (ambient + sensor baseline)
└── notes.txt  capture log
```

640×480, camera and specimen fixed — **only the light changes between frames**.
That is the entire measurement; everything else is computed.

Captured with `../Data_collection/grab4.py`, which drives the Arduino
(`L1`..`L4` over serial at 115200).

---

## 1. Preprocess — photos to comparable brightness maps

Handled inside `bone_depth_batch.py`; you never call it directly.

1. **Dark subtraction** `led_i − dark` — removes ambient light and sensor bias,
   leaving only the light that LED contributed.
2. **Linear luminance** (`solve_luminance`) — undoes sRGB gamma, converts to a
   single brightness channel, inpaints specular glints on the marker beads.
3. **Exposure balance** (`balance_exposure`) — divides each frame by its median.
   The frames are separate JPEGs with independent auto-exposure, and the LEDs
   differ in power by up to 2×. *This step is load-bearing*: removing it makes
   the reconstruction come out **inverted** (correlation with CT goes negative).

---

## 2. Reconstruct — the main event

```bash
cd marker_pipeline
python nearfield_ctfree.py shot_004          # one shot
python nearfield_ctfree.py all               # every shot in CT_CORR
```

Outputs to `../depth_outputs/ct_registered/`:

| file | what it is |
|---|---|
| `shot_004_mesh_ctfree_cam.ply` | mesh in **camera** coordinates — the actual product |
| `shot_004_mesh_ctfree_ct.ply` | same mesh rigidly placed in CT coordinates, for Slicer |

### What it does inside

**a. Normals from shading.** Each LED is a point source at a known position
6.05 mm off-axis, so every pixel gets its own light direction and 1/r²
falloff. Four brightness values + three unknowns (albedo × normal) per pixel →
solve by least squares.

**b. Integrate to a surface.** Frankot–Chellappa turns the field of slopes into
a height map. This is only correct **up to scale and offset** — integration
throws the absolute size away.

**c. Recover the metric scale — the part with no CT in it.** Depth is modelled
as a two-parameter family

```
D(u,v) = b + s · (b/f) · z̃(u,v)
```

`b` = working distance, `s` = relief gain. Both are found by minimising the
**four-image photometric residual**: build the candidate surface, compute its
normals *from that surface*, predict all four images, and compare. Four lights
overdetermine three unknowns per pixel, so a wrong geometry leaves a misfit no
albedo can absorb. Because the LED ring radius (6.05 mm) is a real measured
length, the minimum picks a real millimetre scale.

Two things pin the search:
- `b` is restricted to the lens **focus band 45–70 mm** (a bench property of the
  fixed-focus optics, not CT).
- `μ` (LED beam-width exponent) is a **rig constant** read from `ctfree_mu.json`,
  currently 8.0. It must not float per shot — that opens a (b, μ) degeneracy.

**d. Back-project.** Each pixel becomes a ray via `cv2.undistortPoints`, scaled
by its depth → 3-D point in camera coordinates.

**e. Trim and mesh.** Drop pixels with high photometric residual, depth spikes,
frame borders, and small components; then connect adjacent pixels into
triangles, skipping any quad spanning more than 3 mm of depth.

### Useful flags

```bash
--mu 8              # override the rig emission exponent
--wiring 270,90,180,0   # LED index -> azimuth, for wiring experiments
--aim-dx 30         # convergent-aim LED model (off by default; see §7)
--trim 0.10         # robust objective, drops the worst 10% of pixels
--undistort-grid    # shading solve on undistorted rays (off; see §7)
```

---

## 3. Clean the mesh

```bash
python clean_mesh_edges.py --all                  # every *_ct.ply -> *_clean.ply
python clean_mesh_edges.py in.ply out.ply --peel 2 --max-tilt 60
```

Removes, in order: stretched triangles bridging depth cliffs, **flip-up walls**
(triangles standing more than `--max-tilt` from their patch's dominant
orientation), boundary rings (`--peel`), and small components.

---

## 4. View it in Slicer

Load `shot_004_mesh_ctfree_ct.ply`. Two gotchas:

- **PLY imports as LPS**, Slicer works internally in RAS, so X and Y flip.
  `export_fiducials.py` writes both conventions if you need matching points.
- Point clouds (`*_surface_*.ply`) have no faces and render as nothing. Use the
  `_mesh_` files.

---

## 5. Evaluate (this is where CT finally enters)

```bash
python loo_accuracy.py           # leave-one-marker-out predictive error
python ct_frame_recover.py       # recover the CT_CORR <-> ct_spine.stl transform
python dense_ct_accuracy.py      # every vertex vs the real CT bone surface
```

`ct_frame_recover.py` must be run once before `dense_ct_accuracy.py`; it fits
the 19 beads onto the CT bone surface (1.22 mm residual, beads 0.2–2.9 mm off
the surface exactly as glued beads should be) and writes
`ct_frame_transform.npz`.

Marker correspondences (image pixel ↔ CT mm) live hand-entered in
`calibrate_photometric_ct.py` → `CT_CORR`. To add a shot, add its entry there.

**Why leave-one-out matters:** FRE is measured at the same points the alignment
was fitted to, so it is biased low by √(1−2/n) ≈ 29% at n=4. LOO fits on 3
markers and tests on the 4th, which is the honest predictive number.

---

## 6. Known open problem — error grows away from the fiducials

Measured: 2.09 mm within 10 mm of a bead, 6.99 mm beyond 25 mm. Established by
`dense_ct_accuracy.py`; confirmed **not** an edge artifact by `drift_confound.py`
(partial correlation 0.34–0.51 for fiducial distance vs 0.04–0.06 for edge
distance, and the two are independent).

**Cause.** The entire 30 mm patch is pinned by only two numbers (`b`, `s`).
Photometric stereo gives slopes; integration gives a relative surface; two
parameters scale it. Nothing constrains the surface locally, so drift
accumulates with distance.

**Two fixes tried and rejected — do not repeat them:**

| attempt | script | result |
|---|---|---|
| Fuse per-pixel attenuation depth into the low frequencies | `nearfield_drift_fix.py` | **No effect** (2.68→2.70). Reason: with albedo free per pixel, depth is *unidentifiable* — a 25 mm depth error is reproduced to within 5.9% by rescaling albedo alone. No per-pixel attenuation prior can work. |
| Assume bone albedo is low-frequency-flat, attribute slow brightness variation to depth | `nearfield_albedo_prior.py` | **Worse** (2.68→4.87, 3.07→7.17, 2.78→8.72). Bone albedo *does* vary slowly — and so does lens vignetting, which is indistinguishable from it. |

The second failure points at the real fix: **flat-field the lens vignetting**,
so slow brightness variation can be attributed to geometry honestly. That is a
bench measurement, not a code change (§7).

---

## 7. Bench measurements still needed

These are the blockers on better numbers. All are one-time, ~15 minutes total.

1. **White-target captures.** Photograph a flat white sheet head-on at 3–4
   ruler-measured distances (~10 mm, 40, 55, 70), one frame per LED plus dark.
   This yields at once: μ (beam width), beam aim direction, per-LED power, the
   vignetting flat-field, and the LED index→position wiring in image
   coordinates — every hardware constant currently guessed.
2. **Focus band.** Confirm 45–70 mm is really where the fixed-focus lens is
   sharp. `b` currently sits at the band floor on levels 4/5, so this prior is
   load-bearing for the metric scale.

Until then `μ = 8` in `ctfree_mu.json` is provisional (chosen because the
CT-validated surfaces prefer it — a mild circularity that the bench measurement
removes).

### On the LED wiring

`led_wiring_vote.py` scores all 24 index→azimuth assignments by the CT-free
photometric residual; the wiring in use ranks #1 of 24. **But** the direct
measurement in `led_direction_figure.py` shows led1 lighting the *bottom* and
led2 the *top*, i.e. the vertical pair swapped relative to the code, consistently
across three shots. The vote's #1 and #2 differ by exactly that swap and are
only 3% apart, so it cannot settle the question. Unresolved; the white-target
capture settles it.

---

## 8. Figures

```bash
python paper_figs.py               # fig_landscape.png + fig_overlay.png (Fig. 2)
python led_direction_figure.py     # which side each LED actually lights
python per_led_depth_slide.py      # single-image depth, one panel per LED
python per_led_nearfield_slide.py  # same, but inverting the near-field model
python marker_depth_slides.py --shots shot_004 \
       --shots-dir ../Data_collection/calib_charuco \
       --out-dir ../depth_outputs/calib_charuco_slides
```

`paper_figs.py` reads cached `*_plotdata_mu8.npz`, **not** the PLY files —
regenerate those with `paper_ctfree_eval2.py` if the reconstruction changes, or
the figure will silently show stale results.

---

## 9. Baselines and diagnostics

| script | purpose |
|---|---|
| `farfield_ctfree.py` | far-field relief + the same CT-free scale recovery. Far-field has no depth term at all, so this is generous to it — and it still gives 10.76/1.45/6.32 mm: occasionally accurate, never trustworthy |
| `inverse_square_ct.py` | single-image `1/√I` depth. Correlates +0.20 with CT (near-field: +0.99); LOO 12.6 mm. Structure is real shading, the depth values are not |
| `nearfield_ct.py` | **CT-anchored** near-field (the old "oracle" path — fits an affine to CT marker depths each iteration). Reference ceiling only, 1.3–1.7 mm; not a CT-free result |
| `drift_diagnose.py` | what shape the error is (68–92% high-frequency residual, not a smooth bowl) |
| `led_incidence_check.py` | cast-shadow light-direction test. **Inconclusive by construction**: near-coaxial geometry displaces a bead shadow by only ~3 px |
| `analyze_envelope.py` | working distances and coplanarity per shot |

Legacy `*_shot011*` scripts predate the marker-based work and are kept only for
reference.

---

## Quick reference — the whole thing

```bash
cd marker_pipeline
python nearfield_ctfree.py all        # photos -> metric meshes  (CT-free)
python clean_mesh_edges.py --all      # trim artifacts
python ct_frame_recover.py            # once: recover the CT frame
python dense_ct_accuracy.py           # score against the CT surface
python loo_accuracy.py                # predictive error
python paper_figs.py                  # figures
```

Shared infrastructure lives in `../scripts/reconstruction/`
(`bone_depth_batch.py` for the photometric core and image helpers,
`nearfield_lambertian.py` for the standalone near-field solver and its
synthetic self-test).
