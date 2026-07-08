# Landmark registration — endoscope cloud → CT spine model

Method: closed-form **similarity** (rigid rotation + translation + one uniform
scale) from matched fiducial pairs (Umeyama/Horn). Gives an exact, honest
**fiducial registration error (FRE)** in millimetres and — because the endoscope
depth is only *relative* — recovers the missing metric scale from the CT.

## Why landmarks and not automatic ICP
The endoscope clouds are near-planar brightness-relief patches (depth std ≈ 4 px
vs 640×360 in-plane). Surface ICP onto a curved vertebra is under-constrained and
returns confident-but-arbitrary poses. Matched landmarks remove that ambiguity.

## What you do in 3D Slicer (one time)

1. **Export the CT bone surface**
   - Segment the vertebra (Segment Editor) → *Segmentation* module →
     *Export to files* → STL (or right-click the model → Export).
   - Save it into this `registration/` folder, e.g. `ct_spine.stl`.

2. **Load the endoscope cloud** you want to register
   - File → Add Data → pick one `depth_outputs/ply_allon/locNN_allon.ply`
     (it loads as a Model, with its photo colour). Start with a location that
     covers a **curved / featured** region, not a flat face.

3. **Place matched fiducials (same real point on each, same order)**
   - Markups module → create a point list named **`source`** → click 4–8 clearly
     identifiable features **on the endoscope model** (foramen edge, a ridge, a
     corner, a vessel groove…).
   - Create a second list named **`target`** → click the **same physical points,
     in the same order**, on the CT model.
   - **Important:** don't put all points on a single flat patch — include points
     at different depths so out-of-plane orientation is constrained. Aim for ≥4,
     ideally 6+.
   - Save each list: right-click → *Export* (or File → Save) as
     `source.mrk.json` and `target.mrk.json` into this folder.
   - `.fcsv` or a plain `x,y,z` CSV also work.

## Then I run (or you can run):

```
python landmark_register.py \
    --source-fids registration/source.mrk.json \
    --target-fids registration/target.mrk.json \
    --cloud       depth_outputs/ply_allon/locNN_allon.ply \
    --outdir      registration/reg_out
```

### Optional stage 2 — ICP surface refinement
After the landmark alignment, tighten the fit against the full CT surface:

```
python icp_refine.py \
    --source registration/reg_out/locNN_allon_registered.ply \
    --target registration/ct_spine.stl \
    --outdir registration/reg_out --trim 0.8
```
Point-to-plane, trimmed (rejects the non-overlapping part of the vertebra as
outliers). It only *refines* — it cannot fix a bad landmark start, and on a flat
patch it can slide in-plane, so keep the landmark FRE as the primary trust metric
and eyeball the overlay. Add `--with-scale` only if you want ICP to nudge scale too.

Outputs land in `registration/reg_out/`:
- `locNN_allon_registered.ply` — the cloud moved into CT space (load in Slicer, it
  drops onto the bone; no transform juggling needed).
- `transform_source_to_target.txt` / `.npy` — the 4×4 matrix + recovered scale.
- `transform_itk.tfm` — import via Slicer → Transforms if you'd rather keep the
  original cloud and apply a transform node.
- `registration_report.json` — recovered **scale**, **FRE rms/max**, per-landmark
  residuals, and a near-planar conditioning warning.

## Reading the result
- **FRE rms** ≈ how well the points line up (mm). With careful picking on a
  featured region expect low single-digit mm; large values mean mismatched or
  coplanar points.
- **scale** is the recovered mm-per-cloud-unit. Cross-check: the pipeline's own
  `depth_estimates.csv` implies ≈0.046875 mm/px for Z only — X/Y are raw pixels,
  so the landmark scale tells you the true in-plane mm/px.
- **near_planar_warning = true** → add landmarks with more depth variation.
