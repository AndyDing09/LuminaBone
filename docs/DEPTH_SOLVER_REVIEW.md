# Depth Estimate Solver Review

## Source Of Truth

The current depth estimate solver is in `bone_depth_batch.py`. The exact solve
path is:

1. `process_location()`
2. `solve_luminance()`
3. `balance_exposure()`
4. `photometric_stereo()`
5. `normals_to_depth()`
6. `integrate_frankot_chellappa()`

`bone_depth_values.py`, `depth_recompute.py`, and `make_triangulation_slides.py`
reuse that same solver. Scripts with `allon` in the name use the older
single-image brightness-to-depth approximation from `endoscope_depth.py`; those
are useful for quick colored meshes, but they are not the main three-light depth
estimate.

## Inputs

Each location is expected to have four photos:

- `pN-1`: all LEDs on, used as a visual/color reference
- `pN-2`: right LED only
- `pN-3`: upper LED only
- `pN-4`: left LED only

The depth solve uses only the three single-LED photos. The all-on photo does not
drive the photometric-stereo solve.

## Calculation Flow

First, each single-LED RGB image is converted to linear luminance. The code
undoes sRGB gamma, converts RGB to a brightness channel, removes specular glints,
and balances exposure by dividing each light image by its median brightness.
This matters because photometric stereo assumes brightness differences come from
surface orientation, not camera auto-exposure or shiny highlights.

Next, the solver builds a light-direction matrix `L`. Each row is a known unit
light vector from `SINGLE_LED_AZIMUTH_DEG` and `LIGHT_ELEVATION_DEG`.

For every pixel, the three observed brightnesses form:

```text
L @ g = I
```

where `I` is the pixel brightness under the three LEDs and `g` is the unknown
albedo-scaled normal. With three lights, the code uses an exact inverse:

```text
g = inverse(L) @ I
```

Then:

```text
albedo = norm(g)
normal = g / albedo
```

After normals are estimated, `normals_to_depth()` converts them into depth
gradients. With the repo's convention, larger depth means farther from the
camera:

```text
d(depth)/d(col) = +nx / nz
d(depth)/d(row) = -ny / nz
```

Finally, `integrate_frankot_chellappa()` solves for the depth map whose
gradients best match those two gradient fields. It does this in the frequency
domain with an FFT Poisson solve. The result is a relative depth map.

## Units

The native output is relative depth in working-resolution pixel units. The
relative relief shape is meaningful, but the absolute physical height is not
known from photometric stereo alone.

The scripts now report inch-scaled relief using:

```text
inches_per_pixel = FIELD_WIDTH_IN / image_width_px
relief_in = relief_px * inches_per_pixel
```

This is only physically correct if `FIELD_WIDTH_IN` is the measured real width
of the photographed area. The default keeps the previous 30 mm placeholder,
converted to inches, so update it before treating the inch values as true.

## Review Notes

- The main photometric-stereo math is internally consistent.
- The sign convention is depth-away-from-camera; 3D plots negate depth so bone
  protrusions point upward/toward the viewer.
- The strongest uncertainty is absolute scale. `LIGHT_ELEVATION_DEG` changes
  vertical relief, and `FIELD_WIDTH_IN` converts pixel relief to physical units.
- The `pN-3` upper light is calibrated as an upper-right light, not a pure top
  light. That is already reflected in `SINGLE_LED_AZIMUTH_DEG`.
- The older `allon` mesh scripts use a simpler inverse-square brightness model.
  Use the photometric-stereo outputs for the best available depth relief.
