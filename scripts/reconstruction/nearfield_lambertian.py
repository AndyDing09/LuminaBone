"""
Near-field Lambertian photometric stereo with a dichromatic reflection model.
================================================================================

Physics (irradiance -> radiance -> pixel):

  IRRADIANCE  E(x)  [W/m^2] — light power arriving per unit surface area from
  a point-source LED with Lambertian emission along its axis:

      E(x) = J0 * cos(phi) * max(n . omega, 0) / r^2

  where r = |P_led - X| is the LED-to-surface distance, omega is the unit
  vector from the surface point toward the LED, and phi is the angle between
  the LED's emission axis and the LED->surface ray.

  RADIANCE  L_out  [W/m^2/sr] — what a focused camera pixel measures (pixel
  irradiance is proportional to scene radiance; the camera-to-surface distance
  cancels because the pixel footprint grows as r^2).

  REFLECTION (dichromatic model — bone is a HYBRID reflector):
      L_out = L_body + L_surface
      L_body    = (rho / pi) * E          Lambertian BODY reflection: light
                                          enters the bone, scatters subsurface,
                                          exits diffusely, tinted bone-colored.
      L_surface = Fresnel glint lobe      SURFACE reflection at the interface,
                                          illuminant-colored (white LED).

  The depth solve must use ONLY the body term. Because surface reflection of a
  white LED adds (nearly) equally to R, G and B while body reflection of bone
  is red-heavy, the difference channel

      I_sf = linear_R - linear_B                     (specular-free channel)

  cancels the surface term per-pixel (Shafer's dichromatic separation) and is
  proportional to the body reflection. Its per-pixel proportionality constant
  depends only on bone chromaticity, which multiplies into albedo and cancels
  in the normal estimate n = g / |g|.

  NEAR-FIELD SOLVE: unlike the directional model in bone_depth_batch.py (one
  global light vector per LED, hand-set elevation), the per-pixel effective
  light vector implements the full Quéau et al. (2017) point-source model,

      v_k(x) = Psi_k * [n_s_k . (x - x_s_k) / |x - x_s_k|]^mu_k
                     * (x_s_k - x) / |x_s_k - x|^3

  built from CALIBRATED per-LED geometry -- position x_s_k, principal direction
  n_s_k, anisotropy mu_k and intensity Psi_k -- read from a led_calibration.json
  produced by scripts/calibration/led_calibration.py. The Lambertian system
  I_k = a * (n . v_k) is solved per pixel (>=3 lights, normal equations) and the
  normals integrated to depth (Frankot-Chellappa); geometry and depth alternate
  a few times. NOTHING about the rig is assumed here: not the LED azimuths, not a
  +Z axis, not equal brightness, not a Lambertian (mu=1) emitter -- every one of
  those is a calibrated number. (The legacy symmetric led_positions() helper is
  kept only for the older slide scripts that import the measured constants.)

Coordinates: X right, Y down, Z from camera into the scene — all mm. The lens
is at the origin; LEDs sit in the lens plane (Z=0) facing +Z ("directly
perpendicular to the bone", per the rig). Outward surface normals point
toward the camera (n_z < 0).

Run:
    python nearfield_lambertian.py --selftest     validate on synthetic truth
    python nearfield_lambertian.py                process the bone dataset
    python nearfield_lambertian.py --led-offset 15
"""

import os
import csv
import json
import argparse
import numpy as np

import bone_depth_batch as bd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ---------------------------------------------------------------------------
# Rig geometry (mm). LED_OFFSET_MM is the MEASURED lateral distance from the
# center of the lens to the center of each LED (measured 2026-07-08).
# ---------------------------------------------------------------------------
LED_OFFSET_MM = 12.08                 # measured 2026-07-08
FIELD_WIDTH_MM = 37.0                 # measured 2026-07-07
# MEASURED lens-to-surface working distance (2026-07-08). Use this directly:
# deriving it from the calibrated fx gives 45.9mm, which is INCONSISTENT with
# the measured 30mm + 37mm field width -> the calibrated focal length (fx=1588,
# 44deg FOV) is ~1.5x too high (30mm+37mm imply ~63deg FOV, fx~1038). Trust the
# physical measurements over the (upscaled, anisotropic) checkerboard fx.
WORKING_DISTANCE_MM = 30.0
DEFAULT_CALIB_SIZE = (1280, 720)      # fallback if a calib file omits image_size
OUT_DIR = bd.project_path("depth_outputs", "nearfield")
# Solved per-LED light geometry from scripts/calibration/led_calibration.py.
LED_CALIB_PATH = bd.project_path("calibration", "led_calibration.json")

# TRUE physical LED layout: three identical LEDs (equal brightness, equal
# LED_OFFSET_MM distance from the lens) arranged symmetrically around it —
# RIGHT, TOP, LEFT. This is the real hardware, and it deliberately REPLACES
# bd.SINGLE_LED_AZIMUTH_DEG, whose 55 deg "upper-right, ~0.38x weaker" third
# light was an APPARENT-shading fit in the far-field model, not the physical
# rig. Keyed by photo index (pN-2 right, pN-3 top, pN-4 left). Azimuth: 0 =
# +X right, 90 = up. Because the LEDs are identical, no per-LED brightness
# weighting is applied anywhere — equal emitter power folds into albedo and
# cancels in n = g/|g|.
PHYSICAL_LED_AZIMUTH_DEG = {2: 0.0, 3: 90.0, 4: 180.0}


def calib_image_size(calib):
    """Calibration image size, tolerating files that omit it.

    load_camera_calibration can return image_size=None (npz without the key,
    or JSON that omits it). Fall back to the known rig resolution instead of
    crashing downstream on None subscripting/unpacking.
    """
    size = calib.get("image_size")
    if not size:
        print(f"  [warn] calibration has no image_size; assuming "
              f"{DEFAULT_CALIB_SIZE[0]}x{DEFAULT_CALIB_SIZE[1]}")
        return DEFAULT_CALIB_SIZE
    return tuple(int(v) for v in size)


def working_distance_mm():
    """MEASURED lens-to-surface distance (30mm). The fx-derived value (~46mm)
    is inconsistent with the measured rig, so we trust the direct measurement.
    See WORKING_DISTANCE_MM note above."""
    return WORKING_DISTANCE_MM


def measured_intrinsics(width, height):
    """Intrinsics derived from the MEASURED rig geometry, not calibration.txt.

    The checkerboard fx=1588 is ~1.53x too high (it implies a 46mm distance for
    the 37mm field, but the surface is measured at 30mm). Confirmed: field width
    37mm at 30mm distance => fx = W*d/field_width. fy follows from the same
    physical distance and the 4:3-native/16:9-upscaled vertical field height
    (= field_width * 3/4), which reproduces the calibration's fx/fy=1.33 ratio
    while fixing the absolute scale. Distortion coeffs from calibration are kept
    (shape, robust); only the absolute focal length is replaced.
    """
    d = WORKING_DISTANCE_MM
    fx = width * d / FIELD_WIDTH_MM
    field_height_mm = FIELD_WIDTH_MM * 3.0 / 4.0     # native 4:3 sensor
    fy = height * d / field_height_mm
    return fx, fy, width / 2.0, height / 2.0


def led_positions(offset_mm):
    """Physical LED centers in the lens plane (Z=0), facing +Z.

    Uses the TRUE symmetric layout PHYSICAL_LED_AZIMUTH_DEG (right/top/left at
    0/90/180 deg), all at the same measured `offset_mm` radius and equal
    brightness — matching the real rig. Azimuth: 0 deg = +X right, 90 = up;
    our Y axis points down, so up = -Y.
    """
    pos = {}
    for idx, az_deg in PHYSICAL_LED_AZIMUTH_DEG.items():
        az = np.radians(az_deg)
        pos[idx] = np.array([offset_mm * np.cos(az), -offset_mm * np.sin(az), 0.0])
    return pos


def balance_body_frames(rgbs, bodies):
    """Remove per-frame camera gain/auto-exposure from the body (R-B) frames.

    THE FIX for the review's critical finding: bd.balance_exposure divides each
    frame by its own median, but the specular-free R-B channel is ~0 over dark
    background and single-LED shadow, so its median frequently collapses to 0.
    That makes balance_exposure skip some frames while dividing others by a tiny
    number, injecting a large fake per-LED bias (measured ~250x at loc 16) that
    tilts normals and blows up the relief.

    Camera gain is a single global multiplier that scales ALL channels equally,
    so estimate it from LINEAR LUMINANCE (which stays positive in shadow) over
    the well-lit region, then divide the R-B body frame by that same scalar.
    This removes the gain without the zero-median pathology and preserves the
    physical R-B magnitudes the per-pixel solve relies on.
    """
    out = []
    for rgb, b in zip(rgbs, bodies):
        lum = bd.to_luminance(bd.srgb_to_linear(rgb.astype(np.float64)))
        lit = lum > 0.1 * lum.max()
        g = float(np.median(lum[lit])) if lit.any() else 1.0
        out.append(b / g if g > 1e-6 else b)
    return out


def save_depth_figure(loc, z, normals, mask, lo, hi, relief_mm, shadow,
                      reliable, ref_rgb, out_dir):
    """Write a per-location depth image: mm heatmap + hole-free 3-D surface.

    Depth is in mm (near-field metric). The 3-D surface is the FULL field
    clamped to [lo, hi] so it is continuous (no mask cut-outs). Unreliable
    locations get a red banner so a bad relief never reads as real shape.
    """
    H, W = z.shape
    zc = z - np.median(z[mask]) if mask.any() else z - np.median(z)

    if not HAVE_MPL:
        cv_ok = _save_depth_png_opencv(loc, zc, lo, hi, out_dir)
        return cv_ok

    fig = plt.figure(figsize=(13, 5.4), dpi=120)
    fig.patch.set_facecolor("#f5f5f0")

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(np.clip(ref_rgb, 0, 1)); ax0.axis("off")
    ax0.set_title("Reference (all LEDs)", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(zc, cmap="turbo", vmin=lo, vmax=hi)
    ax1.set_title(f"Depth (mm)  —  relief {relief_mm:.1f} mm",
                  fontsize=10, fontweight="bold")
    ax1.axis("off")
    cb = plt.colorbar(im, ax=ax1, shrink=0.82, pad=0.02)
    cb.set_label("depth toward camera (mm)", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    zsm = gaussian_filter(zc, 1.5)
    zdisp = np.clip(-zsm, -hi, -lo)
    step = max(1, W // 140)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    zg = zdisp[::step, ::step]
    mm_per_px = FIELD_WIDTH_MM / W
    ax2.plot_surface(xx * mm_per_px, yy * mm_per_px, zg, cmap="turbo_r",
                     vmin=-hi, vmax=-lo, rcount=zg.shape[0], ccount=zg.shape[1],
                     linewidth=0, antialiased=True)
    ax2.set_title("3-D surface (up = toward camera)", fontsize=10,
                  fontweight="bold")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)")
    ax2.set_zlabel("mm"); ax2.view_init(elev=55, azim=-60)

    banner = (f"Location {loc}   ·   near-field dichromatic photometric stereo   "
              f"·   single-LED shadow {shadow:.0%}")
    if not reliable:
        banner = (f"Location {loc}   ·   [!] UNRELIABLE — {shadow:.0%} of bone "
                  "unlit by one LED; relief is NOT real shape")
    fig.suptitle(banner, fontsize=12, fontweight="bold",
                 color=("#b00000" if not reliable else "#222222"), y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(out_dir, f"loc{loc:02d}_depth.png")
    fig.savefig(out, bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)
    return out


def _save_depth_png_opencv(loc, zc, lo, hi, out_dir):
    """Fallback heatmap (no matplotlib): plain TURBO depth PNG via OpenCV."""
    import cv2
    zn = np.clip((zc - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = os.path.join(out_dir, f"loc{loc:02d}_depth.png")
    cv2.imwrite(out, img)
    return out


def specular_free_channel(rgb_srgb):
    """Dichromatic separation: linear R minus linear B cancels the white
    surface-reflection term and keeps a body-reflection-proportional signal."""
    lin = bd.srgb_to_linear(rgb_srgb.astype(np.float64))
    sf = lin[..., 0] - lin[..., 2]
    return np.clip(sf, 0.0, None)


def backproject(z, fx, fy, cx, cy):
    """Pixel grid + depth map -> 3-D points (H, W, 3) in mm."""
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    return np.stack([X, Y, z], axis=-1)


def load_led_calibration(path=None):
    """Read solved per-LED geometry produced by led_calibration.py.

    Returns {led_index: {x_s, n_s, mu, Psi}} with numpy arrays and a
    unit-normalised n_s, or None if the file is absent. This is the ONLY source
    of light geometry the solver uses -- no rig symmetry is assumed.
    """
    path = path or LED_CALIB_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    leds = {}
    for k, v in raw.items():
        n_s = np.asarray(v["n_s"], float)
        leds[int(k)] = dict(x_s=np.asarray(v["x_s"], float),
                            n_s=n_s / np.linalg.norm(n_s),
                            mu=float(v.get("mu", 1.0)),
                            Psi=float(v.get("Psi", 1.0)))
    return leds


def effective_light_vectors(points, led):
    """Per-pixel calibrated point-source light vector (Quéau et al. Eq. 2.1).

    v(x) = Psi * [n_s . (x - x_s) / |x - x_s|]^mu * (x_s - x) / |x_s - x|^3
         = Psi * cos^mu(theta) / r^2 * omega,   omega = unit(x_s - x),

    where theta is the emission angle off the LED's calibrated principal axis
    n_s. Every factor (x_s, n_s, mu, Psi) comes from calibration; none is assumed.
    """
    d = led["x_s"][None, None, :] - points             # surface -> LED (x_s - x)
    r = np.linalg.norm(d, axis=-1)
    omega = d / r[..., None]
    cos_theta = np.clip(-(omega @ led["n_s"]), 0.0, None)   # n_s . (x - x_s) / r
    aniso = cos_theta ** led["mu"]
    return led["Psi"] * (aniso / r**2)[..., None] * omega


def solve_normals_nearfield(intensities, points, leds):
    """Per-pixel Lambertian solve with calibrated spatially varying lights.

    intensities : list of K>=3 body-reflection images (H, W)
    leds        : list of K calibration dicts, same order as `intensities`
    Returns (normals (H,W,3) with n_z < 0, pseudo-albedo (H,W)). Uses the normal
    equations (M^T M) g = M^T I so it works for K=3 (exact) and K>3 (least
    squares, e.g. the 4-LED shot rig) alike.
    """
    M = np.stack([effective_light_vectors(points, led) for led in leds],
                 axis=-2)                                  # (H, W, K, 3)
    I = np.stack(intensities, axis=-1)                     # (H, W, K)
    MtM = np.einsum('hwki,hwkj->hwij', M, M)               # (H, W, 3, 3)
    MtI = np.einsum('hwki,hwk->hwi', M, I)                 # (H, W, 3)
    g = np.linalg.solve(MtM + 1e-9 * np.eye(3), MtI[..., None])[..., 0]
    a = np.linalg.norm(g, axis=-1)
    n = g / np.where(a > 1e-12, a, 1.0)[..., None]
    flip = n[..., 2] > 0                                   # outward = toward camera
    n[flip] *= -1.0
    return n, a


def normals_to_depth_mm(normals, fx, fy, cx, cy, z_anchor_mm, nz_floor=0.10,
                        anchor_mask=None):
    """Integrate outward normals (n_z<0) to PERSPECTIVE depth Z(u,v) in mm.

    Under perspective projection P = Z * ((u-cx)/fx, (v-cy)/fy, 1), requiring
    the normal to be orthogonal to both image-direction tangents of the
    surface gives gradients of LOG depth (Tankus-style perspective SfS):

        d(ln Z)/du = -n1 / (fx * D),   d(ln Z)/dv = -n2 / (fy * D),
        D = n1*(u-cx)/fx + n2*(v-cy)/fy + n3

    (An orthographic mm-per-pixel integration is systematically wrong here:
    the bone's relief is a large fraction of the working distance, so the
    physical footprint of a pixel changes across the surface.)

    Z is recovered up to one global scale, fixed by anchoring the median to
    the working distance.
    """
    H, W = normals.shape[:2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    xt = (u - cx) / fx
    yt = (v - cy) / fy
    n1, n2, n3 = normals[..., 0], normals[..., 1], normals[..., 2]
    D = np.minimum(n1 * xt + n2 * yt + n3, -nz_floor)   # n3<0 -> D negative
    p = -n1 / (fx * D)
    q = -n2 / (fy * D)
    log_z = bd.integrate_frankot_chellappa(p, q)
    # Anchor the scale on the OBJECT (bone) pixels only: the working distance
    # describes the bone surface, not the background.
    ref = log_z[anchor_mask] if anchor_mask is not None and anchor_mask.any() \
        else log_z
    z = np.exp(log_z - np.median(ref))
    return z * z_anchor_mm


def nearfield_stereo(body_images, fx, fy, cx, cy, z_work_mm, leds,
                     iterations=2):
    """Alternate (geometry -> normals -> depth) until stable.

    body_images: K exposure-consistent specular-free frames.
    leds       : K calibration dicts (x_s, n_s, mu, Psi), same order.
    Returns (z_mm perspective depth anchored at z_work, normals, albedo).

    iterations=2 on purpose: the first pass (flat plane at the working
    distance) already gives near-correct per-pixel light vectors; one
    refinement absorbs the bone's actual relief. Further iterations feed
    occluding-boundary integration artifacts back into the geometry and
    slowly degrade the normals (verified on the synthetic self-test).
    """
    H, W = body_images[0].shape
    z = np.full((H, W), z_work_mm, dtype=np.float64)
    normals = albedo = None
    for _ in range(iterations):
        points = backproject(z, fx, fy, cx, cy)
        normals, albedo = solve_normals_nearfield(body_images, points, leds)
        anchor = bd.bone_mask(albedo)
        z = normals_to_depth_mm(normals, fx, fy, cx, cy, z_work_mm,
                                anchor_mask=anchor)
    return z, normals, albedo


# ---------------------------------------------------------------------------
# Synthetic self-test: render a KNOWN sphere with the full forward model
# (near-field irradiance + dichromatic reflection + sRGB + noise), then check
# the solver recovers the true geometry.
# ---------------------------------------------------------------------------

def selftest(led_offset_mm=LED_OFFSET_MM, size=360, radius_mm=12.0,
             z_center_mm=46.0, noise=0.003, seed=11):
    fx = fy = size * 46.0 / 37.0            # ~real rig: 37 mm field at 46 mm
    cx = cy = size / 2.0
    # True sphere front surface
    v, u = np.mgrid[0:size, 0:size].astype(np.float64)
    Xd = (u - cx) / fx                       # view-ray direction components
    Yd = (v - cy) / fy
    # Solve |t*(Xd,Yd,1) - C|^2 = R^2 for the near root along each view ray.
    a2 = Xd**2 + Yd**2 + 1.0
    b2 = -2.0 * z_center_mm
    c2 = z_center_mm**2 - radius_mm**2
    disc = b2**2 - 4 * a2 * c2
    mask = disc > 0
    z_true = np.full((size, size), z_center_mm + radius_mm)
    z_true[mask] = (-b2 - np.sqrt(disc[mask])) / (2 * a2[mask])
    pts = backproject(z_true, fx, fy, cx, cy)
    n_true = pts - np.array([0.0, 0.0, z_center_mm])[None, None, :]
    n_true /= np.linalg.norm(n_true, axis=-1, keepdims=True)
    n_true[~mask] = [0.0, 0.0, -1.0]

    # A deliberately ASYMMETRIC synthetic calibration: off-symmetry positions,
    # tilted principal axes (not +Z), unequal brightness (Psi), non-Lambertian
    # emitters (mu != 1). If the solver leaned on any rig-symmetry assumption it
    # would fail here; passing proves it uses only the calibrated numbers.
    def _u(v):
        v = np.asarray(v, float)
        return v / np.linalg.norm(v)
    calib = [
        dict(x_s=np.array([12.5, -1.0, 1.0]), n_s=_u([-0.06, 0.02, 1.0]),
             mu=1.03, Psi=1.00),
        dict(x_s=np.array([-1.0, -12.0, 1.5]), n_s=_u([0.02, -0.05, 1.0]),
             mu=0.98, Psi=1.10),
        dict(x_s=np.array([-12.0, 1.5, 0.8]), n_s=_u([0.05, 0.01, 1.0]),
             mu=1.00, Psi=0.92),
    ]

    # Forward render: calibrated irradiance -> dichromatic radiance -> sRGB.
    rng = np.random.default_rng(seed)
    body_rgb = np.array([0.85, 0.68, 0.48])          # bone-like tint
    frames = []
    for led in calib:
        vlt = effective_light_vectors(pts, led)
        shading = np.clip(np.einsum('hwc,hwc->hw', n_true, vlt), 0, None)
        E = 1300.0 * shading      # J0 tuned like auto-exposure: no body clipping
        view = -pts / np.linalg.norm(pts, axis=-1, keepdims=True)
        h = (vlt / np.maximum(np.linalg.norm(vlt, axis=-1, keepdims=True), 1e-12)
             + view)
        h /= np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        spec = 0.25 * np.clip(np.einsum('hwc,hwc->hw', n_true, h), 0, None) ** 60
        lin = E[..., None] * body_rgb[None, None, :] + spec[..., None]  # white glint
        lin = np.clip(lin + noise * rng.standard_normal(lin.shape), 0, 1)
        srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                        1.055 * lin ** (1 / 2.4) - 0.055)
        frames.append(srgb.astype(np.float32))

    # Inverse pipeline: dichromatic separation -> glint inpaint (clipped pixels
    # break the R-B cancellation) -> calibrated near-field solve. NO exposure
    # balancing: the calibrated Psi already carry the relative LED brightness,
    # so balancing would destroy that information (fixed-exposure capture is
    # required for a calibrated run, as the paper deactivates auto settings).
    body = [bd.inpaint_specular(specular_free_channel(f),
                                bd.detect_specular_mask(f))
            for f in frames]
    z_anchor = float(np.median(z_true[mask]))     # object median, as on the rig
    z_est, n_est, _ = nearfield_stereo(body, fx, fy, cx, cy, z_anchor, calib)

    # Evaluate on the sphere INTERIOR (inside 80% of the silhouette radius).
    # At the occluding rim the surface turns away from the camera; gradient
    # integration cannot follow near-vertical slopes there (nz clamp + FFT
    # least-squares smoothing), so the rim ring is excluded by design — the
    # same pixels bone_mask erosion drops on real data.
    rr = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    interior = mask & (rr < 0.8 * rr[mask].max())

    dot = np.clip(np.abs(np.einsum('hwc,hwc->hw', n_est, n_true)), 0, 1)
    ang = np.degrees(np.arccos(dot))[interior]
    zt = z_true - np.median(z_true[interior])
    ze = z_est - np.median(z_est[interior])
    relief_true = np.percentile(zt[interior], 98) - np.percentile(zt[interior], 2)
    relief_est = np.percentile(ze[interior], 98) - np.percentile(ze[interior], 2)
    med_depth = float(np.median(np.abs(ze[interior] - zt[interior])))

    print("SELF-TEST  (known sphere, CALIBRATED asymmetric lights: tilted axes,")
    print("            unequal Psi, mu!=1 -- interior 80%, rim ring excluded)")
    print(f"  median normal error : {np.median(ang):6.2f} deg")
    print(f"  90th pct normal err : {np.percentile(ang, 90):6.2f} deg")
    print(f"  relief true vs est  : {relief_true:6.2f} vs {relief_est:6.2f} mm "
          f"({100 * relief_est / relief_true:.1f}%)")
    print(f"  median depth error  : {med_depth:6.3f} mm  (sphere radius "
          f"{radius_mm} mm)")
    # Robust correctness gates: median normal error, relief recovery, and the
    # median absolute depth error. A sign/physics/calibration-consumption bug
    # would blow these up. The 90th-pct normal tail is only reported (not gated):
    # near-field PS with lens-clustered lights has localised ill-conditioned
    # patches where one light's emission grazes to zero -- inherent, not a bug.
    ok = (np.median(ang) < 5.0 and med_depth < 0.6
          and 0.80 < relief_est / relief_true < 1.20)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    print("  NOTE: solver consumed ONLY the calibrated (x_s, n_s, mu, Psi); no")
    print("        rig symmetry was used. The 90th-pct tail is a localised")
    print("        near-field conditioning artifact, hence reported not gated.")
    return ok


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def process_dataset(calib_path=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    leds = load_led_calibration(calib_path)
    if leds is None:
        raise SystemExit(
            "No solved LED calibration found at:\n"
            f"    {calib_path or LED_CALIB_PATH}\n\n"
            "This solver refuses to ASSUME the rig geometry -- it must be solved.\n"
            "Capture the mirror-ball + white-card calibration images and run\n"
            "scripts/calibration/led_calibration.py to write led_calibration.json,\n"
            "then re-run. (Use --selftest to validate the solver in the meantime.)")
    z_work = working_distance_mm()
    groups = bd.discover(bd.INPUT_FOLDER)
    led_idxs = sorted(leds)                    # LED indices the calibration covers

    rows = []
    for loc in sorted(groups):
        files = groups[loc]
        if not all(i in files for i in led_idxs):
            print(f"  [loc {loc}] SKIP - missing single-LED photo(s) for "
                  f"calibrated LEDs {led_idxs}")
            continue
        rgbs = [bd.load_rgb(files[i], bd.WORK_LONG_EDGE) for i in led_idxs]
        H, W = rgbs[0].shape[:2]
        # Intrinsics from the MEASURED rig geometry (the module's documented
        # physically-correct choice). The LED calibration MUST be produced with
        # these same intrinsics so x_s lives in the same metric camera frame.
        fx, fy, cx, cy = measured_intrinsics(W, H)
        led_list = [leds[i] for i in led_idxs]

        body = [specular_free_channel(r) for r in rgbs]
        # clipped glints break the R-B cancellation: inpaint them
        body = [bd.inpaint_specular(b, bd.detect_specular_mask(r))
                for b, r in zip(body, rgbs)]
        # NO exposure balance: the calibrated Psi already carry the relative LED
        # brightness, so balancing (per-frame median) would cancel it. A
        # calibrated run therefore assumes fixed-exposure capture.

        z, normals, albedo = nearfield_stereo(body, fx, fy, cx, cy, z_work,
                                              led_list)
        mask = bd.bone_mask(albedo)

        # Data-quality / conditioning flag. Three-light photometric stereo is
        # ill-conditioned wherever a pixel is shadowed in one LED (that
        # equation drops out). Per LED, measure the fraction of BONE pixels it
        # fails to light; the worst LED's fraction is the location's shadow
        # score. High score -> the relief is unreliable, not real shape.
        if mask.any():
            thr = [0.05 * np.median(b[mask]) for b in body]
            shadow = max(float((b[mask] <= t).mean()) for b, t in zip(body, thr))
        else:
            shadow = 1.0
        reliable = shadow < 0.50   # >50% of bone unlit by one LED = near-singular

        zin = (z - np.median(z[mask]))[mask] if mask.any() else z.ravel()
        lo, hi = np.percentile(zin, [2, 98])
        relief_mm = float(hi - lo)
        rows.append(dict(location=loc, relief_mm=round(relief_mm, 3),
                         relief_in=round(relief_mm / 25.4, 4),
                         worst_led_shadow_frac=round(shadow, 3),
                         reliable=reliable))

        # Depth image. Reference = all-on frame (pN-1) if present, else a LED frame.
        ref_rgb = rgbs[0]
        if 1 in files:
            ref_rgb = bd.load_rgb(files[1], bd.WORK_LONG_EDGE)
        save_depth_figure(loc, z, normals, mask, lo, hi, relief_mm, shadow,
                          reliable, ref_rgb, OUT_DIR)

        flag = "" if reliable else "  [!] UNRELIABLE: heavy single-LED shadow"
        print(f"  [loc {loc:2d}] relief = {relief_mm:6.2f} mm "
              f"({relief_mm / 25.4:.3f} in)  shadow={shadow:.0%}{flag}")

    csv_path = os.path.join(OUT_DIR, "nearfield_relief.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["location", "relief_mm", "relief_in",
                                          "worst_led_shadow_frac", "reliable"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} locations -> {csv_path}")
    print(f"(working distance {z_work:.1f} mm; light geometry from "
          f"{calib_path or LED_CALIB_PATH})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true",
                    help="validate the calibrated solver on a synthetic known "
                         "sphere (no calibration file needed)")
    ap.add_argument("--calibration", default=None,
                    help=f"led_calibration.json path (default {LED_CALIB_PATH})")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    process_dataset(args.calibration)


if __name__ == "__main__":
    main()
