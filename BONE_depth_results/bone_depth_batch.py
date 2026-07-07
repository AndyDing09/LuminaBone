"""
Batch photometric-stereo depth estimation for the "BONE PICTURE" dataset.
================================================================================

This processes a whole folder of bone photos automatically. The capture rig
takes 4 photos per physical location/sample:

    pN-1.jpg   ALL LEDs on        (uniform reference - NOT used for stereo)
    pN-2.jpg   RIGHT LED only     (light from the +x / right side)
    pN-3.jpg   TOP   LED only     (light from the +y / upper side)
    pN-4.jpg   LEFT  LED only     (light from the -x / left side)

(The -1 = all-on / -2 = right / -3 = top / -4 = left convention was verified
 on all 17 locations: index 1 is always the most uniform image, and the other
 three are lit from the right, top and left respectively.)

METHOD  (identical in spirit to photometric_depth_test.py)
--------------------------------------------------------------------------------
  1. Take the 3 single-LED images. Each gives one brightness per pixel.
  2. Woodham photometric stereo: for every pixel solve the 3x3 linear system
         L @ g = I        ->     g = L^-1 @ I
     where the rows of L are the 3 known light directions and I is the pixel's
     3 brightnesses. Then  albedo = |g|  and  surface normal n = g / |g|.
  3. Turn the per-pixel normals into surface gradients
         p = dz/dx = -nx/nz ,   q = dz/dy = +ny/nz   (image coords)
     and integrate them into a depth map z(x, y) with the Frankot-Chellappa
     Poisson solver (FFT, fast, no per-pixel loop -- this is the frequency
     domain integrator named in the original script's docstring).
  4. The result is a PER-PIXEL RELATIVE depth map (one heat-map per location)
     plus a few summary numbers per location written to a CSV.

UNITS / WHAT THE NUMBERS MEAN
--------------------------------------------------------------------------------
Plain photometric stereo recovers shape only up to an unknown overall offset
and scale (there is no coaxial point-light calibration image here, and you
asked to ignore the all-on shot). So the depth map is RELATIVE: the *shape*
and the relative height differences are meaningful, the absolute millimetre
value is not. The single number reported per location is the surface "relief"
(robust peak-to-valley range) in working-resolution pixel units. See the note
at the bottom of this file for how to add absolute mm calibration later.

HOW TO RUN
--------------------------------------------------------------------------------
    pip install numpy opencv-python scipy matplotlib
    python bone_depth_batch.py

Edit INPUT_FOLDER below to point at your downloaded copy of the folder.
Everything else (light geometry, working resolution, what to save) is in the
CONFIG block right under the imports so you can tweak it without touching the
algorithm.
"""

import os
import re
import csv
import glob
import numpy as np
import cv2

# Matplotlib is only needed for the labelled heat-map / panel figures.
# If you don't have it, set SAVE_FIGURES = False and you'll still get the
# raw colour depth PNGs (via OpenCV) and the CSV.
try:
    import matplotlib
    matplotlib.use("Agg")          # no display needed, just write files
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# =============================================================================
# CONFIG  -- edit these, not the algorithm below
# =============================================================================

# Folder containing the pN-M.jpg images (downloaded from the Drive folder).
INPUT_FOLDER = "./bone_picture"

# Where results are written (created automatically).
OUTPUT_FOLDER = "./depth_outputs"

# Filename pattern. Group 1 = location number, group 2 = photo number (1..4).
FILENAME_RE = re.compile(r"^p(\d+)-(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

# Which photo index is the "all LEDs on" reference (ignored by the stereo solve,
# used only for colour / visualisation). Set to None if you have no such frame.
ALL_ON_INDEX = 1

# The three single-LED photos and the in-plane azimuth of each light.
#   azimuth 0 deg  = light comes from the RIGHT (+x)
#   azimuth 90 deg = light comes from the TOP / above (+y, "up" in the image)
#   azimuth 180    = light comes from the LEFT (-x)
#   azimuth 270    = light comes from BELOW (-y)
# These match the measured rig (verified on all 17 locations). Adjust if your
# rig differs, or set ESTIMATE_AZIMUTH_FROM_IMAGES = True to auto-detect.
SINGLE_LED_AZIMUTH_DEG = {
    2: 0.0,     # pN-2  -> RIGHT LED
    3: 90.0,    # pN-3  -> TOP   LED
    4: 180.0,   # pN-4  -> LEFT  LED
}

# Elevation of every LED above the surface plane, in degrees. 90 = straight on
# (coaxial), 0 = grazing from the side. The cameras/LEDs here light the bone
# from fairly oblique angles; ~40 deg is a reasonable default. This mainly
# affects the vertical exaggeration of the recovered surface, not its shape.
LIGHT_ELEVATION_DEG = 40.0

# If True, ignore SINGLE_LED_AZIMUTH_DEG and instead estimate each LED's azimuth
# from the data (image minus the per-location mean -> where that LED adds light).
# Robust and rig-agnostic, but the manual values above are usually cleaner.
ESTIMATE_AZIMUTH_FROM_IMAGES = False

# Downscale the long edge to this many pixels before processing. Photometric
# depth doesn't need full resolution and this keeps each location fast.
# Set to None to use full resolution (slower, sharper).
WORK_LONG_EDGE = 640

# Remove specular glints (bright, low-saturation spots) before solving. Bone is
# glossy when wet and specular highlights break the Lambertian assumption.
REMOVE_SPECULAR = True

# Sign of the output depth. With the convention here, larger value = farther
# from the camera. Flip this if your heat-maps look inverted for your rig.
DEPTH_SIGN = +1.0

# What to write per location.
SAVE_DEPTH_PNG = True     # colour depth heat-map (always; uses OpenCV if no MPL)
SAVE_NORMALS_PNG = True   # RGB-encoded surface normals
SAVE_PANEL = True         # multi-panel figure (needs matplotlib)
SAVE_PLY = False          # 3-D point cloud (open in MeshLab/CloudCompare)

# Robust percentiles used both for the colour range and the "relief" number.
LOW_PCT, HIGH_PCT = 2.0, 98.0


# =============================================================================
# Light geometry
# =============================================================================

def light_vector(azimuth_deg, elevation_deg):
    """Unit light direction in camera coords (+x right, +y up, +z toward camera).

    azimuth is measured in the image plane (0 = right, 90 = up); elevation is
    the angle lifted out of that plane toward the camera.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    v = np.array([np.cos(el) * np.cos(az),
                  np.cos(el) * np.sin(az),
                  np.sin(el)], dtype=np.float64)
    return v / np.linalg.norm(v)


def estimate_azimuths(single_imgs):
    """Estimate each single-LED azimuth from the images themselves.

    The mean of the single-LED frames approximates the all-on illumination;
    (frame - mean) isolates where that one LED adds light, and the brightness
    centroid of that residual points along the LED's in-plane direction.
    Returns a list of azimuths (deg) aligned with `single_imgs`.
    """
    mean = np.mean(single_imgs, axis=0)
    H, W = mean.shape
    ys, xs = np.mgrid[0:H, 0:W]
    azs = []
    for g in single_imgs:
        pos = np.clip(g - mean, 0, None)
        tot = pos.sum() + 1e-9
        ox = (xs * pos).sum() / tot - W / 2.0      # +right
        oy = (ys * pos).sum() / tot - H / 2.0      # +down
        azs.append(np.degrees(np.arctan2(-oy, ox)))  # -oy -> +up
    return azs


# =============================================================================
# Image helpers
# =============================================================================

def to_luminance(rgb):
    """RGB (float 0..1, channel order R,G,B) -> single brightness channel.

    Standard Rec.601 luma; bone is near-grey so the exact weights barely matter.
    """
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def load_rgb(path, long_edge=None):
    """Load an image as float RGB in [0, 1], optionally downscaled."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"could not read {path}")
    if long_edge is not None:
        h, w = bgr.shape[:2]
        scale = long_edge / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def detect_specular_mask(rgb, brightness_thresh=0.92, sat_thresh=0.25):
    """Mask pixels that are both very bright AND low-saturation (mirror glints)."""
    lum = to_luminance(rgb)
    max_c = rgb.max(axis=-1)
    min_c = rgb.min(axis=-1)
    sat = np.where(max_c > 1e-6, (max_c - min_c) / (max_c + 1e-6), 0.0)
    return (lum > brightness_thresh) & (sat < sat_thresh)


def inpaint_specular(lum, mask, iterations=15):
    """Fill masked (glint) pixels by repeated neighbour averaging."""
    if not mask.any():
        return lum
    from scipy.ndimage import gaussian_filter
    filled = lum.copy()
    filled[mask] = 0.0
    for _ in range(iterations):
        blur = gaussian_filter(filled, sigma=2.0)
        filled[mask] = blur[mask]
    return filled


# =============================================================================
# Core: photometric stereo  (Woodham 1980, same as photometric_depth_test.py)
# =============================================================================

def photometric_stereo(luminances, light_dirs):
    """Per-pixel surface normals + albedo from >=3 lit images.

    luminances : list of (H, W) brightness arrays, one per light.
    light_dirs : (K, 3) array, row k is the direction of light k.
    Returns (normals (H,W,3), albedo (H,W)).
    """
    K = len(luminances)
    H, W = luminances[0].shape
    I = np.stack([l.ravel() for l in luminances], axis=0)   # (K, N)

    L = np.asarray(light_dirs, dtype=np.float64)            # (K, 3)
    # 3 lights -> exact inverse; >3 lights -> least squares pseudo-inverse.
    L_pinv = np.linalg.inv(L) if K == 3 else np.linalg.pinv(L)
    G = L_pinv @ I                                          # (3, N)

    g = G.T.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=-1)
    safe = np.where(albedo > 1e-5, albedo, 1.0)
    normals = g / safe[..., None]
    # Make normals face the camera (nz >= 0) and avoid div-by-zero later.
    flip = normals[..., 2] < 0
    normals[flip] *= -1.0
    return normals, albedo


# =============================================================================
# Core: integrate normals -> relative depth (Frankot-Chellappa, FFT)
# =============================================================================

def integrate_frankot_chellappa(p, q):
    """Recover z(x,y) from gradients p=dz/dx, q=dz/dy in the frequency domain.

    Global least-squares fit of a depth map whose gradients best match (p, q).
    O(N log N), fully vectorised. Depth is returned up to an additive constant
    (the DC term is set to zero), i.e. RELATIVE depth.
    """
    H, W = p.shape
    fx = np.fft.fftfreq(W) * 2.0 * np.pi
    fy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(fx, fy)
    denom = wx**2 + wy**2
    denom[0, 0] = 1.0                       # avoid 0/0 at DC

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z))
    return z


def normals_to_depth(normals, nz_floor=0.10):
    """Normals -> gradients -> relative depth via Frankot-Chellappa."""
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = np.clip(normals[..., 2], nz_floor, None)   # stop slopes blowing up
    # Image-coordinate gradients (column = +x right, row = +y down).
    p = -nx / nz          # dz/dx
    q = ny / nz           # dz/dy  (image row runs downward -> sign flip on ny)
    z = integrate_frankot_chellappa(p, q)
    return z


# =============================================================================
# Output helpers
# =============================================================================

def colourize_depth(z, low, high):
    """Map a depth array to a uint8 BGR TURBO image for quick viewing."""
    zc = np.clip((z - low) / max(high - low, 1e-9), 0, 1)
    u8 = (zc * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def save_ply(path, z, rgb, sign=1.0):
    """Write a colored point cloud: one vertex per pixel (x, y, depth)."""
    H, W = z.shape
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    X = (j - W / 2.0).astype(np.float32)
    Y = -(i - H / 2.0).astype(np.float32)
    Z = (sign * z).astype(np.float32)
    cols = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    cset = cols.reshape(-1, 3)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, zz), (r, g, b) in zip(pts, cset):
            f.write(f"{x:.2f} {y:.2f} {zz:.4f} {r} {g} {b}\n")


# =============================================================================
# Per-location processing
# =============================================================================

def process_location(loc, files, out_dir):
    """Run the full pipeline for one location. Returns a stats dict (or None)."""
    # Load every available photo for this location.
    imgs_rgb = {}
    for idx, path in sorted(files.items()):
        imgs_rgb[idx] = load_rgb(path, WORK_LONG_EDGE)

    # The single-LED indices we expect for the stereo solve.
    led_indices = sorted(SINGLE_LED_AZIMUTH_DEG.keys())
    missing = [i for i in led_indices if i not in imgs_rgb]
    if missing:
        print(f"  [loc {loc}] SKIP - missing single-LED photo(s) {missing}")
        return None

    H, W = imgs_rgb[led_indices[0]].shape[:2]

    # Luminance per single-LED image (+ optional glint removal).
    luminances = []
    for i in led_indices:
        lum = to_luminance(imgs_rgb[i])
        if REMOVE_SPECULAR:
            lum = inpaint_specular(lum, detect_specular_mask(imgs_rgb[i]))
        luminances.append(lum)

    # Light directions: manual azimuths (default) or estimated from the images.
    if ESTIMATE_AZIMUTH_FROM_IMAGES:
        azimuths = estimate_azimuths(luminances)
    else:
        azimuths = [SINGLE_LED_AZIMUTH_DEG[i] for i in led_indices]
    light_dirs = np.array([light_vector(a, LIGHT_ELEVATION_DEG) for a in azimuths])

    # --- the method ---
    normals, albedo = photometric_stereo(luminances, light_dirs)
    z = DEPTH_SIGN * normals_to_depth(normals)

    # Summary stats (relative, working-pixel units).
    low = np.percentile(z, LOW_PCT)
    high = np.percentile(z, HIGH_PCT)
    relief = float(high - low)
    stats = dict(location=loc, width=W, height=H,
                 n_lights=len(led_indices),
                 azimuths_deg=";".join(f"{a:.0f}" for a in azimuths),
                 depth_min=float(z.min()), depth_max=float(z.max()),
                 depth_mean=float(z.mean()), depth_std=float(z.std()),
                 relief_p2_p98=relief)

    # Reference colour image for overlays/PLY: the all-on frame if present.
    ref_rgb = imgs_rgb.get(ALL_ON_INDEX, imgs_rgb[led_indices[0]])

    # --- outputs ---
    if SAVE_DEPTH_PNG:
        cv2.imwrite(os.path.join(out_dir, f"loc{loc:02d}_depth.png"),
                    colourize_depth(z, low, high))
    if SAVE_NORMALS_PNG:
        nviz = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"loc{loc:02d}_normals.png"),
                    cv2.cvtColor(nviz, cv2.COLOR_RGB2BGR))
    if SAVE_PLY:
        save_ply(os.path.join(out_dir, f"loc{loc:02d}.ply"), z, ref_rgb,
                 sign=1.0)
    if SAVE_PANEL and HAVE_MPL:
        _save_panel(loc, ref_rgb, normals, albedo, z, low, high, out_dir)

    print(f"  [loc {loc}] ok  size={W}x{H}  relief(p2-p98)={relief:8.2f}  "
          f"az=[{stats['azimuths_deg']}]")
    return stats


def _save_panel(loc, ref_rgb, normals, albedo, z, low, high, out_dir):
    """4-panel summary figure: reference photo, normals, depth, 3-D surface."""
    fig = plt.figure(figsize=(13, 9), dpi=110)
    fig.patch.set_facecolor("#f5f5f0")

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(ref_rgb); ax.set_title("Reference (all-LEDs-on)", fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow((normals + 1) / 2); ax.set_title("Surface normals (RGB=xyz)",
                                               fontweight="bold")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3)
    im = ax.imshow(z, cmap="turbo", vmin=low, vmax=high)
    ax.set_title("Relative depth  (blue=near, red=far)", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8, label="relative depth (px units)")

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    H, W = z.shape
    step = max(1, W // 80)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    ax.plot_surface(xx, yy, -z[::step, ::step], cmap="turbo",
                    vmin=-high, vmax=-low, linewidth=0, antialiased=True)
    ax.set_title("3-D reconstruction", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("-depth")
    ax.view_init(elev=55, azim=-60)

    fig.suptitle(f"Location {loc} - photometric-stereo depth from 3 LEDs",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loc{loc:02d}_panel.png"),
                bbox_inches="tight", facecolor="#f5f5f0")
    plt.close(fig)


# =============================================================================
# Discovery + driver
# =============================================================================

def discover(folder):
    """Group image files by location. Returns {loc: {photo_idx: path}}."""
    groups = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"INPUT_FOLDER not found: {folder}")
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        name = os.path.basename(path)
        m = FILENAME_RE.match(name)
        if not m:
            if os.path.isfile(path):
                print(f"  [warn] ignoring unrecognised file: {name}")
            continue
        loc, idx = int(m.group(1)), int(m.group(2))
        groups.setdefault(loc, {})[idx] = path
    return groups


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("=" * 78)
    print(" BONE PICTURE - batch photometric-stereo depth")
    print("=" * 78)
    print(f" input : {os.path.abspath(INPUT_FOLDER)}")
    print(f" output: {os.path.abspath(OUTPUT_FOLDER)}")

    groups = discover(INPUT_FOLDER)
    if not groups:
        print("No pN-M images found - check INPUT_FOLDER.")
        return
    print(f" found {len(groups)} location(s): {sorted(groups)}\n")

    expected = set([ALL_ON_INDEX] if ALL_ON_INDEX else []) \
        | set(SINGLE_LED_AZIMUTH_DEG.keys())

    rows = []
    for loc in sorted(groups):
        files = groups[loc]
        have = set(files.keys())
        if not expected.issubset(have):
            print(f"  [loc {loc}] WARNING incomplete: have {sorted(have)}, "
                  f"expected {sorted(expected)}")
        stats = process_location(loc, files, OUTPUT_FOLDER)
        if stats:
            rows.append(stats)

    # CSV: one row per successfully processed location.
    if rows:
        csv_path = os.path.join(OUTPUT_FOLDER, "depth_summary.csv")
        fields = ["location", "width", "height", "n_lights", "azimuths_deg",
                  "depth_min", "depth_max", "depth_mean", "depth_std",
                  "relief_p2_p98"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows -> {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()


# =============================================================================
# NOTE - getting ABSOLUTE depth (mm) later, if you ever need it
# -----------------------------------------------------------------------------
# Photometric stereo alone gives shape up to scale. To pin the absolute scale
# you need one extra constraint, e.g. a coaxial point-light image (single LED
# next to the lens) where brightness I = albedo * power / r^2, so
#       r = sqrt(albedo * power / I)
# gives true distance at the brightest (nearest, head-on) pixels. Capture such
# a frame, recover r there, and shift+scale this relative z to match. That is
# exactly the calibrate_scale_with_coaxial() step in photometric_depth_test.py.
# =============================================================================
