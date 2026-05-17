"""
Photometric stereo on a sphere lit from three directions.

Pipeline:
  1. Synthesize 3 images of the same sphere, each lit from a different
     known direction (the camera stays put).
  2. For each pixel, solve a 3x3 linear system to recover the surface
     normal and albedo (Woodham 1980).
  3. Integrate the normals into a depth map (Frankot-Chellappa Poisson
     integration in the frequency domain).
  4. Use a single coaxial-light image (inverse-square falloff) to fix
     the absolute scale, so the depth comes out in real centimeters.
  5. Display everything as heatmaps with numerical annotations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 1. Generate a sphere illuminated from three different directions
# ---------------------------------------------------------------------------

def make_sphere_scene(size=300, radius_cm=1.5, center_dist_cm=4.0,
                     albedo=0.7, light_dirs=None):
    """Return (rgb_images, true_depth_cm, true_normals, mask).

    The sphere sits in front of the camera, center at depth `center_dist_cm`.
    Pixels outside the sphere's silhouette are background (mask=False).

    `light_dirs` is a list of 3-vectors (in camera coords; +z toward camera,
    +x right, +y up). They get unit-normalized.
    """
    if light_dirs is None:
        # Three lights from upper-left, upper-right, and below — a classic
        # photometric-stereo configuration.
        light_dirs = [
            np.array([-1.0,  1.0, 1.5]),
            np.array([ 1.0,  1.0, 1.5]),
            np.array([ 0.0, -1.0, 1.5]),
        ]
    light_dirs = [L / np.linalg.norm(L) for L in light_dirs]

    H = W = size
    # Project image coords -> world coords on the sphere surface.
    # Treat the sphere as if it spans roughly 60% of the image width.
    cx, cy = W / 2, H / 2
    # cm per pixel chosen so the sphere of radius 1.5 cm fits comfortably
    pixels_per_cm = (W * 0.3) / radius_cm
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    X = (x - cx) / pixels_per_cm  # cm
    Y = -(y - cy) / pixels_per_cm  # cm (flip y so up is +)

    # The visible front surface of the sphere: z_surface = z_center - sqrt(r^2 - X^2 - Y^2)
    inside = X**2 + Y**2 <= radius_cm**2
    Z = np.zeros_like(X)
    Z[inside] = center_dist_cm - np.sqrt(radius_cm**2 - X[inside]**2 - Y[inside]**2)

    # Surface normals: outward from sphere center (which is at depth center_dist_cm)
    nx = X / radius_cm
    ny = Y / radius_cm
    # The visible surface is the front hemisphere, so nz points toward camera (positive)
    nz_sq = 1 - nx**2 - ny**2
    nz = np.where(inside, np.sqrt(np.maximum(nz_sq, 0)), 0)
    normals = np.stack([nx, ny, nz], axis=-1)
    normals[~inside] = 0

    # Render each lit view: I = albedo * max(n . L, 0)
    images = []
    for L in light_dirs:
        shading = np.clip(normals @ L, 0, None)
        img_gray = albedo * shading
        # Add tiny noise to make it realistic
        rng = np.random.default_rng(42 + len(images))
        img_gray = img_gray + 0.005 * rng.standard_normal(img_gray.shape)
        img_gray = np.clip(img_gray, 0, 1)
        # Convert to RGB (tissue-like reddish tint)
        rgb = np.stack([
            img_gray,
            img_gray * 0.55,
            img_gray * 0.55,
        ], axis=-1)
        # Background dark
        rgb[~inside] = 0.02
        images.append(rgb.astype(np.float32))

    return images, Z, normals, inside, np.array(light_dirs)


# ---------------------------------------------------------------------------
# 2. Solve for normals at every pixel (Woodham photometric stereo)
# ---------------------------------------------------------------------------

def solve_photometric_stereo(images, light_dirs, mask):
    """Recover per-pixel surface normals and albedo.

    For each pixel p, we have a brightness vector I(p) of length 3
    (one entry per light). Solve L @ g(p) = I(p) for the 3-vector g.
    Then albedo = |g|, normal = g / |g|.
    """
    # Convert RGB to luminance for each of the 3 images
    luminances = []
    for img in images:
        lum = 0.15 * img[..., 0] + 0.70 * img[..., 1] + 0.15 * img[..., 2]
        luminances.append(lum)

    H, W = luminances[0].shape
    # Stack into a (3, H*W) matrix
    I_stack = np.stack([l.ravel() for l in luminances], axis=0)  # (3, N)

    # L is (3, 3) — solve once for all pixels
    L = light_dirs  # (3, 3) with each row a light direction
    L_inv = np.linalg.inv(L)
    G = L_inv @ I_stack  # (3, N)

    g = G.T.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=-1)
    # Avoid /0 for background pixels
    safe_alb = np.where(albedo > 1e-4, albedo, 1.0)
    normals = g / safe_alb[..., None]
    # Background: clobber to (0, 0, 1) so integration doesn't blow up
    normals[~mask] = [0, 0, 1]
    albedo[~mask] = 0
    return normals, albedo


# ---------------------------------------------------------------------------
# 3. Integrate the normals into a depth map (Frankot-Chellappa)
# ---------------------------------------------------------------------------

def integrate_normals_to_depth(normals, mask, cm_per_pixel=1.0):
    """Recover depth z(x, y) from surface normals on the masked region.

    The classic gradient relation is:
        nx * dz/dx + ny * dz/dy + nz = 0   (for unit normals)
    Equivalently:
        dz/dx = -nx / nz,   dz/dy = -ny / nz.

    Naively writing one equation per pixel as (z[y, x+1] - z[y, x]) = -nx/nz
    blows up near the silhouette where nz ~ 0. We avoid that by writing
    each equation in its UNDIVIDED form, weighted by nz:

        nz * (z[y, x+1] - z[y, x]) = -nx * cm_per_pixel
        nz * (z[y+1, x] - z[y, x]) = -ny * cm_per_pixel

    Silhouette pixels (nz ~ 0) effectively contribute nothing, exactly
    matching how reliable they are. This is standard for closed-surface
    photometric stereo.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = normals[..., 2]

    H, W = nx.shape
    idx = -np.ones((H, W), dtype=np.int64)
    masked_pixels = np.where(mask)
    n_unknown = len(masked_pixels[0])
    idx[mask] = np.arange(n_unknown)

    rows, cols, data, rhs = [], [], [], []
    eq = 0
    for i in range(n_unknown):
        y, x = masked_pixels[0][i], masked_pixels[1][i]
        w = nz[y, x]  # weight: how reliable this pixel's slope is
        if x + 1 < W and mask[y, x + 1]:
            # nz>0 toward camera, z increases AWAY from camera, so
            # in image space:  dz/dx_img = +nx/nz
            rows.append(eq); cols.append(idx[y, x + 1]); data.append(w)
            rows.append(eq); cols.append(idx[y, x]);     data.append(-w)
            rhs.append(nx[y, x] * cm_per_pixel); eq += 1
        if y + 1 < H and mask[y + 1, x]:
            # Image y increases downward, world Y increases upward, so
            # dz/dy_img = -ny/nz  (sign flip on the y component)
            rows.append(eq); cols.append(idx[y + 1, x]); data.append(w)
            rows.append(eq); cols.append(idx[y, x]);     data.append(-w)
            rhs.append(-ny[y, x] * cm_per_pixel); eq += 1
    # Anchor first masked pixel to 0
    rows.append(eq); cols.append(0); data.append(1.0); rhs.append(0.0); eq += 1

    A = csr_matrix((data, (rows, cols)), shape=(eq, n_unknown))
    z_vec = lsqr(A, np.array(rhs))[0]

    z = np.zeros((H, W))
    z[mask] = z_vec
    return z


# ---------------------------------------------------------------------------
# 4. Use coaxial inverse-square light to anchor the absolute scale
# ---------------------------------------------------------------------------

def calibrate_scale_with_coaxial(z_relative, mask, coaxial_image,
                                 albedo, light_power_cm2):
    """Fix the absolute scale of `z_relative` using a coaxial-light image.

    The photometric-stereo output is up to a constant offset (and possibly
    a sign flip). The coaxial image gives us I = rho * L * cos(theta) / r^2.
    We robustly pick the BRIGHTEST FEW pixels — they are the nearest points,
    where cos(theta) ≈ 1, so r = sqrt(rho * L / I) is most reliable.

    We use the red channel directly because, in our renderer, R preserves
    the raw intensity (tissue tint only affects G and B).
    """
    coax_r = coaxial_image[..., 0]
    masked = np.where(mask, coax_r, -np.inf)
    # Take the top 0.1% brightest pixels and average their inferred r
    flat = masked.ravel()
    thresh = np.percentile(flat[np.isfinite(flat)], 99.9)
    bright_idx = np.where(masked >= thresh)
    I_bright = coax_r[bright_idx].mean()
    r_near = np.sqrt(albedo * light_power_cm2 / max(I_bright, 1e-3))

    # Ensure near pixels have smaller z. Find what photometric-stereo says
    # the depth is at those bright pixels.
    z_at_near = z_relative[bright_idx].mean()
    z_elsewhere = z_relative[mask].mean()
    if z_at_near > z_elsewhere:
        z_relative = -z_relative
        z_at_near = -z_at_near

    offset = r_near - z_at_near
    z_absolute = z_relative + offset
    return z_absolute


def render_coaxial(normals, mask, albedo_map, light_power_cm2,
                   z_true, noise=0.005):
    """Synthesize what the same sphere looks like with a coaxial light.

    We use the *true* geometry here because this is just generating a
    test image for the calibration step — in real life this image would
    come from the actual scope.
    """
    # I = rho * L * cos(theta) / r^2, with cos(theta) = nz (light along -z)
    nz = normals[..., 2]
    I = albedo_map * light_power_cm2 * np.maximum(nz, 0) / np.maximum(z_true, 0.1)**2
    I = np.clip(I, 0, 1)
    rng = np.random.default_rng(7)
    I = I + noise * rng.standard_normal(I.shape)
    I = np.clip(I, 0, 1)
    I[~mask] = 0.02
    rgb = np.stack([I, I * 0.55, I * 0.55], axis=-1)
    return rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Export depth map as a PLY point cloud
# ---------------------------------------------------------------------------

def save_depth_as_ply(z_absolute, mask, normals, color_image, cm_per_pixel,
                      output_path):
    """Write a PLY point cloud from a calibrated depth map.

    Each valid (masked) pixel becomes a 3-D point:
        x_world = (col - image_center_x) * cm_per_pixel
        y_world = -(row - image_center_y) * cm_per_pixel   (flip so +y is up)
        z_world = z_absolute[row, col]

    Per-vertex normals and RGB color (from color_image) are included.
    """
    H, W = z_absolute.shape
    cx, cy = W / 2.0, H / 2.0

    rows, cols = np.where(mask)
    xs = (cols - cx) * cm_per_pixel
    ys = -(rows - cy) * cm_per_pixel
    zs = z_absolute[rows, cols]

    nx = normals[rows, cols, 0]
    ny = normals[rows, cols, 1]
    nz = normals[rows, cols, 2]

    rgb = (np.clip(color_image[rows, cols], 0, 1) * 255).astype(np.uint8)

    num_points = len(xs)
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(num_points):
            f.write(
                f"{xs[i]:.6f} {ys[i]:.6f} {zs[i]:.6f} "
                f"{nx[i]:.6f} {ny[i]:.6f} {nz[i]:.6f} "
                f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n"
            )
    print(f"Saved {num_points} points to {output_path}")


# ---------------------------------------------------------------------------
# 6. Visualize: heatmaps with labels
# ---------------------------------------------------------------------------

def labeled(ax, x, y, text, color='white'):
    t = ax.text(x, y, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
    t.set_path_effects([path_effects.withStroke(linewidth=2.5,
                                                foreground='black')])
    ax.plot(x, y, 'o', markersize=4, markerfacecolor='white',
            markeredgecolor='black', markeredgewidth=1)


def annotate_sphere(ax, depth_cm, mask):
    """Drop depth labels at sample points around and on the sphere."""
    H, W = depth_cm.shape
    # Find sphere bounding box
    ys, xs = np.where(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    radius_px = int((xs.max() - xs.min()) / 2 * 0.7)
    # 5 points: center + 4 around it
    points = [(cy, cx)]
    for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
        py = int(cy + radius_px * np.sin(ang))
        px = int(cx + radius_px * np.cos(ang))
        points.append((py, px))
    for (y, x) in points:
        if 0 <= y < H and 0 <= x < W and mask[y, x]:
            d = depth_cm[max(0,y-2):y+3, max(0,x-2):x+3][mask[max(0,y-2):y+3, max(0,x-2):x+3]].mean()
            labeled(ax, x, y, f'{d:.2f} cm')


def main():
    out_dir = './outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ---- Generate the scene ----
    print("Generating sphere with 3 different light directions...")
    images, true_depth, true_normals, mask, light_dirs = make_sphere_scene(
        size=300, radius_cm=1.5, center_dist_cm=4.0, albedo=0.7)

    # ---- Show the three input images ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=120)
    fig.patch.set_facecolor('#f5f5f0')
    light_labels = ['Light 1: upper-left', 'Light 2: upper-right', 'Light 3: below']
    for ax, img, L, lab in zip(axes, images, light_dirs, light_labels):
        ax.imshow(img)
        ax.set_title(f'{lab}\nL = ({L[0]:+.2f}, {L[1]:+.2f}, {L[2]:+.2f})',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    fig.suptitle('Step 1: Three images, same sphere, three known light directions',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ps_01_inputs.png', dpi=120,
                bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()

    # ---- Solve photometric stereo ----
    print("Solving photometric stereo (3x3 linear system per pixel)...")
    normals_est, albedo_est = solve_photometric_stereo(images, light_dirs, mask)

    # Show recovered normals as an RGB visualization
    # Map normal components to colors: nx -> R, ny -> G, nz -> B
    normals_viz = (normals_est + 1) / 2
    normals_viz[~mask] = 0
    true_normals_viz = (true_normals + 1) / 2
    true_normals_viz[~mask] = 0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=120)
    fig.patch.set_facecolor('#f5f5f0')
    axes[0].imshow(true_normals_viz)
    axes[0].set_title('Ground truth normals', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(normals_viz)
    axes[1].set_title('Recovered normals (from 3 images)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    im = axes[2].imshow(albedo_est, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Recovered albedo', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], shrink=0.7)
    fig.suptitle('Step 2: Per-pixel surface normals from L\u207b\u00b9 I',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ps_02_normals.png', dpi=120,
                bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()

    # ---- Integrate normals into depth ----
    print("Integrating normals into depth...")
    # The scene uses (W * 0.3) / radius_cm pixels per cm
    W = 300
    radius_cm = 1.5
    cm_per_pixel = radius_cm / (W * 0.3)
    z_relative = integrate_normals_to_depth(normals_est, mask, cm_per_pixel=cm_per_pixel)

    # ---- Use coaxial image to fix the absolute scale ----
    print("Calibrating scale using a coaxial-light reference image...")
    coaxial = render_coaxial(true_normals, mask, np.full_like(true_depth, 0.7),
                              light_power_cm2=2.0, z_true=true_depth)
    z_absolute = calibrate_scale_with_coaxial(
        z_relative, mask, coaxial, albedo=0.7, light_power_cm2=2.0)

    # Mask off background for display
    z_display = np.where(mask, z_absolute, np.nan)
    z_true_display = np.where(mask, true_depth, np.nan)

    # ---- The main heatmap ----
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    fig.patch.set_facecolor('#f5f5f0')
    vmin = np.nanpercentile(z_true_display, 2)
    vmax = np.nanpercentile(z_true_display, 98)
    im = ax.imshow(z_display, cmap='turbo', vmin=vmin, vmax=vmax)
    annotate_sphere(ax, z_absolute, mask)
    ax.set_title('Depth heatmap — sphere reconstructed from 3 lit images',
                 fontsize=13, fontweight='bold', pad=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('depth (cm)  —  distance from camera',
                   fontsize=11, fontweight='bold')
    ax.text(0.02, -0.08,
            'BLUE = closest point (front of sphere)   ·   RED = farthest visible point (edges)',
            transform=ax.transAxes, fontsize=9, style='italic')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ps_03_depth_heatmap.png', dpi=120,
                bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()

    # ---- Side-by-side comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=120)
    fig.patch.set_facecolor('#f5f5f0')
    im0 = axes[0].imshow(z_display, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0].set_title('Estimated depth (cm)', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], shrink=0.75, label='cm')
    im1 = axes[1].imshow(z_true_display, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground truth depth (cm)', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], shrink=0.75, label='cm')
    err_cm = np.abs(z_absolute[mask] - true_depth[mask]).mean()
    pct = 100 * err_cm / true_depth[mask].mean()
    fig.suptitle(f'Mean absolute error: {err_cm:.3f} cm  ({pct:.2f}% of mean depth)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ps_04_comparison.png', dpi=120,
                bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()

    # ---- 3D surface view ----
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10, 6), dpi=120)
    fig.patch.set_facecolor('#f5f5f0')
    ax = fig.add_subplot(111, projection='3d')
    H, W = z_absolute.shape
    step = 6
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    zz = z_absolute[::step, ::step]
    mm = mask[::step, ::step]
    zz_m = np.where(mm, zz, np.nan)
    surf = ax.plot_surface(xx, yy, -zz_m, cmap='turbo',
                           vmin=-vmax, vmax=-vmin,
                           linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_zlabel('-depth (cm)  (toward camera = up)')
    ax.set_title('Reconstructed 3D surface of the sphere',
                 fontsize=12, fontweight='bold')
    plt.colorbar(surf, ax=ax, shrink=0.6, label='-depth (cm)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/ps_05_3d_surface.png', dpi=120,
                bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()

    # ---- Export PLY point cloud ----
    color_for_ply = images[1]  # front-lit image for vertex color
    ply_path = f'{out_dir}/depth_map.ply'
    save_depth_as_ply(z_absolute, mask, normals_est, color_for_ply,
                      cm_per_pixel, ply_path)

    # ---- Visualize PLY in interactive 3D viewer ----
    import pyvista as pv
    cloud = pv.read(ply_path)
    cloud.plot(eye_dome_lighting=True)

    print(f"\nMean depth error: {err_cm:.3f} cm ({pct:.2f}%)")
    print(f"True depth at center: {true_depth[mask].min():.3f} cm (nearest point)")
    print(f"True depth at edge:   {true_depth[mask].max():.3f} cm (silhouette)")
    print(f"\nAll figures written to {out_dir}/")


if __name__ == '__main__':
    main()