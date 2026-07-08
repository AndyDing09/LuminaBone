"""
Self-contained endoscope depth estimation script.

Just put this file in your repo with a folder called pdf_images/ containing
the three p6-*.png images from the Wu et al. 2009 paper. Then run:
    python endoscope_depth.py

Outputs per image (in outputs/):
  - paper_endoscope_a.png   (6-panel figure: input, masks, depth, 3D plot)
  - paper_endoscope_a.ply   (colored point cloud for MeshLab/CloudCompare)
  - Same for views b and c
  - paper_endoscope_summary.png  (combined 3-view comparison)

Algorithm:
  Single-image depth from inverse-square + Lambertian shading.
  I = albedo * light / r^2  =>  r = sqrt(albedo * light / I)
  Then: specular masking + edge-aware smoothing.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =============================================================================
# Core algorithm (was in photometric_depth.py)
# =============================================================================

def to_luminance(rgb):
    """Weighted RGB -> single brightness channel (green-heavy for tissue)."""
    return 0.15 * rgb[..., 0] + 0.70 * rgb[..., 1] + 0.15 * rgb[..., 2]


def detect_specular_mask(rgb, brightness_thresh=0.85, sat_thresh=0.3):
    """Mask pixels that are both very bright AND low-saturation (glints)."""
    luminance = to_luminance(rgb)
    max_c = rgb.max(axis=-1)
    min_c = rgb.min(axis=-1)
    saturation = np.where(max_c > 1e-6, (max_c - min_c) / (max_c + 1e-6), 0.0)
    return (luminance > brightness_thresh) & (saturation < sat_thresh)


def inpaint_mask(image, mask, iterations=20):
    """Iterative neighbor-averaging fill of masked pixels."""
    filled = image.copy()
    filled[mask] = 0.0
    for _ in range(iterations):
        blurred = gaussian_filter(filled, sigma=2.0)
        filled[mask] = blurred[mask]
    return filled


def brightness_to_depth(luminance, albedo=0.5, light_power=0.3, eps=1e-3):
    """Inverse-square inversion: r = sqrt(rho * L / I)."""
    safe_I = np.maximum(luminance, eps)
    return np.sqrt(albedo * light_power / safe_I)


def bilateral_smooth(depth, guide, spatial_sigma=10.0, range_sigma=0.08,
                     iterations=4):
    """Edge-aware smoothing of depth using the luminance image as guide."""
    smoothed = depth.copy()
    for _ in range(iterations):
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_depth = np.roll(smoothed, shift, axis=(0, 1))
            shifted_guide = np.roll(guide, shift, axis=(0, 1))
            weight = np.exp(-((guide - shifted_guide) ** 2) /
                            (2 * range_sigma ** 2))
            smoothed = (1 - 0.25 * weight) * smoothed + \
                       (0.25 * weight) * shifted_depth
    return gaussian_filter(smoothed, sigma=spatial_sigma * 0.3)


# =============================================================================
# PLY export
# =============================================================================

def write_ply(filename, points_xyz, colors_rgb=None):
    """Write a colored point cloud to ASCII PLY format."""
    n = len(points_xyz)
    has_color = colors_rgb is not None
    header = ["ply", "format ascii 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    header.append("end_header")
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        if has_color:
            for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z) in points_xyz:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def depth_to_pointcloud(depth, rgb, mask, depth_scale=30.0):
    """Convert depth map -> colored 3D points. Close pixels get high Z."""
    H, W = depth.shape
    j, i = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = W / 2.0, H / 2.0
    X = (j - cx).astype(np.float32)
    Y = -(i - cy).astype(np.float32)
    Z = -depth.astype(np.float32) * depth_scale
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=-1)
    cols = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    cols = cols[mask]
    return pts, cols


# =============================================================================
# Main per-image processing
# =============================================================================

def print_depth_stats(label, depth, mask):
    valid = depth[mask]
    if len(valid) == 0:
        print(f"  [{label}] no valid pixels!")
        return
    print(f"  --- Depth statistics ({label}) ---")
    print(f"    valid pixels:    {valid.size:>8} of {depth.size} ({100*valid.size/depth.size:.1f}%)")
    print(f"    depth min:       {valid.min():>8.3f}  (closest to scope)")
    print(f"    depth max:       {valid.max():>8.3f}  (farthest in view)")
    print(f"    depth mean:      {valid.mean():>8.3f}")
    print(f"    depth std:       {valid.std():>8.3f}")
    print(f"    depth median:    {np.median(valid):>8.3f}")
    print(f"    range (max-min): {valid.max() - valid.min():>8.3f}")


def process_endoscope_image(path, out_png_path, out_ply_path, title):
    img = Image.open(path).convert('RGB')
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    print(f"\nLoaded {path}: shape {rgb.shape}")

    lum_raw = to_luminance(rgb)
    fov_mask = lum_raw > 0.05
    print(f"  FOV pixels: {fov_mask.sum()} of {fov_mask.size}")

    specular_mask = detect_specular_mask(rgb)
    print(f"  Specular highlights: {specular_mask.sum()} pixels")

    cleaned = rgb.copy()
    for c in range(3):
        cleaned[..., c] = inpaint_mask(rgb[..., c], specular_mask)

    lum = to_luminance(cleaned)
    depth_raw = brightness_to_depth(lum)
    depth = bilateral_smooth(depth_raw, lum)

    print_depth_stats(title, depth, fov_mask)

    pts, cols = depth_to_pointcloud(depth, rgb, fov_mask)
    write_ply(out_ply_path, pts, cols)
    print(f"  Wrote PLY: {out_ply_path}  ({len(pts)} points)")

    # 6-panel figure
    depth_disp = np.where(fov_mask, depth, np.nan)
    fig = plt.figure(figsize=(15, 9), dpi=130)
    fig.patch.set_facecolor('#f5f5f0')
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    ax.set_title('Real endoscope image\n(from Wu et al. 2009)',
                  fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb)
    overlay = np.zeros((*specular_mask.shape, 4))
    overlay[specular_mask] = [1, 0, 0, 0.5]
    ax.imshow(overlay)
    ax.set_title(f'Specular highlights\n({specular_mask.sum()} pixels)',
                  fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.clip(cleaned, 0, 1))
    ax.set_title('After inpainting', fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 0])
    vmin = np.nanpercentile(depth_disp, 5)
    vmax = np.nanpercentile(depth_disp, 95)
    im = ax.imshow(depth_disp, cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_title('Estimated depth\nBLUE = close, RED = far',
                  fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.75, label='relative depth')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(rgb * 0.55)
    ax.imshow(depth_disp, cmap='turbo', vmin=vmin, vmax=vmax, alpha=0.55)
    ax.set_title('Depth overlay on image',
                  fontsize=11, fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2], projection='3d')
    Hh, Ww = depth.shape
    step = max(1, Ww // 60)
    yy, xx = np.mgrid[0:Hh:step, 0:Ww:step]
    zz = depth[::step, ::step]
    mm = fov_mask[::step, ::step]
    zz_m = np.where(mm, zz, np.nan)
    ax.plot_surface(xx, yy, -zz_m, cmap='turbo',
                     vmin=-vmax, vmax=-vmin,
                     linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title('3D reconstruction', fontsize=11, fontweight='bold')
    ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)'); ax.set_zlabel('-depth')
    ax.view_init(elev=50, azim=-65)

    fig.suptitle(f'Photometric depth on REAL endoscope photo: {title}',
                  fontsize=13, y=1.0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=130, bbox_inches='tight', facecolor='#f5f5f0')
    plt.close()
    print(f"  Wrote PNG: {out_png_path}")
    return depth, fov_mask


# =============================================================================
# Driver
# =============================================================================

def main():
    out_dir = './outputs'
    os.makedirs(out_dir, exist_ok=True)

    candidates = [
        ('./pdf_images/p6-000.png', 'paper_endoscope_a',
         'view A (artificial spine, view 1)'),
        ('./pdf_images/p6-006.png', 'paper_endoscope_b',
         'view B (artificial spine, view 2)'),
        ('./pdf_images/p6-007.png', 'paper_endoscope_c',
         'view C (artificial spine, view 3)'),
    ]

    print("=" * 70)
    print(" PHOTOMETRIC DEPTH ESTIMATION ON REAL ENDOSCOPE PHOTOS")
    print(" Source: Wu, Narasimhan, Jaramaz (2009), IJCV - Figure 3")
    print("=" * 70)

    results = []
    for src, base, title in candidates:
        if not os.path.exists(src):
            print(f"\n!! Skipping {src}: file not found")
            continue
        png_path = f'{out_dir}/{base}.png'
        ply_path = f'{out_dir}/{base}.ply'
        depth, mask = process_endoscope_image(src, png_path, ply_path, title)
        results.append((src, base, depth, mask))

    if len(results) >= 3:
        print("\n" + "=" * 70)
        print(" Building combined summary figure...")
        print("=" * 70)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=130)
        fig.patch.set_facecolor('#f5f5f0')
        for col, (src, base, depth, mask) in enumerate(results[:3]):
            img = np.asarray(Image.open(src).convert('RGB'), dtype=np.float32) / 255.0
            axes[0, col].imshow(img)
            axes[0, col].set_title(f'Endoscope view {chr(65+col)}',
                                    fontsize=11, fontweight='bold')
            axes[0, col].axis('off')
            depth_disp = np.where(mask, depth, np.nan)
            vmin = np.nanpercentile(depth_disp, 5)
            vmax = np.nanpercentile(depth_disp, 95)
            im = axes[1, col].imshow(depth_disp, cmap='turbo',
                                      vmin=vmin, vmax=vmax)
            axes[1, col].set_title('Estimated depth',
                                    fontsize=11, fontweight='bold')
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], shrink=0.7, label='relative depth')
        fig.suptitle('Photometric depth estimation on REAL endoscope photos\n'
                      '(Wu et al. 2009, Figure 3 source images)',
                      fontsize=13, y=1.02, fontweight='bold')
        plt.tight_layout()
        summary = f'{out_dir}/paper_endoscope_summary.png'
        plt.savefig(summary, dpi=130, bbox_inches='tight', facecolor='#f5f5f0')
        plt.close()
        print(f"\n>>> Combined summary written to: {summary}\n")

    print("=" * 70)
    print(" Output files in", out_dir + ':')
    print("=" * 70)
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:<40s}  ({size_kb:>7.1f} KB)")
    print("\nPLY files can be viewed with MeshLab, CloudCompare, or Blender.\n")


if __name__ == '__main__':
    main()
