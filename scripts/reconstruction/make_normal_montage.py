"""
Classic photometric-stereo teaching figure, built from one triangulation slide.

Recreates the well-known montage (several single-light images -> arrows -> one
RGB surface-normal map, with a normal-encoding reference sphere) but using the
THREE single-LED bone photos that make up one of our triangulation slides:

    p{loc}-2  RIGHT LED   ->  top-left  input
    p{loc}-3  TOP   LED   ->  top-mid   input
    p{loc}-4  LEFT  LED   ->  top-right input
                     |  |  |
                     v  v  v
            RGB surface-normal map  (R,G,B = nx,ny,nz)   + reference sphere

The normals come from exactly the same Woodham solve + measured light geometry
(RIGHT/TOP/LEFT at azimuth 0/90/180, one shared ~75 deg elevation from the
12.08 mm LED offset) that make_triangulation_slides.py already uses, so this
figure is consistent with the depth/3-D slides for the same location.

Run:
    py make_normal_montage.py            # default location (1)
    py make_normal_montage.py 7          # location 7
    py make_normal_montage.py 7 --color  # show input photos in colour, not grey
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import bone_depth_batch as bd
import nearfield_lambertian as nf

OUT_DIR = bd.project_path("depth_outputs", "triangulation_slides")

LED_NAME = {2: "RIGHT", 3: "TOP", 4: "LEFT"}
ORDER = [2, 3, 4]                                   # left-to-right on the figure

# Same measured, symmetric, equal-LED geometry as the triangulation slides.
LED_AZIMUTH_DEG = {2: 0.0, 3: 90.0, 4: 180.0}
WORKING_DISTANCE_MM = nf.working_distance_mm()
LIGHT_ELEVATION_DEG = math.degrees(math.atan2(WORKING_DISTANCE_MM,
                                              nf.LED_OFFSET_MM))


def inpaint_normals(normals, holes, iterations=40):
    """Fill `holes` in a normal map by repeated neighbour averaging + renorm.

    Same idea as bone_depth_batch.inpaint_specular, but on the 3-vector field:
    diffuse the trusted normals into the untrusted region so ink marks / glints
    / background don't punch flat-colour blobs through the map.
    """
    if not holes.any():
        return normals
    from scipy.ndimage import gaussian_filter
    out = normals.copy()
    out[holes] = 0.0
    for _ in range(iterations):
        for c in range(3):
            blur = gaussian_filter(out[..., c], sigma=2.0)
            out[..., c][holes] = blur[holes]
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.where(norm > 1e-6, norm, 1.0)
    return out


def normal_reference_sphere(size=220):
    """Synthetic hemisphere coloured by the SAME (n+1)/2 encoding as the map.

    Pixels outside the unit disk are returned transparent so the sphere floats
    on the figure background, exactly like the little reference ball in the
    classic figure. Row 0 is the TOP of the sphere, so +y (green) points up to
    match how the normal map is displayed with imshow.
    """
    ys, xs = np.mgrid[0:size, 0:size]
    x = (xs - (size - 1) / 2) / ((size - 1) / 2)       # -1 .. +1, +x right
    y = ((size - 1) / 2 - ys) / ((size - 1) / 2)       # -1 .. +1, +y up
    r2 = x * x + y * y
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    n = np.stack([x, y, z], axis=-1)
    rgb = (n + 1.0) / 2.0
    rgba = np.zeros((size, size, 4), dtype=np.float64)
    rgba[..., :3] = rgb
    rgba[..., 3] = inside.astype(np.float64)           # alpha = 1 on the disk
    return rgba


def build_montage(loc, files, color_inputs=False):
    # --- load the three inputs + their light directions ---
    rgbs, lums, Ls = [], [], []
    for i in ORDER:
        rgb = bd.load_rgb(files[i], bd.WORK_LONG_EDGE)
        Ls.append(bd.light_vector(LED_AZIMUTH_DEG[i], LIGHT_ELEVATION_DEG))
        rgbs.append(rgb)
        lums.append(bd.solve_luminance(rgb))
    lums = bd.balance_exposure(lums)

    # --- the method: 3 images -> per-pixel surface normals ---
    normals, albedo = bd.photometric_stereo(lums, np.array(Ls))

    # Untrusted pixels (dark background, ink marks, specular pits) carry no
    # reliable normal. Rather than leave flat-colour blobs, fill them from
    # trusted neighbours so the map reads clean like the classic figure.
    mask = bd.bone_mask(albedo)
    if mask.any():
        normals = inpaint_normals(normals, ~mask)
    normal_rgb = (normals + 1.0) / 2.0                 # R,G,B <- nx,ny,nz

    # Grey display of the shading each LED produced (what the solve actually
    # sees), rendered like the classic figure; --color keeps the raw photos.
    def input_display(k):
        if color_inputs:
            return rgbs[k], None
        g = bd.to_luminance(rgbs[k])
        g = np.clip(g / max(np.percentile(g, 99), 1e-6), 0, 1)   # gentle norm
        return g, "gray"

    # =========================== draw ===========================
    fig = plt.figure(figsize=(12.5, 11.0), dpi=130)
    fig.patch.set_facecolor("#0f0f12")                 # dark like the reference

    # Top row: the three single-LED inputs.
    top_axes = []
    top_positions = [0.055, 0.375, 0.695]              # left edges (fig coords)
    top_w, top_h, top_y = 0.25, 0.30, 0.63
    for k, i in enumerate(ORDER):
        ax = fig.add_axes([top_positions[k], top_y, top_w, top_h])
        disp, cmap = input_display(k)
        ax.imshow(disp, cmap=cmap)
        L = Ls[k]
        ax.set_title(f"p{loc}-{i}.jpg   {LED_NAME[i]} LED\n"
                     f"L = ({L[0]:+.2f}, {L[1]:+.2f}, {L[2]:+.2f})",
                     fontsize=10.5, color="#f0f0f0", fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#f0f0f0"); s.set_linewidth(1.2)
        top_axes.append(ax)

    # Centre: the RGB surface-normal map (the "result").
    map_w, map_h = 0.40, 0.36
    map_x = 0.5 - map_w / 2
    map_y = 0.10
    axm = fig.add_axes([map_x, map_y, map_w, map_h])
    axm.imshow(normal_rgb)
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_color("#f0f0f0"); s.set_linewidth(1.4)
    axm.set_title("Surface normal map   (R, G, B  =  n$_x$, n$_y$, n$_z$)",
                  fontsize=12.5, color="#ffffff", fontweight="bold", pad=8)

    # Reference sphere, tucked at the lower-right of the map like the original.
    sph_size = 0.11
    axs = fig.add_axes([map_x + map_w - sph_size + 0.005,
                        map_y - sph_size * 0.55, sph_size, sph_size])
    axs.imshow(normal_reference_sphere())
    axs.axis("off")

    # Converging arrows from each input down to the top of the normal map.
    map_top_center = (0.5, map_y + map_h)
    for ax in top_axes:
        bbox = ax.get_position()
        start = (bbox.x0 + bbox.width / 2, bbox.y0 - 0.008)
        arrow = FancyArrowPatch(start, map_top_center,
                                transform=fig.transFigure,
                                arrowstyle="-|>", mutation_scale=22,
                                lw=2.2, color="#e8e8e8",
                                shrinkA=2, shrinkB=10)
        fig.patches.append(arrow)

    fig.text(0.5, 0.975,
             f"Location {loc}  —  three-image photometric stereo  →  surface normals",
             ha="center", va="top", fontsize=15, color="#ffffff",
             fontweight="bold")
    fig.text(0.5, 0.032,
             f"Woodham solve on RIGHT+TOP+LEFT LEDs (azimuth 0/90/180°, "
             f"{LIGHT_ELEVATION_DEG:.0f}° elevation from the {nf.LED_OFFSET_MM:.2f} mm "
             f"LED offset). Reference sphere: same n→RGB colour key.",
             ha="center", va="bottom", fontsize=8.5, style="italic",
             color="#9a9aa2")

    suffix = "_color" if color_inputs else ""
    out = os.path.join(OUT_DIR, f"loc{loc:02d}_normal_montage{suffix}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    color_inputs = "--color" in args
    args = [a for a in args if not a.startswith("--")]
    loc = int(args[0]) if args else 1

    groups = bd.discover(bd.INPUT_FOLDER)
    if loc not in groups:
        raise SystemExit(f"location {loc} not found; have {sorted(groups)}")
    files = groups[loc]
    missing = [i for i in ORDER if i not in files]
    if missing:
        raise SystemExit(f"location {loc} missing p{loc}-{missing} single-LED photo(s)")

    out = build_montage(loc, files, color_inputs=color_inputs)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
