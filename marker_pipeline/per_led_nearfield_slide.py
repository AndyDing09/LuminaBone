"""
Per-LED NEAR-FIELD depth: invert the near-field attenuation model on EACH
single LED frame.

per_led_depth_slide.py does the naive thing -- treat brightness as depth,
r ~ 1/sqrt(I) -- which ignores that the LED sits 6.05 mm OFF the optical axis.
This script does it properly. For LED s at position P_s the model is

    I_s = k_s * cos^mu(theta_s) / r_s^2 ,    cos(theta_s) = D / r_s
    r_s = || P_s - X(D) || ,   X(D) = D * [x_n, -y_n, -1]

so for each pixel we solve that scalar equation for the depth D along its own
ray (bisection; the right side is monotonic in D). The result is depth from the
LENS in millimetres, correctly accounting for the off-axis source -- not a
brightness re-colouring.

Scale: k_s is the unknown LED power. It is fixed CT-FREE by requiring the
median depth over bone to equal the working distance b that the CT-free
near-field solver recovered for this shot (nearfield_ctfree.py). So all four
panels are in real mm and directly comparable.

Still a single-image estimate: with one light, surface tilt (n.l) and albedo
are inseparable from distance, so the maps disagree with each other. The
4-light near-field solve is what removes that ambiguity.

Run: python per_led_nearfield_slide.py [shot_004]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "reconstruction"))
import bone_depth_batch as bd
bd.UNDISTORT_INPUTS = False
import nearfield_ctfree as CF
import marker_depth_slides as S

MEASURED = {1: "lights BOTTOM", 2: "lights TOP",
            3: "lights LEFT", 4: "lights RIGHT"}
D_LO, D_HI = 5.0, 200.0


def att_one(D, led, mu, xn, yn):
    """near-field attenuation cos^mu / r^2 for ONE led, per pixel."""
    P = CF.P[led]
    Xw = np.stack([xn * D, -yn * D, -D], axis=-1)
    w = P[None, None, :] - Xw
    r2 = np.sum(w * w, axis=-1)
    r = np.sqrt(r2)
    a = 1.0 / r2
    if mu:
        a = a * np.clip(D / r, 0.0, None) ** mu
    return a


def invert_depth(I, led, mu, k, xn, yn, iters=48):
    """solve k * att(D) = I for D, per pixel, by bisection (att decreases in D)."""
    lo = np.full(I.shape, D_LO)
    hi = np.full(I.shape, D_HI)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pred = k * att_one(mid, led, mu, xn, yn)
        too_bright = pred > I          # predicted brighter => surface too near
        lo = np.where(too_bright, mid, lo)
        hi = np.where(too_bright, hi, mid)
    return 0.5 * (lo + hi)


def fit_k(I, led, mu, xn, yn, mask, b_target, iters=40):
    """choose the LED power k so the median recovered depth equals b_target."""
    klo, khi = 1e-6, 1e6
    for _ in range(iters):
        k = np.sqrt(klo * khi)
        D = invert_depth(I, led, mu, k, xn, yn)
        # larger k => brighter prediction => bisection pushes the surface
        # farther => larger D. So too-deep means k is too LARGE.
        if np.median(D[mask]) > b_target:
            khi = k
        else:
            klo = k
    return np.sqrt(klo * khi)


def main():
    shot = next((a for a in sys.argv[1:] if a.startswith("shot_")), "shot_004")
    d = bd.project_path("Data_collection", "calib_charuco", shot)
    dark = bd.load_rgb(os.path.join(d, "dark.png"), bd.WORK_LONG_EDGE)
    rgbs, lums = [], []
    for i in (1, 2, 3, 4):
        rgb = bd.load_rgb(os.path.join(d, f"led{i}.png"), bd.WORK_LONG_EDGE)
        rgbs.append(rgb)
        lums.append(bd.solve_luminance(np.clip(rgb - dark, 0, 1)))
    lums = bd.balance_exposure(lums)
    mask = bd.bone_mask(np.stack(lums, 0).mean(0))

    print(f"{shot}: running the CT-free near-field solve for the scale anchor ...")
    D4, _, _, keep, _, hist = CF.reconstruct(shot, mu_fixed=8.0, verbose=False)
    b, s_gain, mu, resid = hist[-1]
    print(f"  4-light solve: b = {b:.2f} mm, mu = {mu:.1f}")

    H, W = lums[0].shape
    xn, yn = CF.norm_grid(CF.FX, CF.FY, CF.CX, CF.CY, H, W)

    zs, spans = [], []
    for k_led in (1, 2, 3, 4):
        I = np.maximum(lums[k_led - 1], 0.05 * np.median(lums[k_led - 1][mask]))
        kpow = fit_k(I, k_led, mu, xn, yn, mask, b)
        D = invert_depth(I, k_led, mu, kpow, xn, yn)
        zs.append(np.where(mask, D, np.nan))
        v = D[mask]
        spans.append((np.percentile(v, 3), np.percentile(v, 97)))
        print(f"  led{k_led}: k = {kpow:.3g}   depth over bone "
              f"{spans[-1][0]:6.2f} .. {spans[-1][1]:6.2f} mm "
              f"(median {np.median(v):.2f})")

    stack = np.concatenate([z[mask] for z in zs])
    lo, hi = np.percentile(stack, [3, 97])

    markers = S.drop_other_segment(
        shot, S.find_markers({i: S.undistorted_bgr(os.path.join(d, f"led{i}.png"))
                              for i in S.ORDER}))

    fig, axs = plt.subplots(2, 4, figsize=(18, 8.6),
                            gridspec_kw=dict(height_ratios=[1.0, 1.35]))
    fig.patch.set_facecolor("#f5f5f0")
    for c in range(4):
        axs[0, c].imshow(rgbs[c])
        axs[0, c].set_title(f"led{c+1}.png   ({MEASURED[c+1]})",
                            fontsize=11, fontweight="bold")
        axs[0, c].axis("off")

        im = axs[1, c].imshow(zs[c], cmap="turbo_r", vmin=lo, vmax=hi)
        vals = []
        for kk, (x, y, r, _n) in enumerate(markers, 1):
            axs[1, c].add_patch(plt.Circle((x, y), max(r, 8) + 3, fill=False,
                                           edgecolor="white", linewidth=1.6))
            axs[1, c].plot(x, y, "+", color="white", ms=6, mew=1.3)
            axs[1, c].annotate(str(kk), (x, y), (x + max(r, 8) + 5, y),
                               color="white", fontsize=9, fontweight="bold",
                               ha="left", va="center",
                               bbox=dict(boxstyle="circle,pad=0.14", fc="black",
                                         alpha=0.6, ec="white", lw=0.7))
            vals.append(float(np.nanmedian(
                zs[c][max(0, int(y) - 6):int(y) + 7,
                      max(0, int(x) - 6):int(x) + 7])))
        axs[1, c].set_title("near-field depth from this LED alone (mm)\n" +
                            "  ".join(f"{i}:{v:.0f}"
                                      for i, v in enumerate(vals, 1)),
                            fontsize=10)
        axs[1, c].axis("off")

    cb = fig.colorbar(im, ax=axs[1, :].tolist(), shrink=0.85, pad=0.012)
    cb.set_label("depth from lens (mm), blue = near", fontsize=10)
    fig.suptitle(f"{shot} - NEAR-FIELD single-LED depth "
                 f"(inverting cos$^\\mu$/r$^2$ from each frame separately)",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             f"Each pixel's depth solves I = k cos^mu(theta)/r^2 along its own "
             f"ray for an LED at 6.05 mm off-axis, so the off-axis geometry is "
             f"handled properly - unlike r ~ 1/sqrt(I). Power k is set CT-free "
             f"so each median matches the working distance b = {b:.1f} mm that "
             f"the 4-light CT-free solve recovered. The panels still disagree: "
             f"one light cannot separate distance from surface tilt or albedo. "
             f"Using all four together is what resolves it (leave-one-out "
             f"5.1 mm, vs 12.6 mm for single-image depth).",
             ha="center", fontsize=8.5, style="italic", color="#555555",
             wrap=True)
    out = bd.project_path("depth_outputs", "calib_charuco_slides",
                          f"{shot}_per_led_nearfield_depth.png")
    fig.savefig(out, dpi=115, bbox_inches="tight", facecolor="#f5f5f0")
    print("wrote", out)


if __name__ == "__main__":
    main()
