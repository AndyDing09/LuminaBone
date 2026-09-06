"""
Presentation slide for the camera-calibration step.

Builds one figure, in the same light style as the triangulation slides:

    [ Raw (lens-distorted) ]  --->  [ Undistorted (corrected) ]
    -------------------------------------------------------------
    The lens fingerprint (fx, fy, cx, cy)  |  Lens distortion + quality

The "after" panel is undistorted with alpha=1 so the barrel correction shows as
the characteristic curved black border - the visual proof the lens warp was
removed. Numbers are read straight from calibration/calibration.txt.

Run:
    py make_calibration_slide.py
    py make_calibration_slide.py --image <path-to-a-checkerboard-or-bone-photo>
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


CALIB_TXT = project_path("calibration", "calibration.txt")
OUT = project_path("depth_outputs", "triangulation_slides", "camera_calibration_slide.png")


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
    return data, K, dist


def pick_demo_image(explicit):
    if explicit:
        return explicit
    # Prefer a 1280x720 checkerboard (straight lines make the barrel obvious).
    cands = sorted(glob.glob(project_path("calibration", "calib_images_1280x720", "*.jpg")))
    if cands:
        return cands[len(cands) // 2]           # a middle one, board usually centred
    raise SystemExit("no calibration images found; pass --image <path>")


def main():
    args = sys.argv[1:]
    explicit = None
    if "--image" in args:
        explicit = args[args.index("--image") + 1]

    data, K, dist = load_calibration(CALIB_TXT)
    img_path = pick_demo_image(explicit)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise SystemExit(f"could not read {img_path}")
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # alpha=1 keeps the whole field of view, so the corrected image shows the
    # curved black border that proves distortion was pulled out.
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(bgr, K, dist, None, new_K)
    undist_rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    d = dist.ravel()
    err = data["reprojection_error"]
    n_imgs = data.get("num_images_used", "?")
    iw, ih = data["image_size"]

    # ---------------- draw ----------------
    fig = plt.figure(figsize=(14, 9.6), dpi=115)
    fig.patch.set_facecolor("#f5f5f0")

    # Top row: before / after.
    axL = fig.add_axes([0.045, 0.46, 0.40, 0.40])
    axL.imshow(rgb); axL.set_xticks([]); axL.set_yticks([])
    axL.set_title("Raw photo  (lens-distorted)", fontsize=12, fontweight="bold")
    for s in axL.spines.values():
        s.set_color("#c0392b"); s.set_linewidth(2)

    axR = fig.add_axes([0.555, 0.46, 0.40, 0.40])
    axR.imshow(undist_rgb); axR.set_xticks([]); axR.set_yticks([])
    axR.set_title("Undistorted  (straightened by the calibration)",
                  fontsize=12, fontweight="bold")
    for s in axR.spines.values():
        s.set_color("#27ae60"); s.set_linewidth(2)

    arrow = FancyArrowPatch((0.455, 0.66), (0.55, 0.66),
                            transform=fig.transFigure, arrowstyle="-|>",
                            mutation_scale=34, lw=3, color="#333333")
    fig.patches.append(arrow)
    fig.text(0.503, 0.70, "undistort", ha="center", fontsize=10,
             style="italic", color="#333333")

    # Bottom-left: the camera matrix K, shown as the real 3x3 from calibration.txt.
    axtl = fig.add_axes([0.045, 0.04, 0.44, 0.35]); axtl.axis("off")
    axtl.set_title("Camera matrix  K   (from calibration.txt)",
                   fontsize=12.5, fontweight="bold", loc="left")
    Kmatrix = (
        f"      | {fx:9.2f}   {0.0:7.2f}   {cx:7.2f} |\n"
        f" K =  | {0.0:9.2f}   {fy:7.2f}   {cy:7.2f} |\n"
        f"      | {0.0:9.2f}   {0.0:7.2f}   {1.0:7.2f} |\n"
    )
    axtl.text(0.0, 0.88, Kmatrix, va="top", ha="left", family="monospace",
              fontsize=11.5, color="#1a5276", transform=axtl.transAxes)
    klabels = (
        f"fx, fy = {fx:.0f}, {fy:.0f} px   how zoomed-in (across / down)\n"
        f"cx, cy = {cx:.0f}, {cy:.0f} px   true optical centre\n"
    )
    axtl.text(0.0, 0.36, klabels, va="top", ha="left", family="monospace",
              fontsize=10.5, color="#222222", transform=axtl.transAxes)
    axtl.text(0.0, 0.06, "Maps a real 3-D point onto the correct pixel.",
              va="top", ha="left", fontsize=9.5, style="italic",
              color="#555555", transform=axtl.transAxes)

    # Bottom-right: the distortion vector, shown exactly as stored, + quality.
    axtr = fig.add_axes([0.515, 0.04, 0.44, 0.35]); axtr.axis("off")
    axtr.set_title("Distortion coefficients  +  quality",
                   fontsize=12.5, fontweight="bold", loc="left")
    distvec = (
        f"dist = [ {d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f},\n"
        f"         {d[3]:+.4f}, {d[4]:+.4f} ]\n"
    )
    axtr.text(0.0, 0.88, distvec, va="top", ha="left", family="monospace",
              fontsize=11.0, color="#1a5276", transform=axtr.transAxes)
    axtr.text(0.0, 0.56,
              "order (k1, k2, p1, p2, k3):\n"
              "k1, k2, k3 = radial 'barrel' warp    p1, p2 = lens tilt",
              va="top", ha="left", family="monospace", fontsize=9.5,
              color="#222222", transform=axtr.transAxes)
    quality = (
        f"resolution         {iw} x {ih}\n"
        f"photos used        {n_imgs}\n"
        f"reprojection error {err:.4f} px   (<0.5 good; ~1/9 px)\n"
    )
    axtr.text(0.0, 0.30, quality, va="top", ha="left", family="monospace",
              fontsize=10.5, color="#1a5276", transform=axtr.transAxes)

    fig.suptitle("Camera Calibration  —  measuring & removing the endoscope's lens distortion",
                 fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.015,
             f"A 25 mm checkerboard photographed from {n_imgs} angles pins down how "
             "the lens zooms, centres and warps the image, so every bone photo is "
             "straightened before we measure its shape.",
             ha="center", fontsize=9, style="italic", color="#555555")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}\n(demo image: {img_path})")


if __name__ == "__main__":
    main()
